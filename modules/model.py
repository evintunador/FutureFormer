import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.mqa import precompute_freqs_cis
from modules.layer import Layer
from modules.loss import splice_future_indices, create_multi_hot_vector
from modules.pool_ops import PoolingHub

class Model(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device

        ### regular/general hyperparameters
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len + 3 # the 3 is the bos, eos, and padding tokens
        self.max_batch_size = cfg.max_batch_size

        ### Future Sight specific hyperparameters
        self.fs_mult = cfg.fs_mult
        self.fs_periods = cfg.fs_periods

        ### the initial embedding
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim)
        self.scale = cfg.dim ** 0.5 if cfg.scale_first_resid else 1.0

        ### the body of the model
        # the shared body of the model before we start doing future-sight prediction
        self.body_layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_layers - cfg.fs_periods))
        # the unique layer that gets to use both cross-attention and kv caching
        self.first_fs_layer = Layer(cfg, i = cfg.num_layers - cfg.fs_periods, cross_attn = True, kv_cache = True)
        # the remaining layers which do not have the luxury of kv caching
        self.fs_layers = nn.ModuleList(
            Layer(cfg, i = cfg.num_layers - cfg.fs_periods + 1 + i, cross_attn = True, kv_cache = False
                 ) for i in range(cfg.fs_periods - 1))

        ### Pooling
        # optional norm before pooling
        if cfg.pre_pool_norm: self.pre_pool_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)
        # an object full of the pooling modules for each fs level
        self.pooler = PoolingHub(
            cfg.compress_freq, 
            cfg.compress_freq_n, 
            cfg.pool_type,
            cfg.dim,
            cfg.fs_mult,
            cfg.fs_periods,
            cfg.pool_output_linear
        )

        ### the model's output projection(s)
        self.output_projection = nn.Sequential(
            Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps),
            nn.Linear(cfg.dim, self.vocab_len, bias=False)
        )
        # optional weight tying for the output projections to the embedding matrix
        if cfg.weight_tying: self.token_embedder.weight = self.output_projection[1].weight

        ### RoPE
        freqs_cis = precompute_freqs_cis(cfg.head_dim,cfg.max_seq_len,cfg.theta).to(cfg.device)
        self.register_buffer('freqs_cis', freqs_cis)
        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf"), device=cfg.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        ### Loss Functions
        self.fs_criterion = nn.BCEWithLogitsLoss() # i guess the padding token vector is actually going to be useful
        self.ntp_criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_len - 1) # ignore the padding token
        # parameter to weight the FS loss with geometric decay
        self.fs_loss_lambda = cfg.fs_loss_lambda

    @log_io
    def forward(
        self, 
        input_token_ids: torch.Tensor, 
        cache_len: int = 0, 
        target_token_ids: Optional[torch.Tensor] = None
    ) -> (torch.Tensor, torch.Tensor):
        return self.forward_inference(input_token_ids, cache_len) if target_token_ids is None else self.forward_train(input_token_ids, target_token_ids)

    @log_io
    def forward_train(self, input_token_ids: torch.Tensor, target_token_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        input_token_ids, target_token_ids = input_token_ids.to(self.device), target_token_ids.to(self.device) # ensure correct device
        batch_size, max_seq_len = input_token_ids.shape
        assert input_token_ids.shape == target_token_ids.shape
        assert max_seq_len == self.max_seq_len
        mask, freqs_cis = self.mask, self.freqs_cis
        
        ### setting up Future Sight
        # the subset of the targets we'll have our Future Sight mechanism 1) train on and 2) embed then pool then pay attention to
        future_targets = splice_future_indices(target_token_ids, 
                                               padding_token = self.vocab_len - 1, 
                                               mult_factor = self.fs_mult, 
                                               max_iter = self.fs_periods)
            # [fs_periods x (batch_size, max_seq_len, fs_mult ** i)]
            # order is further-in-the-future to less-far-in-the-future
        # for now we'll assume a sum pooling module and figure out how to make it configurable later
        future_vecs = [self.pooler(self.pre_pool_norm(self.token_embedder(ft)), i) for i, ft in enumerate(future_targets)]
            # [fs_periods x (batch_size, max_seq_len, fs_mult**i, dim)]
        
        ### initialize first residual state and run the first few regular layers of the model
        x = self.token_embedder(input_token_ids) * self.scale # (batch_size, max_seq_len, dim)
        for layer in self.body_layers:
            x = layer(x, freqs_cis, mask, training = True) # (batch_size, max_seq_len, dim)
        # get our first Future Sight output
        z = self.output_projection(x) # (batch_size, max_seq_len, vocab_len)
        logits_list = [z]
        
        ### run our unique in-between layer
        # gets to use both kv caching (in the self-attention) and cross-attention
        x = self.first_fs_layer(x, freqs_cis, mask, y = future_vecs[0], training = True) # (batch_size, max_seq_len, dim)
        # get our second FS output
        z = self.output_projection(x) # (batch_size, max_seq_len, vocab_len)
        logits_list.append(z)
        
        ### loop thru remaining layers & their outputs
        for i, layer in enumerate(self.fs_layers):
            y = torch.concat(future_vecs[:i+1], dim = 2)
            x = layer(x, freqs_cis, mask, y = y, training = True) # (batch_size, max_seq_len, dim)
            z = self.output_projection(x) # (batch_size, max_seq_len, vocab_len)
            logits_list.append(z)

        ### calculating loss
        # turn the targets into multi-hot vectors compatible with BCELoss
        multi_hots = [create_multi_hot_vector(target, self.vocab_len) for target in future_targets] 
            # (batch_size, max_seq_len, vocab_len)
        # initialize Future Sight loss
        fs_loss = 0.
        # iterate over output logits & calculate loss
        for i, logits in enumerate(logits_list):
            if i != (len(logits_list) - 1): # if we're messing with FS output logits
                # we weight the FS loss to give more weight to NTP
                fs_lambda = self.fs_loss_lambda ** (self.fs_periods - i)
                fs_loss = fs_loss + fs_lambda * self.fs_criterion(
                    logits.view(batch_size * max_seq_len, self.vocab_len),
                    multi_hots[i].view(batch_size * max_seq_len, self.vocab_len)
                )
            else: # if we're messing with the final NTP output logits
                ntp_loss = self.ntp_criterion(
                    logits.view(batch_size * max_seq_len, self.vocab_len),
                    target_token_ids.reshape(batch_size * max_seq_len)
                )
                # the final logits object will be our NTP logits so we can just return it
        
        loss = ntp_loss + fs_loss
        return logits, loss

    @log_io
    def forward_inference(self, input_token_ids: torch.Tensor, cache_len: int) -> (torch.Tensor, None):
        # setup
        input_token_ids = input_token_ids.to(self.device)
        batch_size, seq_len = input_token_ids.shape
        assert batch_size <= self.max_batch_size # we had to initialize the kv cache to some maximum possible size
        freqs_cis = self.freqs_cis[cache_len : cache_len + seq_len]
        mask = self.mask[:seq_len, :seq_len]
        mask = torch.hstack([torch.zeros((seq_len, cache_len), device=self.device), mask])

        ### initialize first residual state and run the first few regular layers of the model
        x = self.token_embedder(input_token_ids) * self.scale # (batch_size, seq_len, dim)
        for layer in self.body_layers:
            x = layer(x, freqs_cis, mask, cache_len) # (batch_size, seq_len, dim)
        
        ### get our first Future Sight output
        z = self.output_projection(x) # (batch_size, seq_len, vocab_len)
        # figure out how many tokens we want to select & then grab their indices greedily
        fs_indices = torch.topk(z, k = self.fs_mult ** self.fs_periods, dim=2).indices # (batch_size, seq_len, fs_mult ** fs_periods)
        # embed & then pool them to get the output we can cross-attend to
        fs_vecs = self.token_embedder(fs_indices) # (batch_size, seq_len, fs_mult ** fs_periods, dim)
        y = self.pooler(self.pre_pool_norm(fs_vecs), i=0) # (batch_size, seq_len, pooled_len, dim)
            # pooled_len is determined by self.scheddy

        ### run our unique in-between layer 
        # it gets to use both kv caching (in the self-attention) and cross-attention
        x = self.first_fs_layer(x, freqs_cis, mask, cache_len, y) # (batch_size, max_seq_len, dim)
        # get our second FS output & turn it into the vector we can cross-attend to
        z = self.output_projection(x) # (batch_size, max_seq_len, vocab_len)
        fs_indices = torch.topk(z, k = self.fs_mult ** (self.fs_periods - 1), dim=2).indices 
            # (batch_size, seq_len, fs_mult ** (fs_periods - 1))
        fs_vecs = self.token_embedder(fs_indices) # (batch_size, seq_len, fs_mult ** (fs_periods - 1), dim)
        y = self.pooler(self.pre_pool_norm(fs_vecs), i=1) # (batch_size, seq_len, pooled_len, dim)
        
        ### loop thru remaining layers & their outputs
        for i, layer in enumerate(self.fs_layers):
            x = layer(x, freqs_cis, mask, y = y) # (batch_size, max_seq_len, dim)
            z = self.output_projection(x) # (batch_size, max_seq_len, vocab_len)
            if i != (self.fs_periods - 1):
                fs_indices = torch.topk(z, k = self.fs_mult ** (self.fs_periods - 2 - i), dim=2).indices 
                    # (batch_size, seq_len, fs_mult ** (fs_periods - 2 - i))
                fs_vecs = self.token_embedder(fs_indices) # (batch_size, seq_len, fs_mult ** (fs_periods - 2 - i), dim)
                y = self.pooler(self.pre_pool_norm(fs_vecs), i+2) # (batch_size, seq_len, pooled_len, dim)
            
        return z, None