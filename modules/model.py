import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.self_mqa import precompute_freqs_cis
from modules.layer import Layer
from modules.mlp import MLP
from modules.loss import create_multi_hot_vector, splice_future_indices
from modules.pool_mechanism import CompressionSchedule
from modules.pool_operations import AttentionPooling

class Model(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len + 3 # the 3 is the bos, eos, and padding tokens
        self.max_batch_size = cfg.max_batch_size

        self.fs_mult_factor = cfg.fs_mult_factor
        self.fs_max_iter = cfg.fs_max_iter
        
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim)
        self.scale = cfg.dim ** 0.5 if cfg.scale_first_resid else 1.0
        
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))

        self.scheddy = CompressionSchedule(cfg.compress_freq, cfg.compress_freq_n)
        self.pool = nn.sequential(Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps),
                                  AttentionPooling(dim = cfg.dim, output_seq_len = 1, use_output_linear = cfg.pool_output_linear),
                                  Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps))
            # TODO:
            # - make the pre-pool norm optional
            # - make a dynamic version that can do different output lengths. gotta pre-calc the total number of modules needed
            # - allow choosing bw different pooling mechanisms in config
        
        self.output_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)
        self.output_projections = nn.ModuleList(nn.linear(cfg.dim, self.vocab_len) for _ in range(cfg.max_iter))

        freqs_cis = precompute_freqs_cis(cfg.head_dim,cfg.max_seq_len,cfg.theta).to(cfg.device)
        self.register_buffer('freqs_cis', freqs_cis)
        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf"), device=cfg.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.fs_criterion = nn.BCEWithLogitsLoss() # i guess the padding token vector is actually going to be useful
        self.ntp_criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_len - 1) # ignore the padding token

    @log_io
    def forward(self, inputs: torch.Tensor, cache_len: int = 0, targets: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        return self.inference(inputs, cache_len) if targets is None else self.training(inputs, targets)

    @log_io
    def training(inputs, targets):
        # setup
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        batch_size, seq_len = inputs.shape
        assert inputs.shape == targets.shape
        assert seq_len == self.max_seq_len
        mask = self.mask
        freqs_cis = self.freqs_cis

        # the subset of the targets we'll have our future sight mechanism 1) train on and 2) pool & then pay attention to
        future_targets = splice_future_indices(targets, 
                                               padding_token = self.vocab_len - 1, 
                                               mult_factor = self.fs_mult_factor, 
                                               max_iter = self.fs_max_iter)
        # for now we'll assume a single attention pooling module and figure out how to make it configurable later
        future_vecs = [self.pool(self.token_embedder(m)) for m in future_targets]
        
        # initialize first residual state and run first layer of the model
        x = self.token_embedder(inputs) * self.scale # [batch_size, seq_len, dim]
        x = self.layers[0](x, freqs_cis, mask, training = True)

        # encourage retrieve model's best guess for future vectors & use bag-of-words loss on em
        for out in self.output_projections:
            future_pred_chunk = out(x)

        # all following layers
        for i in range(1,len(self.layers)):
            x, c = self.layers[i](x, freqs_cis, mask, c, training = True)
        
        x = self.output_norm(x)
        logits = x @ self.token_embedder.weight.t() # [batch_size, seq_len, vocab_len]
            
        loss = self.ntp_criterion(
            logits.view(batch_size * seq_len, self.vocab_len),
            targets.reshape(batch_size * seq_len)
        )

        return logits, loss

    @log_io
    def inference(inputs, cache_len):
        # setup
        inputs = inputs.to(self.device)
        batch_size, seq_len = inputs.shape
        assert batch_size <= self.max_batch_size # we had to initialize the kv cache to some maximum possible size
        freqs_cis = self.freqs_cis[cache_len : cache_len + seq_len]
        mask = self.mask[:seq_len, :seq_len]
        mask = torch.hstack([torch.zeros((seq_len, cache_len), device=self.device), mask])
        
        # initialize first residual state and run the model
        x = self.token_embedder(inputs) * self.scale # [batch_size, seq_len, dim]
        for layer in self.layers:
            x, c = layer(x, c, freqs_cis, mask, cache_len, training = False)
            
        x = self.output_norm(x)
        logits = x @ self.token_embedder.weight.t() # [batch_size, seq_len, vocab_len]
            
        return logits, None