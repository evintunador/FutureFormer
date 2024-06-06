import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.self_mqa import precompute_freqs_cis
from modules.layer import Layer
from modules.mlp import MLP
from modules.loss import create_multi_hot_vector, splice_future_indices
from modules.pool_mech import CompressionSchedule
from modules.pool_ops import *

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

        # the shared body of the model before we start doing future-sight prediction
        self.body_layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_layers - cfg.fs_periods - 1))
        # the unique layer that gets to use both cross-attention and kv caching
        self.first_fs_layer = Layer(cfg, i = cfg.num_layers - cfg.fs_periods - 1, cross_attn = True, kv_cache = True)
        # the remaining layers which do not have the luxury of kv caching
        self.fs_layers = nn.ModuleList(
            Layer(cfg, i = cfg.num_layers - cfg.fs_periods + i, cross_attn = True, kv_cache = False
                 ) for i in range(cfg.fs_periods))

        self.scheddy = CompressionSchedule(cfg.compress_freq, cfg.compress_freq_n)
        self.pool = nn.sequential(
            Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps),
            SumPooling(dim = cfg.dim, output_seq_len = 1, use_output_linear = cfg.pool_output_linear))
            # TODO:
            # - make the pre-pool norm optional
            # - make a dynamic version that can do different output lengths. gotta pre-calc the total number of modules needed
            # - allow choosing bw different pooling mechanisms in config
        
        self.output_projections = nn.ModuleList(
            nn.sequential(
                Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps),
                nn.linear(cfg.dim, self.vocab_len)
            ) for _ in range(cfg.fs_periods + 1))
            # TODO:
            # - make final NTP projection optionally shareable with self.token_embedder
            # - make all output projections optionally shareable with self.token_embedder

        freqs_cis = precompute_freqs_cis(cfg.head_dim,cfg.max_seq_len,cfg.theta).to(cfg.device)
        self.register_buffer('freqs_cis', freqs_cis)
        mask = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf"), device=cfg.device)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

        self.fs_criterion = nn.BCEWithLogitsLoss() # i guess the padding token vector is actually going to be useful
        self.ntp_criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_len - 1) # ignore the padding token
            # TODO:
            # - add a parameter to adjust weight of fs loss. the code should exist but i'll prolly just set it to 1

    @log_io
    def forward(self, inputs: torch.Tensor, cache_len: int = 0, targets: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor):
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
                                               max_iter = self.fs_periods) # [fs_periods x (batch_size, max_seq_len, fs_mult_factor ** i)]
        # for now we'll assume a single attention pooling module and figure out how to make it configurable later
        future_vecs = [self.pool(self.token_embedder(m)) for m in future_targets] # [fs_periods x (batch_size, max_seq_len, fs_mult_factor**i, dim)]
        
        # initialize first residual state and run first layer of the model
        x = self.token_embedder(inputs) * self.scale # (batch_size, seq_len, dim)
        for layer in self.body_layers:
            x = layer(x, freqs_cis, mask, training = True) # (batch_size, seq_len, dim)
        # get our first fs output
        y = self.output_projections[0](x) # (batch_size, seq_len, vocab_len)
        logits_list = [y]
        
        # run our unique in-between layer that gets to use both kv caching (in the self-attention) and cross-attention
        x = self.first_fs_layer(x, freqs_cis, mask, y = y, training = True) # (batch_size, seq_len, dim)
        # get our second fs output
        y = self.output_projections[1](x) # (batch_size, seq_len, vocab_len)
        logits_list = logits_list.append(y)
        
        # loop thru remaining layers & their outputs
        for i, layer in enumerate(self.fs_layers):
            x = layer(x, freqs_cis, mask, y = y, training = True)
            y = self.output_projections[i+2](x)
            logits_list = logits_list.append(y)
        #######################################################
        ### ok somewhere in these last three blocks ^ i'm mis-counting the length of output_projections & fs_layers
        ### also i'm supposed to be cross-attending to future_vecs instead of the y i just made since this is training
        
        fs_loss = 0
        for i, logits in enumerate(logits_list):
            if i != len(logits_list) - 1): # if we're messing with fs output logits
                fs_loss = fs_loss + self.fs_criterion(
                    logits.view(batch_size * seq_len, self.vocab_len),
                    # what do i put here? how do i use BCELoss???
                )
            else: # if we're messing with the final NTP output logits
                ntp_loss = self.ntp_criterion(
                    logits.view(batch_size * seq_len, self.vocab_len),
                    targets.reshape(batch_size * seq_len)
                )
                # the final logits object will be our NTP logits so we can just return it
                
        loss = ntp_loss + fs_loss

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