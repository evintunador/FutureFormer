import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.self_mqa import selfMQA
from modules.cross_mqa import crossMQA
from modules.mlp import MLP

class Layer(LoggingModule):
    def __init__(self, cfg, i: int = None, cross_attn: bool = False):
        super().__init__()
        self.i = i # the layer's id number
        self.second_norm = cfg.second_resid_norm
        self.dropout_rate = cfg.dropout_rate

        # self-attention connection
        self.pre_self_attn_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)
        self.self_attn = selfMQA(
            cfg.dim, cfg.head_dim,
            cfg.num_q_heads, cfg.num_kv_heads,
            cfg.max_batch_size, cfg.max_seq_len,
            cfg.dropout_rate,
            cfg.device
        )
        if self.second_norm: 
            self.post_self_attn_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)

        # cross-attention connection
        self.cross_attn = cross_attn
        if cross_attn:
            self.pre_cross_attn_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)
            self.cross_attn = crossMQA(cfg.dim, cfg.ca_head_dim, cfg.ca_num_q_heads, cfg.ca_num_kv_heads, cfg.dropout_rate, cfg.device)
            if self.second_norm: 
                self.post_cross_attn_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)

        # feedforward connection
        self.pre_mlp_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps) 
        # `mult` ensures mlp_hidden_mult maintains the same parameter count if gated == True
        mult = cfg.mlp_hidden_mult * 2/3 if cfg.mlp_gated else cfg.mlp_hidden_mult
        self.mlp = MLP(
            cfg.dim, int(cfg.dim * mult), cfg.dim,
            cfg.mlp_nonlinearity, cfg.mlp_gated,
            cfg.mlp_bias,
            cfg.dropout_rate
        )
        if self.second_norm: 
            self.post_mlp_norm = Norm(cfg.dim, cfg.norm_type, cfg.norm_affine, cfg.norm_bias, cfg.eps)

    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        c: torch.Tensor = None,
        cache_len: int = None,
        training = False,
    ) -> torch.Tensor:
        x = x + self.self_attn_connect(x, freqs_cis, mask, cache_len, training)
        if self.cross_attn & (c is not None):
            x = x + self.cross_attn_connect(x, c, freqs_cis, training)
        elif self.cross_attn & (c is None):
            raise InputError(f'Vectors to cross-attend to expected in layer {self.i}, but none were inputted')
        x = x + self.mlp_connect(x, training)
        return x

    @log_io
    def self_attn_connect(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        mask: torch.Tensor, 
        cache_len: int, 
        training: bool
    ) -> torch.Tensor:
        dx = self.self_attn(self.pre_self_attn_norm(x),freqs_cis, mask, cache_len, training)
        if training: F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_self_attn_norm(dx)
        return dx

    @log_io
    def cross_attn_connect(self, x: torch.Tensor, c: torch.Tensor, training: bool,) -> torch.Tensor:
        dx = self.cross_attn(self.pre_cross_attn_norm(x), c, training) # c should already be normed, no? i guess it makes sense to do it here
        if training: F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_cross_attn_norm(dx)
        return dx

    @log_io
    def mlp_connect(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        dx = self.mlp(self.pre_mlp_norm(x))
        if training: F.dropout(dx, self.dropout_rate)
        if self.second_norm: dx = self.post_mlp_norm(dx)
        return dx