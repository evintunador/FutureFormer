import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io

class crossMQA(LoggingModule): # multi-query cross-attention
    def __init__(
        self, 
        dim: int,
        ca_head_dim: int,
        ca_num_q_heads: int, # at some point it'll prolly make sense to give this fewer heads
        ca_num_kv_heads: int,
        dropout_rate: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.num_q_heads = ca_num_q_heads
        self.num_kv_heads = ca_num_q_heads if ca_num_kv_heads is None else ca_num_kv_heads
        assert ca_num_q_heads % ca_num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // ca_num_q_heads if ca_head_dim is None else ca_head_dim
        self.dropout_rate = dropout_rate

        self.Wq = nn.Linear(dim, self.num_q_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.num_q_heads * self.head_dim, dim, bias=False)
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        freqs_cis: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len_x, _ = x.shape
        batch_size_c, seq_len_c, _ = c.shape
        assert batch_size == batch_size_c
        
        xq, ck, cv = self.Wq(x), self.Wk(c), self.Wv(c)

        xq = xq.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        ck = ck.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        cv = cv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        xq, ck = self.apply_rotary_emb(xq, ck, freqs_cis)
        ### if that ^ doesn't work, take a look at this old code from next-concept predictor v11.3
        #if self.use_RoPE:
            #expand = input_len_x // input_len_c
            #ck = ck.repeat_interleave(expand, dim=1) 
            #cv = cv.repeat_interleave(expand, dim=1) # cv need to be expanded for their use later on if we do this
            #xq, ck = self.RoPE(xq, ck)

        # adjusts ck and cv to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            ck, cv = self.match_headcount(ck, cv) # (bs, cache_len + seq_len, num_q_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, num_q_heads, seq_len, head_dim)
        ck = ck.transpose(1, 2)  # (bs, num_q_heads, cache_len + seq_len, head_dim)
        cv = cv.transpose(1, 2)  # (bs, num_q_heads, cache_len + seq_len, head_dim)
        
        logits = self.attend(xq, ck, training)
        scores = self.calc_output(logits, cv, training) 
        
        output = self.Wo(scores)
        if training: output = F.dropout(output, self.dropout_rate)
        
        return output
    
    @log_io
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        ck: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        ck_ = torch.view_as_complex(ck.float().reshape(*ck.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis.to(xq.device), xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        ck_out = torch.view_as_real(ck_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), ck_out.type_as(ck)

    @log_io
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(self, ck: torch.Tensor, cv: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        ck = torch.repeat_interleave(ck, self.num_q_heads // self.num_kv_heads, dim=2)
        cv = torch.repeat_interleave(cv, self.num_q_heads // self.num_kv_heads, dim=2)
        return ck, cv

    @log_io
    def attend(self, xq: torch.Tensor, ck: torch.Tensor, training: bool) -> torch.Tensor:
        return torch.matmul(xq, ck.transpose(2, 3)) * (self.head_dim ** -0.5)
    
    @log_io
    def calc_output(self, logits: torch.Tensor, cv: torch.Tensor, training: bool) -> torch.Tensor:
        batch_size, _, seq_len, _ = logits.shape
        scores = F.softmax(logits, dim=-1)
        if training: scores = F.dropout(scores, self.dropout_rate)
        output = scores @ cv # [batch_size, n_heads, seq_len, head_dim]
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [batch_size, seq_len, n_heads * head_dim]
