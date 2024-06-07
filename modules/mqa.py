import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io

def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class MQA(LoggingModule):
    """
    multi-query attention https://arxiv.org/abatch_size/1911.02150
    also reduces to grouped-query attention or multi-head attention depending on your hyperparameters

    this module is flexible to either act as self-attention or as cross-attention. 
    For the latter, just pass the encoder-output into xk & xv. However it does assume that dim is the same

    Optionally disable kv caching by letting max_batch_size default to None
    Optionally disable Rotary Positional Encoding by just not passing in a freqs_cis tensor
    """
    def __init__(
        self, 
        dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        max_batch_size: Optional[int] = None, # If None then kv cache will not be used
        dropout_rate: float = 0.1,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        assert num_q_heads % num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // num_q_heads if head_dim is None else head_dim
        self.dropout_rate = dropout_rate

        self.Wq = nn.Linear(dim, num_q_heads * head_dim, bias=False).to(device)
        self.Wk = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False).to(device)
        self.Wv = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False).to(device)
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=False).to(device)

        # various layers wil not be able to use kv caching to prevent information leakage
        self.kv_cache = True if max_batch_size is not None else False
        if self.kv_cache:
            self.cache_k = torch.zeros(
                (max_batch_size, max_seq_len, num_kv_heads, head_dim),
                requires_grad = False).to(device)
            self.cache_v = torch.zeros(
                (max_batch_size, max_seq_len, num_kv_heads, head_dim),
                requires_grad = False).to(device)
    
    @log_io
    def forward(
        self,
        q: torch.Tensor, # for self-attention you'd input the same tensor into each of these three
        k: torch.Tensor, # for cross-attention you'd input the encoder output into k & v
        v: torch.Tensor, # self-attention they're all (batch_size, seq_len, dim) but for cross-attentionxk & v are (batch_size, pool_output_len, dim)
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache_len: Optional[int] = None,
        training: bool = False,
    ) -> torch.Tensor:
        assert q.shape[0] == k.shape[0] == v.shape[0] 
        assert k.shape[1] == v.shape[1] # seq_len does not need to match with queries if we're doing either kv caching or cross-attention
        assert q.shape[2] == k.shape[2] == v.shape[2]
        batch_size, seq_len_q, _ = q.shape
        seq_len_kv = k.shape[1]
        
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)

        q = q.view(batch_size, seq_len_q, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim)

        if freqs_cis is not None: q, k = self.apply_rotary_emb(q, k, freqs_cis)

        if self.kv_cache & (cache_len is not None): # if we're performing inference and using kv caching. it'll init at 0
            self.cache_k = self.cache_k.to(q)
            self.cache_v = self.cache_v.to(q)

            self.cache_k[:batch_size, cache_len : cache_len + seq_len_kv] = k
            self.cache_v[:batch_size, cache_len : cache_len + seq_len_kv] = v

            k = self.cache_k[:batch_size, : cache_len + seq_len_kv]
            v = self.cache_v[:batch_size, : cache_len + seq_len_kv]

        # adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            k, v = self.match_headcount(k, v) # (batch_size, cache_len + seq_len_kv, num_q_heads, head_dim)

        q = q.transpose(1, 2)  # (batch_size, num_q_heads, seq_len_q, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_q_heads, cache_len + seq_len_kv, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_q_heads, cache_len + seq_len_kv, head_dim)
        
        logits = self.attend(q, k) # (batch_size, num_q_heads, seq_len_q, cache_len + seq_len_kv)
        if mask is not None: logits = logits + mask  
        scores = self.calc_output(logits, v, training) # (batch_size, seq_len_q, n_heads * head_dim)
        
        output = self.Wo(scores) # (batch_size, seq_len_q, dim)
        return output
    
    @log_io
    def apply_rotary_emb(self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,) -> (torch.Tensor, torch.Tensor):
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis.to(q.device), q_)
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)

    @log_io
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(self, k: torch.Tensor, v: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        k = torch.repeat_interleave(k, self.num_q_heads // self.num_kv_heads, dim=2)
        v = torch.repeat_interleave(v, self.num_q_heads // self.num_kv_heads, dim=2)
        return k, v # (batch_size, cache_len + seq_len_kv, num_q_heads, head_dim)

    @log_io
    def attend(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        k = k.transpose(2, 3)  # (batch_size, num_q_heads, head_dim, cache_len + seq_len_kv)
        return (q @ k) * (self.head_dim ** -0.5) # (batch_size, num_q_heads, seq_len_q, cache_len + seq_len_kv)
    
    @log_io
    def calc_output(self, logits: torch.Tensor, v: torch.Tensor, training: bool) -> torch.Tensor:
        batch_size, _, seq_len_q, _ = logits.shape # (batch_size, n_heads, seq_len_q, seq_len_kv)
        scores = F.softmax(logits, dim=-1) # (batch_size, n_heads, seq_len_q, seq_len_kv)
        if training: scores = F.dropout(scores, self.dropout_rate)
        output = scores @ v # (batch_size, n_heads, seq_len_q, head_dim)
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1) # (batch_size, seq_len_q, n_heads * head_dim)
        
class futureSightMQA(LoggingModule):
    """
    like the above except designed for the k&v being cross-attended to to have shape (batch_size, seq_len, pool_seq_len, dim)
    """
    def __init__(
        self, 
        dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        assert num_q_heads % num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // num_q_heads if head_dim is None else head_dim
        self.dropout_rate = dropout_rate

        self.Wq = nn.Linear(dim, num_q_heads * head_dim, bias=False).to(device)
        self.Wk = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False).to(device)
        self.Wv = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False).to(device)
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=False).to(device)
    
    @log_io
    def forward(
        self,
        q: torch.Tensor, # (batch_size, seq_len, dim)
        kv: torch.Tensor, # k & v are (batch_size, seq_len, seq_len_fs, dim)
        training: bool = False,
    ) -> torch.Tensor:
        assert q.shape[0] == kv.shape[0] # batch_size
        assert q.shape[1] == kv.shape[1] # seq_len
        assert q.shape[2] == kv.shape[3] # dim
        batch_size, seq_len, _ = q.shape
        seq_len_fs = kv.shape[2]
        
        q, k, v = self.Wq(q), self.Wk(kv), self.Wv(kv)

        q = q.view(batch_size*seq_len, 1, self.num_q_heads, self.head_dim)
        k = k.view(batch_size*seq_len, seq_len_fs, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size*seq_len, seq_len_fs, self.num_kv_heads, self.head_dim)

        # adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            k, v = self.match_headcount(k, v) # (batch_size*seq_len, seq_len_fs, num_q_heads, head_dim)

        q = q.transpose(1, 2)  # (batch_size*seq_len, num_q_heads, 1, head_dim)
        k = k.transpose(1, 2)  # (batch_size*seq_len, num_q_heads, seq_len_fs, head_dim)
        v = v.transpose(1, 2)  # (batch_size*seq_len, num_q_heads, seq_len_fs, head_dim)
        
        logits = self.attend(q, k) # (batch_size*seq_len, num_q_heads, 1, seq_len_fs)
        scores = self.calc_output(logits, v, training) # (batch_size*seq_len, 1, n_heads * head_dim)
        
        output = self.Wo(scores) # (batch_size*seq_len, 1, dim)
        return output.view(batch_size, seq_len, -1) # (batch_size, seq_len, dim)

    @log_io
    def match_headcount(self, k: torch.Tensor, v: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        k = torch.repeat_interleave(k, self.num_q_heads // self.num_kv_heads, dim=2)
        v = torch.repeat_interleave(v, self.num_q_heads // self.num_kv_heads, dim=2)
        return k, v # (batch_size*seq_len, seq_len_fs, num_q_heads, head_dim)

    @log_io
    def attend(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        k = k.transpose(2, 3) # (batch_size*seq_len, num_q_heads, head_dim, seq_len_fs)
        return (q @ k) * (self.head_dim ** -0.5) # (batch_size*seq_len, num_q_heads, 1, seq_len_fs)
    
    @log_io
    def calc_output(self, logits: torch.Tensor, v: torch.Tensor, training: bool) -> torch.Tensor:
        bs_x_sl, _, seq_len_q, seq_len_fs = logits.shape # (batch_size*seq_len, n_heads, 1, seq_len_fs)
        scores = F.softmax(logits, dim=-1) # (batch_size*seq_len, n_heads, 1, seq_len_fs)
        if training: scores = F.dropout(scores, self.dropout_rate)
        output = scores @ v # (batch_size*seq_len, n_heads, 1, head_dim)
        return output.transpose(1, 2).contiguous().view(bs_x_sl, 1, -1) # (batch_size*seq_len, 1, n_heads * head_dim)
