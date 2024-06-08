import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from modules.logging import LoggingModule, log_io

class CompressionSchedule:
    def __init__(self, compress_freq, compress_freq_n):
        self.compress_freq = compress_freq
        self.compress_freq_n = compress_freq_n
        
        if compress_freq_n < 1: 
            raise ValueError(f"compress_freq_n must be >= 1")
        if compress_freq == 'log' and compress_freq_n < 2: 
            raise ValueError(f"if using compress_freq=='log' then compress_freq_n must be >= 2")

    def __call__(self, i: int) -> int:
        if self.compress_freq == 'constant':
            return self.constant(i)
        elif self.compress_freq == 'linear':
            return self.linear(i)
        elif self.compress_freq == 'root':
            return self.root(i)
        elif self.compress_freq == 'log':
            return self.log(i)
        elif self.compress_freq == 'poly':
            return self.poly(i)
        else:
            raise ValueError(f"Invalid compression frequency type. {self.compress_freq} is unknkown")

    def constant(self, i: int) -> int:
        return math.floor(self.compress_freq_n)

    def linear(self, i: int) -> int:
        return math.floor(self.compress_freq_n * i + 1)

    def root(self, i: int) -> int:
        return math.floor((i+1) ** (1 / self.compress_freq_n))

    def log(self, i: int) -> int:
        return math.floor(math.log((i+1), self.compress_freq_n))+1

    def poly(self, i: int) -> int:
        return math.floor((i+1) ** self.compress_freq_n)


class MaxPooling(LoggingModule):
    def __init__(self):
        super().__init__()
    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        return torch.max(x, dim=2)[0].unsqueeze(2)  # (batch_size, max_seq_len, 1, dim)


class SumPooling(LoggingModule):
    def __init__(self):
        super().__init__()
    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        return torch.sum(x, dim=2).unsqueeze(2)  # (batch_size, max_seq_len, 1, dim)


class ParametricMaxPooling(LoggingModule):
    def __init__(self, dim, output_seq_len, use_output_linear: bool = False, bias: bool = False):
        super().__init__()
        
        self.output_seq_len = output_seq_len
        self.linears = nn.ModuleList([nn.Linear(dim, dim, bias=bias) for _ in range(output_seq_len)])
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=bias)

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        versions = [linear(x) for linear in self.linears]  # List of (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        pooled = torch.max(torch.stack(versions, dim=2), dim=3)[0]  # (batch_size, max_seq_len, output_seq_len, dim)
        if self.use_output_linear: pooled = self.out(pooled)  # (batch_size, max_seq_len, output_seq_len, dim)
        return pooled


class ParametricSumPooling(LoggingModule):
    def __init__(self, dim, output_seq_len, use_output_linear: bool = False, bias: bool = False):
        super().__init__()
        
        self.output_seq_len = output_seq_len
        self.linears = nn.ModuleList([nn.Linear(dim, dim, bias=bias) for _ in range(output_seq_len)])
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=bias)

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        versions = [linear(x) for linear in self.linears]  # List of (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        pooled = torch.sum(torch.stack(versions, dim=2), dim=3)  # (batch_size, max_seq_len, output_seq_len, dim)
        if self.use_output_linear: pooled = self.out(pooled)  # (batch_size, max_seq_len, output_seq_len, dim)
        return pooled


class FlattenProjectionPooling(LoggingModule):
    def __init__(self, dim, to_be_pooled_seq_len, output_seq_len, bias: bool = False):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Linear(to_be_pooled_seq_len * dim, dim, bias=bias) for _ in range(output_seq_len)
        ])

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        batch_size, max_seq_len, _, _ = x.shape
        flattened = x.view(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, to_be_pooled_seq_len * dim)
        pooled = torch.stack([proj(flattened) for proj in self.projections], dim=2)  # (batch_size, max_seq_len, output_seq_len, dim)
        return pooled


class ConvPooling(LoggingModule):
    def __init__(self, dim, to_be_pooled_seq_len, output_seq_len, use_output_linear: bool = False, bias: bool = False):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=to_be_pooled_seq_len, bias=bias)
            for _ in range(output_seq_len)
        ])
        self.output_seq_len = output_seq_len
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=bias)

    @log_io
    def forward(self, x):
        batch_size, max_seq_len, to_be_pooled_seq_len, dim = x.shape
        x = x.view(batch_size * max_seq_len, to_be_pooled_seq_len, dim)  # (batch_size * max_seq_len, to_be_pooled_seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size*max_seq_len, dim, to_be_pooled_seq_len)
        
        pooled = [conv(x) for conv in self.convs]  # [(batch_size*max_seq_len, dim, 1)] * output_seq_len
        pooled = torch.cat(pooled, dim=2)  # (batch_size*max_seq_len, dim, output_seq_len)
        pooled = pooled.transpose(1, 2)  # (batch_size*max_seq_len, output_seq_len, dim)
        
        if self.use_output_linear: pooled = self.out(pooled)  # (batch_size*max_seq_len, output_seq_len, dim)
        return pooled.view(batch_size, max_seq_len, -1, dim)


class AttentionPooling(LoggingModule):
    def __init__(self, dim, output_seq_len, use_output_linear: bool = False, bias: bool = False):
        super().__init__()
        self.query = nn.Linear(dim, output_seq_len, bias=bias)
        self.output_seq_len = output_seq_len
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=bias)
    
    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        scores = self.query(x)  # (batch_size, max_seq_len, to_be_pooled_seq_len, output_seq_len)
        attn_weights = F.softmax(scores, dim=2)  # (batch_size, max_seq_len, to_be_pooled_seq_len, output_seq_len)
        
        # Reshape attn_weights for broadcasting
        attn_weights = attn_weights.permute(0, 1, 3, 2)  # (batch_size, max_seq_len, output_seq_len, to_be_pooled_seq_len)
        pooled = torch.matmul(attn_weights, x)  # (batch_size, output_seq_len, dim)
        
        if self.use_output_linear: pooled = self.out(pooled)  # (batch_size, output_seq_len, dim)
        return pooled
        

class SelfAttentionPooling(LoggingModule): # TODO: figure out how to make queries input-dependent
    def __init__(self, dim, output_seq_len, bias: bool = False):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        self.output_queries = nn.Parameter(torch.randn(output_seq_len, dim))
        self.softmax = nn.Softmax(dim=2)

    @log_io
    def forward(self, x):
        batch_size, max_seq_len, to_be_pooled_seq_len, dim = x.shape
        x = x.view(batch_size*max_seq_len, to_be_pooled_seq_len, dim)
        
        keys = self.key(x)  # (batch_size*max_seq_len, to_be_pooled_seq_len, dim)
        values = self.value(x)  # (batch_size*max_seq_len, to_be_pooled_seq_len, dim)

        # (batch_size*max_seq_len, output_seq_len, dim)
        output_queries = self.output_queries.unsqueeze(0).expand(x.size(0), -1, -1)  
        
        attn_scores = torch.bmm(output_queries, keys.transpose(1, 2))  # (batch_size*max_seq_len, output_seq_len, to_be_pooled_seq_len)
        attn_weights = self.softmax(attn_scores)  # (batch_size*max_seq_len, output_seq_len, to_be_pooled_seq_len)
        
        pooled = torch.bmm(attn_weights, values)  # (batch_size*max_seq_len, output_seq_len, dim)
        
        return pooled.view(batch_size, max_seq_len, -1, dim)


class PoolingHub(LoggingModule):
    def __init__(
        self, 
        compress_freq: int,
        compress_freq_n: int,
        pool_type: str,
        dim: int,
        fs_mult,
        fs_periods: int,
        output_linear: bool
    ):
        super().__init__()

        # schedule determines how many vectors the pooling mechanism should to compress into
        self.scheddy = CompressionSchedule(compress_freq, compress_freq_n)

        ### selecting your choice of pooling mechanisms
        self.pool_type = pool_type
        if pool_type == 'sum': self.pooler = SumPooling()
        elif pool_type == 'max': self.pooler = MaxPooling()
        elif pool_type == 'parametric_sum':
            self.pooler_list = nn.ModuleList(ParametricSumPooling(dim = dim, 
                                                                output_seq_len = self.scheddy(fs_periods - 1 - i), 
                                                                use_output_linear = pool_output_linear
                                                               ) for i in range(fs_periods))
        elif pool_type == 'parametric_max':
            self.pooler_list = nn.ModuleList(ParametricMaxPooling(dim = dim, 
                                                                output_seq_len = self.scheddy(fs_periods - 1 - i), 
                                                                use_output_linear = pool_output_linear
                                                               ) for i in range(fs_periods))
        elif pool_type == 'flatten':
            self.pooler_list = nn.ModuleList(FlattenProjectionPooling(dim = dim, 
                                                                    to_be_pooled_seq_len = fs_mult ** (fs_periods - i),
                                                                    output_seq_len = self.scheddy(fs_periods - 1 - i)
                                                                   ) for i in range(fs_periods))
        elif pool_type == 'conv':
            self.pooler_list = nn.ModuleList(ConvPooling(dim = dim, 
                                                       to_be_pooled_seq_len = fs_mult ** (fs_periods - i),
                                                       output_seq_len = self.scheddy(fs_periods - 1 - i),
                                                       use_output_linear = pool_output_linear
                                                      ) for i in range(fs_periods))
        elif pool_type == 'attention':
            self.pooler_list = nn.ModuleList(AttentionPooling(dim = dim, 
                                                            output_seq_len = self.scheddy(fs_periods - 1 - i),
                                                            use_output_linear = pool_output_linear
                                                           ) for i in range(fs_periods))
        elif pool_type == 'self_attention': # TODO: fix self-attention pooling to make queries actually input-dependent
            self.pooler_list = nn.ModuleList(SelfAttentionPooling(dim = dim, 
                                                                output_seq_len = self.scheddy(fs_periods - 1 - i)
                                                               ) for i in range(fs_periods))
        else:
            raise InputError(f'pool_type {pool_type} unrecognized')

    @log_io
    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return  self.pooler(x) if self.pool_type in ['sum', 'max'] else self.pooler_list[i](x)