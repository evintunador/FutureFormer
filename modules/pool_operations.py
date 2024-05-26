import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class MaxPooling(LoggingModule):
    def __init__(self):
        super().__init__()

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        return torch.max(x, dim=2)[0]  # (batch_size, max_seq_len, dim)


class SumPooling(LoggingModule):
    def __init__(self):
        super().__init__()

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        return torch.sum(x, dim=2)  # (batch_size, max_seq_len, dim)


class ParametricMaxPooling(LoggingModule):
    def __init__(self, dim, output_seq_len, use_output_linear=False):
        super().__init__()
        
        self.output_seq_len = output_seq_len
        self.linears = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(output_seq_len)])
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=False)

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        versions = [linear(x) for linear in self.linears]  # List of (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        pooled = torch.max(torch.stack(versions, dim=2), dim=3)[0]  # (batch_size, max_seq_len, output_seq_len, dim)
        if self.use_output_linear: pooled = self.out(pooled)  # (batch_size, max_seq_len, output_seq_len, dim)
        return pooled


class ParametricSumPooling(LoggingModule):
    def __init__(self, dim, output_seq_len, use_output_linear=False):
        super().__init__()
        
        self.output_seq_len = output_seq_len
        self.linears = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(output_seq_len)])
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=False)

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        versions = [linear(x) for linear in self.linears]  # List of (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        pooled = torch.sum(torch.stack(versions, dim=2), dim=3)  # (batch_size, max_seq_len, output_seq_len, dim)
        if self.use_output_linear: pooled = self.out(pooled)  # (batch_size, max_seq_len, output_seq_len, dim)
        return pooled


class FlattenProjectionPooling(LoggingModule):
    def __init__(self, to_be_pooled_seq_len, dim, output_seq_len):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Linear(to_be_pooled_seq_len * dim, dim, bias=False) for _ in range(output_seq_len)
        ])

    @log_io
    def forward(self, x):
        # x: (batch_size, max_seq_len, to_be_pooled_seq_len, dim)
        batch_size, max_seq_len, _, _ = x.shape
        flattened = x.view(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, to_be_pooled_seq_len * dim)
        pooled = torch.stack([proj(flattened) for proj in self.projections], dim=2)  # (batch_size, max_seq_len, output_seq_len, dim)
        return pooled


class ConvPooling(LoggingModule):
    def __init__(self, to_be_pooled_seq_len, dim, output_seq_len, use_output_linear=False):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=to_be_pooled_seq_len)
            for _ in range(output_seq_len)
        ])
        self.output_seq_len = output_seq_len
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=False)

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
    def __init__(self, dim, output_seq_len, use_output_linear=False):
        super().__init__()
        self.query = nn.Linear(dim, output_seq_len, bias=False)
        self.output_seq_len = output_seq_len
        
        self.use_output_linear = use_output_linear
        if use_output_linear: self.out = nn.Linear(dim, dim, bias=False)
    
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
    def __init__(self, dim, output_seq_len):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
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
