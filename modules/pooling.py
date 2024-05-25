import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

class AttentionPooling(LoggingModule):
    def __init__(self, embed_dim, output_seq_len, use_output_layer=True):
        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(embed_dim, output_seq_len)
        self.output_seq_len = output_seq_len
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.out = nn.Linear(embed_dim, embed_dim)
    
    @log_io
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        scores = self.query(x)  # (batch_size, seq_len, output_seq_len)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, output_seq_len)
        
        # Reshape attn_weights for broadcasting
        attn_weights = attn_weights.permute(0, 2, 1)  # (batch_size, output_seq_len, seq_len)
        
        pooled = torch.matmul(attn_weights, x)  # (batch_size, output_seq_len, embed_dim)
        
        if self.use_output_layer:
            pooled = self.out(pooled)  # (batch_size, output_seq_len, embed_dim)
        
        return pooled


class ConvPooling(LoggingModule):
    def __init__(self, seq_len, embed_dim, output_seq_len, use_output_layer=True):
        super(ConvPooling, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=seq_len)
            for _ in range(output_seq_len)
        ])
        self.output_seq_len = output_seq_len
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.out = nn.Linear(embed_dim, embed_dim)

    @log_io
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        pooled = [conv(x) for conv in self.convs]  # [(batch_size, embed_dim, 1)] * output_seq_len
        pooled = torch.cat(pooled, dim=2)  # (batch_size, embed_dim, output_seq_len)
        pooled = pooled.transpose(1, 2)  # (batch_size, output_seq_len, embed_dim)
        
        if self.use_output_layer:
            pooled = self.out(pooled)  # (batch_size, output_seq_len, embed_dim)
        
        return pooled
        

class SelfAttentionPooling(LoggingModule):
    def __init__(self, embed_dim, output_seq_len, use_output_layer=True):
        super(SelfAttentionPooling, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_queries = nn.Parameter(torch.randn(output_seq_len, embed_dim))
        self.softmax = nn.Softmax(dim=2)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.out = nn.Linear(embed_dim, embed_dim)

    @log_io
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        queries = self.query(x)  # (batch_size, seq_len, embed_dim)
        keys = self.key(x)  # (batch_size, seq_len, embed_dim)
        values = self.value(x)  # (batch_size, seq_len, embed_dim)

        output_queries = self.output_queries.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch_size, output_seq_len, embed_dim)
        
        attn_scores = torch.bmm(output_queries, keys.transpose(1, 2))  # (batch_size, output_seq_len, seq_len)
        attn_weights = self.softmax(attn_scores)  # (batch_size, output_seq_len, seq_len)
        
        pooled = torch.bmm(attn_weights, values)  # (batch_size, output_seq_len, embed_dim)
        
        if self.use_output_layer:
            pooled = self.out(pooled)  # (batch_size, output_seq_len, embed_dim)
        
        return pooled


class MaxPooling(LoggingModule):
    def __init__(self, embed_dim, output_seq_len, use_output_layer=True):
        super(MaxPooling, self).__init__()
        self.output_seq_len = output_seq_len
        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(output_seq_len)])
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.out = nn.Linear(embed_dim, embed_dim)

    @log_io
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        versions = [linear(x) for linear in self.linears]  # List of (batch_size, seq_len, embed_dim)
        pooled = torch.max(torch.stack(versions, dim=1), dim=2)[0]  # (batch_size, output_seq_len, embed_dim)
        
        if self.use_output_layer:
            pooled = self.out(pooled)  # (batch_size, output_seq_len, embed_dim)
        
        return pooled


class SumPooling(LoggingModule):
    def __init__(self, embed_dim, output_seq_len, use_output_layer=True):
        super(SumPooling, self).__init__()
        self.output_seq_len = output_seq_len
        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(output_seq_len)])
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.out = nn.Linear(embed_dim, embed_dim)

    @log_io
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        versions = [linear(x) for linear in self.linears]  # List of (batch_size, seq_len, embed_dim)
        pooled = torch.sum(torch.stack(versions, dim=1), dim=2)  # (batch_size, output_seq_len, embed_dim)
        
        if self.use_output_layer:
            pooled = self.out(pooled)  # (batch_size, output_seq_len, embed_dim)
        
        return pooled


class FlattenProjectionPooling(LoggingModule):
    def __init__(self, seq_len, embed_dim, output_seq_len, use_output_layer=True):
        super(FlattenProjectionPooling, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.output_seq_len = output_seq_len
        self.projections = nn.ModuleList([
            nn.Linear(seq_len * embed_dim, embed_dim) for _ in range(output_seq_len)
        ])
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.out = nn.Linear(embed_dim, embed_dim)

    @log_io
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size = x.size(0)
        flattened = x.view(batch_size, -1)  # (batch_size, seq_len * embed_dim)
        projected = torch.stack([proj(flattened) for proj in self.projections], dim=1)  # (batch_size, output_seq_len, embed_dim)
        
        if self.use_output_layer:
            projected = self.out(projected)  # (batch_size, output_seq_len, embed_dim)
        
        return projected
