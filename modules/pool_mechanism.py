import torch
import torch.nn as nn

from modules.pool_operations import *

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

