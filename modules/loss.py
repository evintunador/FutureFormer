import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

def create_multi_hot_vector(sequence, vocab_size):
    multi_hot = torch.zeros(vocab_size)
    for token in sequence:
        multi_hot[token] = 1
    return multi_hot

