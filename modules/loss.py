import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io

def splice_future_indices(target_tokens, padding_token, mult_factor, max_iter):
    batch_size, max_seq_len = target_tokens.size()
    matrices = []

    length = mult_factor
    tot_length = 1 + length 
    j = 0

    while (tot_length <= max_seq_len) and (j < max_iter):
        matrix = []
        for i in range(max_seq_len):
            subseq = target_tokens[:, i+1:i+1+length]  # slice the target tokens
            
            # If the subsequence is shorter than the required length, pad it with padding_token
            if subseq.size(1) < length:
                padding = torch.full((batch_size, length - subseq.size(1)), padding_token, dtype=torch.long)
                subseq = torch.cat([subseq, padding], dim=1)
            
            matrix.append(subseq)
        
        matrices.append(torch.stack(matrix, dim=1))
        
        length *= mult_factor
        tot_length += length
        j += 1

    return matrices
    
def create_multi_hot_vector(sequence, vocab_size):
    multi_hot = torch.zeros(vocab_size)
    for token in sequence:
        multi_hot[token] = 1
    return multi_hot

