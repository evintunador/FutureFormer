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

    while j < max_iter:# (tot_length <= max_seq_len) and
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

    return matrices[::-1] # the splicing reverses the order so furthest in future vectors are first in the list
    
def create_multi_hot_vector(sequence, vocab_size):
    batch_size, seq_len, number_of_targets = sequence.shape
    
    # Initialize target tensor with zeros
    multi_hot = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float)
    
    # Generate the batch and sequence indices
    batch_indices = torch.arange(batch_size).view(-1, 1, 1)
    seq_indices = torch.arange(seq_len).view(1, -1, 1)
    
    # Expand indices to match the number_of_targets dimension
    batch_indices = batch_indices.expand(-1, seq_len, number_of_targets)
    seq_indices = seq_indices.expand(batch_size, -1, number_of_targets)
    
    # Use advanced indexing to set the appropriate positions in the targets tensor to 1
    multi_hot[batch_indices, seq_indices, sequence] = 1
    return multi_hot

