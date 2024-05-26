from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import time

@dataclass
class ModelConfig:
    """
    Design your FutureFormer here
    Yes I know dropout_rate should probably be in TrainConfig but it was easier to implement from here
    """
    # general hyperparameters
    dim: int = 128
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' # can't do MPS bc metal doesn't support complex64 used in RoPE
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization

    # tokenizer
    tokenizer: str = 'bpe_v1' # must choose from one of the folders in 'tokenizers/'. current options: 'bpe_v1', 'bpe_v2'
    vocab_len: int = 8192 # options assuming 'bpe' are 95 (character-wise), 128, 256, 512, 1024, 2048, 4096, & 8192
        # ^ that number does not include the three tokens bos, eos, and pad

    # Residual Layers
    num_layers: int = 6 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    second_resid_norm: bool = False # True adds an extra Norm after the attn & MLP, like in Grok. Only recommended if using RMSNorm
    
    # Multi-Layer Perceptrion
    mlp_hidden_mult: int = 4 # how wide the hidden dimension of the MLP should be. common range is 2 to 8, usually 4
    mlp_bias: bool = False # whether to use bias weights. Llama3 does not and I'm partial to their choice
    mlp_nonlinearity: str = 'SiLU' # options are 'GeLU', 'SiLU', and 'ReLU'(not recommended)
    mlp_gated: bool = True # Turns SiLU into SwiGLU, GeLU into GeGLU, etc
        # ^ if gated == True, mlp_hidden_mult will automatically adjust to maintain parameter count

    # Self-Attention
    num_q_heads: int = 2 # `num_q_heads % num_kv_heads == 0` must be true
    num_kv_heads: int = 1 # set =num_q_heads to revert to regular multi-head attention (not recommended)
    head_dim: int = dim // num_q_heads # most common choices are 32, 64 and especially 128 bc those are what works with FlashAttention
    theta: float = 10_000 # 10_000 is the most common choice. Llama3 uses 50_000
    max_seq_len: int = 64 # 512 is the most my 8gb of ram can handle

    # Cross-Attention
    ca_num_q_heads: int = num_q_heads # is it worth messing around with cross-attention's parameters?
    ca_num_kv_heads: int = num_kv_heads
    ca_head_dim: int = dim // ca_num_q_heads

    # normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm'(recommended), 'LayerNorm', and 'CosineNorm'. Add more options in 'norm.py'
    norm_affine: bool = True # whether to use a linear layer after each norm. recommended especially if you're using LayerNorm or CosineNorm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-6 # small constant to prevent division by 0. Not really worth editing

    # Pooling Operation
    pool_type: str = 'sum' # options are 'sum', 'max', 'parametric_sum', 'parametric_max', 'flatten', 'conv', 'attention', 'self_attention'
        # 'sum' and 'max' have no learnable parameters and will ignore compress_freq, compress_freq_n, and pool_output_layer
        # 'flatten' will ignore pool_output_layer
    pre_pool_norm: bool = True # whether to norm embedding vectors before pooling. 
        # they do always get normed after pooling. the characteristics of the norms are defined above under # normalization
    pool_output_linear: bool = False # 
    compress_freq: str = 'linear' # options are 'constant', 'linear', 'root', 'log', and 'poly'
    compress_freq_n: int = 1 # defines compress_freq. below are recommended values & explanations
        # (compress_freq=='constant')&(compress_freq_n==1) -> y=n
            # for n=1, everyy block gets compressed into one single embedding vector
        # (compress_freq=='linear')&(compress_freq_n==1) -> y=nx+1
            # for n=1, 1st block gets 1 vector, 2nd gets 2, 3rd gets 3, etc 
        # (compress_freq=='root')&(compress_freq_n==2) -> y=floor(x**(1/n))
            # for n=2, 1st thru 3rd blocks get 1 vector, 4th thru 8th get 2, 9th thru 15th get 3, etc
        # (compress_freq=='log')&(compress_freq_n==2) -> y=floor(log_n(x))
            # for n=2, 1st gets 1 vector, 2nd thru 8th get 2, 9th thru 15th get 3, etc 
        # (compress_freq=='poly')&(compress_freq_n==1.2) -> y=x**n
            # for n=2, 1st gets 1 vector, 2nd gets 2, 3rd gets 3, 4th gets 5, 5th gets 7, 6th gets 9, etc 

    # Future Sight
    fs_mult_factor: int = 4 # sequence length of first set of future vectors to be pooled & the mult factor of each successive larger future time chunk
    fs_max_iter: int = 3 # maximum number of future chunks to look at
        # for fs_mult_factor=2 and fs_max_iter=6, we've got chunk sizes 2,4,8,16,32,64 for a total of 126 future-sight tokens
    assert max_seq_len >= sum([fs_mult_factor**i for i in range(fs_max_iter)]), f'future sight prediction chunks cannot be longer than max_seq_len'

    # inference (kv caching)
    max_batch_size: int = 1 # i haven't tried changing this from 1
    # it needs to be set >1 at the first model initialization if you ever want to be able to do batched inference. i should fix that
    # i think batched inference is probably broken rn bc of my shitty tokenizer. might fix in future

@dataclass
class TrainConfig:
    """
    Design your training loop here
    """
    # name of the folder the model will be saved into
    model_name = f'{time.strftime("%Y-%m-%d|%H-%M-%S")}' # defaults to the time that config.py was imported
    
    weight_decay: float = 0.05
    batch_size: int = 32
    
    # total number of batches to run over the course of training
    max_iters: int = 20 # i recommend at least 1_000
    # how often to print out an update on how training is going
    eval_interval: int = max_iters // 10 # doing this too often slows things down hella but also gives detailed log data
    # how many samples to take at each evaluation. more means a more accurate loss/perplexity calculation
    eval_samples: int = 1 # this number can slow things down. each sample is almost like doing an extra training iteration
    # how often to save a model checkpoint
    checkpoint_interval: int = None # eval_interval # set to None if you don't want checkpoints
    
    ### to visualize the learning rate schedule you define here, see cell 7 of training.ipynb

    # Initial learning rate to start from during the warmup
    lr_init: float = 1e-6
    # Maximum and minimum learning rates during annealing
    lr_max: float = 1e-1
    lr_min: float = 1e-3
    # if you'd like a flat learning rate, set lr_init = lr_min = lr_max and ignore the variables below
    
    # number of iterations for a linear warmup from lr_min to lr_max
    warmup_iters: int = int(max_iters * 0.01) # if you don't want to use a lr warmup, set = 0
    # number of iterations for a constant learning rate of lr_min at the end of training
    final_flat_iters: int = int(max_iters * 0.1) # if you don't want to use a final flat lr at the end, set = 0
    
    # type of annealment to use. Annealment is when the learning rate decreases over the course of training
    anneal_type: str = 'cos' # options: 'cos'(recommended) and 'lin'
    # number of times to bring the learning rate back up from lr_min to lr_max in-between the warmup & final flat
    num_restarts: int = 3 # if you don't want to use warm restarts, set =0 and ignore T_mult
    # relative length of each warm restart compared to the previous.
    T_mult: int = 2 # =1 makes all restarts the same length, <1 means they get shorter and >1 makes them longer
    
    # Calculates T_0 in a way that ensures smooth transition to the final flat learning rate
    def T_0(self): # I DO NOT RECOMMEND EDITING THIS
        middle_section = self.max_iters - self.warmup_iters - self.final_flat_iters
        return middle_section / sum(self.T_mult ** i for i in range(self.num_restarts+1))