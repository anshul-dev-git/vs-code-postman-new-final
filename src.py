import torch
import torch.nn.functional as F
import math

def get_dense_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k) #getting the score.
    #transposing the k matrix to get same dimensions
    if mask is not None:
        #removing those values which will come in future and are marked false by mark filter by swappig them with -infinity, which will be set - by softmax
        scores = scores.masked_fill(mask == False, float('-inf'))   
    probs = F.softmax(scores, dim=-1)
    # fix NaNs if an entire row is masked out (happens at block boundaries when there are no other numbers to look at)
    probs = torch.nan_to_num(probs, nan=0.0)
    
    return probs @ v, probs

def get_sliding_window_mask(seq_len, window_size):
    # coordinate grids
    r = torch.arange(seq_len).unsqueeze(1) #making a list of the items in the row
    c = torch.arange(seq_len).unsqueeze(0) #similar to whaat we did to rows
    
    # only the past word is assured by causal and within thhe distance is used by window
    causal = c <= r
    window = (r - c) < window_size
    
    return causal & window #both the conditions to be sastisfied
def get_bigbird_mask(seq_len, window_size=3, num_global=2, random_prob=0.05):
    # Coordinate grids
    r = torch.arange(seq_len).unsqueeze(1)
    c = torch.arange(seq_len).unsqueeze(0)
    # Base rule: No time travel allowed
    causal = c <= r
    # Rule 1: Local (Sliding Window)
    local_mask = (r - c) < window_size
    # Rule 2: Global (Always look at the first 'num_global' tokens)
    global_mask = c < num_global
    # Rule 3: Random (Look at a small percentage of random tokens)
    # torch.rand gives a matrix of decimals between 0.0 and 1.0. 
    # Checking < 0.05 gives us a 5% chance for any cell to be True.
    random_mask = torch.rand(seq_len, seq_len) < random_prob
    # Combine: Must be in the past AND (local OR global OR random)
    return causal & (local_mask | global_mask | random_mask)