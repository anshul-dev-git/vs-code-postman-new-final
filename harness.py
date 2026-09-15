import torch
import torch.nn.functional as F
from src import get_dense_attention, get_sliding_window_mask, get_bigbird_mask

def run_correctness_harness():
    print("running the correctness harness")
    
    # Lock the seed so random numbers are the same every time we test
    torch.manual_seed(18)
    
    # setting up standard transformer parameters
    batch_size = 2
    num_heads = 4
    seq_len = 16
    d_k = 64
    window_size = 3
    
    # Generating mock random data to test
    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    # Test 1: Sliding Window Pattern
    print("\n[Test 1] Verifying Sliding Window Mask")
    sliding_mask = get_sliding_window_mask(seq_len, window_size)
    
    # running the manual function i made
    manual_sliding, _ = get_dense_attention(q, k, v, mask=sliding_mask)
    # Run PyTorch's official standard
    ref_sliding = F.scaled_dot_product_attention(q, k, v, attn_mask=sliding_mask)
    
    if torch.allclose(manual_sliding, ref_sliding, atol=1e-5):
        print("PASS: Manual sliding window perfectly matches reference.")
    else:
        print("FAIL: Outputs do not match.")

    # Test 2: BigBird-Style Pattern
    print("\n[Test 2] Verifying BigBird-Style Mask")
    bigbird_mask = get_bigbird_mask(seq_len, window_size=3, num_global=2, random_prob=0.1)
    
    manual_bigbird, _ = get_dense_attention(q, k, v, mask=bigbird_mask)
    ref_bigbird = F.scaled_dot_product_attention(q, k, v, attn_mask=bigbird_mask)
    
    if torch.allclose(manual_bigbird, ref_bigbird, atol=1e-5):
        print("PASS: Manual BigBird attention perfectly matches reference.")
    else:
        print("FAIL: Outputs do not match.")

if __name__ == "__main__":
    run_correctness_harness()