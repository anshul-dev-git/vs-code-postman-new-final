import time
import torch
import matplotlib.pyplot as plt
from src import get_dense_attention, get_sliding_window_mask, get_bigbird_mask

def run_benchmark():
    print("Running Benchmark")
     #getting the hardware on which it ran
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Device: {gpu_name}")
    else:
        device = "cpu" #if it runs on a cpu
        print("Hardware Device: CPU")
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    batch_size = 1
    num_heads = 4
    d_k = 64
    window_size = 128
    # Storage arrays for all three variants
    dense_times, dense_mems = [], []
    sliding_times, sliding_mems = [], []
    bigbird_times, bigbird_mems = [], []
    
    for seq_len in seq_lengths:
        print(f"\nTesting Sequence Length: {seq_len}")
        # Setup Mock Data
        q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
        
        # Generate both masks
        sliding_mask = get_sliding_window_mask(seq_len, window_size).to(device)
        bigbird_mask = get_bigbird_mask(seq_len, window_size=window_size, num_global=16, random_prob=0.05).to(device)
        
        # Dense Baseline Run
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_time = time.time()
        _ = get_dense_attention(q, k, v, mask=None)
        torch.cuda.synchronize()
        
        dense_times.append(time.time() - start_time)
        dense_mems.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        
        #Sliding Window Run
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_time = time.time()
        _ = get_dense_attention(q, k, v, mask=sliding_mask)
        torch.cuda.synchronize()
        
        sliding_times.append(time.time() - start_time)
        sliding_mems.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        
        #BigBird-Style Run
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start_time = time.time()
        _ = get_dense_attention(q, k, v, mask=bigbird_mask)
        torch.cuda.synchronize()
        
        bigbird_times.append(time.time() - start_time)
        bigbird_mems.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        
        # Print results for this sequence length
        print(f"  Dense   -> Time: {dense_times[-1]:.4f}s | Mem: {dense_mems[-1]:.2f} MB")
        print(f"  Sliding -> Time: {sliding_times[-1]:.4f}s | Mem: {sliding_mems[-1]:.2f} MB")
        print(f"  BigBird -> Time: {bigbird_times[-1]:.4f}s | Mem: {bigbird_mems[-1]:.2f} MB")  
    # Plotting the Results
    print("\nGenerating benchmark_plot.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Wall-Clock Time
    ax1.plot(seq_lengths, dense_times, label='Dense', marker='o')
    ax1.plot(seq_lengths, sliding_times, label='Sliding Window', marker='s')
    ax1.plot(seq_lengths, bigbird_times, label='BigBird-Style', marker='^')
    ax1.set_title('Forward Pass: Wall-Clock Time')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Seconds')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Peak Memory
    ax2.plot(seq_lengths, dense_mems, label='Dense', marker='o')
    ax2.plot(seq_lengths, sliding_mems, label='Sliding Window', marker='s')
    ax2.plot(seq_lengths, bigbird_mems, label='BigBird-Style', marker='^')
    ax2.set_title('Forward Pass: Peak Memory')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Megabytes (MB)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_plot.png')
    print("image saved in a new folder.")

if __name__ == "__main__":
    run_benchmark()