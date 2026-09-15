# Sparse Attention from Scratch 

**Author:** Anshul Mittal
**Task:** Postman AI/ML Recruitment Task (25 Batch) - Task 1

This repository contains a manual, from-scratch implementation of sparse attention mechanisms (Sliding Window and BigBird-style block-sparse). It bypasses high-level framework wrappers like `F.scaled_dot_product_attention` to directly manipulate the linear algebra and masking logic, evaluating the empirical costs and benefits of algorithmic sparsity.

## Hardware Tested
- **GPU:** NVIDIA GeForce RTX 4050 (6GB VRAM)
- **Environment:** Local execution (CUDA enabled)

## Repository Structure
- `src.py`: Contains the core mathematical implementations for dense attention, the sliding window mask, and the BigBird (local + global + random) mask.
- `harness.py`: The correctness test. Verifies that the manual sparse outputs perfectly match PyTorch's native C++ backend on unmasked positions.
- `benchmark.py`: Measures wall-clock time and peak VRAM allocation for dense vs. sparse variants across sequence lengths (512 -> 8192). Generates `benchmark_plot.png`.
- `tiny_gpt.py`: A minimal 2-layer character-level GPT. Injects the custom attention math to train on TinyShakespeare and evaluates the physical information loss of each sparsity pattern.
- `WRITEUP.md`: The theoretical breakdown of NaN handling, the memory materialization trap, and global token information retention.

## Dependencies
Ensure you have the following installed in your Python environment:
```bash
pip install torch torchvision torchaudio matplotlib requests
