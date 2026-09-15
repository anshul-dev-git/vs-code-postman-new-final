# Sparse Attention from Scratch

**Postman AI/ML Recruitment Task - 25 Batch**
**Author:** Anshul Mittal 
**Hardware:** NVIDIA GeForce RTX 4050


## 1. Implementation & Correctness (Deliverables 1.1 – 1.3)

### The Dense Baseline
To establish a ground truth, I implemented a manual dense attention forward pass ($Q K^T / \sqrt{d_k}$) combined with a standard causal mask and softmax[cite: 1]. This explicitly bypasses PyTorch's optimized `F.scaled_dot_product_attention` to ensure the core linear algebra serves as an un-abstracted reference[cite: 1].

### Sparsity Patterns
I implemented two distinct sparsity patterns using vectorized tensor broadcasting to avoid slow control-flow loops[cite: 1]:
*   **Sliding Window:** A strictly local pattern where tokens can only attend to a fixed historical window (e.g., $N=32$)[cite: 1]. 
*   **BigBird-Style (Block-Sparse):** A composite pattern combining a local sliding window, a small set of global tokens at the start of the sequence, and a 5% random token sample to create attention shortcuts[cite: 1].

### Correctness Harness
To verify the math, the correctness harness tests both sparse variants against the dense reference on unmasked positions[cite: 1]. By asserting `torch.allclose` with a `1e-5` tolerance, the harness mathematically proves that the manual masking and scaling logic perfectly matches PyTorch's native C++ backend[cite: 1].

---

## 2. Edge Cases & NaN Handling (Deliverable 1.4)

When building sparse attention, an architectural edge case emerges that does not exist in standard dense transformers. When strict sparsity constraints (like a sliding window) overlap with a causal mask, it is possible for a query token's entire allowed attention set to be masked out[cite: 1].

*   When an entire row is masked, the `masked_fill` operation replaces all raw scores with $-\infty$[cite: 1]. 
*   Passing a uniform row of $-\infty$ through the softmax function results in an attempt to divide zero by zero, which physically crashes the tensor state into `NaN` (Not a Number)[cite: 1]. This happens for real at block boundaries[cite: 1]. 

**The Fix:** I handled this by injecting `torch.nan_to_num(probs, nan=0.0)` immediately after the softmax operation. This catches any completely masked rows and forces them to a clean `0.0`, ensuring those tokens safely contribute zero information to the value matrix without poisoning the network's forward or backward pass.

---

## 3. Benchmarking & The Materialization Trap (Deliverable 1.5)

The benchmarking suite measured wall-clock time and peak memory allocation across sequence lengths from 512 up to 8192[cite: 1]. The benchmark revealed a critical difference between *algorithmic* sparsity and *hardware* sparsity.

While the algorithmic masking successfully restricted sequence dependency, the peak VRAM consumption remained virtually identical across Dense, Sliding Window, and BigBird variants[cite: 1].

**Why this happens:** Under the hood, PyTorch executes the matrix multiplication first. This physically allocates the full $O(N^2)$ memory grid in VRAM *before* the boolean mask is applied via `masked_fill`. We successfully prevent the blocked data from altering the math, but we still pay the upfront VRAM cost to hold the empty array. 

To realize true memory savings, this masking logic must be pushed down into a custom fused GPU kernel (e.g., via Triton or CUDA C++). A custom kernel would compute the block-sparse tiles directly in the GPU's fast SRAM and write the output back to HBM without ever instantiating the global dense matrix. 

---

## 4. Quality Evaluation & Information Loss (Deliverables 1.6 – 1.7)

To evaluate the empirical cost of sparsity, I trained a 2-layer character-level GPT on the TinyShakespeare dataset[cite: 1]. The networks were trained on a 256-token context window with a sparse window restriction of 32 tokens. 

**Performance Hierarchy (Cross-Entropy Loss):**
1.  **Dense Baseline (Best):** Retained perfect memory of the sequence[cite: 1].
2.  **BigBird-Style:** Near-dense performance[cite: 1].
3.  **Sliding Window (Worst):** Noticeable degradation in validation loss[cite: 1].

**Which pattern loses what information?**
The strict Sliding Window suffers from induced architectural amnesia[cite: 1]. By rigidly cutting off long-range dependencies, the network is physically blocked from utilizing early-sequence semantic context. By the time it generates the end of a sentence, it has mathematically forgotten the subject at the beginning.

**Why global tokens matter disproportionately:**
The BigBird pattern dramatically closed the loss gap with the dense baseline because of its global tokens[cite: 1]. These first few tokens act as critical "attention sinks." Because every subsequent token in the sequence is permitted to attend to them, they naturally learn to aggregate sequence-wide context and broadcast it forward. They act as an information bridge, neutralizing the blind spots created by the local window and proving that you don't need $O(N^2)$ compute to maintain global context[cite: 1].