---
title: NVIDIA AI Engineering
aliases: [NVIDIA, CUDA, GPU, NeMo, TensorRT]
tags: [company, hardware, cuda, llm-serving, distributed-training, triton]
related: [flash-attention, llm-serving-infra, kv-cache, transformer-architecture]
sources: [training-knowledge, nvidia-developer-blog, cuda-docs]
relevance: 9
last_updated: 2025-01-15
status: current
---

# NVIDIA AI Engineering

## Company Context

NVIDIA's $2T+ valuation is entirely predicated on AI. GPUs (A100, H100, H200, B200/Blackwell) dominate AI training and inference. NVIDIA's software stack (CUDA, cuDNN, TensorRT, NeMo, Triton Inference Server, RAPIDS) creates ecosystem lock-in. An NVIDIA AI engineer works on: GPU hardware/software co-design, LLM training/serving optimization, CUDA kernel development, and the TensorRT-LLM inference stack.

**Key products:** A100/H100/H200 GPUs, CUDA, cuDNN, TensorRT, Triton Inference Server, NeMo (LLM training framework), RAPIDS (GPU-accelerated data science), DGX systems, NVLink/NVSwitch.

---

## What NVIDIA AI Engineers Work On

### 1. GPU Architecture (Hardware-Software Co-design)

```
H100 SXM5 Specs:
- 80GB HBM3 (3.35 TB/s bandwidth)
- 989 TFLOPS FP16
- 1979 TFLOPS FP8 (with sparsity)
- 3958 TFLOPS INT8
- 900 GB/s NVLink (per GPU, in NVLink Switch system)
- 4th-gen Tensor Cores
- Transformer Engine (auto FP8/FP16 switching per layer)

Blackwell B200:
- 192GB HBM3e
- ~4 PFLOPs FP4
- FP8 training (first generation)
- 1.8 TB/s NVLink
```

**Key architectural concepts:**

```
H100 memory hierarchy:
  HBM3:        80 GB, 3.35 TB/s  (off-chip, high latency)
  L2 cache:    50 MB, ~20 TB/s
  SRAM (SM):   256 KB per SM × 132 SMs = 33 MB total, ~200 TB/s
  Registers:   256 KB per SM (fastest, private to threads)
```

Flash Attention was invented precisely to exploit this hierarchy — keeping data in fast SRAM rather than roundtripping to HBM.

### 2. TensorRT-LLM: Production LLM Serving

TensorRT-LLM is NVIDIA's optimized LLM inference library. Key techniques:

```python
# Key optimizations in TensorRT-LLM:

# 1. Continuous batching (in-flight batching)
#    - Requests join/leave the batch mid-generation
#    - GPU never idle waiting for full batch to finish

# 2. Paged KV cache (vLLM-style)
#    - KV cache in fixed-size "pages" (blocks)
#    - Eliminates memory fragmentation
#    - Enables efficient memory sharing for parallel decoding

# 3. Speculative decoding
#    - Draft model generates 4-5 tokens, target model verifies in one pass
#    - 2-3× throughput for latency-constrained serving

# 4. Quantization
#    - INT8 weights + FP16 activations (W8A16)
#    - FP8 (H100 only): 2× throughput vs FP16
#    - AWQ (Activation-aware Weight Quantization): best quality at INT4

# 5. Tensor parallelism
#    - Split attention heads across GPUs (column-parallel linear for QKV)
#    - All-reduce after each attention layer

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir(
    engine_dir="./engine",
    rank=0,
    max_batch_size=64,
    max_input_len=2048,
    max_output_len=512
)
```

### 3. CUDA Kernel Development

Custom CUDA kernels are the core of NVIDIA's competitive moat. Key concepts:

```c
// Flash Attention kernel sketch — shows GPU programming concepts
// Each thread block handles one Q tile

__global__ void flash_attention_fwd(
    float* Q, float* K, float* V, float* O,
    int N, int d, float scale
) {
    // Shared memory for this thread block's Q tile, K tile, V tile
    __shared__ float Q_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ float K_tile[BLOCK_SIZE][HEAD_DIM];
    
    int q_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Load Q tile from HBM to SRAM
    for (int d_idx = 0; d_idx < HEAD_DIM; d_idx++) {
        Q_tile[threadIdx.x][d_idx] = Q[q_idx * HEAD_DIM + d_idx];
    }
    __syncthreads();  // ensure all threads loaded before compute
    
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output[HEAD_DIM] = {0};
    
    // Iterate over K/V tiles (stays in SRAM, avoids HBM roundtrip)
    for (int kv_block = 0; kv_block < N / BLOCK_SIZE; kv_block++) {
        // Load K tile
        // Compute QK^T scores
        // Online softmax update
        // Accumulate output
    }
    
    // Write output back to HBM
}
```

### 4. NeMo: LLM Training Framework

```python
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.core.config import hydra_runner

# NeMo uses Megatron-LM for tensor parallelism + pipeline parallelism
# Training configuration
config = {
    "model": {
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 4,
        "num_layers": 96,
        "hidden_size": 12288,
        "num_attention_heads": 96,
    },
    "trainer": {
        "devices": 64,  # 64 GPUs
        "precision": "bf16-mixed",
        "gradient_clip_val": 1.0,
    }
}
```

---

## Key Concepts

### Memory Bandwidth vs Compute Bound

```
Arithmetic intensity = FLOPs / bytes_accessed

Attention (decode, batch=1):
  - Load K,V cache: 2 × n_layers × seq × d_model × 2 bytes
  - Compute: 2 × seq × d_model FLOPs per token
  - Intensity: very low → memory bandwidth bound

Matrix multiply (large batch):
  - A[M,K] × B[K,N]: 2MKN FLOPs
  - Memory: (MK + KN + MN) × 2 bytes  
  - For M=N=K=8192: intensity = 5461 FLOP/byte → compute bound

→ LLM inference at small batch: maximize memory bandwidth (HBM3 vs HBM2)
→ LLM training / large batch inference: maximize compute (TFLOPS)
```

### Quantization Details

| Method | Bits | Speedup | Quality loss | When |
|---|---|---|---|---|
| FP16 | 16 | baseline | none | training |
| BF16 | 16 | baseline | none | training (better dynamic range) |
| INT8 (W8A8) | 8 | 2× | minimal | inference, older GPUs |
| FP8 (W8A8) | 8 | 2× | minimal | H100/H200 (hardware support) |
| AWQ (W4A16) | 4 | 2-3× | small | production inference |
| GPTQ (W4A16) | 4 | 2-3× | small | community models |
| INT4 (W4A4) | 4 | 4× | moderate | edge/mobile |

---

## Key Questions

**Systems / Architecture:**
- "Explain the H100 memory hierarchy and how it affects LLM inference performance"
- "What is continuous batching and how does it improve GPU utilization?"
- "How does speculative decoding work? When does it help and when doesn't it?"
- "Explain tensor parallelism for transformer inference across 8 GPUs"
- "How does paged KV cache (from vLLM) work and why is it needed?"

**CUDA / Low-level:**
- "What is arithmetic intensity and how do you determine if a kernel is compute-bound or memory-bound?"
- "Explain the roofline model for GPU performance analysis"
- "What is warp divergence and how do you avoid it in CUDA kernels?"
- "How does Flash Attention use the GPU memory hierarchy to speed up attention?"

**ML Depth:**
- "Compare AWQ vs GPTQ quantization. Which preserves quality better?"
- "How does FP8 training work on H100? What's the training recipe?"
- "What is MoE (Mixture of Experts) and what are the GPU communication challenges?"

---

## Red Flags at NVIDIA

- **Not knowing GPU architecture:** NVIDIA engineers are expected to know the hardware deeply. "GPUs are fast" is not enough.
- **No CUDA experience:** Python-only ML experience may not be sufficient for core roles.
- **Ignoring memory bandwidth:** Most LLM inference bottlenecks are bandwidth, not compute.
- **Not knowing quantization methods:** AWQ, GPTQ, FP8 — must know the trade-offs.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | H100/A100 architecture: SM, SRAM, HBM, Tensor Cores, NVLink |
| 2 | Flash Attention: IO analysis, tiling, SRAM usage, FA1/FA2/FA3 |
| 3 | LLM inference: continuous batching, paged KV cache, speculative decoding |
| 4 | Tensor/pipeline/data parallelism for training and serving |
| 5 | Quantization: INT8/FP8/AWQ/GPTQ — algorithms and trade-offs |
| 6 | CUDA programming basics: threads, blocks, shared memory, memory coalescing |
| 7 | System design: LLM serving system at scale, multi-GPU inference |
