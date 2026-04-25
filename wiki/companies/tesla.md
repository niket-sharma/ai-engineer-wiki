---
title: Tesla AI Engineering
aliases: [Tesla, Autopilot, FSD, Dojo, Tesla AI]
tags: [company, autonomous-driving, computer-vision, video-prediction, custom-silicon]
related: [transformer-architecture, llm-serving-infra, flash-attention]
sources: [training-knowledge, tesla-ai-day, tesla-research]
relevance: 7
last_updated: 2025-01-15
status: current
---

# Tesla AI Engineering

## Company Context

Tesla's AI team (based in Palo Alto, ~200 ML engineers) builds Autopilot and Full Self-Driving (FSD). This is unique among tech companies: pure perception → planning → control pipeline with real-world consequences (lives at stake). Tesla's fleet of 5M+ cars generates ~1 petabyte of video per day. Andrej Karpathy (ex-Tesla AI director) described it as "the most difficult AI problem in the world."

**Key products:** Autopilot (lane keeping, adaptive cruise), FSD (full autonomy in supervised mode), Dojo (custom AI training supercomputer), FSD Computer (HW3/HW4 inference chip), Tesla Bot (Optimus).

**What makes Tesla different:** 
- **No LiDAR:** Vision-only (8 cameras) vs competitors using LiDAR
- **Fleet learning:** 5M cars generate training data at scale no competitor can match
- **Vertical integration:** Custom chip (FSD Computer), custom supercomputer (Dojo), custom software (FSD stack)

---

## What Tesla AI Engineers Work On

### 1. Full Self-Driving (FSD) Perception Pipeline

The FSD perception system processes video from 8 cameras simultaneously:

```
8 cameras (1.2MP each, 36 fps)
      ↓
HW4 FSD Computer (72 TOPS, 2× neural processing units)
      ↓
Video backbone (processes temporal sequences, not single frames)
      ↓
BEV (Bird's Eye View) transformer
  - Projects 8-camera features into unified top-down 3D space
  - No explicit depth estimation — geometry learned end-to-end
      ↓
Detection heads:
  - Objects (vehicles, pedestrians, cyclists)
  - Road/lane geometry
  - Traffic signs + signals
  - Occupancy networks (3D voxel-level scene understanding)
      ↓
Planner: trajectory prediction → motion planning → control
```

**Key architectural shift (2022–2023):** Tesla moved from a modular pipeline (detect objects → localize → plan) to an end-to-end neural network that directly predicts driving trajectories from raw video.

### 2. Occupancy Networks

Occupancy networks replaced traditional object detection boxes (2023):

```python
# Traditional approach: predict bounding boxes
# Problem: doesn't handle irregular shapes, debris, unknown objects

# Occupancy network approach: predict 3D voxel occupancy grid
# Each voxel is classified: free | occupied | occluded

class OccupancyHead(nn.Module):
    def __init__(self, bev_channels: int, voxel_resolution: tuple[int, int, int]):
        super().__init__()
        H, W, Z = voxel_resolution  # e.g., 200×200×16 voxels
        self.voxel_resolution = voxel_resolution
        self.head = nn.Sequential(
            nn.Conv2d(bev_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, Z * 3, 1),  # 3 classes per voxel height
        )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        # bev_features: [B, C, H, W]
        B, C, H, W = bev_features.shape
        H_vox, W_vox, Z_vox = self.voxel_resolution
        
        logits = self.head(bev_features)  # [B, Z*3, H, W]
        # Reshape to voxel grid
        logits = logits.view(B, Z_vox, 3, H_vox, W_vox)
        # logits[b, z, class, h, w] = score for voxel (h, w, z) being of `class`
        return logits

# Why occupancy networks win:
# - Handles any object shape (not limited to boxes)
# - Handles debris, roadkill, unusual obstacles
# - Naturally handles occlusion (occluded voxels predicted via context)
# - Unified representation for all "stuff" in the scene
```

### 3. Video Transformers and Temporal Modeling

FSD processes video (sequences of frames) not individual images:

```python
class VideoTransformerBackbone(nn.Module):
    """
    Process T frames from N cameras into spatiotemporal feature volume.
    Tesla calls this the "video package" — key insight from 2022 AI Day.
    """
    def __init__(self, n_cameras: int = 8, n_frames: int = 8, d_model: int = 512):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(d_model)  # CNN per frame
        self.temporal_attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
        )
        # Cross-camera fusion via cross-attention
        self.camera_fusion = nn.MultiheadAttention(d_model, 8, batch_first=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, N_cameras, T_frames, C, H, W]
        B, N, T, C, H, W = frames.shape
        
        # Encode each frame spatially
        spatial_features = self.spatial_encoder(frames.view(B*N*T, C, H, W))
        spatial_features = spatial_features.view(B, N, T, -1, self.d_model)
        
        # Attend over time (within each camera)
        temporal_features = self.temporal_attn(
            spatial_features.view(B*N, T*H_feat*W_feat, self.d_model)
        )
        
        return temporal_features
```

### 4. Dojo: Tesla's Custom AI Supercomputer

Dojo (announced 2021, first ExaPOD in 2023) is Tesla's purpose-built training cluster:

```
Dojo D1 chip:
- 354 TFLOPS BF16
- 4× faster on-chip bandwidth vs A100
- 576 GB/s chip-to-chip bandwidth (on training tile)
- Custom ISA (not CUDA) — runs TensorFlow, not PyTorch

Dojo ExaPOD:
- 120 training tiles × 25 D1 chips = 3000 D1 chips per ExaPOD
- 1.1 ExaFLOP BF16 compute
- 1.3 TB/s aggregate bandwidth between tiles
- Optimized for video: large batch multi-camera training

Why custom silicon?
- NVIDIA GPUs general-purpose; Dojo optimized for Tesla's exact workloads
- Bandwidth-optimized for streaming video data (vs compute-heavy LLMs)
- Long-term: avoid $1B/year NVIDIA GPU bills
```

### 5. Data Engine and Auto-Labeling

With 1PB/day of video data, manual labeling is impossible:

```
Tesla Data Engine (closed-loop data flywheel):

1. Fleet triggers: FSD logs "interesting" scenarios based on:
   - Uncertainty in model predictions
   - Disengagement events (human override)
   - Rare scene detections (specific traffic light types, unusual objects)

2. Auto-labeling pipeline:
   - Run multiple specialized models on the clip
   - 3D reconstruction using Structure-from-Motion on fleet data
   - Weak supervision: combine multiple imperfect labels
   - Human verification on samples / hard cases only

3. Curation: select for diversity (don't re-train on common easy cases)

4. Retrain: periodic retraining with expanded dataset

5. Evaluate: shadow mode testing (FSD runs in background, compares to human)

Shadow mode metrics:
- Intervention rate: how often human overrides FSD
- "Miles per intervention" as the key safety metric
```

---

## Key Questions

**Computer Vision / Perception:**
- "Why did Tesla abandon LiDAR? What are the engineering trade-offs?"
- "How do you project 8 camera views into a unified Bird's Eye View?"
- "Explain occupancy networks vs bounding box detection — when does each win?"
- "How do you handle occlusion in 3D object detection?"
- "What is the challenge of detecting objects at night, rain, or direct sunlight?"

**ML Systems:**
- "Design Tesla's auto-labeling data pipeline for video data"
- "How would you build a 'shadow mode' evaluation system for autonomous driving?"
- "What metrics would you use to measure FSD safety?"
- "How do you handle distribution shift when deploying in a new country?"

**Architecture:**
- "How does a video transformer differ from an image transformer?"
- "Why use BEV (Bird's Eye View) representation for autonomous driving?"
- "How does Tesla's custom Dojo chip differ from a standard GPU? What trade-offs?"

---

## Red Flags at Tesla

- **Safety-dismissive:** Autonomous driving has life-or-death consequences. "We'll ship fast and fix bugs" does not work here.
- **No computer vision depth:** Tesla is pure vision. Knowing only LLMs / NLP is insufficient.
- **Not knowing temporal modeling:** Single-frame object detection is solved; video-level reasoning is the hard part.
- **Ignoring edge cases:** Tesla's system needs to handle snow, fog, faded lane markings, construction zones — edge cases are the product.

---

## 7-Day Learning Path

| Day | Focus |
|---|---|
| 1 | 3D object detection: DETR, BEV projection, coordinate transforms |
| 2 | Occupancy networks: voxel grids, signed distance functions, 3D representations |
| 3 | Video transformers: temporal attention, spatiotemporal features |
| 4 | Autonomous driving stack: perception → prediction → planning → control |
| 5 | Data flywheels: auto-labeling, weak supervision, active learning |
| 6 | System design: FSD data pipeline, shadow mode evaluation |
| 7 | Coding: PyTorch (3D convs, attention masks), numpy geometry operations |
