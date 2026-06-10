# ScenePilot: Retrieval-Augmented Grow-and-Repair for Text-Driven 3D Indoor Scene Generation

ScenePilot is a text-driven 3D indoor scene generation framework designed to synthesize physically plausible, semantically coherent, and functionally complete indoor layouts from natural language descriptions.

The project builds upon the ReSpace-style structured scene representation and introduces a retrieval-augmented grow-and-repair pipeline. Given a room description such as *“a cozy bedroom with a bed, two nightstands and a desk”*, ScenePilot progressively constructs the scene using retrieved spatial priors, group-wise object growth, lightweight physical optimization, and vision-language-model-based scene repair.

## Overview

Text-driven 3D indoor scene generation is challenging because natural language descriptions often omit detailed object relations, functional constraints, and precise spatial layouts. Direct generation methods may produce object collisions, out-of-bound placements, missing functional objects, or semantically unreasonable arrangements.

ScenePilot addresses these issues through three main components:

1. **Retrieval-Augmented Prior Grounding**
   Retrieves room-level and anchor-level spatial priors from a mined scene memory to guide object selection and layout planning.

2. **Group-wise Scene Growth**
   Decomposes a room into functional object groups and generates the scene progressively, rather than predicting the full layout in one step.

3. **Vision-Guided Repair Loop**
   Uses rendered visual feedback and structured scene metrics to iteratively repair layout errors through executable actions such as move, rotate, and scale.

## Key Features

* Text-driven 3D indoor scene synthesis
* Retrieval-augmented generation with FAISS-based spatial prior memory
* Group-wise autoregressive scene construction
* Local deterministic optimization after each object group
* VLM-based final scene repair
* Structured JSON scene representation
* Support for move, rotate, and scale editing actions
* Physical plausibility evaluation using voxelization-based metrics
* SFT and GRPO training pipeline for layout repair
* Evaluation with physical, relational, functional, and VLM-based metrics

## Pipeline

<p align="center">
  <img src="docs/figures/pipeline.svg" alt="ScenePilot Pipeline" width="900">
</p>

## Method

### 1. Retrieval-Augmented Prior Grounding

ScenePilot first retrieves useful spatial priors from a pre-built RAG memory. The memory is constructed from mined object groups and contains information such as:

* common object co-occurrence patterns
* anchor-object relationships
* preferred object distances
* local offsets between related objects
* room-specific object group priors

The retrieved priors help compensate for missing spatial information in short or underspecified text prompts.

Example:

```text
Input: "a bedroom with a bed, desk and chair"

Retrieved priors:
- A bed is often paired with nightstands and a rug.
- A desk usually appears with a chair.
- A chair is commonly placed near the front side of the desk.
```

### 2. Group-wise Scene Growth

Instead of generating all objects at once, ScenePilot decomposes the scene into several functional groups. Each group contains an anchor object and its related companion objects.

Examples:

```text
Bedroom:
- Sleeping group: bed, nightstand, rug
- Working group: desk, chair, lamp
- Storage group: wardrobe, dresser

Living room:
- Seating group: sofa, coffee table, side table
- Media group: TV stand, television
- Decoration group: plant, shelf, rug
```

The system inserts one group at a time. After each group is generated, a lightweight optimizer adjusts the layout to reduce physical violations.

### 3. Deterministic Physical Optimization

After each group is placed, ScenePilot applies a simple deterministic optimization strategy, including:

* anchor snapping
* collision pushing
* out-of-bound projection
* local position adjustment

The optimizer mainly reduces physical layout errors before the next group is generated.

### 4. VLM-based Scene Repair

After the full scene is generated, a vision-language model receives rendered views of the scene and predicts structured repair actions.

Supported actions include:

```json
{
  "action": "move",
  "object_index": 3,
  "dx": 0.2,
  "dy": 0.0,
  "dz": -0.1
}
```

```json
{
  "action": "rotate",
  "object_index": 2,
  "yaw_deg": 90
}
```

```json
{
  "action": "scale",
  "object_index": 5,
  "sx": 1.1,
  "sy": 1.0,
  "sz": 0.9
}
```

The repair action is accepted only when it improves the scene metrics.

## Evaluation Metrics

ScenePilot evaluates generated scenes from both physical and semantic perspectives.

### Physical Metrics

* **OOB**: Out-of-bound loss
* **MBL**: Mesh-based collision loss
* **PBL**: Physical-based loss

```text
PBL = OOB + MBL
```

A lower PBL indicates better physical plausibility.

### Relational Metric

* **REL** evaluates whether object-object spatial relations are reasonable.
* Example: a chair should be close to and facing a desk.

### Functional Metric

* **FUNC** evaluates whether the layout satisfies functional usage constraints.
* Example: a bed should have accessible space, and a chair should be usable with a desk.

### VLM-based Metrics

A vision-language model is used to judge the rendered scene from three aspects:

* **LC**: layout correctness
* **SPA**: semantic plausibility
* **FC**: functional completeness
* **Overall**: average score of LC, SPA, and FC

## Training

ScenePilot includes a two-stage training pipeline for the visual repair model.

### Supervised Fine-Tuning

The SFT stage teaches the model to imitate repair actions from constructed scene trajectories.

Input:

* diagonal rendered image
* annotated top-down image
* current scene JSON

Output:

* structured JSON repair action

### GRPO Training

The GRPO stage further improves the model through reward-based optimization.

The final reward is defined as:

```text
R = 0.15 R_format + 0.15 R_apply + 0.50 R_phys + 0.20 R_vlm
```

Where:

* `R_format`: whether the output is valid JSON
* `R_apply`: whether the action can be executed
* `R_phys`: whether physical violations are reduced
* `R_vlm`: whether the visual quality improves according to VLM judgment

## Dataset

The project uses indoor scene data derived from 3D-FRONT and 3D-FUTURE assets.

Main data components include:

* cleaned benchmark scenes
* reverse construction trajectories
* scale-only repair trajectories
* rendered visual feedback images
* RAG spatial prior memory
* SFT training data
* GRPO training data

Supported room types include:

* bedroom
* living room
* dining room
* library
* laundry room

## Project Structure

```text
ScenePilot/
├── data/
│   ├── benchmark/
│   ├── training_data/
│   └── rag_memory/
│
├── assets/
│   └── 3D-FUTURE-model/
│
├── src/
│   ├── planning/
│   ├── rag/
│   ├── growth/
│   ├── repair/
│   ├── optimization/
│   ├── evaluation/
│   └── visualization/
│
├── scripts/
│   ├── build_rag_index.py
│   ├── run_generation.py
│   ├── run_repair.py
│   ├── train_sft.py
│   ├── train_grpo.py
│   └── eval_batch.py
│
├── results/
│   ├── batch_outputs/
│   ├── renders/
│   └── metrics/
│
├── README.md
└── requirements.txt
```

## Installation

Create a Python environment:

```bash
conda create -n scenepilot python=3.10
conda activate scenepilot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install additional packages if needed:

```bash
pip install torch torchvision
pip install transformers accelerate peft trl
pip install faiss-cpu
pip install trimesh open3d
```

For GPU training, please install the PyTorch version that matches your CUDA version.

## Build RAG Index

Before generation, build the retrieval index from mined group priors:

```bash
python scripts/build_rag_index.py \
  --input data/rag_memory/rag_templates.jsonl \
  --output data/rag_memory/faiss_index \
  --embedding_model /path/to/qwen3-embedding-8B
```

## Run Scene Generation

Generate a scene from a text prompt:

```bash
python scripts/run_generation.py \
  --prompt "a cozy bedroom with a queen-size bed, two nightstands, a desk and a chair" \
  --room_type bedroom \
  --output results/demo_bedroom
```

The output folder contains:

```text
results/demo_bedroom/
├── scene.json
├── renders/
│   ├── diag.jpg
│   ├── top.jpg
│   └── annotated_top.jpg
└── metrics.json
```

## Run VLM Repair

Apply final visual repair to a generated scene:

```bash
python scripts/run_repair.py \
  --scene results/demo_bedroom/scene.json \
  --image results/demo_bedroom/renders/diag.jpg \
  --annotated_top results/demo_bedroom/renders/annotated_top.jpg \
  --model /path/to/qwen3-sft-grpo \
  --output results/demo_bedroom_repaired
```

## Evaluation

Evaluate a batch of generated scenes:

```bash
python scripts/eval_batch.py \
  --input results/batch_outputs \
  --output results/metrics/eval_results.json
```

The evaluation reports:

* OOB
* MBL
* PBL
* REL
* FUNC
* valid scene ratio
* VLM-based LC / SPA / FC / Overall

## Example Output

Input prompt:

```text
A cozy bedroom with a queen-size bed, two nightstands, a desk, a chair, a wardrobe and a rug.
```

Generated scene:

```text
- The bed is placed against the wall.
- Nightstands are placed on both sides of the bed.
- The desk and chair form a working area.
- The wardrobe is placed near the wall.
- The rug is placed under the bed area.
- Object collisions and out-of-bound violations are reduced through repair.
```

## Main Contributions

* A retrieval-augmented framework for text-driven 3D indoor scene generation.
* A group-wise grow-and-repair pipeline that improves layout stability.
* A lightweight physical optimizer for progressive object placement.
* A VLM-based repair model trained with SFT and GRPO.
* A unified evaluation protocol combining physical, relational, functional, and visual-semantic metrics.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{scenepilot2026,
  title  = {ScenePilot: Retrieval-Augmented Grow-and-Repair for Text-Driven 3D Indoor Scene Generation},
  author = {Anonymous},
  year   = {2026},
  note   = {Project repository}
}
```

## Acknowledgements

This project builds on recent progress in text-driven 3D indoor scene generation, vision-language reasoning, retrieval-augmented generation, and reinforcement learning for spatial layout optimization.

The project uses indoor scene assets and layouts derived from public 3D scene datasets and follows the structured scene representation paradigm for controllable 3D scene synthesis and editing.

## License

This project is released for academic research purposes only. Please check the licenses of the original datasets, pretrained models, and 3D assets before redistribution or commercial use.
