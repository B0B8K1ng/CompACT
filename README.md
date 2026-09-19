# Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model

**CVPR 2026**

[Dongwon Kim](https://kdwonn.github.io)<sup>1</sup>, [Gawon Seo](https://www.linkedin.com/in/gawon-seo-9a3588279/)<sup>2</sup>, [Jinsung Lee](https://jinsingsangsung.github.io/)<sup>2</sup>, [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/)<sup>2,3</sup>, [Suha Kwak](https://suhakwak.github.io/)<sup>2</sup>

<sup>1</sup>KAIST &nbsp; <sup>2</sup>POSTECH &nbsp; <sup>3</sup>RLWRLD

[[Project Page](https://kdwonn.github.io/CompACT)] [[Paper](https://arxiv.org/abs/2603.05438)]

## Overview

This repo consists of two main training pipelines:
1. **Tokenizer training**: Train CompACT, compact tokenizer that compress image up to 8 discrete tokens
2. **World Model Training**: Trains the CDiT model using the learned tokenizers for navigation prediction


## Quick Start

This project uses `uv` for dependency management. Simply run:

```bash
# Install dependencies (first time only)
uv sync

# Run any script
uv run <script>

# Examples:
uv run train_tokenizer.py  # Train tokenizer
uv run bash scripts/train.sh --nproc=4 -- ++tokenizer_path=<TOKENIZER_DIR>  # Train world model
```

## Data & Checkpoint Preparation

- Requires ImageNet in webdataset format for tokenizer training. For navigation dataset, see original [NWM repo](https://github.com/facebookresearch/nwm/).
- Download following checkpoints: [MAGE](https://github.com/LTH14/mage), [DINOv3](https://github.com/facebookresearch/dinov3)
- Set following environment variables for ckpt and dataset dirs: `DATASET_PREFIX`, `BASE_TOKENIZER_CKPT`

## Architecture

### Tokenizer Training Pipeline

```bash
# Train tokenizer with default configuration
uv run train_tokenizer.py

# With custom image size (default: 224)
uv run train_tokenizer.py ++dataset.image_size=256 ++dataset.dinov2_image_size=256
```

### World Model Training Pipeline

The main CDiT model supports diverse tokenizer architectures and is trained on navigation data:

#### Supported Tokenizers

The codebase supports multiple tokenizer types (`conf/model/tokenizer/`):
- **CompactTok**: Efficient tokenization with extreme compression (proposed)
- **FlexTok**: Flexible tokenization with variable token length
- **SDVAE**: Stable Diffusion VAE-based tokenization

```bash
# Train world model (requires tokenizer checkpoint)
uv run bash scripts/train.sh --nproc=4 -- ++tokenizer_path=<TOKENIZER_DIR>
```

## Configuration System

The project uses [Hydra](https://hydra.cc/) for configuration management with two separate configuration directories:

### Main Training Configuration (`conf/`)
```
conf/
├── config.yaml              # Main configuration file
├── model/
│   ├── generator/          # CDiT model configurations (B, B_disc, L_disc, XL)
│   ├── tokenizer/          # Tokenizer configurations
│   └── diffusion/          # Diffusion configurations (gaussian, discrete, discrete_history)
├── training/               # Training configurations
├── dataset/                # Dataset configurations
└── scheduler/              # LR scheduler configurations
```

### Tokenizer Training Configuration (`conf_tokenizer/`)
```
conf_tokenizer/
├── config.yaml             # Tokenizer training configuration
├── model/
│   ├── compact_tokenizer/  # CompactTokenizer (DINOv3, ViT-based, RIN) configs
│   ├── base_tokenizer/     # Base tokenizer (MAGE, OpenMAGViT) configs
│   ├── diffusion/          # Discrete diffusion configs
│   └── rep_guidance/       # Representation guidance loss configs
├── training/               # Training configurations
├── dataset/                # Dataset configs (ImageNet, etc.)
└── scheduler/              # LR scheduler configurations
```

## Training

### 1. Tokenizer Training

Train a tokenizer to learn efficient visual representations. Defaults use DINOv3-B QFormer encoder with MMDiT-L decoder and discrete diffusion. Trained using 8 H100 GPUS.

```bash
# Single GPU training with defaults
uv run train_tokenizer.py

# With 256x256 images
uv run train_tokenizer.py \
  ++dataset.image_size=256 \
  ++dataset.dinov2_image_size=256
```

### 2. World Model Training

Train the CDiT world model. Defaults use CDiT-B with maksed token modeling + history masking and CompactTok tokenizer. Trained using 4 RTX 6000 ada GPUs.

```bash
# Distributed training (requires tokenizer checkpoint)
uv run bash scripts/train.sh --nproc=4 -- \
  ++tokenizer_path=<TOKENIZER_DIR>

# With larger model
uv run bash scripts/train.sh --nproc=8 -- \
  ++tokenizer_path=<TOKENIZER_DIR> \
  model/generator=cdit_xl
```

## Planning Evaluation

```bash
# Run planning evaluation
uv run bash scripts/plan.sh --nproc=1 -- \
  ++exp_dir=<WORLD_MODEL_DIR>

# Multi-GPU with custom checkpoint
uv run bash scripts/plan.sh --nproc=4 -- \
  ++exp_dir=<WORLD_MODEL_DIR> \
  ++ckp="latest"
```

## Reproducible NWM Benchmark Suite

The benchmark runner evaluates registered checkpoints with one shared protocol,
updates one append-only-style JSON registry, and renders a comparison against
previous models and the NWM/CompACT paper baselines:

```bash
# Recommended: run the complete standard suite in the background. When GPU IDs
# are omitted, four GPUs with less than 1 GiB allocated are selected automatically.
scripts/start_nwm_benchmark.sh nwm-latentpt-ft

# The launcher prints PID/log/status paths; follow progress with its printed command.
tail -f /file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/logs/<LOG_FILE>

# Reuse completed results and run every missing task for all registered models.
scripts/run_nwm_benchmark.sh --gpus 0,1,2,3

# Select models and tasks. `unseen` is the fixed Go Stanford one-shot task.
scripts/run_nwm_benchmark.sh \
  --models nwm-base,nwm-real,nwm-release \
  --tasks unseen,navigation \
  --gpus 0,1,2,3

# Run the standardized full Go Stanford autoregressive rollout benchmark and
# render GT/models side by side. Any positive GPU count is supported.
scripts/run_nwm_benchmark.sh \
  --models nwm-real,nwm-timept-ft \
  --tasks unseen_rollout \
  --gpus 0,1

# Fill only a missing navigation dataset; optionally limit CEM peak memory.
scripts/run_nwm_benchmark.sh \
  --models nwm-release --tasks navigation \
  --navigation-datasets scand --planning-microbatch-size 40 \
  --gpus 0,1

# Inspect commands without launching jobs.
scripts/run_nwm_benchmark.sh --models nwm-base --tasks unseen --dry-run
```

The canonical outputs are:

- `/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/benchmark_results.json`
- `/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/benchmark_results.md`
- `/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/visualizations/go_stanford_unseen_rollout_v1/`

`unseen_rollout` uses all 150 fixed entries in the Go Stanford `rollout.pkl`.
The split SHA-256 is recorded in the registry. The protocol keeps 1 fps and
4 fps autoregressive rollouts, 16 seconds, 250 DDPM steps for NWM (50 Euler
steps for RAE-NWM), and seed 0. Exact strided sharding plus sample-ID keyed
randomness makes results independent of GPU count and batch partitioning up to
floating-point error. It scores all 150 samples at 1/2/4/8/16-second horizons
and visualizes eight fixed IDs. Prediction artifacts live under
`protocol_runs/go_stanford_unseen_rollout_v1`. To compare a newly
registered model, include it together with the desired baselines in `--models`;
completed baseline audits are reused and one new side-by-side visualization set
is made.

## NavAnywhere Stage-1 pretraining

For coverage-balanced, exactly replayable NavAnywhere sampling, the shared
TimePT/GeoPT/IDMPT/LatentPT recipe, one-command W&B training, and resumable
eight-GPU SD-VAE posterior precompute, see
[NAVANYWHERE_STAGE1.md](NAVANYWHERE_STAGE1.md).

Use `scripts/nwm_benchmark_registry.py` to register/import another model's
audited results. Registry writes are atomic and file-locked, so independent GPU
jobs can safely append results concurrently. Navigation also checkpoints every
trajectory under the model's planning directory and resumes completed samples
after interruption.

## Acknowledgements

This codebase builds on the following repositories:
- [NWM](https://github.com/facebookresearch/nwm/)
- [MAGE](https://github.com/LTH14/mage)
- [SEED-Voken](https://github.com/TencentARC/SEED-Voken)
- [FlexTok](https://github.com/apple/ml-flextok)

## Citation

```bibtex
@inproceedings{kim2026planning,
  title={Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model},
  author={Kim, Dongwon and Seo, Gawon and Lee, Jinsung and Cho, Minsu and Kwak, Suha},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
