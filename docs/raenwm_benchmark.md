# RAE-NWM benchmark

This integration evaluates the public `zmkun20/raenwm` weights with the same
fixed samples, ground truth, and LPIPS/DreamSim/PSNR implementation as the local
NWM benchmark. It deliberately keeps RAE-NWM's model-specific preprocessing:
official `[-64, 64]` action normalization, SE(2) delta composition, DINOv2-RAE,
and 50-step Euler flow sampling.

Pinned upstream inputs:

- `20robo/raenwm` revision `0219ce41c44d515f86719dd763c1efe7c7f72519`
- `zmkun20/raenwm` revision `3d21560bdbdbc8cc3d4a796e1e110d60e920d273`
- `nyu-visionx/RAE-collections` revision `1be4f03273523431f099a934da4cf1940dc6039f`
- `facebook/dinov2-with-registers-base` revision `a1d738ccfa7ae170945f210395d99dde8adb1805`

Create the official Python 3.11 environment:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda create -y -n raenwm python=3.11.10
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run -n raenwm \
  python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run -n raenwm \
  python -m pip install 'numpy<2' pyyaml omegaconf huggingface_hub decord einops \
  evo 'transformers==4.48.0' diffusers tqdm timm dreamsim torcheval lpips 'accelerate>=0.26.0' \
  torchdiffeq==0.2.5 scipy matplotlib
```

Download and hash the approximately 8 GB of model assets on NAS. The setup
script checks free space before writing:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run -n nwm \
  python scripts/setup_raenwm_assets.py
```

Run only the two direct-prediction tasks used by the paper table:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run -n nwm \
  python scripts/run_nwm_benchmark.py \
  --models rae-nwm \
  --tasks recon_prediction,unseen \
  --gpus 2,6 \
  --raenwm-python /file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python
```

The table reads the `4s` entries from the two generated audit JSON files under
`.../nwm_benchmark/predictions/rae-nwm/`. The inference manifest beside the
predictions records source and split revisions, file hashes, seed, sampler, and
distributed execution settings.

The runner uses the official batch size 16 and compiled CDiT path. A local
16-sample smoke test on one 46 GB L20 used about 5.05 GiB of allocated CUDA
memory after model setup. Execution settings are recorded in the inference
manifest.

RAE-NWM also supports autoregressive rollout. This is implemented in its
upstream `infer.py`/`run_eval.sh` and is not a paper or model limitation. The
old local `--tasks unseen_rollout` wrapper only handled NWM's legacy artifact
layout; use the unified entry point instead:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run \
  --no-capture-output -n nwm \
  python scripts/run_nwm_benchmark.py \
  --models rae-nwm --metrics rollout --datasets go_stanford \
  --gpus 5 --batch-size 16
```

## Navigation on RECON and SCAND

RAE-NWM navigation uses the same registered `navigation_cem80_v1` protocol as
the other local models, rather than the defaults in upstream RAE-NWM's planning
script:

- the exact 100-sample RECON and SCAND `navigation_eval.pkl` files;
- line/constant-delta CEM with population 80, top-k 5, one optimization step,
  three stochastic evaluations, horizon 8, rollout stride 1, and seed 42;
- LPIPS-Alex measured against each model tokenizer's reconstruction of the goal;
- the same ATE, translation RPE, final-position, and yaw-error definitions.

The pinned split SHA-256 values are
`c62cd08be9f124cbeec48d914460da8630e089bf0bdb84c5018013a82d12ec54`
for RECON and
`8acb4062561cbf1549e27f39a6e80241b97294a8c0e55c345787ae6a47be55ce`
for SCAND.

Candidate trajectories remain in the shared benchmark coordinate system. Only
the model input is converted to RAE-NWM's `[-64,64]` training normalization.
RECON has the same 0.25 m waypoint spacing in both systems; SCAND is converted
from the benchmark's 0.38 m spacing to RAE-NWM's 0.36 m training spacing. Yaw is
not scaled.

Choose the model-native Euler integration budget explicitly. The public
RAE-NWM default is 50 steps; 250 is supported for a matched step-count budget,
but is still Euler ODE integration rather than DDPM diffusion:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run -n nwm \
  python scripts/run_nwm_benchmark.py \
  --models rae-nwm \
  --tasks navigation \
  --navigation-datasets recon,scand \
  --gpus 0,1,2,3 \
  --planning-microbatch-size 80 \
  --raenwm-planning-steps 50 \
  --raenwm-python /file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python
```

Per-sample metrics are written after every sample. Candidate and model noise is
keyed by sample/candidate/repetition/rollout-step identity, so a resumed run may
change GPU count or microbatch size without changing the random streams. Results
for the two integration budgets are isolated under `planning/rae-nwm/euler50`
or `planning/rae-nwm/euler250`; their `planning_manifest.json` records the
source/checkpoint hashes, split hashes, CEM settings, action-spacing bridge, and
sampling budget.
