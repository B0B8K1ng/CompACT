# Two-stage NWM training

This extension keeps the legacy training path as the default. Selecting a
`two_stage` overlay opts into one of the runs below; no overlay means
`training_stage=legacy` and preserves the old configuration/checkpoint behavior.
The implementation adds neither a motion embedding nor an action-type embedding.

## 1. Data split contract

Stage 1 uses **NavAnywhere only**. Set `NWM_NAVANYWHERE_ROOT` to the processed
video root whose immediate children are source directories such as
`BotanicGarden/` and `CASIA-Nav/`. Sources and trajectories are discovered
directly; there is no assumed `navanywhere/` directory level. An optional
JSON/JSONL/text trajectory manifest can be supplied with
`NWM_NAVANYWHERE_MANIFEST`. `conf/dataset/navanywhere.yaml` retains signed
target offsets from -64 through 64. RECON, SCAND, TartanDrive, HuRoN/SACSoN,
and Go Stanford are excluded from stage-1 training.

Stage 2 uses the training splits of exactly these four sources:

- `recon` (RECON)
- `scand` (SCAND)
- `tartan_drive` (the processed directory is named `tartan`)
- `sacson` (the repository's historical key for HuRoN)

`go_stanford` is disabled and evaluation-only. NavAnywhere is not mixed into
stage 2. Set `NWM_DATA_ROOT` to the parent of the four processed datasets.
Stage 2 defaults to eight GPUs, eight data-loader workers per rank, and the
completed four-dataset SD-VAE posterior cache. Relocate that cache with
`NWM_FINETUNE_VAE_LATENT_ROOT`.

## 2. Stage-1 modes

All modes retain the existing NWM diffusion loss and the full [-64, 64] goal
sampling range.

| Overlay | `action_mode` | Condition |
| --- | --- | --- |
| `timept` | `none` | diffusion timestep + relative time only |
| `geopt` | `geometry` | time condition plus an eligible cached geometry proxy |
| `idmpt` | `idm` | time condition plus an eligible cached IDM proxy |
| `latentpt` | `latent` | time condition plus an eligible cached DreamDojo latent |
| `latentonlypt` | `latent` | eligible cached DreamDojo latent only; time fallback otherwise |

HybridPT and ShuffledPT are not part of this workflow.

GeoPT, IDMPT, and LatentPT checkpoints are stage-1 initialization artifacts,
not real-action deployment checkpoints. Directly feeding a `real` action group
to one of them is rejected instead of silently using the untrained `E_real`.
Run inference or planning from the corresponding stage-2 A/B/C checkpoint.
TimePT remains a valid time-only model and ignores action tensors entirely.

The proxy window is not the sampling window. A target at any signed offset in
[-64, 64] contributes diffusion loss. A proxy is eligible only when
`abs(frame_offset) <= 8`, and it is used only if its exact record exists, has
the configured shape, is finite, and passes store validation. The dataset does
not access the proxy store for longer offsets. An invalid or long-range sample
is time-only; a zero placeholder used for collation has no conditioning
meaning. Masking is applied after the action encoder, so encoder bias cannot
leak into an invalid sample.

Conceptually, stage 1 computes:

```text
c = diffusion_timestep_embedding(diffusion_t) + relative_time_embedding(rel_t)
if proxy_valid:
    c += proxy_encoder(proxy_action)
prediction = NWM(noisy_target, context, c)
```

`latentonlypt` changes only the relative-time/action composition. The diffusion
timestep embedding is always retained because it identifies the diffusion noise
level. For each flattened target row it computes:

```text
if abs(frame_offset) <= 8 and latent_valid:
    c = diffusion_timestep_embedding(diffusion_t) + latent_encoder(latent_action)
else:
    c = diffusion_timestep_embedding(diffusion_t) + relative_time_embedding(rel_t)
```

Thus local valid rows do not receive relative time, while long-range rows and
local cache misses remain time-conditioned. `proxy.relative_time_mode=fallback`
records this identity in checkpoint metadata; regular `latentpt` uses `always`.

The eligibility test always uses the original signed integer `frame_offset`,
never normalized `rel_t`.

## 3. Stage-2 schemes and two-step schedule

Every scheme runs adapter warm-up, saves a transition checkpoint, and then
runs joint fine-tuning. A single launch performs both steps. Optimizers are
rebuilt at the transition and contain only parameters with
`requires_grad=True`.

| Scheme / overlay | Warm-up | Joint |
| --- | --- | --- |
| A, `*_reset` | train new `E_real`; frozen NWM remains in the autograd graph; diffusion loss | train `E_real` at `adapter_lr` and NWM/time modules at `backbone_lr`; diffusion loss |
| B, `latent_align` | train `E_real`; frozen `E_z` supplies local teacher embeddings; alignment loss only | train `E_real` + NWM/time modules; diffusion loss on all offsets plus local valid alignment |
| C, `latent_real_to_latent` | train `G_real_to_latent`; gradients traverse frozen `E_z` and NWM; diffusion loss | train `G_real_to_latent`, `E_z`, and NWM/time modules; diffusion loss only |
| D, `latent_state_controller` | train the state-conditioned controller for 3,000 steps with local latent L2; freeze all Stage-1 modules | train controller + CDiT backbone for 200,000 steps at fixed `1e-4`; freeze `x_embedder` and `E_z`; diffusion loss on all offsets plus local latent L2 |

The VAE stays frozen. Unused stage-1 proxy encoders are frozen and excluded from
the optimizer. In B, `E_z` is frozen in both steps and is absent from inference.
In C, it is frozen during warm-up and becomes trainable in joint fine-tuning.
In D, `x_embedder` and `E_z` remain frozen throughout; only the controller is
trainable in warm-up, then the rest of the CDiT backbone joins it in joint.
These freeze policies and the transition-checkpoint save are enforced by the
scheme/substage state machine, so the example overlays intentionally do not
expose misleading `freeze_latent_action_encoder` or
`save_transition_checkpoint` switches.

EmbeddingAlign decides whether the warm-up step is skipped from the globally
reduced valid-pair count, so every DDP rank executes the same backward and
optimizer step. Its first-step nonzero-gradient assertion is rank-local: a rank
with no local teacher pair participates through the zero graph without being
mistakenly rejected when another rank supplies the valid pair.

Scheme A can initialize from TimePT, GeoPT, IDMPT, or LatentPT. Schemes B, C,
and D
require a LatentPT checkpoint and fail before training if the checkpoint action
mode, latent dimension/normalization, hidden dimension, model size, context size,
or other required metadata is incompatible.

### Why action encoders do not receive `rel_t`

`rel_t` already has its own embedding. `E_real`, the geometry/IDM encoders,
`E_z`, and `G_real_to_latent` encode action coordinates only. Keeping time out
of these encoders prevents duplicated time semantics and preserves the learned
stage-1 latent-action interface.

### Why B aligns only local pairs

DreamDojo targets are cached only for reliable pairs with
`abs(frame_offset) <= 8`. B computes cosine plus L1 alignment only where
`latent_valid=true`; longer or invalid pairs still receive diffusion loss in
joint training. A batch with no valid teacher target has a device/dtype-correct
zero alignment loss, not a division by zero.

At inference and during planning, B uses only:

```text
real_action -> E_real -> NWM
```

No future frame, DreamDojo model, cache, or `E_z` teacher is needed.

### Why C needs no latent target

C predicts latent coordinates from real action and trains them only through
the NWM diffusion objective:

```text
real_action -> G_real_to_latent -> E_z -> NWM
```

There is no ground-truth latent lookup and no alignment term. The output
coordinate convention must exactly match the one recorded by LatentPT: if
LatentPT's `E_z` consumes standardized latent coordinates,
`G_real_to_latent` outputs standardized latent coordinates directly; if it
uses another established convention, C preserves that convention unchanged.
Inference uses this identical path.

### Scheme D: state-conditioned latent controller

Scheme C remains action-only and diffusion-supervised, so it is not the
controller used in [Learning Latent Action World Models In The
Wild](https://arxiv.org/html/2601.05230v2). The paper uses the final previous
frame's frozen V-JEPA representation, two self-attention blocks, a three-layer
real-action MLP, cross-attention, and direct latent L2 supervision. It does not
use an SD-VAE latent or CDiT's patch embedder. D is therefore a separate
CompACT-native adaptation and leaves C unchanged.

The implemented path is:

```text
last context-frame SD-VAE latent
  -> frozen CDiT patch embedder -> 2 self-attention blocks -> visual tokens
normalized real action
  -> 3-layer MLP -> action query
action query cross-attends to visual tokens
  -> linear projection -> predicted 32-D latent action -> frozen E_z -> NWM
```

Reusing the CDiT patch embedder is an engineering hypothesis: it exposes the
same spatial token coordinates consumed by the pretrained NWM and avoids a
second visual encoder. It is not implied by the paper, and should be compared
against a frozen LAM/V-JEPA-style representation. Treating the action token as
the single cross-attention query is likewise a natural way to obtain one latent
token through action-dependent spatial pooling, not a uniquely justified
architecture; pooled concatenation, FiLM, or a joint `[ACT]` transformer token
are valid controls.

The teacher is the frozen PixelActionLAM latent for the same
`(current_frame, goal_frame)` pair. The existing four-dataset cache at
`/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/finetune_nav1_pixel_action_step100000_four_datasets`
already stores raw 32-D `z_mu` targets for eligible
`abs(frame_offset) <= 8` pairs. Controller pretraining minimizes direct L2 on
valid targets while freezing SD-VAE, the Stage-1 NWM, and `E_z`. It runs 3,000
steps with AdamW at `1e-3`, weight decay `0.04`, a 300-step linear warm-up, and
cosine decay.

Use the last context frame rather than the first frame of the whole trajectory:
the controller must see the camera state from which the real action is applied.
The latent L2 term is available only inside the cache's local window, but that
does not require long-range samples to be time-only during Stage 2. The existing
fine-tuning contract conditions all `[-64, 64]` samples on real actions and
applies diffusion loss everywhere; only teacher alignment is locally masked.
A repository-compatible controller run should follow the same rule:

```text
all offsets:          diffusion loss using controller(state, real_action)
abs(offset) <= 8:   + latent L2 against the cached teacher
```

The overlay then runs 200,000 joint steps. Controller and CDiT backbone both
use a fixed `1e-4`, matching nwm-real, while `x_embedder` and `E_z` remain
frozen. This architecture and auxiliary supervision cannot guarantee an
improvement over the end-to-end nwm-real baseline; controller L2, perceptual
loss, and downstream navigation must all be reported.

CDiT checkpoints also fix the number of context frames through their learned
positional embeddings. Therefore `eval_context_size` and, for planning,
`trajectory_eval_context_size` must equal the training
`dataset.context_size`. Both entry points validate this immediately and report
the conflicting keys instead of failing later with a tensor shape mismatch.

## 4. Configurations

Stage 1:

- `conf/two_stage/timept.yaml`
- `conf/two_stage/geopt.yaml`
- `conf/two_stage/idmpt.yaml`
- `conf/two_stage/latentpt.yaml`
- `conf/two_stage/latentonlypt.yaml`

Stage 2:

- `conf/two_stage/no_pretrain.yaml`
- `conf/two_stage/latent_reset.yaml`
- `conf/two_stage/latent_align.yaml`
- `conf/two_stage/latent_real_to_latent.yaml`
- `conf/two_stage/latent_state_controller.yaml`
- `conf/two_stage/time_reset.yaml`
- `conf/two_stage/geo_reset.yaml`
- `conf/two_stage/idm_reset.yaml`

Every overlay uses top-level `training_stage`, `action_mode`, `proxy`, and
`finetune` fields. `finetune.warmup_steps`, `finetune.joint_steps`, both learning
rates, and alignment weights are ordinary Hydra overrides.

`proxy.max_abs_frame_offset=8` and the dataset range `[-64,64]` are invariants,
not tuning knobs. Proxy-bearing stage-1 runs, B, and D set
`proxy.use_precomputed_only=true`; the training process never launches an
extractor. The suffixless `proxy.file_pattern` is shared by stage 1, B, and D,
so they accept exactly one `.pt` or `.npz` shard per source/trajectory without a
format-specific configuration change.

Cache environment variables are:

| Run | Required cache variables |
| --- | --- |
| GeoPT | `NWM_GEOMETRY_PROXY_ROOT` |
| IDMPT | `NWM_IDM_PROXY_ROOT` |
| LatentPT | `NWM_LATENT_PROXY_ROOT` |
| EmbeddingAlign | `NWM_FINETUNE_LATENT_ROOT` |
| State-conditioned controller | `NWM_FINETUNE_LATENT_ROOT` (shared NAS default available) |
| Every stage-2 run (SD-VAE posterior) | `NWM_FINETUNE_VAE_LATENT_ROOT` (optional override) |

The current latent adapter uses its existing parameter-free, per-sample
`layer_norm`, so these overlays do not invent an external statistics file. Its
normalization identifier is still saved in checkpoint metadata and checked by
B/C/D. If an experiment explicitly switches to a mean/std implementation, those
statistics must come only from the NavAnywhere 1000-hour train split, their
path/statistics must be saved in stage 1, and B/C/D must reuse them unchanged.

## 5. Launch commands

`two_stage_nwm.sh` wraps the existing `scripts/train.sh`. Put additional Hydra
overrides after `--`. Full training is long-running, so launch it with the
durable experiment runner. The wrapper resolves data roots, cache roots,
manifest paths, result roots, and checkpoint arguments to absolute paths before
Hydra changes the working directory. When invoking `train.py` directly, supply
absolute paths yourself.

TimePT on one GPU:

```bash
export NWM_NAVANYWHERE_ROOT=/path/to/NavAnywhere
export NWM_RESULTS_DIR=/path/to/nwm-checkpoints
export NWM_CONDA_ENV=nwm
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 timept --gpus=0 --nproc=1
```

GeoPT, IDMPT, and LatentPT use the same form after setting the cache variables:

```bash
export NWM_GEOMETRY_PROXY_ROOT=/path/to/proxy-cache/geometry
export NWM_IDM_PROXY_ROOT=/path/to/proxy-cache/idm
export NWM_LATENT_PROXY_ROOT=/path/to/proxy-cache/dreamdojo-latent
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 geopt --gpus=0,1,2,3 --nproc=4
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 idmpt --gpus=0,1,2,3 --nproc=4
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 latentpt --gpus=0,1,2,3 --nproc=4
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 latentonlypt --gpus=0,1,2,3 --nproc=4
```

Start stage 2 from a LatentPT checkpoint (the same syntax works for every reset
overlay):

```bash
export NWM_DATA_ROOT=/path/to/processed-navigation-data
export NWM_RESULTS_DIR=/path/to/nwm-checkpoints
export NWM_STAGE1_CHECKPOINT=/path/to/checkpoints/latentpt.pt
export NWM_FINETUNE_VAE_LATENT_ROOT=/path/to/vae-cache/four-datasets
export NWM_FINETUNE_LATENT_ROOT=/path/to/proxy-cache/finetune-dreamdojo-latent
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_align --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"

conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_real_to_latent --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"

conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_state_controller --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"

conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_reset --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"
```

Reset from TimePT, GeoPT, or IDMPT by selecting the matching overlay and source
checkpoint:

```bash
export NWM_SOURCE_CHECKPOINT=/path/to/checkpoints/source-stage1.pt
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 time_reset --stage1-checkpoint="${NWM_SOURCE_CHECKPOINT}"
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 geo_reset --stage1-checkpoint="${NWM_SOURCE_CHECKPOINT}"
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 idm_reset --stage1-checkpoint="${NWM_SOURCE_CHECKPOINT}"
```

Run the strict no-pretraining control with the same four-dataset stage-2 data
budget. It constructs CDiT and the real-action adapter from the standard
seeded initialization and intentionally accepts no stage-1 checkpoint. Unlike
checkpoint-initialized reset runs, it starts joint CDiT + adapter optimization
immediately: a fresh CDiT's zero-initialized output and adaLN layers make an
adapter-only warm-up unable to propagate diffusion gradients. The first two
joint updates open those standard zero-initialized paths, so the E_real
gradient assertion is performed on the third backward. The frozen
SD-VAE/cached SD-VAE posteriors remain shared with all other runs and are not
part of the NWM pretraining comparison:

```bash
export NWM_DATA_ROOT=/path/to/processed-navigation-data
export NWM_RESULTS_DIR=/path/to/nwm-checkpoints
export NWM_FINETUNE_VAE_LATENT_ROOT=/path/to/vae-cache/four-datasets
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 no_pretrain
```

Resume a warm-up checkpoint, then separately resume a transition or joint
checkpoint with the same launcher. The checkpoint's `finetune_substage` and
completed-step counters decide whether to remain in warm-up or enter/continue
joint training:

```bash
export NWM_WARMUP_CHECKPOINT=/path/to/checkpoints/latent-align-warmup.pt
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_align --resume="${NWM_WARMUP_CHECKPOINT}"

export NWM_JOINT_CHECKPOINT=/path/to/checkpoints/latent-align-transition.pt
conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_align --resume="${NWM_JOINT_CHECKPOINT}"
```

Minimal one-GPU smoke launch (two warm-up and three joint steps):

```bash
WANDB_MODE=offline conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_real_to_latent --gpus=0 --nproc=1 --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}" -- training.batch_size=2 training.num_workers=0 finetune.warmup_steps=2 finetune.joint_steps=3 eval_at_first_step=false
```

The repository's data-free synthetic smoke suites run TimePT, LatentPT, and
the warm-up/joint paths for A/B/C/D with batch size 2. They are the quickest check
of masking, gradients, finite losses, save, and resume:

```bash
TWO_STAGE_SMOKE_DEVICE=cuda:0 conda run -n "${NWM_CONDA_ENV}" python tests/test_two_stage_smoke.py -q
conda run -n "${NWM_CONDA_ENV}" python -m unittest tests.test_state_conditioned_controller
```

Command-only checks do not require a GPU, data mount, or cache:

```bash
./two_stage_nwm.sh stage1 latentpt --gpus=0 --nproc=1 --dry-run -- training.batch_size=2 max_train_steps=3
./two_stage_nwm.sh stage2 latent_align --dry-run --stage1-checkpoint=/path/to/checkpoints/latentpt.pt
./two_stage_nwm.sh stage2 latent_state_controller --dry-run --stage1-checkpoint=/path/to/checkpoints/latentpt.pt
```

For multi-node DDP, run one launcher per node with the same host/port and the
appropriate rank, for example `--nodes=2 --host=rank0.example.internal --port=29501
--rank=0` and `--rank=1`. `--nproc` is the process count per node.

## 6. Proxy cache key and tensor contract

The logical key is the complete tuple:

```text
(dataset_or_source_id, trajectory_or_video_id,
 current_frame_index, target_frame_index)
```

Never key a cache by batch position or randomized sample index. The configured
suffixless pattern is `root/{source_id}/{trajectory_id}`. Exactly one of
`root/{source_id}/{trajectory_id}.pt` and
`root/{source_id}/{trajectory_id}.npz` must exist; having both is an ambiguity
error. A compatible trajectory-sharded mapping stores `frame_pairs` as int64
`[N, 2]` and `motion` as finite floating-point `[N, D]`, where `D` equals
`proxy.dim` (geometry 3, the repository's current IDM output 7, and DreamDojo
latent 32 by default). An optional `proxy_valid`, `motion_valid`, or
`valid_mask` vector is boolean/binary `[N]`.
Each selected sample is a fixed `[D]` tensor. Store metadata records source,
proxy type, dimension, pair direction (`current_to_target`), normalization
identity/statistics, and extractor checkpoint identity.

The stage-2 teacher cache follows the same key/shape contract but only needs
valid pairs in [-8, 8]. Schemes B and D read it; Scheme C does not. With
`proxy.strict_loading=true`, a missing, malformed, non-finite, or otherwise
invalid eligible record raises an error containing the full logical key.

## 7. Checkpoint conversion and recovery

A stage-1 checkpoint records the stage/action/proxy metadata, model and context
dimensions, latent normalization, and the usual model state. Starting stage 2
loads the backbone, context/image modules, timestep and relative-time embedders,
position/final layers, plus only the stage-1 proxy encoder required by the
chosen scheme. A fresh stage-2 optimizer is created; the stage-1 optimizer is
not imported. No offline checkpoint-conversion script is needed: the stage-2
loader performs the validated stage transition directly and reports the loaded
and intentionally skipped module prefixes.

Stage-2 checkpoints additionally record the scheme, current substage,
`completed_warmup_steps`, and `completed_joint_steps`. Periodic and phase-final
checkpoints preserve the exact `(epoch, batch_in_epoch)` position; the epoch is
advanced only when the loader was exhausted. Consequently, increasing a saved
phase's step target resumes the remaining batches instead of silently starting
the next epoch. Each checkpoint also stores every DDP rank's Python, NumPy,
torch, proxy-metric, and alignment-progress state. Even when a resumed phase is
already complete and its loop is skipped, that saved RNG state is carried into
joint training and restored after joint iterator creation.

With the default `finetune.warmup_steps=10000`,
`finetune.joint_steps=100000`, and `ckpt_every=10000`, stage 2 retains
`warmup_0010000.pth.tar` and `joint_0010000.pth.tar` through
`joint_0100000.pth.tar`. The interval is counted independently inside each
substage. A phase target that is not an exact interval multiple still gets one
phase-final retained checkpoint. `latest.pth.tar` is an atomically replaced
relative symlink to the newest retained file and is intended as a convenient
resume path; retained files are never overwritten by later intervals. Stage 1
uses the same publication contract: with 200,000 steps and
`ckpt_every=10000`, it retains `pretrain_0010000.pth.tar` through
`pretrain_0200000.pth.tar`, while `latest.pth.tar` points to the newest file.

Scheme D instead retains its phase-final `warmup_0003000.pth.tar`, then
`joint_0010000.pth.tar` through `joint_0200000.pth.tar`; the complete run is
3,000 + 200,000 optimizer steps.

Stage 1 and Stage 2 both run lightweight inline evaluation. With the shared
defaults they evaluate the EMA model at global step 1 and every
`eval_every=5000` steps, compute DreamSim on the first distributed test batch,
and write up to ten condition/ground-truth/prediction panels. Stage 1 uses only
the trajectory-disjoint NavAnywhere split with its cached latent actions, writes
to `viz/navanywhere/<step>`, and logs
`eval/navanywhere_perceptual_loss` plus the compatibility alias
`eval/perceptual_loss`. It deliberately does not run Base/nwm-real test data
through the unadapted real-action path. Stage 2 evaluates the normal Base test
loader, writes to `viz/<step>`, and logs `eval/perceptual_loss`. Cached SD-VAE
posteriors remain training-only: each eval loader reads RGB frames and a frozen
SD-VAE is loaded for encoding and decoding. Evaluation RNG consumption is
restored so it does not change the training or exact-resume trajectory.

Scheme D uses phase-local evaluation axes. Controller pretraining evaluates at
steps 1, 1,000, 2,000, and 3,000 under `controller_pretrain/*`; joint evaluates
at step 1 and every 5,000 joint steps under `joint/*`. Visualizations go to
`viz/controller_pretrain/<step>` and `viz/joint/<step>`. The unprefixed
compatibility metrics remain available, while phase-prefixed curves align the
joint 0--200,000 step range directly with nwm-real.

"First distributed batch" is not one sample. For the standard 8-rank,
batch-16 Stage-1 run it contains 128 observation sequences; four goals per
sequence produce 512 predictions in the reported DreamSim mean. It remains a
fast regression curve rather than a full-validation estimate.

This inline score is a quick regression signal, not a complete checkpoint
selection benchmark. Evaluate retained checkpoints offline by passing their
stem (for example `ckp=joint_0030000`) to `isolated_nwm_infer.py`, then compare
LPIPS, DreamSim, PSNR, and generated frames on one frozen validation set. Run
the more expensive navigation `planning_eval.py` on the strongest candidates.
Do not select a checkpoint by repeatedly inspecting the test set; reserve test
for the final chosen model.

The warm-up completion transition artifact is always the epoch-zero, cursor-zero
entry point for joint training. It omits the warm-up optimizer and scheduler,
because joint constructs its two learning-rate parameter groups, but preserves
the shared AMP GradScaler state. Unexpected checkpoint keys or missing keys
outside explicitly allowed new-module prefixes are errors rather than being
silently swallowed by unrestricted `strict=False` loading.

New checkpoints include a SHA-256 data-resume fingerprint covering the seed,
batch size, worker count, substage, dataset selection/configuration, proxy
cache configuration, motion configuration, and fixed sampler/loader contract.
Phase checkpoints also record observed dataset and loader lengths. These values
are checked before cursor replay; changing one is a hard error rather than a
best-effort resume. An in-epoch resume requires the same DDP world size. Older
schema-v1 checkpoints that predate this fingerprint remain compatible only at
their historical epoch-boundary cursor (`batch_in_epoch=0`); an old nonzero
cursor cannot be certified for exact replay and is rejected.
