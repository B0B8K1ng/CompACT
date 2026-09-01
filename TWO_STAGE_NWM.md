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

## 2. Stage-1 modes

All modes retain the existing NWM diffusion loss and the full [-64, 64] goal
sampling range.

| Overlay | `action_mode` | Condition |
| --- | --- | --- |
| `timept` | `none` | diffusion timestep + relative time only |
| `geopt` | `geometry` | time condition plus an eligible cached geometry proxy |
| `idmpt` | `idm` | time condition plus an eligible cached IDM proxy |
| `latentpt` | `latent` | time condition plus an eligible cached DreamDojo latent |

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

The VAE stays frozen. Unused stage-1 proxy encoders are frozen and excluded from
the optimizer. In B, `E_z` is frozen in both steps and is absent from inference.
In C, it is frozen during warm-up and becomes trainable in joint fine-tuning.
These freeze policies and the transition-checkpoint save are enforced by the
scheme/substage state machine, so the example overlays intentionally do not
expose misleading `freeze_latent_action_encoder` or
`save_transition_checkpoint` switches.

EmbeddingAlign decides whether the warm-up step is skipped from the globally
reduced valid-pair count, so every DDP rank executes the same backward and
optimizer step. Its first-step nonzero-gradient assertion is rank-local: a rank
with no local teacher pair participates through the zero graph without being
mistakenly rejected when another rank supplies the valid pair.

Scheme A can initialize from TimePT, GeoPT, IDMPT, or LatentPT. Schemes B and C
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

Stage 2:

- `conf/two_stage/latent_reset.yaml`
- `conf/two_stage/latent_align.yaml`
- `conf/two_stage/latent_real_to_latent.yaml`
- `conf/two_stage/time_reset.yaml`
- `conf/two_stage/geo_reset.yaml`
- `conf/two_stage/idm_reset.yaml`

Every overlay uses top-level `training_stage`, `action_mode`, `proxy`, and
`finetune` fields. `finetune.warmup_steps`, `finetune.joint_steps`, both learning
rates, and alignment weights are ordinary Hydra overrides.

`proxy.max_abs_frame_offset=8` and the dataset range `[-64,64]` are invariants,
not tuning knobs. Proxy-bearing stage-1 runs and B set
`proxy.use_precomputed_only=true`; the training process never launches an
extractor. The suffixless `proxy.file_pattern` is shared by stage 1 and B, so
both accept exactly one `.pt` or `.npz` shard per source/trajectory without a
format-specific configuration change.

Cache environment variables are:

| Run | Required cache variables |
| --- | --- |
| GeoPT | `NWM_GEOMETRY_PROXY_ROOT` |
| IDMPT | `NWM_IDM_PROXY_ROOT` |
| LatentPT | `NWM_LATENT_PROXY_ROOT` |
| EmbeddingAlign | `NWM_FINETUNE_LATENT_ROOT` |

The current latent adapter uses its existing parameter-free, per-sample
`layer_norm`, so these overlays do not invent an external statistics file. Its
normalization identifier is still saved in checkpoint metadata and checked by
B/C. If an experiment explicitly switches to a mean/std implementation, those
statistics must come only from the NavAnywhere 1000-hour train split, their
path/statistics must be saved in stage 1, and B/C must reuse them unchanged.

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
codex-exp start nwm-timept -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 timept --gpus=0 --nproc=1
```

GeoPT, IDMPT, and LatentPT use the same form after setting the cache variables:

```bash
export NWM_GEOMETRY_PROXY_ROOT=/path/to/proxy-cache/geometry
export NWM_IDM_PROXY_ROOT=/path/to/proxy-cache/idm
export NWM_LATENT_PROXY_ROOT=/path/to/proxy-cache/dreamdojo-latent
codex-exp start nwm-geopt -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 geopt --gpus=0,1,2,3 --nproc=4
codex-exp start nwm-idmpt -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 idmpt --gpus=0,1,2,3 --nproc=4
codex-exp start nwm-latentpt -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage1 latentpt --gpus=0,1,2,3 --nproc=4
```

Start stage 2 from a LatentPT checkpoint (the same syntax works for every reset
overlay):

```bash
export NWM_DATA_ROOT=/path/to/processed-navigation-data
export NWM_RESULTS_DIR=/path/to/nwm-checkpoints
export NWM_STAGE1_CHECKPOINT=/path/to/checkpoints/latentpt.pt
export NWM_FINETUNE_LATENT_ROOT=/path/to/proxy-cache/finetune-dreamdojo-latent
codex-exp start nwm-align -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_align --gpus=0,1,2,3 --nproc=4 --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"

codex-exp start nwm-real-to-latent -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_real_to_latent --gpus=0,1,2,3 --nproc=4 --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"

codex-exp start nwm-latent-reset -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_reset --gpus=0,1,2,3 --nproc=4 --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}"
```

Reset from TimePT, GeoPT, or IDMPT by selecting the matching overlay and source
checkpoint:

```bash
export NWM_SOURCE_CHECKPOINT=/path/to/checkpoints/source-stage1.pt
codex-exp start nwm-time-reset -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 time_reset --gpus=0,1,2,3 --nproc=4 --stage1-checkpoint="${NWM_SOURCE_CHECKPOINT}"
codex-exp start nwm-geo-reset -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 geo_reset --gpus=0,1,2,3 --nproc=4 --stage1-checkpoint="${NWM_SOURCE_CHECKPOINT}"
codex-exp start nwm-idm-reset -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 idm_reset --gpus=0,1,2,3 --nproc=4 --stage1-checkpoint="${NWM_SOURCE_CHECKPOINT}"
```

Resume a warm-up checkpoint, then separately resume a transition or joint
checkpoint with the same launcher. The checkpoint's `finetune_substage` and
completed-step counters decide whether to remain in warm-up or enter/continue
joint training:

```bash
export NWM_WARMUP_CHECKPOINT=/path/to/checkpoints/latent-align-warmup.pt
codex-exp start nwm-align-warmup-resume -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_align --gpus=0,1,2,3 --nproc=4 --resume="${NWM_WARMUP_CHECKPOINT}"

export NWM_JOINT_CHECKPOINT=/path/to/checkpoints/latent-align-transition.pt
codex-exp start nwm-align-joint-resume -- conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_align --gpus=0,1,2,3 --nproc=4 --resume="${NWM_JOINT_CHECKPOINT}"
```

Minimal one-GPU smoke launch (two warm-up and three joint steps):

```bash
WANDB_MODE=offline conda run -n "${NWM_CONDA_ENV}" ./two_stage_nwm.sh stage2 latent_real_to_latent --gpus=0 --nproc=1 --stage1-checkpoint="${NWM_STAGE1_CHECKPOINT}" -- training.batch_size=2 training.num_workers=0 finetune.warmup_steps=2 finetune.joint_steps=3 eval_at_first_step=false
```

The repository's data-free synthetic smoke suite runs TimePT, LatentPT, and
the warm-up/joint paths for A/B/C with batch size 2. It is the quickest check
of masking, gradients, finite losses, save, and resume:

```bash
TWO_STAGE_SMOKE_DEVICE=cuda:0 conda run -n "${NWM_CONDA_ENV}" python tests/test_two_stage_smoke.py -q
```

Command-only checks do not require a GPU, data mount, or cache:

```bash
./two_stage_nwm.sh stage1 latentpt --gpus=0 --nproc=1 --dry-run -- training.batch_size=2 max_train_steps=3
./two_stage_nwm.sh stage2 latent_align --gpus=0,1,2,3 --nproc=4 --dry-run --stage1-checkpoint=/path/to/checkpoints/latentpt.pt
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

The stage-2 alignment cache follows the same key/shape contract but only needs
valid pairs in [-8, 8]. Scheme C does not read it. With
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
