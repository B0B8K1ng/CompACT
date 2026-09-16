# NavAnywhere v2: 15-source LatentPT pretraining

NavAnywhere v2 adds `ego4d` and `The_Great_Outdoors` to the frozen 13-source
NavAnywhere v1 recipe.  It is fully opt-in: v1 keeps its original launcher,
recipe, cache paths, and validation-disabled default.

## Frozen sampling and split

- Seed: `20260901`
- Sampling policy: uniform source, then uniform trajectory within source
- Logical source weight: exactly `1/15` (`6.6666666667%`) for every source
- Train/validation split unit: complete trajectory; overlap is zero
- Validation holdout: one trajectory per source
- Validation recipe: 128 logical slots per source, 1,920 slots total
- Evaluator consumption: one distributed batch = 8 GPUs x 16 observations =
  128 observations and 512 goal draws, covering all 15 sources

The 200,000-step, 8-GPU, batch-16/GPU training contract consumes exactly
25,600,000 observation draws.  Because the second epoch is partial and
shuffled, realized source counts differ slightly from the ideal `1/15` while
remaining deterministic:

| Source | Train trajectories | Train observations | Exact observation draws | Realized share | Exact unique latent pairs | Validation trajectory (observations) |
|---|---:|---:|---:|---:|---:|---|
| BotanicGarden | 6 | 11,868 | 1,706,750 | 6.666992188% | 201,450 | `1018_00_VLIO` (383) |
| CASIA-Nav | 8 | 4,432 | 1,706,611 | 6.666449219% | 74,936 | `olympic_tower` (314) |
| CityWalker | 17 | 98,970 | 1,705,610 | 6.662539063% | 1,444,967 | `traj_nav_06` (3,059) |
| DL3DV-10K | 20,351 | 1,119,264 | 1,706,620 | 6.666484375% | 3,311,720 | `96df94e8849ebd36c9f70c4f95402667ad25d7957042acc3b337c618f357a6f4_1` (62) |
| EgoWalk | 238 | 422,416 | 1,707,194 | 6.668726563% | 2,447,899 | `2024_08_15__19_45_11` (256) |
| KrishnaCam | 435 | 995,099 | 1,705,738 | 6.663039062% | 3,001,418 | `20150227_125852_852.mp4_cropped` (259) |
| LAVN | 311 | 442,047 | 1,707,627 | 6.670417969% | 2,653,361 | `Gibson2_traj_Ewell` (261) |
| ROVER | 38 | 75,035 | 1,706,687 | 6.666746094% | 1,159,317 | `garden_small_night-light_2024-05-29_4` (1,290) |
| RealEstate10K | 43,765 | 577,573 | 1,706,253 | 6.665050781% | 3,049,918 | `a8317f5de4f6f22a` (59) |
| SANPO | 2,660 | 120,001 | 1,706,897 | 6.667566406% | 1,192,375 | `Y9DwcKE3c7y_ZEUzj04ipPzOurWLLK6P` (247) |
| The_Great_Outdoors | 16 | 68,280 | 1,706,431 | 6.665746094% | 1,053,753 | `tamu_full_high_speed_run_1_2024-04-23-15-46-20` (831) |
| Walking_Tours | 9 | 191,713 | 1,706,741 | 6.666957031% | 2,100,629 | `Walking Tour Zurich` (6,354) |
| ego4d | 1,618 | 13,082,984 | 1,708,127 | 6.672371094% | 3,350,896 | `990e12e3-686c-4087-8727-caced22f523a` (257) |
| i2Nav-Robot | 9 | 50,649 | 1,706,296 | 6.665218750% | 844,827 | `parking01` (4,523) |
| uB-VisioGeoloc | 15 | 5,601 | 1,706,418 | 6.665695313% | 94,452 | `12` (733) |
| **Total** | **69,496** | **17,265,932** | **25,600,000** | **100%** | **25,981,918** | **15 trajectories / 18,888 observations** |

The exact training sample set is not an estimate.  It is frozen by the train
recipe plus pair bitmap below.  The extractor decodes that bitmap into explicit
sorted `(current_frame, target_frame)` rows in every trajectory `.pt` file.
Thus the completed cache manifests are also a directly inspectable enumeration
of every pair required by training.

## Frozen artifacts and hashes

All generated artifacts are under
`/file_system/nas/algorithm/dujun.nie/nwm/compact`.

| Artifact | Path | SHA-256 |
|---|---|---|
| Frozen 13-source v1 input recipe | `recipes/navanywhere_balanced_seed20260901.json` | `604fd1e3ad4dc541198cf41b0a430adb586c07719e7994089872d0639b85bc1b` |
| Frozen Ego4D input recipe | `recipes/ego4d_balanced_seed20260901.json` | `a6e324930f27eedf75301246bbf6c21c0f30a87f5f71da0298e37a3f9ca7f3d7` |
| Frozen GO input recipe | `recipes/the_great_outdoors_4fps_seed20260901.json` | `9988c7d028db777dca8bed0690c0e16d80d5fff051077b70b66ed433c9631d35` |
| Train recipe | `recipes/navanywhere_v2_15src_train_seed20260901.json` | `dbd778aad418c6d9c43d4717d835528c4c675f0a7e4ec8d1e7e1d384e52da95b` |
| Validation recipe | `recipes/navanywhere_v2_15src_val_seed20260901.json` | `55867a5f27830ee22692403f31d308029e0bda73af35dddeb22ce607234deebf` |
| Split report file | `recipes/navanywhere_v2_15src_split_seed20260901.json` | `d6dcb954e2adf23c2d4c181ec272ba98dceec5a3224ed0311385620f1c23723d` |
| Train plan identity | `plans/navanywhere_v2_15src_train_latent_action_seed20260901_ws8_bs16_steps200000.json` | `e34c6e4e7deacbe47f45306382b3174c9c794c6c582e6725446a19a036ac26c4` |
| Train pair bitmap | `plans/navanywhere_v2_15src_train_latent_action_seed20260901_ws8_bs16_steps200000.pairs.bin` | `e945e84133b42a45b277dbb32c748cf6462308ad11ea385bd7aed54ebd65c48b` |
| Validation plan identity | `plans/navanywhere_v2_15src_val_latent_action_seed20260901_ws8_bs16_batches1.json` | `ec91755f6f0f4b02eb6876202e0913b6d54d7fd26674bf415f39becb6f87e85d` |
| Validation pair bitmap | `plans/navanywhere_v2_15src_val_latent_action_seed20260901_ws8_bs16_batches1.pairs.bin` | `68e74710894939dd052ef6218a4d8dfb8341488e36dc48bd382c509d190b6d14` |
| Completed validation cache metadata | `cache/navanywhere_v2_15src_val_nav1_pixel_action_step100000_ws8_bs16_batches1/metadata.json` | `d62d119e5011f9c2e0549aac07e8e40bc7f46ce2ed6fa1d8c9d93b97c14f9c7e` |

Plan totals:

- Training: 102,400,000 goal draws; 54,800,226 local-pair draws;
  25,981,918 unique local pairs; all 69,496 train trajectories touched.
- Validation: 128 observation draws; 512 goal draws; 256 unique local pairs;
  all 15 validation trajectories touched.
- Relative to extracting every possible +/-8-frame pair, the exact train plan
  avoids 91.04% of pair inference.
- Existing v1 and partial Ego4D caches can reuse an estimated 14,675,315 exact
  rows (56.48%); 11,306,603 rows require new inference. Runtime validation still
  checks every reused row's frame, checkpoint, extraction, shape, dtype, and
  finite-value fingerprints before accepting it.

The measured PixelActionLAM streaming rate on this host is about 57--60 newly
inferred pairs/s/GPU. At that rate, three continuously available GPUs need
about 18 hours for inference alone and roughly 20--24 hours including frame
scans, JPEG reads, cache reuse, and many small output files. Eight continuously
available GPUs need about 6.6 hours for inference alone and roughly 8--10 hours
end to end. These are workload estimates, not deadlines; the dynamic supervisor
automatically shortens the remaining time whenever another allowed GPU becomes
idle.

## Latent-action extraction

Run in the foreground without tmux or `codex-exp`:

```bash
cd /file_system/vepfs/algorithm/dujun.nie/code/CompACT

GPU_IDS=0,1,2,3,4,5,6,7 \
./run_navanywhere_v2_latent_actions.sh
```

The supervisor starts workers only on GPUs with at least 30,000 MiB free and
at most 20% utilization. It waits for busy GPUs instead of interfering with
their jobs. The command is resumable: rerun exactly the same command after an
interruption. Completed trajectory files are validated and skipped, stale
generated claims are cleaned, and missing or incompatible rows are inferred.
Validation is extracted first, followed by training.

Current outputs:

- Train cache:
  `/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/navanywhere_v2_15src_train_nav1_pixel_action_step100000_ws8_bs16_steps200000`
- Validation cache:
  `/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/navanywhere_v2_15src_val_nav1_pixel_action_step100000_ws8_bs16_batches1`

Do not start training until both cache roots contain validated `metadata.json`
and `_SUCCESS.json` files. The v2 training launcher enforces this condition.

## NWM LatentPT training

Once extraction is complete and all eight GPUs are idle:

```bash
cd /file_system/vepfs/algorithm/dujun.nie/code/CompACT

GPU_IDS=0,1,2,3,4,5,6,7 \
NPROC=8 \
BATCH_SIZE=16 \
MAX_TRAIN_STEPS=200000 \
EVAL_EVERY=5000 \
EVAL_AT_FIRST_STEP=true \
WANDB_PROJECT=compact-nwm-navanywhere-v2 \
./run_navanywhere_v2_stage1.sh
```

The launcher uses online SD-VAE encoding by default because no complete,
recipe-bound v2 posterior cache exists yet. Validation uses pixels plus the
separate validation latent-action cache so DreamSim evaluation can encode the
conditions and decode predictions. It evaluates the EMA model at step 1 and
every 5,000 steps after that. Only the trajectory-disjoint NavAnywhere
latent-action validation set is evaluated during Stage 1; Base/nwm-real test
data is intentionally not loaded because the Stage-1 real-action path has not
been adapted yet. W&B records `eval/navanywhere_perceptual_loss` (and the
backward-compatible alias `eval/perceptual_loss`).

Every 10,000 steps a numbered checkpoint such as
`pretrain_0010000.pth.tar` is retained, and `latest.pth.tar` is atomically
updated to point to it. The 200,000-step run therefore retains 20 Stage-1
checkpoints. Existing CDiT-B checkpoints are about 3.14 GB each (about 63 GB
decimal for 20); verify available space separately for larger CDiT variants.

Set `DRY_RUN=1` to inspect the resolved launch command without requiring
completed latent caches.
