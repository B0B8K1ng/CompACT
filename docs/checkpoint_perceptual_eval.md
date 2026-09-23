# Local evaluation of saved training checkpoints

`scripts/evaluate_training_checkpoints.py` evaluates the saved EMA weights of a
two-stage real-action run using `train_utils.evaluate`. It does not resume
training or write to W&B.

The defaults match the previous `nwm-nav1-latentpt-pixel-action-ft-reset`
training curve's evaluation layout: seed 0, eight logical ranks, 16 observations
per rank, eight loader workers, one batch and four goals per observation (512
goal images). The full training diffusion schedule is used: **1000 steps**, not
the independent inference configuration's 250 steps. BF16 and normalization
come from the saved checkpoint config. All checkpoints use the same examples
and random streams within this evaluation.

The original non-shuffled validation loader takes only its first batch per
rank. With the current concatenation order this is a **RECON prefix**, not a
balanced average of the four datasets and not full validation. Historical code,
data, library and RNG changes can prevent numerical reproduction of old W&B
points even when the metric and sampling layout match. This run's original
training seed is 20260901; the evaluation seed is explicitly 0 for comparison
with the earlier curves.

Run from the repository with an explicit Python environment, for example:

```bash
conda run --no-capture-output -n nwm python -B -u scripts/evaluate_training_checkpoints.py \
  --run-dir /absolute/path/to/run \
  --output /absolute/path/to/new/output \
  --source-run entity/project/run_id \
  --gpus 5,6,7
```

Inspect GPU availability before selecting devices. The script caps each worker
at 55% of a GPU's memory; it does not reserve hardware or terminate other jobs.
Each worker runs a singleton process group and replays its assigned logical
ranks. Their goal-weighted means reproduce the distributed reduction (up to
floating-point summation order) without changing per-rank sampling or batch size.

To evaluate only one saved weight, add `--checkpoint joint_0100000.pth.tar`
or `--checkpoint latest.pth.tar`. The selected file is still fingerprinted and
its aliases are recorded; the remaining checkpoints are not loaded.

The checkpoint directory is snapshotted at launch. Real files are evaluated;
symbolic aliases such as `latest.pth.tar` are recorded separately. EMA content
digests identify duplicate weights such as a warmup/transition pair. Strict
state loading and file-stat checks catch incompatible or replaced checkpoints.
The table's `train_steps` includes warmup plus joint training; the joint step
appears in the checkpoint filename and metadata.

Outputs are `manifest.json`, `results.json` (including per-rank scores),
`results.csv`, `perceptual_loss.png`, and `status.json`. Successful completion
removes temporary visualizations, rendezvous files, worker logs and intermediate
worker JSON. A failure retains diagnostics under `.temporary`; preserve the
failure details before removing that task-owned directory. Shared model and
dataset caches are reused and are not deleted.

After an interruption, rerun the same command with `--resume`. Keep the same
evaluation arguments and physical worker count. Completed shards are loaded
from the original worker JSON files; only missing shards are computed. Resume
checks the saved protocol and checkpoint file stats before starting workers.
After the current evaluation has exited, add `--resume --include-new` to evaluate
checkpoints saved by ongoing training since the previous snapshot. Completed
scores are restored from `results.json` if intermediate worker files were cleaned.
Never start two coordinators against the same output directory concurrently.
