"""Convert cleaned Go2 videos to NWM trajectories without crossing stream gaps.

Run in the nwm-preprocess environment. Source episodes, not segments, define
the train/test boundary. Images and poses always come from the same CSV row.
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_indices(times, fps):
    """Select actual paired frames nearest to a regular grid, never interpolate."""
    times = np.asarray(times, dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Segment timestamps must be finite and strictly increasing')
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * fps)) + 1) / fps
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right)
    if np.any(np.diff(selected) <= 0) or np.max(np.abs(times[selected] - grid)) > 0.5 / fps:
        raise ValueError('Insufficient source frame cadence for requested output FPS')
    return selected, grid


def local_deltas(position, yaw, offset=1):
    delta = position[offset:] - position[:-offset]
    c, s = np.cos(yaw[:-offset]), np.sin(yaw[:-offset])
    dyaw = (yaw[offset:] - yaw[:-offset] + np.pi) % (2 * np.pi) - np.pi
    return np.stack([delta[:, 0] * c + delta[:, 1] * s,
                     -delta[:, 0] * s + delta[:, 1] * c, dyaw], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=4.0)
    parser.add_argument('--test-episodes', type=int, default=2)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    index = json.loads((args.source / 'index.json').read_text())
    episodes = sorted(index['episodes'], key=lambda x: x['start_utc'])
    if not 0 < args.test_episodes < len(episodes):
        parser.error('Need nonempty training and test episode sets')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to mix with an existing conversion: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    test_ids = {x['episode'] for x in episodes[-args.test_episodes:]}
    names = {'train': [], 'test': []}
    all_names = {'train': [], 'test': []}
    records, episode_records = [], []
    motions = {'train': {}, 'test': {}}
    distances = []
    for episode in episodes:
        episode_id = episode['episode']
        source = args.source / episode_id
        manifest = json.loads((source / 'manifest.json').read_text())
        for filename, expected in manifest['output_sha256'].items():
            if sha256(source / filename) != expected:
                raise ValueError(f'Input checksum mismatch: {source / filename}')
        rows = list(csv.DictReader((source / 'aligned_pose.csv').open()))
        if len(rows) != manifest['frames']:
            raise ValueError('CSV/manifest frame count mismatch')
        segments = json.loads((source / 'segments.json').read_text())
        if segments != manifest['segments']:
            raise ValueError('Segment manifests disagree')
        if [i for seg in segments for i in range(seg['first_frame'], seg['last_frame'] + 1)] != list(range(len(rows))):
            raise ValueError('Segments must cover all CSV rows exactly once')
        split = 'test' if episode_id in test_ids else 'train'
        targets = {}
        for segment in segments:
            a, b = segment['first_frame'], segment['last_frame']
            block = rows[a:b + 1]
            for j, row in enumerate(block, a):
                if (int(row['frame_index']) != j or row['pair_valid'].lower() != 'true'
                        or int(row['segment_id']) != segment['segment_id']):
                    raise ValueError(f'Invalid paired frame: {episode_id}:{j}')
                if abs(int(row['nearest_pair_delta_ns'])) > 10_000_000 or int(row['pose_bracket_gap_ns']) > 40_000_000:
                    raise ValueError(f'Invalid pose timing: {episode_id}:{j}')
            times = np.array([float(r['source_elapsed_s']) for r in block])
            selected, grid = nearest_indices(times, args.fps)
            chosen = [block[i] for i in selected]
            position = np.array([[float(r[k]) for k in ('x', 'y')] for r in chosen], dtype=np.float64)
            yaw = np.array([float(r['yaw']) for r in chosen], dtype=np.float64)
            if not np.isfinite(position).all() or not np.isfinite(yaw).all():
                raise ValueError('Nonfinite pose')
            name = f'{episode_id}__seg{segment["segment_id"]:02d}'
            out = args.output / 'go2' / name
            out.mkdir(parents=True)
            with (out / 'traj_data.pkl').open('wb') as stream:
                # NWM casts EVERY value to float: only numerical arrays belong here.
                pickle.dump({'position': position, 'yaw': yaw}, stream, protocol=4)
            fields = ['output_frame', 'grid_source_elapsed_s', *rows[0].keys()]
            with (out / 'frame_mapping.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for j, row in enumerate(chosen):
                    writer.writerow({'output_frame': j, 'grid_source_elapsed_s': grid[j], **row})
                    targets[int(row['frame_index'])] = out / f'{j}.jpg'
            anchors = max(0, len(chosen) - 64 - 4 + 1)
            all_names[split].append(name)
            if anchors:
                names[split].append(name)
            steps = local_deltas(position, yaw)
            for offset in (1, 4, 8, 16, 64):
                if len(yaw) > offset:
                    motions[split].setdefault(offset, []).append(local_deltas(position, yaw, offset))
            if split == 'train' and anchors:
                distances.extend(np.linalg.norm(np.diff(position, axis=0), axis=1).tolist())
            record = dict(episode=episode_id, trajectory=name, split=split,
                          source_frames=len(block), frames=len(chosen), training_anchors=anchors,
                          eligible=bool(anchors), source_duration_s=float(times[-1] - times[0]),
                          sampled_duration_s=float(times[selected[-1]] - times[selected[0]]),
                          nearest_grid_error_max_s=float(np.max(np.abs(times[selected] - grid))),
                          path_length_m=float(np.linalg.norm(steps[:, :2], axis=1).sum()),
                          max_source_gap_s=float(np.diff(times).max()) if len(times) > 1 else 0,
                          source_interrupted=episode['source_interrupted'])
            (out / 'metadata.json').write_text(json.dumps(record, indent=2))
            records.append(record)
        cap = cv2.VideoCapture(str(source / 'video.mp4'))
        count, emitted = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in targets:
                if not cv2.imwrite(str(targets[count]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f'Failed to write {targets[count]}')
                emitted += 1
            count += 1
        cap.release()
        if count != len(rows) or emitted != len(targets):
            raise ValueError(f'Video/CSV mismatch in {episode_id}: decoded={count}, rows={len(rows)}')
        episode_records.append(dict(episode=episode_id, split=split, raw_frames=len(rows),
                                    output_frames=emitted, duration_s=float(rows[-1]['source_elapsed_s'])))
        print(f'{episode_id} {split}: decoded={count}, sampled={emitted}', flush=True)
    spacing = float(np.mean(distances))
    if spacing <= 0:
        raise ValueError('Training data has no translation')
    for split in names:
        folder = args.output / 'data_splits' / 'go2' / split
        folder.mkdir(parents=True)
        (folder / 'traj_names.txt').write_text('\n'.join(names[split]) + '\n')
        (folder / 'all_traj_names.txt').write_text('\n'.join(all_names[split]) + '\n')
        (folder / 'episode_names.txt').write_text('\n'.join(x['episode'] for x in episode_records if x['split'] == split) + '\n')
    motion_stats = {}
    for split, offsets in motions.items():
        motion_stats[split] = {}
        for offset, arrays in offsets.items():
            values = np.concatenate(arrays)
            norm = values / np.array([spacing, spacing, 1.0])
            motion_stats[split][str(offset)] = dict(
                pairs=len(values), metric_quantiles=np.quantile(values, [0, .05, .5, .95, 1], axis=0).tolist(),
                normalized_quantiles=np.quantile(norm, [0, .05, .5, .95, 1], axis=0).tolist(),
                backward_fraction=float(np.mean(values[:, 0] < -.02)),
                sideways_fraction=float(np.mean(np.abs(values[:, 1]) > np.maximum(.02, np.abs(values[:, 0])))),
                turning_fraction=float(np.mean(np.abs(values[:, 2]) > .05)))
    report = dict(source=str(args.source.resolve()), fps=args.fps, time_basis='source_elapsed_s',
                  pose_policy='same source row as nearest actual video frame; no interpolation',
                  split_policy=f'last {args.test_episodes} source episodes by start_utc held out',
                  image_policy='original resolution JPEG quality 95; NWM loader applies 4:3 crop and 224x224 resize',
                  physical_sync_verified=False, metric_waypoint_spacing=spacing,
                  spacing_policy='mean consecutive displacement of train segments with >=68 sampled frames only',
                  checkpoint_context_size=4, len_traj_pred=64,
                  quantile_levels=[0, .05, .5, .95, 1], action_columns=['dx', 'dy', 'dyaw'],
                  episodes=episode_records, segments=records, motion_stats=motion_stats)
    (args.output / 'conversion_report.json').write_text(json.dumps(report, indent=2))
    # The converter generates the measured value; no test data enters normalization.
    (args.output / 'normalization.json').write_text(json.dumps({
        'metric_waypoint_spacing': spacing, 'fit_split': 'train', 'fps': args.fps}, indent=2))
    print(f'CONVERSION_COMPLETE train={len(names["train"])} test={len(names["test"])} '
          f'segments; spacing={spacing:.9f}', flush=True)


if __name__ == '__main__':
    main()
