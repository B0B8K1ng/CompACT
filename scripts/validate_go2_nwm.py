"""CPU audit of converted Go2 files through the repository's actual NWM loader."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from datasets import TrainingDataset
from misc import get_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / 'conversion_report.json').read_text())
    os.environ['NWM_GO2_ROOT'] = str(args.root)
    os.environ['NWM_INDEX_ROOT'] = str(args.root / 'index_cache')
    config_dir = str(Path(__file__).resolve().parents[1] / 'conf')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name='nwm', overrides=['dataset=go2'], return_hydra_config=True)
    HydraConfig.instance().set_config(config)
    spacing = float(config.dataset.datasets.go2.metric_waypoint_spacing)
    assert abs(spacing - report['metric_waypoint_spacing']) < 1e-12
    assert {x['episode'] for x in report['episodes'] if x['split'] == 'train'}.isdisjoint(
        {x['episode'] for x in report['episodes'] if x['split'] == 'test'})
    image_hashes = {'train': set(), 'test': set()}
    checks = []
    image_count = 0
    max_quaternion_yaw_error = 0.0
    torch.set_num_threads(2)
    for record in report['segments']:
        path = args.root / 'go2' / record['trajectory']
        rows = list(csv.DictReader((path / 'frame_mapping.csv').open()))
        with (path / 'traj_data.pkl').open('rb') as stream:
            pose = pickle.load(stream)
        assert len(rows) == record['frames'] == len(pose['yaw'])
        assert pose['position'].shape == (len(rows), 2)
        assert np.isfinite(pose['position']).all() and np.isfinite(pose['yaw']).all()
        assert len({row['segment_id'] for row in rows}) == 1
        assert len({row['original_frame_index'] for row in rows}) == len(rows)
        for i, row in enumerate(rows):
            np.testing.assert_array_equal(pose['position'][i], [float(row['x']), float(row['y'])])
            assert pose['yaw'][i] == float(row['yaw'])
            w, x, y, z = [float(row[k]) for k in ('qw', 'qx', 'qy', 'qz')]
            expected_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            error = abs((expected_yaw - pose['yaw'][i] + np.pi) % (2 * np.pi) - np.pi)
            max_quaternion_yaw_error = max(max_quaternion_yaw_error, error)
            assert error < 1e-5
            image_path = path / f'{i}.jpg'
            with Image.open(image_path) as image:
                image.load()
                assert image.size == (1280, 720)
            image_hashes[record['split']].add(hashlib.sha256(image_path.read_bytes()).hexdigest())
            image_count += 1
    assert image_hashes['train'].isdisjoint(image_hashes['test']), 'Identical images across splits'
    for split in ('train', 'test'):
        dataset = TrainingDataset(
            data_folder=str(args.root / 'go2'),
            data_split_folder=str(args.root / 'data_splits' / 'go2' / split),
            dataset_name='go2', image_size=224, min_dist_cat=-64, max_dist_cat=64,
            len_traj_pred=64, traj_stride=1, context_size=4,
            transform=get_transform(), action_stats=config.dataset.action_stats,
            waypoint_spacing=spacing, goals_per_obs=4, motion_condition=config.motion_condition,
            motion_types=['real'])
        expected = sum(x['training_anchors'] for x in report['segments'] if x['split'] == split)
        assert len(dataset) == expected
        # Every anchor respects its continuous segment, including +/-64 targets.
        for trajectory, current, low, high in dataset.index_to_data:
            pose = dataset._get_trajectory(trajectory)
            current, low, high = int(current), int(low), int(high)
            assert current >= 3 and current + 64 < len(pose['yaw'])
            targets = np.array([current + low, current, current + high])
            actual = dataset._get_motion('real', trajectory, current, targets).numpy()
            d = pose['position'][targets] - pose['position'][current]
            yaw = pose['yaw'][current]
            expected_xy = np.stack([np.cos(yaw)*d[:, 0]+np.sin(yaw)*d[:, 1],
                                    -np.sin(yaw)*d[:, 0]+np.cos(yaw)*d[:, 1]], -1) / spacing
            expected_yaw = (pose['yaw'][targets] - yaw + np.pi) % (2*np.pi)-np.pi
            np.testing.assert_allclose(actual, np.column_stack([expected_xy, expected_yaw]), atol=1e-5)
        sample_indices = np.unique(np.linspace(0, len(dataset)-1, min(16, len(dataset))).astype(int))
        for i in sample_indices:
            sample = dataset[int(i)]
            assert sample['motion_type'] == 'real'
            assert sample['motion'].shape == (4, 3)
            assert sample['video'].shape == (8, 3, 224, 224)
            assert torch.isfinite(sample['video']).all() and torch.isfinite(sample['motion']).all()
        checks.append({'split': split, 'anchors_checked': len(dataset), 'image_samples_checked': len(sample_indices),
                       'motion_shape': [4, 3], 'video_shape': [8, 3, 224, 224]})
        print(f'LOADER_OK {split} anchors={len(dataset)} samples={len(sample_indices)}', flush=True)
    summary = {'status': 'passed', 'gpu_used': False, 'images_decoded': image_count,
               'split_episode_overlap': 0, 'split_exact_image_overlap': 0,
               'quaternion_yaw_max_error_rad': max_quaternion_yaw_error, 'loader_checks': checks,
               'limitation': 'Does not establish exposure-time synchronization or disjoint physical routes.'}
    (args.root / 'validation_report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
