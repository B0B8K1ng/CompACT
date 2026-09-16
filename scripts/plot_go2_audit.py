"""Plot actual navigation coverage of the converted Go2 training trajectories."""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prepare_go2_nwm import local_deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / 'conversion_report.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {'train': '#287a9f', 'test': '#e48636'}
    coverage = {}
    for split in ('train', 'test'):
        motion, durations = [], []
        first = True
        for record in report['segments']:
            if record['split'] != split:
                continue
            with (args.root / 'go2' / record['trajectory'] / 'traj_data.pkl').open('rb') as stream:
                pose = pickle.load(stream)
            p, yaw = pose['position'], pose['yaw']
            axes[0].plot(p[:, 0], p[:, 1], color=colors[split], alpha=.65, lw=1.2,
                         label=split if first else None)
            first = False
            if record['eligible']:
                motion.append(local_deltas(p, yaw))
                durations.append(record['sampled_duration_s'])
        values = np.concatenate(motion)
        axes[1].scatter(values[:, 0], values[:, 1], s=8, alpha=.3,
                        color=colors[split], label=split, rasterized=True)
        axes[2].hist(values[:, 2], bins=np.linspace(-.4, .4, 41), density=True,
                     alpha=.55, color=colors[split], label=split)
        coverage[split] = dict(
            eligible_adjacent_pairs=len(values),
            forward_over_2cm_fraction=float(np.mean(values[:, 0] > .02)),
            backward_over_2cm_fraction=float(np.mean(values[:, 0] < -.02)),
            sideways_dominant_over_2cm_fraction=float(np.mean(np.abs(values[:, 1]) > np.maximum(.02, np.abs(values[:, 0])))),
            near_static_fraction=float(np.mean((np.linalg.norm(values[:, :2], axis=1) < .01) & (np.abs(values[:, 2]) < .02))),
            left_turn_over_005rad_fraction=float(np.mean(values[:, 2] > .05)),
            right_turn_over_005rad_fraction=float(np.mean(values[:, 2] < -.05)),
            quantile_levels=[.05, .5, .95],
            dx_dy_dyaw_quantiles=np.quantile(values, [.05, .5, .95], axis=0).tolist())
    axes[0].set(title='Odometry paths (gaps kept separate)', xlabel='World x (m)', ylabel='World y (m)')
    axes[0].axis('equal')
    axes[1].set(title='Observed local translation per ~0.25 s', xlabel='Forward dx (m)', ylabel='Left dy (m)')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(-.18, .18)
    axes[1].set_ylim(-.18, .18)
    axes[1].axhline(0, color='grey', lw=.7)
    axes[1].axvline(0, color='grey', lw=.7)
    axes[2].set(title='Observed yaw change per ~0.25 s', xlabel='CCW dyaw (rad)', ylabel='Density')
    for axis in axes:
        axis.grid(alpha=.15)
        axis.legend()
    fig.suptitle('Go2: 5 training episodes / 2 held-out episodes; navigation motion coverage', fontsize=13)
    fig.savefig(args.root / 'motion_coverage.png', dpi=160)
    fig.savefig(args.root / 'motion_coverage.pdf')
    (args.root / 'eligible_action_coverage.json').write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))


if __name__ == '__main__':
    main()
