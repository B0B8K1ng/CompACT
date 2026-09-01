# Navigation data audit for the CompACT NWM baseline

## Paper population

The CompACT navigation world model uses RECON, SCAND, and the public
low-resolution HuRoN/SACSoN release. It follows the NWM train/test split files
but excludes TartanDrive and Ego4D.

Primary sources:

- Paper, Appendix G.1: https://arxiv.org/html/2603.05438
- Official CompACT repository: https://github.com/kdwonn/CompACT
- Official public HuRoN directory: https://rail.eecs.berkeley.edu/datasets/huron/
- HuRoN/SACSoN project: https://github.com/NHirose/SACSoN

## Current NAS coverage

The NAS root is `/file_system/nas/algorithm/dujun.nie/nwm/data`.

| Dataset | Train split coverage | Test split coverage | Train windows | Test windows |
| --- | ---: | ---: | ---: | ---: |
| RECON | 9,468 / 9,468 | 2,367 / 2,367 | 132,929 | 31,711 |
| HuRoN/SACSoN | 2,278 / 2,451 | 561 / 613 | 106,716 | 27,587 |
| SCAND | 483 / 483 | 121 / 121 | 64,646 | 18,138 |
| Total | 12,229 / 12,402 | 3,049 / 3,101 | 304,291 | 77,436 |

“Windows” are indexed training examples, not image or trajectory counts.

## HuRoN public/private split difference

All 558 public HuRoN bags referenced by the released splits were downloaded
and processed at 4 Hz with NoMaD's `process_bags.py`. Of 3,061 split
trajectories associated with those bags, the public release reproduced 2,839;
222 split trajectories were not generated. A further three split trajectories
refer to `Dec-22-2022/00000000.bag`, which is absent from the official public
RAIL directory. Thus 225 released split names are unavailable in the current
public-data reproduction.

The per-bag verification records are under:

`/file_system/nas/algorithm/dujun.nie/nwm/logs/huron_public_pipeline_state`

This is not an unfinished download: it is a reproducible difference between
the released NWM split names and the public low-resolution HuRoN bags. The
CompACT loader explicitly skips split trajectories that do not exist, which is
consistent with using the public release, but the paper does not publish its
final post-scan trajectory manifest. Therefore bit-for-bit equality with the
authors' processed directory cannot be proven.

## Unofficial candidate

An unofficial Hugging Face upload,
`konpat/huron-sacson-hdf5`, advertises 2,955 trajectories in a 997 MB HDF5
file. It includes some names missing from our public processing, but it is not
an author release, its dataset card has inconsistent provenance, and its HDF5
layout is not directly accepted by CompACT. It should not be merged into the
training data without checking trajectory names, odometry, frame content,
license, and conversion behavior.
