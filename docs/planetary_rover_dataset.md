# Planetary Rover unseen test set

`planetary_rover` is one evaluation-only dataset containing real Mars and Moon
rover observations. It has no train split and is disabled by default in both
NWM dataset configurations, so it cannot silently enter training or latent
precomputation.

## Authoritative sources

- Mars localization: NASA Planetary Data System, Mars 2020 Rover PLACES,
  [DOI 10.17189/btz6-5a82](https://doi.org/10.17189/btz6-5a82), bundle
  `urn:nasa:pds:mars2020_rover_places::16.0`.
- Mars images and PDS4 labels: NASA PDS Mars 2020 Navcam Operations Raw,
  [DOI 10.17189/d3nm-pp09](https://doi.org/10.17189/d3nm-pp09), bundle
  `mars2020_navcam_ops_raw`. Product discovery uses the official JPL PDS
  Imaging Atlas API; only products carrying a published Atlas release ID are
  accepted.
- Moon images and labels: the
  [Chinese Lunar Exploration Program scientific data system](https://moon.bao.ac.cn/)
  (CLEP), Chang'e-4/Yutu-2 PCAM calibrated level 2B, catalogue ID 672. The
  official catalogue lists
  [2019](https://doi.org/10.12350/CLPDS.GRAS.CE4.PCAM-2B-2019.vB),
  [2020](https://doi.org/10.12350/CLPDS.GRAS.CE4.PCAM-2B-2020.vB), and
  [2021](https://doi.org/10.12350/CLPDS.GRAS.CE4.PCAM-2B-2021.vB) DOIs and
  identifies the producer as the National Astronomical Observatories/CLEP
  Ground Research and Application System.

Every selected frame records its official label URL, image URL, label SHA-256,
downloaded source-image SHA-256, and processed-JPEG SHA-256 in
`frame_metadata.jsonl`. Checksums for the source indexes and PLACES tables are
in the cache `sources/source_manifest.json`. The CLEP host currently serves an
expired TLS certificate; the builder records this fact, connects only to the
fixed official host, and hashes every returned payload. Users redistributing
the source products remain responsible for the providers' citation and usage
terms.

No simulated image, synthesized frame, visual odometry, learned pose, SLAM
pose, or interpolated image is used.

## Selection and poses

Mars uses the left regular EDR Navcam view and keeps one representative product
per `(site, drive)`. Only `VCE_` and `TRAV` activities with absolute rover-frame
camera azimuth at most 5 degrees are eligible. Position is the official PLACES
`best_interp` solution when the RMC is present. Otherwise, official image-time
telemetry is mapped into PLACES with the local similarity transform bracketed
by neighboring PLACES RMCs. Yaw comes from the official telemetry quaternion.
Segments do not cross a site, a displacement above 5 m, an SCLK gap above
900,000 s, or a wrapped yaw step above 120 degrees.

For Moon, a PCAM observation sequence is a panorama taken at one rover pose.
Using panorama frames as vehicle motion would be incorrect, so exactly one
left-eye image minimizing the magnitude of the official wrapped mast yaw and
mast pitch is selected from each sequence; the accepted score must be within
30 degrees. Rover body yaw is recovered entirely from two official label
quantities:

```text
rover_yaw = wrap(atan2(center_point_observe_vector.y,
                       center_point_observe_vector.x)
                 - Rotation_Angle.yawing)
```

The circular mean across all views at that fixed pose is used. A sequence is
accepted only when its circular standard deviation is at most 2 degrees; the
median over all official sequences is about 0.30 degrees. As a direct audit,
five independently panned views from sequence 0362 recover one body heading
with circular standard deviation below 0.003 degrees. The within-sequence
pose-span check is 0.05 m.

The Moon observations are genuinely sparse: the raw nonzero adjacent-distance
median is `11.200689830543231 m`, and timestamps are irregular. They remain a
sparse spatial sequence; no frames are filled in. To put Mars and Moon into one
dataset with one action scale, Moon XY written to `traj_data.pkl` is transformed
reversibly:

```text
evaluation_xy = (official_rover_xy_m - [-1.759764, -12.692296])
                * 0.08697457907173892
official_rover_xy_m = evaluation_xy / 0.08697457907173892
                      + [-1.759764, -12.692296]
```

This makes the lunar nonzero median step equal to the Mars waypoint spacing,
`0.9741752833246042`. The unmodified official `rover_xyz`, explicit raw XY,
transform, timestamps, and label values remain in per-frame provenance.

CLEP stores the grayscale DN in an `UnsignedLSB2` container; selected products
populate the 10-bit range `0..1023`. The fixed display conversion retains the
most-significant eight bits (`uint8 = uint16_dn >> 2`) before cropping and
resizing. It is scene-independent—there is no per-image histogram or percentile
stretch—and each frame records its observed source DN range and formula.

## Layout and action convention

The installed data root is:

```text
/file_system/nas/algorithm/dujun.nie/nwm/data/planetary_rover/
  mars_perseverance_navcam_0000/
    0.jpg
    1.jpg
    ...
    traj_data.pkl
    frame_metadata.jsonl
    metadata.json
  ...
  moon_yutu2_pcam_0000/
  dataset_report.json
  _SUCCESS
```

This is the same loader-facing layout as RECON, SCAND, HuRoN/SACSoN,
TartanDrive, and Go Stanford. Each image is RGB `224 x 224`; the official image
is center-square cropped and resized. `traj_data.pkl` has exactly:

```python
{
    "position": np.ndarray[N, 2],
    "yaw": np.ndarray[N],
}
```

Yaw is in radians. `BaseDataset` expresses future positions in the observation
frame and divides XY by the configured waypoint spacing. `EvalDataset` then
produces `[dx, dy, dyaw]`. For this dataset only, `dyaw` is the shortest signed
angle in `[-pi, pi)`; this prevents false `2*pi` jumps at the branch cut while
leaving historical datasets unchanged.

The only split is
`data_splits/planetary_rover/test/traj_names.txt`. There is deliberately no
train path. Inference uses the complete deterministic test index rather than a
separately sampled `time.pkl` or `rollout.pkl`; set `NWM_INDEX_ROOT` to a NAS
cache directory to keep generated loader indexes out of the repository.

Frame indices in this dataset are spatial observation indices, not a fixed-rate
camera clock. In particular, lunar timestamps can be separated by days. Do not
label frame offsets as seconds or compare time-based FPS metrics directly with
4 Hz terrestrial datasets. Use spatial-step or action-conditioned evaluation;
the config records `temporal_semantics: spatial_index` to make this explicit.

## Frozen scale

With `context_size=4` and `len_traj_pred=64`, a trajectory of length `N`
contributes `N-67` windows:

| subset | trajectories | frames | windows |
|---|---:|---:|---:|
| Mars | 75 | 11,278 | 6,253 |
| Moon | 1 | 135 | 68 |
| combined test set | 76 | 11,413 | 6,321 |

For comparison, counts from the locally installed test splits are:

| dataset | frames | windows |
|---|---:|---:|
| TartanDrive | 13,197 | 5,742 |
| planetary_rover | 11,413 | 6,321 |
| Go Stanford | 26,463 | 16,815 |
| SCAND | 22,757 | 18,138 |
| HuRoN/SACSoN (available trajectories) | 46,901 | 27,587 |
| RECON | 120,920 | 31,711 |

Thus the new test set is closest to TartanDrive in both frame and window
count. It is large enough to be a full unseen test set but is never an
adaptation or training set.

## Rebuild and validation

The builder is `scripts/prepare_planetary_rover.py`. Large data and all
authoritative caches stay on NAS:

```bash
conda run --no-capture-output -n nwm-preprocess python -u \
  scripts/prepare_planetary_rover.py all \
  --cache-root /file_system/nas/algorithm/dujun.nie/datasets/planetary_rover/cache \
  --output-root /file_system/nas/algorithm/dujun.nie/nwm/data/planetary_rover \
  --split-output data_splits/planetary_rover/test/traj_names.txt \
  --workers 32

conda run --no-capture-output -n nwm-preprocess python -u \
  scripts/prepare_planetary_rover.py validate \
  --cache-root /file_system/nas/algorithm/dujun.nie/datasets/planetary_rover/cache \
  --output-root /file_system/nas/algorithm/dujun.nie/nwm/data/planetary_rover \
  --split-output data_splits/planetary_rover/test/traj_names.txt
```

The build is frame-resumable. A trajectory is committed only after every
image, pose array, provenance row, checksum, and shape check succeeds.
