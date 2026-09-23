# Planetary rover real-colour evaluation set

`planetary_rover` is a small, evaluation-only NWM test set made from real
mission imagery. It contains 15 Chang'e-4/Yutu-2 Moon clips and
13 Tianwen-1/Zhurong Mars clips. All 239 frames are native colour
captures with 211 adjacent-pose actions. No simulated, colourised,
cropped-to-hide-hardware, duplicated, padded, or temporally interpolated image
is included.

The dataset remains disabled in both `conf/dataset/nwm.yaml` and
`conf/dataset/nwm_real.yaml`. It has no training split and must not enter model
training or latent precomputation.

## Contents

| body | mission and camera | trajectories | frames | actions | view |
|---|---|---:|---:|---:|---|
| Moon | Chang'e-4/Yutu-2 PCAML Level 2B | 15 | 66 | 51 | original full-frame ego view, forward terrain and horizon, no rover component or rover shadow |
| Mars | Tianwen-1/Zhurong NaTeCamA Level 2C | 13 | 173 | 160 | rover-mounted forward or forward-oblique terrain with horizon; no nadir view |
| **total** |  | **28** | **239** | **211** |  |

The exact trajectory names and per-trajectory counts are frozen in
`dataset_report.json` and `source_manifest.json`. The source products and labels
come from the NAOC/CNSA Ground Research and Application System planetary
archive:

- Moon: [Chang'e-4 rover archive](https://moon.bao.ac.cn/PUBDATA/CE4ROLL/),
  PCAML `COLOR` Level 2B products.
- Mars: [Tianwen-1 rover archive](https://moon.bao.ac.cn/WEBDATA/HX1ROLL/HX1-Ro/NaTeCamA/2C/),
  NaTeCamA Level 2C products.

`source_manifest.json` and each `frame_metadata.jsonl` record the exact product
and label URLs and SHA-256 hashes. Provider citation and redistribution terms
still apply to the original products.

## Image selection and processing

All 66 accepted Moon source frames were reviewed as complete original
images. Every frame shows forward terrain near the horizon and contains neither
a rover component nor a rover shadow. No crop, mask, inpainting, retouching, or
artificial colourisation was used to meet that condition. Frames with a
downward-looking composition, severe glare or overexposure, or colour-row
artifacts were excluded; no action crosses an excluded frame or clip boundary.

The lunar products contain a physical 10-bit RGGB Bayer mosaic. Processing is
fixed for every frame: little-endian `uint16`, OpenCV
`COLOR_BayerRGGB2RGB`, then `rgb_uint8 = rgb_uint16 >> 2`. There is no
scene-dependent stretch, white balance, or artificial colourisation. The
natural lunar colour is consequently low in saturation. The complete
2352 x 1728 image content is resized to 224 x 224 with LANCZOS; it is not
cropped.

The 173 Mars frames are native three-channel NaTeCamA products. Their
complete source frames are resized to 224 x 224 with LANCZOS. Every accepted
view is rover-mounted, forward or forward-oblique, and includes the horizon;
drone, nadir, and downward-looking views are excluded. A thin yellow
rover-mounted element remains visible at the lower edge of some Mars frames;
the full images are retained instead of hiding it.

All 211 adjacent image pairs passed reciprocal feature matching and robust
geometric verification. These checks establish useful visual overlap between
the selected observations; they do not imply a fixed capture rate or an
observed continuous drive.

## Official poses and action semantics

The dataset does not use three-dimensional reconstruction, visual odometry,
SLAM, learned poses, or interpolated poses:

- Moon and Mars camera XY endpoints are the sum of official rover XY and the
  official exterior-orientation camera-center XY offset in the declared global
  coordinate frame. Camera yaw is the official optical-axis direction projected
  into that same frame. The camera-center offset is already global and is never
  rotated a second time by rover yaw.

For loader compatibility, each trajectory translates its first official XY
position to the origin. A moving trajectory applies one uniform scale so that
its median nonzero adjacent translation is 1. A rotation-only trajectory keeps
its fixed XY at zero. The exact official metre coordinates and reversible
transform remain in `frame_metadata.jsonl` and `metadata.json`. Thus
`metric_waypoint_spacing: 1.0` denotes sequence-normalized units, not metres;
metric navigation evaluation is explicitly disabled.

For adjacent captures `i` and `i + 1`, the action is the net planar pose change
expressed in camera/ego axes at capture `i`:

```text
world_delta = normalized_xy[i + 1] - normalized_xy[i]
forward     =  cos(yaw[i]) * world_delta.x + sin(yaw[i]) * world_delta.y
left        = -sin(yaw[i]) * world_delta.x + cos(yaw[i]) * world_delta.y
delta_yaw   = wrap(yaw[i + 1] - yaw[i])
action      = [forward, left, delta_yaw]
```

This is an official-endpoint-derived net SE(2) displacement. It is not
throttle, steering, wheel speed, motor command, or a record of the unobserved
path between the two captures.

Some Moon clips are camera-turn observations at a fixed official rover XY.
Their camera-center endpoints can still move because the published global
camera-center offset changes with the camera pose. The recorded action uses
those camera endpoints and the wrapped difference between adjacent official
optical-axis yaws. It does not claim wheel travel or a low-level vehicle
control.

## Installed layout

The dataset root is:

```text
/file_system/nas/algorithm/dujun.nie/nwm/data/planetary_rover/
  moon_yutu2_pcam_color_*/
  mars_zhurong_natecam_color_*/
  dataset_config.json
  dataset_report.json
  source_manifest.json
  validation.json
  preview.jpg
  _SUCCESS
```

Each trajectory contains contiguous `0.jpg`, `1.jpg`, ... images,
`traj_data.pkl`, `actions.json`, `frame_metadata.jsonl`, `metadata.json`, and
`pair_geometry.json`. `traj_data.pkl` keeps the standard loader payload:

```python
{
    "position": np.ndarray[N, 2],  # sequence-normalized units
    "yaw": np.ndarray[N],          # radians
}
```

The evaluation split is installed at
`data_splits/planetary_rover/test/`. Its frozen counts are:

| index | samples | configured future length | source trajectories |
|---|---:|---:|---|
| `time.pkl` | 10 | 16 | eligible Mars clips only |
| `navigation_eval.pkl` | 47 | 8 | eligible Mars clips only |
| `rollout.pkl` | 0 | 64 | none |

`traj_names.txt` lists all 28 accepted trajectories. Short clips remain
available for qualitative, action-conditioned sequence tests but do not enter
an indexed task unless they contain a complete context and future window. The
indexes contain no duplicates, padding, cross-clip samples, or samples that
cross a failed visual-overlap edge. Inference therefore uses the
planetary-only overrides `prediction_sample_count: 10` and
`eval_len_traj_pred: 16`; planning uses `navigation_sample_count: 47`.
Other datasets retain their existing defaults.

Capture intervals are irregular mission observation intervals, sometimes much
longer than terrestrial video frame intervals. Frame indices are ordered
observations and do not define a fixed FPS. Do not report time-based FPS claims
or metre-scale navigation metrics from this normalized test set. Use it for
small qualitative and out-of-distribution NWM evaluation, and report the exact
split and action convention with any result.
