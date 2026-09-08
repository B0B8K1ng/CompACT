#!/usr/bin/env python3
"""Roll out an NWM from one image and a user-defined navigation trajectory.

The demo follows the protocol used by NWM Figures 11--15: one first image is
repeated to cold-start the model context, then up to 64 future frames are
generated autoregressively at 4 FPS.  The output contains every frame,
synchronized MP4 comparisons, the resolved actions, a white-background yellow
trajectory panel, and a paper-style contact sheet.  Rollouts up to 4 seconds
include 1s, 2s, 3s, and 4s snapshots; longer rollouts include 2s onward.

Examples
--------
Use one of the built-in trajectories::

    conda run -n nwm python demo_nwm_rollout.py \
      --checkpoint /path/to/experiment/checkpoints/joint_0100000.pth.tar \
      --first-image /path/to/image.jpg \
      --preset forward_then_left \
      --output-dir /path/to/output

Use custom per-frame deltas from JSON::

    uv run demo_nwm_rollout.py ... --actions actions.json

``actions.json`` may contain ``actions`` (one ``[dx, dy, d_yaw]`` row per
frame) or motion ``segments``.  Run with ``--write-actions-example PATH`` to
write a documented template.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FPS = 4
MAX_SECONDS = 16.0
ACTION_DIM = 3
BACKGROUND = (18, 21, 27)
LABEL = (244, 245, 247)
MUTED = (170, 177, 190)
YELLOW = (255, 210, 45)

PRESET_DESCRIPTIONS = {
    "forward": "匀速直行",
    "forward_then_left": "前 1/4 时长直行，随后向左转弯",
    "forward_then_right": "前 1/4 时长直行，随后向右转弯",
    "s_curve": "直行后先左转、再右转",
    "stop_and_go": "直行 4 秒、停止 4 秒、再直行",
}


def _segments_for_preset(
    name: str, seconds: float, speed: float, turn_rate_deg: float
) -> list[dict[str, float]]:
    """Build body-frame velocity segments for a named trajectory."""
    if name not in PRESET_DESCRIPTIONS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESET_DESCRIPTIONS)}")
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    def segment(duration: float, yaw_rate: float = 0.0, velocity: float = speed):
        return {
            "duration_seconds": duration,
            "forward_speed": velocity,
            "lateral_speed": 0.0,
            "yaw_rate_deg": yaw_rate,
        }

    if name == "forward":
        return [segment(seconds)]
    if name == "stop_and_go":
        first = min(4.0, seconds)
        stop = min(4.0, max(0.0, seconds - first))
        result = [segment(first)]
        if stop:
            result.append(segment(stop, velocity=0.0))
        if seconds > first + stop:
            result.append(segment(seconds - first - stop))
        return result

    # Preserve the paper-style 4s straight prefix for a 16s rollout while
    # ensuring short rollouts still contain a visible turn (1s + 3s at 4s).
    straight = seconds / 4.0
    remaining = seconds - straight
    result = [segment(straight)]
    if remaining <= 0:
        return result
    if name == "forward_then_left":
        return result + [segment(remaining, abs(turn_rate_deg))]
    if name == "forward_then_right":
        return result + [segment(remaining, -abs(turn_rate_deg))]

    first_turn = remaining / 2.0
    return result + [
        segment(first_turn, abs(turn_rate_deg)),
        segment(remaining - first_turn, -abs(turn_rate_deg)),
    ]


def segments_to_actions(
    segments: Sequence[Mapping[str, Any]], fps: int = FPS
) -> np.ndarray:
    """Integrate body-frame velocities into initial-frame pose deltas.

    Translation fields are velocities per second.  Positive ``forward_speed``
    moves along the current heading, positive ``lateral_speed`` moves left, and
    positive yaw is a counter-clockwise/left turn.  Returned translation deltas
    are expressed in the coordinate frame of the first image, matching the
    evaluation code in :mod:`datasets`.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    actions: list[list[float]] = []
    heading = 0.0
    for index, segment in enumerate(segments):
        duration = float(segment.get("duration_seconds", 0.0))
        exact_frames = duration * fps
        frame_count = round(exact_frames)
        if duration <= 0 or not math.isclose(exact_frames, frame_count, abs_tol=1e-6):
            raise ValueError(
                f"segments[{index}].duration_seconds must be a positive multiple of {1 / fps:g}"
            )
        forward = float(segment.get("forward_speed", 0.0))
        lateral = float(segment.get("lateral_speed", 0.0))
        if "yaw_rate_rad" in segment and "yaw_rate_deg" in segment:
            raise ValueError(
                f"segments[{index}] cannot define both yaw_rate_rad and yaw_rate_deg"
            )
        yaw_rate = (
            float(segment["yaw_rate_rad"])
            if "yaw_rate_rad" in segment
            else math.radians(float(segment.get("yaw_rate_deg", 0.0)))
        )
        if not all(math.isfinite(value) for value in (forward, lateral, yaw_rate)):
            raise ValueError(f"segments[{index}] contains a non-finite value")

        dt = 1.0 / fps
        for _ in range(frame_count):
            delta_yaw = yaw_rate * dt
            middle_heading = heading + delta_yaw / 2.0
            delta_forward = forward * dt
            delta_lateral = lateral * dt
            delta_x = (
                delta_forward * math.cos(middle_heading)
                - delta_lateral * math.sin(middle_heading)
            )
            delta_y = (
                delta_forward * math.sin(middle_heading)
                + delta_lateral * math.cos(middle_heading)
            )
            actions.append([delta_x, delta_y, delta_yaw])
            heading += delta_yaw
    return np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM)


def _convert_yaw_to_radians(actions: np.ndarray, yaw_unit: str) -> np.ndarray:
    unit = yaw_unit.strip().lower()
    if unit in {"radian", "radians", "rad"}:
        return actions
    if unit in {"degree", "degrees", "deg"}:
        result = actions.copy()
        result[:, 2] = np.deg2rad(result[:, 2])
        return result
    raise ValueError("yaw_unit must be 'radians' or 'degrees'")


def load_actions_file(
    path: Path, expected_frames: int, waypoint_spacing_meters: float | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load custom actions or velocity segments from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("The actions JSON root must be an object")
    file_fps = int(payload.get("fps", FPS))
    if file_fps != FPS:
        raise ValueError(f"This demo requires actions at {FPS} FPS; JSON declares {file_fps}")

    if ("actions" in payload) == ("segments" in payload):
        raise ValueError("The actions JSON must define exactly one of 'actions' or 'segments'")
    if "segments" in payload:
        segments = payload["segments"]
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise TypeError("segments must be a JSON list")
        actions = segments_to_actions(segments, file_fps)
        source_format = "segments"
    else:
        coordinate_frame = str(payload.get("coordinate_frame", "first_image"))
        if coordinate_frame != "first_image":
            raise ValueError(
                "Per-frame actions require coordinate_frame='first_image'. "
                "Use 'segments' for body-frame forward/lateral velocities."
            )
        actions = np.asarray(payload["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must have shape [N, {ACTION_DIM}], got {actions.shape}")
        actions = _convert_yaw_to_radians(actions, str(payload.get("yaw_unit", "radians")))
        source_format = "per_frame_deltas"

    if len(actions) != expected_frames:
        raise ValueError(
            f"Expected exactly {expected_frames} action rows, got {len(actions)}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions contains NaN or infinity")

    translation_unit = str(
        payload.get("translation_unit", "waypoint_spacing_units")
    ).strip().lower()
    spacing = payload.get("waypoint_spacing_meters", waypoint_spacing_meters)
    if translation_unit == "meters":
        if spacing is None or float(spacing) <= 0:
            raise ValueError(
                "Meter actions require positive waypoint_spacing_meters in JSON "
                "or --waypoint-spacing-meters"
            )
        actions = actions.copy()
        actions[:, :2] /= float(spacing)
    elif translation_unit != "waypoint_spacing_units":
        raise ValueError(
            "translation_unit must be 'waypoint_spacing_units' or 'meters'"
        )
    return actions, {
        "kind": "file",
        "path": str(path.resolve()),
        "format": source_format,
        "input_translation_unit": translation_unit,
        "waypoint_spacing_meters": float(spacing) if spacing is not None else None,
    }


def make_preset_actions(
    name: str, seconds: float, speed: float, turn_rate_deg: float
) -> tuple[np.ndarray, dict[str, Any]]:
    segments = _segments_for_preset(name, seconds, speed, turn_rate_deg)
    actions = segments_to_actions(segments, FPS)
    return actions, {
        "kind": "preset",
        "name": name,
        "description": PRESET_DESCRIPTIONS[name],
        "speed_waypoint_units_per_second": speed,
        "turn_rate_degrees_per_second": turn_rate_deg,
        "segments": segments,
    }


def write_actions_example(path: Path) -> None:
    example = {
        "fps": FPS,
        "translation_unit": "waypoint_spacing_units",
        "yaw_unit": "radians",
        "coordinate_frame": "first_image",
        "notes": (
            "Use exactly 4*seconds [dx,dy,d_yaw] rows. Alternatively remove "
            "actions and add segments like the example_segments field."
        ),
        "actions": [[0.25, 0.0, 0.0] for _ in range(64)],
        "example_segments": [
            {"duration_seconds": 4, "forward_speed": 1.0, "yaw_rate_deg": 0},
            {"duration_seconds": 12, "forward_speed": 1.0, "yaw_rate_deg": 7.5},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_experiment_config(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        config = explicit.expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(f"Config does not exist: {config}")
        return config
    current = checkpoint.expanduser().resolve().parent
    for _ in range(5):
        candidate = current / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        "Could not auto-discover .hydra/config.yaml above the checkpoint. "
        "Pass the training config explicitly with --config."
    )


def _checkpoint_state_dict(checkpoint: Any, weights_key: str) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    if weights_key in checkpoint:
        state = checkpoint[weights_key]
    elif checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        state = checkpoint
    else:
        raise KeyError(
            f"Checkpoint has no {weights_key!r} weights. Available keys: "
            f"{sorted(map(str, checkpoint.keys()))}"
        )
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint field {weights_key!r} is not a state dict")
    return state


def _canonical_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = ("module._orig_mod.", "_orig_mod.", "module.")
    result: dict[str, Any] = {}
    for original_key, value in state.items():
        key = str(original_key)
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key in result:
            raise ValueError(f"State-dict key collision after wrapper removal: {key}")
        result[key] = value
    return result


def load_runtime(
    checkpoint_path: Path,
    config_path: Path,
    weights_key: str,
    device_name: str,
    diffusion_steps: int | None,
    compile_model: bool,
):
    """Construct the exact training architecture and load its EMA weights."""
    import torch
    from omegaconf import OmegaConf, open_dict

    import hydra_utils  # noqa: F401 - registers the ${divide:...} resolver
    from train_utils import setup_diffusion, setup_model, setup_tokenizer

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device = torch.device(device_name)
    config = OmegaConf.load(config_path)
    if diffusion_steps is not None:
        if diffusion_steps <= 0:
            raise ValueError("--diffusion-steps must be positive")
        with open_dict(config):
            config.model.diffusion.eval_timestep_respacing = int(diffusion_steps)

    print(f"Loading architecture from {config_path}", flush=True)
    tokenizer = setup_tokenizer(config, device).eval()
    model = setup_model(config, device)
    print(f"Loading {weights_key!r} weights from {checkpoint_path}", flush=True)
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _canonical_state_dict(_checkpoint_state_dict(payload, weights_key))
    model.load_state_dict(state, strict=True)
    del state, payload
    gc.collect()
    model.eval().to(device)
    if compile_model:
        print("Compiling the NWM (the first prediction will take longer)...", flush=True)
        model = torch.compile(model)
    diffusion = setup_diffusion(config, for_eval=True, device=device)
    return config, model, diffusion, tokenizer, device


def _legacy_action_contract(raw_deltas: np.ndarray, config: Any) -> np.ndarray:
    """Reproduce EvalDataset's historical normalization for old checkpoints."""
    cumulative = np.cumsum(raw_deltas, axis=0, dtype=np.float64)
    stats = config.dataset.action_stats
    minimum = np.asarray(stats["min"], dtype=np.float64)
    maximum = np.asarray(stats["max"], dtype=np.float64)
    cumulative[:, :2] = 2.0 * (cumulative[:, :2] - minimum) / (maximum - minimum) - 1.0
    padded = np.concatenate([np.zeros((1, ACTION_DIM)), cumulative], axis=0)
    return np.diff(padded, axis=0).astype(np.float32)


def _model_uses_motion_condition(model: Any) -> bool:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "motion_condition_enabled"):
            return bool(current.motion_condition_enabled)
        current = getattr(current, "_orig_mod", getattr(current, "module", None))
        if current is None:
            break
    return False


def prepare_first_image(path: Path, config: Any):
    from datasets import load_image
    from misc import get_transform, get_unnormalize

    image = load_image(path)
    transform = get_transform(
        int(config.dataset.image_size),
        list(config.dataset.mean),
        list(config.dataset.std),
    )
    normalized = transform(image)
    display = get_unnormalize(
        list(config.dataset.mean), list(config.dataset.std)
    )(normalized.clone()).clamp(0, 1)
    return normalized, display


def load_display_frames(
    frame_dir: Path, start_index: int, frame_count: int, config: Any
) -> list[Image.Image]:
    """Load and preprocess numbered GT frames exactly like model inputs."""
    frames: list[Image.Image] = []
    for index in range(start_index, start_index + frame_count):
        candidates = [
            frame_dir / f"{index}.jpg",
            frame_dir / f"{index}.png",
            frame_dir / f"{index:06d}.jpg",
            frame_dir / f"{index:06d}.png",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"Could not find GT frame {index} in {frame_dir}; tried jpg/png "
                "with plain and six-digit names"
            )
        _, display = prepare_first_image(path, config)
        frames.append(tensor_to_image(display))
    return frames


def run_rollout(
    model: Any,
    diffusion: Any,
    tokenizer: Any,
    first_image: Any,
    actions: np.ndarray,
    context_size: int,
    device: Any,
    seed: int,
    image_mean: Sequence[float],
    image_std: Sequence[float],
    feedback_mode: str = "pixel",
) -> list[Any]:
    """Generate frames using an official NWM autoregressive feedback path.

    ``pixel`` matches :func:`isolated_nwm_infer.generate_rollout`: decode each
    prediction, normalize it, and encode the moving context again on the next
    call. ``latent`` matches ``generate_rollout_efficient``: encode the initial
    context once and recursively append predicted latents.
    """
    import torch
    from isolated_nwm_infer import model_forward_wrapper
    from misc import get_normalize

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    action_tensor = torch.as_tensor(actions, device=device, dtype=torch.float32)
    context = (
        first_image.unsqueeze(0)
        .unsqueeze(1)
        .repeat(1, context_size, 1, 1, 1)
        .to(device)
    )
    if feedback_mode not in {"pixel", "latent"}:
        raise ValueError("feedback_mode must be 'pixel' or 'latent'")
    normalize_for_context = get_normalize(list(image_mean), list(image_std))
    generated: list[Any] = []

    with torch.inference_mode():
        latent_context = None
        if feedback_mode == "latent":
            latent_context = tokenizer.encode(context.flatten(0, 1)).unflatten(
                0, (context.shape[0], context.shape[1])
            )
        for index in range(len(actions)):
            if feedback_mode == "pixel":
                prediction_pixels = model_forward_wrapper(
                    (model, diffusion, tokenizer),
                    context,
                    action_tensor[index : index + 1],
                    num_timesteps=1,
                    latent_size=None,
                    device=device,
                    num_cond=context_size,
                    num_goals=1,
                    progress=False,
                    skip_tokenizer=False,
                    motion_type="real",
                )
                prediction_context = normalize_for_context(prediction_pixels)
                context = torch.cat(
                    [context[:, 1:], prediction_context.unsqueeze(1)], dim=1
                )
            else:
                prediction_latents = model_forward_wrapper(
                    (model, diffusion, tokenizer),
                    latent_context,
                    action_tensor[index : index + 1],
                    num_timesteps=1,
                    latent_size=None,
                    device=device,
                    num_cond=context_size,
                    num_goals=1,
                    progress=False,
                    skip_tokenizer=True,
                    motion_type="real",
                )
                prediction_pixels = tokenizer.decode(
                    prediction_latents, denormalize=True
                )
                latent_context = torch.cat(
                    [latent_context[:, 1:], prediction_latents.unsqueeze(1)], dim=1
                )
            generated.append(
                prediction_pixels[0].detach().float().clamp(0, 1).cpu()
            )
            if (index + 1) % FPS == 0 or index + 1 == len(actions):
                print(
                    f"Generated {index + 1:02d}/{len(actions)} frames "
                    f"({(index + 1) / FPS:.2f}s)",
                    flush=True,
                )
    return generated


def tensor_to_image(tensor: Any) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidate = Path("/usr/share/fonts/truetype/dejavu") / name
    if candidate.is_file():
        return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _trajectory_points(
    actions: np.ndarray, size: tuple[int, int]
) -> list[tuple[int, int]]:
    """Project poses with a fixed lower-center origin and forward pointing up."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [N, {ACTION_DIM}], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contains NaN or infinity")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("trajectory image must have positive dimensions")

    poses = np.concatenate(
        [np.zeros((1, ACTION_DIM), dtype=np.float32), np.cumsum(actions, axis=0)],
        axis=0,
    )
    offsets = np.column_stack((-poses[:, 1], -poses[:, 0])).astype(np.float64)
    origin = np.asarray([width * 0.5, height * 0.76], dtype=np.float64)
    padding = max(8.0, min(width, height) * 0.06)
    min_u, min_v = offsets.min(axis=0)
    max_u, max_v = offsets.max(axis=0)
    candidates: list[float] = []
    for extent, room in (
        (max_u, width - padding - origin[0]),
        (-min_u, origin[0] - padding),
        (max_v, height - padding - origin[1]),
        (-min_v, origin[1] - padding),
    ):
        if extent > 1e-8:
            candidates.append(float(room / extent))
    scale = min(candidates) * 0.94 if candidates else 1.0
    projected = offsets * scale + origin
    return [(int(round(x)), int(round(y))) for x, y in projected]


def render_trajectory_panel(
    actions: np.ndarray,
    size: tuple[int, int] = (224, 224),
    visible_steps: int | None = None,
) -> Image.Image:
    """Draw a yellow trajectory on white, optionally revealing only a prefix.

    Geometry and scale always come from the complete trajectory, so the path
    extends without moving or rescaling in the synchronized rollout video.
    """
    points = _trajectory_points(actions, size)
    if visible_steps is None:
        visible_steps = len(points) - 1
    visible_steps = max(0, min(int(visible_steps), len(points) - 1))
    visible = points[: visible_steps + 1]
    canvas = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    unit = max(1, round(min(size) / 224))
    if len(visible) > 1:
        draw.line(visible, fill=YELLOW, width=max(3, 4 * unit), joint="curve")
    marker_radius = max(2, 3 * unit)
    for frame in range(FPS, visible_steps + 1, FPS):
        x, y = points[frame]
        draw.ellipse(
            (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
            fill=YELLOW,
        )
    start_radius = max(3, 5 * unit)
    sx, sy = points[0]
    draw.ellipse(
        (sx - start_radius, sy - start_radius, sx + start_radius, sy + start_radius),
        fill=YELLOW,
    )
    if visible_steps > 0:
        cx, cy = points[visible_steps]
        current_radius = max(3, 5 * unit)
        draw.ellipse(
            (cx - current_radius, cy - current_radius, cx + current_radius, cy + current_radius),
            fill=YELLOW,
        )
    return canvas


def render_contact_sheet(
    trajectory: Image.Image,
    first: Image.Image,
    generated: Sequence[Image.Image],
    output: Path,
) -> list[float]:
    last_complete_second = min(MAX_SECONDS, len(generated) / FPS)
    first_sample_second = 1 if last_complete_second <= 4 else 2
    seconds = [
        float(value)
        for value in range(first_sample_second, int(last_complete_second) + 1)
    ]
    panels = [trajectory, first, *[generated[int(second * FPS) - 1] for second in seconds]]
    labels = ["Action trajectory", "First image", *[f"{second:g}s" for second in seconds]]
    cell = (224, 224)
    header = 38
    canvas = Image.new("RGB", (cell[0] * len(panels), cell[1] + header), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = _font(16, bold=True)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * cell[0]
        fitted = panel.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x, header))
        draw.text((x + 10, 9), label, fill=LABEL, font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return seconds


def _labeled_panels(
    panels: Sequence[Image.Image], labels: Sequence[str]
) -> Image.Image:
    if len(panels) != len(labels) or not panels:
        raise ValueError("panels and labels must have the same non-zero length")
    cell = panels[0].size
    header = 32
    canvas = Image.new("RGB", (cell[0] * len(panels), cell[1] + header), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = _font(15, bold=True)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * cell[0]
        canvas.paste(panel.convert("RGB").resize(cell, Image.Resampling.LANCZOS), (x, header))
        draw.text((x + 9, 7), label, fill=LABEL, font=font)
    return canvas


def render_video_frames(
    generated: Sequence[Image.Image],
    actions: np.ndarray,
    output: Path,
    gt_frames: Sequence[Image.Image] | None = None,
) -> None:
    if gt_frames is not None and len(gt_frames) != len(generated):
        raise ValueError("GT and rollout must have the same number of frames")
    output.mkdir(parents=True, exist_ok=False)
    for index, rollout in enumerate(generated, start=1):
        trajectory = render_trajectory_panel(actions, rollout.size, visible_steps=index)
        if gt_frames is None:
            panels = [rollout, trajectory]
            labels = ["NWM rollout", f"Action through {index / FPS:.2f}s"]
        else:
            panels = [gt_frames[index - 1], rollout, trajectory]
            labels = ["Ground truth", "NWM rollout", f"Action through {index / FPS:.2f}s"]
        _labeled_panels(panels, labels).save(
            output / f"frame_{index:04d}.png", optimize=True
        )


def render_gt_comparison_sheet(
    trajectory: Image.Image,
    first: Image.Image,
    generated: Sequence[Image.Image],
    gt_frames: Sequence[Image.Image],
    output: Path,
) -> list[float]:
    if len(generated) != len(gt_frames):
        raise ValueError("GT and rollout must have the same number of frames")
    duration = len(generated) / FPS
    first_sample_second = 1 if duration <= 4 else 2
    seconds = [
        float(value) for value in range(first_sample_second, int(duration) + 1)
    ]
    cell = (224, 224)
    header = 38
    row_label = 112
    columns = 2 + len(seconds)
    canvas = Image.new(
        "RGB", (row_label + cell[0] * columns, header + cell[1] * 2), BACKGROUND
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16, bold=True)
    column_labels = ["Trajectory", "First image", *[f"{second:g}s" for second in seconds]]
    for column, label in enumerate(column_labels):
        draw.text((row_label + column * cell[0] + 9, 9), label, fill=LABEL, font=font)
    draw.text((9, header + 100), "Ground truth", fill=LABEL, font=font)
    draw.text((9, header + cell[1] + 100), "NWM rollout", fill=LABEL, font=font)
    common = [trajectory, first]
    gt_panels = [*common, *[gt_frames[int(second * FPS) - 1] for second in seconds]]
    rollout_panels = [*common, *[generated[int(second * FPS) - 1] for second in seconds]]
    for row, panels in enumerate((gt_panels, rollout_panels)):
        for column, panel in enumerate(panels):
            fitted = panel.convert("RGB").resize(cell, Image.Resampling.LANCZOS)
            canvas.paste(fitted, (row_label + column * cell[0], header + row * cell[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return seconds


def encode_video(frame_dir: Path, output: Path, frame_count: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create rollout.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        "1",
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def save_visual_artifacts(
    output: Path,
    first: Image.Image,
    generated: Sequence[Image.Image],
    actions: np.ndarray,
    gt_frames: Sequence[Image.Image] | None = None,
) -> tuple[dict[str, str], list[float]]:
    """Save figures and videos from already-generated display-space frames."""
    trajectory = render_trajectory_panel(actions, first.size)
    trajectory_path = output / "trajectory.png"
    trajectory.save(trajectory_path, optimize=True)

    frames_dir = output / "frames"
    if not frames_dir.exists():
        frames_dir.mkdir()
        first.save(frames_dir / "frame_0000.png", optimize=True)
        for index, image in enumerate(generated, start=1):
            image.save(frames_dir / f"frame_{index:04d}.png", optimize=True)

    prediction_only_path = output / "rollout_prediction_only.mp4"
    encode_video(frames_dir, prediction_only_path, len(generated))

    rollout_video_frames = output / "rollout_video_frames"
    render_video_frames(generated, actions, rollout_video_frames)
    rollout_path = output / "rollout.mp4"
    encode_video(rollout_video_frames, rollout_path, len(generated))

    figure_path = output / "figure_11_15_style.png"
    sampled_seconds = render_contact_sheet(
        trajectory, first, generated, figure_path
    )
    artifacts = {
        "frames": str(frames_dir),
        "prediction_only_video": str(prediction_only_path),
        "rollout_with_trajectory_video": str(rollout_path),
        "rollout_video_frames": str(rollout_video_frames),
        "figure": str(figure_path),
        "trajectory": str(trajectory_path),
    }

    if gt_frames is not None:
        gt_dir = output / "gt_frames"
        gt_dir.mkdir()
        for index, image in enumerate(gt_frames, start=1):
            image.save(gt_dir / f"frame_{index:04d}.png", optimize=True)
        gt_path = output / "ground_truth.mp4"
        encode_video(gt_dir, gt_path, len(gt_frames))

        comparison_frames = output / "comparison_video_frames"
        render_video_frames(generated, actions, comparison_frames, gt_frames)
        comparison_path = output / "ground_truth_vs_rollout.mp4"
        encode_video(comparison_frames, comparison_path, len(generated))
        comparison_figure = output / "ground_truth_vs_rollout.png"
        render_gt_comparison_sheet(
            trajectory, first, generated, gt_frames, comparison_figure
        )
        artifacts.update(
            {
                "ground_truth_frames": str(gt_dir),
                "ground_truth_video": str(gt_path),
                "comparison_video": str(comparison_path),
                "comparison_video_frames": str(comparison_frames),
                "comparison_figure": str(comparison_figure),
            }
        )
    return artifacts, sampled_seconds


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_action_record(output: Path, actions: np.ndarray, source: Mapping[str, Any]) -> Path:
    poses = np.cumsum(actions, axis=0)
    records = [
        {
            "frame": index + 1,
            "time_seconds": (index + 1) / FPS,
            "delta": [float(value) for value in delta],
            "pose_from_first_image": [float(value) for value in pose],
        }
        for index, (delta, pose) in enumerate(zip(actions, poses))
    ]
    path = output / "resolved_actions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": FPS,
                "translation_unit": "waypoint_spacing_units",
                "yaw_unit": "radians",
                "coordinate_frame": "first_image",
                "source": dict(source),
                "steps": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 4-FPS, up-to-16-second autoregressive NWM rollout from one image."
    )
    parser.add_argument("--checkpoint", type=Path, help="NWM .pth.tar checkpoint")
    parser.add_argument(
        "--config", type=Path, help="Training config; auto-discovered from the checkpoint"
    )
    parser.add_argument("--first-image", type=Path, help="Path to the first RGB image")
    parser.add_argument("--output-dir", type=Path, help="A new or empty output directory")
    parser.add_argument(
        "--postprocess-from",
        type=Path,
        help="Reuse raw frames/actions from an earlier demo output without model inference",
    )
    parser.add_argument(
        "--gt-frames-dir",
        type=Path,
        help="Directory containing numbered source frames for GT comparison",
    )
    parser.add_argument(
        "--gt-start-index",
        type=int,
        help="Index of the first future GT frame (normally current_time + 1)",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--actions", type=Path, help="Custom action/segment JSON")
    action_group.add_argument(
        "--preset", choices=sorted(PRESET_DESCRIPTIONS), default="forward"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=MAX_SECONDS,
        help="Rollout length, maximum 16 (use 4 for real-world images)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Preset forward speed in waypoint units/second"
    )
    parser.add_argument(
        "--turn-rate-deg", type=float, default=7.5, help="Preset turn rate in degrees/second"
    )
    parser.add_argument(
        "--waypoint-spacing-meters",
        type=float,
        help="Convert meter-valued custom translation actions to training waypoint units",
    )
    parser.add_argument("--weights-key", default="ema", help="Checkpoint state-dict field")
    parser.add_argument("--device", default="cuda", help="For example cuda, cuda:1, or cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feedback-mode",
        choices=("pixel", "latent"),
        default="pixel",
        help="generate_rollout decoded-pixel feedback or latent-only autoregression",
    )
    parser.add_argument("--diffusion-steps", type=int, help="Override checkpoint evaluation steps")
    parser.add_argument("--compile", action="store_true", help="Compile the NWM before rollout")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--write-actions-example", type=Path, metavar="PATH")
    return parser.parse_args()


def _validated_frame_count(seconds: float) -> int:
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError(f"--seconds must be in (0, {MAX_SECONDS:g}]")
    exact = seconds * FPS
    frames = round(exact)
    if not math.isclose(exact, frames, abs_tol=1e-6):
        raise ValueError(f"--seconds must be a multiple of {1 / FPS:g}")
    return frames


def _load_resolved_actions(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    actions = np.asarray([step["delta"] for step in steps], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Invalid resolved action record: {path}")
    return actions, dict(payload.get("source", {"kind": "postprocessed"}))


def _validate_gt_args(args: argparse.Namespace) -> tuple[Path, int] | None:
    if (args.gt_frames_dir is None) != (args.gt_start_index is None):
        raise ValueError("--gt-frames-dir and --gt-start-index must be provided together")
    if args.gt_frames_dir is None:
        return None
    frame_dir = args.gt_frames_dir.expanduser().resolve()
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"GT frames directory does not exist: {frame_dir}")
    return frame_dir, int(args.gt_start_index)


def _postprocess_existing(args: argparse.Namespace, gt_spec: tuple[Path, int] | None) -> None:
    from omegaconf import OmegaConf

    source = args.postprocess_from.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Existing demo output does not exist: {source}")
    source_manifest_path = source / "manifest.json"
    source_actions_path = source / "resolved_actions.json"
    source_frames = source / "frames"
    for required in (source_manifest_path, source_actions_path, source_frames):
        if not required.exists():
            raise FileNotFoundError(f"Postprocess source is missing: {required}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    actions, action_source = _load_resolved_actions(source_actions_path)
    frame_count = _validated_frame_count(args.seconds)
    if frame_count > len(actions):
        raise ValueError(
            f"Requested {frame_count} postprocessed frames, but {source} contains "
            f"only {len(actions)}"
        )
    actions = actions[:frame_count]
    first = Image.open(source_frames / "frame_0000.png").convert("RGB")
    generated = [
        Image.open(source_frames / f"frame_{index:04d}.png").convert("RGB")
        for index in range(1, frame_count + 1)
    ]
    output = _prepare_output(args.output_dir)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    first.save(frames_dir / "frame_0000.png", optimize=True)
    for index, image in enumerate(generated, start=1):
        image.save(frames_dir / f"frame_{index:04d}.png", optimize=True)
    action_record = _save_action_record(output, actions, action_source)

    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else Path(source_manifest["config"]).expanduser().resolve()
    )
    config = OmegaConf.load(config_path)
    gt_frames = None
    if gt_spec is not None:
        gt_frames = load_display_frames(gt_spec[0], gt_spec[1], frame_count, config)
    artifacts, sampled_seconds = save_visual_artifacts(
        output, first, generated, actions, gt_frames
    )
    artifacts["actions"] = str(action_record)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "postprocessed_from": str(source),
            "seconds": frame_count / FPS,
            "generated_frame_count": frame_count,
            "sampled_seconds": sampled_seconds,
            "gt_frames_dir": str(gt_spec[0]) if gt_spec is not None else None,
            "gt_start_index": gt_spec[1] if gt_spec is not None else None,
            "image_preprocessing": {
                "center_crop_aspect_ratio": "4:3",
                "resize": [int(config.dataset.image_size)] * 2,
                "crop_order": "center_crop_then_resize",
            },
            "artifacts": artifacts,
        }
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Done (no model inference): {output}")
    print(f"Complete rollout: {artifacts['rollout_with_trajectory_video']}")
    if gt_frames is not None:
        print(f"GT comparison: {artifacts['comparison_video']}")


def main() -> None:
    args = parse_args()
    if args.list_presets:
        for name, description in PRESET_DESCRIPTIONS.items():
            print(f"{name:20s} {description}")
        return
    if args.write_actions_example is not None:
        write_actions_example(args.write_actions_example.expanduser().resolve())
        print(args.write_actions_example.expanduser().resolve())
        return
    if args.output_dir is None:
        raise ValueError("Missing required argument: --output-dir")
    gt_spec = _validate_gt_args(args)
    if args.postprocess_from is not None:
        _postprocess_existing(args, gt_spec)
        return

    missing = [
        name
        for name, value in (
            ("--checkpoint", args.checkpoint),
            ("--first-image", args.first_image),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    checkpoint = args.checkpoint.expanduser().resolve()
    first_image_path = args.first_image.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not first_image_path.is_file():
        raise FileNotFoundError(f"First image does not exist: {first_image_path}")
    frame_count = _validated_frame_count(args.seconds)
    if args.actions is not None:
        action_path = args.actions.expanduser().resolve()
        if not action_path.is_file():
            raise FileNotFoundError(f"Actions file does not exist: {action_path}")
        raw_actions, action_source = load_actions_file(
            action_path, frame_count, args.waypoint_spacing_meters
        )
    else:
        raw_actions, action_source = make_preset_actions(
            args.preset, args.seconds, args.speed, args.turn_rate_deg
        )
    if len(raw_actions) != frame_count:
        raise AssertionError("Internal action generator returned the wrong number of frames")

    output = _prepare_output(args.output_dir)
    action_record = _save_action_record(output, raw_actions, action_source)
    config_path = find_experiment_config(checkpoint, args.config)

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    config, model, diffusion, tokenizer, device = load_runtime(
        checkpoint,
        config_path,
        args.weights_key,
        args.device,
        args.diffusion_steps,
        args.compile,
    )
    context_size = int(config.dataset.context_size)
    normalized_first, display_first = prepare_first_image(first_image_path, config)
    uses_motion = _model_uses_motion_condition(model)
    model_actions = (
        raw_actions if uses_motion else _legacy_action_contract(raw_actions, config)
    )
    generated_pixels = run_rollout(
        model,
        diffusion,
        tokenizer,
        normalized_first,
        model_actions,
        context_size,
        device,
        args.seed,
        config.dataset.mean,
        config.dataset.std,
        args.feedback_mode,
    )

    frames_dir = output / "frames"
    frames_dir.mkdir()
    first_pil = tensor_to_image(display_first)
    first_pil.save(frames_dir / "frame_0000.png", optimize=True)
    generated_pil = [tensor_to_image(frame) for frame in generated_pixels]
    for index, image in enumerate(generated_pil, start=1):
        image.save(frames_dir / f"frame_{index:04d}.png", optimize=True)
    gt_frames = None
    if gt_spec is not None:
        gt_frames = load_display_frames(gt_spec[0], gt_spec[1], frame_count, config)
    artifacts, sampled_seconds = save_visual_artifacts(
        output, first_pil, generated_pil, raw_actions, gt_frames
    )
    artifacts["actions"] = str(action_record)

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "weights_key": args.weights_key,
        "first_image": str(first_image_path),
        "seed": args.seed,
        "fps": FPS,
        "seconds": args.seconds,
        "generated_frame_count": len(generated_pil),
        "context_size": context_size,
        "context_initialization": "repeat_one_normalized_first_image",
        "context_update": "append_prediction_and_keep_latest_context_size_frames",
        "autoregressive_feedback": (
            "decode_normalize_reencode"
            if args.feedback_mode == "pixel"
            else "latent_only"
        ),
        "motion_condition_enabled": uses_motion,
        "diffusion_steps_override": args.diffusion_steps,
        "action_source": action_source,
        "sampled_seconds": sampled_seconds,
        "gt_frames_dir": str(gt_spec[0]) if gt_spec is not None else None,
        "gt_start_index": gt_spec[1] if gt_spec is not None else None,
        "image_preprocessing": {
            "center_crop_aspect_ratio": "4:3",
            "resize": [int(config.dataset.image_size)] * 2,
            "crop_order": "center_crop_then_resize",
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Done: {output}")
    print(f"Paper-style figure: {artifacts['figure']}")
    print(f"Complete rollout: {artifacts['rollout_with_trajectory_video']}")
    if gt_frames is not None:
        print(f"GT comparison: {artifacts['comparison_video']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
