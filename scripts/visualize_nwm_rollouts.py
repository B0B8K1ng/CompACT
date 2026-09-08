#!/usr/bin/env python3
"""Render deterministic side-by-side videos and horizon sheets for NWM rollouts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_BENCHMARK_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark"
)
DEFAULT_PROTOCOL = "go_stanford_unseen_rollout_10_v1"
HORIZONS = (1, 2, 4, 8, 16)
BACKGROUND = (20, 23, 29)
LABEL = (242, 244, 248)
MUTED = (174, 181, 193)


def csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def frame_index(horizon_seconds: int, fps: int) -> int:
    if horizon_seconds <= 0 or fps <= 0:
        raise ValueError("horizon and fps must be positive")
    return horizon_seconds * fps - 1


def frame_paths(directory: Path, expected_count: int) -> list[Path]:
    paths = [directory / f"{index}.png" for index in range(expected_count)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {len(missing)}/{expected_count} frames; "
            f"first missing: {missing[0]}"
        )
    return paths


def load_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        return rgb.copy()


def render_video_frame(
    paths: list[Path], labels: list[str], timestamp: str, panel_size: tuple[int, int]
) -> Image.Image:
    panel_width, panel_height = panel_size
    title_height = 48
    canvas = Image.new("RGB", (panel_width * len(paths), title_height + panel_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18, bold=True)
    time_font = load_font(15)
    for column, (path, panel_label) in enumerate(zip(paths, labels)):
        x = column * panel_width
        canvas.paste(load_panel(path, panel_size), (x, title_height))
        draw.text((x + 10, 7), panel_label, fill=LABEL, font=title_font)
        draw.text((x + 10, 28), timestamp, fill=MUTED, font=time_font)
    return canvas


def render_contact_sheet(
    rows: list[tuple[str, list[Path]]], output: Path, fps: int
) -> None:
    cell = (180, 180)
    row_label_width = 170
    header_height = 42
    canvas = Image.new(
        "RGB",
        (row_label_width + cell[0] * len(HORIZONS), header_height + cell[1] * len(rows)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(17, bold=True)
    label_font = load_font(16, bold=True)
    for column, horizon in enumerate(HORIZONS):
        draw.text(
            (row_label_width + column * cell[0] + 12, 11),
            f"t = {horizon}s",
            fill=LABEL,
            font=header_font,
        )
    for row_index, (row_label, paths) in enumerate(rows):
        y = header_height + row_index * cell[1]
        draw.text((12, y + 78), row_label, fill=LABEL, font=label_font)
        for column, path in enumerate(paths):
            canvas.paste(load_panel(path, cell), (row_label_width + column * cell[0], y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def encode_video(frames: list[Image.Image], output: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode rollout visualizations")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nwm-rollout-", dir=output.parent) as temporary:
        frame_root = Path(temporary)
        for index, frame in enumerate(frames):
            frame.save(frame_root / f"frame_{index:05d}.png", optimize=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_root / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)


def validate_protocol(
    registry: dict, models: list[str], protocol_name: str
) -> tuple[dict, list[int]]:
    try:
        protocol = registry["protocols"][protocol_name]
    except KeyError as exc:
        raise KeyError(f"registry does not define {protocol_name}") from exc
    unknown = sorted(set(models) - set(registry.get("models", {})))
    if unknown:
        raise ValueError(f"models are not registered: {unknown}")
    sample_ids = [int(value) for value in protocol["visualization_sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("visualization sample ids must be unique")
    return protocol, sample_ids


def render(
    registry_path: Path,
    benchmark_root: Path,
    output_root: Path,
    models: list[str],
    dataset: str,
    protocol_name: str = DEFAULT_PROTOCOL,
) -> Path:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    protocol, sample_ids = validate_protocol(registry, models, protocol_name)
    labels = ["Ground truth", *models]
    inputs = [benchmark_root / "gt", *[benchmark_root / "predictions" / model for model in models]]
    artifacts: list[dict[str, object]] = []

    for sample_id in sample_ids:
        sample_dir = output_root / f"id_{sample_id}"
        for mode, fps in zip(protocol["evaluation"], protocol["rollout_fps"]):
            source_mode = protocol.get("source_evaluations", {}).get(mode, mode)
            expected_count = 16 * fps
            source_sequences = [
                frame_paths(
                    root / dataset / source_mode / f"id_{sample_id}", expected_count
                )
                for root in inputs
            ]
            frames = [
                render_video_frame(
                    [sequence[index] for sequence in source_sequences],
                    labels,
                    f"Go Stanford unseen | {mode} | t = {(index + 1) / fps:.2f}s",
                    panel_size=(224, 224),
                )
                for index in range(expected_count)
            ]
            video = sample_dir / f"{mode}.mp4"
            encode_video(frames, video, fps)

            horizon_rows = [
                (
                    label,
                    [sequence[frame_index(horizon, fps)] for horizon in HORIZONS],
                )
                for label, sequence in zip(labels, source_sequences)
            ]
            sheet = sample_dir / f"{mode}_horizons.png"
            render_contact_sheet(horizon_rows, sheet, fps)
            artifacts.append(
                {
                    "sample_id": sample_id,
                    "mode": mode,
                    "source_mode": source_mode,
                    "video": str(video.resolve()),
                    "horizon_sheet": str(sheet.resolve()),
                }
            )

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol": protocol_name,
        "protocol_config": protocol,
        "registry": str(registry_path.resolve()),
        "benchmark_root": str(benchmark_root.resolve()),
        "dataset": dataset,
        "models": {
            model: {
                key: registry["models"][model].get(key)
                for key in ("architecture", "checkpoint", "checkpoint_id", "checkpoint_step", "sha256")
            }
            for model in models
        },
        "panel_order": labels,
        "artifacts": artifacts,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    temporary = manifest_path.with_suffix(f".tmp.{manifest_path.suffix.lstrip('.')}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", type=csv, required=True)
    parser.add_argument("--dataset", default="go_stanford")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render(
        args.registry,
        args.benchmark_root,
        args.output_root,
        args.models,
        args.dataset,
        args.protocol,
    )
    print(manifest)


if __name__ == "__main__":
    main()
