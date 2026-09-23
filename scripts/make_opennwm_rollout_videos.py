#!/usr/bin/env python3
"""Make synchronized GT/NWM/RAE-NWM/OpenNWM videos for every saved rollout case."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/"
    "opennwm_rollout_showcase_20260923"
)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
IMAGE_SIZE = 224
TITLE_HEIGHT = 36
LABEL_HEIGHT = 28
WIDTH = 2 * IMAGE_SIZE
HEIGHT = TITLE_HEIGHT + 2 * (LABEL_HEIGHT + IMAGE_SIZE)
PANELS = (
    ("GT", "GT", "#23a6a0"),
    ("NWM", "nwm-release", "#5c8edb"),
    ("RAE-NWM", "rae-nwm", "#a279d5"),
    ("OpenNWM", "opennwm-finalLAM-100k", "#eb9b45"),
)


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,codec_name",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def validate_video(path: Path, count: int, fps: int) -> None:
    stream = probe(path)
    expected = (WIDTH, HEIGHT, str(count), f"{fps}/1", "h264")
    actual = (
        stream["width"], stream["height"], stream.get("nb_frames"),
        stream["avg_frame_rate"], stream["codec_name"],
    )
    if actual != expected:
        raise RuntimeError(f"Invalid video {path}: expected {expected}, got {actual}")


def draw_frame(case: dict, case_dir: Path, step: int, title_font: ImageFont.FreeTypeFont,
               label_font: ImageFont.FreeTypeFont, initial_image: Path | None = None) -> bytes:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#111820")
    draw = ImageDraw.Draw(image)
    count = case["seconds"] * case["fps"]
    heading = (f"{case['group']}  {case['dataset']}/id_{case['sample_id']}"
               f"   t={(step + 1) / case['fps']:.2f}s   {step + 1}/{count}")
    draw.text((8, 8), heading, font=title_font, fill="white")
    for panel, (label, folder, color) in enumerate(PANELS):
        row, col = divmod(panel, 2)
        x = col * IMAGE_SIZE
        y = TITLE_HEIGHT + row * (LABEL_HEIGHT + IMAGE_SIZE)
        draw.rectangle((x, y, x + 5, y + LABEL_HEIGHT - 1), fill=color)
        draw.text((x + 11, y + 4), label, font=label_font, fill="white")
        frame = (initial_image if step < 0 else
                 case_dir / "frames" / folder / f"{step:03d}.png")
        if frame is None:
            raise ValueError("Initial frame was requested without an image")
        with Image.open(frame) as source:
            if source.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Unexpected image size: {frame}: {source.size}")
            image.paste(source.convert("RGB"), (x, y + LABEL_HEIGHT))
    return image.tobytes()


def make_video(case: dict, root: Path, force: bool) -> dict:
    case_dir = root / "examples" / case["dataset"] / f"id_{case['sample_id']}"
    count = case["seconds"] * case["fps"]
    include_initial = (root / "protocol.json").is_file()
    initial_image = (root / "initial" / case["dataset"] /
                     f"id_{case['sample_id']}.png") if include_initial else None
    if initial_image is not None and not initial_image.is_file():
        raise FileNotFoundError(initial_image)
    video_count = count + int(include_initial)
    for _, folder, _ in PANELS:
        frames = sorted((case_dir / "frames" / folder).glob("*.png"))
        expected = [f"{step:03d}.png" for step in range(count)]
        if [path.name for path in frames] != expected:
            raise RuntimeError(f"Incomplete frames: {case_dir / 'frames' / folder}")
    output = case_dir / "comparison.mp4"
    if output.is_file() and not force:
        validate_video(output, video_count, case["fps"])
        print(f"SKIP verified {output}", flush=True)
    else:
        temporary = case_dir / "comparison.tmp.mp4"
        temporary.unlink(missing_ok=True)
        command = [
            "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(case["fps"]),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-threads", "1", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ]
        print(f"RUN {' '.join(command)}", flush=True)
        title_font = ImageFont.truetype(str(FONT_PATH), 16)
        label_font = ImageFont.truetype(str(FONT_PATH), 16)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"PID {process.pid} {case['dataset']}/id_{case['sample_id']}", flush=True)
        try:
            assert process.stdin is not None
            for step in range(-int(include_initial), count):
                process.stdin.write(draw_frame(
                    case, case_dir, step, title_font, label_font, initial_image
                ))
            process.stdin.close()
            assert process.stderr is not None
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            status = process.wait()
            print(f"EXIT {status} {case['dataset']}/id_{case['sample_id']}", flush=True)
            if status:
                raise RuntimeError(f"ffmpeg failed for {output}: {stderr}")
            validate_video(temporary, video_count, case["fps"])
            os.replace(temporary, output)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            temporary.unlink(missing_ok=True)
            raise
    return {
        "group": case["group"], "dataset": case["dataset"],
        "sample_id": case["sample_id"], "seconds": case["seconds"],
        "fps": case["fps"], "frame_count": video_count,
        "initial_frame_included": include_initial,
        "width": WIDTH, "height": HEIGHT,
        "video": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cases = json.loads((root / "summary.json").read_text())
    if not cases:
        raise RuntimeError("No cases in summary.json")
    index = [make_video(case, root, args.force) for case in cases]
    (root / "video_index.json").write_text(json.dumps({
        "layout": "2x2: GT, NWM, RAE-NWM, OpenNWM",
        "encoder": "ffmpeg libx264 CRF 18 preset medium, yuv420p, no audio",
        "videos": index,
    }, indent=2) + "\n")
    print(f"DONE {len(index)} videos", flush=True)


if __name__ == "__main__":
    main()
