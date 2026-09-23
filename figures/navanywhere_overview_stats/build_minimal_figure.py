#!/usr/bin/env python3
"""Build a restrained, integrated NavAnywhere overview.

The figure keeps the original overview's visual hierarchy. Dataset composition
is a single quiet dumbbell plot along the left edge, while scene shares are
written directly beside their image rows. All 47 overview-v3 images remain.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from pathlib import Path

from build_integrated_figure import (
    COLORS,
    DATASET_CSV,
    DISPLAY_DATASET_NAMES,
    OVERVIEW_SVG,
    SCENE_COLORS,
    SCENE_CSV,
    EmbeddedImage,
    compass_logo,
    extract_images,
    image_markup,
    read_csv,
    sha256,
    validate_data,
)


DEFAULT_OUTPUT_DIR = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/visualizations/2026/09/22/"
    "01a0c742-f9c0-7ab1-9bd0-dca2be5385fa/"
    "navanywhere_overview_minimal_20260922"
)

BASE_NAME = "navanywhere_overview_minimal_20260922"
CANVAS_W = 2400
CANVAS_H = 1288
PAGE_WIDTH_IN = 7.0
PAGE_HEIGHT_IN = PAGE_WIDTH_IN * CANVAS_H / CANVAS_W


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def header_markup() -> str:
    return f"""
  {compass_logo()}
  <text x="111" y="78" class="title">NavAnywhere</text>
  <text x="620" y="29" class="section">CORPUS SCALE</text>
  <text x="620" y="84" class="real-number">97.3%</text>
  <text x="820" y="77" class="real-label">real-world video</text>
  <line x1="1039" y1="25" x2="1039" y2="99" class="divider"/>
  <text x="1068" y="65" class="stat-number">70,756</text>
  <text x="1070" y="94" class="stat-label">SEQUENCES</text>
  <text x="1340" y="65" class="stat-number">1,189 h</text>
  <text x="1342" y="94" class="stat-label">VIDEO HOURS</text>
  <text x="1601" y="65" class="stat-number">17.5M</text>
  <text x="1603" y="94" class="stat-label">SAMPLED FRAMES</text>
  <text x="1907" y="65" class="stat-number">17+</text>
  <text x="1909" y="94" class="stat-label">COUNTRIES</text>
  <line x1="41" y1="113" x2="2359" y2="113" class="divider"/>
"""


def coverage_markup() -> str:
    return """
  <text x="41" y="139" class="section">EMBODIMENT</text>
  <g transform="translate(0 -72)">
    <g fill="none" stroke="#37847D" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="220" cy="227" r="8"/>
      <path d="M220,237 L220,262 M205,250 L220,242 L235,250 M220,262 L207,282 M220,262 L233,282"/>
      <path d="M413,244 H463 V266 H413 Z"/>
      <path d="M430,244 V227 H451 V244 M441,227 V218"/>
      <circle cx="423" cy="272" r="8"/><circle cx="455" cy="272" r="8"/>
      <rect x="650" y="244" width="20" height="16" rx="4"/>
      <path d="M650,249 L630,234 M670,249 L690,234 M650,255 L630,270 M670,255 L690,270"/>
      <ellipse cx="624" cy="232" rx="13" ry="5"/><ellipse cx="696" cy="232" rx="13" ry="5"/>
      <ellipse cx="624" cy="272" rx="13" ry="5"/><ellipse cx="696" cy="272" rx="13" ry="5"/>
    </g>
    <text x="220" y="309" text-anchor="middle" class="icon-label">Human</text>
    <text x="440" y="309" text-anchor="middle" class="icon-label">Ground robot</text>
    <text x="660" y="309" text-anchor="middle" class="icon-label">Drone</text>
  </g>
  <line x1="811" y1="126" x2="811" y2="246" class="divider"/>
  <text x="858" y="139" class="section">CAPTURE CONDITIONS</text>
  <g transform="translate(0 -72)">
    <g fill="none" stroke="#607F9B" stroke-width="3.3" stroke-linecap="round">
      <circle cx="1040" cy="252" r="18"/>
      <path d="M1040,222 L1040,212 M1040,282 L1040,292 M1010,252 L1000,252 M1070,252 L1080,252 M1019,231 L1011,223 M1061,231 L1069,223 M1019,273 L1011,281 M1061,273 L1069,281"/>
    </g>
    <path d="M1244,224 A31,31 0 1,0 1255,274 A25,25 0 1,1 1244,224 Z" fill="none" stroke="#607F9B" stroke-width="3.4"/>
    <g fill="none" stroke="#607F9B" stroke-width="3.3" stroke-linecap="round" stroke-linejoin="round">
      <path d="M1393,256 C1385,234 1412,225 1422,239 C1434,227 1457,242 1448,259 H1395"/>
      <path d="M1402,270 l-5,12 M1420,270 l-5,12 M1438,270 l-5,12"/>
    </g>
    <g fill="none" stroke="#607F9B" stroke-width="3.1" stroke-linecap="round">
      <path d="M1610,252 V218 M1610,231 L1603,224 M1610,231 L1617,224 M1610,252 L1639,235 M1628,242 L1631,232 M1628,242 L1638,244 M1610,252 L1639,269 M1628,263 L1638,260 M1628,263 L1631,272 M1610,252 V286 M1610,273 L1617,280 M1610,273 L1603,280 M1610,252 L1581,269 M1592,263 L1589,272 M1592,263 L1582,260 M1610,252 L1581,235 M1592,242 L1582,244 M1592,242 L1589,232"/>
    </g>
    <g fill="none" stroke-linecap="round" stroke-linejoin="round">
      <g stroke="#5E8D70" stroke-width="2.2"><circle cx="1792" cy="234" r="2.3" fill="#5E8D70"/><circle cx="1792" cy="228" r="3.7"/><circle cx="1798" cy="234" r="3.7"/><circle cx="1792" cy="240" r="3.7"/><circle cx="1786" cy="234" r="3.7"/></g>
      <g stroke="#B78B3D" stroke-width="2.1"><circle cx="1828" cy="234" r="5.3"/><path d="M1836,234 H1840 M1834,240 L1836,242 M1828,242 V246 M1822,240 L1820,242 M1820,234 H1816 M1822,228 L1820,226 M1828,226 V222 M1834,228 L1836,226"/></g>
      <g stroke="#9A7156" stroke-width="2.2"><path d="M1784,274 C1784,264 1794,260 1801,263 C1800,272 1794,279 1784,274 Z"/><path d="M1785,274 L1800,263 M1785,274 L1781,279"/></g>
      <g stroke="#607F9B" stroke-width="2.2"><path d="M1817,270 H1839 M1823,260 L1834,280 M1834,260 L1823,280"/><circle cx="1828" cy="270" r="1.7" fill="#607F9B" stroke="none"/></g>
    </g>
    <text x="1040" y="309" text-anchor="middle" class="icon-label">Day</text>
    <text x="1230" y="309" text-anchor="middle" class="icon-label">Night</text>
    <text x="1420" y="309" text-anchor="middle" class="icon-label">Rain</text>
    <text x="1610" y="309" text-anchor="middle" class="icon-label">Snow</text>
    <text x="1810" y="309" text-anchor="middle" class="icon-label">Seasons</text>
  </g>
  <line x1="41" y1="253" x2="2359" y2="253" class="divider"/>
"""


def log_x(value: float, x0: float = 208.0, x1: float = 425.0) -> float:
    lo = math.log10(0.01)
    hi = math.log10(100.0)
    return x0 + (math.log10(value) - lo) / (hi - lo) * (x1 - x0)


def compact_share(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def source_plot_markup(dataset_rows: list[dict[str, str]]) -> str:
    rows = sorted(
        dataset_rows,
        key=lambda row: float(row["video_hours_share_pct"]),
        reverse=True,
    )
    row_y0 = 321.0
    row_gap = 42.0
    plot_top = 302.0
    axis_y = 927.0
    parts = [
        '<text x="41" y="282" class="body-title">SOURCE MIX</text>',
        f'<circle cx="174" cy="277" r="4.5" fill="{COLORS["teal"]}"/>',
        '<text x="185" y="282" class="key-text">H</text>',
        f'<circle cx="217" cy="277" r="4.5" fill="{COLORS["coral"]}"/>',
        '<text x="228" y="282" class="key-text">F</text>',
        '<text x="440" y="282" class="key-text" text-anchor="end">share (%) · log</text>',
    ]
    for tick, label in zip([0.01, 0.1, 1, 10, 100], [".01", ".1", "1", "10", "100"]):
        x = log_x(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_top:.1f}" x2="{x:.2f}" y2="{axis_y:.1f}" class="source-grid"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="949" class="source-tick" text-anchor="middle">{label}</text>'
        )

    for index, row in enumerate(rows):
        y = row_y0 + index * row_gap
        hours = float(row["video_hours_share_pct"])
        frames = float(row["sampled_frames_share_pct"])
        xh = log_x(hours)
        xf = log_x(frames)
        name = DISPLAY_DATASET_NAMES.get(row["dataset"], row["dataset"])
        parts.extend(
            [
                f'<text x="190" y="{y + 5:.1f}" class="source-label" text-anchor="end">{esc(name)}</text>',
                f'<line x1="{xh:.2f}" y1="{y - 4:.1f}" x2="{xf:.2f}" y2="{y + 4:.1f}" class="pair-line"/>',
                f'<circle cx="{xh:.2f}" cy="{y - 4:.1f}" r="5.2" fill="{COLORS["teal"]}" stroke="#FFFFFF" stroke-width="1.5"/>',
                f'<circle cx="{xf:.2f}" cy="{y + 4:.1f}" r="5.2" fill="{COLORS["coral"]}" stroke="#FFFFFF" stroke-width="1.5"/>',
            ]
        )
    return "".join(parts)


def gallery_markup(
    images: list[EmbeddedImage], scene_rows: list[dict[str, str]]
) -> tuple[list[str], str]:
    clips: list[str] = []
    parts = [
        '<text x="465" y="282" class="body-title">SCENE COVERAGE · 35 REPRESENTATIVE FRAMES</text>',
        '<text x="2359" y="282" class="key-text" text-anchor="end">H · video hours   F · sampled frames</text>',
    ]
    x_positions = [465.0 + 235.0 * index for index in range(7)]
    row_bases = [305.0 + 134.0 * index for index in range(5)]
    y_offsets = [0.0, 4.0, 1.0, 5.0, 1.0, 4.0, 0.0]

    for row_index, (scene, base_y) in enumerate(zip(scene_rows, row_bases)):
        center_y = base_y + 58.0
        color = SCENE_COLORS[scene["category"]]
        parts.append(
            f'<line x1="451" y1="{center_y:.1f}" x2="2121" y2="{center_y:.1f}" stroke="{color}" stroke-width="3.5" stroke-linecap="round" opacity="0.72"/>'
        )
        parts.append(
            f'<circle cx="2121" cy="{center_y:.1f}" r="17" fill="{COLORS["paper"]}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="2121" y="{center_y - 25:.1f}" class="waypoint-index" text-anchor="middle">{row_index + 1:02d}</text>'
        )

        category_lines = {
            "Residential": ["Residential"],
            "Public & Commercial": ["Public & Commercial"],
            "Urban & Transport": ["Urban & Transport"],
            "Parks & Gardens": ["Parks & Gardens"],
            "Natural & Off-road": ["Natural & Off-road"],
        }[scene["category"]]
        parts.append(
            f'<text x="2154" y="{center_y - 11:.1f}" class="scene-name">{esc(category_lines[0])}</text>'
        )
        hours = float(scene["video_hours_share_pct"])
        frames = float(scene["sampled_frames_share_pct"])
        parts.append(
            f'<text x="2154" y="{center_y + 20:.1f}" class="scene-share">'
            f'<tspan fill="{COLORS["teal"]}" font-weight="700">H</tspan>'
            f'<tspan> {hours:.1f}%   </tspan>'
            f'<tspan fill="{COLORS["coral"]}" font-weight="700">F</tspan>'
            f'<tspan> {frames:.1f}%</tspan></text>'
        )

        for col_index, x in enumerate(x_positions):
            idx = row_index * 7 + col_index
            y = base_y + y_offsets[col_index]
            clip, body = image_markup(
                images[idx],
                f"minimal-main-{row_index}-{col_index}",
                x,
                y,
                226.0,
                116.0,
                COLORS["line"],
            )
            clips.append(clip)
            parts.append(body)
    return clips, "".join(parts)


def ood_markup(images: list[EmbeddedImage]) -> tuple[list[str], str]:
    clips: list[str] = []
    parts = [
        '<path d="M49,1008 H2355" fill="none" stroke="#718FA5" stroke-width="3" stroke-dasharray="10 10" stroke-linecap="round" opacity="0.85"/>',
        f'<rect x="35" y="976" width="640" height="43" fill="{COLORS["paper"]}"/>',
        '<text x="42" y="1004" class="ood-heading">OUT-OF-DOMAIN DATASETS FOR EVALUATION</text>',
    ]
    group_x = [42.0, 626.0, 1210.0, 1794.0]
    group_names = [
        ("UZH-FPV", ""),
        ("TUM RGB-D SLAM", ""),
        ("Office-Go2", " · in-house captured"),
        ("Planetary Rover", " · in-house web-curated"),
    ]
    for group_index, (base_x, (name, tag)) in enumerate(zip(group_x, group_names)):
        center = base_x + 272.5
        parts.append(
            f'<circle cx="{center:.1f}" cy="1008" r="7" fill="{COLORS["paper"]}" stroke="{COLORS["blue"]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{base_x:.1f}" y="1060" class="ood-name"><tspan>{esc(name)}</tspan>'
            + (f'<tspan class="ood-tag">{esc(tag)}</tspan>' if tag else "")
            + "</text>"
        )
        for col_index in range(3):
            x = base_x + 184.0 * col_index
            idx = group_index * 3 + col_index
            clip, body = image_markup(
                images[idx],
                f"minimal-ood-{group_index}-{col_index}",
                x,
                1080.0,
                177.0,
                164.0,
                "#C8D7E0",
            )
            clips.append(clip)
            parts.append(body)
    return clips, "".join(parts)


def style_markup() -> str:
    return f"""
  <style>
    text {{ font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; }}
    .title {{ font-size: 61px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: -1px; }}
    .section {{ font-size: 18px; font-weight: 700; fill: {COLORS['muted']}; letter-spacing: 2.4px; }}
    .real-number {{ font-size: 56px; font-weight: 700; fill: {COLORS['teal']}; letter-spacing: -1.5px; }}
    .real-label {{ font-size: 22px; font-weight: 700; fill: {COLORS['ink']}; }}
    .stat-number {{ font-size: 33px; font-weight: 700; fill: {COLORS['ink']}; }}
    .stat-label {{ font-size: 14px; font-weight: 700; fill: {COLORS['muted']}; letter-spacing: 1.25px; }}
    .divider {{ stroke: {COLORS['line']}; stroke-width: 1.5; }}
    .icon-label {{ font-size: 19px; font-weight: 600; fill: {COLORS['ink']}; }}
    .body-title {{ font-size: 18px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: 1.45px; }}
    .key-text {{ font-size: 14px; font-weight: 600; fill: {COLORS['muted']}; }}
    .source-label {{ font-size: 16px; font-weight: 600; fill: {COLORS['ink']}; }}
    .source-values {{ font-size: 13.5px; font-weight: 600; fill: {COLORS['muted']}; }}
    .source-grid {{ stroke: {COLORS['line_soft']}; stroke-width: 1.1; }}
    .source-tick {{ font-size: 12px; font-weight: 500; fill: {COLORS['faint']}; }}
    .pair-line {{ stroke: #AAB8B6; stroke-width: 2; }}
    .scene-name {{ font-size: 19px; font-weight: 700; fill: {COLORS['ink']}; }}
    .scene-share {{ font-size: 16px; font-weight: 600; fill: {COLORS['muted']}; }}
    .waypoint-index {{ font-size: 13px; font-weight: 700; fill: {COLORS['faint']}; letter-spacing: 1px; }}
    .ood-heading {{ font-size: 20px; font-weight: 700; fill: {COLORS['blue']}; letter-spacing: 2.1px; }}
    .ood-name {{ font-size: 22px; font-weight: 700; fill: {COLORS['ink']}; }}
    .ood-tag {{ font-size: 15px; font-weight: 600; fill: {COLORS['muted']}; }}
  </style>
"""


def build_svg(
    images: list[EmbeddedImage],
    dataset_rows: list[dict[str, str]],
    scene_rows: list[dict[str, str]],
) -> str:
    main_clips, gallery = gallery_markup(images[:35], scene_rows)
    ood_clips, ood = ood_markup(images[35:])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{PAGE_WIDTH_IN:.6f}in" height="{PAGE_HEIGHT_IN:.6f}in"
     viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-labelledby="title desc">
  <title id="title">Minimal integrated NavAnywhere overview</title>
  <desc id="desc">A quiet source-composition dumbbell plot beside 35 scene-aligned frames, followed by 12 out-of-domain evaluation frames.</desc>
  <defs>{''.join(main_clips + ood_clips)}</defs>
  {style_markup()}
  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{COLORS['paper']}"/>
  <g fill="none" stroke="{COLORS['line_soft']}" stroke-width="1.4" opacity="0.62">
    <path d="M440,370 C680,308 850,336 1030,385 S1400,442 1590,378 S1980,319 2400,361"/>
    <path d="M440,650 C680,590 870,606 1060,662 S1440,720 1640,652 S2020,589 2400,635"/>
    <path d="M440,910 C700,854 890,872 1090,921 S1470,972 1690,908 S2060,851 2400,895"/>
  </g>
  {header_markup()}
  {coverage_markup()}
  {source_plot_markup(dataset_rows)}
  {gallery}
  {ood}
</svg>
"""


def build_html(svg_name: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {PAGE_WIDTH_IN:.6f}in {PAGE_HEIGHT_IN:.6f}in; margin: 0; }}
html,body {{ margin:0; padding:0; width:{PAGE_WIDTH_IN:.6f}in; height:{PAGE_HEIGHT_IN:.6f}in; overflow:hidden; background:{COLORS['paper']}; }}
img {{ display:block; width:{PAGE_WIDTH_IN:.6f}in; height:{PAGE_HEIGHT_IN:.6f}in; }}
</style></head><body><img src="{esc(svg_name)}" alt="Minimal integrated NavAnywhere overview"></body></html>
"""


def build_caption() -> str:
    return rf"""\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/{BASE_NAME}_macos.pdf}}
  \caption{{\textbf{{NavAnywhere overview and corpus composition.}}
  The left dumbbell plot compares each formal source's share of video hours
  (H) and sampled RGB frames (F) on a logarithmic scale. Thirty-five
  representative frames are arranged by scene category, with the corresponding
  H/F shares reported beside each row. The bottom strip contains twelve frames
  from four out-of-domain evaluation datasets.}}
  \label{{fig:navanywhere-overview-minimal}}
\end{{figure*}}
"""


def write_provenance(output_dir: Path, images: list[EmbeddedImage]) -> None:
    manifest = {
        "artifact": BASE_NAME,
        "design": "minimal left-rail source plot with row-aligned scene shares",
        "canvas": {
            "viewbox": [0, 0, CANVAS_W, CANVAS_H],
            "physical_size_inches": [PAGE_WIDTH_IN, PAGE_HEIGHT_IN],
        },
        "image_count": {"total": 47, "in_domain": 35, "out_of_domain": 12},
        "embedded_image_href_sha256": [
            hashlib.sha256(image.href.encode("ascii")).hexdigest() for image in images
        ],
        "inputs": {
            "overview_v3_svg": {"path": str(OVERVIEW_SVG), "sha256": sha256(OVERVIEW_SVG)},
            "dataset_composition_csv": {"path": str(DATASET_CSV), "sha256": sha256(DATASET_CSV)},
            "scene_composition_csv": {"path": str(SCENE_CSV), "sha256": sha256(SCENE_CSV)},
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    images = extract_images(OVERVIEW_SVG)
    dataset_rows = read_csv(DATASET_CSV)
    scene_rows = read_csv(SCENE_CSV)
    validate_data(dataset_rows, scene_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{BASE_NAME}.svg"
    html_path = output_dir / f"{BASE_NAME}_print.html"
    svg_path.write_text(build_svg(images, dataset_rows, scene_rows), encoding="utf-8")
    html_path.write_text(build_html(svg_path.name), encoding="utf-8")
    (output_dir / "caption.tex").write_text(build_caption(), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Minimal integrated NavAnywhere overview\n\n"
        "The source statistics form one restrained left-side plot; scene shares "
        "are attached directly to their image rows. All 47 overview-v3 image "
        "payloads are retained byte-for-byte.\n",
        encoding="utf-8",
    )
    shutil.copy2(DATASET_CSV, output_dir / "dataset_composition.csv")
    shutil.copy2(SCENE_CSV, output_dir / "scene_composition.csv")
    write_provenance(output_dir, images)
    print(f"Wrote {svg_path}")
    print(f"Embedded images: {len(images)}")
    print(f"Page: {PAGE_WIDTH_IN:.4f} x {PAGE_HEIGHT_IN:.4f} in")


if __name__ == "__main__":
    main()
