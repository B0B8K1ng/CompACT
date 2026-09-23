#!/usr/bin/env python3
"""Build an integrated NavAnywhere overview and corpus composition figure.

The source composition is encoded as a compact paired-bar fingerprint in the
header. Scene statistics are aligned with the five corresponding image rows.
All 47 embedded JPEG payloads from overview v3 are retained byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from lxml import etree


OVERVIEW_SVG = Path(
    "/file_system/nas/algorithm/dujun.nie/navanywhere_paper_figure_20260920/"
    "output/navanywhere_overview_v3.svg"
)
DATASET_CSV = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/stats/navanywhere_iclr_20260914/"
    "tables/navanywhere_dataset_composition.csv"
)
SCENE_CSV = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/stats/navanywhere_iclr_20260914/"
    "tables/navanywhere_scene_composition.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/visualizations/2026/09/22/"
    "01a0c742-f9c0-7ab1-9bd0-dca2be5385fa/"
    "navanywhere_overview_integrated_20260922"
)

BASE_NAME = "navanywhere_overview_integrated_20260922"
CANVAS_W = 2400
CANVAS_H = 1530
PAGE_WIDTH_IN = 7.0
PAGE_HEIGHT_IN = PAGE_WIDTH_IN * CANVAS_H / CANVAS_W

COLORS = {
    "paper": "#FCFCFA",
    "ink": "#17343D",
    "muted": "#607379",
    "faint": "#8B9A9E",
    "line": "#D6E0DF",
    "line_soft": "#E9EFEE",
    "teal": "#37847D",
    "teal_soft": "#DCEBE8",
    "coral": "#C97962",
    "coral_soft": "#F1E2DC",
    "blue": "#718FA5",
    "residential": "#88749A",
    "public": "#B0785B",
    "urban": "#607F9B",
    "parks": "#60876D",
    "natural": "#89805E",
}

SCENE_COLORS = {
    "Residential": COLORS["residential"],
    "Public & Commercial": COLORS["public"],
    "Urban & Transport": COLORS["urban"],
    "Parks & Gardens": COLORS["parks"],
    "Natural & Off-road": COLORS["natural"],
}

DISPLAY_DATASET_NAMES = {
    "CASIA-Nav": "In-house Collected",
    "The Great Outdoors": "Great Outdoors",
}


@dataclass(frozen=True)
class EmbeddedImage:
    href: str
    aria_label: str


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def extract_images(path: Path) -> list[EmbeddedImage]:
    parser = etree.XMLParser(resolve_entities=False, huge_tree=True)
    root = etree.parse(str(path), parser).getroot()
    images: list[EmbeddedImage] = []
    for image in root.xpath("//*[local-name()='image']"):
        href = image.get("href") or image.get(
            "{http://www.w3.org/1999/xlink}href"
        )
        if not href or not href.startswith("data:image/jpeg;base64,"):
            raise RuntimeError("Expected a self-contained embedded JPEG")
        parent = image.getparent()
        label = parent.get("aria-label") if parent is not None else None
        images.append(EmbeddedImage(href=href, aria_label=label or "dataset frame"))
    if len(images) != 47:
        raise RuntimeError(f"Expected 47 embedded images, found {len(images)}")
    return images


def validate_data(
    dataset_rows: list[dict[str, str]], scene_rows: list[dict[str, str]]
) -> None:
    if len(dataset_rows) != 15:
        raise RuntimeError(f"Expected 15 formal sources, found {len(dataset_rows)}")
    if len(scene_rows) != 5:
        raise RuntimeError(f"Expected five scene categories, found {len(scene_rows)}")
    if "CASIA-Nav" not in {row["dataset"] for row in dataset_rows}:
        raise RuntimeError("Expected CASIA-Nav before display anonymization")
    if [row["category"] for row in scene_rows] != list(SCENE_COLORS):
        raise RuntimeError("Unexpected scene category order")
    for field in ("video_hours_share_pct", "sampled_frames_share_pct"):
        dataset_sum = sum(float(row[field]) for row in dataset_rows)
        scene_sum = sum(float(row[field]) for row in scene_rows)
        if abs(dataset_sum - 100.0) > 0.02:
            raise RuntimeError(f"Dataset {field} sums to {dataset_sum}")
        if abs(scene_sum - 100.0) > 0.02:
            raise RuntimeError(f"Scene {field} sums to {scene_sum}")


def fmt_share(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}%"
    if value >= 0.1:
        return f"{value:.2f}%"
    return f"{value:.3f}%"


def log_bar_width(value: float, max_width: float = 154.0) -> float:
    lo = math.log10(0.02)
    hi = math.log10(100.0)
    t = (math.log10(value) - lo) / (hi - lo)
    return 12.0 + max(0.0, min(1.0, t)) * (max_width - 12.0)


def compass_logo() -> str:
    return """
    <g transform="translate(65 63) scale(1.12)">
      <circle cx="0" cy="0" r="21" fill="none" stroke="#17343D" stroke-width="1.6"/>
      <circle cx="0" cy="0" r="14" fill="none" stroke="#17343D" stroke-width="0.8" opacity="0.45"/>
      <path d="M0,-27 L5,-5 L0,0 L-5,-5 Z" fill="#37847D"/>
      <path d="M0,27 L-5,5 L0,0 L5,5 Z" fill="#17343D"/>
      <path d="M27,0 L5,5 L0,0 L5,-5 Z" fill="#17343D" opacity="0.65"/>
      <path d="M-27,0 L-5,-5 L0,0 L-5,5 Z" fill="#17343D" opacity="0.65"/>
      <circle cx="0" cy="0" r="3.2" fill="#C97962"/>
    </g>
    """


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


def source_fingerprint_markup(dataset_rows: list[dict[str, str]]) -> str:
    rows = sorted(
        dataset_rows,
        key=lambda row: float(row["video_hours_share_pct"]),
        reverse=True,
    )
    x0 = 41.0
    cell_w = 2318.0 / 8.0
    row_tops = [147.0, 239.0]
    parts = [
        '<text x="41" y="137" class="band-title">SOURCE COMPOSITION · 15 FORMAL DATASETS</text>',
        '<text x="2359" y="137" class="band-note" text-anchor="end">paired log share of total corpus</text>',
    ]
    for index, row in enumerate(rows):
        row_index = 0 if index < 8 else 1
        col_index = index if index < 8 else index - 8
        x = x0 + col_index * cell_w
        y = row_tops[row_index]
        name = DISPLAY_DATASET_NAMES.get(row["dataset"], row["dataset"])
        hours = float(row["video_hours_share_pct"])
        frames = float(row["sampled_frames_share_pct"])
        if col_index:
            parts.append(
                f'<line x1="{x - 12:.2f}" y1="{y + 4:.2f}" x2="{x - 12:.2f}" '
                f'y2="{y + 78:.2f}" class="source-separator"/>'
            )
        parts.extend(
            [
                f'<text x="{x + 4:.2f}" y="{y + 18:.2f}" class="source-name">{esc(name)}</text>',
                f'<text x="{x + cell_w - 20:.2f}" y="{y + 18:.2f}" class="source-rank" text-anchor="end">{index + 1:02d}</text>',
                f'<text x="{x + 4:.2f}" y="{y + 45:.2f}" class="metric-letter metric-hours">H</text>',
                f'<line x1="{x + 28:.2f}" y1="{y + 40:.2f}" x2="{x + 182:.2f}" y2="{y + 40:.2f}" class="source-track"/>',
                f'<line x1="{x + 28:.2f}" y1="{y + 40:.2f}" x2="{x + 28 + log_bar_width(hours):.2f}" y2="{y + 40:.2f}" class="source-hours"/>',
                f'<circle cx="{x + 28 + log_bar_width(hours):.2f}" cy="{y + 40:.2f}" r="4.2" class="source-hours-dot"/>',
                f'<text x="{x + 195:.2f}" y="{y + 45:.2f}" class="source-value">{fmt_share(hours)}</text>',
                f'<text x="{x + 4:.2f}" y="{y + 72:.2f}" class="metric-letter metric-frames">F</text>',
                f'<line x1="{x + 28:.2f}" y1="{y + 67:.2f}" x2="{x + 182:.2f}" y2="{y + 67:.2f}" class="source-track"/>',
                f'<line x1="{x + 28:.2f}" y1="{y + 67:.2f}" x2="{x + 28 + log_bar_width(frames):.2f}" y2="{y + 67:.2f}" class="source-frames"/>',
                f'<circle cx="{x + 28 + log_bar_width(frames):.2f}" cy="{y + 67:.2f}" r="4.2" class="source-frames-dot"/>',
                f'<text x="{x + 195:.2f}" y="{y + 72:.2f}" class="source-value">{fmt_share(frames)}</text>',
            ]
        )

    legend_x = x0 + 7 * cell_w
    legend_y = row_tops[1]
    parts.extend(
        [
            f'<line x1="{legend_x - 12:.2f}" y1="{legend_y + 4:.2f}" x2="{legend_x - 12:.2f}" y2="{legend_y + 78:.2f}" class="source-separator"/>',
            f'<text x="{legend_x + 4:.2f}" y="{legend_y + 18:.2f}" class="source-key-title">READING THE GLYPHS</text>',
            f'<circle cx="{legend_x + 10:.2f}" cy="{legend_y + 40:.2f}" r="5" fill="{COLORS["teal"]}"/>',
            f'<text x="{legend_x + 24:.2f}" y="{legend_y + 45:.2f}" class="source-key">H · video hours</text>',
            f'<circle cx="{legend_x + 10:.2f}" cy="{legend_y + 67:.2f}" r="5" fill="{COLORS["coral"]}"/>',
            f'<text x="{legend_x + 24:.2f}" y="{legend_y + 72:.2f}" class="source-key">F · sampled frames</text>',
            '<line x1="41" y1="334" x2="2359" y2="334" class="divider"/>',
        ]
    )
    return "".join(parts)


def coverage_markup() -> str:
    # The icon vocabulary follows overview v3 but is compacted into one band.
    return """
  <text x="41" y="361" class="section">EMBODIMENT</text>
  <g transform="translate(0 148)">
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

  <line x1="811" y1="350" x2="811" y2="470" class="divider"/>
  <text x="858" y="361" class="section">CAPTURE CONDITIONS</text>
  <g transform="translate(0 148)">
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
  <line x1="41" y1="474" x2="2359" y2="474" class="divider"/>
"""


def image_markup(
    image: EmbeddedImage,
    clip_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    border: str,
) -> tuple[str, str]:
    clip = (
        f'<clipPath id="{clip_id}"><rect x="{x:.2f}" y="{y:.2f}" '
        f'width="{width:.2f}" height="{height:.2f}" rx="7"/></clipPath>'
    )
    body = (
        f'<g aria-label="{esc(image.aria_label)}">'
        f'<image x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'href="{image.href}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})"/>'
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="7" '
        f'fill="none" stroke="#FFFFFF" stroke-width="5"/>'
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="7" '
        f'fill="none" stroke="{border}" stroke-width="1.3"/>'
        "</g>"
    )
    return clip, body


def gallery_markup(
    main_images: list[EmbeddedImage], scene_rows: list[dict[str, str]]
) -> tuple[list[str], str]:
    if len(main_images) != 35:
        raise RuntimeError("Expected 35 in-domain images")
    parts = [
        '<text x="41" y="505" class="band-title">SCENE COMPOSITION · REPRESENTATIVE NAVANYWHERE FRAMES</text>',
        '<text x="2359" y="505" class="band-note" text-anchor="end">H · video hours   F · sampled frames</text>',
    ]
    clips: list[str] = []
    x_positions = [350.0 + 252.0 * index for index in range(7)]
    row_bases = [526.0 + 136.0 * index for index in range(5)]
    y_offsets = [0.0, 7.0, 2.0, 8.0, 1.0, 6.0, 0.0]
    centers = [base + 62.0 for base in row_bases]

    # A continuous path keeps the original route metaphor, with each row
    # carrying the color of the scene category it represents.
    route = f"M315,{centers[0]:.1f} H2110"
    for row_index in range(1, 5):
        previous = centers[row_index - 1]
        current = centers[row_index]
        if row_index % 2:
            route += f" C2182,{previous:.1f} 2182,{current:.1f} 2110,{current:.1f} H315"
        else:
            route += f" C243,{previous:.1f} 243,{current:.1f} 315,{current:.1f} H2110"
    parts.append(
        f'<path d="{route}" fill="none" stroke="#C8D3D1" stroke-width="6" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    for row_index, (scene, base_y, center_y) in enumerate(
        zip(scene_rows, row_bases, centers)
    ):
        category = scene["category"]
        color = SCENE_COLORS[category]
        direction_right = row_index % 2 == 0
        line_start = 315 if direction_right else 2110
        line_end = 2110 if direction_right else 315
        marker = f"scene-arrow-{row_index}"
        parts.append(
            f'<path d="M{line_start},{center_y:.1f} H{line_end}" fill="none" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round" '
            f'marker-end="url(#{marker})"/>'
        )

        label_lines = {
            "Residential": ["Residential"],
            "Public & Commercial": ["Public &", "Commercial"],
            "Urban & Transport": ["Urban &", "Transport"],
            "Parks & Gardens": ["Parks &", "Gardens"],
            "Natural & Off-road": ["Natural &", "Off-road"],
        }[category]
        label_y = center_y - (12 if len(label_lines) == 2 else 0)
        for line_index, line in enumerate(label_lines):
            parts.append(
                f'<text x="268" y="{label_y + line_index * 24:.1f}" class="category" '
                f'text-anchor="end">{esc(line)}</text>'
            )
        parts.extend(
            [
                f'<circle cx="315" cy="{center_y:.1f}" r="21" fill="{COLORS["paper"]}" stroke="{color}" stroke-width="5"/>',
                f'<circle cx="315" cy="{center_y:.1f}" r="4" fill="{color}"/>',
                f'<text x="315" y="{center_y - 30:.1f}" class="waypoint-index" text-anchor="middle">{row_index + 1:02d}</text>',
            ]
        )

        hours = float(scene["video_hours_share_pct"])
        frames = float(scene["sampled_frames_share_pct"])
        stat_x = 2140.0
        track_x = stat_x + 24.0
        track_w = 126.0
        value_x = 2358.0
        for metric_index, (letter, value, metric_color) in enumerate(
            (("H", hours, COLORS["teal"]), ("F", frames, COLORS["coral"]))
        ):
            y = center_y - 11.0 + metric_index * 25.0
            width = track_w * value / 35.0
            parts.extend(
                [
                    f'<text x="{stat_x:.1f}" y="{y + 5:.1f}" class="scene-metric" fill="{metric_color}">{letter}</text>',
                    f'<line x1="{track_x:.1f}" y1="{y:.1f}" x2="{track_x + track_w:.1f}" y2="{y:.1f}" class="scene-track"/>',
                    f'<line x1="{track_x:.1f}" y1="{y:.1f}" x2="{track_x + width:.1f}" y2="{y:.1f}" stroke="{metric_color}" stroke-width="8" stroke-linecap="round"/>',
                    f'<text x="{value_x:.1f}" y="{y + 5:.1f}" class="scene-value" text-anchor="end">{value:.1f}%</text>',
                ]
            )

        for col_index, x in enumerate(x_positions):
            image_index = row_index * 7 + col_index
            y = base_y + y_offsets[col_index]
            clip, body = image_markup(
                main_images[image_index],
                f"integrated-main-{row_index}-{col_index}",
                x,
                y,
                242.0,
                124.0,
                COLORS["line"],
            )
            clips.append(clip)
            parts.append(body)
    return clips, "".join(parts)


def ood_markup(ood_images: list[EmbeddedImage]) -> tuple[list[str], str]:
    if len(ood_images) != 12:
        raise RuntimeError("Expected 12 out-of-domain images")
    clips: list[str] = []
    parts = [
        '<path d="M2110,1174 C2290,1190 2355,1231 2355,1253 H49" fill="none" stroke="#718FA5" stroke-width="3" stroke-dasharray="10 10" stroke-linecap="round" opacity="0.85"/>',
        f'<rect x="35" y="1219" width="640" height="44" fill="{COLORS["paper"]}"/>',
        '<text x="42" y="1248" class="ood-heading">OUT-OF-DOMAIN DATASETS FOR EVALUATION</text>',
    ]
    group_x = [42.0, 626.0, 1210.0, 1794.0]
    group_names = [
        ("UZH-FPV", ""),
        ("TUM RGB-D SLAM", ""),
        ("Office-Go2", " · in-house captured"),
        ("Planetary Rover", " · in-house web-curated"),
    ]
    image_y = 1317.0
    for group_index, (base_x, (name, tag)) in enumerate(zip(group_x, group_names)):
        center = base_x + 272.5
        parts.append(
            f'<circle cx="{center:.1f}" cy="1253" r="8" fill="{COLORS["paper"]}" stroke="{COLORS["blue"]}" stroke-width="3"/>'
        )
        if tag:
            parts.append(
                f'<text x="{base_x:.1f}" y="1297" class="ood-name"><tspan>{esc(name)}</tspan><tspan class="ood-tag">{esc(tag)}</tspan></text>'
            )
        else:
            parts.append(
                f'<text x="{base_x:.1f}" y="1297" class="ood-name">{esc(name)}</text>'
            )
        for col_index in range(3):
            x = base_x + 184.0 * col_index
            image_index = group_index * 3 + col_index
            clip, body = image_markup(
                ood_images[image_index],
                f"integrated-ood-{group_index}-{col_index}",
                x,
                image_y,
                177.0,
                164.0,
                "#C8D7E0",
            )
            clips.append(clip)
            parts.append(body)
    return clips, "".join(parts)


def marker_defs() -> str:
    parts = []
    for index, color in enumerate(SCENE_COLORS.values()):
        parts.append(
            f'<marker id="scene-arrow-{index}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 Z" fill="{color}"/></marker>'
        )
    return "".join(parts)


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
    .band-title {{ font-size: 19px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: 1.65px; }}
    .band-note {{ font-size: 16px; font-weight: 600; fill: {COLORS['muted']}; }}
    .source-name {{ font-size: 17.5px; font-weight: 700; fill: {COLORS['ink']}; }}
    .source-rank {{ font-size: 13px; font-weight: 700; fill: {COLORS['faint']}; letter-spacing: 1px; }}
    .source-value {{ font-size: 14.5px; font-weight: 600; fill: {COLORS['ink']}; }}
    .source-key-title {{ font-size: 15px; font-weight: 700; fill: {COLORS['muted']}; letter-spacing: 1.1px; }}
    .source-key {{ font-size: 15px; font-weight: 600; fill: {COLORS['ink']}; }}
    .source-separator {{ stroke: {COLORS['line_soft']}; stroke-width: 1.3; }}
    .source-track {{ stroke: {COLORS['line_soft']}; stroke-width: 7; stroke-linecap: round; }}
    .source-hours {{ stroke: {COLORS['teal']}; stroke-width: 7; stroke-linecap: round; }}
    .source-frames {{ stroke: {COLORS['coral']}; stroke-width: 7; stroke-linecap: round; }}
    .source-hours-dot {{ fill: {COLORS['teal']}; stroke: #FFFFFF; stroke-width: 1.6; }}
    .source-frames-dot {{ fill: {COLORS['coral']}; stroke: #FFFFFF; stroke-width: 1.6; }}
    .metric-letter {{ font-size: 14px; font-weight: 800; }}
    .metric-hours {{ fill: {COLORS['teal']}; }}
    .metric-frames {{ fill: {COLORS['coral']}; }}
    .icon-label {{ font-size: 19px; font-weight: 600; fill: {COLORS['ink']}; }}
    .category {{ font-size: 23px; font-weight: 700; fill: {COLORS['ink']}; }}
    .waypoint-index {{ font-size: 14px; font-weight: 700; fill: {COLORS['faint']}; letter-spacing: 1px; }}
    .scene-track {{ stroke: {COLORS['line_soft']}; stroke-width: 8; stroke-linecap: round; }}
    .scene-metric {{ font-size: 15px; font-weight: 800; }}
    .scene-value {{ font-size: 15px; font-weight: 700; fill: {COLORS['ink']}; }}
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
    defs = marker_defs() + "".join(main_clips + ood_clips)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{PAGE_WIDTH_IN:.6f}in" height="{PAGE_HEIGHT_IN:.6f}in"
     viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-labelledby="title desc">
  <title id="title">NavAnywhere overview with integrated corpus composition</title>
  <desc id="desc">Fifteen source-composition glyphs, thirty-five in-domain frames aligned to five scene-composition rows, and twelve out-of-domain evaluation frames.</desc>
  <defs>{defs}</defs>
  {style_markup()}
  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{COLORS['paper']}"/>
  <g fill="none" stroke="{COLORS['line_soft']}" stroke-width="1.4" opacity="0.70">
    <path d="M0,586 C190,526 380,542 540,592 S900,656 1080,592 S1440,524 1630,584 S2030,654 2400,562"/>
    <path d="M0,860 C220,792 420,808 610,868 S1000,936 1200,862 S1580,790 1790,854 S2150,928 2400,838"/>
    <path d="M0,1116 C250,1052 455,1068 650,1122 S1030,1180 1250,1114 S1620,1046 1850,1110 S2180,1168 2400,1094"/>
  </g>
  {header_markup()}
  {source_fingerprint_markup(dataset_rows)}
  {coverage_markup()}
  {gallery}
  {ood}
</svg>
"""


def build_html(svg_name: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: {PAGE_WIDTH_IN:.6f}in {PAGE_HEIGHT_IN:.6f}in; margin: 0; }}
html, body {{ margin: 0; padding: 0; width: {PAGE_WIDTH_IN:.6f}in; height: {PAGE_HEIGHT_IN:.6f}in; overflow: hidden; background: {COLORS['paper']}; }}
img {{ display: block; width: {PAGE_WIDTH_IN:.6f}in; height: {PAGE_HEIGHT_IN:.6f}in; }}
</style></head><body><img src="{esc(svg_name)}" alt="Integrated NavAnywhere overview"></body></html>
"""


def build_caption() -> str:
    return rf"""\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/{BASE_NAME}_macos.pdf}}
  \caption{{\textbf{{NavAnywhere corpus overview and composition.}}
  The source fingerprint jointly reports each formal dataset's share of video
  hours (H) and sampled RGB frames (F) on a logarithmic scale. The central
  route aligns 35 representative frames with five scene categories; paired
  bars on each row report the corresponding duration and frame shares. Twelve
  additional frames summarize four out-of-domain evaluation datasets.}}
  \label{{fig:navanywhere-overview-integrated}}
\end{{figure*}}
"""


def build_readme() -> str:
    return f"""# Integrated NavAnywhere overview

This redesign treats corpus statistics as part of the overview rather than as
separate appended charts.

- Source composition is shown as 15 paired log-scale glyphs in the header.
- Scene composition is attached directly to the five corresponding image rows.
- All 47 original overview-v3 images are retained: 35 in-domain and 12 OOD.
- Embedded JPEG payloads are copied byte-for-byte from overview v3.
- `CASIA-Nav` is displayed as `In-house Collected`.

Recommended files:

- `{BASE_NAME}_macos.pdf`: PDF 1.4 for Preview.app.
- `{BASE_NAME}_600dpi.png`: high-resolution RGB fallback.
- `{BASE_NAME}.svg`: editable, self-contained vector source.
- `{BASE_NAME}_bundle.zip`: complete delivery package.
"""


def write_provenance(output_dir: Path, images: list[EmbeddedImage]) -> None:
    payload_hashes = [
        hashlib.sha256(image.href.encode("ascii")).hexdigest() for image in images
    ]
    manifest = {
        "artifact": BASE_NAME,
        "design": "integrated source fingerprint and row-aligned scene statistics",
        "canvas": {
            "viewbox": [0, 0, CANVAS_W, CANVAS_H],
            "physical_size_inches": [PAGE_WIDTH_IN, PAGE_HEIGHT_IN],
        },
        "image_count": {"total": 47, "in_domain": 35, "out_of_domain": 12},
        "embedded_image_href_sha256": payload_hashes,
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

    for path in (OVERVIEW_SVG, DATASET_CSV, SCENE_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)

    images = extract_images(OVERVIEW_SVG)
    dataset_rows = read_csv(DATASET_CSV)
    scene_rows = read_csv(SCENE_CSV)
    validate_data(dataset_rows, scene_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{BASE_NAME}.svg"
    html_path = output_dir / f"{BASE_NAME}_print.html"
    svg_path.write_text(
        build_svg(images, dataset_rows, scene_rows), encoding="utf-8"
    )
    html_path.write_text(build_html(svg_path.name), encoding="utf-8")
    (output_dir / "caption.tex").write_text(build_caption(), encoding="utf-8")
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")
    shutil.copy2(DATASET_CSV, output_dir / "dataset_composition.csv")
    shutil.copy2(SCENE_CSV, output_dir / "scene_composition.csv")
    write_provenance(output_dir, images)
    print(f"Wrote {svg_path}")
    print(f"Wrote {html_path}")
    print(f"Embedded images: {len(images)}")
    print(f"Page: {PAGE_WIDTH_IN:.4f} x {PAGE_HEIGHT_IN:.4f} in")


if __name__ == "__main__":
    main()
