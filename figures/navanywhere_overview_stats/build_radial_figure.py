#!/usr/bin/env python3
"""Build a dense radial NavAnywhere overview around a scene-composition pie.

The original 35 in-domain frames are retained byte-for-byte and augmented with
75 audited NavAnywhere frames, giving 22 examples per scene category.  The
held-out band contains TUM RGB-D SLAM, Office-Go2, and Planetary Rover; UZH-FPV
is intentionally excluded.  Added frames receive only a centered 16:9 crop,
Lanczos resize, and JPEG encoding.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import shutil
from pathlib import Path

from lxml import etree
from PIL import Image, ImageOps

from build_integrated_figure import (
    COLORS,
    DATASET_CSV,
    DISPLAY_DATASET_NAMES,
    OVERVIEW_SVG,
    SCENE_CSV,
    EmbeddedImage,
    esc,
    extract_images,
    read_csv,
    sha256,
    validate_data,
)


DEFAULT_OUTPUT_DIR = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/visualizations/2026/09/22/"
    "01a0c742-f9c0-7ab1-9bd0-dca2be5385fa/"
    "navanywhere_overview_radial_20260922"
)

BASE_NAME = "navanywhere_overview_radial_20260922"
LOGO_NAME = "navanywhere_compass_logo.svg"
CANVAS_W = 2600
CANVAS_H = 1440
PAGE_WIDTH_IN = 7.0
PAGE_HEIGHT_IN = PAGE_WIDTH_IN * CANVAS_H / CANVAS_W

PIE_CX = 1300.0
PIE_CY = 623.0
PIE_R = 210.0
PIE_INNER_R = 135.0
PIE_START_DEG = -148.8

# A brighter, more saturated palette for the pie, labels, and image borders.
SCENE_COLORS = {
    "Residential": "#A276D2",
    "Public & Commercial": "#E48346",
    "Urban & Transport": "#4B97D1",
    "Parks & Gardens": "#53B575",
    "Natural & Off-road": "#B99A32",
}

MAIN_IMAGE_W = 164.0
MAIN_IMAGE_H = MAIN_IMAGE_W * 9.0 / 16.0
MAIN_GAP_X = 7.0
MAIN_GAP_Y = 7.0
MAIN_GRID_COLS = 15
MAIN_GRID_ROWS = 9
MAIN_GRID_X = (CANVAS_W - (MAIN_GRID_COLS * MAIN_IMAGE_W + (MAIN_GRID_COLS - 1) * MAIN_GAP_X)) / 2.0
MAIN_GRID_Y = 180.0
MAIN_HOLE_COLS = range(5, 10)
MAIN_HOLE_ROWS = range(2, 7)

OOD_IMAGE_W = 245.0
OOD_IMAGE_H = OOD_IMAGE_W * 9.0 / 16.0
OOD_GAP_X = 10.0

SCENE_REVIEW_CSV = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/stats/"
    "navanywhere_iclr_20260914/evidence/scene_review_labels.csv"
)

# Fifteen additional audited frames per scene category.  The order is chosen for
# visual variety while keeping every original representative frame intact.
ADDED_REVIEW_IDS = {
    "Residential": [
        "R0030", "R0316", "R0457", "R0036", "R0318", "R0325", "R0326", "R0028",
        "R0035", "R0042", "R0319", "R0322", "R0329", "R0044", "R0331",
    ],
    "Public & Commercial": [
        "R0006", "R0137", "R0481", "R0045", "R0010", "R0047", "R0479", "R0141",
        "R0057", "R0055", "R0059", "R0144", "R0338", "R0485", "R0514",
    ],
    "Urban & Transport": [
        "R0458", "R0531", "R0397", "R0015", "R0054", "R0244", "R0157", "R0399",
        "R0075", "R0158", "R0161", "R0256", "R0400", "R0492", "R0532",
    ],
    "Parks & Gardens": [
        "R0062", "R0520", "R0027", "R0178", "R0288", "R0404", "R0555", "R0111",
        "R0186", "R0074", "R0183", "R0192", "R0441", "R0203", "R0446",
    ],
    "Natural & Off-road": [
        "R0107", "R0212", "R0430", "R0103", "R0222", "R0293", "R0119", "R0449",
        "R0118", "R0123", "R0213", "R0305", "R0427", "R0436", "R0463",
    ],
}


def polar(cx: float, cy: float, radius: float, degrees: float) -> tuple[float, float]:
    angle = math.radians(degrees)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def sector_path(start_deg: float, end_deg: float) -> str:
    x0, y0 = polar(PIE_CX, PIE_CY, PIE_R, start_deg)
    x1, y1 = polar(PIE_CX, PIE_CY, PIE_R, end_deg)
    large_arc = 1 if end_deg - start_deg > 180.0 else 0
    return (
        f"M {PIE_CX:.2f},{PIE_CY:.2f} L {x0:.2f},{y0:.2f} "
        f"A {PIE_R:.2f},{PIE_R:.2f} 0 {large_arc} 1 {x1:.2f},{y1:.2f} Z"
    )


def compass_logo(cx: float, cy: float, scale: float = 1.0) -> str:
    return f"""
    <g transform="translate({cx:.2f} {cy:.2f}) scale({scale:.3f})">
      <circle cx="0" cy="0" r="21" fill="none" stroke="{COLORS['ink']}" stroke-width="1.6"/>
      <circle cx="0" cy="0" r="14" fill="none" stroke="{COLORS['ink']}" stroke-width="0.8" opacity="0.45"/>
      <path d="M0,-27 L5,-5 L0,0 L-5,-5 Z" fill="{COLORS['teal']}"/>
      <path d="M0,27 L-5,5 L0,0 L5,5 Z" fill="{COLORS['ink']}"/>
      <path d="M27,0 L5,5 L0,0 L5,-5 Z" fill="{COLORS['ink']}" opacity="0.65"/>
      <path d="M-27,0 L-5,-5 L0,0 L-5,5 Z" fill="{COLORS['ink']}" opacity="0.65"/>
      <circle cx="0" cy="0" r="3.2" fill="{COLORS['coral']}"/>
    </g>
    """


def embodiment_icons(x: float, y: float) -> str:
    """Three compact vector glyphs for human, ground robot, and drone."""
    return f"""
    <g transform="translate({x:.2f} {y:.2f})" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <g transform="translate(0 0)" stroke="{COLORS['teal']}" stroke-width="3.2" aria-label="Human">
        <title>Human</title>
        <circle cx="0" cy="-14" r="5.5"/>
        <path d="M0,-7 V10 M-12,1 L0,-5 L12,1 M0,10 L-10,24 M0,10 L10,24"/>
      </g>
      <g transform="translate(96 1)" stroke="{COLORS['blue']}" stroke-width="3.0" aria-label="Ground robot">
        <title>Ground robot</title>
        <path d="M-20,-5 H20 V12 H-20 Z M-8,-5 V-17 H10 V-5 M1,-17 V-23"/>
        <circle cx="-12" cy="17" r="5.5"/><circle cx="13" cy="17" r="5.5"/>
      </g>
      <g transform="translate(192 0)" stroke="{COLORS['coral']}" stroke-width="2.8" aria-label="Drone">
        <title>Drone</title>
        <rect x="-9" y="-5" width="18" height="12" rx="3"/>
        <path d="M-9,-1 L-25,-13 M9,-1 L25,-13 M-9,3 L-25,15 M9,3 L25,15"/>
        <ellipse cx="-30" cy="-16" rx="9" ry="3.5"/><ellipse cx="30" cy="-16" rx="9" ry="3.5"/>
        <ellipse cx="-30" cy="18" rx="9" ry="3.5"/><ellipse cx="30" cy="18" rx="9" ry="3.5"/>
      </g>
    </g>
    """


def condition_icons(x: float, y: float) -> str:
    """Day/night above spring, summer, autumn, and winter glyphs."""
    return f"""
    <g transform="translate({x:.2f} {y:.2f})" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <g transform="translate(68 0)" stroke="#B78B3D" stroke-width="2.6" aria-label="Day">
        <title>Day</title><circle cx="0" cy="0" r="8"/>
        <path d="M0,-14 V-20 M0,14 V20 M-14,0 H-20 M14,0 H20 M-10,-10 L-14,-14 M10,-10 L14,-14 M-10,10 L-14,14 M10,10 L14,14"/>
      </g>
      <g transform="translate(138 0)" stroke="{COLORS['blue']}" stroke-width="2.8" aria-label="Night">
        <title>Night</title><path d="M4,-16 A17,17 0 1,0 11,14 A14,14 0 1,1 4,-16 Z"/>
      </g>
      <g transform="translate(0 35)" stroke="#5E8D70" stroke-width="2.2" aria-label="Spring">
        <title>Spring</title><circle cx="0" cy="0" r="2.3" fill="#5E8D70"/>
        <circle cx="0" cy="-6" r="4"/><circle cx="6" cy="0" r="4"/>
        <circle cx="0" cy="6" r="4"/><circle cx="-6" cy="0" r="4"/>
      </g>
      <g transform="translate(69 35)" stroke="#D09A32" stroke-width="2.2" aria-label="Summer">
        <title>Summer</title><circle cx="0" cy="-2" r="6"/>
        <path d="M0,-12 V-16 M0,8 V12 M-10,-2 H-14 M10,-2 H14 M-7,-9 L-10,-12 M7,-9 L10,-12 M-7,5 L-10,8 M7,5 L10,8 M-13,15 Q-7,11 0,15 T13,15"/>
      </g>
      <g transform="translate(138 35)" stroke="#9A7156" stroke-width="2.4" aria-label="Autumn">
        <title>Autumn</title><path d="M-10,5 C-11,-6 1,-13 11,-10 C10,2 2,11 -10,5 Z M-9,6 L10,-10 M-8,7 L-12,12"/>
      </g>
      <g transform="translate(207 35)" stroke="#607F9B" stroke-width="2.3" aria-label="Winter">
        <title>Winter</title><path d="M-12,0 H12 M-6,-10 L6,10 M6,-10 L-6,10 M-12,0 L-8,-4 M-12,0 L-8,4 M12,0 L8,-4 M12,0 L8,4"/>
      </g>
    </g>
    """


def image_markup(
    image: EmbeddedImage,
    clip_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    border_color: str,
    radius: float = 8.0,
    border_width: float = 1.8,
) -> tuple[str, str]:
    clip = (
        f'<clipPath id="{clip_id}"><rect x="{x:.2f}" y="{y:.2f}" '
        f'width="{width:.2f}" height="{height:.2f}" rx="{radius:.2f}"/></clipPath>'
    )
    body = f"""
    <g aria-label="{esc(image.aria_label)}">
      <image x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}"
             href="{image.href}" preserveAspectRatio="xMidYMid slice"
             clip-path="url(#{clip_id})"/>
      <rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}"
            rx="{radius:.2f}" fill="none" stroke="#FFFFFF" stroke-width="5"/>
      <rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}"
            rx="{radius:.2f}" fill="none" stroke="{border_color}" stroke-opacity="0.98"
            stroke-width="{border_width:.2f}"/>
    </g>
    """
    return clip, body


def derive_added_frame(row: dict[str, str]) -> tuple[EmbeddedImage, dict[str, object]]:
    source = Path(row["frame_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        opened.verify()
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        source_dimensions = [image.width, image.height]
        image = image.convert("RGB")
        image = ImageOps.fit(
            image,
            (720, 405),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=93,
            subsampling=0,
            optimize=True,
            progressive=False,
        )
    payload = buffer.getvalue()
    href = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
    label = (
        f"NavAnywhere · {row['review_category']} · {row['dataset']} · "
        f"{row['review_id']}"
    )
    metadata: dict[str, object] = {
        "review_id": row["review_id"],
        "category": row["review_category"],
        "dataset": row["dataset"],
        "trajectory": row["trajectory"],
        "review_confidence": row["model_confidence"],
        "source_path": str(source),
        "source_dimensions": source_dimensions,
        "derived_dimensions": [720, 405],
        "operation": "center crop to 16:9; Lanczos resize; JPEG quality 93, 4:4:4",
        "content_editing": "none",
        "source_sha256": sha256(source),
        "derived_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return EmbeddedImage(href=href, aria_label=label), metadata


def load_added_images() -> tuple[dict[str, list[EmbeddedImage]], list[dict[str, object]]]:
    requested = {
        review_id
        for review_ids in ADDED_REVIEW_IDS.values()
        for review_id in review_ids
    }
    rows = {
        row["review_id"]: row
        for row in read_csv(SCENE_REVIEW_CSV)
        if row["review_id"] in requested
    }
    if set(rows) != requested:
        raise RuntimeError(f"Missing added review frames: {sorted(requested - set(rows))}")

    by_category: dict[str, list[EmbeddedImage]] = {}
    metadata: list[dict[str, object]] = []
    for category, review_ids in ADDED_REVIEW_IDS.items():
        category_images: list[EmbeddedImage] = []
        for review_id in review_ids:
            row = rows[review_id]
            if row["review_category"] != category:
                raise RuntimeError(
                    f"Category mismatch for {review_id}: "
                    f"{row['review_category']} != {category}"
                )
            embedded, record = derive_added_frame(row)
            category_images.append(embedded)
            metadata.append(record)
        by_category[category] = category_images
    return by_category, metadata


def ring_scene_positions() -> list[list[tuple[float, float]]]:
    """Fill a rectangular image field while reserving only the central pie."""
    cells: list[tuple[float, float]] = []
    for row in range(MAIN_GRID_ROWS):
        for column in range(MAIN_GRID_COLS):
            if row in MAIN_HOLE_ROWS and column in MAIN_HOLE_COLS:
                continue
            cells.append(
                (
                    MAIN_GRID_X + column * (MAIN_IMAGE_W + MAIN_GAP_X),
                    MAIN_GRID_Y + row * (MAIN_IMAGE_H + MAIN_GAP_Y),
                )
            )

    def angular_key(position: tuple[float, float]) -> tuple[float, float]:
        x, y = position
        center_x = x + MAIN_IMAGE_W / 2.0
        center_y = y + MAIN_IMAGE_H / 2.0
        angle = math.degrees(math.atan2(center_y - PIE_CY, center_x - PIE_CX))
        radius = math.hypot(center_x - PIE_CX, center_y - PIE_CY)
        return ((angle - PIE_START_DEG) % 360.0, radius)

    ordered = sorted(cells, key=angular_key)
    if len(ordered) != 110:
        raise RuntimeError(f"Expected 110 ring positions, found {len(ordered)}")
    return [ordered[index * 22 : (index + 1) * 22] for index in range(5)]


def hole_bounds() -> tuple[float, float, float, float]:
    left = MAIN_GRID_X + min(MAIN_HOLE_COLS) * (MAIN_IMAGE_W + MAIN_GAP_X)
    right = (
        MAIN_GRID_X
        + max(MAIN_HOLE_COLS) * (MAIN_IMAGE_W + MAIN_GAP_X)
        + MAIN_IMAGE_W
    )
    top = MAIN_GRID_Y + min(MAIN_HOLE_ROWS) * (MAIN_IMAGE_H + MAIN_GAP_Y)
    bottom = (
        MAIN_GRID_Y
        + max(MAIN_HOLE_ROWS) * (MAIN_IMAGE_H + MAIN_GAP_Y)
        + MAIN_IMAGE_H
    )
    return left, top, right, bottom


def ray_to_hole_edge(degrees: float) -> tuple[float, float]:
    """Intersect a ray from the pie center with the reserved grid opening."""
    left, top, right, bottom = hole_bounds()
    angle = math.radians(degrees)
    dx = math.cos(angle)
    dy = math.sin(angle)
    candidates: list[float] = []
    if dx > 0:
        candidates.append((right - PIE_CX) / dx)
    elif dx < 0:
        candidates.append((left - PIE_CX) / dx)
    if dy > 0:
        candidates.append((bottom - PIE_CY) / dy)
    elif dy < 0:
        candidates.append((top - PIE_CY) / dy)
    for distance in sorted(value for value in candidates if value > 0):
        x = PIE_CX + distance * dx
        y = PIE_CY + distance * dy
        if left - 0.01 <= x <= right + 0.01 and top - 0.01 <= y <= bottom + 0.01:
            return x, y
    raise RuntimeError(f"Unable to intersect scene ray at {degrees:.2f} degrees")


def top_metrics_markup() -> str:
    metrics = [
        ("70,756", "SEQUENCES"),
        ("1,189 h", "VIDEO"),
        ("17.5M", "SAMPLED FRAMES"),
        ("15", "FORMAL SOURCES"),
        ("17+", "COUNTRIES"),
        ("97.3%", "REAL-WORLD VIDEO"),
    ]
    left = 70.0
    cell = (CANVAS_W - 2 * left) / 8.0
    parts = [
        '<g id="dataset-scale">',
        '<text x="70" y="28" class="eyebrow">DATASET SCALE &amp; DIVERSITY</text>',
    ]
    for index, (value, label) in enumerate(metrics):
        x = left + index * cell
        if index:
            parts.append(
                f'<line x1="{x - 18:.2f}" y1="43" x2="{x - 18:.2f}" y2="118" class="metric-divider"/>'
            )
        parts.extend(
            [
                f'<text x="{x:.2f}" y="80" class="metric-number">{value}</text>',
                f'<text x="{x + 1:.2f}" y="109" class="metric-label">{label}</text>',
            ]
        )
    for index in (6, 7):
        x = left + index * cell
        parts.append(
            f'<line x1="{x - 18:.2f}" y1="43" x2="{x - 18:.2f}" y2="118" class="metric-divider"/>'
        )
    embodiment_x = left + 6 * cell
    conditions_x = left + 7 * cell
    parts.extend(
        [
            f'<text x="{embodiment_x:.2f}" y="53" class="icon-panel-title">EMBODIMENT</text>',
            f'<text x="{embodiment_x + 137.0:.2f}" y="94" text-anchor="middle" class="panel-value">Human · Ground robot · Drone</text>',
            f'<text x="{conditions_x:.2f}" y="53" class="icon-panel-title">CONDITIONS</text>',
            f'<text x="{conditions_x + 145.0:.2f}" y="80" text-anchor="middle" class="panel-value">Day · Night</text>',
            f'<text x="{conditions_x + 145.0:.2f}" y="108" text-anchor="middle" class="panel-value-compact">Spring · Summer · Autumn · Winter</text>',
        ]
    )
    parts.extend(
        [
            '<line x1="70" y1="138" x2="2530" y2="138" class="rule"/>',
            "</g>",
        ]
    )
    return "".join(parts)


def radial_scene_markup(
    main_images: list[EmbeddedImage], scene_rows: list[dict[str, str]]
) -> tuple[list[str], str]:
    if len(main_images) != 110:
        raise RuntimeError("Expected 110 in-domain images")

    clips: list[str] = []
    parts = ['<g id="scene-composition">']

    # The 15 x 9 field has a 5 x 5 central opening.  Its remaining 110 cells
    # form five contiguous angular groups of 22 images around the pie.
    position_groups = ring_scene_positions()
    for scene_index, (scene, positions) in enumerate(zip(scene_rows, position_groups)):
        color = SCENE_COLORS[scene["category"]]
        scene_images = main_images[scene_index * 22 : (scene_index + 1) * 22]
        for local_index, ((image_x, image_y), image) in enumerate(
            zip(positions, scene_images)
        ):
            clip, body = image_markup(
                image,
                f"radial-main-{scene_index}-{local_index}",
                image_x,
                image_y,
                MAIN_IMAGE_W,
                MAIN_IMAGE_H,
                color,
                radius=6.0,
                border_width=3.6,
            )
            clips.append(clip)
            parts.append(body)

    parts.extend(
        [
            '<text x="1300" y="400" text-anchor="middle" class="radial-heading">SCENE COMPOSITION</text>',
        ]
    )

    starts: list[float] = []
    ends: list[float] = []
    mids: list[float] = []
    cursor = PIE_START_DEG
    for scene in scene_rows:
        span = 360.0 * float(scene["sampled_frames_share_pct"]) / 100.0
        starts.append(cursor)
        ends.append(cursor + span)
        mids.append(cursor + span / 2.0)
        cursor += span

    # The proportional pie is drawn before its white center.
    for scene, start_deg, end_deg, mid_deg in zip(scene_rows, starts, ends, mids):
        color = SCENE_COLORS[scene["category"]]
        frames = float(scene["sampled_frames_share_pct"])
        parts.append(
            f'<path d="{sector_path(start_deg, end_deg)}" fill="{color}" '
            'stroke="#FCFCFA" stroke-width="7" stroke-linejoin="round"/>'
        )
        value_x, value_y = polar(
            PIE_CX, PIE_CY, (PIE_R + PIE_INNER_R) / 2.0, mid_deg
        )
        parts.append(
            f'<text x="{value_x:.2f}" y="{value_y + 6:.2f}" text-anchor="middle" '
            f'class="sector-value">{frames:.1f}%</text>'
        )

    parts.extend(
        [
            f'<circle cx="{PIE_CX:.2f}" cy="{PIE_CY:.2f}" r="{PIE_INNER_R:.2f}" class="pie-center"/>',
            compass_logo(PIE_CX, PIE_CY - 48.0, 1.30),
            f'<text x="{PIE_CX:.2f}" y="{PIE_CY + 24:.2f}" text-anchor="middle" class="center-title">NavAnywhere</text>',
            f'<text x="{PIE_CX:.2f}" y="{PIE_CY + 54:.2f}" text-anchor="middle" class="center-subtitle">DATASET</text>',
        ]
    )

    # Category names follow the mid-angle of their corresponding sector.
    for scene, mid_deg in zip(scene_rows, mids):
        category = scene["category"]
        color = SCENE_COLORS[category]
        label_x, label_y = polar(PIE_CX, PIE_CY, PIE_R + 30.0, mid_deg)
        cosine = math.cos(math.radians(mid_deg))
        anchor = "start" if cosine > 0.25 else "end" if cosine < -0.25 else "middle"
        sizing = ""
        if category == "Natural & Off-road":
            # Keep the full label inside the reserved 5 x 5 center opening.
            label_x = 1080.0
            anchor = "end"
            sizing = ' textLength="195" lengthAdjust="spacingAndGlyphs"'
        parts.append(
            f'<text x="{label_x:.2f}" y="{label_y + 7.0:.2f}" text-anchor="{anchor}" '
            f'class="scene-name" fill="{color}"{sizing}>{esc(category)}</text>'
        )

    parts.append("</g>")
    return clips, "".join(parts)


def source_line_markup(dataset_rows: list[dict[str, str]]) -> str:
    names = [
        DISPLAY_DATASET_NAMES.get(row["dataset"], row["dataset"])
        for row in dataset_rows
    ]
    source_text = "  ·  ".join(names)
    return f"""
    <g id="formal-sources">
      <line x1="70" y1="1088" x2="2530" y2="1088" class="rule"/>
      <text x="70" y="1126" class="source-heading">15 SOURCES</text>
      <text x="260" y="1126" class="source-list" textLength="2270" lengthAdjust="spacing">{esc(source_text)}</text>
      <line x1="70" y1="1153" x2="2530" y2="1153" class="rule-soft"/>
    </g>
    """


def ood_markup(ood_images: list[EmbeddedImage]) -> tuple[list[str], str]:
    if len(ood_images) != 9:
        raise RuntimeError("Expected 9 out-of-domain images")
    groups = [
        ("TUM RGB-D SLAM", ""),
        ("Office-Go2", " · in-house captured"),
        ("Planetary Rover", " · in-house web-curated"),
    ]
    group_width = 3 * OOD_IMAGE_W + 2 * OOD_GAP_X
    left = 70.0
    gap = (CANVAS_W - 2 * left - 3 * group_width) / 2.0
    image_y = 1265.0
    clips: list[str] = []
    parts = [
        '<g id="out-of-domain-evaluation">',
        '<text x="70" y="1195" class="ood-heading">OUT-OF-DOMAIN EVALUATION</text>',
    ]
    for group_index, (name, tag) in enumerate(groups):
        x0 = left + group_index * (group_width + gap)
        parts.append(
            f'<text x="{x0:.2f}" y="1239" class="ood-name">{esc(name)}'
            f'<tspan class="ood-tag">{esc(tag)}</tspan></text>'
        )
        parts.append(
            f'<line x1="{x0:.2f}" y1="1252" x2="{x0 + group_width:.2f}" y2="1252" class="ood-group-rule"/>'
        )
        for local_index in range(3):
            image = ood_images[group_index * 3 + local_index]
            image_x = x0 + local_index * (OOD_IMAGE_W + OOD_GAP_X)
            clip, body = image_markup(
                image,
                f"radial-ood-{group_index}-{local_index}",
                image_x,
                image_y,
                OOD_IMAGE_W,
                OOD_IMAGE_H,
                COLORS["blue"],
                radius=7.0,
            )
            clips.append(clip)
            parts.append(body)
    parts.append("</g>")
    return clips, "".join(parts)


def style_markup() -> str:
    return f"""
  <style>
    text {{ font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; }}
    .eyebrow {{ font-size: 22px; font-weight: 700; fill: {COLORS['muted']}; letter-spacing: 2.2px; }}
    .metric-number {{ font-size: 39px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: -0.4px; }}
    .metric-label {{ font-size: 15px; font-weight: 700; fill: {COLORS['faint']}; letter-spacing: 1.1px; }}
    .icon-panel-title {{ font-size: 15px; font-weight: 800; fill: {COLORS['faint']}; letter-spacing: 1.2px; }}
    .panel-value {{ font-size: 18px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: -0.15px; }}
    .panel-value-compact {{ font-size: 16px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: -0.25px; }}
    .metric-divider {{ stroke: {COLORS['line']}; stroke-width: 1.2; }}
    .rule {{ stroke: {COLORS['ink']}; stroke-width: 1.35; }}
    .rule-soft {{ stroke: {COLORS['line']}; stroke-width: 1.2; }}
    .radial-heading {{ font-size: 22px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: 2.0px; }}
    .sector-value {{ font-size: 22px; font-weight: 800; fill: #FFFFFF; letter-spacing: -0.2px; paint-order: stroke; stroke: #17343D; stroke-opacity: 0.28; stroke-width: 1.4px; }}
    .pie-center {{ fill: {COLORS['paper']}; stroke: {COLORS['ink']}; stroke-width: 1.4; }}
    .center-title {{ font-size: 34px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: -0.7px; }}
    .center-subtitle {{ font-size: 15px; font-weight: 700; fill: {COLORS['muted']}; letter-spacing: 2.8px; }}
    .scene-name {{ font-size: 22px; font-weight: 750; paint-order: stroke; stroke: {COLORS['paper']}; stroke-width: 5px; stroke-linejoin: round; }}
    .source-heading {{ font-size: 20px; font-weight: 700; fill: {COLORS['teal']}; letter-spacing: 1.8px; }}
    .source-list {{ font-size: 20px; font-weight: 500; fill: {COLORS['muted']}; }}
    .ood-heading {{ font-size: 24px; font-weight: 700; fill: {COLORS['blue']}; letter-spacing: 2.1px; }}
    .ood-name {{ font-size: 24px; font-weight: 700; fill: {COLORS['ink']}; }}
    .ood-tag {{ font-size: 18px; font-weight: 600; fill: {COLORS['muted']}; }}
    .ood-group-rule {{ stroke: {COLORS['line']}; stroke-width: 1.2; }}
  </style>
    """


def build_svg(
    images: list[EmbeddedImage],
    dataset_rows: list[dict[str, str]],
    scene_rows: list[dict[str, str]],
) -> str:
    main_clips, radial = radial_scene_markup(images[:110], scene_rows)
    ood_clips, ood = ood_markup(images[110:])
    defs = "".join(main_clips + ood_clips)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{PAGE_WIDTH_IN:.6f}in" height="{PAGE_HEIGHT_IN:.6f}in"
     viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-labelledby="title desc">
  <title id="title">NavAnywhere radial dataset overview</title>
  <desc id="desc">A sampled-frame scene-composition pie radiates through a dense field of 110 in-domain frames; 9 frames from three out-of-domain evaluation datasets and corpus-scale statistics complete the overview.</desc>
  <defs>{defs}</defs>
  {style_markup()}
  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{COLORS['paper']}"/>
  {top_metrics_markup()}
  {radial}
  {source_line_markup(dataset_rows)}
  {ood}
</svg>
"""


def build_html(svg_name: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {PAGE_WIDTH_IN:.6f}in {PAGE_HEIGHT_IN:.6f}in; margin: 0; }}
html,body {{ margin:0; padding:0; width:{PAGE_WIDTH_IN:.6f}in; height:{PAGE_HEIGHT_IN:.6f}in; overflow:hidden; background:{COLORS['paper']}; }}
img {{ display:block; width:{PAGE_WIDTH_IN:.6f}in; height:{PAGE_HEIGHT_IN:.6f}in; }}
</style></head><body><img src="{svg_name}" alt="NavAnywhere radial dataset overview"></body></html>
"""


def build_logo_svg() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"
     viewBox="0 0 256 256" role="img" aria-labelledby="logo-title logo-desc">
  <title id="logo-title">NavAnywhere compass logo</title>
  <desc id="logo-desc">Standalone vector compass icon used in the NavAnywhere dataset overview.</desc>
  {compass_logo(128.0, 128.0, 3.75)}
</svg>
"""


def build_caption() -> str:
    return rf"""\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/{BASE_NAME}_macos.pdf}}
  \caption{{\textbf{{Overview of NavAnywhere.}}
  The central pie encodes the sampled-frame share of five scene categories. Each sector
  radiates to twenty-two representative in-domain frames. The lower band presents
  nine frames from three out-of-domain evaluation datasets, while the upper
  band summarizes corpus scale and diversity.}}
  \label{{fig:navanywhere-overview-radial}}
\end{{figure*}}
"""


def validate_embedded_images(
    source: Path, output: Path, expected: list[EmbeddedImage]
) -> None:
    parser = etree.XMLParser(resolve_entities=False, huge_tree=True)

    def hrefs(path: Path) -> list[str]:
        root = etree.parse(str(path), parser).getroot()
        return [
            image.get("href")
            or image.get("{http://www.w3.org/1999/xlink}href")
            for image in root.xpath("//*[local-name()='image']")
        ]

    source_hrefs = hrefs(source)
    output_hrefs = hrefs(output)
    expected_hrefs = [image.href for image in expected]
    if len(source_hrefs) != 47 or len(output_hrefs) != 119:
        raise RuntimeError("Unexpected source or output image count")
    if output_hrefs != expected_hrefs:
        raise RuntimeError("Embedded image ordering changed")
    retained = source_hrefs[:35] + source_hrefs[38:]
    removed_uzh = source_hrefs[35:38]
    if any(output_hrefs.count(href) != 1 for href in retained):
        raise RuntimeError("An original retained image is missing or duplicated")
    if any(href in output_hrefs for href in removed_uzh):
        raise RuntimeError("A removed UZH-FPV image is still embedded")
    added = [href for href in output_hrefs if href not in source_hrefs]
    if len(added) != 75 or len(set(added)) != 75:
        raise RuntimeError("Expected 75 unique added NavAnywhere payloads")


def write_provenance(
    output_dir: Path,
    images: list[EmbeddedImage],
    added_metadata: list[dict[str, object]],
) -> None:
    manifest = {
        "artifact": BASE_NAME,
        "design": "central proportional scene pie inside five contiguous radial image regions",
        "canvas": {
            "viewbox": [0, 0, CANVAS_W, CANVAS_H],
            "physical_size_inches": [PAGE_WIDTH_IN, PAGE_HEIGHT_IN],
        },
        "image_count": {"total": 119, "in_domain": 110, "out_of_domain": 9},
        "image_grouping": "five scene groups of twenty-two plus three OOD groups of three",
        "scene_composition_basis": "sampled_frames_share_pct",
        "scene_colors": SCENE_COLORS,
        "standalone_vector_logo": LOGO_NAME,
        "retained_original_payloads": {
            "in_domain": 35,
            "out_of_domain": 9,
            "policy": "embedded JPEG hrefs retained byte-for-byte from overview v3",
        },
        "removed_out_of_domain_dataset": "UZH-FPV",
        "added_frame_policy": (
            "center crop to 16:9; Lanczos resize to 720x405; JPEG quality 93, "
            "4:4:4; no content editing"
        ),
        "added_navanywhere_frames": added_metadata,
        "embedded_image_href_sha256": [
            hashlib.sha256(image.href.encode("ascii")).hexdigest() for image in images
        ],
        "inputs": {
            "overview_v3_svg": {"path": str(OVERVIEW_SVG), "sha256": sha256(OVERVIEW_SVG)},
            "dataset_composition_csv": {"path": str(DATASET_CSV), "sha256": sha256(DATASET_CSV)},
            "scene_composition_csv": {"path": str(SCENE_CSV), "sha256": sha256(SCENE_CSV)},
            "scene_review_labels_csv": {
                "path": str(SCENE_REVIEW_CSV),
                "sha256": sha256(SCENE_REVIEW_CSV),
            },
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_images = extract_images(OVERVIEW_SVG)
    dataset_rows = read_csv(DATASET_CSV)
    scene_rows = read_csv(SCENE_CSV)
    validate_data(dataset_rows, scene_rows)
    added_by_category, added_metadata = load_added_images()

    main_images: list[EmbeddedImage] = []
    for scene_index, scene in enumerate(scene_rows):
        category = scene["category"]
        main_images.extend(source_images[scene_index * 7 : (scene_index + 1) * 7])
        main_images.extend(added_by_category[category])
    # The first three OOD payloads in overview v3 are UZH-FPV.
    ood_images = source_images[38:]
    images = main_images + ood_images
    if len(main_images) != 110 or len(ood_images) != 9 or len(images) != 119:
        raise RuntimeError("Unexpected assembled image counts")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{BASE_NAME}.svg"
    html_path = output_dir / f"{BASE_NAME}_print.html"
    logo_path = output_dir / LOGO_NAME

    svg_path.write_text(
        build_svg(images, dataset_rows, scene_rows), encoding="utf-8"
    )
    validate_embedded_images(OVERVIEW_SVG, svg_path, images)
    html_path.write_text(build_html(svg_path.name), encoding="utf-8")
    logo_path.write_text(build_logo_svg(), encoding="utf-8")
    (output_dir / "caption.tex").write_text(build_caption(), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Radial NavAnywhere overview\n\n"
        "The central pie encodes scene composition by sampled-frame share. Five contiguous "
        "radial regions show twenty-two in-domain frames per category: all 35 originals "
        "plus 75 audited "
        "additions. The lower band contains TUM RGB-D SLAM, Office-Go2, and "
        "Planetary Rover (three frames each); UZH-FPV is excluded. The 44 retained "
        "JPEG payloads are byte-identical to overview v3. Added frames are center "
        "cropped to 16:9 and resized without content editing.\n"
        "The central compass icon is also provided as `navanywhere_compass_logo.svg`.\n",
        encoding="utf-8",
    )
    shutil.copy2(DATASET_CSV, output_dir / "dataset_composition.csv")
    shutil.copy2(SCENE_CSV, output_dir / "scene_composition.csv")
    write_provenance(output_dir, images, added_metadata)
    print(f"Wrote {svg_path}")
    print("Embedded images: 119 (110 NavAnywhere + 9 OOD; UZH-FPV removed)")
    print("Retained source payloads: 44 byte-identical; added audited frames: 75")
    print(f"Page: {PAGE_WIDTH_IN:.4f} x {PAGE_HEIGHT_IN:.4f} in")


if __name__ == "__main__":
    main()
