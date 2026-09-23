#!/usr/bin/env python3
"""Build a clean editorial NavAnywhere overview with integrated statistics.

The original overview-v3 image layout is preserved. Source composition is
summarized typographically, and paired scene shares replace the original
single percentage beside each scene row. No chart bars or dense axes are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from lxml import etree

from build_integrated_figure import (
    COLORS,
    DATASET_CSV,
    OVERVIEW_SVG,
    SCENE_CSV,
    extract_images,
    read_csv,
    sha256,
    validate_data,
)


DEFAULT_OUTPUT_DIR = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/visualizations/2026/09/22/"
    "01a0c742-f9c0-7ab1-9bd0-dca2be5385fa/"
    "navanywhere_overview_editorial_20260922"
)

BASE_NAME = "navanywhere_overview_editorial_20260922"
CANVAS_W = 2400
CANVAS_H = 1500
PAGE_WIDTH_IN = 7.0
PAGE_HEIGHT_IN = PAGE_WIDTH_IN * CANVAS_H / CANVAS_W
CONTENT_SHIFT_Y = 100


def source_summary_markup(dataset_rows: list[dict[str, str]]) -> str:
    ranked = sorted(
        dataset_rows,
        key=lambda row: float(row["video_hours_share_pct"]),
        reverse=True,
    )
    dominant = ranked[0]
    next_three = ranked[1:4]
    remaining = ranked[4:]

    def totals(rows: list[dict[str, str]]) -> tuple[float, float]:
        return (
            sum(float(row["video_hours_share_pct"]) for row in rows),
            sum(float(row["sampled_frames_share_pct"]) for row in rows),
        )

    groups = [totals([dominant]), totals(next_three), totals(remaining)]
    positions = [375.0, 965.0, 1690.0]
    labels = [
        "Ego4D · dominant source",
        "DL3DV-10K · KrishnaCam · RealEstate10K",
        "remaining 11 formal sources",
    ]
    parts = [
        '<g id="source-composition-summary" aria-label="Source composition summary">',
        '<text x="41" y="216" class="summary-heading">SOURCE BALANCE</text>',
        '<text x="41" y="246" class="summary-key">H · video hours</text>',
        '<text x="41" y="270" class="summary-key">F · sampled frames</text>',
        '<line x1="318" y1="201" x2="318" y2="272" class="summary-divider"/>',
        '<line x1="908" y1="201" x2="908" y2="272" class="summary-divider"/>',
        '<line x1="1633" y1="201" x2="1633" y2="272" class="summary-divider"/>',
    ]
    for x, (hours, frames), label in zip(positions, groups, labels):
        parts.extend(
            [
                f'<text x="{x:.1f}" y="232" class="summary-number">'
                f'<tspan fill="{COLORS["teal"]}">{hours:.1f}</tspan>'
                f'<tspan class="summary-slash"> / </tspan>'
                f'<tspan fill="{COLORS["coral"]}">{frames:.1f}</tspan>'
                '<tspan class="summary-unit">%</tspan></text>',
                f'<text x="{x:.1f}" y="263" class="summary-label">{label}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="2359" y="216" class="summary-order" text-anchor="end">H / F</text>',
            '<line x1="41" y1="284" x2="2359" y2="284" class="summary-divider"/>',
            "</g>",
        ]
    )
    return "".join(parts)


def scene_pair_text(
    x: str, y: str, anchor: str, hours: float, frames: float
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="scene-pair">'
        f'<tspan class="scene-h">H</tspan><tspan> {hours:.1f}</tspan>'
        '<tspan class="scene-dot"> · </tspan>'
        f'<tspan class="scene-f">F</tspan><tspan> {frames:.1f}</tspan>'
        "</text>"
    )


def replace_scene_shares(svg: str, scene_rows: list[dict[str, str]]) -> str:
    specs = [
        ("238", "424", "end"),
        ("2167", "581", "start"),
        ("238", "728", "end"),
        ("2167", "875", "start"),
        ("238", "1022", "end"),
    ]
    for row, (x, y, anchor) in zip(scene_rows, specs):
        hours = float(row["video_hours_share_pct"])
        frames = float(row["sampled_frames_share_pct"])
        old_pattern = (
            rf'<text x="{x}" y="{y}" text-anchor="{anchor}" class="share">'
            r"[^<]+</text>"
        )
        svg, count = re.subn(
            old_pattern,
            scene_pair_text(x, y, anchor, hours, frames),
            svg,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"Could not replace scene share at {x},{y}")
    return svg


def build_svg(
    original_svg: str,
    dataset_rows: list[dict[str, str]],
    scene_rows: list[dict[str, str]],
) -> str:
    svg, count = re.subn(
        r'width="7\.0in" height="4\.083333in"',
        f'width="{PAGE_WIDTH_IN:.6f}in" height="{PAGE_HEIGHT_IN:.6f}in"',
        original_svg,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update physical dimensions")
    svg = svg.replace(
        'viewBox="0 0 2400 1400"', f'viewBox="0 0 {CANVAS_W} {CANVAS_H}"', 1
    )
    svg = svg.replace(
        '<title id="title">NavAnywhere: navigation diversity along a visual route</title>',
        '<title id="title">NavAnywhere overview and corpus composition</title>',
        1,
    )
    svg = svg.replace(
        '<desc id="desc">Thirty-five authentic NavAnywhere frames form a continuous route through five scene categories, followed by twelve separate out-of-domain evaluation frames.</desc>',
        '<desc id="desc">The original 47-frame NavAnywhere overview with a typographic source summary and paired scene-composition annotations.</desc>',
        1,
    )
    svg = svg.replace(
        '<rect width="2400" height="1400" fill="#FCFCFA"/>',
        f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#FCFCFA"/>',
        1,
    )
    svg = svg.replace(
        '<g fill="none" stroke="#E9EFEE" stroke-width="1.5" opacity="0.8">',
        f'<g transform="translate(0 {CONTENT_SHIFT_Y})" fill="none" stroke="#E9EFEE" stroke-width="1.5" opacity="0.8">',
        1,
    )
    svg = replace_scene_shares(svg, scene_rows)

    extra_style = f"""
    .summary-heading {{ font-size: 18px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: 1.8px; }}
    .summary-key {{ font-size: 15px; font-weight: 600; fill: {COLORS['muted']}; }}
    .summary-number {{ font-size: 33px; font-weight: 700; letter-spacing: -0.4px; }}
    .summary-slash {{ fill: {COLORS['faint']}; font-weight: 500; }}
    .summary-unit {{ fill: {COLORS['muted']}; font-size: 19px; }}
    .summary-label {{ font-size: 16px; font-weight: 600; fill: {COLORS['muted']}; }}
    .summary-order {{ font-size: 14px; font-weight: 700; fill: {COLORS['faint']}; letter-spacing: 1.2px; }}
    .summary-divider {{ stroke: {COLORS['line']}; stroke-width: 1.4; }}
    .scene-pair {{ font-size: 17px; font-weight: 650; fill: {COLORS['muted']}; }}
    .scene-h {{ fill: {COLORS['teal']}; font-weight: 800; }}
    .scene-f {{ fill: {COLORS['coral']}; font-weight: 800; }}
    .scene-dot {{ fill: {COLORS['faint']}; }}
"""
    svg = svg.replace("  </style>", extra_style + "  </style>", 1)

    start_token = '  <text x="41" y="210" class="section">EMBODIMENT</text>'
    if svg.count(start_token) != 1:
        raise RuntimeError("Could not locate overview body")
    summary = source_summary_markup(dataset_rows)
    svg = svg.replace(
        start_token,
        summary
        + f'\n  <g id="shifted-overview-body" transform="translate(0 {CONTENT_SHIFT_Y})">\n'
        + start_token,
        1,
    )
    if svg.count("</svg>") != 1:
        raise RuntimeError("Unexpected SVG closing tag count")
    svg = svg.replace("</svg>", "  </g>\n</svg>", 1)
    return svg


def build_html(svg_name: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {PAGE_WIDTH_IN:.6f}in {PAGE_HEIGHT_IN:.6f}in; margin: 0; }}
html,body {{ margin:0; padding:0; width:{PAGE_WIDTH_IN:.6f}in; height:{PAGE_HEIGHT_IN:.6f}in; overflow:hidden; background:#FCFCFA; }}
img {{ display:block; width:{PAGE_WIDTH_IN:.6f}in; height:{PAGE_HEIGHT_IN:.6f}in; }}
</style></head><body><img src="{svg_name}" alt="NavAnywhere overview and corpus composition"></body></html>
"""


def build_caption() -> str:
    return rf"""\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/{BASE_NAME}_macos.pdf}}
  \caption{{\textbf{{Overview and composition of NavAnywhere.}}
  The source summary reports video-hour (H) and sampled-frame (F) shares for
  the dominant source, the next three sources, and the remaining long tail.
  Thirty-five representative frames follow a visual route through five scene
  categories; the paired H/F shares are printed beside the corresponding row.
  Twelve additional frames show four out-of-domain evaluation datasets.}}
  \label{{fig:navanywhere-overview-editorial}}
\end{{figure*}}
"""


def validate_embedded_images(source: Path, output: Path) -> None:
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
    if len(output_hrefs) != 47 or source_hrefs != output_hrefs:
        raise RuntimeError("Embedded image payloads changed")


def write_provenance(output_dir: Path) -> None:
    images = extract_images(OVERVIEW_SVG)
    manifest = {
        "artifact": BASE_NAME,
        "design": "editorial source summary and direct scene annotations",
        "canvas": {
            "viewbox": [0, 0, CANVAS_W, CANVAS_H],
            "physical_size_inches": [PAGE_WIDTH_IN, PAGE_HEIGHT_IN],
        },
        "image_layout": "overview v3 positions and dimensions shifted down 100 units",
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
    if len(images) != 47:
        raise RuntimeError("Expected 47 images")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{BASE_NAME}.svg"
    html_path = output_dir / f"{BASE_NAME}_print.html"
    svg_path.write_text(
        build_svg(OVERVIEW_SVG.read_text(encoding="utf-8"), dataset_rows, scene_rows),
        encoding="utf-8",
    )
    validate_embedded_images(OVERVIEW_SVG, svg_path)
    html_path.write_text(build_html(svg_path.name), encoding="utf-8")
    (output_dir / "caption.tex").write_text(build_caption(), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Editorial NavAnywhere overview\n\n"
        "This version preserves the overview-v3 image layout and all 47 embedded "
        "JPEG payloads. Statistics are integrated as typography and direct scene "
        "annotations; there are no bar-chart panels.\n",
        encoding="utf-8",
    )
    shutil.copy2(DATASET_CSV, output_dir / "dataset_composition.csv")
    shutil.copy2(SCENE_CSV, output_dir / "scene_composition.csv")
    write_provenance(output_dir)
    print(f"Wrote {svg_path}")
    print("Embedded images: 47 (payloads identical)")
    print(f"Page: {PAGE_WIDTH_IN:.4f} x {PAGE_HEIGHT_IN:.4f} in")


if __name__ == "__main__":
    main()
