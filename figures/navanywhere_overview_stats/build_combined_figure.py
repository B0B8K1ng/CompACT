#!/usr/bin/env python3
"""Build the combined NavAnywhere overview and corpus statistics figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
from pathlib import Path


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
CONFIRMED_DATASET_REFERENCE = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/generated_images/"
    "01a0a34a-6b7a-7220-a28d-ee2f3139d439/"
    "exec-3a435b7e-f239-4d87-8d20-d120d0719976.png"
)
CONFIRMED_SCENE_REFERENCE = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/generated_images/"
    "01a0a34a-6b7a-7220-a28d-ee2f3139d439/"
    "exec-76dbbbf8-4b65-4ff7-887d-8dae22c38ac7.png"
)

DEFAULT_OUTPUT_DIR = Path(
    "/file_system/vepfs/algorithm/dujun.nie/.codex/visualizations/2026/09/22/"
    "01a0c742-f9c0-7ab1-9bd0-dca2be5385fa/"
    "navanywhere_overview_stats_combined_20260922"
)

BASE_NAME = "navanywhere_overview_with_statistics_20260922"
CANVAS_W = 2400
OVERVIEW_H = 1400
CANVAS_H = 2250
PAGE_WIDTH_IN = 7.0
PAGE_HEIGHT_IN = PAGE_WIDTH_IN * CANVAS_H / CANVAS_W

COLORS = {
    "paper": "#FCFCFA",
    "card": "#FFFFFF",
    "ink": "#17343D",
    "muted": "#607379",
    "faint": "#8B9A9E",
    "line": "#D6E0DF",
    "line_soft": "#E8EEED",
    "grid_minor": "#F0F3F2",
    "teal": "#37847D",
    "coral": "#C97962",
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


def log_x(value: float, x0: float, x1: float) -> float:
    if value <= 0:
        raise ValueError(f"Log-scale value must be positive, got {value}")
    lo = math.log10(0.01)
    hi = math.log10(100.0)
    return x0 + (math.log10(value) - lo) / (hi - lo) * (x1 - x0)


def fmt_source_share(value: float) -> str:
    return f"{value:.2f}%" if value >= 0.1 else f"{value:.3f}%"


def validate_data(
    dataset_rows: list[dict[str, str]], scene_rows: list[dict[str, str]]
) -> None:
    if len(dataset_rows) != 15:
        raise RuntimeError(f"Expected 15 formal sources, found {len(dataset_rows)}")
    if len(scene_rows) != 5:
        raise RuntimeError(f"Expected five scene categories, found {len(scene_rows)}")
    dataset_names = {row["dataset"] for row in dataset_rows}
    if "CASIA-Nav" not in dataset_names:
        raise RuntimeError("Expected CASIA-Nav source before display anonymization")
    scene_names = [row["category"] for row in scene_rows]
    if scene_names != list(SCENE_COLORS):
        raise RuntimeError(f"Unexpected scene order: {scene_names}")
    for field in ("video_hours_share_pct", "sampled_frames_share_pct"):
        dataset_sum = sum(float(row[field]) for row in dataset_rows)
        scene_sum = sum(float(row[field]) for row in scene_rows)
        if abs(dataset_sum - 100.0) > 0.02:
            raise RuntimeError(f"Dataset shares for {field} sum to {dataset_sum}")
        if abs(scene_sum - 100.0) > 0.02:
            raise RuntimeError(f"Scene shares for {field} sum to {scene_sum}")


def source_chart_markup(dataset_rows: list[dict[str, str]]) -> str:
    rows = sorted(
        dataset_rows,
        key=lambda row: float(row["video_hours_share_pct"]),
        reverse=True,
    )
    display_names = {
        "CASIA-Nav": "In-house Collected",
    }
    label_x = 262.0
    plot_specs = [
        (292.0, 806.0, "video_hours_share_pct", COLORS["teal"]),
        (947.0, 1461.0, "sampled_frames_share_pct", COLORS["coral"]),
    ]
    row_top = 1592.0
    row_gap = 34.0
    axis_y = 2093.0
    chart_top = row_top - 19.0
    parts: list[str] = []

    major_ticks = [0.01, 0.1, 1.0, 10.0, 100.0]
    minor_ticks: list[float] = []
    for exponent in (-2, -1, 0, 1):
        minor_ticks.extend(multiplier * 10**exponent for multiplier in range(2, 10))

    for x0, x1, _, _ in plot_specs:
        for value in minor_ticks:
            x = log_x(value, x0, x1)
            parts.append(
                f'<line x1="{x:.2f}" y1="{chart_top:.2f}" x2="{x:.2f}" '
                f'y2="{axis_y:.2f}" class="stats-grid-minor"/>'
            )
        for value in major_ticks:
            x = log_x(value, x0, x1)
            parts.append(
                f'<line x1="{x:.2f}" y1="{chart_top:.2f}" x2="{x:.2f}" '
                f'y2="{axis_y:.2f}" class="stats-grid-major"/>'
            )

    for index, row in enumerate(rows):
        y = row_top + index * row_gap
        name = display_names.get(row["dataset"], row["dataset"])
        parts.append(
            f'<text x="{label_x:.2f}" y="{y:.2f}" class="stats-source-label" '
            f'text-anchor="end" dominant-baseline="middle">{esc(name)}</text>'
        )
        for x0, x1, field, color in plot_specs:
            value = float(row[field])
            x = log_x(value, x0, x1)
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="3.1" stroke-linecap="round" opacity="0.72"/>'
            )
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7.2" fill="{color}" '
                f'stroke="#FFFFFF" stroke-width="2.1"/>'
            )
            if x > x1 - 64:
                text_x = x - 12
                anchor = "end"
            else:
                text_x = x + 12
                anchor = "start"
            parts.append(
                f'<text x="{text_x:.2f}" y="{y:.2f}" class="stats-value" '
                f'text-anchor="{anchor}" dominant-baseline="middle">'
                f'{fmt_source_share(value)}</text>'
            )

    tick_labels = ["0.01", "0.1", "1", "10", "100"]
    for x0, x1, _, _ in plot_specs:
        parts.append(
            f'<line x1="{x0:.2f}" y1="{axis_y:.2f}" x2="{x1:.2f}" '
            f'y2="{axis_y:.2f}" class="stats-axis"/>'
        )
        for value, tick_label in zip(major_ticks, tick_labels):
            x = log_x(value, x0, x1)
            parts.append(
                f'<line x1="{x:.2f}" y1="{axis_y:.2f}" x2="{x:.2f}" '
                f'y2="{axis_y + 8:.2f}" class="stats-axis"/>'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{axis_y + 31:.2f}" class="stats-tick" '
                f'text-anchor="middle">{tick_label}</text>'
            )

    return "".join(parts)


def scene_chart_markup(scene_rows: list[dict[str, str]]) -> str:
    x0 = 1743.0
    x1 = 2348.0
    width = x1 - x0
    bar_height = 76.0
    bar_centers = [1778.0, 1958.0]
    row_fields = ["video_hours_share_pct", "sampled_frames_share_pct"]
    row_labels = ["Video hours", "Sampled RGB frames"]
    axis_y = 2041.0
    parts: list[str] = [
        '<defs>'
        f'<clipPath id="stats-scene-hours"><rect x="{x0}" y="{bar_centers[0] - bar_height / 2}" '
        f'width="{width}" height="{bar_height}" rx="9"/></clipPath>'
        f'<clipPath id="stats-scene-frames"><rect x="{x0}" y="{bar_centers[1] - bar_height / 2}" '
        f'width="{width}" height="{bar_height}" rx="9"/></clipPath>'
        '</defs>'
    ]

    legend_positions = [
        (1568.0, 1556.0),
        (1837.0, 1556.0),
        (2110.0, 1556.0),
        (1568.0, 1598.0),
        (1878.0, 1598.0),
    ]
    for (category, color), (x, y) in zip(SCENE_COLORS.items(), legend_positions):
        parts.append(
            f'<rect x="{x:.2f}" y="{y - 12:.2f}" width="24" height="15" rx="3" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + 34:.2f}" y="{y:.2f}" class="stats-legend">'
            f'{esc(category)}</text>'
        )

    for tick in range(0, 101, 20):
        x = x0 + width * tick / 100.0
        parts.append(
            f'<line x1="{x:.2f}" y1="1688" x2="{x:.2f}" y2="{axis_y:.2f}" '
            f'class="stats-grid-major"/>'
        )

    clip_ids = ["stats-scene-hours", "stats-scene-frames"]
    for center_y, field, row_label, clip_id in zip(
        bar_centers, row_fields, row_labels, clip_ids
    ):
        parts.append(
            f'<text x="1718" y="{center_y:.2f}" class="stats-row-label" '
            f'text-anchor="end" dominant-baseline="middle">{esc(row_label)}</text>'
        )
        left = x0
        for row in scene_rows:
            category = row["category"]
            value = float(row[field])
            segment_width = width * value / 100.0
            color = SCENE_COLORS[category]
            parts.append(
                f'<rect x="{left:.2f}" y="{center_y - bar_height / 2:.2f}" '
                f'width="{segment_width + 0.25:.2f}" height="{bar_height:.2f}" '
                f'fill="{color}" clip-path="url(#{clip_id})"/>'
            )
            parts.append(
                f'<text x="{left + segment_width / 2:.2f}" y="{center_y:.2f}" '
                f'class="stats-scene-value" text-anchor="middle" '
                f'dominant-baseline="middle">{value:.1f}%</text>'
            )
            left += segment_width
        parts.append(
            f'<rect x="{x0:.2f}" y="{center_y - bar_height / 2:.2f}" '
            f'width="{width:.2f}" height="{bar_height:.2f}" rx="9" fill="none" '
            f'stroke="#FFFFFF" stroke-width="2"/>'
        )

    parts.append(
        f'<line x1="{x0:.2f}" y1="{axis_y:.2f}" x2="{x1:.2f}" '
        f'y2="{axis_y:.2f}" class="stats-axis"/>'
    )
    for tick in range(0, 101, 20):
        x = x0 + width * tick / 100.0
        parts.append(
            f'<line x1="{x:.2f}" y1="{axis_y:.2f}" x2="{x:.2f}" '
            f'y2="{axis_y + 8:.2f}" class="stats-axis"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{axis_y + 34:.2f}" class="stats-tick" '
            f'text-anchor="middle">{tick}</text>'
        )
    return "".join(parts)


def statistics_markup(
    dataset_rows: list[dict[str, str]], scene_rows: list[dict[str, str]]
) -> str:
    return f"""
  <style>
    .stats-panel-title {{ font-size: 25px; font-weight: 700; fill: {COLORS['ink']}; letter-spacing: 2.2px; }}
    .stats-panel-subtitle {{ font-size: 18.5px; font-weight: 500; fill: {COLORS['muted']}; }}
    .stats-subhead {{ font-size: 24px; font-weight: 700; }}
    .stats-source-label {{ font-size: 21px; font-weight: 500; fill: {COLORS['ink']}; }}
    .stats-value {{ font-size: 18.5px; font-weight: 600; fill: {COLORS['ink']}; paint-order: stroke; stroke: #FFFFFF; stroke-width: 5px; stroke-linejoin: round; }}
    .stats-tick {{ font-size: 18px; font-weight: 500; fill: {COLORS['muted']}; }}
    .stats-axis-label {{ font-size: 20px; font-weight: 600; fill: {COLORS['ink']}; }}
    .stats-axis {{ stroke: {COLORS['ink']}; stroke-width: 1.5; }}
    .stats-grid-major {{ stroke: {COLORS['line']}; stroke-width: 1.15; }}
    .stats-grid-minor {{ stroke: {COLORS['grid_minor']}; stroke-width: 0.8; }}
    .stats-legend {{ font-size: 18px; font-weight: 600; fill: {COLORS['ink']}; }}
    .stats-row-label {{ font-size: 19.5px; font-weight: 600; fill: {COLORS['ink']}; }}
    .stats-scene-value {{ font-size: 19px; font-weight: 700; fill: #FFFFFF; paint-order: stroke; stroke: rgba(23,52,61,0.20); stroke-width: 1.2px; }}
  </style>
  <g id="corpus-statistics" aria-label="NavAnywhere corpus composition statistics" shape-rendering="geometricPrecision">
    <rect x="0" y="{OVERVIEW_H}" width="{CANVAS_W}" height="{CANVAS_H - OVERVIEW_H}" fill="{COLORS['paper']}"/>
    <line x1="41" y1="1417" x2="2359" y2="1417" stroke="{COLORS['line']}" stroke-width="1.5"/>
    <rect x="24" y="1438" width="1490" height="772" rx="18" fill="{COLORS['card']}" stroke="{COLORS['line_soft']}" stroke-width="1.5"/>
    <rect x="1534" y="1438" width="842" height="772" rx="18" fill="{COLORS['card']}" stroke="{COLORS['line_soft']}" stroke-width="1.5"/>

    <text x="49" y="1482" class="stats-panel-title">DATASET COMPOSITION</text>
    <text x="49" y="1514" class="stats-panel-subtitle">Share of total corpus by formal source · log scale</text>
    <text x="549" y="1554" class="stats-subhead" text-anchor="middle" fill="{COLORS['teal']}">Video hours</text>
    <text x="1204" y="1554" class="stats-subhead" text-anchor="middle" fill="{COLORS['coral']}">Sampled RGB frames</text>
    {source_chart_markup(dataset_rows)}
    <text x="876" y="2171" class="stats-axis-label" text-anchor="middle">Share of NavAnywhere (%)</text>

    <text x="1559" y="1482" class="stats-panel-title">SCENE COMPOSITION</text>
    <text x="1559" y="1514" class="stats-panel-subtitle">Estimated share by scene category</text>
    {scene_chart_markup(scene_rows)}
    <text x="2046" y="2128" class="stats-axis-label" text-anchor="middle">Share of NavAnywhere (%)</text>
  </g>
"""


def build_svg(
    overview_svg: str,
    dataset_rows: list[dict[str, str]],
    scene_rows: list[dict[str, str]],
) -> str:
    updated, count = re.subn(
        r'width="[0-9.]+in"\s+height="[0-9.]+in"',
        f'width="{PAGE_WIDTH_IN:.6f}in" height="{PAGE_HEIGHT_IN:.6f}in"',
        overview_svg,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update SVG physical dimensions")
    updated, count = re.subn(
        rf'viewBox="0 0 {CANVAS_W} {OVERVIEW_H}"',
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}"',
        updated,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update SVG viewBox")
    updated, count = re.subn(
        r'<title id="title">.*?</title>',
        '<title id="title">NavAnywhere overview and corpus composition</title>',
        updated,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update SVG title")
    updated, count = re.subn(
        r'<desc id="desc">.*?</desc>',
        '<desc id="desc">NavAnywhere route overview with source and scene composition statistics.</desc>',
        updated,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update SVG description")
    markup = statistics_markup(dataset_rows, scene_rows)
    if updated.count("</svg>") != 1:
        raise RuntimeError("Unexpected SVG closing-tag count")
    return updated.replace("</svg>", f"{markup}\n</svg>")


def build_html(svg_name: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page {{ size: {PAGE_WIDTH_IN:.6f}in {PAGE_HEIGHT_IN:.6f}in; margin: 0; }}
    html, body {{
      margin: 0; padding: 0; width: {PAGE_WIDTH_IN:.6f}in;
      height: {PAGE_HEIGHT_IN:.6f}in; overflow: hidden; background: {COLORS['paper']};
    }}
    img {{ display: block; width: {PAGE_WIDTH_IN:.6f}in; height: {PAGE_HEIGHT_IN:.6f}in; }}
  </style>
</head>
<body><img src="{esc(svg_name)}" alt="NavAnywhere overview with corpus statistics"></body>
</html>
"""


def build_caption() -> str:
    return rf"""\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/{BASE_NAME}_macos.pdf}}
  \caption{{\textbf{{Overview and composition of NavAnywhere.}}
  The upper panel summarizes the scale and diversity of the corpus using
  representative frames from its 15 formal sources and four independent
  out-of-domain evaluation datasets. The lower-left panel reports each formal
  source's share of total video hours and sampled RGB frames on logarithmic
  axes. The lower-right panel reports the estimated scene-category composition
  by video duration and sampled frames.}}
  \label{{fig:navanywhere-overview-statistics}}
\end{{figure*}}
"""


def build_readme() -> str:
    return f"""# NavAnywhere overview with corpus statistics

This package combines the final NavAnywhere overview v3 with two confirmed
statistics panels. The statistics are redrawn from the audited CSV values so
that typography, line weight, spacing, and colors match the overview.

The display-only edits requested for the source charts are preserved:

- `Main-view video hours` is shown as `Video hours`.
- `CASIA-Nav` is shown as `In-house Collected`.
- Standalone chart titles and explanatory footnotes are omitted.

## Recommended files

- `{BASE_NAME}_macos.pdf`: PDF 1.4 compatibility copy for Preview.app.
- `{BASE_NAME}_600dpi.png`: 600 dpi RGB fallback rendered from that PDF.
- `{BASE_NAME}.svg`: self-contained editable vector source.
- `{BASE_NAME}_bundle.zip`: all deliverables, source snapshots, provenance,
  checksums, and rebuild scripts.

## Integrity check on macOS

```bash
unzip -t {BASE_NAME}_bundle.zip
shasum -a 256 -c SHA256SUMS.txt
```

The two `confirmed_*_reference.png` files are the exact images approved before
composition. They are included for provenance; the combined figure uses the
CSV values rather than embedding those PNGs.
"""


def write_provenance(output_dir: Path) -> None:
    inputs = {
        "overview_v3_svg": OVERVIEW_SVG,
        "dataset_composition_csv": DATASET_CSV,
        "scene_composition_csv": SCENE_CSV,
        "confirmed_dataset_reference": CONFIRMED_DATASET_REFERENCE,
        "confirmed_scene_reference": CONFIRMED_SCENE_REFERENCE,
    }
    manifest = {
        "artifact": BASE_NAME,
        "canvas": {
            "viewbox": [0, 0, CANVAS_W, CANVAS_H],
            "physical_size_inches": [PAGE_WIDTH_IN, PAGE_HEIGHT_IN],
        },
        "display_edits": {
            "Main-view video hours": "Video hours",
            "CASIA-Nav": "In-house Collected",
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in inputs.items()
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    required = [
        OVERVIEW_SVG,
        DATASET_CSV,
        SCENE_CSV,
        CONFIRMED_DATASET_REFERENCE,
        CONFIRMED_SCENE_REFERENCE,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    dataset_rows = read_csv(DATASET_CSV)
    scene_rows = read_csv(SCENE_CSV)
    validate_data(dataset_rows, scene_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{BASE_NAME}.svg"
    html_path = output_dir / f"{BASE_NAME}_print.html"

    svg = build_svg(
        OVERVIEW_SVG.read_text(encoding="utf-8"), dataset_rows, scene_rows
    )
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(build_html(svg_path.name), encoding="utf-8")
    (output_dir / "caption.tex").write_text(build_caption(), encoding="utf-8")
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")
    shutil.copy2(DATASET_CSV, output_dir / "dataset_composition.csv")
    shutil.copy2(SCENE_CSV, output_dir / "scene_composition.csv")
    shutil.copy2(
        CONFIRMED_DATASET_REFERENCE,
        output_dir / "confirmed_dataset_composition_reference.png",
    )
    shutil.copy2(
        CONFIRMED_SCENE_REFERENCE,
        output_dir / "confirmed_scene_composition_reference.png",
    )
    write_provenance(output_dir)

    print(f"Wrote {svg_path}")
    print(f"Wrote {html_path}")
    print(f"Page: {PAGE_WIDTH_IN:.4f} x {PAGE_HEIGHT_IN:.4f} in")


if __name__ == "__main__":
    main()
