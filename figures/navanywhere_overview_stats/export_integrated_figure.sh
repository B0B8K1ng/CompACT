#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_dir=/file_system/vepfs/algorithm/dujun.nie/.codex/visualizations/2026/09/22/01a0c742-f9c0-7ab1-9bd0-dca2be5385fa/navanywhere_overview_integrated_20260922
python_bin=/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/python
chrome_bin=/opt/google/chrome/google-chrome
base=navanywhere_overview_integrated_20260922

mkdir -p "$output_dir"

"$python_bin" "$script_dir/build_integrated_figure.py" --output-dir "$output_dir"
cp "$script_dir/build_integrated_figure.py" "$output_dir/build_integrated_figure.py"
cp "$script_dir/export_integrated_figure.sh" "$output_dir/export_integrated_figure.sh"

"$chrome_bin" \
  --headless=new \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --allow-file-access-from-files \
  --run-all-compositor-stages-before-draw \
  --no-pdf-header-footer \
  --print-to-pdf="$output_dir/$base.pdf" \
  "file://$output_dir/${base}_print.html" \
  2>"$output_dir/chrome_pdf.log"

gs -q -dBATCH -dNOPAUSE -dSAFER \
  -sDEVICE=pdfwrite \
  -dCompatibilityLevel=1.4 \
  -dPDFSETTINGS=/prepress \
  -dAutoRotatePages=/None \
  -dEmbedAllFonts=true \
  -dSubsetFonts=true \
  -dDetectDuplicateImages=true \
  -sOutputFile="$output_dir/${base}_macos.pdf" \
  "$output_dir/$base.pdf"

gs -q -dBATCH -dNOPAUSE -dSAFER \
  -sDEVICE=png16m -r600 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
  -sOutputFile="$output_dir/${base}_600dpi.png" \
  "$output_dir/${base}_macos.pdf"

gs -q -dBATCH -dNOPAUSE -dSAFER \
  -sDEVICE=png16m -r300 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
  -sOutputFile="$output_dir/${base}_macos_300dpi.png" \
  "$output_dir/${base}_macos.pdf"

gs -q -dBATCH -dNOPAUSE -dSAFER \
  -sDEVICE=png16m -r171.428571 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
  -sOutputFile="$output_dir/${base}_preview.png" \
  "$output_dir/${base}_macos.pdf"

(
  cd "$output_dir"
  sha256sum \
    "$base.svg" \
    "$base.pdf" \
    "${base}_macos.pdf" \
    "${base}_600dpi.png" \
    "${base}_macos_300dpi.png" \
    "${base}_preview.png" \
    "${base}_print.html" \
    README.md caption.tex provenance.json \
    dataset_composition.csv scene_composition.csv \
    build_integrated_figure.py export_integrated_figure.sh \
    > SHA256SUMS.txt
)

zip -j -9 -q \
  "$output_dir/${base}_bundle.zip" \
  "$output_dir/$base.svg" \
  "$output_dir/$base.pdf" \
  "$output_dir/${base}_macos.pdf" \
  "$output_dir/${base}_600dpi.png" \
  "$output_dir/${base}_macos_300dpi.png" \
  "$output_dir/${base}_preview.png" \
  "$output_dir/${base}_print.html" \
  "$output_dir/SHA256SUMS.txt" \
  "$output_dir/README.md" \
  "$output_dir/caption.tex" \
  "$output_dir/provenance.json" \
  "$output_dir/dataset_composition.csv" \
  "$output_dir/scene_composition.csv" \
  "$output_dir/build_integrated_figure.py" \
  "$output_dir/export_integrated_figure.sh"

unzip -t "$output_dir/${base}_bundle.zip" >"$output_dir/zip_test.log"

ls -lh \
  "$output_dir/$base.svg" \
  "$output_dir/$base.pdf" \
  "$output_dir/${base}_macos.pdf" \
  "$output_dir/${base}_600dpi.png" \
  "$output_dir/${base}_macos_300dpi.png" \
  "$output_dir/${base}_preview.png" \
  "$output_dir/${base}_bundle.zip"
