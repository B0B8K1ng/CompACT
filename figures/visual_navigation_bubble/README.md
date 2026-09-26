# Visual prediction and navigation bubble plot

Source: the active (uncommented) main-results table supplied by the user on
2026-09-26. The CSV transcribes NWM, RAE-NWM, and OpenNWM from that table.

- Horizontal coordinate: reported four-dataset ID average PSNR at 4 s
  (RECON, SCAND, HuRoN, TartanDrive).
- Vertical coordinate: unweighted arithmetic mean of ATE on RECON, SCAND,
  and Office Go2. No rounding is applied before plotting.
- Both axes use their standard direction: lower right indicates higher PSNR
  and lower navigation error.
- Circle area is proportional to parameters: 279M, 867M, and 280M.
- Linear axes; no data-coordinate offsets or nonlinear scaling.

| Method | ID PSNR (dB) | Mean navigation ATE | Parameters |
| --- | ---: | ---: | ---: |
| NWM | 13.811 | 1.723333333 | 279M |
| RAE-NWM | 13.957 | 1.836666667 | 867M |
| OpenNWM | 14.299 | 1.663333333 | 280M |

The table does not report navigation for NWM CDiT-XL + Ego4D or visual
prediction for GNM, NoMaD, and CompACT. These methods therefore cannot be
placed on the same two-metric plot without inventing a missing coordinate.

Run from the repository root:

```sh
conda run -n nwm python figures/visual_navigation_bubble/plot.py
```

Exports PDF with embedded TrueType fonts, editable-text SVG, and 450 dpi PNG.

Suggested paper caption:

Visual prediction and navigation performance. PSNR is averaged over the
four ID datasets at a 4 s horizon, and ATE is averaged over RECON, SCAND, and
Office Go2. The lower right is better. Bubble area
denotes world-model parameter count.
