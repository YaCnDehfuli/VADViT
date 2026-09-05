#!/usr/bin/env python3
"""Generate docs/figures/architecture.svg for the VADViT README.

One-row pipeline, mid-tone fills (no pure white or black), readable at 800px
in GitHub light and dark themes. Stdlib only.

    python docs/figures/generate_architecture.py
"""

from pathlib import Path

OUT = Path(__file__).with_name("architecture.svg")

# Neutral mid-tone palette. Avoid #000 and #fff fills.
INK = "#2f3a42"
LINE = "#5a6570"
TEXT = "#f3efe6"
MUTED = "#e4ddd0"
BG = "#7d8289"
FILLS = [
    "#4f6574",  # dump
    "#4f6d62",  # vadinfo
    "#5c5d74",  # vad files
    "#6a5a4a",  # channels
    "#5a4f6a",  # rgb
    "#4a5f6c",  # grid
    "#4d5f52",  # vit
    "#6a5348",  # heads
    "#5a4e62",  # attention
]


def main():
    # Explicit lines so the row stays readable when GitHub scales the SVG to ~800px.
    stages = [
        ["Windows memory", "dump"],
        ["Volatility 3", "windows.vadinfo", "PID-targeted"],
        ["VAD region files", "sorted by", "address"],
        ["Three channels", "intensity · entropy", "· Markov"],
        ["Fused RGB", "patches"],
        ["Process-level", "grid image"],
        ["Fine-tuned ViT", "via timm"],
        ["Binary + family", "heads"],
        ["Attention-ranked", "regions"],
    ]

    width, height = 1180, 268
    pad_x, pad_y = 12, 36
    n = len(stages)
    gap = 14
    box_w = (width - 2 * pad_x - (n - 1) * gap) / n
    box_h = 132
    box_y = 52

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="VADViT pipeline from Windows memory dump to attention-ranked VAD regions">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}"/>',
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
        f'font-family="ui-sans-serif, Helvetica, Arial, sans-serif" '
        f'font-size="15" font-weight="700" fill="{TEXT}">VADViT pipeline</text>',
    ]

    xs = []
    for i, label in enumerate(stages):
        x = pad_x + i * (box_w + gap)
        xs.append(x)
        fill = FILLS[i]
        parts.append(
            f'<rect x="{x:.1f}" y="{box_y}" width="{box_w:.1f}" height="{box_h}" '
            f'rx="8" ry="8" fill="{fill}" stroke="{INK}" stroke-width="1.2"/>'
        )
        lines = label
        line_h = 15
        block_h = len(lines) * line_h
        start = box_y + (box_h - block_h) / 2 + 11
        for j, line in enumerate(lines):
            ty = start + j * line_h
            parts.append(
                f'<text x="{x + box_w/2:.1f}" y="{ty:.1f}" text-anchor="middle" '
                f'font-family="ui-sans-serif, Helvetica, Arial, sans-serif" '
                f'font-size="11" fill="{TEXT}">{_esc(line)}</text>'
            )
        if i < n - 1:
            x1 = x + box_w
            x2 = x + box_w + gap
            mid = box_y + box_h / 2
            parts.append(
                f'<line x1="{x1:.1f}" y1="{mid:.1f}" x2="{x2 - 7:.1f}" y2="{mid:.1f}" '
                f'stroke="{LINE}" stroke-width="2"/>'
            )
            parts.append(
                f'<polygon points="{x2-1:.1f},{mid:.1f} {x2-8:.1f},{mid-5:.1f} '
                f'{x2-8:.1f},{mid+5:.1f}" fill="{LINE}"/>'
            )

    key_y = box_y + box_h + 22
    parts.append(
        f'<text x="{width/2:.1f}" y="{key_y:.1f}" text-anchor="middle" '
        f'font-family="ui-sans-serif, Helvetica, Arial, sans-serif" '
        f'font-size="11" fill="{MUTED}">'
        "R: VAD-feature intensity · G: windowed Shannon entropy · "
        "B: Markov byte-transition</text>"
    )

    parts.append("</svg>")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


def _esc(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    main()
