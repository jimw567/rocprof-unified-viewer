#!/usr/bin/env python3
"""Annotate docs/fusion-analysis.png, focused on the Fusion Analysis panel."""
from PIL import Image, ImageDraw, ImageFont

SRC = "docs/fusion-analysis.png"
DST = "docs/fusion-analysis-annotated.png"

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
def font(name, sz):
    return ImageFont.truetype(FONT_DIR + name, sz)

f_badge   = font("DejaVuSans-Bold.ttf", 18)
f_zone    = font("DejaVuSans-Bold.ttf", 16)
f_legend  = font("DejaVuSans.ttf", 15)
f_legendb = font("DejaVuSans-Bold.ttf", 15)
f_title   = font("DejaVuSans-Bold.ttf", 20)

base = Image.open(SRC).convert("RGBA")
W, H = base.size

MARGIN = 300  # bottom strip for the legend
canvas = Image.new("RGBA", (W, H + MARGIN), (18, 18, 22, 255))
canvas.paste(base, (0, 0))
ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
d = ImageDraw.Draw(ov)

ACCENT = (255, 210, 40, 255)
ZONE   = (80, 200, 255, 255)

def zone(x0, y0, x1, y1, label, col=ZONE):
    d.rectangle([x0, y0, x1, y1], outline=col, width=3)
    tw = d.textlength(label, font=f_zone)
    pad = 6
    ty = y0 - 24 if y0 - 24 > 0 else y0 + 4
    d.rectangle([x0, ty, x0 + tw + 2 * pad, ty + 22], fill=(10, 10, 12, 235))
    d.text((x0 + pad, ty + 3), label, font=f_zone, fill=col)

def badge(n, bx, by, tx, ty):
    r = 15
    d.line([bx, by, tx, ty], fill=ACCENT, width=3)
    d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=ACCENT)
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=(10, 10, 12, 255),
              outline=ACCENT, width=3)
    s = str(n)
    tw = d.textlength(s, font=f_badge)
    d.text((bx - tw / 2, by - 11), s, font=f_badge, fill=ACCENT)

# ---- zone boxes ----
zone(306, 232, 726, 528, "FUSION ANALYSIS")
zone(10, 232, 296, 370, "SELECTION")

# ---- numbered badges pointing at each row ----
# (n, badge_x, badge_y, target_x, target_y)
badge(1, 1055, 50, 996, 50)     # easy selection -> the 3 highlighted kernels
badge(2, 152, 205, 152, 233)    # SELECTION panel (from lasso)
badge(3, 780, 268, 470, 268)    # span (wall)
badge(4, 810, 291, 470, 291)    # busy (sum kernels)
badge(5, 780, 314, 645, 313)    # inter-kernel idle (reclaimable)
badge(6, 810, 386, 600, 386)    # per-family VGPR/LDS/scratch/occ table
badge(7, 780, 437, 715, 437)    # fused VGPR/wave
badge(8, 810, 459, 630, 459)    # fused LDS/block
badge(9, 780, 481, 650, 481)    # fused occupancy (modeled)
badge(10, 780, 515, 720, 515)   # FUSION verdict line

# ---- legend in the bottom margin, two columns ----
d.text((20, H + 12), "Fusion Analysis — feature guide", font=f_title,
       fill=(240, 240, 245, 255))

items = [
    ("1", "easy selection", "drag a lasso on the timeline; ctrl/cmd+click adds/removes one."),
    ("2", "Selection panel", "families you picked: count, kernel time, % of selection."),
    ("3", "span (wall)", "wall-clock time covered by the selected kernel group."),
    ("4", "busy (sum kernels)", "GPU time actually spent inside those kernels."),
    ("5", "inter-kernel idle", "span - busy = idle between kernels; reclaimable by fusion."),
    ("6", "per-family resources", "VGPR, LDS/block, scratch, occupancy per family."),
    ("7", "fused VGPR/wave", "modeled registers if fused (96=full occ, 256=spill)."),
    ("8", "fused LDS/block", "modeled LDS use vs the per-WGP LDS budget."),
    ("9", "fused occupancy", "modeled occupancy of the fused kernel (slots-bound here)."),
    ("10", "fusion verdict", "plain-language call: worth it, and within budget?"),
]

col_w = 730
rx = [20, 20 + col_w]
rows_per_col = 5
row_h = 42
top = H + 48
for i, (num, head, desc) in enumerate(items):
    x = rx[i // rows_per_col]
    y = top + (i % rows_per_col) * row_h
    d.ellipse([x, y, x + 24, y + 24], fill=(10, 10, 12, 255), outline=ACCENT, width=3)
    tw = d.textlength(num, font=f_legendb)
    d.text((x + 12 - tw / 2, y + 3), num, font=f_legendb, fill=ACCENT)
    d.text((x + 34, y - 1), head, font=f_legendb, fill=(255, 235, 150, 255))
    d.text((x + 34, y + 18), desc, font=f_legend, fill=(205, 205, 210, 255))

out = Image.alpha_composite(canvas, ov).convert("RGB")
out.save(DST)
print("wrote", DST, out.size)
