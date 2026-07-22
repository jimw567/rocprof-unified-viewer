#!/usr/bin/env python3
"""Annotate docs/system-view.png with numbered callouts + zone boxes for docs."""
from PIL import Image, ImageDraw, ImageFont

SRC = "docs/system-view.png"
DST = "docs/system-view-annotated.png"

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
def font(name, sz):
    return ImageFont.truetype(FONT_DIR + name, sz)

f_badge   = font("DejaVuSans-Bold.ttf", 20)
f_zone    = font("DejaVuSans-Bold.ttf", 17)
f_legend  = font("DejaVuSans.ttf", 17)
f_legendb = font("DejaVuSans-Bold.ttf", 17)
f_title   = font("DejaVuSans-Bold.ttf", 22)

base = Image.open(SRC).convert("RGBA")
W, H = base.size

MARGIN = 360  # bottom strip for the legend
canvas = Image.new("RGBA", (W, H + MARGIN), (18, 18, 22, 255))
canvas.paste(base, (0, 0))

# overlay layer for translucent shapes
ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
d = ImageDraw.Draw(ov)

ACCENT = (255, 210, 40, 255)     # badge / leader color
ZONE   = (80, 200, 255, 255)     # zone outline color

def zone(x0, y0, x1, y1, label, col=ZONE):
    d.rectangle([x0, y0, x1, y1], outline=col, width=3)
    tw = d.textlength(label, font=f_zone)
    pad = 6
    ty = y0 - 26 if y0 - 26 > 0 else y0 + 4
    d.rectangle([x0, ty, x0 + tw + 2 * pad, ty + 24], fill=(10, 10, 12, 235))
    d.text((x0 + pad, ty + 3), label, font=f_zone, fill=col)

def badge(n, bx, by, tx, ty):
    """circle numbered badge at (bx,by), leader line to target (tx,ty)."""
    r = 17
    d.line([bx, by, tx, ty], fill=ACCENT, width=3)
    d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=ACCENT)
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=(10, 10, 12, 255),
              outline=ACCENT, width=3)
    s = str(n)
    tw = d.textlength(s, font=f_badge)
    d.text((bx - tw / 2, by - 12), s, font=f_badge, fill=ACCENT)

# ---- major zone outlines ----
zone(14, 68, 1102, 101, "NAVIGATION & MARKERS")
zone(8, 138, 1446, 392, "MULTI-LANE TIMELINE  (host HIP  |  GPU kernels  |  phase  |  layer)")
zone(12, 400, 1446, 429, "GPU BUSY / IDLE")
zone(12, 470, 938, 893, "SELECTED-KERNEL DETAIL & ROOFLINE")
zone(1458, 72, 1893, 786, "PER-KERNEL-FAMILY / TOKEN TABLE")

# ---- numbered badges: (n, badge_x, badge_y, target_x, target_y) ----
badge(1, 470, 45, 210, 14)      # title: device + peak BW
badge(2, 900, 45, 560, 34)      # baked-span summary stats
badge(3, 640, 128, 648, 100)    # gap-nav dropdown / find
badge(4, 1010, 55, 1009, 84)    # RDNA 3.5 HW button
badge(5, 300, 128, 240, 118)    # bottleneck color legend
badge(6, 60, 240, 120, 197)     # host HIP lane
badge(7, 728, 148, 728, 183)    # A/B markers + dt
badge(8, 1120, 240, 700, 292)   # GPU kernel slices (color=bottleneck)
badge(9, 1250, 340, 1025, 340)  # phase/family lane
badge(10, 1250, 366, 470, 366)  # layer lane
badge(11, 470, 448, 90, 542)    # bottleneck diagnosis line
badge(12, 690, 690, 300, 710)   # roofline metrics table
badge(13, 560, 448, 420, 510)   # drill-down buttons
badge(14, 1420, 55, 1660, 133)  # family table stall class + footer

# ---- legend in bottom margin ----
ly0 = H + 16
d.text((20, ly0), "rocprof unified viewer — main view: annotated feature guide",
       font=f_title, fill=(240, 240, 245, 255))

items = [
    ("1", "Device + peak DRAM BW", "gfx1151, 230 GB/s — the roofline denominator for all BW%."),
    ("2", "Baked-span summary", "tokens baked, GPU slices, HIP calls, window GPU-busy, GGUF matvec coverage, decode tok/s."),
    ("3", "Gap navigation", "jump between largest intra-token gaps; Find next/prev steps through occurrences."),
    ("4", "RDNA 3.5 HW view", "opens the embedded WGP hardware reference diagram."),
    ("5", "Bottleneck color key", "memory-bound / compute / occupancy-latency / LDS / copy / no-PMC."),
    ("6", "Host HIP-API lane", "CPU-side HIP calls (e.g. hipStreamSynchronize) with host duration on hover."),
    ("7", "A / B markers + dt", "drop two markers to measure an interval (here dt = 70.3 µs)."),
    ("8", "GPU kernel slices", "each dispatch, width = duration, color = its dominant bottleneck."),
    ("9", "Phase / family lane", "groups slices into qkv / attn / ffn / o_proj phases."),
    ("10", "Layer lane", "maps slices to model layers/blocks (L10 GDN, L11 ATTN)."),
    ("11", "Bottleneck diagnosis", "plain-language verdict for the selected kernel (DRAM-bound, etc.)."),
    ("12", "Roofline metrics", "duration, weight tensor, effective vs peak BW, FETCH_SIZE, L2 hit, MemUnitBusy."),
    ("13", "Drill-down actions", "Open tiling view / Run trace / Open trace view for the selected kernel."),
    ("14", "Family table + footer", "per-family cnt/tok, time%, stall class; footer rolls up totals & eff BW%."),
]

col_w = 940
rows_per_col = 7
rx = [20, 20 + col_w]
row_h = 44
top = ly0 + 40
for i, (num, head, desc) in enumerate(items):
    c = i // rows_per_col
    r = i % rows_per_col
    x = rx[c]
    y = top + r * row_h
    # badge chip
    d.ellipse([x, y, x + 26, y + 26], fill=(10, 10, 12, 255), outline=ACCENT, width=3)
    tw = d.textlength(num, font=f_legendb)
    d.text((x + 13 - tw / 2, y + 4), num, font=f_legendb, fill=ACCENT)
    d.text((x + 38, y - 1), head, font=f_legendb, fill=(255, 235, 150, 255))
    d.text((x + 38, y + 20), desc, font=f_legend, fill=(205, 205, 210, 255))

out = Image.alpha_composite(canvas, ov).convert("RGB")
out.save(DST)
print("wrote", DST, out.size)
