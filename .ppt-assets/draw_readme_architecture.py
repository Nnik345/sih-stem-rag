#!/usr/bin/env python3
"""README.md Architecture — same stages, same names, two columns for the slide."""

from PIL import Image, ImageDraw, ImageFont

W, H = 2000, 1320
BLUE = (0, 112, 192)
BLUE_DK = (0, 84, 150)
TEAL = (13, 148, 136)
ORANGE = (180, 60, 12)
MUTED = (71, 85, 105)
WHITE = (255, 255, 255)
NAVY = (15, 23, 42)
BAND_L = (239, 246, 255)
BAND_R = (248, 250, 252)

FONT_DIR = "/usr/share/fonts/TTF"
fb = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 19)
fs = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 14)
fband = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 22)


def ts(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def box(draw, x, y, w, h, title, subs=None, outline=BLUE, tfill=BLUE_DK):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=WHITE, outline=outline, width=3)
    cx = x + w / 2
    items = [(title, fb, tfill)] + [(s, fs, MUTED) for s in (subs or [])]
    measured = []
    total = 0
    for text, font, col in items:
        tw, th = ts(draw, text, font)
        if tw <= w - 20:
            measured.append((text, font, col, tw, th))
            total += th + 2
            continue
        words, cur = text.split(), ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if ts(draw, trial, font)[0] <= w - 20:
                cur = trial
            else:
                tw2, th2 = ts(draw, cur, font)
                measured.append((cur, font, col, tw2, th2))
                total += th2 + 2
                cur = wd
        if cur:
            tw2, th2 = ts(draw, cur, font)
            measured.append((cur, font, col, tw2, th2))
            total += th2 + 2
    cy = y + max(6, (h - total) / 2)
    for text, font, col, tw, th in measured:
        draw.text((cx - tw / 2, cy), text, font=font, fill=col)
        cy += th + 2
    return y + h


def v_arrow(draw, x, y1, y2):
    draw.line((x, y1, x, y2 - 11), fill=BLUE, width=4)
    draw.polygon([(x, y2), (x - 7, y2 - 13), (x + 7, y2 - 13)], fill=BLUE)


img = Image.new("RGB", (W, H), WHITE)
d = ImageDraw.Draw(img)
d.text((40, 18), "Architecture  (README.md)", font=fband, fill=NAVY)

d.rounded_rectangle((28, 62, 984, 1288), radius=18, fill=BAND_L, outline=(186, 216, 245), width=2)
d.rounded_rectangle((1016, 62, 1972, 1288), radius=18, fill=BAND_R, outline=(226, 232, 240), width=2)
d.text((52, 80), "Corpus", font=fband, fill=BLUE_DK)
d.text((1040, 80), "Student query", font=fband, fill=BLUE_DK)

# left
lx, lw, cx = 64, 884, 64 + 884 / 2
left = [
    (110, "Approved STEM sources (PDF/ePUB)", None, BLUE, BLUE_DK),
    (110, "PyMuPDF structured parsing", ["text, headings, embedded images, page geometry"], BLUE, BLUE_DK),
    (110, "Hierarchical chunking", ["Document → Page → Section → Chunk"], BLUE, BLUE_DK),
    (96, "BGE-M3 dense embeddings", None, BLUE, BLUE_DK),
    (200, "Neo4j", [
        "curriculum knowledge graph",
        "dense vector index (cosine)",
        "full-text (Lucene) index",
        "metadata properties on Chunk nodes",
    ], TEAL, TEAL),
]
y = 128
prev = None
for h, title, subs, outline, tfill in left:
    if prev is not None:
        v_arrow(d, cx, prev, y)
    prev = box(d, lx, y, lw, h, title, subs, outline=outline, tfill=tfill)
    y = prev + 26

# right
rx, rw, cxr = 1052, 884, 1052 + 884 / 2
right = [
    (72, "Student query", None, BLUE, BLUE_DK),
    (88, "Metadata filtering", ["grade + subject, required from the caller"], BLUE, BLUE_DK),
    (88, "Qwen3-VL-2B query rewrite", ["text and optional student photo; then unloaded"], BLUE, BLUE_DK),
    (88, "SigLIP image kNN", ["if a photo was uploaded; then unloaded"], BLUE, BLUE_DK),
    (168, "Retrieval channels", [
        "dense semantic retrieval",
        "lexical / full-text retrieval",
        "bounded graph expansion",
        "page chunks from matched textbook figures",
    ], TEAL, TEAL),
    (72, "Weighted Reciprocal Rank Fusion", None, BLUE, BLUE_DK),
    (72, "BGE-reranker-v2-m3", None, BLUE, BLUE_DK),
    (88, "Evidence sufficiency gate", ["if insufficient → do not invent"], ORANGE, ORANGE),
    (72, "Qwen3-VL-8B-Instruct", None, BLUE, BLUE_DK),
    (72, "Socratic tutoring response (streamed)", None, BLUE, BLUE_DK),
]
y = 128
prev = None
for h, title, subs, outline, tfill in right:
    if prev is not None:
        v_arrow(d, cxr, prev, y)
    prev = box(d, rx, y, rw, h, title, subs, outline=outline, tfill=tfill)
    y = prev + 14

out = "/home/nnik345/Projects/sih-stem-rag/.ppt-assets/slide3-readme-architecture.png"
img.save(out, "PNG")
print("wrote", out, img.size)
