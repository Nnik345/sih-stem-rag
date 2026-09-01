#!/usr/bin/env python3
"""Wide SIH Slide 3 architecture: full pipeline, simple English, readable labels."""

from PIL import Image, ImageDraw, ImageFont

W, H = 2400, 1020
BG = (255, 255, 255)
BLUE = (0, 112, 192)
BLUE_DK = (0, 84, 150)
TEAL = (15, 118, 110)
ORANGE = (180, 60, 12)
GREEN = (4, 110, 80)
TEXT = (30, 41, 59)
MUTED = (71, 85, 105)
WHITE = (255, 255, 255)
BAND1 = (232, 244, 252)
BAND2 = (248, 250, 252)
BOX_LINE = (0, 112, 192)

FONT_DIR = "/usr/share/fonts/TTF"
font_box = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 21)
font_sub = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 16)
font_tiny = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 15)
font_band = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 20)


def text_size(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def rounded(draw, xy, fill, outline, radius=16, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def box(draw, x, y, w, h, title, sub=None, fill=WHITE, outline=BOX_LINE, title_fill=BLUE_DK):
    rounded(draw, (x, y, x + w, y + h), fill=fill, outline=outline)
    lines = [(title, font_box, title_fill)]
    if sub:
        lines.append((sub, font_sub, MUTED))
    total = 0
    measured = []
    for line, f, c in lines:
        tw, th = text_size(draw, line, f)
        measured.append((line, f, c, tw, th))
        total += th
    total += 6 if sub else 0
    cy = y + (h - total) / 2
    for i, (line, f, c, tw, th) in enumerate(measured):
        draw.text((x + (w - tw) / 2, cy), line, font=f, fill=c)
        cy += th + (6 if i == 0 and sub else 0)
    return {"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2, "r": x + w, "l": x, "t": y, "b": y + h}


def h_arrow(draw, x1, x2, y, color=BLUE, width=4):
    if x2 < x1:
        x1, x2 = x2, x1
    draw.line((x1, y, x2 - 12, y), fill=color, width=width)
    draw.polygon([(x2, y), (x2 - 14, y - 8), (x2 - 14, y + 8)], fill=color)


def v_arrow(draw, x, y1, y2, color=BLUE, width=4):
    down = y2 > y1
    draw.line((x, y1, x, y2 - (12 if down else -12)), fill=color, width=width)
    if down:
        draw.polygon([(x, y2), (x - 8, y2 - 14), (x + 8, y2 - 14)], fill=color)
    else:
        draw.polygon([(x, y2), (x - 8, y2 + 14), (x + 8, y2 + 14)], fill=color)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# ===== 1. BUILD ONCE =====
rounded(d, (28, 20, W - 28, 248), fill=BAND1, outline=(186, 216, 245), radius=22, width=2)
d.text((52, 36), "1. BUILD ONCE  —  turn the textbooks into a map", font=font_band, fill=BLUE_DK)

bw, bh = 340, 112
gap = 52
xs = 70
by = 88
build = [
    ("NCERT books", "official PDFs only"),
    ("Read the PDF", "text + figures"),
    ("Cut into chunks", "page and section"),
    ("Make searchable", "turn text into vectors"),
    ("Neo4j book map", "graph + words + figures"),
]
boxes = []
x = xs
for title, sub in build:
    boxes.append(box(d, x, by, bw, bh, title, sub))
    x += bw + gap
for a, b_ in zip(boxes, boxes[1:]):
    h_arrow(d, a["r"] + 2, b_["l"] - 2, a["cy"])

# ===== 2. EVERY DOUBT =====
rounded(d, (28, 272, W - 28, H - 28), fill=BAND2, outline=(226, 232, 240), radius=22, width=2)
d.text((52, 290), "2. EVERY STUDENT DOUBT  —  search, check, then teach", font=font_band, fill=BLUE_DK)

# Column 1: student + dashboard
b_stu = box(d, 56, 350, 280, 96, "Student", "types a doubt or sends a photo")
b_dash = box(d, 56, 478, 280, 96, "Dashboard", "locks class and subject")
v_arrow(d, b_stu["cx"], b_stu["b"] + 2, b_dash["t"] - 2)

# Column 2: rewrite
b_rw = box(d, 400, 414, 300, 110, "Rewrite the doubt", "so it sounds like the book")
h_arrow(d, b_dash["r"] + 2, b_rw["l"] - 2, b_rw["cy"])

# Column 3: four searches
sx, sy = 770, 348
sw, sh = 300, 72
sgap = 12
search_items = [
    ("Meaning search", "what it is about"),
    ("Word search", "exact words in the book"),
    ("Chapter map", "linked pages nearby"),
    ("Figure match", "if the student sent a photo"),
]
searches = []
for i, (t, s) in enumerate(search_items):
    searches.append(box(d, sx, sy + i * (sh + sgap), sw, sh, t, s))

# rewrite -> middle of search stack
h_arrow(d, b_rw["r"] + 2, sx - 2, b_rw["cy"])

# Column 4: combine
b_merge = box(d, 1140, 414, 280, 110, "Combine and rank", "best pages first")
h_arrow(d, sx + sw + 2, b_merge["l"] - 2, b_merge["cy"])

# Column 5: book check
b_gate = box(d, 1488, 400, 270, 96, "Book check", "is this enough proof?")
h_arrow(d, b_merge["r"] + 2, b_gate["l"] - 2, b_gate["cy"])

# Column 6: yes / no
b_tutor = box(d, 1860, 348, 480, 100, "Tutor", "hint like a teacher, from those pages")
b_stop = box(
    d,
    1860,
    500,
    480,
    96,
    "Stop",
    "book is not enough  •  do not invent",
    outline=ORANGE,
    title_fill=ORANGE,
)
h_arrow(d, b_gate["r"] + 2, b_tutor["l"] - 2, b_tutor["cy"], color=GREEN)
d.text((b_gate["r"] + 8, b_tutor["cy"] - 26), "yes", font=font_tiny, fill=GREEN)

# no path: down then right
d.line((b_gate["cx"], b_gate["b"] + 2, b_gate["cx"], b_stop["cy"]), fill=ORANGE, width=4)
h_arrow(d, b_gate["cx"], b_stop["l"] - 2, b_stop["cy"], color=ORANGE)
d.text((b_gate["cx"] + 10, b_gate["b"] + 8), "no", font=font_tiny, fill=ORANGE)

# Neo4j feeds searches — bar under the four boxes
feed = box(
    d,
    sx,
    720,
    sw,
    84,
    "Fed by the Neo4j book map",
    "the store from step 1",
    outline=TEAL,
    title_fill=TEAL,
)
v_arrow(d, feed["cx"], feed["t"] - 2, searches[-1]["b"] + 2, color=TEAL)
# wait that's wrong direction - we need UP from feed to last search. v_arrow with y2 < y1
# Actually v_arrow I defined: down if y2>y1. From feed top UP to last search bottom... 
# last search bottom is above feed. So y1=feed.t, y2=searches[-1].b which is smaller - UP arrow.
# But I drew v_arrow from feed.t DOWN to searches[-1].b which would be backwards.
# Fix: draw upward from feed.t to searches[-1].b

# Caption
d.text((56, 920), "All of this stays on one school PC. No cloud.", font=font_sub, fill=MUTED)
d.text(
    (1860, 620),
    "The hint and the page trail show on the dashboard.",
    font=font_tiny,
    fill=MUTED,
)

out = "/home/nnik345/Projects/sih-stem-rag/.ppt-assets/slide3-full-architecture.png"
img.save(out, "PNG")
print("wrote", out)
