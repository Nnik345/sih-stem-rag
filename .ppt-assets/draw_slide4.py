#!/usr/bin/env python3
"""Slide 4: Feasibility / Risks / Strategies — three equal cards."""

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 980
WHITE = (255, 255, 255)
NAVY = (15, 23, 42)
MUTED = (71, 85, 105)
BLUE = (0, 112, 192)
GREEN = (4, 120, 87)
ORANGE = (180, 60, 12)
TEAL = (13, 148, 136)

FONT_DIR = "/usr/share/fonts/TTF"
ft = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 26)
fb = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 18)
fr = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 17)

img = Image.new("RGB", (W, H), WHITE)
d = ImageDraw.Draw(img)


def ts(text, font):
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def wrap(text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if ts(trial, font)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def card(x, y, w, h, header, color, bullets):
    d.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=(248, 250, 252), outline=color, width=4)
    # header bar
    d.rounded_rectangle((x, y, x + w, y + 64), radius=18, fill=color, outline=color)
    d.rectangle((x, y + 32, x + w, y + 64), fill=color)
    tw, th = ts(header, ft)
    d.text((x + (w - tw) / 2, y + (64 - th) / 2), header, font=ft, fill=WHITE)
    yy = y + 84
    for b in bullets:
        for i, line in enumerate(wrap(b, fr, w - 56)):
            prefix = "•  " if i == 0 else "    "
            d.text((x + 22, yy), prefix + line, font=fr, fill=NAVY)
            yy += 26
        yy += 10


gap = 36
cw = 590
ch = 900
y0 = 40
cards = [
    (
        BLUE,
        "Already possible",
        [
            "Working prototype runs on one PC today.",
            "Python + Neo4j tarball. No Docker. No cloud bill.",
            "Official NCERT zips from ncert.nic.in.",
            "Models load, work, then unload — fits 8–12 GB GPUs (slower).",
            "Teacher can see the page trail, not a black box.",
        ],
    ),
    (
        ORANGE,
        "Risks to watch",
        [
            "The 8B tutor is slow on a small GPU.",
            "Evidence-gate numbers are still heuristics.",
            "No labelled retrieval score yet.",
            "Full NCERT ingest needs time and ~25 GB disk.",
            "No OCR — image-only pages give no text.",
            "NCERT is copyrighted: local use, do not republish.",
        ],
    ),
    (
        TEAL,
        "How we handle it",
        [
            "Retrieval-only and strict mode: skip the tutor if the book is thin.",
            "Unload models between steps so memory fits.",
            "Dashboard to see which page was used.",
            "Tune the gate on a labelled set next.",
            "Start with one class if the school PC is small.",
            "Student photos are deleted after the query.",
        ],
    ),
]

x = 28
for color, header, bullets in cards:
    card(x, y0, cw, ch, header, color, bullets)
    x += cw + gap

out = "/home/nnik345/Projects/sih-stem-rag/.ppt-assets/slide4-feasibility.png"
img.save(out, "PNG")
print("wrote", out)
