#!/usr/bin/env python3
"""Slide 5 Impact and Benefits — two columns matching SIH pointers."""

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 980
WHITE = (255, 255, 255)
NAVY = (15, 23, 42)
MUTED = (71, 85, 105)
BLUE = (0, 112, 192)
TEAL = (13, 148, 136)

FONT_DIR = "/usr/share/fonts/TTF"
ft = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 24)
fs = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 18)
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


def card(x, y, w, h, header, color, blocks):
    d.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=(248, 250, 252), outline=color, width=4)
    d.rounded_rectangle((x, y, x + w, y + 70), radius=18, fill=color, outline=color)
    d.rectangle((x, y + 36, x + w, y + 70), fill=color)
    lines = wrap(header, ft, w - 28)
    yy = y + 10
    for line in lines:
        tw, th = ts(line, ft)
        d.text((x + (w - tw) / 2, yy), line, font=ft, fill=WHITE)
        yy += th + 2
    yy = y + 88
    for block in blocks:
        if block.startswith("#"):
            d.text((x + 22, yy), block[1:], font=fs, fill=color)
            yy += 30
            continue
        for i, line in enumerate(wrap(block, fr, w - 56)):
            prefix = "•  " if i == 0 else "    "
            d.text((x + 22, yy), prefix + line, font=fr, fill=NAVY)
            yy += 26
        yy += 8


gap = 40
cw = 900
ch = 900
y0 = 40
card(
    28,
    y0,
    cw,
    ch,
    "Who it helps",
    BLUE,
    [
        "# Students (classes 1–12)",
        "Hints from their NCERT book, not a random chatbot.",
        "They have to think. The full answer is not dumped.",
        "They can type a doubt or send a photo of a question.",
        "# Teachers",
        "See which page was used. Easy to trust or correct.",
        "# Schools and parents",
        "Help after class without sending doubts to the internet.",
        "Less copying of made-up, off-syllabus answers.",
    ],
)
card(
    28 + cw + gap,
    y0,
    cw,
    ch,
    "Social · Economic · Environment",
    TEAL,
    [
        "# Social",
        "Same book as the classroom. Right class, right subject.",
        "Works where the internet is weak: it runs on a school PC.",
        "Student photos are deleted after the query.",
        "# Economic",
        "No monthly AI API bill for every student.",
        "Uses a PC and NCERT books the school already has.",
        "# Environmental",
        "Each doubt is answered on one local machine,",
        "not a remote data centre for every question.",
    ],
)

out = "/home/nnik345/Projects/sih-stem-rag/.ppt-assets/slide5-impact.png"
img.save(out, "PNG")
print("wrote", out)
