"""Label the layouts preview and assemble the 2x2 contact sheet (PIL only).

    python3 scripts/assets/make_contact_sheet.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PREV = Path(__file__).resolve().parents[2] / "assets/vtvl_v1/preview"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# layouts shot: vehicles at x = -6.9 + 4.6 i, camera 23.5 m away, 38 mm lens on a 36 mm sensor (render_previews)
LAYOUT_LABELS = [("gimbal_rcs", "1 gimballed engine"), ("tri_diff", "3 engines, diff. throttle"),
                 ("quad_diff", "4 engines, diff. throttle"), ("coldgas", "1 engine + cold-gas ring")]


def label_layouts(src: Path, dst: Path):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im, "RGBA")
    fb, fs = ImageFont.truetype(FONT_B, int(h * 0.030)), ImageFont.truetype(FONT, int(h * 0.022))
    for i, (name, desc) in enumerate(LAYOUT_LABELS):
        x_m = -6.9 + 4.6 * i
        xf = 0.5 + (38.0 / 36.0) * x_m / 20.6
        cx = int(xf * w)
        y0 = int(h * 0.86)
        tw = max(d.textlength(name, font=fb), d.textlength(desc, font=fs))
        d.rounded_rectangle([cx - tw / 2 - 10, y0 - 6, cx + tw / 2 + 10, y0 + h * 0.075], 6, fill=(0, 0, 0, 120))
        d.text((cx, y0), name, font=fb, fill=(255, 255, 255), anchor="ma")
        d.text((cx, y0 + h * 0.038), desc, font=fs, fill=(225, 225, 225), anchor="ma")
    im.save(dst)


def contact(paths, dst: Path, cell=(960, 540)):
    sheet = Image.new("RGB", (cell[0] * 2 + 12, cell[1] * 2 + 12), (18, 18, 20))
    f = ImageFont.truetype(FONT_B, 20)
    titles = ["a  hero, golden hour", "b  aft end: legs, skirt, nozzle", "c  engine layouts from below", "d  landing site"]
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize(cell, Image.LANCZOS)
        x, y = 4 + (i % 2) * (cell[0] + 4), 4 + (i // 2) * (cell[1] + 4)
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet, "RGBA")
        d.rectangle([x, y, x + 12 + d.textlength(titles[i], font=f), y + 32], fill=(0, 0, 0, 140))
        d.text((x + 6, y + 5), titles[i], font=f, fill=(255, 255, 255))
    sheet.save(dst, optimize=True)


def downsample_supersampled(size=(1280, 720)):
    """<name>_ss2.png -> <name>.png (box filter: averages the 2x2 samples, i.e. ~4x effective samples)."""
    for p in sorted(PREV.glob("*_ss*.png")):
        dst = PREV / (p.stem.rsplit("_ss", 1)[0] + ".png")
        Image.open(p).convert("RGB").resize(size, Image.BOX).save(dst, optimize=True)
        p.unlink()


if __name__ == "__main__":
    downsample_supersampled()
    label_layouts(PREV / "layouts_raw.png", PREV / "layouts.png")
    contact([PREV / "hero.png", PREV / "aft.png", PREV / "layouts.png", PREV / "site.png"], PREV / "contact.png")
    print("wrote", PREV / "contact.png")
