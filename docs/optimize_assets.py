"""Regenerate website derivatives without changing the original figure files.

Run with Python and Pillow from any directory.
"""
from pathlib import Path
from PIL import Image, ImageChops

ASSETS = Path(__file__).resolve().parent / "assets"

def export(source, name, width, *, emblem=False):
    image = Image.open(ASSETS / source).convert("RGB")
    if emblem:
        # Remove only the near-white outer margin, preserving the original mark.
        mask = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        mask = mask.convert("L").point(lambda value: 255 if value > 35 else 0)
        image = image.crop(mask.getbbox())
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    target = ASSETS / name
    image.save(target, "WEBP", quality=88, method=6)
    print(f"{name}: {width}x{height}, {target.stat().st_size:,} bytes")

if __name__ == "__main__":
    export("zhejiang-university.png", "zhejiang-university-emblem.webp", 128, emblem=True)
    export("Zhe.jpg", "sahzu-emblem.webp", 128, emblem=True)
    export("anestrace-primary-logo.png", "anestrace-primary-logo.webp", 384)
    for width in (1200, 2400):
        export("anestrace-overview.png", f"anestrace-overview-{width}.webp", width)
    for width in (1040, 2080):
        export("representative-l3-evaluation-trajectory.png", f"representative-l3-{width}.webp", width)
