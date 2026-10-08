"""Сжатие картинок-аватаров. Оригиналы читаются из assets/originals."""
from pathlib import Path

from PIL import Image

for name in ("left_profile.png", "right_profile.png"):
    src = Path("assets/originals") / name
    before_mb = src.stat().st_size / 1024 / 1024

    image = Image.open(src).convert("RGB")
    width, height = image.size
    image = image.resize((1200, round(height * 1200 / width)), Image.LANCZOS)

    # Рисунок контурный, на тёмном фоне — палитры в 128 цветов достаточно.
    image = image.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    image.save(name, "PNG", optimize=True)

    after_kb = Path(name).stat().st_size / 1024
    print(f"{name}: {before_mb:.1f} MB -> {after_kb:.0f} KB")
