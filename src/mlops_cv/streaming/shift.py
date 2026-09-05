"""A synthetic photometric shift the producer can dial into the stream.

Two modes, and the pair is the point:

- **defocus** — a Gaussian blur.
- **brightness** — a multiplier.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter

DEFAULT_AMOUNTS: dict[str, float] = {"defocus": 4.0, "brightness": 0.35}
MODES: tuple[str, ...] = tuple(DEFAULT_AMOUNTS)

JPEG_QUALITY = 95

_UNITS = {"defocus": "Gaussian blur radius", "brightness": "multiplier"}


@dataclass(frozen=True)
class Shift:
    """One labeled synthetic transform, applied to a frame's bytes before it is published."""

    mode: str
    amount: float

    def __post_init__(self) -> None:
        if self.mode not in DEFAULT_AMOUNTS:
            raise ValueError(f"unknown shift mode {self.mode!r}; the modes are {list(MODES)}")
        if self.amount <= 0:
            raise ValueError(f"the shift amount must be positive, not {self.amount}")

    def describe(self) -> str:
        """How the shift announces itself — in the startup log line, and nowhere unlabeled."""
        return f"synthetic {self.mode} shift ({_UNITS[self.mode]} {self.amount:g})"

    def apply(self, data: bytes) -> bytes:
        """Transform one frame; raises on data that is not a decodable image."""
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            shifted = self._transform(im.convert("RGB"))
        buf = io.BytesIO()
        shifted.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

    def _transform(self, im: Image.Image) -> Image.Image:
        if self.mode == "defocus":
            return im.filter(ImageFilter.GaussianBlur(radius=self.amount))
        return ImageEnhance.Brightness(im).enhance(self.amount)
