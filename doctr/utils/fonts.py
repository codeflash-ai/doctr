# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import platform
from functools import lru_cache

from PIL import ImageFont

_SYSTEM_FONT = None

__all__ = ["get_font"]


def get_font(font_family: str | None = None, font_size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolves a compatible ImageFont for the system

    Args:
        font_family: the font family to use
        font_size: the size of the font upon rendering

    Returns:
        the Pillow font
    """
    # Font selection
    if font_family is None:
        try:
            system_font = _get_system_font()
            font = _cached_truetype(system_font, font_size)
        except OSError:  # pragma: no cover
            font = _cached_load_default()
            logging.warning(
                "unable to load recommended font family. Loading default PIL font,"
                "font size issues may be expected."
                "To prevent this, it is recommended to specify the value of 'font_family'."
            )
    else:  # pragma: no cover
        font = _cached_truetype(font_family, font_size)

    return font


def _get_system_font() -> str:
    global _SYSTEM_FONT
    if _SYSTEM_FONT is not None:
        return _SYSTEM_FONT
    _SYSTEM_FONT = "FreeMono.ttf" if platform.system() == "Linux" else "Arial.ttf"
    return _SYSTEM_FONT


@lru_cache(maxsize=32)
def _cached_truetype(font_family: str, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.truetype(font_family, font_size)


@lru_cache(maxsize=8)
def _cached_load_default() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default()  # type: ignore[assignment]
