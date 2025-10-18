# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import platform

from PIL import ImageFont

_PLATFORM_SYSTEM = platform.system()

# Cache font loading results for default system fonts for common sizes
_default_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

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
        font_key = (_PLATFORM_SYSTEM, font_size)
        if font_key in _default_font_cache:
            return _default_font_cache[font_key]

        try:
            font = ImageFont.truetype(
                "FreeMono.ttf" if _PLATFORM_SYSTEM == "Linux" else "Arial.ttf",
                font_size
            )
        except OSError:  # pragma: no cover
            font = ImageFont.load_default()  # type: ignore[assignment]
            # Logging is not cached, so only log on miss - keeps warning as original
            logging.warning(
                "unable to load recommended font family. Loading default PIL font,"
                "font size issues may be expected."
                "To prevent this, it is recommended to specify the value of 'font_family'."
            )
        _default_font_cache[font_key] = font
        return font
    else:  # pragma: no cover
        return ImageFont.truetype(font_family, font_size)
