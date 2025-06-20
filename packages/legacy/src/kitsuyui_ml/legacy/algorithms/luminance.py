# References
# https://www.w3.org/TR/WCAG20/

Uint8 = int


def uint8_color_to_float(color: Uint8) -> float:
    """Convert a color value from uint8 to float."""
    return color / 255


def relative_luminance(r_s: float, g_s: float, b_s: float) -> float:
    """Calculate the relative luminance of a color in the sRGB color space.
    The relative luminance of a color is defined in the WCAG 2.0 spec as:
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    http://www.w3.org/TR/2008/REC-WCAG20-20081211/#relativeluminancedef
    """
    r = decode_gumma_to_linear(r_s)
    g = decode_gumma_to_linear(g_s)
    b = decode_gumma_to_linear(b_s)
    # https://en.wikipedia.org/wiki/Relative_luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance


def decode_gumma_to_linear(value: float) -> float:
    """Decode a color value from sRGB value to linear value."""
    if value <= 0.03928:
        return float(value / 12.92)
    return float(((value + 0.055) / 1.055) ** 2.4)


def relative_luminance_from_uint8(r: Uint8, g: Uint8, b: Uint8) -> float:
    """Calculate the relative luminance of a color in 8-bit sRGB color space.
    The relative luminance of a color is defined in the WCAG 2.0 spec as:
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    http://www.w3.org/TR/2008/REC-WCAG20-20081211/#relativeluminancedef
    """
    r_s_rgb = uint8_color_to_float(r)
    g_s_rgb = uint8_color_to_float(g)
    b_s_rgb = uint8_color_to_float(b)
    return relative_luminance(r_s_rgb, g_s_rgb, b_s_rgb)


def relative_luminance_from_hex(hex_color: str) -> float:
    """Calculate the relative luminance of a color in hex format.

    The relative luminance of a color is defined in the WCAG 2.0 spec as:
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    http://www.w3.org/TR/2008/REC-WCAG20-20081211/#relativeluminancedef
    """
    r, g, b = hex_color_to_rgb(hex_color)
    return relative_luminance_from_uint8(r, g, b)


def hex_color_to_rgb(hex_color: str) -> tuple[Uint8, Uint8, Uint8]:
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(2 * s for s in hex_color)
    r_hex, g_hex, b_hex = hex_color[:2], hex_color[2:4], hex_color[4:]
    r, g, b = int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)
    return r, g, b


def text_should_be_black_by_background_color(color: str) -> bool:
    """Determine whether text should be black or white based on the background color.
    When the luminance is great, the text should be black (True).
    When the luminance is small, the text should be white (False).
    """
    luminance = relative_luminance_from_hex(color)
    return luminance > THRESHOLD


# Same as choose_best_contrast_color in white and black
# This means solve the following inequality in (0.0 <= L <= 1.0):
# Proofs:
# (Black contrast ratio) > (White contrast ratio)
# (L + 0.05) / (0.0 + 0.05) > (1.0 + 0.05) / (L + 0.05)
# (L + 0.05) * (L + 0.05) > (1.0 + 0.05) * (0.0 + 0.05)
# (L + 0.05) ** 2 > 1.05 * 0.05
# L + 0.05 > (1.05 * 0.05) ** 0.5
# L > (1.05 * 0.05) ** 0.5 - 0.05
# (1.05 * 0.05) ** 0.5 - 0.05 == 0.17912878474779204
THRESHOLD = 0.17912878474779204


def contrast_ratio_1(lighter: float, darker: float) -> float:
    """Calculate the contrast ratio between two luminance values.
    The first argument must be greater than the second argument.

    https://www.w3.org/TR/2008/REC-WCAG20-20081211/#contrast-ratiodef
    """
    return (lighter + 0.05) / (darker + 0.05)


def contrast_ratio(l1: float, l2: float) -> float:
    """Calculate the contrast ratio between two luminance values.
    from 21.0 (white with black) to 1.0 (same color)

    https://www.w3.org/TR/2008/REC-WCAG20-20081211/#contrast-ratiodef
    """
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return contrast_ratio_1(lighter, darker)


def choose_best_contrast_color(
    fixed_color: str,
    colors: list[str],
) -> str:
    """Choose the color with the best contrast ratio from a list of colors.
    The color with the best contrast ratio is the color with the highest
    contrast ratio from the fixed color.

    This function is useful for choosing a better monochrome text color for a given background color.
    But this function only works for monochrome colors. It does not work for colors with different hues.
    """
    fixed_color_luminance = relative_luminance_from_hex(fixed_color)
    luminances = [relative_luminance_from_hex(color) for color in colors]
    contrast_ratios = [
        contrast_ratio(fixed_color_luminance, luminance) for luminance in luminances
    ]
    max_contrast_ratio = max(contrast_ratios)
    max_contrast_ratio_index = contrast_ratios.index(max_contrast_ratio)
    return colors[max_contrast_ratio_index]


BLACK_LUMINANCE = relative_luminance_from_hex("#000000")
WHITE_LUMINANCE = relative_luminance_from_hex("#FFFFFF")
