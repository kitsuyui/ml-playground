import kitsuyui_ml.legacy.algorithms.luminance as luminance


def test_luminance() -> None:
    assert luminance.BLACK_LUMINANCE == 0.0
    assert luminance.WHITE_LUMINANCE == 1.0


def test_text_black_or_white_from_background_color() -> None:
    assert luminance.text_should_be_black_by_background_color("#FFFFFF")
    assert not luminance.text_should_be_black_by_background_color("#000000")


def test_contrast_ratio() -> None:
    assert luminance.contrast_ratio(1.0, 0.0) == 21.0
    assert luminance.contrast_ratio(0.5, 0.5) == 1.0
    assert luminance.contrast_ratio(0.0, 0.0) == 1.0
    assert luminance.contrast_ratio(1.0, 1.0) == 1.0


def test_contrast_ratio_from_luminance_pair() -> None:
    assert luminance.contrast_ratio(1.0, 0.0) == 21.0
    assert luminance.contrast_ratio(0.0, 1.0) == 21.0
    assert luminance.contrast_ratio(0.5, 0.5) == 1.0
    assert luminance.contrast_ratio(0.0, 0.0) == 1.0
    assert luminance.contrast_ratio(1.0, 1.0) == 1.0


def test_choose_best_contrast_color() -> None:
    assert (
        luminance.choose_best_contrast_color("#FFFFFF", ["#000000", "#FFFFFF"])
        == "#000000"
    )
    assert (
        luminance.choose_best_contrast_color("#000000", ["#000000", "#FFFFFF"])
        == "#FFFFFF"
    )
    assert (
        luminance.choose_best_contrast_color("#000000", ["#111111", "#EEEEEE"])
        == "#EEEEEE"
    )
    assert (
        luminance.choose_best_contrast_color("#CCCCCC", ["#111111", "#EEEEEE"])
        == "#111111"
    )
    assert luminance.choose_best_contrast_color("#000", ["#000", "#FFF"]) == "#FFF"
