import numpy as np
import pytest

from segi_freshness import reference as ref
from segi_freshness.config import ColorThresholdsConfig

from .conftest import BROWN, DARK, GREEN, MOLD, RED, YELLOW, solid

T = ColorThresholdsConfig()


@pytest.mark.parametrize(
    "color,index",
    [(BROWN, ref.BROWN), (DARK, ref.DARK), (MOLD, ref.MOLD), (YELLOW, ref.YELLOW)],
)
def test_color_stats_classifies_pure_colours(color, index):
    img = solid(color)
    stats = ref.color_stats(img, np.array([[0, 0, 160, 120]]), 16, T)[0]
    assert stats[index] == pytest.approx(1.0)
    others = [i for i in (ref.DARK, ref.BROWN, ref.MOLD, ref.YELLOW) if i != index]
    assert stats[others].sum() == pytest.approx(0.0)


@pytest.mark.parametrize("color", [GREEN, RED])
def test_healthy_colours_have_no_defects(color):
    stats = ref.color_stats(solid(color), np.array([[0, 0, 160, 120]]), 16, T)[0]
    assert stats[:4].sum() == pytest.approx(0.0)


def test_color_stats_ratio_matches_area():
    img = solid(GREEN, 100, 100)
    img[:, :25] = BROWN  # left quarter browned
    stats = ref.color_stats(img, np.array([[0, 0, 100, 100]]), 20, T)[0]
    assert stats[ref.BROWN] == pytest.approx(0.25)


def test_color_stats_clamps_boxes_outside_image():
    stats = ref.color_stats(solid(DARK), np.array([[-50, -50, 500, 500]]), 8, T)
    assert stats[0, ref.DARK] == pytest.approx(1.0)


def test_letterbox_params_and_padding():
    img = solid(RED, 360, 640)
    chw, info = ref.letterbox(img, 320)
    assert chw.shape == (3, 320, 320)
    assert (info.new_w, info.new_h, info.pad_x, info.pad_y) == (320, 180, 0, 70)
    assert chw[:, 0, 0] == pytest.approx([ref.LETTERBOX_PAD] * 3)
    centre = chw[:, 160, 160]
    assert centre == pytest.approx([205 / 255, 30 / 255, 30 / 255], abs=1e-5)  # RGB order


def test_letterbox_inverse_maps_boxes():
    info = ref.letterbox_params(1920, 1080, 640)
    net = np.array([[info.pad_x, info.pad_y, info.pad_x + info.new_w, info.pad_y + info.new_h]], np.float32)
    assert info.to_source(net)[0] == pytest.approx([0, 0, 1920, 1080], abs=0.5)


def test_crop_resize_constant_and_normalisation():
    img = solid(RED)
    out = ref.crop_resize(img, np.array([[10, 10, 90, 70]]), 32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert out.shape == (1, 3, 32, 32)
    assert out[0, 0].mean() == pytest.approx((205 / 255 - 0.5) / 0.5, abs=1e-5)
    assert out[0, 2].mean() == pytest.approx((30 / 255 - 0.5) / 0.5, abs=1e-5)


def test_bgrx_input_is_accepted():
    img = np.dstack([solid(BROWN), np.full((120, 160), 255, np.uint8)])
    stats = ref.color_stats(img, np.array([[0, 0, 160, 120]]), 8, T)[0]
    assert stats[ref.BROWN] == pytest.approx(1.0)


def test_shrink_boxes():
    out = ref.shrink_boxes(np.array([[0, 0, 100, 50]]), 0.1)
    assert out[0] == pytest.approx([10, 5, 90, 45])
