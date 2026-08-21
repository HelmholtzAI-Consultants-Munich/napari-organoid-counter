from napari_organoid_counter import _utils as utils


def test_get_adaptive_edge_width_scales_with_image_size():
    assert utils.get_adaptive_edge_width((5000, 5000)) == 12
    assert utils.get_adaptive_edge_width((2500, 2500)) == 6
    assert utils.get_adaptive_edge_width((1000, 1000)) == 2
    assert utils.get_adaptive_edge_width((25000, 10000)) == 48


def test_get_adaptive_edge_width_handles_rgb_shape_and_fallback():
    assert utils.get_adaptive_edge_width((768, 1024, 3)) == 2
    assert utils.get_adaptive_edge_width(()) == 12
