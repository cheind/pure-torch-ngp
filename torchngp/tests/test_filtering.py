import torch
from torch.testing import assert_close

from torchngp import filtering

from .test_radiance import ColorGradientRadianceField


def test_bounds_filter():
    f = filtering.BoundsFilter()
    nxyz = torch.empty((100, 3)).uniform_(-1.0, 1.0)
    assert f.test(nxyz).all()

    nxyz = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    assert f.test(nxyz).all()


def test_occupancy_grid_filter_deterministic():
    rf = ColorGradientRadianceField(surface_dim=0, surface_pos=0.0, density_scale=1.0)
    f = filtering.OccupancyGridFilter(
        res=4,
        update_decay=1.0,
        update_selection_rate=1.0,
        update_noise_scale=0.0,
        density_initial=0.0,
        density_threshold=0.0,
    )
    f.update(rf)

    # Note dim order is DHW and indexing is xyz. The plane in color gradient field
    # is parallel to x. Plane is at zero in NDC, so at right most x border we have
    # density 0.5 (according to config above). Then the last voxel center is actually
    # a bit before that
    assert_close(f.grid[0, 0, :], torch.tensor([0.0000, 0.0000, 0.1250, 0.3750]))
    assert_close(f.grid[0, :, 0], torch.tensor([0.0000, 0.0000, 0.0000, 0.0000]))
    assert_close(f.grid[:, 0, 0], torch.tensor([0.0000, 0.0000, 0.0000, 0.0000]))

    pts = torch.empty((1000, 3)).uniform_(-1.0, 1.0)
    m = f.test(pts)
    expected = pts[:, 0] > 0.0

    assert_close(m, expected)


def test_occupancy_grid_filter_non_deterministic():
    rf = ColorGradientRadianceField(surface_dim=0, surface_pos=0.0, density_scale=1.0)
    f = filtering.OccupancyGridFilter(
        res=4,
        update_decay=0.2,
        update_selection_rate=1.0,
        update_noise_scale=None,
        density_initial=0.0,
        density_threshold=0.0,
    )
    [f.update(rf) for _ in range(1000)]

    # Note dim order is DHW and indexing is xyz. The plane in color gradient field
    # is parallel to x. Plane is at zero in NDC, so at right most x border we have
    # density 0.5 (according to config above). The grid now contains the maximum
    # densities over 1000 iterations for random points within the voxel (no decay)
    assert_close(f.grid[0, 0, :2], torch.tensor([0.0000, 0.0000]))
    assert f.grid[0, 0, 2] > 0.0 and f.grid[0, 0, 2] <= 0.25
    assert f.grid[0, 0, 3] > 0.25 and f.grid[0, 0, 2] <= 0.5
    assert_close(f.grid[0, :, 0], torch.tensor([0.0000, 0.0000, 0.0000, 0.0000]))
    assert_close(f.grid[:, 0, 0], torch.tensor([0.0000, 0.0000, 0.0000, 0.0000]))

    pts = torch.empty((1000, 3)).uniform_(-1.0, 1.0)
    m = f.test(pts)
    expected = pts[:, 0] > 0.0
    assert_close(m, expected)
