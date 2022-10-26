import torch
import torch.nn.functional as F
from torch.testing import assert_close


from ngp_nerf.nerf2.encoding import (
    MultiLevelHybridHashEncoding,
    _compute_bilinear_params,
    _hash_ravel,
)


def test_encoding_output_shape_3d():
    mlh = MultiLevelHybridHashEncoding(
        n_encodings=2**16,
        n_input_dims=3,
        n_embed_dims=2,
        n_levels=4,
        min_res=32,
        max_res=64,
        max_n_dense=48 * 48,
    )

    f = mlh(torch.empty((10, 3)).uniform_(-1, 1))
    assert f.shape == (10, 4, 2)


def test_encoding_output_shape_2d():
    mlh = MultiLevelHybridHashEncoding(
        n_encodings=2**14,
        n_input_dims=2,
        n_embed_dims=2,
        n_levels=4,
        min_res=32,
        max_res=64,
        max_n_dense=48 * 48,
    )

    f = mlh(torch.empty((10, 2)).uniform_(-1, 1))
    assert f.shape == (10, 4, 2)


def test_encoding_forward_2d():
    mlh = MultiLevelHybridHashEncoding(
        n_encodings=16,
        n_input_dims=2,
        n_embed_dims=1,
        n_levels=2,
        min_res=2,
        max_res=4,
        max_n_dense=4,
    )
    assert all([not li.hashing for li in mlh.level_infos])
    assert sum([li.dense for li in mlh.level_infos]) == 1
    mlh.level_emb_matrix0.data.copy_(torch.arange(4).view(1, 4) + 1)  # dense
    mlh.level_emb_matrix1.data.copy_(torch.arange(16).view(16, 1) + 1)  # sparse

    x = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    f = mlh(x)
    # level 0 (2x2)
    assert_close(f[0, 0], torch.tensor([0.25]))
    assert_close(f[1, 0], torch.tensor([6.0 / 4]))
    assert_close(f[2, 0], torch.tensor([7.0 / 4]))
    assert_close(f[3, 0], torch.tensor([4.0 / 4]))
    assert_close(f[4, 0], torch.tensor([10.0 / 4]))

    # level 1 (4x4)
    assert_close(f[0, 1], torch.tensor([0.25]))
    assert_close(f[1, 1], torch.tensor([(8.0 + 12.0) / 4]))
    assert_close(f[2, 1], torch.tensor([(14.0 + 15.0) / 4]))
    assert_close(f[3, 1], torch.tensor([16.0 / 4]))
    assert_close(f[4, 1], torch.tensor([(6.0 + 7.0 + 10.0 + 11.0) / 4]))

    # Random points
    x = torch.empty((100, 2)).uniform_(-1.0, 1.0)
    f = mlh(x)

    a = (
        F.grid_sample(
            mlh.level_emb_matrix0.view(1, 2, 2).unsqueeze(0),
            x.view(1, 100, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .view(1, 100)
        .permute(1, 0)
    )
    assert (f[:, 0] - a).abs().max() < 1e-5

    b = (
        F.grid_sample(
            mlh.level_emb_matrix1.permute(1, 0).view(1, 4, 4).unsqueeze(0),
            x.view(1, 100, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .view(1, 100)
        .permute(1, 0)
    )
    assert (f[:, 1] - b).abs().max() < 1e-5


def test_encoding_gradients():
    mlh = MultiLevelHybridHashEncoding(
        n_encodings=2**14,
        n_input_dims=2,
        n_embed_dims=1,
        n_levels=4,
        min_res=32,
        max_res=64,
        max_n_dense=48 * 48,
    )
    # Set all embeddings to one
    mlh.level_emb_matrix0.data.copy_(torch.ones_like(mlh.level_emb_matrix0))
    mlh.level_emb_matrix1.data.copy_(torch.ones_like(mlh.level_emb_matrix1))
    mlh.level_emb_matrix2.data.copy_(torch.ones_like(mlh.level_emb_matrix2))
    mlh.level_emb_matrix3.data.copy_(torch.ones_like(mlh.level_emb_matrix3))

    f = mlh(torch.tensor([[0.31231, 0.7312]]))
    loss = f.square().sum()
    loss.backward()
    # If there isn't a collison, we should get 4 embedding vectors to receive
    # gradient (scaled by the weights of the bilinear interpolation). Note, the
    # coordinate above is chosen, so that on no level it falls exactly one a grid
    # point
    # mask = mlh.embmatrix.grad[..., 1] != 0.0
    # print(mlh.embmatrix.grad[..., 1][mask])
    assert (mlh.level_emb_matrix0.grad != 0.0).sum().int() == 4
    assert (mlh.level_emb_matrix1.grad != 0.0).sum().int() == 4
    assert (mlh.level_emb_matrix2.grad != 0.0).sum().int() == 4
    assert (mlh.level_emb_matrix3.grad != 0.0).sum().int() == 4

    # 3D
    mlh = MultiLevelHybridHashEncoding(
        n_encodings=2**14,
        n_input_dims=3,
        n_embed_dims=1,
        n_levels=4,
        min_res=32,
        max_res=64,
        max_n_dense=48 * 48 * 48,
    )
    # Set all embeddings to one
    mlh.level_emb_matrix0.data.copy_(torch.ones_like(mlh.level_emb_matrix0))
    mlh.level_emb_matrix1.data.copy_(torch.ones_like(mlh.level_emb_matrix1))
    mlh.level_emb_matrix2.data.copy_(torch.ones_like(mlh.level_emb_matrix2))
    mlh.level_emb_matrix3.data.copy_(torch.ones_like(mlh.level_emb_matrix3))

    f = mlh(torch.tensor([[0.31231, 0.7312, 0.12345]]))
    loss = f.square().sum()
    loss.backward()
    # If there isn't a collison, we should get 4 embedding vectors to receive
    # gradient (scaled by the weights of the bilinear interpolation). Note, the
    # coordinate above is chosen, so that on no level it falls exactly one a grid
    # point
    # mask = mlh.embmatrix.grad[..., 1] != 0.0
    # print(mlh.embmatrix.grad[..., 1][mask])
    assert (mlh.level_emb_matrix0.grad != 0.0).sum().int() == 8
    assert (mlh.level_emb_matrix1.grad != 0.0).sum().int() == 8
    assert (mlh.level_emb_matrix2.grad != 0.0).sum().int() == 8
    assert (mlh.level_emb_matrix3.grad != 0.0).sum().int() == 8


def _bilinear_interpolate(input: torch.Tensor, x: torch.Tensor):
    """Testing method to compare with grid_sample.

    Should equal grid_sample with options:
        padding=0
        mode=bilinear
        align_corners=False

    Params:
        input: (C,H,W), (C,D,H,W)
        x: (B,2) or (B,3) in range [-1,+1]

    Returns:
        interp: (B,C)
    """
    C, *spatial = input.shape

    # unnormalized coords
    xu = (x + 1) * torch.tensor([spatial[::-1]]) * 0.5 - 0.5
    # get bilinear weights
    cids, w, m = _compute_bilinear_params(xu, spatial)
    # Mark all corner ids outside to select zero-th element instead
    cids[~m] = 0

    dims = len(spatial)
    if dims == 2:
        input = input.permute(1, 2, 0)  # HWC
        v = input[cids[..., 1], cids[..., 0]].clone()  # (B,4,C)
    else:
        input = input.permute(1, 2, 3, 0)  # DHWC
        v = input[cids[..., 2], cids[..., 1], cids[..., 0]].clone()  # (B,8,C)

    # Zero pad
    v[~m] = 0.0
    # Compute result
    return (v * w[..., None]).sum(1)


def test_bilinear_interpolation():
    H, W = 200, 320
    img = torch.randn((2, H, W))
    x = torch.empty((1000, 2)).uniform_(-1.0, 1.0)

    y = (
        F.grid_sample(
            img.unsqueeze(0),
            x.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .view(2, 1000)
        .permute(1, 0)
        .clone()
    )

    # we call the testing function instead of
    # compute_bilinear_params directly
    yhat = _bilinear_interpolate(img, x)
    assert (y - yhat).abs().max() < 1e-4


def test_trilinear_interpolation():
    D, H, W = 100, 200, 320
    img = torch.randn((2, D, H, W))
    x = torch.empty((1000, 3)).uniform_(-1.0, 1.0)

    y = (
        F.grid_sample(
            img.unsqueeze(0),
            x.view(1, -1, 1, 1, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .view(2, 1000)
        .permute(1, 0)
        .clone()
    )

    # we call the testing function instead of
    # compute_bilinear_params directly
    yhat = _bilinear_interpolate(img, x)
    assert (y - yhat).abs().max() < 1e-4


def test_hash_ravel():
    import numpy as np

    D, H, W = 10, 4, 6
    x = torch.tensor([[0, 0, 0], [1, 1, 1], [5, 3, 9], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

    ids_2d_numpy = np.ravel_multi_index(x[..., :2].T.numpy(), (W, H), order="F")
    ids_3d_numpy = np.ravel_multi_index(x.T.numpy(), (W, H, D), order="F")

    ids_2d = _hash_ravel(x[..., :2], shape=(H, W))
    ids_3d = _hash_ravel(x, shape=(D, H, W))
    # print(ids_3d, ids_3d_numpy)
    assert_close(ids_2d, torch.tensor(ids_2d_numpy).long())
    assert_close(ids_3d, torch.tensor(ids_3d_numpy).long())
