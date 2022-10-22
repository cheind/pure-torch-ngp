import torch
import torch.nn.functional as F
from torch.testing import assert_close
from ngp_nerf.encoding import MultiLevelHybridHashEncoding


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
