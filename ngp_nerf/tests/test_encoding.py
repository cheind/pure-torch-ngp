import torch
from ngp_nerf.encoding import MultiLevelHashEncoding


def test_encoding_output_shape_3d():
    mlh = MultiLevelHashEncoding(
        n_encodings=2**16,
        n_input_dims=3,
        n_embed_dims=2,
        n_levels=4,
        min_res=32,
        max_res=64,
    )

    f = mlh(torch.empty((10, 3)).uniform_(-1, 1))
    assert f.shape == (10, 4, 2)


def test_encoding_output_shape_2d():
    mlh = MultiLevelHashEncoding(
        n_encodings=2**14,
        n_input_dims=2,
        n_embed_dims=2,
        n_levels=4,
        min_res=32,
        max_res=64,
    )

    f = mlh(torch.empty((10, 2)).uniform_(-1, 1))
    assert f.shape == (10, 4, 2)


def test_encoding_gradients():
    mlh = MultiLevelHashEncoding(
        n_encodings=2**14,
        n_input_dims=2,
        n_embed_dims=1,
        n_levels=4,
        min_res=32,
        max_res=64,
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
