import torch
from ngp_nerf.encoding import MultiLevelHashEncoding


def test_encoding_output_shape_3d():
    mlh = MultiLevelHashEncoding(
        n_encodings=2**14,
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
    mlh.embmatrix.data.copy_(torch.ones_like(mlh.embmatrix) * 2)

    f = mlh(torch.tensor([[0.125, 0.125]]))
    loss = f.square().sum()
    loss.backward()
    # If there isn't a collison, we should get 4 embedding vectors to receive
    # gradient of 1.0 per level
    assert (mlh.embmatrix.grad[..., 0] > 0.0).sum().int() == 4
    assert (mlh.embmatrix.grad[..., 1] > 0.0).sum().int() == 4
    assert (mlh.embmatrix.grad[..., 2] > 0.0).sum().int() == 4
    assert (mlh.embmatrix.grad[..., 3] > 0.0).sum().int() == 4
