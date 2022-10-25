from dataclasses import dataclass

import torch
import torch.nn
import torch.nn.functional as F

from .. import pixels


@dataclass
class LevelInfo:
    res: int
    shape: tuple[int, ...]
    dense: bool
    hashing: bool
    n_encodings: int


class MultiLevelHybridHashEncoding(torch.nn.Module):
    def __init__(
        self,
        n_encodings: int = 2**14,
        n_input_dims: int = 3,
        n_embed_dims: int = 2,
        n_levels: int = 16,
        min_res: int = 16,
        max_res: int = 512,
        max_n_dense: int = 2 ** (8 * 3),
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        assert n_input_dims in [2, 3], "Only 2D and 3D inputs are supported"

        self.n_levels = n_levels
        self.n_input_dims = n_input_dims
        self.n_embed_dims = n_embed_dims
        self.max_n_encodings = n_encodings
        self.min_res = min_res
        self.max_res = max_res
        self.max_n_dense = max_n_dense

        self.level_infos: list[LevelInfo] = []

        resolutions = _compute_resolutions(
            n_levels=n_levels, min_res=min_res, max_res=max_res
        )

        for level, res in enumerate(resolutions):
            n_elements = res**self.n_input_dims
            li = LevelInfo(
                res=res,
                shape=(res,) * self.n_input_dims,
                dense=n_elements <= self.max_n_dense,
                hashing=n_elements > self.max_n_encodings,
                n_encodings=min(n_elements, self.max_n_encodings),
            )
            self.level_infos.append(li)

            # Note: the embedding matrices for dense (E,T) and sparse (T,E)
            # levels are permuted. This is done to better match its usage.

            if li.dense:
                dense_ids = self._compute_dense_indices(li)
                self.register_buffer(
                    "level_emb_indices" + str(level),
                    dense_ids,
                )
                self.register_parameter(
                    "level_emb_matrix" + str(level),
                    torch.nn.Parameter(
                        torch.empty(n_embed_dims, li.n_encodings).uniform_(
                            -init_scale, init_scale
                        )
                    ),
                )
            else:
                # Add extra encoding that will attract all locs outside.
                emb = torch.empty(li.n_encodings, n_embed_dims).uniform_(
                    -init_scale, init_scale
                )
                self.register_parameter(
                    "level_emb_matrix" + str(level),
                    torch.nn.Parameter(emb),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the multi-resolutional feature emebeddings for all query locations.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].

        Returns:
            features: (B,L,E) array of features with dims (E)
                for each query (B) and level (L).
        """
        features = []
        for level, li in enumerate(self.level_infos):
            if li.dense:
                f = self._forward_dense(x, level)
            else:
                f = self._forward_sparse(x, level)
            features.append(f)
        f = torch.stack(features, 1)
        return f

    def _forward_dense(self, x: torch.Tensor, level: int):
        """Returns the multi-resolutional feature emebeddings for all query locations.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].
            level: level index

        Returns:
            features: (B,E) array of features with dims (E)
                for each query (B).
        """
        B, D = x.shape
        # Note, we re-interpret the query locations as a sampling grid by
        # by shuffling the batch dimension into the first image dimension.
        # Turned out to be faster (many times on cpu) than using expand.
        x = x.view(1, B, *([1] * (D - 1)), D)  # (1,B,1,2) or (1,B,1,1,3)
        indices = getattr(self, "level_emb_indices" + str(level))
        embmatrix = getattr(self, "level_emb_matrix" + str(level))
        # Note for myself: don't store the following as a buffer, it won't work.
        # We need to perform the indexing on-line.
        levelmap = embmatrix[:, indices].unsqueeze(0)
        # Bilinearly interpolate the sampling locations using the levelmap.
        f = F.grid_sample(
            levelmap,
            x,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (1,E,B,1) or (1,E,B,1,1)
        # Shuffle back into (B,E) tensor
        return f.view(self.n_embed_dims, B).permute(1, 0)

    def _forward_sparse(self, x: torch.Tensor, level: int):
        embmatrix = getattr(self, "level_emb_matrix" + str(level))
        ids, w = self._compute_sparse_indices(x, level)
        # (B,4,E) or (B,8,E) -> (B,E)
        return (embmatrix[ids] * w[..., None]).sum(1)

    @torch.no_grad()
    def _compute_dense_indices(self, li: LevelInfo) -> torch.LongTensor:
        index_coords = pixels.generate_grid_coords(li.shape, indexing="xy")
        if not li.hashing:
            ids = _hash_ravel(index_coords, li.shape, li.n_encodings)
        else:
            ids = _hash_xor(index_coords, li.n_encodings)
        return ids

    @torch.no_grad()
    def _compute_sparse_indices(self, x: torch.Tensor, level: int):
        li = self.level_infos[level]

        # Normalized to pixel [-0.5,R+0.5]
        x = (x + 1) * li.res * 0.5 - 0.5  # (B,C)
        c, w, m = _compute_bilinear_params(x, li.shape)

        # Compute indices
        if li.hashing:
            ids = _hash_xor(c, li.n_encodings)
        else:
            ids = _hash_ravel(c, li.shape, li.n_encodings)

        # Point outside elements to the first element, but set
        # all weights zero to simulate zero-padding.
        w[~m] = 0.0
        ids[~m] = 0
        return ids, w


def _compute_resolutions(
    n_levels: int = 16,
    min_res: int = 16,
    max_res: int = 512,
):
    """Computes grid resolutions for each level

    Equation 2 and 3 in the paper to determine the number of grid vertices
    per resolution level

    https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    """
    growth_factor = torch.exp(
        (torch.log(torch.tensor(max_res)) - torch.log(torch.tensor(min_res)))
        / (n_levels - 1)
    )
    resolutions = (
        torch.floor(
            torch.tensor(
                [min_res * growth_factor**level for level in range(n_levels)]
            )
        )
        .long()
        .tolist()
    )
    return resolutions


def _compute_bilinear_params(
    x: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    """Computes bilinear/trilinear interpolation parameter

    Params:
        x: (B,C) points with C=2 or C=3
        shape: (H,W) or (D,H,W) of interpolation grid size

    Returns:
        c: (B,4,2) or (B,8,2) corner integer coordinates
        w: (B,4) or (B,8) weights per corner (sum to one)
        m: (B) mask of valid points x.
    """
    B, C = x.shape
    if C == 2:
        c, w, m = _bilinear_params_2d(x, shape)
    elif C == 3:
        c, w, m = _bilinear_params_3d(x, shape)
    else:
        raise NotImplementedError
    return c, w, m


def _bilinear_params_2d(
    x: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    o = x.new_tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    xl = x.floor()
    xf = x - xl

    # Compute corners
    c = (xl.long().unsqueeze(1) + o[None, ...]).contiguous()

    # Compute mask
    m = ((c >= 0) & (c < c.new_tensor(shape[::-1])[None, None, :])).all(-1)  # B,4

    # Compute weights
    one_min = 1.0 - xf
    w11 = one_min[:, 0] * one_min[:, 1]
    w12 = one_min[:, 0] * xf[:, 1]
    w21 = xf[:, 0] * one_min[:, 1]
    w22 = xf[:, 0] * xf[:, 1]
    w = torch.stack((w11, w12, w21, w22), 1)  # B,4

    return c, w, m


def _bilinear_params_3d(
    x: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    o = x.new_tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )

    xl = x.floor()
    xf = x - xl

    # Compute corners
    c = (xl.long().unsqueeze(1) + o[None, ...]).contiguous()

    # Compute mask
    m = ((c >= 0) & (c < c.new_tensor(shape[::-1])[None, None, :])).all(-1)  # B,8

    # Compute weights
    one_min = 1 - xf
    w000 = one_min[:, 0] * one_min[:, 1] * one_min[:, 2]
    w001 = one_min[:, 0] * one_min[:, 1] * xf[:, 2]
    w010 = one_min[:, 0] * xf[:, 1] * one_min[:, 2]
    w011 = one_min[:, 0] * xf[:, 1] * xf[:, 2]
    w100 = xf[:, 0] * one_min[:, 1] * one_min[:, 2]
    w101 = xf[:, 0] * one_min[:, 1] * xf[:, 2]
    w110 = xf[:, 0] * xf[:, 1] * one_min[:, 2]
    w111 = xf[:, 0] * xf[:, 1] * xf[:, 2]

    w = torch.stack((w000, w001, w010, w011, w100, w101, w110, w111), 1)  # B,8

    return c, w, m


def _hash_xor(x: torch.LongTensor, n_indices: int) -> torch.LongTensor:
    # See https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
    N = x.dim() - 1
    Q = x.shape[-1]
    pi = torch.tensor([1, 2654435761, 805459861], dtype=x.dtype, device=x.device)[:Q]
    pi = pi.view(*([1] * N), Q)

    xpi_int = x * pi
    indices = xpi_int[..., 0]
    for d in range(1, Q):
        indices = torch.bitwise_xor(indices, xpi_int[..., d])
    return indices % n_indices


def _hash_ravel(
    x: torch.LongTensor, shape: tuple[int, ...], n_indices: int
) -> torch.LongTensor:
    Q = x.shape[-1]
    assert Q in [2, 3], "Only implemented for 2D and 3D"
    if Q == 2:
        indices = x[..., 0] + x[..., 1] * shape[0]
    elif Q == 3:
        indices = x[..., 0] + x[..., 1] * shape[0] + x[..., 2] * (shape[0] * shape[1])
    return indices % n_indices