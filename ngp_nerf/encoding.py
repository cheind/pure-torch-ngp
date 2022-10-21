import torch
import torch.nn
import torch.nn.functional as F

from . import hashing, pixels, interpolation


def compute_resolutions(
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


class MultiLevelHashEncoding(torch.nn.Module):
    """Multi Resolution Hash Encoding module.

    Encodes 2D/3D query locations by extracting bilinear interpolated
    encoding vectors from varying grid resolutions.

    This implementation uses a an encoding matrix per layer with
    encoding `n=min(res**dims,n_encodings)` encoding vectors. For each
    grid vertex, the proposed hashing/direct mapping method is used
    to identify the corresponding encoding indices. The indices are pre-computed
    per level and stored as dense images/volumes. The query locations are reshaped
    into a grid format compatible with `F.grid_sample` to interpolate the encodings.

    Images/Volumes to be samples are stored densly and can thus lead
    to huge memory consumptions independent of the number of queries.

    Based on
    https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    """

    def __init__(
        self,
        n_encodings: int = 2**14,
        n_input_dims: int = 3,
        n_embed_dims: int = 2,
        n_levels: int = 16,
        min_res: int = 16,
        max_res: int = 512,
        init_scale: float = 1e-4,
    ) -> None:
        """Initialize the module.

        Params:
            n_encodings: Number of encoding vectors per resolution level
            n_input_dims: Number of query dimensions (2 or 3).
            n_embed_dims: Number of embedding dimensions.
            n_levels: Number of levels to generate
            min_res: Lowest grid resolution (isotropic in each dimension)
            max_res: Highest grid resolution (isotropic in each dimension)
        """
        super().__init__()

        assert n_input_dims in [2, 3], "Only 2D and 3D inputs are supported"

        self.n_levels = n_levels
        self.n_input_dims = n_input_dims
        self.n_embed_dims = n_embed_dims
        self.n_encodings = n_encodings
        self.min_res = min_res
        self.max_res = max_res

        with torch.no_grad():
            resolutions = compute_resolutions(
                n_levels=n_levels, min_res=min_res, max_res=max_res
            )

            # For each resolution R, we form a (E,R,R,R) levelmap whose vertices
            # contain a view of the global embedding vectors in first dimension.
            # The encoding vector indices are computed by hashing the spatial
            # grid location. We precompute the encoding vector indices here.
            for level, res in enumerate(resolutions):
                n_level_encodings = min(res**self.n_input_dims, self.n_encodings)
                index_coords = pixels.generate_grid_coords(
                    [res] * self.n_input_dims, indexing="xy"
                )
                if res**self.n_input_dims <= n_level_encodings:
                    res_shape = [res] * self.n_input_dims
                    indices = hashing.ravel_index(
                        index_coords, res_shape, n_level_encodings
                    )
                else:
                    indices = hashing.xor_index_hashing(index_coords, n_level_encodings)
                self.register_buffer(
                    "level_emb_indices" + str(level),
                    indices,
                )
                self.register_parameter(
                    "level_emb_matrix" + str(level),
                    torch.nn.Parameter(
                        torch.empty(n_embed_dims, n_level_encodings).uniform_(
                            -init_scale, init_scale
                        )
                    ),
                )

    def forward(self, x):
        """Returns the multi-resolutional feature emebeddings for all query locations.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].

        Returns:
            features: (B,L,E) array of features with dims (E)
                for each query (B) and level (L).
        """
        B, D = x.shape
        # Note, we re-interpret the query locations as a sampling grid by
        # by shuffling the batch dimension into the first image dimension.
        # Turned out to be faster (many times on cpu) than using expand.
        x = x.view(1, B, *([1] * (D - 1)), D)  # (1,B,1,2) or (1,B,1,1,3)
        features = []
        for level in range(self.n_levels):
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
            features.append(f.view(self.n_embed_dims, B).permute(1, 0))
        f = torch.stack(features, 1)
        return f


class MultiLevelSparseHashEncoding(torch.nn.Module):
    """Multi Resolution Sparse Hash Encoding module.

    Similar to `MultiLevelHashEncoding`, but computes hashing and bilinear
    interpolation on the fly. Therfore brings memory advantages over
    `MultiLevelHashEncoding` for large resolutions, but is slower at query
    time.

    Based on
    https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    """

    def __init__(
        self,
        n_encodings: int = 2**14,
        n_input_dims: int = 3,
        n_embed_dims: int = 2,
        n_levels: int = 16,
        min_res: int = 16,
        max_res: int = 512,
        init_scale: float = 1e-4,
    ) -> None:
        """Initialize the module.

        Params:
            n_encodings: Number of encoding vectors per resolution level
            n_input_dims: Number of query dimensions (2 or 3).
            n_embed_dims: Number of embedding dimensions.
            n_levels: Number of levels to generate
            min_res: Lowest grid resolution (isotropic in each dimension)
            max_res: Highest grid resolution (isotropic in each dimension)
        """
        super().__init__()

        assert n_input_dims in [2, 3], "Only 2D and 3D inputs are supported"

        self.n_levels = n_levels
        self.n_input_dims = n_input_dims
        self.n_embed_dims = n_embed_dims
        self.n_encodings = n_encodings
        self.min_res = min_res
        self.max_res = max_res

        with torch.no_grad():
            # Equation 2 and 3 in the paper to determine the number of grid vertices
            # per resolution level
            self.resolutions = compute_resolutions(
                n_levels=n_levels, min_res=min_res, max_res=max_res
            )
            self.direct_mappings = [
                (res**self.n_input_dims < self.n_encodings)
                for res in self.resolutions
            ]
            self.n_level_encodings = [
                min(res**self.n_input_dims, self.n_encodings)
                for res in self.resolutions
            ]

            for level, n_level_encodings in enumerate(self.n_level_encodings):
                emb = torch.empty(n_level_encodings + 1, n_embed_dims).uniform_(
                    -init_scale, init_scale
                )
                emb[-1, :] = 0.0
                self.register_parameter(
                    "level_emb_matrix" + str(level),
                    torch.nn.Parameter(emb),
                )

    def forward(self, x):
        """Returns the multi-resolutional feature emebeddings for all query locations.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].

        Returns:
            features: (B,L,E) array of features with dims (E)
                for each query (B) and level (L).
        """
        B, D = x.shape
        features = []
        for level in range(self.n_levels):
            embmatrix = getattr(self, "level_emb_matrix" + str(level))
            ids, w, _ = self._find_embeddings(x, level)

            # f = embmatrix[:, ids]  # (B,4,E)
            f = (embmatrix[ids] * w[..., None]).sum(1)
            features.append(f)
        f = torch.stack(features, 1)
        return f

    @torch.no_grad()
    def _find_embeddings(self, x: torch.Tensor, level: int):
        R = self.resolutions[level]
        direct = self.direct_mappings[level]
        n_encs = self.n_level_encodings[level]
        res = [R] * self.n_input_dims
        # Normalized to pixel [-0.5,R+0.5]
        x = (x + 1) * R * 0.5 - 0.5  # (B,C)

        c, w, m = interpolation.linear_interpolate_info(x, res)

        if direct:
            ids = hashing.ravel_index(c, res, n_encs)
        else:
            ids = hashing.xor_index_hashing(c, n_encs)

        ids[~m] = n_encs  # point to last+1

        return ids, w, m


if __name__ == "__main__":
    mlh_dense = MultiLevelHashEncoding(
        n_encodings=2**8,
        n_input_dims=2,
        n_embed_dims=1,
        n_levels=2,
        min_res=4,
        max_res=256,
        init_scale=1.0,
    )

    mlh_sparse = MultiLevelSparseHashEncoding(
        n_encodings=2**8,
        n_input_dims=2,
        n_embed_dims=1,
        n_levels=2,
        min_res=4,
        max_res=256,
        init_scale=1.0,
    )
    mlh_sparse.level_emb_matrix0.data[:-1].copy_(mlh_dense.level_emb_matrix0.T)
    mlh_sparse.level_emb_matrix1.data[:-1].copy_(mlh_dense.level_emb_matrix1.T)

    with torch.no_grad():
        x = torch.empty((100, 2)).uniform_(-1, 1.0)
        f_dense = mlh_dense(x)
        f_sparse = mlh_sparse(x)
        print((f_dense - f_sparse).abs().sum())

        x = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, 1.0]])
        f_dense = mlh_dense(x)
        f_sparse = mlh_sparse(x)
        print((f_dense - f_sparse).abs().sum())

    # 3D
    mlh_dense = MultiLevelHashEncoding(
        n_encodings=2**8,
        n_input_dims=3,
        n_embed_dims=1,
        n_levels=2,
        min_res=4,
        max_res=256,
        init_scale=1.0,
    )

    mlh_sparse = MultiLevelSparseHashEncoding(
        n_encodings=2**8,
        n_input_dims=3,
        n_embed_dims=1,
        n_levels=2,
        min_res=4,
        max_res=256,
        init_scale=1.0,
    )
    mlh_sparse.level_emb_matrix0.data[:-1].copy_(mlh_dense.level_emb_matrix0.T)
    mlh_sparse.level_emb_matrix1.data[:-1].copy_(mlh_dense.level_emb_matrix1.T)

    with torch.no_grad():
        x = torch.empty((100, 3)).uniform_(-1, 1.0)
        f_dense = mlh_dense(x)
        f_sparse = mlh_sparse(x)
        print((f_dense - f_sparse).abs().sum())

        x = torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]
        )
        f_dense = mlh_dense(x)
        f_sparse = mlh_sparse(x)
        print((f_dense - f_sparse).abs().sum())

    # Performance

    mlh_dense = MultiLevelHashEncoding(
        n_encodings=2**12,
        n_input_dims=3,
        n_embed_dims=2,
        n_levels=32,
        min_res=64,
        max_res=2**8,
    ).cuda()

    mlh_sparse = MultiLevelSparseHashEncoding(
        n_encodings=2**12,
        n_input_dims=3,
        n_embed_dims=2,
        n_levels=32,
        min_res=64,
        max_res=2**8,
    ).cuda()

    x = torch.empty((100000, 3)).uniform_(-1, 1.0).cuda()

    import time

    with torch.no_grad():
        for _ in range(10):
            mlh_dense(x)
            mlh_sparse(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            mlh_dense(x)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 10)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            mlh_sparse(x)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end) / 10)

    # print(f_dense)
    # print(f_sparse)

    # mlh2d = MultiLevelHashEncoding(
    #     n_encodings=2**14,
    #     n_input_dims=2,
    #     n_embed_dims=2,
    #     n_levels=4,
    #     min_res=32,
    #     max_res=512,
    # )
    # f = mlh2d(torch.empty((10000, 2)).uniform_(-1, 1))
    # print(f.shape)

    # import torch.optim

    # mlh2d = MultiLevelHashEncoding(
    #     n_encodings=1024,
    #     n_input_dims=2,
    #     n_embed_dims=2,
    #     n_levels=1,
    #     min_res=32,
    #     max_res=32,
    # ).cuda()

    # opt = torch.optim.SGD(mlh2d.parameters(), lr=1e-1)
    # for _ in range(100):
    #     myfeatures = mlh2d(torch.zeros(20000, 2).cuda())
    #     loss = F.mse_loss(myfeatures, torch.zeros_like(myfeatures))
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()
    #     print(loss.item())
