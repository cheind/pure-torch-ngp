import torch
import torch.nn
import torch.nn.functional as F

from . import hashing


class MultiLevelHashEncoding(torch.nn.Module):
    """Multi Resolution Hash Encoding module.

    Encodes 2D/3D query locations by extracting bilinear interpolated
    encoding vectors from varying grid resolutions.

    This implementation uses a global encoding matrix and equal number
    of encoding vectors for each resolution level. The module employs the
    proposed hashing method to identify the indices of (potentially shared)
    encoding vectors for each grid vertex. The grids (called levelmaps) are
    treated as images/volumes and can be pre-computed. The query
    locations are reshaped into a grid format compatible with `F.grid_sample`
    to interpolate the encodings.

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
            growth_factor = torch.exp(
                (
                    torch.log(torch.tensor(self.max_res))
                    - torch.log(torch.tensor(self.min_res))
                )
                / (self.n_levels - 1)
            )
            resolutions = torch.floor(
                torch.tensor(
                    [
                        self.min_res * growth_factor**level
                        for level in range(self.n_levels)
                    ]
                )
            ).long()
            # For each resolution R, we form a (E,R,R,R) levelmap whose vertices
            # contain a view of the global embedding vectors in first dimension.
            # The encoding vector indices are computed by hashing the spatial
            # grid location. We precompute the encoding vector indices here.
            for level, res in enumerate(resolutions):
                n_level_encodings = min(res**self.n_input_dims, self.n_encodings)
                index_coords = torch.stack(
                    torch.meshgrid(
                        [torch.arange(res)] * self.n_input_dims, indexing="ij"
                    ),
                    -1,
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
                        torch.empty(n_embed_dims, n_level_encodings).uniform_(-0.5, 0.5)
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


if __name__ == "__main__":
    mlh3d = MultiLevelHashEncoding(
        n_encodings=2**14,
        n_input_dims=3,
        n_embed_dims=2,
        n_levels=8,
        min_res=32,
        max_res=512,
    )
    f = mlh3d(torch.empty((10000, 3)).uniform_(-1, 1))

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
