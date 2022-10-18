import torch
import torch.nn
import torch.nn.functional as F


class MultiLevelHashEncoding(torch.nn.Module):
    def __init__(
        self,
        n_encodings: int = 2**14,
        n_input_dims: int = 3,
        n_embed_dims: int = 2,
        n_levels: int = 16,
        min_res: int = 16,
        max_res: int = 512,
    ) -> None:
        super().__init__()

        assert n_input_dims in [2, 3], "Only 2D and 3D inputs are supported"

        embmatrix = torch.empty(n_embed_dims, n_encodings, n_levels).uniform_(-0.5, 0.5)
        self.register_parameter("embmatrix", torch.nn.Parameter(embmatrix))
        self.embmatrix: torch.Tensor

        self.n_levels = n_levels
        self.n_input_dims = n_input_dims
        self.n_encodings = n_encodings
        self.min_res = min_res
        self.max_res = max_res

        resolutions = torch.ceil(
            torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(self.min_res)),
                    torch.log(torch.tensor(self.max_res)),
                    self.n_levels,
                )
            )
        ).int()
        print("Pregenerating data")
        for level, res in enumerate(resolutions):
            # At each resolution R, we form a (E,R,R,R) levelmap whose vertices
            # contain the embedding vectors given by the corresponding hashed
            # index from position.
            xyz_int = self._levelmap_embedding_indices(res)  # (R,R,R)
            self.register_buffer("level_embedding_indices" + str(level), xyz_int)

    @torch.no_grad()
    def _levelmap_embedding_indices(self, res: int) -> torch.LongTensor:
        """Returns the embedding indices for each vertex in a grid of given resolution.

        Params:
            res: Grid resolution in each direction

        Returns:
            indices: (R,R) or (R,R,R) tensor of embedding indices in range [0,n_encodings]
        """
        res_range = torch.arange(res)
        xyz = torch.meshgrid([res_range] * self.n_input_dims, indexing="ij")

        # TODO: don't hash if res**self.n_input_dims < self.n_encodings

        pis = torch.tensor([1, 2654435761, 805459861])[: self.n_input_dims]
        pis = pis.view(*([1] * self.n_input_dims), self.n_input_dims)
        xyz = torch.stack(xyz, -1)
        xyz_pi_int = (xyz * pis).long()
        xyz_int = xyz_pi_int[..., 0]
        for d in range(1, self.n_input_dims):
            xyz_int = torch.bitwise_xor(xyz_int, xyz_pi_int[..., d]) % self.n_encodings

        u = torch.unique(xyz_int)
        print(u.numel(), xyz_int.numel())
        print(f"res {res} collisions:", (xyz_int.numel() - u.numel()) / xyz_int.numel())

        return xyz_int

    def forward(self, x):
        """Returns the multi-resolutional feature emebeddings for all query locations.

        Params:
            x: (B,2) or (B,3) array of query locations in value range [-1,+1].

        Returns:
            features: (B,L,E) array of features with dims (E)
                for each query (B) and level (L).
        """
        B, D = x.shape
        x = x.view(1, B, *([1] * (D - 1)), D)  # (1,B,1,2) or (1,B,1,1,3)
        features = []
        for level in range(self.n_levels):
            indices = getattr(self, "level_embedding_indices" + str(level))
            # Note for myself: don't store the following as a buffer. it won't work.

            levelmap = self.embmatrix[:, indices, level].unsqueeze(0)
            f = F.grid_sample(
                levelmap,
                x,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # (1,E,B,1) or (1,E,B,1,1)
            features.append(f.view(self.embmatrix.shape[0], B).permute(1, 0))
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
    print(f.shape)

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
