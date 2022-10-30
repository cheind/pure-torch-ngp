import torch
import torch.nn
from typing import Union


class MultiViewCamera(torch.nn.Module):
    """Perspective camera with multiple poses.

    All camera models treat pixels as squares with pixel centers
    corresponding to integer pixel coordinates. That is, a pixel
    (u,v) extends from (u+0.5,v+0.5) to `(u+0.5,v+0.5)`.

    In addition we generalize to `N` poses by batching along
    the first dimension for attributes `R` and `T`.
    """

    def __init__(
        self,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        size: torch.IntTensor,
        R: Union[torch.Tensor, list[torch.Tensor]],
        T: Union[torch.Tensor, list[torch.Tensor]],
        tnear: torch.Tensor,
        tfar: torch.Tensor,
    ) -> None:
        super().__init__()

        focal_length = torch.as_tensor(focal_length).view(2).float()
        principal_point = torch.as_tensor(principal_point).view(2).float()
        size = torch.as_tensor(size).view(2).int()
        tnear = torch.as_tensor(tnear).view(1).float()
        tfar = torch.as_tensor(tfar).view(1).float()
        if isinstance(R, list):
            R = torch.stack(R, 0)
        R = R.view(-1, 3, 3).float()
        if isinstance(T, list):
            T = torch.stack(T, 0)
        T = T.view(-1, 3, 1).float()

        self.register_buffer("focal_length", focal_length)
        self.register_buffer("principal_point", principal_point)
        self.register_buffer("size", size)
        self.register_buffer("tnear", tnear)
        self.register_buffer("tfar", tfar)
        self.register_buffer("R", R)
        self.register_buffer("T", T)

        self.focal_length: torch.Tensor
        self.principal_point: torch.Tensor
        self.size: torch.IntTensor
        self.tnear: torch.Tensor
        self.tfar: torch.Tensor
        self.R: torch.Tensor
        self.T: torch.Tensor

    def __getitem__(self, index) -> "MultiViewCamera":
        """Slice a subset of camera poses."""
        return MultiViewCamera(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            size=self.size,
            R=self.R[index],
            T=self.T[index],
            tnear=self.tnear,
            tfar=self.tfar,
        )

    @property
    def K(self):
        """Return the 3x3 camera intrinsic matrix"""
        K = self.R.new_zeros((3, 3))
        K[0, 0] = self.focal_length[0]
        K[1, 1] = self.focal_length[1]
        K[0, 2] = self.principal_point[0]
        K[1, 2] = self.principal_point[1]
        K[2, 2] = 1
        return K

    @property
    def E(self):
        """Return the (N,4,4) extrinsic pose matrix."""
        N = self.n_views
        t = self.R.new_zeros((N, 4, 4))
        t[:, :3, :3] = self.R
        t[:, :3, 3:4] = self.T
        t[:, -1, -1] = 1
        return t

    @property
    def n_views(self):
        """Return the number of views."""
        return self.R.shape[0]

    def make_uv_grid(self):
        """Generates uv-pixel grid coordinates.

        Returns:
            uv: (N,H,W,2) tensor of grid coordinates using
                'xy' indexing.
        """
        N = self.n_views
        dev = self.focal_length.device
        dtype = self.focal_length.dtype
        uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.size[0], dtype=dtype, device=dev),
                    torch.arange(self.size[1], dtype=dtype, device=dev),
                    indexing="xy",
                ),
                -1,
            )
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
        )
        return uv

    def extra_repr(self):
        return "\n".join([key + ": " + repr(mod) for key, mod in self._buffers.items()])


if __name__ == "__main__":
    torch.random.manual_seed(123)
    c = MultiViewCamera(
        focal_length=[100.0, 100.0],
        principal_point=[160, 120],
        size=[320, 200],
        R=[torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)],
        T=[torch.randn(3), torch.randn(3), torch.randn(3)],
        tnear=0.0,
        tfar=10.0,
    ).cuda()

    print(c[1:])
    print(c.K, c.E)
    print(c[1:].K, c[1:].E)
    print(c[1:].uv_grid().shape)
