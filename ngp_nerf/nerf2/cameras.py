import torch
import torch.nn
from typing import Union, Optional


class BaseCamera(torch.nn.Module):
    """Base class for perspective cameras.

    All camera models treat pixels as squares with pixel centers
    corresponding to integer pixel coordinates. That is, a pixel
    (u,v) extends from (u+0.5,v+0.5) to `(u+0.5,v+0.5)`.

    In addition we generalize to `N` cameras by batching along
    the first dimension.
    """

    def __init__(self):
        super().__init__()
        self.focal_length: torch.Tensor
        self.principal_point: torch.Tensor
        self.R: torch.Tensor
        self.T: torch.Tensor
        self.size: torch.Tensor
        self.tnear: torch.Tensor
        self.tfar: torch.Tensor

    @property
    def K(self):
        N = self.focal_length.shape[0]
        K = self.focal_length.new_zeros((N, 3, 3))
        K[:, 0, 0] = self.focal_length[:, 0]
        K[:, 1, 1] = self.focal_length[:, 1]
        K[:, 0, 2] = self.principal_point[:, 0]
        K[:, 1, 2] = self.principal_point[:, 1]
        K[:, 2, 2] = 1
        return K

    @property
    def t4x4(self):
        N = self.focal_length.shape[0]
        t = self.focal_length.new_zeros((N, 4, 4))
        t[:, :3, :3] = self.R
        t[:, :3, 3] = self.T
        t[:, -1, -1] = 1
        return t

    def uv_grid(self):
        """Generates uv-pixel grid coordinates.

        Returns:
            uv: (N,H,W,2) tensor of grid coordinates using
                'xy' indexing.
        """
        N = self.focal_length.shape[0]
        dev = self.focal_length.device
        dtype = self.focal_length.dtype
        uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.size[0, 0], dtype=dtype, device=dev),
                    torch.arange(self.size[0, 1], dtype=dtype, device=dev),
                    indexing="xy",
                ),
                -1,
            )
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
        )
        return uv

    def unproject_uv(
        self, uv: torch.Tensor = None, depth: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """Unprojects uv-pixel coordinates to view space.

        Params:
            uv: (N,...,2) uv coordinates to unproject. If not specified, defaults
                to all grid coordiantes.
            depth: scalar or any shape broadcastable to (N,...,1) representing
                the depths of unprojected pixels.

        Returns:
            xyz: (N,...,3) tensor of coordinates.
        """
        # uv is (N,...,2)
        if uv is None:
            uv = self.uv_grid()
        N = self.focal_length.shape[0]
        mbatch = uv.shape[1:-1]
        mbatch_ones = (1,) * len(mbatch)

        if not torch.is_tensor(depth):
            depth = uv.new_tensor(depth)

        depth = depth.expand((N,) + mbatch + (1,))
        pp = self.principal_point.view((N,) + mbatch_ones + (2,))
        fl = self.focal_length.view((N,) + mbatch_ones + (2,))

        xy = (uv - pp) / fl * depth
        xyz = torch.cat((xy, depth), -1)
        return xyz


class Camera(BaseCamera):
    """A single perspective camera.

    To comply with BaseCamera and batching, parameters are
    stored with a prepended batch dimension `N=1`.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        R: torch.tensor = None,
        T: torch.tensor = None,
        tnear: float = 0.0,
        tfar: float = 100.0,
    ) -> None:
        super().__init__()
        if R is None:
            R = torch.eye(3)
        if T is None:
            T = torch.zeros(3)
        self.register_buffer("focal_length", torch.tensor([[fx, fy]]).float())
        self.register_buffer("principal_point", torch.tensor([[cx, cy]]).float())
        self.register_buffer("size", torch.tensor([[width, height]]).int())
        self.register_buffer("R", R.unsqueeze(0).float())
        self.register_buffer("T", T.unsqueeze(0).float())
        self.register_buffer("tnear", torch.tensor([[tnear]]).float())
        self.register_buffer("tfar", torch.tensor([[tfar]]).float())


class CameraBatch(BaseCamera):
    """A batch of N perspective cameras."""

    def __init__(self, cams: list[Camera]) -> None:
        super().__init__()
        self.register_buffer(
            "focal_length", torch.cat([c.focal_length for c in cams], 0)
        )
        self.register_buffer(
            "principal_point", torch.cat([c.principal_point for c in cams], 0)
        )
        self.register_buffer("size", torch.cat([c.size for c in cams], 0))
        self.register_buffer("R", torch.cat([c.R for c in cams]), 0)
        self.register_buffer("T", torch.cat([c.T for c in cams]), 0)
        self.register_buffer("tnear", torch.cat([c.tnear for c in cams]), 0)
        self.register_buffer("tfar", torch.cat([c.tfar for c in cams]), 0)


class MultiViewScene(torch.nn.Module):
    """A multiple view scene setup."""

    def __init__(
        self,
        cameras: BaseCamera,
        aabb: torch.Tensor,
        images: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cameras = cameras
        self.register_buffer("aabb", aabb)
        self.register_buffer("images", images)
        self.aabb: torch.Tensor
        self.images: torch.Tensor
