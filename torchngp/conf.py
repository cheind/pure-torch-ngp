import torch
import dataclasses
from hydra_zen import make_custom_builds_fn

from . import functional

builds = make_custom_builds_fn(populate_full_signature=True)


class Camera(torch.nn.Module):
    def __init__(
        self, focal_length: float, size: tuple[int, int], T: torch.Tensor
    ) -> None:
        super().__init__()
        self.register_buffer("focal_length", torch.tensor([focal_length, focal_length]))
        self.register_buffer("size", torch.as_tensor(size))
        self.register_buffer("T", torch.as_tensor(T))

    def extra_repr(self) -> str:
        out = ""
        out += f"focal_length={self.focal_length}, "
        out += f"size={self.size}, "
        out += f"n_poses={self.T.shape[0]}"
        return out


@dataclasses.dataclass
class Pose:
    rx: float
    ry: float
    rz: float
    tx: float
    ty: float
    tz: float


def poses_to_tensor(poses: list[Pose]) -> torch.Tensor:
    rvecs = torch.tensor([(p.rx, p.ry, p.rz) for p in poses]).to(torch.float64)
    tvecs = torch.tensor([(p.tx, p.ty, p.tz) for p in poses])

    theta = torch.linalg.vector_norm(rvecs, dim=-1)
    axis = torch.zeros_like(rvecs)
    axis[..., 0] = 1.0
    mask = theta > 1e-7
    axis[mask] = rvecs[mask] / theta[mask][..., None]
    R = functional.rotation_matrix(axis, theta)

    T = torch.eye(4).unsqueeze(0).expand(rvecs.shape[0], 4, 4).contiguous()
    T[:, :3, :3] = R.float()
    T[:, :3, 3] = tvecs
    return T


class RadianceField(torch.nn.Module):
    def __init__(self, res: int = 64) -> None:
        super().__init__()
        self.res = res
        self.params = torch.nn.Linear(3, 3)


class SpatialFilter(torch.nn.Module):
    def __init__(self, res: int = 32) -> None:
        super().__init__()
        self.res = res


class BoxTransform(torch.nn.Module):
    def __init__(
        self, minc: tuple[float, float, float], maxc: tuple[float, float, float]
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", torch.as_tensor([minc, maxc]))


class Volume(torch.nn.Module):
    def __init__(
        self,
        rf: RadianceField,
        filter: SpatialFilter,
        boxt: BoxTransform,
    ) -> None:
        super().__init__()

        self.rf = rf
        self.filter = filter
        self.box_transform = boxt


class Scene(torch.nn.Module):
    def __init__(self, cams: dict[str, Camera], volume: Volume) -> None:
        super().__init__()
        self.cameras = torch.nn.ModuleDict(cams)
        self.volume = volume


from hydra_zen import to_yaml

PosesConf = builds(poses_to_tensor)
CameraConf = builds(
    Camera,
    T=PosesConf([Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]),
)
RadianceFieldConf = builds(RadianceField)
SpatialFilterConf = builds(SpatialFilter)
BoxTransformConf = builds(BoxTransform)
VolConf = builds(
    Volume,
    rf=RadianceFieldConf(),
    filter=SpatialFilterConf(),
    boxt=BoxTransformConf((-1.0,) * 3, (1.0,) * 3),
)

SceneConfig = builds(
    Scene,
    cams={},
    volume=VolConf(),
)
