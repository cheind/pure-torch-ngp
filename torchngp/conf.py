import torch

from hydra_zen import make_custom_builds_fn


builds = make_custom_builds_fn(populate_full_signature=True)


class Camera(torch.nn.Module):
    def __init__(self, focal_length: float, size: tuple[int, int]) -> None:
        super().__init__()
        self.register_buffer("focal_length", torch.tensor([focal_length, focal_length]))
        self.register_buffer("size", torch.as_tensor(size))

    def extra_repr(self) -> str:
        out = ""
        out += f"focal_length={self.focal_length}, "
        out += f"size={self.size}"
        return out


class RadianceField(torch.nn.Module):
    def __init__(self, res: int = 64) -> None:
        super().__init__()
        self.res = res


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


CameraConf = builds(Camera, populate_full_signature=True)
RadianceFieldConf = builds(RadianceField, populate_full_signature=True)
SpatialFilterConf = builds(SpatialFilter, populate_full_signature=True)
BoxTransformConf = builds(BoxTransform, populate_full_signature=True)
VolConf = builds(
    Volume,
    rf=RadianceFieldConf(),
    filter=SpatialFilterConf(),
    boxt=BoxTransformConf((-1.0,) * 3, (1.0,) * 3),
    populate_full_signature=True,
)

SceneConfig = builds(
    Scene,
    cams={"cam_train": CameraConf(1.0, (100, 200))},
    volume=VolConf(),
    populate_full_signature=True,
)
