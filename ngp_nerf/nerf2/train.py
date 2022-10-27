from itertools import islice
from .cameras import MultiViewScene

from . import sampling

import torch.utils.data


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        mvs: MultiViewScene,
        n_samples_per_cam: int = None,
        random: bool = True,
        subpixel: bool = True,
    ):
        self.mvs = mvs
        self.n_pixels_per_cam = self.mvs.cameras.size[0].prod().item()
        if n_samples_per_cam is None:
            # width of first cam (one row)
            n_samples_per_cam = mvs.cameras.size[0, 0].item()
        assert self.n_pixels_per_cam % n_samples_per_cam == 0
        self.n_samples_per_cam = n_samples_per_cam
        self.random = random
        self.subpixel = subpixel if random else False

    def __iter__(self):
        if self.random:
            return islice(
                sampling.generate_random_uv_samples(
                    camera=self.mvs.cameras,
                    image=self.mvs.images,
                    n_samples_per_cam=self.n_samples_per_cam,
                    subpixel=self.subpixel,
                ),
                len(self),
            )
        else:
            return sampling.generate_sequential_uv_samples(
                camera=self.mvs.cameras,
                image=self.mvs.images,
                n_samples_per_cam=self.n_samples_per_cam,
                n_passes=1,
            )

    def __len__(self) -> int:
        # Number of mini-batches required to match with number of total pixels
        return self.n_pixels_per_cam // self.n_samples_per_cam


if __name__ == "__main__":
    from .io import load_scene_from_json
    from . import plotting

    mvs = load_scene_from_json("data/suzanne/transforms.json", load_images=True)
    ds = MultiViewDataset(mvs)

    # ax = plotting.plot_camera(mvs.cameras)
    # ax = plotting.plot_box(mvs.aabb)
    # ax.set_aspect("equal")
    # ax.relim()  # make sure all the data fits
    # ax.autoscale()
    import matplotlib.pyplot as plt

    # plt.show()
    img = torch.empty((2, 4, 30, 40)).uniform_(0.0, 1.0)
    plotting.plot_image(img, scale=0.5)
    plt.show()

    # dl = torch.utils.data.DataLoader(ds, batch_size=4)
    # print(len(dl))
    # for idx, (uv, uv_f) in enumerate(dl):
    #     print(idx, uv.shape)
    # print(idx)
