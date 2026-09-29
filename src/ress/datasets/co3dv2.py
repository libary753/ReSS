"""CO3Dv2 loader -- the calibration set behind the HeSS Fisher head-importance tables.

This is the only dataset loader here: CO3Dv2 is the calibration set, used by
`scripts/calibrate_fisher_scores.py` alone, which is what produces the tables shipped in
`ress/calibration/`. Nothing in the forward path reads a dataset.

Reading it needs pytorch3d's implicitron dataset provider. Only the CPU dataset-I/O path
is used, so pytorch3d may be built without CUDA (`FORCE_CUDA=0`) -- see the setup docs,
since a full CUDA build costs close to an hour and none of its kernels run here.

VIEW SAMPLING IS NOT SEEDED HERE, ON PURPOSE. `get_scan` draws its views with a bare
`torch.randperm`, so the CALLER owns reproducibility by seeding immediately before each
call (`scripts/calibrate_fisher_scores.py` does: `torch.manual_seed(view_seed + idx)`).
The seed cannot live inside `get_scan` because the draw is retried -- the method returns
None when any drawn view falls below `min_quality` -- and a seed fixed per scene would
make a rejected scene reject forever. This matters more than it looks: the tables the
paper's HeSS row stands on predate that caller-side seeding and are therefore not
reproducible draw-to-draw. See the README's "Calibration" section for how much the
draw actually moves the resulting table.
"""

# ruff: noqa: I001 -- the import order below is load-bearing, not stylistic:
# `ress._pathsetup` is what puts the VGGT tree on sys.path, so isort's
# third-party-before-first-party order would import `vggt` before it exists.
import json
import os

import torch
from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2,
)
from pytorch3d.implicitron.tools.config import expand_args_fields

from ress import _pathsetup  # noqa: F401  (puts the VGGT tree on sys.path)
from ress.datasets._preprocess import load_and_preprocess_depths, load_ply, update_K
from vggt.utils.load_fn import load_and_preprocess_images


def get_dataset_map(dataset_root: str, category: str, subset_name: str) -> DatasetMap:
    """Obtain the dataset map that contains the train/val/test dataset objects."""
    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map_provider = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        dataset_root=dataset_root,
        test_on_train=True,
        only_test_set=False,
        load_eval_batches=True,
        dataset_JsonIndexDataset_args=DictConfig({"remove_empty_masks": False}),
    )
    return dataset_map_provider.get_dataset_map()


def make_K_from_ndc_isotropic(f_ndc, W, H, principal_at_center=True):
    """CO3Dv2 stores focal length in PyTorch3D's NDC convention (normalised by the
    SHORTER image side); this turns it back into pixel units."""
    s = min(W, H) / 2.0
    f_pix = f_ndc * s
    cx = W / 2.0 if principal_at_center else 0.0
    cy = H / 2.0 if principal_at_center else 0.0
    K = torch.tensor([[f_pix, 0.0, cx], [0.0, f_pix, cy], [0.0, 0.0, 1.0]])

    return K


class Co3Dv2:
    """Category-and-subset indexed view of a CO3Dv2 download.

    A "scan" here is one (category, subset_name) pair, and `get_scan` draws `n_views`
    frames from it. Only the sequences whose subset name contains `split` are indexed.
    """

    align_voxel_size: float = 0.01
    eval_dist_thr: float = 1

    def __init__(self, dataset_dir, split: str, min_quality: float = 0.5):
        if split not in ("dev", "test", "train"):
            raise ValueError(f"Undefined split: {split}")
        self.split = split
        self.dataset_dir = dataset_dir
        self.min_quality = min_quality
        self.build_scan_list()

    def build_scan_list(self):
        with open(
            os.path.join(self.dataset_dir, "category_to_subset_name_list.json"),
            "r",
            encoding="utf-8",
        ) as f:
            subset_name_dict = json.load(f)

        self.scans = []
        for category, subset_names in subset_name_dict.items():
            for subset_name in subset_names:
                if subset_name.find(self.split) >= 0:
                    self.scans.append((category, subset_name))

    def get_scan(self, scan_idx, n_views):
        """Draw `n_views` random frames from scan `scan_idx`.

        Returns None -- meaning "skip this scan" -- if ANY drawn view's
        `camera_quality_score` is below `min_quality`, since a scene is only useful for
        calibration if every view in it has a trustworthy pose. See the module docstring
        for why the draw is seeded by the caller and not here.
        """
        category, subset_name = self.scans[scan_idx]
        dataset_map = get_dataset_map(
            dataset_root=self.dataset_dir,
            category=category,
            subset_name=subset_name,
        )
        dataset = dataset_map.val
        if self.split == "dev":
            dataset = dataset_map.val
        elif self.split == "test":
            dataset = dataset_map.test
        elif self.split == "train":
            dataset = dataset_map.train
        else:
            raise ValueError(f"Undefined split: {self.split}")

        # View sampling -- unseeded by design; the caller seeds. See module docstring.
        img_ids = torch.randperm(len(dataset))[:n_views]

        img_paths = []
        depth_paths = []
        Rs = []
        ts = []
        Ks = []
        for img_id in img_ids:
            img_id = img_id.item()
            frame = dataset[img_id]
            if frame.camera_quality_score.item() < self.min_quality:
                return None

            img_paths.append(frame.image_path)
            depth_paths.append(frame.depth_path)
            Rs.append(frame.camera.R[0].T)
            ts.append(frame.camera.T[0])
            H = frame.image_size_hw[0].item()
            W = frame.image_size_hw[1].item()
            f = dataset.frame_annots[img_id]["frame_annotation"].viewpoint.focal_length[0]
            K = make_K_from_ndc_isotropic(f, W=W, H=H)
            K = torch.tensor(update_K(K.numpy(), height=H, width=W))
            Ks.append(K)

        images = load_and_preprocess_images(img_paths)
        depths, _ = load_and_preprocess_depths(depth_paths, convert_to_float16=True)
        depths = depths.squeeze(1)  # (N, 1, H, W) -> (N, H, W)

        depth_masks = load_and_preprocess_images(
            [
                img_path.replace("images", "depth_masks").replace(".jpg", ".png")
                for img_path in img_paths
            ]
        ) == 1.0
        depth_masks = depth_masks[:, 0, :, :]

        Rs = torch.stack(Rs)
        ts = torch.stack(ts)

        # PyTorch3D's camera convention has +x left and +y up; flipping both axes puts
        # the extrinsics in the +x right / +y down convention the rest of this project
        # (and VGGT) uses. The third axis is shared, hence the (-1, -1, +1) diagonal.
        Rts = torch.cat([Rs, ts.unsqueeze(-1)], dim=-1)
        Rts = torch.tensor((((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),)) @ Rts

        Ks = torch.stack(Ks)

        pcd = (
            load_ply(
                os.path.join(
                    self.dataset_dir, dataset.seq_annots[frame.sequence_name].point_cloud.path
                )
            )
            if self.split != "test"
            else None
        )

        scene_data = {
            "images": images.cuda(),  # torch.Tensor (N, 3, H, W)
            "depth": depths,  # torch.Tensor (N, H, W)
            "K": Ks,  # torch.Tensor (N, 3, 3)
            "Rt": Rts,  # torch.Tensor (N, 3, 4)
            "depth_mask": depth_masks,  # torch.Tensor (N, H, W)
            "Rt_pcd": torch.eye(4)[:3],  # torch.Tensor (N, 3, 4)
            "pcd": pcd,  # np.array (M, 3)
            "obs_mask": None,
            "BB": None,
            "res": None,
            "quality_score": frame.camera_quality_score.item(),
        }

        return scene_data

    def size(self):
        return len(self.scans)
