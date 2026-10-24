from pathlib import Path

import nibabel as nib
import numpy as np

from datasets.base_dataset import BaseShortAxisDataset


class ACDCDataset(BaseShortAxisDataset):
    def __init__(
        self,
        split: str = "train",
        apply_augmentation: bool = True,
        load_unlabeled: bool = True,
        fourier_m: int = 4,
    ):
        super().__init__(apply_augmentation=apply_augmentation, fourier_m=fourier_m)
        if split == "train":
            self.root = Path("data/ACDC/database/training")
        elif split in {"val", "test"}:
            self.root = Path("data/ACDC/database/testing")
        elif split == "all":
            self.root = None
        else:
            raise ValueError(f"Unsupported ACDC split: {split}")
        self.split = split
        self.load_unlabeled = load_unlabeled
        self.samples = self._build_samples()

    def get_curve_label_groups(self):
        return (
            (3,),
            (2, 3),
            (1, 2, 3),
        )

    def _build_samples(self):
        samples = []
        roots = [self.root] if self.root is not None else [
            Path("data/ACDC/database/training"),
            Path("data/ACDC/database/testing"),
        ]
        for root in roots:
            if not root.exists():
                raise RuntimeError(f"ACDC split folder not found: {root}")

            for patient_dir in sorted(root.glob("patient*")):
                if not patient_dir.is_dir():
                    continue

                for gt_path in sorted(patient_dir.glob("*_frame??_gt.nii.gz")):
                    image_name = gt_path.name.replace("_gt.nii.gz", ".nii.gz")
                    image_path = gt_path.with_name(image_name)
                    if not image_path.exists():
                        continue

                    samples.append(
                        {
                            "case_id": f"acdc_{image_path.name.replace('.nii.gz', '')}",
                            "image_path": image_path,
                            "gt_path": gt_path,
                        }
                    )

                if root.name == "training" and self.load_unlabeled:
                    cine_path = patient_dir / f"{patient_dir.name}_4d.nii.gz"
                    if cine_path.exists():
                        cine_nii = nib.load(str(cine_path))
                        num_frames = cine_nii.shape[-1] if len(cine_nii.shape) == 4 else 1
                        for frame_idx in range(num_frames):
                            samples.append(
                                {
                                    "case_id": f"acdc_{cine_path.name.replace('.nii.gz', '')}_f{frame_idx:02d}",
                                    "image_path": cine_path,
                                    "gt_path": None,
                                    "frame_idx": frame_idx if num_frames > 1 else None,
                                }
                            )
        return samples

    def adjust_image_affine(self, img_nii, affine: np.ndarray) -> np.ndarray:
        zooms = img_nii.header.get_zooms()[:3]
        corrected_affine = affine.astype(np.float32, copy=True)
        for axis_idx, zoom in enumerate(zooms):
            column = corrected_affine[:3, axis_idx]
            norm = float(np.linalg.norm(column))
            if norm > 1e-6:
                corrected_affine[:3, axis_idx] = column / norm * float(zoom)
            else:
                corrected_affine[:3, axis_idx] = 0.0
                corrected_affine[axis_idx, axis_idx] = float(zoom)
        return corrected_affine

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self._load_sample(
            item["image_path"],
            item["gt_path"],
            item["case_id"],
            frame_idx=item.get("frame_idx"),
        )
