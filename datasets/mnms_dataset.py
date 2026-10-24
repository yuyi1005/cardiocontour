from pathlib import Path
import nibabel as nib

from datasets.base_dataset import BaseShortAxisDataset


class MnMsDataset(BaseShortAxisDataset):
    def __init__(
        self,
        split: str = "train",
        apply_augmentation: bool = True,
        load_unlabeled: bool = True,
        fourier_m: int = 4,
    ):
        super().__init__(apply_augmentation=apply_augmentation, fourier_m=fourier_m)
        self.root = Path("data/MnMs")
        self.split = split
        self.load_unlabeled = load_unlabeled
        self.samples = self._build_samples()

    def _iter_frames(self, image_path: Path, gt_path: Path | None, prefix: str, keep_only_labeled: bool):
        image_nii = nib.load(str(image_path))
        num_frames = image_nii.shape[-1] if len(image_nii.shape) == 4 else 1
        gt_data = None
        if gt_path is not None:
            gt_data = nib.load(str(gt_path)).get_fdata()
        samples = []
        for frame_idx in range(num_frames):
            frame_gt_path = gt_path
            if gt_data is not None:
                gt_frame = gt_data[..., frame_idx] if gt_data.ndim == 4 else gt_data
                has_label = bool((gt_frame != 0).any())
                if not has_label:
                    if keep_only_labeled or not self.load_unlabeled:
                        continue
                    frame_gt_path = None
            samples.append(
                {
                    "case_id": f"mnms_{prefix}_f{frame_idx:02d}",
                    "image_path": image_path,
                    "gt_path": frame_gt_path,
                    "frame_idx": frame_idx if num_frames > 1 else None,
                }
            )
        return samples

    def _build_samples(self):
        split_to_groups = {
            "train": [
                ("Training/Labeled", True, False),
                ("Training/Unlabeled", False, False),
                ("Validation", True, False),
            ],
            "test": [
                ("Testing", True, True),
            ],
            "all": [
                ("Training/Labeled", True, False),
                ("Training/Unlabeled", False, False),
                ("Validation", True, False),
                ("Testing", True, True),
            ],
        }

        if self.split not in split_to_groups:
            raise ValueError(f"Unsupported MnMs split: {self.split}")

        samples = []
        for rel_dir, use_gt, keep_only_labeled in split_to_groups[self.split]:
            if rel_dir == "Training/Unlabeled" and not self.load_unlabeled:
                continue
            split_dir = self.root / rel_dir
            if not split_dir.exists():
                continue

            for case_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                image_path = case_dir / f"{case_dir.name}_sa.nii.gz"
                gt_path = case_dir / f"{case_dir.name}_sa_gt.nii.gz"
                if not image_path.exists():
                    continue
                sample_gt_path = gt_path if use_gt and gt_path.exists() else None
                samples.extend(
                    self._iter_frames(
                        image_path=image_path,
                        gt_path=sample_gt_path,
                        prefix=image_path.name.replace(".nii.gz", ""),
                        keep_only_labeled=keep_only_labeled,
                    )
                )

        if len(samples) == 0:
            raise RuntimeError(f"No MnMs samples found for split={self.split} under {self.root}")
        return samples

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
