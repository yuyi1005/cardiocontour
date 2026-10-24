from pathlib import Path
import nibabel as nib

from datasets.base_dataset import BaseShortAxisDataset


class MnMs2Dataset(BaseShortAxisDataset):
    def __init__(
        self,
        split: str = "train",
        apply_augmentation: bool = True,
        load_unlabeled: bool = True,
        fourier_m: int = 4,
    ):
        super().__init__(apply_augmentation=apply_augmentation, fourier_m=fourier_m)
        self.root = Path("data/MnMs2/dataset")
        self.split = split
        self.load_unlabeled = load_unlabeled
        self.samples = self._build_samples()

    def _build_samples(self):
        case_ids = sorted(
            [p.name for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()]
        )
        if len(case_ids) == 0:
            raise RuntimeError(f"No case folders found under {self.root}")

        # Official split: cases 001-200 (train + val) for training, the last 160 cases for testing.
        num_test_cases = 160
        if len(case_ids) <= num_test_cases:
            raise RuntimeError(f"Expected more than {num_test_cases} cases under {self.root}, found {len(case_ids)}")
        split_index = len(case_ids) - num_test_cases
        if self.split == "train":
            split_ids = case_ids[:split_index]
        elif self.split in {"test", "val"}:
            split_ids = case_ids[split_index:]
        elif self.split == "all":
            split_ids = case_ids
        else:
            raise ValueError(f"Unsupported MnMs2 split: {self.split}")

        samples = []
        for case_id in split_ids:
            case_dir = self.root / case_id
            for phase in ("ED", "ES"):
                image_path = case_dir / f"{case_id}_SA_{phase}.nii.gz"
                gt_path = case_dir / f"{case_id}_SA_{phase}_gt.nii.gz"
                if image_path.exists() and gt_path.exists():
                    samples.append(
                        {
                            "case_id": f"mnms2_{image_path.name.replace('.nii.gz', '')}",
                            "image_path": image_path,
                            "gt_path": gt_path,
                        }
                    )
            if self.split == "train" and self.load_unlabeled:
                cine_path = case_dir / f"{case_id}_SA_CINE.nii.gz"
                if cine_path.exists():
                    cine_nii = nib.load(str(cine_path))
                    num_frames = cine_nii.shape[-1] if len(cine_nii.shape) == 4 else 1
                    for frame_idx in range(num_frames):
                        samples.append(
                            {
                                "case_id": f"mnms2_{cine_path.name.replace('.nii.gz', '')}_f{frame_idx:02d}",
                                "image_path": cine_path,
                                "gt_path": None,
                                "frame_idx": frame_idx if num_frames > 1 else None,
                            }
                        )
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
