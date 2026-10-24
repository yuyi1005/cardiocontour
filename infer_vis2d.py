import argparse
from pathlib import Path

import torch

from datasets.base_dataset import BaseShortAxisDataset
from models.cardiocontour import CardioContour
from test import visualize_fourier_overlay_grid
from train import seed_everything


class NiftiSampleDataset(BaseShortAxisDataset):
    def __init__(
        self,
        nifti_path: str | Path,
        fourier_m: int,
        gt_path: str | Path | None = None,
        frame_idx: int | None = None,
    ):
        super().__init__(apply_augmentation=False, fourier_m=fourier_m)
        self.nifti_path = Path(nifti_path)
        if not self.nifti_path.exists():
            raise FileNotFoundError(f"NIfTI not found: {self.nifti_path}")
        self.gt_path = None if gt_path is None else Path(gt_path)
        if self.gt_path is not None and not self.gt_path.exists():
            raise FileNotFoundError(f"GT NIfTI not found: {self.gt_path}")
        self.frame_idx = frame_idx
        self.case_id = self.nifti_path.name.removesuffix(".nii.gz").removesuffix(".nii")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self._load_sample(
            self.nifti_path,
            self.gt_path,
            self.case_id,
            frame_idx=self.frame_idx,
        )


def build_model(fourier_m: int, sampling_k: int, device: torch.device):
    return CardioContour(fourier_m=fourier_m, sampling_k=sampling_k).to(device)


def load_checkpoint(model, checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Checkpoint is missing required model weights: {incompatible.missing_keys}")
    print(f"Loaded checkpoint from {checkpoint_path}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys: {incompatible.unexpected_keys}")


def run_inference(args):
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NiftiSampleDataset(
        args.nifti_path,
        fourier_m=args.fourier_m,
        gt_path=args.gt_path,
        frame_idx=args.frame_idx,
    )
    model = build_model(args.fourier_m, args.sampling_k, device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    sample = dataset[0]
    images = sample["images"].to(device)
    with torch.no_grad():
        output = model([images])

    pred_contours = output["contours"][0]
    pred_labels = (torch.sigmoid(output["cls_logits"][0]) > args.cls_threshold).float()
    gt_contours = sample["contours"]
    gt_labels = sample["labels"]

    print(
        f"{sample['case_id']} | slices={images.shape[0]} | "
        f"positive_curves={int(pred_labels.sum().item())} | gt={'yes' if gt_contours is not None else 'no'}"
    )

    overlay_path = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = output_dir / f"{sample['case_id']}.png"

    valid_slices = torch.ones(images.shape[0], dtype=torch.bool, device=device)
    visualize_fourier_overlay_grid(
        images=images,
        gt_contours=None if gt_contours is None else gt_contours.to(device),
        pred_contours=pred_contours,
        gt_labels=None if gt_labels is None else gt_labels.to(device),
        pred_labels=pred_labels,
        valid_slices=valid_slices,
        save_path=overlay_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 2D CardioContour predictions from one NIfTI file.")
    parser.add_argument("--nifti_path", type=str, default="data/ACDC/database/testing/patient101/patient101_frame01.nii.gz")
    parser.add_argument("--gt-path", type=str, default="data/ACDC/database/testing/patient101/patient101_frame01_gt.nii.gz")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame to select when the NIfTI is 4D.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cardiocontour_v1_pretrained_m6_last.pt")
    parser.add_argument("--fourier_m", type=int, default=6)
    parser.add_argument("--sampling-k", type=int, default=64)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="work_dirs/exp_m6/infer_vis2d", help="Save the contour PNG here; if omitted, show it interactively.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
