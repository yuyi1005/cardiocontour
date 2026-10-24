import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from utils.contour_utils import contour_to_mask
from datasets.base_dataset import BaseShortAxisDataset
from infer_vis3d import (
    build_model,
    load_checkpoint,
    smooth_predictions_along_slices,
)
from train import seed_everything


class CineNiftiDataset(BaseShortAxisDataset):
    def __init__(self, cine_path: str | Path, fourier_m: int, frame_idx: int | None = None):
        super().__init__(apply_augmentation=False, fourier_m=fourier_m)
        self.cine_path = Path(cine_path)
        if not self.cine_path.exists():
            raise FileNotFoundError(f"NIfTI not found: {self.cine_path}")

        nii = nib.load(str(self.cine_path))
        if len(nii.shape) == 4:
            frame_indices = list(range(nii.shape[-1])) if frame_idx is None else [int(frame_idx)]
        elif len(nii.shape) == 3:
            if frame_idx is not None:
                raise ValueError(f"--frame-idx was set, but input is 3D: {self.cine_path}")
            frame_indices = [None]
        else:
            raise ValueError(f"Expected 3D or 4D NIfTI, got shape {nii.shape} from {self.cine_path}")

        stem = self.cine_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        self.samples = [
            {
                "case_id": stem if idx is None else f"{stem}_f{idx:02d}",
                "image_path": self.cine_path,
                "gt_path": None,
                "frame_idx": idx,
            }
            for idx in frame_indices
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._load_sample(
            sample["image_path"],
            sample["gt_path"],
            sample["case_id"],
            frame_idx=sample["frame_idx"],
        )


def optional_float(value):
    if value is None:
        return None
    if str(value).lower() in {"none", "off", "false"}:
        return None
    return float(value)


def build_label_volume(pred_contours: torch.Tensor, pred_labels: torch.Tensor, height: int, width: int):
    contours_np = pred_contours.detach().cpu().numpy()
    labels_np = pred_labels.detach().cpu().numpy()
    num_slices = contours_np.shape[0]
    label_volume = np.zeros((height, width, num_slices), dtype=np.uint8)

    for slice_idx in range(num_slices):
        # Curves are cumulative: 2 contains RV+MYO+LV, 1 contains MYO+LV, 0 contains LV.
        # Paint broad-to-narrow so LV and MYO overwrite the larger regions.
        for curve_idx, label_value in ((2, 3), (1, 2), (0, 1)):
            if labels_np[slice_idx, curve_idx] <= 0.5:
                continue
            mask = contour_to_mask(contours_np[slice_idx, curve_idx], height, width)
            label_volume[:, :, slice_idx][mask] = label_value

    return label_volume


def keep_largest_component(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num_components = ndimage.label(mask, structure=structure)
    if num_components <= 1:
        return mask
    component_sizes = ndimage.sum(mask, labeled, index=np.arange(1, num_components + 1))
    largest_label = int(np.argmax(component_sizes) + 1)
    return labeled == largest_label


def keep_largest_component_per_label(label_volume: np.ndarray):
    if label_volume.ndim == 4:
        frames = [keep_largest_component_per_label(label_volume[..., frame_idx]) for frame_idx in range(label_volume.shape[3])]
        return np.stack(frames, axis=-1).astype(np.uint8, copy=False)

    output = np.zeros_like(label_volume, dtype=np.uint8)
    for label_value in (3, 2, 1):
        kept = keep_largest_component(label_volume == label_value)
        output[kept] = label_value
    return output


def save_label_volume(label_volume: np.ndarray, affine: np.ndarray, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(label_volume.astype(np.uint8, copy=False), affine)
    nii.header.set_xyzt_units("mm")
    nii.header.set_data_dtype(np.uint8)
    if label_volume.ndim == 4:
        spatial_zooms = tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))
        nii.header.set_zooms(spatial_zooms + (1.0,))
    nib.save(nii, str(output_path))


def gaussian_kernel_1d(sigma: float):
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"--frame-smooth must be positive or None, got {sigma}")
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum().clamp_min(1e-8)


def smooth_predictions_across_frames(contours_by_frame, logits_by_frame, sigma: float):
    if len(contours_by_frame) < 2:
        return contours_by_frame, logits_by_frame

    kernel = gaussian_kernel_1d(sigma).to(dtype=contours_by_frame[0].dtype, device=contours_by_frame[0].device)
    radius = int(kernel.numel() // 2)
    smoothed_contours = []
    smoothed_logits = []

    for frame_idx in range(len(contours_by_frame)):
        contour_accum = torch.zeros_like(contours_by_frame[frame_idx])
        logit_accum = torch.zeros_like(logits_by_frame[frame_idx])
        weight_accum = contours_by_frame[frame_idx].new_zeros(())

        for kernel_idx, weight in enumerate(kernel):
            neighbor_idx = min(max(frame_idx + kernel_idx - radius, 0), len(contours_by_frame) - 1)
            contour_accum = contour_accum + weight * contours_by_frame[neighbor_idx]
            logit_accum = logit_accum + weight * logits_by_frame[neighbor_idx]
            weight_accum = weight_accum + weight

        smoothed_contours.append(contour_accum / weight_accum)
        smoothed_logits.append(logit_accum / weight_accum)

    return smoothed_contours, smoothed_logits


def run_inference(args):
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CineNiftiDataset(args.nifti_path, fourier_m=args.fourier_m, frame_idx=args.frame_idx)
    model = build_model(args.fourier_m, device, sampling_k=args.sampling_k)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    samples = []
    contours_by_frame = []
    logits_by_frame = []
    print(f"Running segmentation on {len(dataset)} frame(s)")
    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            images = sample["images"].to(device)
            output = model([images])
            samples.append(sample)
            contours_by_frame.append(output["contours"][0].cpu())
            logits_by_frame.append(output["cls_logits"][0].cpu())
            print(f"Inferred frame {sample_idx + 1}/{len(dataset)}", flush=True)

    if args.frame_smooth is not None:
        contours_by_frame, logits_by_frame = smooth_predictions_across_frames(
            contours_by_frame,
            logits_by_frame,
            sigma=args.frame_smooth,
        )

    label_volumes = []
    for sample_idx, (sample, pred_contours, cls_logits) in enumerate(zip(samples, contours_by_frame, logits_by_frame)):
        if args.smooth is not None:
            pred_contours, cls_logits = smooth_predictions_along_slices(pred_contours, cls_logits, sigma=args.smooth)
        pred_labels = (torch.sigmoid(cls_logits) > args.cls_threshold).float()
        images = sample["images"]
        label_volumes.append(
            build_label_volume(
                pred_contours=pred_contours,
                pred_labels=pred_labels,
                height=images.shape[1],
                width=images.shape[2],
            )
        )
        print(f"Rasterized frame {sample_idx + 1}/{len(samples)}", flush=True)

    label_volume = label_volumes[0] if len(label_volumes) == 1 else np.stack(label_volumes, axis=-1).astype(np.uint8, copy=False)
    label_volume = keep_largest_component_per_label(label_volume)

    save_label_volume(label_volume, samples[0]["affines"][0].detach().cpu().numpy(), args.output)
    counts = {label: int((label_volume == label).sum()) for label in (1, 2, 3)}
    print(
        f"Saved predicted mask: {args.output} | "
        f"shape={label_volume.shape} | LV={counts[1]} MYO={counts[2]} RV={counts[3]}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Save a predicted LV/MYO/RV segmentation mask from one NIfTI file.")
    parser.add_argument("--nifti_path", default="data/ACDC/database/testing/patient101/patient101_4d.nii.gz", type=str)
    parser.add_argument("--output", default="work_dirs/exp_m6/infer_segment/patient101_4d_pred_mask.nii.gz", type=str)
    parser.add_argument("--frame-idx", type=int, default=None, help="Optional frame to select when the NIfTI is 4D; omit to process all frames.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cardiocontour_v1_pretrained_m6_last.pt")
    parser.add_argument("--fourier_m", type=int, default=6)
    parser.add_argument("--sampling-k", type=int, default=64)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--smooth", type=optional_float, default=2, help="Gaussian sigma in slices for smoothing contours and logits; use None/off to disable.")
    parser.add_argument("--frame-smooth", type=optional_float, default=2, help="Gaussian sigma in frames for smoothing contours and logits; use None/off to disable.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
