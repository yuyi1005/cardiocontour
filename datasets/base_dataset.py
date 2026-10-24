from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.augmentation import (
    Compose3D,
    normalize_intensity,
    RandomBiasField,
    RandomCenterCrop,
    RandomGammaIntensity,
    RandomGaussianNoise,
    RandomInPlaneAffine,
    ResizeKeepAspect,
    PadToShape,
)
from utils.contour_utils import contour_to_fourier, mask_to_contour, fourier_to_contour


def make_slice_affine(base_affine: np.ndarray, slice_index: int) -> np.ndarray:
    """
    Create a per-slice affine for slice k from a 3D NIfTI affine.

    For a volume with voxel coordinates [i, j, k, 1]^T, the 3D affine maps
    to world coordinates. For a 2D slice at fixed k, we fold that k into the
    translation term so the returned affine maps [i, j, 1]^T-like in-plane
    coordinates to the correct world plane.

    Output is still 4x4 for convenience / consistency.
    """
    A = base_affine.copy()

    # world origin for this slice = A[:3, 3] + k * A[:3, 2]
    A2 = np.eye(4, dtype=np.float32)
    A2[:3, 0] = A[:3, 0]
    A2[:3, 1] = A[:3, 1]
    A2[:3, 2] = A[:3, 2]
    A2[:3, 3] = A[:3, 3] + slice_index * A[:3, 2]
    return A2


def short_axis_collate_fn(batch):
    return {
        "images": [sample["images"] for sample in batch],
        "affines": [sample["affines"] for sample in batch],
        "fourier_coeffs": [sample["fourier_coeffs"] for sample in batch],
        "contours": [sample["contours"] for sample in batch],
        "labels": [sample["labels"] for sample in batch],
        "volumes": [sample["volumes"] for sample in batch],
        "mask": [sample["mask"] for sample in batch],
        "case_id": [sample["case_id"] for sample in batch],
    }


class BaseShortAxisDataset(Dataset):
    def __init__(self, apply_augmentation: bool = True, fourier_m: int = 4, sampling_k: int = 64):
        self.fourier_m = int(fourier_m)
        self.sampling_k = int(sampling_k)
        self.resize = ResizeKeepAspect(
            max_h=256,
            max_w=256,
            image_order=1,
            label_order=0,
            image_cval=0.0,
            label_cval=0.0,
            update_affine=True,
        )
        self.pre_crop = (
            RandomCenterCrop(
                p=0.5,
                min_fraction=0.8,
                update_affine=True,
            )
            if apply_augmentation
            else None
        )
        self.transform = (
            RandomInPlaneAffine(
                p=0.8,
                rotate_deg=180.0,
                translate_px=(48.0, 48.0),
                scale_range=(0.8, 1.2),
                flip_lr_prob=0.5,
                flip_ud_prob=0.5,
                image_order=1,
                label_order=0,
                image_cval=0.0,
                label_cval=0.0,
                update_affine=True,
            )
            if apply_augmentation
            else None
        )
        self.intensity_transform = (
            Compose3D(
                [
                    RandomBiasField(p=0.25, coeff_range=(0.2, 0.45), grid_shape=(3, 3)),
                    RandomGammaIntensity(p=0.25, gamma_range=(0.7, 1.4)),
                    RandomGaussianNoise(p=0.25, sigma_range=(0.01, 0.04)),
                ]
            )
            if apply_augmentation
            else None
        )
        self.pad = PadToShape(
            max_h=256,
            max_w=256,
            image_cval=0.0,
            label_cval=0.0,
        )

    def get_curve_label_groups(self):
        return (
            (1,),
            (1, 2),
            (1, 2, 3),
        )

    def adjust_image_affine(self, img_nii, affine: np.ndarray) -> np.ndarray:
        return affine

    def _load_sample(self, image_path: Path, gt_path: Path | None, case_id: str, frame_idx: int | None = None):
        img_nii = nib.load(str(image_path))

        image = img_nii.get_fdata().astype(np.float32)
        affine = self.adjust_image_affine(img_nii, img_nii.affine.astype(np.float32))
        has_gt = gt_path is not None

        if frame_idx is not None and image.ndim == 4:
            image = image[..., frame_idx]

        if has_gt:
            gt_nii = nib.load(str(gt_path))
            gt = gt_nii.get_fdata().astype(np.float32)
            if frame_idx is not None and gt.ndim == 4:
                gt = gt[..., frame_idx]
        else:
            gt = np.zeros(image.shape[:3] if image.ndim == 4 else image.shape, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., None]
        if gt.ndim == 2:
            gt = gt[..., None]

        if self.transform is not None:
            image, gt, affine = self.transform(image, gt, affine)

        image, gt, affine = self.resize(image, gt, affine)

        if self.pre_crop is not None:
            image, gt, affine = self.pre_crop(image, gt, affine)

        if self.intensity_transform is not None:
            image, gt, affine = self.intensity_transform(image, gt, affine)

        image, gt, affine = self.pad(image, gt, affine)
        image = normalize_intensity(image)

        num_slices = image.shape[2]
        images_i = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        affines_i = torch.stack(
            [torch.from_numpy(make_slice_affine(affine, s)).float() for s in range(num_slices)],
            dim=0,
        )

        fourier_tensor = None
        contour_tensor = None
        label_tensor = None
        volume_tensor = None
        mask_tensor = None
        if has_gt:
            fourier = []
            contours = []
            label = []
            masks = []
            volume = np.zeros(len(self.get_curve_label_groups()), dtype=np.float32)
            slice_area = float(image.shape[0] * image.shape[1])
            out_dim = 2 * self.fourier_m + 3
            cumulative_labels = self.get_curve_label_groups()
            for slice_idx in range(num_slices):
                slice_gt = np.rint(gt[:, :, slice_idx]).astype(np.int32, copy=False)
                slice_fourier = []
                slice_contours = []
                slice_label = []
                slice_masks = []
                for group_idx, class_group in enumerate(cumulative_labels):
                    slice_mask = np.isin(slice_gt, class_group).astype(np.float32)
                    volume[group_idx] += float(slice_mask.sum()) / slice_area
                    slice_masks.append(slice_mask.astype(bool, copy=False))
                    contour_xy, cls = mask_to_contour(slice_mask)
                    coeffs, cls = (
                        contour_to_fourier(contour_xy, fourier_m=self.fourier_m)
                        if cls > 0.5
                        else (np.zeros(out_dim, dtype=np.float32), 0.0)
                    )
                    if cls > 0.5:
                        x, y = fourier_to_contour(coeffs, sampling_k=self.sampling_k, fourier_m=self.fourier_m)
                        contour = np.stack([x, y], axis=-1).astype(np.float32, copy=False)
                    else:
                        contour = np.zeros((self.sampling_k, 2), dtype=np.float32)
                    slice_fourier.append(coeffs)
                    slice_contours.append(contour)
                    slice_label.append(cls)
                fourier.append(np.stack(slice_fourier, axis=0))
                contours.append(np.stack(slice_contours, axis=0))
                label.append(np.asarray(slice_label, dtype=np.float32))
                masks.append(np.stack(slice_masks, axis=0))
            regional_volume = np.asarray([volume[0], volume[1] - volume[0], volume[2] - volume[1]], dtype=np.float32)
            volume_tensor = torch.from_numpy(regional_volume).float()
            fourier_tensor = torch.from_numpy(np.stack(fourier, axis=0)).float()
            contour_tensor = torch.from_numpy(np.stack(contours, axis=0)).float()
            label_tensor = torch.from_numpy(np.asarray(label, dtype=np.float32)).float()
            mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).bool()
        return {
            "images": images_i,
            "affines": affines_i,
            "fourier_coeffs": fourier_tensor,
            "contours": contour_tensor,
            "labels": label_tensor,
            "volumes": volume_tensor,
            "mask": mask_tensor,
            "case_id": case_id,
        }
