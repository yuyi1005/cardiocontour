import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from train import seed_everything, build_ground_truth, format_loss_metrics
from utils.losses import CombinedTrainingLoss, EdgeAlignmentLoss, FourierContourLoss, VolumeRegressionLoss
from datasets.mnms_dataset import MnMsDataset
from datasets.mnms2_dataset import MnMs2Dataset
from datasets.acdc_dataset import ACDCDataset
from datasets.base_dataset import short_axis_collate_fn
from models.cardiocontour import CardioContour
from utils.contour_utils import contour_iou, contour_to_mask

CURVE_NAMES = ("LV", "MYO", "RV")


def build_test_dataset(dataset_name, split, apply_augmentation, fourier_m):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnms":
        return MnMsDataset(split=split, apply_augmentation=apply_augmentation, fourier_m=fourier_m)
    if dataset_name == "mnms2":
        return MnMs2Dataset(split=split, apply_augmentation=apply_augmentation, fourier_m=fourier_m)
    if dataset_name == "acdc":
        return ACDCDataset(split=split, apply_augmentation=apply_augmentation, fourier_m=fourier_m)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def visualize_fourier_overlay_grid(
    images,
    gt_contours,
    pred_contours,
    gt_labels,
    pred_labels,
    valid_slices,
    save_path=None,
):
    curve_colors = ("r", "g", "b")
    valid_idx = torch.nonzero(valid_slices, as_tuple=False).squeeze(1).tolist()
    if len(valid_idx) == 0:
        valid_idx = [0]

    n = len(valid_idx)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, idx in zip(axes, valid_idx):
        img = images[idx].detach().cpu().numpy()
        pred_contour_np = pred_contours[idx].detach().cpu().numpy()
        gt_contour_np = None if gt_contours is None else gt_contours[idx].detach().cpu().numpy()
        ax.imshow(img, cmap="gray")

        title_lines = [f"Slice {idx}"]
        for curve_idx in range(len(curve_colors) - 1, -1, -1):
            color = curve_colors[curve_idx]
            curve_name = CURVE_NAMES[curve_idx]
            iou = None
            if pred_labels[idx, curve_idx].item() > 0.5:
                x_pred = pred_contour_np[curve_idx, :, 0] * img.shape[1]
                y_pred = pred_contour_np[curve_idx, :, 1] * img.shape[0]
                ax.plot(x_pred, y_pred, color=color, linestyle="--", linewidth=1.5)
            if (
                gt_labels is not None
                and gt_contour_np is not None
                and gt_labels[idx, curve_idx].item() > 0.5
                and pred_labels[idx, curve_idx].item() > 0.5
            ):
                iou = contour_iou(gt_contour_np[curve_idx], pred_contour_np[curve_idx])

            curve_line = curve_name
            if iou is not None:
                curve_line += f" IoU={iou:.2f}"
            title_lines.append(curve_line)

        ax.set_title("\n".join(title_lines))
        ax.axis("equal")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def visualize_mask_overlay_grid(images, region_masks, save_path=None):
    mask_colors = {
        "LV": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "MYO": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "RV": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    if torch.is_tensor(images):
        images_np = images.detach().cpu().numpy()
    else:
        images_np = np.asarray(images)

    num_slices = images_np.shape[0]
    ncols = min(4, num_slices)
    nrows = (num_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for slice_idx, ax in enumerate(axes):
        if slice_idx >= num_slices:
            ax.axis("off")
            continue

        image = np.asarray(images_np[slice_idx], dtype=np.float32)
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)
        image_rgb = np.repeat(image[..., None], 3, axis=-1)

        overlay = image_rgb.copy()
        for curve_name in CURVE_NAMES:
            mask_slice = np.asarray(region_masks[curve_name][slice_idx], dtype=bool)
            if not np.any(mask_slice):
                continue
            overlay[mask_slice] = 0.55 * overlay[mask_slice] + 0.45 * mask_colors[curve_name]

        ax.imshow(overlay)
        ax.set_title(f"Slice {slice_idx}")
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def calculate_regional_volumes_ml(volume, images, affines):
    volume_np = volume.detach().cpu().numpy().astype(np.float64)
    affine = affines.detach().cpu().numpy()[0]
    height, width = images.shape[-2:]
    pixel_area_mm2 = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    slice_spacing_mm = float(np.linalg.norm(affine[:3, 2]))
    volumes_ml = volume_np * float(height * width) * pixel_area_mm2 * slice_spacing_mm / 1000.0
    return {name: (float(value) if value >= 0.0 else -1.0) for name, value in zip(CURVE_NAMES, volumes_ml)}


def match_pred_and_gt_volumes(case_id, gt_volumes_ml, pred_volumes_ml):
    if gt_volumes_ml is None or pred_volumes_ml is None:
        return {}

    if set(gt_volumes_ml.keys()) == set(CURVE_NAMES):
        return {
            key: (gt_volumes_ml[key], pred_volumes_ml[key])
            for key in CURVE_NAMES
            if key in pred_volumes_ml and gt_volumes_ml[key] >= 0
        }

    phase = None
    if case_id.endswith("_ED"):
        phase = "ED"
    elif case_id.endswith("_ES"):
        phase = "ES"

    if phase is None:
        return {}

    matched = {}
    lv_key = f"LV{phase}V"
    rv_key = f"RV{phase}V"
    if lv_key in gt_volumes_ml:
        matched["LV"] = (gt_volumes_ml[lv_key], pred_volumes_ml["LV"])
    if rv_key in gt_volumes_ml:
        matched["RV"] = (gt_volumes_ml[rv_key], pred_volumes_ml["RV"])
    return matched


def build_patient_region_masks(contours, labels, images):
    contours_np = contours.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    num_slices, height, width = images.shape
    cumulative_masks = np.zeros((num_slices, len(CURVE_NAMES), height, width), dtype=bool)
    for slice_idx in range(num_slices):
        for curve_idx in range(len(CURVE_NAMES)):
            if labels_np[slice_idx, curve_idx] <= 0.5:
                continue
            cumulative_masks[slice_idx, curve_idx] = contour_to_mask(
                contours_np[slice_idx, curve_idx], height=height, width=width
            )
    return {
        "LV": cumulative_masks[:, 0],
        "MYO": np.logical_and(cumulative_masks[:, 1], np.logical_not(cumulative_masks[:, 0])),
        "RV": np.logical_and(cumulative_masks[:, 2], np.logical_not(cumulative_masks[:, 1])),
    }


def keep_largest_connected_component(mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask.copy()
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape (S, H, W), got {mask.shape}")

    structure = ndimage.generate_binary_structure(rank=2, connectivity=1)
    output = np.zeros_like(mask, dtype=bool)
    for slice_idx in range(mask.shape[0]):
        slice_mask = mask[slice_idx]
        if not np.any(slice_mask):
            continue
        labeled, num_components = ndimage.label(slice_mask, structure=structure)
        if num_components <= 1:
            output[slice_idx] = slice_mask
            continue
        component_sizes = ndimage.sum(slice_mask, labeled, index=np.arange(1, num_components + 1))
        largest_label = int(np.argmax(component_sizes)) + 1
        output[slice_idx] = labeled == largest_label
    return output


def postprocess_region_masks(region_masks, mode):
    if mode == 0:
        return {
            curve_name: np.asarray(mask, dtype=bool).copy()
            for curve_name, mask in region_masks.items()
        }
    if mode == 1:
        return {
            curve_name: keep_largest_connected_component(mask)
            for curve_name, mask in region_masks.items()
        }
    raise ValueError(f"Unsupported postprocess mode: {mode}")


def build_patient_region_masks_from_gt(gt_masks):
    gt_masks_np = gt_masks.detach().cpu().numpy().astype(bool)
    return {
        "LV": gt_masks_np[:, 0],
        "MYO": np.logical_and(gt_masks_np[:, 1], np.logical_not(gt_masks_np[:, 0])),
        "RV": np.logical_and(gt_masks_np[:, 2], np.logical_not(gt_masks_np[:, 1])),
    }


def calculate_patient_mask_volumes_ml(region_masks, images, affines):
    affines_np = affines.detach().cpu().numpy()
    num_slices, height, width = images.shape
    volumes_mm3 = {curve_name: 0.0 for curve_name in CURVE_NAMES}

    for slice_idx in range(num_slices):
        affine = affines_np[slice_idx]
        pixel_area_mm2 = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
        slice_spacing_mm = float(np.linalg.norm(affine[:3, 2]))
        voxel_volume_mm3 = pixel_area_mm2 * slice_spacing_mm
        for curve_name in CURVE_NAMES:
            volumes_mm3[curve_name] += float(region_masks[curve_name][slice_idx].sum()) * voxel_volume_mm3

    return {
        curve_name: volumes_mm3[curve_name] / 1000.0
        for curve_name in CURVE_NAMES
    }


def dice_score(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    total = mask_a.sum(dtype=np.float64) + mask_b.sum(dtype=np.float64)
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def calculate_patient_dice_scores(gt_masks, pred_masks):
    gt_masks = build_patient_region_masks_from_gt(gt_masks)

    dice_scores = {}
    for curve_name in CURVE_NAMES:
        dice_scores[curve_name] = dice_score(gt_masks[curve_name], pred_masks[curve_name])
    dice_scores["mean"] = float(np.mean([dice_scores[curve_name] for curve_name in CURVE_NAMES]))
    return dice_scores


def get_volume_spacing_mm(affines):
    affines_np = affines.detach().cpu().numpy()
    affine = affines_np[0]
    spacing_y_mm = float(np.linalg.norm(affine[:3, 1]))
    spacing_x_mm = float(np.linalg.norm(affine[:3, 0]))
    spacing_z_mm = float(np.linalg.norm(affine[:3, 2]))
    return (spacing_z_mm, spacing_y_mm, spacing_x_mm)


def extract_surface(mask):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def compute_hd95_and_nsd(mask_gt, mask_pred, spacing_mm, tolerance_mm):
    mask_gt = np.asarray(mask_gt, dtype=bool)
    mask_pred = np.asarray(mask_pred, dtype=bool)

    if not np.any(mask_gt) and not np.any(mask_pred):
        return {"hd95": 0.0, "nsd": 1.0}
    if not np.any(mask_gt) or not np.any(mask_pred):
        return {"hd95": float("inf"), "nsd": 0.0}

    surface_gt = extract_surface(mask_gt)
    surface_pred = extract_surface(mask_pred)

    if not np.any(surface_gt) and not np.any(surface_pred):
        return {"hd95": 0.0, "nsd": 1.0}
    if not np.any(surface_gt) or not np.any(surface_pred):
        return {"hd95": float("inf"), "nsd": 0.0}

    distance_to_gt = ndimage.distance_transform_edt(~surface_gt, sampling=spacing_mm)
    distance_to_pred = ndimage.distance_transform_edt(~surface_pred, sampling=spacing_mm)

    distances_pred_to_gt = distance_to_gt[surface_pred]
    distances_gt_to_pred = distance_to_pred[surface_gt]
    all_surface_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred], axis=0)
    hd95 = float(np.percentile(all_surface_distances, 95.0))

    within_tol = float(np.count_nonzero(distances_pred_to_gt <= tolerance_mm) + np.count_nonzero(distances_gt_to_pred <= tolerance_mm))
    total_surface_points = float(distances_pred_to_gt.size + distances_gt_to_pred.size)
    nsd = within_tol / total_surface_points if total_surface_points > 0 else 1.0
    return {"hd95": hd95, "nsd": nsd}


def calculate_patient_surface_metrics(gt_masks, pred_masks, affines, nsd_tolerance_mm):
    gt_masks = build_patient_region_masks_from_gt(gt_masks)
    spacing_mm = get_volume_spacing_mm(affines)

    metrics = {}
    hd95_values = []
    nsd_values = []
    for curve_name in CURVE_NAMES:
        curve_metrics = compute_hd95_and_nsd(
            mask_gt=gt_masks[curve_name],
            mask_pred=pred_masks[curve_name],
            spacing_mm=spacing_mm,
            tolerance_mm=nsd_tolerance_mm,
        )
        metrics[curve_name] = curve_metrics
        if np.isfinite(curve_metrics["hd95"]):
            hd95_values.append(curve_metrics["hd95"])
        nsd_values.append(curve_metrics["nsd"])

    metrics["mean"] = {
        "hd95": float(np.mean(hd95_values)) if hd95_values else float("inf"),
        "nsd": float(np.mean(nsd_values)) if nsd_values else 0.0,
    }
    return metrics


def calculate_positive_curve_iou_stats(gt_contours, gt_labels, pred_contours, pred_labels):
    gt_contours_np = gt_contours.detach().cpu().numpy()
    gt_labels_np = gt_labels.detach().cpu().numpy()
    pred_contours_np = pred_contours.detach().cpu().numpy()
    pred_labels_np = pred_labels.detach().cpu().numpy()

    ious = []
    for slice_idx in range(gt_contours_np.shape[0]):
        for curve_idx in range(len(CURVE_NAMES)):
            if gt_labels_np[slice_idx, curve_idx] <= 0.5:
                continue
            if pred_labels_np[slice_idx, curve_idx] <= 0.5:
                ious.append(0.0)
                continue
            ious.append(contour_iou(gt_contours_np[slice_idx, curve_idx], pred_contours_np[slice_idx, curve_idx]))
    if not ious:
        return {"sum": 0.0, "count": 0, "mean": 0.0}
    return {"sum": float(np.sum(ious)), "count": len(ious), "mean": float(np.mean(ious))}


@torch.no_grad()
def test_one_epoch(model, loader, criterion, device, vis_dir, max_vis=None, nsd_tolerance_mm=2.0, mask_postprocess=0):
    model.eval()
    loss_sums = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_coef": 0.0,
        "loss_cont": 0.0,
        "loss_meas": 0.0,
        "loss_edge": 0.0,
    }

    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    save_idx = 0
    patient_results = []
    positive_curve_iou_sum = 0.0
    positive_curve_iou_count = 0
    mask_gt_count = 0

    for batch_idx, batch in enumerate(loader):
        images = [x.to(device) for x in batch["images"]]
        affines = [x.to(device) for x in batch["affines"]]
        ground_truth = build_ground_truth(batch, device)
        mask = [x.to(device) if x is not None else None for x in batch["mask"]]

        out = model(images)

        pred = out["fourier_coeffs"]
        cls_logits = out["cls_logits"]
        pred_label = [(torch.sigmoid(x) > 0.5).float() for x in cls_logits]

        losses = criterion(out, ground_truth, images)

        batch_size = len(ground_truth["volumes"])
        for key in loss_sums:
            loss_sums[key] += losses[key].item() * batch_size

        batch_size = len(images)
        for b in range(batch_size):
            has_mask_gt = ground_truth["fourier_coeffs"][b] is not None
            has_volume_gt = ground_truth["volumes"][b] is not None
            gt_volumes_ml = calculate_regional_volumes_ml(ground_truth["volumes"][b], images[b], affines[b]) if has_volume_gt else None
            pred_masks = build_patient_region_masks(out["contours"][b], pred_label[b], images[b])
            pred_masks = postprocess_region_masks(pred_masks, mode=mask_postprocess)
            pred_fourier_volumes_ml = calculate_regional_volumes_ml(out["volumes"][b], images[b], affines[b])
            pred_mask_volumes_ml = calculate_patient_mask_volumes_ml(
                pred_masks,
                images[b],
                affines[b],
            )
            matched_volume_pairs = match_pred_and_gt_volumes(
                case_id=batch["case_id"][b],
                gt_volumes_ml=gt_volumes_ml,
                pred_volumes_ml=pred_fourier_volumes_ml,
            )
            matched_mask_volume_pairs = match_pred_and_gt_volumes(
                case_id=batch["case_id"][b],
                gt_volumes_ml=gt_volumes_ml,
                pred_volumes_ml=pred_mask_volumes_ml,
            )

            patient_result = {
                "case_id": batch["case_id"][b],
                "gt_ml": gt_volumes_ml,
                "pred_fourier": pred_fourier_volumes_ml,
                "pred_mask": pred_mask_volumes_ml,
                "matched_volume_pairs": matched_volume_pairs,
                "matched_mask_volume_pairs": matched_mask_volume_pairs,
                "dice": None,
                "positive_curve_iou": None,
                "positive_curve_iou_count": 0,
            }

            if has_mask_gt:
                mask_gt_count += 1
                patient_dice = calculate_patient_dice_scores(
                    gt_masks=mask[b],
                    pred_masks=pred_masks,
                )
                patient_surface_metrics = calculate_patient_surface_metrics(
                    gt_masks=mask[b],
                    pred_masks=pred_masks,
                    affines=affines[b],
                    nsd_tolerance_mm=nsd_tolerance_mm,
                )
                patient_iou_stats = calculate_positive_curve_iou_stats(
                    gt_contours=ground_truth["contours"][b],
                    gt_labels=ground_truth["labels"][b],
                    pred_contours=out["contours"][b],
                    pred_labels=pred_label[b],
                )
                positive_curve_iou_sum += patient_iou_stats["sum"]
                positive_curve_iou_count += patient_iou_stats["count"]
                patient_result["dice"] = patient_dice
                patient_result["surface_metrics"] = patient_surface_metrics
                patient_result["positive_curve_iou"] = patient_iou_stats["mean"]
                patient_result["positive_curve_iou_count"] = patient_iou_stats["count"]
            else:
                patient_result["surface_metrics"] = None

            patient_results.append(patient_result)

            if max_vis is not None and max_vis > 0 and save_idx >= max_vis:
                continue
            save_path = vis_dir / f"{batch['case_id'][b]}.png"
            visualize_fourier_overlay_grid(
                images=images[b],
                gt_contours=None if not has_mask_gt else ground_truth["contours"][b],
                pred_contours=out["contours"][b],
                gt_labels=None if not has_mask_gt else ground_truth["labels"][b],
                pred_labels=pred_label[b],
                valid_slices=torch.ones(images[b].shape[0], dtype=torch.bool, device=device),
                save_path=save_path,
            )
            mask_save_path = vis_dir / f"{batch['case_id'][b]}_mask.png"
            visualize_mask_overlay_grid(
                images=images[b],
                region_masks=pred_masks,
                save_path=mask_save_path,
            )
            save_idx += 1

    test_metrics = {key: value / len(loader.dataset) for key, value in loss_sums.items()}
    test_metrics["positive_curve_iou"] = (
        positive_curve_iou_sum / positive_curve_iou_count if positive_curve_iou_count > 0 else 0.0
    )
    test_metrics["positive_curve_iou_count"] = positive_curve_iou_count
    test_metrics["mask_gt_count"] = mask_gt_count
    return test_metrics, patient_results


def write_volume_report(report_path, dataset_name, patient_results, test_losses):
    matched_metric_names = sorted(
        {
            metric_name
            for item in patient_results
            for metric_name in item["matched_volume_pairs"].keys()
        }
    )
    avg_abs_diff = {}
    std_abs_diff = {}
    for metric_name in matched_metric_names:
        diffs = [
            abs(item["matched_volume_pairs"][metric_name][1] - item["matched_volume_pairs"][metric_name][0])
            for item in patient_results
            if metric_name in item["matched_volume_pairs"]
        ]
        avg_abs_diff[metric_name] = float(np.mean(diffs)) if diffs else 0.0
        std_abs_diff[metric_name] = float(np.std(diffs)) if diffs else 0.0

    matched_mask_metric_names = sorted(
        {
            metric_name
            for item in patient_results
            for metric_name in item["matched_mask_volume_pairs"].keys()
        }
    )
    avg_mask_abs_diff = {}
    std_mask_abs_diff = {}
    for metric_name in matched_mask_metric_names:
        diffs = [
            abs(item["matched_mask_volume_pairs"][metric_name][1] - item["matched_mask_volume_pairs"][metric_name][0])
            for item in patient_results
            if metric_name in item["matched_mask_volume_pairs"]
        ]
        avg_mask_abs_diff[metric_name] = float(np.mean(diffs)) if diffs else 0.0
        std_mask_abs_diff[metric_name] = float(np.std(diffs)) if diffs else 0.0

    mask_patient_results = [item for item in patient_results if item["dice"] is not None]
    avg_dice = {}
    for curve_name in CURVE_NAMES:
        dice_values = [item["dice"][curve_name] for item in mask_patient_results]
        avg_dice[curve_name] = float(np.mean(dice_values)) if dice_values else 0.0
    avg_dice["mean"] = (
        float(np.mean([item["dice"]["mean"] for item in mask_patient_results]))
        if mask_patient_results
        else 0.0
    )
    avg_surface_metrics = {}
    for curve_name in CURVE_NAMES:
        hd95_values = [
            item["surface_metrics"][curve_name]["hd95"]
            for item in mask_patient_results
            if item["surface_metrics"] is not None and np.isfinite(item["surface_metrics"][curve_name]["hd95"])
        ]
        nsd_values = [
            item["surface_metrics"][curve_name]["nsd"]
            for item in mask_patient_results
            if item["surface_metrics"] is not None
        ]
        avg_surface_metrics[curve_name] = {
            "hd95": float(np.mean(hd95_values)) if hd95_values else float("inf"),
            "nsd": float(np.mean(nsd_values)) if nsd_values else 0.0,
        }
    mean_hd95_values = [
        item["surface_metrics"]["mean"]["hd95"]
        for item in mask_patient_results
        if item["surface_metrics"] is not None and np.isfinite(item["surface_metrics"]["mean"]["hd95"])
    ]
    mean_nsd_values = [
        item["surface_metrics"]["mean"]["nsd"]
        for item in mask_patient_results
        if item["surface_metrics"] is not None
    ]
    avg_surface_metrics["mean"] = {
        "hd95": float(np.mean(mean_hd95_values)) if mean_hd95_values else float("inf"),
        "nsd": float(np.mean(mean_nsd_values)) if mean_nsd_values else 0.0,
    }

    lines = [
        f"Dataset: {dataset_name}",
        (
            f"Test {format_loss_metrics(test_losses, prefix='loss_', precision=6)} | "
            f"pos_curve_iou={test_losses['positive_curve_iou']:.6f}"
        ),
        "",
    ]

    for item in patient_results:
        pred_fourier_text = ", ".join(f"{key}={value:.2f}" for key, value in item["pred_fourier"].items())
        pred_mask_text = ", ".join(f"{key}={value:.2f}" for key, value in item["pred_mask"].items())
        if item["gt_ml"] is not None:
            gt_ml_text = ", ".join(
                f"{key}={value:.2f}" if value >= 0 else f"{key}=-1"
                for key, value in item["gt_ml"].items()
            )
            lines.append(f"{item['case_id']} GT volume (mL) | {gt_ml_text}")
        lines.append(f"{item['case_id']} pred Fourier volume (mL) | {pred_fourier_text}")
        lines.append(f"{item['case_id']} pred mask volume (mL) | {pred_mask_text}")

        if item["matched_volume_pairs"]:
            match_text = ", ".join(
                f"{key} absdiff={abs(pred - gt):.2f}"
                for key, (gt, pred) in item["matched_volume_pairs"].items()
                if gt >= 0
            )
            lines.append(f"{item['case_id']} Fourier-vs-GT volume diff | {match_text}")
        if item["matched_mask_volume_pairs"]:
            match_text = ", ".join(
                f"{key} absdiff={abs(pred - gt):.2f}"
                for key, (gt, pred) in item["matched_mask_volume_pairs"].items()
                if gt >= 0
            )
            lines.append(f"{item['case_id']} Mask-vs-GT volume diff | {match_text}")

        if item["dice"] is not None:
            lines.append(
                f"{item['case_id']} Dice | "
                f"LV={item['dice']['LV']:.4f}, MYO={item['dice']['MYO']:.4f}, RV={item['dice']['RV']:.4f}, "
                f"Mean={item['dice']['mean']:.4f} | "
                f"PosCurveIoU={item['positive_curve_iou']:.4f} over {item['positive_curve_iou_count']} curves"
            )
            lines.append(
                f"{item['case_id']} Surface | "
                f"LV HD95={item['surface_metrics']['LV']['hd95']:.4f}mm NSD={item['surface_metrics']['LV']['nsd']:.4f}, "
                f"MYO HD95={item['surface_metrics']['MYO']['hd95']:.4f}mm NSD={item['surface_metrics']['MYO']['nsd']:.4f}, "
                f"RV HD95={item['surface_metrics']['RV']['hd95']:.4f}mm NSD={item['surface_metrics']['RV']['nsd']:.4f}, "
                f"Mean HD95={item['surface_metrics']['mean']['hd95']:.4f}mm NSD={item['surface_metrics']['mean']['nsd']:.4f}"
            )
        else:
            lines.append(f"{item['case_id']} Dice/IoU/Surface | skipped (no mask ground truth)")

    lines.extend([""])
    if matched_metric_names:
        avg_diff_text = ", ".join(
            f"{key}={avg_abs_diff[key]:.2f}±{std_abs_diff[key]:.2f}"
            for key in matched_metric_names
        )
        lines.extend(
            [
                "Average Fourier-vs-GT volume absolute difference (mL)",
                avg_diff_text,
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Average Fourier-vs-GT volume absolute difference (mL)",
                "Skipped: no comparable GT/pred volume pairs were available.",
                "",
            ]
        )

    if matched_mask_metric_names:
        avg_diff_text = ", ".join(
            f"{key}={avg_mask_abs_diff[key]:.2f}±{std_mask_abs_diff[key]:.2f}"
            for key in matched_mask_metric_names
        )
        lines.extend(
            [
                "Average Mask-vs-GT volume absolute difference (mL)",
                avg_diff_text,
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Average Mask-vs-GT volume absolute difference (mL)",
                "Skipped: no comparable GT/pred mask volume pairs were available.",
                "",
            ]
        )

    if mask_patient_results:
        lines.extend(
            [
                "Average Dice over patients",
                f"LV={avg_dice['LV']:.4f}, MYO={avg_dice['MYO']:.4f}, RV={avg_dice['RV']:.4f}, Mean={avg_dice['mean']:.4f}",
                "",
                "Average surface metrics over patients",
                (
                    f"LV HD95={avg_surface_metrics['LV']['hd95']:.4f}mm NSD={avg_surface_metrics['LV']['nsd']:.4f}, "
                    f"MYO HD95={avg_surface_metrics['MYO']['hd95']:.4f}mm NSD={avg_surface_metrics['MYO']['nsd']:.4f}, "
                    f"RV HD95={avg_surface_metrics['RV']['hd95']:.4f}mm NSD={avg_surface_metrics['RV']['nsd']:.4f}, "
                    f"Mean HD95={avg_surface_metrics['mean']['hd95']:.4f}mm NSD={avg_surface_metrics['mean']['nsd']:.4f}"
                ),
                "",
                "Average IoU over all GT-positive curves",
                f"IoU={test_losses['positive_curve_iou']:.4f} over {test_losses['positive_curve_iou_count']} curves",
            ]
        )
    else:
        lines.extend(
            [
                "Average Dice over patients",
                "Skipped: no mask ground truth was available.",
                "",
                "Average surface metrics over patients",
                "Skipped: no mask ground truth was available.",
                "",
                "Average IoU over all GT-positive curves",
                "Skipped: no mask ground truth was available.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = build_test_dataset(args.dataset, args.split, args.apply_augmentation, args.fourier_m)
    print(f"Total test cases: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=short_axis_collate_fn,
    )

    model = CardioContour(fourier_m=args.fourier_m).to(device)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    incompatible_keys = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if incompatible_keys.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing model weights required by the current architecture: "
            f"{incompatible_keys.missing_keys}"
        )

    criterion = CombinedTrainingLoss(
        supervised_losses=[FourierContourLoss(), VolumeRegressionLoss()],
        unsupervised_losses=[EdgeAlignmentLoss(fourier_m=args.fourier_m)],
    )

    test_losses, patient_results = test_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        vis_dir=args.vis_dir,
        max_vis=args.max_vis,
        nsd_tolerance_mm=args.nsd_tolerance_mm,
        mask_postprocess=args.mask_postprocess,
    )

    dataset_name = args.dataset.lower()
    report_path = Path(args.vis_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}.txt"
    write_volume_report(report_path, dataset_name, patient_results, test_losses)

    print(
        f"Test {format_loss_metrics(test_losses, prefix='loss_', precision=6)} | "
        f"pos_curve_iou={test_losses['positive_curve_iou']:.6f}"
    )
    print(f"Visualization saved to: {args.vis_dir}")
    print(f"Volume report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, default="checkpoints/cardiocontour_v1_pretrained_m6_last.pt")
    parser.add_argument("--vis_dir", type=str, default="work_dirs/exp_m6/test")
    parser.add_argument("--dataset", type=str, default="acdc", choices=["mnms", "mnms2", "acdc"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--apply_augmentation", default=False, action="store_true")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_vis", type=int, default=None)
    parser.add_argument("--nsd_tolerance_mm", type=float, default=2.0)
    parser.add_argument("--fourier_m", type=int, default=6)
    parser.add_argument("--mask_postprocess", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    main(args)
