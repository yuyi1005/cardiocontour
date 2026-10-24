from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class FourierContourLoss(nn.Module):
    def __init__(self, lambda_cont=50.0, lambda_coef=20.0, lambda_cls=1.0):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_coef = lambda_coef
        self.lambda_cls = lambda_cls

    def zero_losses(self, device=None, dtype=None, reference_tensor=None):
        zero = reference_tensor.new_zeros(()) if reference_tensor is not None else torch.zeros((), device=device, dtype=dtype or torch.float32)
        return {"loss_cls": zero, "loss_coef": zero, "loss_cont": zero}

    def forward(self, predictions, ground_truth):
        pred_coeffs = predictions["fourier_coeffs"]
        if len(pred_coeffs) == 0:
            return self.zero_losses()
        indices = [i for i, coeffs in enumerate(ground_truth["fourier_coeffs"]) if coeffs is not None]
        if not indices:
            return self.zero_losses(reference_tensor=pred_coeffs[0])

        pred_coeffs = torch.cat([predictions["fourier_coeffs"][i] for i in indices], dim=0)
        pred_logits = torch.cat([predictions["cls_logits"][i] for i in indices], dim=0)
        pred_contours = torch.cat([predictions["contours"][i] for i in indices], dim=0)
        gt_coeffs = torch.cat([ground_truth["fourier_coeffs"][i] for i in indices], dim=0)
        gt_labels = torch.cat([ground_truth["labels"][i] for i in indices], dim=0)
        gt_contours = torch.cat([ground_truth["contours"][i] for i in indices], dim=0)

        pred_coeffs = pred_coeffs.reshape(-1, pred_coeffs.shape[-1])
        pred_logits = pred_logits.reshape(-1)
        pred_contours = pred_contours.reshape(-1, pred_contours.shape[-2], pred_contours.shape[-1])
        gt_coeffs = gt_coeffs.reshape(-1, gt_coeffs.shape[-1])
        gt_labels = gt_labels.reshape(-1)
        gt_contours = gt_contours.reshape(-1, gt_contours.shape[-2], gt_contours.shape[-1])

        loss_cls = pred_coeffs.new_zeros(())
        loss_coef = pred_coeffs.new_zeros(())
        loss_cont = pred_coeffs.new_zeros(())
        if self.lambda_cls > 0:
            loss_cls = self.lambda_cls * F.binary_cross_entropy_with_logits(pred_logits, gt_labels)
        pos_mask = gt_labels > 0.5
        if self.lambda_coef > 0 and pos_mask.any():
            loss_coef = self.lambda_coef * F.l1_loss(pred_coeffs[pos_mask], gt_coeffs[pos_mask])
        if self.lambda_cont > 0 and pos_mask.any():
            loss_cont = self.lambda_cont * F.l1_loss(pred_contours[pos_mask], gt_contours[pos_mask])
        return {"loss_cls": loss_cls, "loss_coef": loss_coef, "loss_cont": loss_cont}


class VolumeRegressionLoss(nn.Module):
    def __init__(self, lambda_meas=10):
        super().__init__()
        self.lambda_meas = lambda_meas

    def zero_losses(self, device=None, dtype=None, reference_tensor=None):
        zero = reference_tensor.new_zeros(()) if reference_tensor is not None else torch.zeros((), device=device, dtype=dtype or torch.float32)
        return {"loss_meas": zero}

    def forward(self, predictions, ground_truth):
        pred_volumes = predictions["volumes"]
        if len(pred_volumes) == 0:
            return self.zero_losses()
        per_sample_losses = []
        for pred_volume, gt_volume in zip(pred_volumes, ground_truth["volumes"]):
            if gt_volume is None:
                continue
            gt_volume = gt_volume.to(device=pred_volume.device, dtype=pred_volume.dtype)
            valid_mask = gt_volume > 0
            if torch.any(valid_mask):
                per_sample_losses.append(F.l1_loss(pred_volume[valid_mask], gt_volume[valid_mask], reduction="mean"))
        if not per_sample_losses:
            return self.zero_losses(reference_tensor=pred_volumes[0])
        return {"loss_meas": self.lambda_meas * torch.stack(per_sample_losses).mean()}


class EdgeAlignmentLoss(nn.Module):
    def __init__(
        self,
        sampling_k=64,
        fourier_m=4,
        lambda_edge=0.1,
        start_epoch=120,
        contour_sigma_px=1.0,
        pred_threshold=0.5,
        debug_dir="edge_debug",
        max_debug_saves=None,
    ):
        super().__init__()
        self.sampling_k = sampling_k
        self.fourier_m = fourier_m
        self.lambda_edge = lambda_edge
        self.start_epoch = start_epoch
        self.contour_sigma_px = contour_sigma_px
        self.pred_threshold = pred_threshold
        self.debug_dir = Path(debug_dir)
        self.max_debug_saves = max_debug_saves
        if self.max_debug_saves is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_counter = 0
        self._local_offset_cache = {}
        self.edge_polarity = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)

    def forward(self, predictions, images, epoch=None):
        pred_contours = predictions["contours"]
        pred_logits = predictions["cls_logits"]
        if len(pred_contours) == 0:
            zero = torch.zeros(())
            return {"loss_edge": zero}

        debug_slice_idx = images[0].shape[0] // 2
        pred_contours = torch.cat(pred_contours, dim=0)
        pred_logits = torch.cat(pred_logits, dim=0)
        images = torch.cat(images, dim=0)
        dummy_cls_loss = 0.0 * pred_logits.sum()

        if epoch is not None and epoch < self.start_epoch:
            return {"loss_edge": dummy_cls_loss}

        pos_mask = torch.sigmoid(pred_logits) > self.pred_threshold
        if not pos_mask.any():
            return {"loss_edge": dummy_cls_loss}

        edge_maps = self._run_sobel(images)
        num_curves = pred_contours.shape[1]
        polarity_map = self.edge_polarity.to(device=pred_contours.device, dtype=pred_contours.dtype).view(1, num_curves)
        polarity_map = polarity_map.expand(pred_contours.shape[0], -1)
        pos_contours = pred_contours[pos_mask]
        pos_polarity = polarity_map[pos_mask]
        pos_edge_x_maps = edge_maps["edge_x_maps"].unsqueeze(1).expand(-1, num_curves, -1, -1, -1)[pos_mask]
        pos_edge_y_maps = edge_maps["edge_y_maps"].unsqueeze(1).expand(-1, num_curves, -1, -1, -1)[pos_mask]
        contour_x_maps, contour_y_maps = self._render_soft_contours(
            pos_contours,
            images.shape[-2],
            images.shape[-1],
            polarity=pos_polarity,
        )
        refined_edge_maps = pos_edge_x_maps * contour_x_maps + pos_edge_y_maps * contour_y_maps
        refined_edge_maps = refined_edge_maps.clamp_min(0.0)
        edge_alignment = refined_edge_maps.sum(dim=(-2, -1)).squeeze(-1) / float(self.sampling_k)
        loss_edge = self.lambda_edge * torch.exp(-edge_alignment.mean())

        should_save_debug = (
            self.max_debug_saves is not None
            and self._debug_counter < self.max_debug_saves
            and (
                not dist.is_available()
                or not dist.is_initialized()
                or dist.get_rank() == 0
            )
        )
        if should_save_debug:
            debug_contour_x_maps = contour_x_maps.detach().new_zeros(
                (images.shape[0], num_curves, 1, images.shape[-2], images.shape[-1])
            )
            debug_contour_y_maps = contour_y_maps.detach().new_zeros(
                (images.shape[0], num_curves, 1, images.shape[-2], images.shape[-1])
            )
            debug_refined_maps = refined_edge_maps.detach().new_zeros(
                (images.shape[0], num_curves, 1, images.shape[-2], images.shape[-1])
            )
            debug_contour_x_maps[pos_mask] = contour_x_maps.detach()
            debug_contour_y_maps[pos_mask] = contour_y_maps.detach()
            debug_refined_maps[pos_mask] = refined_edge_maps.detach()
            self._save_debug_images(
                images,
                edge_maps["edge_x_maps"],
                edge_maps["edge_y_maps"],
                debug_contour_x_maps,
                debug_contour_y_maps,
                debug_refined_maps,
                pos_mask,
                slice_idx=debug_slice_idx,
            )

        return {"loss_edge": loss_edge + dummy_cls_loss}

    def _run_sobel(self, images):
        image_min = images.amin(dim=(-2, -1), keepdim=True)
        image_max = images.amax(dim=(-2, -1), keepdim=True)
        sobel_input = (images - image_min) / (image_max - image_min).clamp_min(1e-6)
        sobel_input = sobel_input.unsqueeze(1)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(sobel_input, sobel_x, padding=1)
        grad_y = F.conv2d(sobel_input, sobel_y, padding=1)
        return {
            "edge_x_maps": grad_x,
            "edge_y_maps": grad_y,
        }


    def _render_soft_contours(self, contours, height, width, polarity=None):
        x = contours[..., 0].clamp(0.0, 1.0) * (width - 1)
        y = contours[..., 1].clamp(0.0, 1.0) * (height - 1)
        tangent_x = torch.roll(x, shifts=-1, dims=1) - torch.roll(x, shifts=1, dims=1)
        tangent_y = torch.roll(y, shifts=-1, dims=1) - torch.roll(y, shifts=1, dims=1)
        tangent_norm = torch.sqrt(tangent_x.square() + tangent_y.square() + 1e-6)
        tangent_x = tangent_x / tangent_norm
        tangent_y = tangent_y / tangent_norm
        normal_x = -tangent_y
        normal_y = tangent_x
        if polarity is not None:
            polarity = polarity.to(device=contours.device, dtype=contours.dtype).view(-1, 1)
            if polarity.shape[0] != contours.shape[0]:
                raise ValueError(
                    f"Expected polarity length {contours.shape[0]}, got {polarity.shape[0]}"
                )
            normal_x = normal_x * polarity
            normal_y = normal_y * polarity
        dx, dy = self._get_local_offsets(contours.device, contours.dtype)

        center_x = torch.floor(x).long()
        center_y = torch.floor(y).long()
        patch_x = (center_x.unsqueeze(-1) + dx.view(1, 1, -1)).clamp(0, width - 1)
        patch_y = (center_y.unsqueeze(-1) + dy.view(1, 1, -1)).clamp(0, height - 1)

        patch_x_float = patch_x.to(contours.dtype)
        patch_y_float = patch_y.to(contours.dtype)
        dist2 = (patch_x_float - x.unsqueeze(-1)) ** 2 + (patch_y_float - y.unsqueeze(-1)) ** 2
        weights = torch.exp(-dist2 / (2.0 * self.contour_sigma_px ** 2))

        flat_x_maps = contours.new_zeros((contours.shape[0], height * width))
        flat_y_maps = contours.new_zeros((contours.shape[0], height * width))
        flat_index = patch_y * width + patch_x
        weighted_x = weights * normal_x.unsqueeze(-1)
        weighted_y = weights * normal_y.unsqueeze(-1)
        flat_x_maps.scatter_add_(
            1,
            flat_index.reshape(contours.shape[0], -1),
            weighted_x.reshape(contours.shape[0], -1),
        )
        flat_y_maps.scatter_add_(
            1,
            flat_index.reshape(contours.shape[0], -1),
            weighted_y.reshape(contours.shape[0], -1),
        )

        contour_x_maps = flat_x_maps.view(contours.shape[0], 1, height, width)
        contour_y_maps = flat_y_maps.view(contours.shape[0], 1, height, width)
        contour_x_maps = contour_x_maps / contour_x_maps.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        contour_y_maps = contour_y_maps / contour_y_maps.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return contour_x_maps, contour_y_maps

    def _get_local_offsets(self, device, dtype):
        radius = max(int(np.ceil(3.0 * self.contour_sigma_px)), 1)
        cache_key = (str(device), str(dtype), radius)
        if cache_key in self._local_offset_cache:
            return self._local_offset_cache[cache_key]

        offset_range = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)
        dy, dx = torch.meshgrid(offset_range, offset_range, indexing="ij")
        dx = dx.reshape(-1)
        dy = dy.reshape(-1)
        self._local_offset_cache[cache_key] = (dx, dy)
        return dx, dy

    def _normalize_for_display(self, image):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image_min = float(image.min())
            image_max = float(image.max())
            if image_max > image_min:
                return (image - image_min) / (image_max - image_min)
            return np.zeros_like(image)
        image_min = image.min(axis=(0, 1), keepdims=True)
        image_max = image.max(axis=(0, 1), keepdims=True)
        denom = np.where(image_max > image_min, image_max - image_min, 1.0)
        return (image - image_min) / denom

    def _save_debug_images(
        self,
        images,
        edge_x_maps,
        edge_y_maps,
        contour_x_maps,
        contour_y_maps,
        refined_edge_maps,
        pos_mask,
        slice_idx,
    ):
        idx = int(slice_idx)
        debug_idx = self._debug_counter
        image_np = self._normalize_for_display(images[idx].detach().cpu().numpy())
        edge_x_np = self._normalize_for_display(edge_x_maps[idx, 0].detach().cpu().numpy())
        edge_y_np = self._normalize_for_display(edge_y_maps[idx, 0].detach().cpu().numpy())
        edge_mag_np = self._normalize_for_display(
            torch.sqrt(edge_x_maps[idx, 0].square() + edge_y_maps[idx, 0].square() + 1e-6).detach().cpu().numpy()
        )

        curve_x_rgb = contour_x_maps[idx, :, 0].detach().cpu().numpy().transpose(1, 2, 0)
        curve_y_rgb = contour_y_maps[idx, :, 0].detach().cpu().numpy().transpose(1, 2, 0)
        curve_mag_rgb = np.sqrt(curve_x_rgb ** 2 + curve_y_rgb ** 2)
        refined_rgb = refined_edge_maps[idx, :, 0].detach().cpu().numpy().transpose(1, 2, 0)
        active_mask = pos_mask[idx].detach().cpu().numpy().astype(np.float32).reshape(1, 1, -1)
        curve_x_rgb = self._normalize_for_display(curve_x_rgb * active_mask)
        curve_y_rgb = self._normalize_for_display(curve_y_rgb * active_mask)
        curve_mag_rgb = self._normalize_for_display(curve_mag_rgb * active_mask)
        refined_rgb = self._normalize_for_display(refined_rgb * active_mask)
        refined_heatmap = self._normalize_for_display(refined_rgb.mean(axis=-1))
        image_rgb = np.repeat(image_np[..., None], 3, axis=-1)
        curve_x_overlay = np.clip(0.7 * image_rgb + 0.8 * curve_x_rgb, 0.0, 1.0)
        curve_y_overlay = np.clip(0.7 * image_rgb + 0.8 * curve_y_rgb, 0.0, 1.0)
        curve_mag_overlay = np.clip(0.7 * image_rgb + 0.8 * curve_mag_rgb, 0.0, 1.0)
        active_curve_count = int(pos_mask[idx].sum().item())
        if active_curve_count > 0:
            slice_curve_alignment = refined_edge_maps[idx, :, 0].sum(dim=(-2, -1)) / float(self.sampling_k)
            slice_alignment = float(slice_curve_alignment[pos_mask[idx]].mean().item())
        else:
            slice_alignment = 0.0

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.reshape(-1)
        axes[0].imshow(image_np, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(edge_x_np, cmap="gray")
        axes[1].set_title("Edge X")
        axes[2].imshow(curve_x_overlay)
        axes[2].set_title("Contour X")
        axes[3].imshow(edge_mag_np, cmap="gray")
        axes[3].set_title("Edge Magnitude")
        axes[4].imshow(refined_heatmap, cmap="magma")
        axes[4].set_title(f"Agreement Heatmap\nalign={slice_alignment:.4f}")
        axes[5].imshow(edge_y_np, cmap="gray")
        axes[5].set_title("Edge Y")
        axes[6].imshow(curve_y_overlay)
        axes[6].set_title("Contour Y")
        axes[7].imshow(curve_mag_overlay)
        axes[7].set_title("Contour Magnitude")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.debug_dir / f"{debug_idx:04d}_sample.png", dpi=300)
        plt.close(fig)
        self._debug_counter += 1


class CombinedTrainingLoss(nn.Module):
    def __init__(self, supervised_losses: list[nn.Module], unsupervised_losses: list[nn.Module]):
        super().__init__()
        self.supervised_losses = nn.ModuleList(supervised_losses)
        self.unsupervised_losses = nn.ModuleList(unsupervised_losses)

    def forward(self, predictions, ground_truth, images, epoch=None):
        merged_losses = {}
        for loss_fn in self.supervised_losses:
            merged_losses.update(loss_fn(predictions, ground_truth))
        for loss_fn in self.unsupervised_losses:
            merged_losses.update(loss_fn(predictions, images, epoch=epoch))
        loss_total = None
        for value in merged_losses.values():
            loss_total = value if loss_total is None else loss_total + value
        if loss_total is None:
            zero = predictions["fourier_coeffs"][0].new_zeros(()) if len(predictions["fourier_coeffs"]) > 0 else torch.zeros(())
            loss_total = zero
        merged_losses["loss_total"] = loss_total
        return merged_losses