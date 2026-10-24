import os
import random
import argparse
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from models.cardiocontour import CardioContour
from datasets.mnms_dataset import MnMsDataset
from datasets.mnms2_dataset import MnMs2Dataset
from datasets.acdc_dataset import ACDCDataset
from datasets.base_dataset import short_axis_collate_fn
from utils.losses import FourierContourLoss, EdgeAlignmentLoss, VolumeRegressionLoss, CombinedTrainingLoss


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_ground_truth(batch, device):
    return {
        key: [value.to(device) if torch.is_tensor(value) else value for value in batch[key]]
        for key in ("fourier_coeffs", "contours", "labels", "volumes")
    }


def format_loss_metrics(losses, prefix="", precision=4):
    return " | ".join(
        f"{prefix}{name.removeprefix('loss_')}={value.item() if torch.is_tensor(value) else value:.{precision}f}"
        for name, value in losses.items()
    )


def merge_short_axis_batches(*batches):
    merged = {
        "images": [],
        "affines": [],
        "fourier_coeffs": [],
        "contours": [],
        "labels": [],
        "volumes": [],
        "mask": [],
        "case_id": [],
    }
    for batch in batches:
        for key in merged:
            merged[key].extend(batch[key])
    return merged


class MixedBatchLoader:
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def __len__(self):
        return len(self.labeled_loader)

    def __iter__(self):
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader) if self.unlabeled_loader is not None else None

        for labeled_batch in labeled_iter:
            if unlabeled_iter is None:
                yield labeled_batch
                continue

            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            yield merge_short_axis_batches(labeled_batch, unlabeled_batch)


def build_subset_loader(dataset, indices, batch_size, distributed, world_size, rank, shuffle, num_workers):
    subset = Subset(dataset, indices)
    sampler = (
        DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        if distributed
        else None
    )
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=short_axis_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, sampler


def split_labeled_unlabeled_indices(dataset):
    if isinstance(dataset, ConcatDataset):
        labeled_indices = []
        unlabeled_indices = []
        offset = 0
        for child_dataset in dataset.datasets:
            if not hasattr(child_dataset, "samples"):
                raise ValueError("Expected each training dataset to expose a .samples list")
            for local_idx, sample in enumerate(child_dataset.samples):
                global_idx = offset + local_idx
                if sample.get("gt_path") is None:
                    unlabeled_indices.append(global_idx)
                else:
                    labeled_indices.append(global_idx)
            offset += len(child_dataset)
        return labeled_indices, unlabeled_indices

    if not hasattr(dataset, "samples"):
        raise ValueError("Expected the training dataset to expose a .samples list")

    labeled_indices = []
    unlabeled_indices = []
    for idx, sample in enumerate(dataset.samples):
        if sample.get("gt_path") is None:
            unlabeled_indices.append(idx)
        else:
            labeled_indices.append(idx)
    return labeled_indices, unlabeled_indices


class ConsistencyLearningWrapper:
    def __init__(
        self,
        scale_range: tuple[float, float] = (0.8, 1.2),
        lambda_consistency: float = 5.0,
        positive_logit_threshold: float = 0.0,
        center_valid_margin: float = 0.1,
        debug_dir: str = "consistency_debug",
        max_debug_saves: int | None = None,
        fourier_m: int = 4,
    ):
        self.scale_range = scale_range
        self.lambda_consistency = lambda_consistency
        self.positive_logit_threshold = positive_logit_threshold
        self.center_valid_margin = center_valid_margin
        self.debug_dir = Path(debug_dir)
        self.max_debug_saves = max_debug_saves
        if self.max_debug_saves is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_counter = 0
        self._current_scale_factor = 1.0
        self.fourier_m = int(fourier_m)

    def view_augmentation(self, images):
        self._current_scale_factor = random.uniform(*self.scale_range)
        augmented_images = list(images)
        for image in images:
            augmented_images.append(self._scale_view(image, self._current_scale_factor))
        return augmented_images

    def consistency_loss(self, coeffs, cls_logits):
        if len(coeffs) % 2 != 0:
            raise ValueError(f"Expected an even number of coefficient tensors, got {len(coeffs)}")
        if len(cls_logits) != len(coeffs):
            raise ValueError(f"Expected cls_logits and coeffs to have the same length, got {len(cls_logits)} and {len(coeffs)}")

        batch_size = len(coeffs) // 2
        original_coeffs = coeffs[:batch_size]
        scaled_coeffs = coeffs[batch_size:]
        original_logits = cls_logits[:batch_size]
        scaled_logits = cls_logits[batch_size:]

        original = torch.cat(original_coeffs, dim=0)
        scaled = torch.cat(scaled_coeffs, dim=0)
        original_logit = torch.cat(original_logits, dim=0)
        scaled_logit = torch.cat(scaled_logits, dim=0)

        expected_scaled = original.clone()
        expected_scaled[..., :3] = expected_scaled[..., :3] * self._current_scale_factor

        min_center = self.center_valid_margin
        max_center = 1.0 - self.center_valid_margin
        center_in_bounds = (
            (expected_scaled[..., 0] >= min_center)
            & (expected_scaled[..., 0] <= max_center)
            & (expected_scaled[..., 1] >= min_center)
            & (expected_scaled[..., 1] <= max_center)
        )
        positive_mask = (
            (original_logit > self.positive_logit_threshold)
            & (scaled_logit > self.positive_logit_threshold)
            & center_in_bounds
        )
        if not torch.any(positive_mask):
            return coeffs[0].sum() * 0.0

        scaled_pos = scaled[positive_mask]
        expected_scaled_pos = expected_scaled[positive_mask]
        original_pos = original[positive_mask]

        loss_cxcy = torch.abs(scaled_pos[..., :2] - expected_scaled_pos[..., :2])
        loss_a0 = torch.abs(scaled_pos[..., 2:3] - expected_scaled_pos[..., 2:3])
        loss_rest = torch.abs(scaled_pos[..., 3:] - original_pos[..., 3:])
        loss_all = torch.cat([loss_cxcy, loss_a0, loss_rest], dim=-1)
        return self.lambda_consistency * loss_all.mean()

    def _scale_view(self, image, scale_factor):
        if image.ndim != 3:
            raise ValueError(f"Expected image with shape (S, H, W), got {tuple(image.shape)}")

        num_slices, height, width = image.shape
        scaled = F.interpolate(
            image.unsqueeze(1),
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ).squeeze(1)
        scaled_h, scaled_w = scaled.shape[-2:]

        if scaled_h >= height:
            scaled = scaled[:, :height, :]
        else:
            pad_bottom = height - scaled_h
            scaled = F.pad(scaled.unsqueeze(1), (0, 0, 0, pad_bottom), mode="replicate").squeeze(1)

        if scaled_w >= width:
            scaled = scaled[:, :, :width]
        else:
            pad_right = width - scaled_w
            scaled = F.pad(scaled.unsqueeze(1), (0, pad_right, 0, 0), mode="replicate").squeeze(1)

        return scaled

    def __call__(self, model, images):
        batch_size = len(images)
        debug_slice_idx = images[0].shape[0] // 2
        augmented_images = self.view_augmentation(images)

        out = model(augmented_images)
        all_coeffs = out["fourier_coeffs"]
        all_logits = out["cls_logits"]
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
            self._save_debug_visualization(
                original_image=images[0][debug_slice_idx],
                augmented_image=augmented_images[batch_size][debug_slice_idx],
                original_coeffs=all_coeffs[0][debug_slice_idx],
                augmented_coeffs=all_coeffs[batch_size][debug_slice_idx],
                original_logits=all_logits[0][debug_slice_idx],
                augmented_logits=all_logits[batch_size][debug_slice_idx],
            )

        wrapped_out = dict(out)
        for key in ("fourier_coeffs", "cls_logits", "contours", "volumes", "query_tokens"):
            wrapped_out[key] = out[key][:batch_size]
        wrapped_out["loss_cons"] = self.consistency_loss(all_coeffs, all_logits)
        return wrapped_out

    def _normalize_for_display(self, image):
        image = np.asarray(image, dtype=np.float32)
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max > image_min:
            return (image - image_min) / (image_max - image_min)
        return np.zeros_like(image)

    def _coeffs_to_contour(self, coeffs, num_points=128, fourier_order=None):
        if fourier_order is None:
            fourier_order = self.fourier_m
        coeffs = np.asarray(coeffs, dtype=np.float32)
        theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float32)
        r = np.full((num_points,), coeffs[2], dtype=np.float32)
        for k in range(1, fourier_order + 1):
            ck_real = coeffs[2 * k + 1]
            ck_imag = coeffs[2 * k + 2]
            r = r + 2.0 * coeffs[2] * (
                ck_real * np.cos(k * theta) - ck_imag * np.sin(k * theta)
            )
        x = r * np.cos(theta) + coeffs[0]
        y = r * np.sin(theta) + coeffs[1]
        return x, y

    def _overlay_curves(self, image, coeffs, logits):
        image_np = self._normalize_for_display(image.detach().cpu().numpy())
        overlay = np.repeat(image_np[..., None], 3, axis=-1)
        coeffs_np = coeffs.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        colors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        h, w = image_np.shape
        for curve_idx in range(min(coeffs_np.shape[0], 3)):
            if logits_np[curve_idx] <= 0.0:
                continue
            x, y = self._coeffs_to_contour(coeffs_np[curve_idx])
            x = np.clip(x * (w - 1), 0, w - 1)
            y = np.clip(y * (h - 1), 0, h - 1)
            xi = np.rint(x).astype(np.int32)
            yi = np.rint(y).astype(np.int32)
            overlay[yi, xi] = np.maximum(overlay[yi, xi], colors[curve_idx])
        return overlay

    def _save_debug_visualization(
        self,
        original_image,
        augmented_image,
        original_coeffs,
        augmented_coeffs,
        original_logits,
        augmented_logits,
    ):
        if self.max_debug_saves is None or self._debug_counter >= self.max_debug_saves:
            return

        original_overlay = self._overlay_curves(original_image, original_coeffs, original_logits)
        augmented_overlay = self._overlay_curves(augmented_image, augmented_coeffs, augmented_logits)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_overlay)
        axes[0].set_title("Original View")
        axes[1].imshow(augmented_overlay)
        axes[1].set_title("Scaled View")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.debug_dir / f"{self._debug_counter:04d}_sample.png", dpi=300)
        plt.close(fig)
        self._debug_counter += 1


# ============================================================
# Train / validation
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    world_size=1,
    max_norm=0.1,
    consistency_wrapper=None,
    warmup_scheduler=None,
    warmup_steps=0,
    epoch=None,
):
    model.train()
    loss_sums = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_coef": 0.0,
        "loss_cont": 0.0,
        "loss_meas": 0.0,
        "loss_edge": 0.0,
        "loss_cons": 0.0,
    }
    sample_count = 0
    prev_step_end = time.perf_counter()

    for step_idx, batch in enumerate(loader):
        sync_if_needed(device)
        step_start = time.perf_counter()
        data_time = step_start - prev_step_end

        data_start = time.perf_counter()
        images = [x.to(device) for x in batch["images"]]
        ground_truth = build_ground_truth(batch, device)
        sync_if_needed(device)

        optimizer.zero_grad()
        sync_if_needed(device)
        data_time += time.perf_counter() - data_start

        forward_start = time.perf_counter()
        if consistency_wrapper is not None:
            out = consistency_wrapper(model, images)
        else:
            out = model(images)
        pred = out["fourier_coeffs"]
        sync_if_needed(device)
        forward_time = time.perf_counter() - forward_start

        loss_start = time.perf_counter()
        losses = criterion(out, ground_truth, images, epoch=epoch)
        if "loss_cons" in out:
            losses["loss_cons"] = out["loss_cons"]
            losses["loss_total"] = losses["loss_total"] + out["loss_cons"]
        else:
            losses["loss_cons"] = pred[0].new_zeros(()) if len(pred) > 0 else torch.zeros((), device=device)
        loss = losses["loss_total"]
        sync_if_needed(device)
        loss_time = time.perf_counter() - loss_start

        backward_start = time.perf_counter()
        loss.backward()
        sync_if_needed(device)
        backward_time = time.perf_counter() - backward_start

        optim_start = time.perf_counter()
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            sync_if_needed(device)

        optimizer.step()
        if warmup_scheduler is not None and warmup_scheduler.last_epoch < warmup_steps:
            warmup_scheduler.step()
        sync_if_needed(device)
        optimizer_time = time.perf_counter() - optim_start
        current_lrs = "[" + ", ".join(f"{pg['lr']:.6g}" for pg in optimizer.param_groups) + "]"

        batch_size = len(ground_truth["volumes"])
        sample_count += batch_size
        for key in loss_sums:
            loss_sums[key] += losses[key].item() * batch_size

        step_total_time = time.perf_counter() - step_start
        prev_step_end = time.perf_counter()
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            if step_idx % 50 == 0:
                print(
                    f"[train step {step_idx:04d}] "
                    f"{format_loss_metrics(losses, prefix='loss_')} | "
                    f"data={data_time:.4f}s | "
                    f"forward={forward_time:.4f}s | "
                    f"loss={loss_time:.4f}s | "
                    f"backward={backward_time:.4f}s | "
                    f"optim={optimizer_time:.4f}s | "
                    f"total={step_total_time:.4f}s | "
                    f"lr={current_lrs}"
                )

    if world_size > 1:
        totals = torch.tensor(
            [
                loss_sums["loss_total"],
                loss_sums["loss_cls"],
                loss_sums["loss_coef"],
                loss_sums["loss_cont"],
                loss_sums["loss_meas"],
                loss_sums["loss_edge"],
                loss_sums["loss_cons"],
                float(sample_count),
            ],
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sums = {
            "loss_total": totals[0].item(),
            "loss_cls": totals[1].item(),
            "loss_coef": totals[2].item(),
            "loss_cont": totals[3].item(),
            "loss_meas": totals[4].item(),
            "loss_edge": totals[5].item(),
            "loss_cons": totals[6].item(),
        }
        sample_count = max(int(totals[7].item()), 1)

    return {key: value / sample_count for key, value in loss_sums.items()}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, world_size=1, consistency_wrapper=None, epoch=None):
    model.eval()
    loss_sums = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_coef": 0.0,
        "loss_cont": 0.0,
        "loss_meas": 0.0,
        "loss_edge": 0.0,
        "loss_cons": 0.0,
    }
    sample_count = 0

    for batch in loader:
        images = [x.to(device) for x in batch["images"]]
        ground_truth = build_ground_truth(batch, device)

        if consistency_wrapper is not None:
            out = consistency_wrapper(model, images)
        else:
            out = model(images)
        pred = out["fourier_coeffs"]
        losses = criterion(out, ground_truth, images, epoch=epoch)
        if "loss_cons" in out:
            losses["loss_cons"] = out["loss_cons"]
            losses["loss_total"] = losses["loss_total"] + out["loss_cons"]
        else:
            losses["loss_cons"] = pred[0].new_zeros(()) if len(pred) > 0 else torch.zeros((), device=device)

        batch_size = len(ground_truth["volumes"])
        sample_count += batch_size
        for key in loss_sums:
            loss_sums[key] += losses[key].item() * batch_size

    if world_size > 1:
        totals = torch.tensor(
            [
                loss_sums["loss_total"],
                loss_sums["loss_cls"],
                loss_sums["loss_coef"],
                loss_sums["loss_cont"],
                loss_sums["loss_meas"],
                loss_sums["loss_edge"],
                loss_sums["loss_cons"],
                float(sample_count),
            ],
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sums = {
            "loss_total": totals[0].item(),
            "loss_cls": totals[1].item(),
            "loss_coef": totals[2].item(),
            "loss_cont": totals[3].item(),
            "loss_meas": totals[4].item(),
            "loss_edge": totals[5].item(),
            "loss_cons": totals[6].item(),
        }
        sample_count = max(int(totals[7].item()), 1)

    return {key: value / sample_count for key, value in loss_sums.items()}


# ============================================================
# Optimizer / scheduler helpers
# ============================================================
def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def build_schedulers(optimizer, lr_milestones, warmup_steps=0, warmup_start_factor=0.1):
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=0.1,
    )
    warmup_scheduler = None
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
    return main_scheduler, warmup_scheduler


def load_from_model(model, ckpt_path, rank=0):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if is_main_process(rank):
        print(f"Loaded pretrained model from {ckpt_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")


def resume_training_state(model, optimizer, scheduler, warmup_scheduler, ckpt_path, rank=0):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain optimizer_state_dict")
    optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain scheduler_state_dict")
    scheduler.load_state_dict(scheduler_state)

    warmup_state = checkpoint.get("warmup_scheduler_state_dict")
    if warmup_scheduler is not None and warmup_state is not None:
        warmup_scheduler.load_state_dict(warmup_state)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    if is_main_process(rank):
        print(f"Resumed training state from {ckpt_path}")
        print(f"Resume epoch: {start_epoch}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    return start_epoch


def main(args):
    distributed, rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed + rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    if args.labeled_batch_size <= 0 or args.labeled_batch_size >= args.batch_size:
        raise ValueError(
            f"Expected labeled_batch_size to be in [1, {args.batch_size - 1}], got {args.labeled_batch_size}"
        )
    unlabeled_batch_size = args.batch_size - args.labeled_batch_size

    train_mnms_dataset = MnMsDataset(split="train", apply_augmentation=True, fourier_m=args.fourier_m)
    train_mnms2_dataset = MnMs2Dataset(split="train", apply_augmentation=True, fourier_m=args.fourier_m)
    train_acdc_dataset = ACDCDataset(split="train", apply_augmentation=True, fourier_m=args.fourier_m)
    val_dataset = ACDCDataset(split="test", apply_augmentation=False, fourier_m=args.fourier_m)

    train_dataset = ConcatDataset([train_mnms_dataset, train_mnms2_dataset, train_acdc_dataset])

    labeled_indices, unlabeled_indices = split_labeled_unlabeled_indices(train_dataset)
    if len(labeled_indices) == 0:
        raise ValueError("No labeled training samples were found")
    if len(unlabeled_indices) == 0:
        raise ValueError("No unlabeled training samples were found")

    labeled_loader, labeled_sampler = build_subset_loader(
        dataset=train_dataset,
        indices=labeled_indices,
        batch_size=args.labeled_batch_size,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        num_workers=args.num_workers,
    )
    unlabeled_loader, unlabeled_sampler = build_subset_loader(
        dataset=train_dataset,
        indices=unlabeled_indices,
        batch_size=unlabeled_batch_size,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        num_workers=args.num_workers,
    )
    train_loader = MixedBatchLoader(labeled_loader, unlabeled_loader)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    if is_main_process(rank):
        print(
            f"MnMs train samples: {len(train_mnms_dataset)}, "
            f"MnMs2 train samples: {len(train_mnms2_dataset)}, "
            f"ACDC train samples: {len(train_acdc_dataset)}, "
            f"labeled train samples: {len(labeled_indices)}, "
            f"unlabeled train samples: {len(unlabeled_indices)}, "
            f"mixed batch ratio: {args.labeled_batch_size}:{unlabeled_batch_size}, "
            f"ACDC val samples: {len(val_dataset)}"
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=short_axis_collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    base_model = CardioContour(fourier_m=args.fourier_m).to(device)
    if args.resume is not None and args.load_from is not None:
        raise ValueError("Use either --resume or --load_from, not both")
    if args.load_from is not None:
        load_from_model(base_model, args.load_from, rank)
    consistency_wrapper = ConsistencyLearningWrapper(fourier_m=args.fourier_m)
    model = base_model
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = CombinedTrainingLoss(
        supervised_losses=[FourierContourLoss(), VolumeRegressionLoss()],
        unsupervised_losses=[EdgeAlignmentLoss(fourier_m=args.fourier_m)],
    )

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler, warmup_scheduler = build_schedulers(
        optimizer=optimizer,
        lr_milestones=args.lr_milestones,
        warmup_steps=args.warmup_steps,
        warmup_start_factor=args.warmup_start_factor,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume is not None:
        model_to_resume = model.module if isinstance(model, DDP) else model
        start_epoch = resume_training_state(
            model=model_to_resume,
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_scheduler=warmup_scheduler,
            ckpt_path=args.resume,
            rank=rank,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if labeled_sampler is not None:
            labeled_sampler.set_epoch(epoch)
        if unlabeled_sampler is not None:
            unlabeled_sampler.set_epoch(epoch)

        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            world_size,
            args.clip_max_norm,
            consistency_wrapper,
            warmup_scheduler,
            args.warmup_steps,
            epoch,
        )
        val_losses = validate_one_epoch(
            model, val_loader, criterion, device, world_size, consistency_wrapper, epoch
        )

        scheduler.step()

        if is_main_process(rank):
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"{format_loss_metrics(train_losses, prefix='train_', precision=6)} | "
                f"{format_loss_metrics(val_losses, prefix='val_', precision=6)}"
            )

        model_to_save = model.module if isinstance(model, DDP) else model
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "warmup_scheduler_state_dict": None if warmup_scheduler is None else warmup_scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "args": vars(args),
        }

        if is_main_process(rank):
            torch.save(ckpt, save_dir / "last.pt")


    if is_main_process(rank):
        print("Done.")
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="work_dirs/exp_m6")

    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--labeled_batch_size", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lr_milestones", type=int, nargs="+", default=[240, 330])
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_start_factor", type=float, default=0.05)
    parser.add_argument("--clip_max_norm", type=float, default=0.5)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fourier_m", type=int, default=6)

    args = parser.parse_args()
    main(args)
