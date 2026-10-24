import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infer_vis3d import (
    CURVE_RGBA,
    add_predicted_vtk_actors,
    add_renderer_title,
    add_scene_lights,
    build_model,
    load_checkpoint,
)
import nibabel as nib

from datasets.base_dataset import BaseShortAxisDataset
from train import seed_everything


class CineNiftiDataset(BaseShortAxisDataset):
    def __init__(self, cine_path: str | Path, fourier_m: int):
        super().__init__(apply_augmentation=False, fourier_m=fourier_m)
        self.cine_path = Path(cine_path)
        if not self.cine_path.exists():
            raise FileNotFoundError(f"Cine NIfTI not found: {self.cine_path}")
        nii = nib.load(str(self.cine_path))
        if len(nii.shape) != 4:
            raise ValueError(f"Expected a 4D cine NIfTI, got shape {nii.shape} from {self.cine_path}")
        self.samples = [
            {
                "case_id": f"{self.cine_path.name.removesuffix('.nii.gz').removesuffix('.nii')}_f{frame_idx:02d}",
                "image_path": self.cine_path,
                "gt_path": None,
                "frame_idx": frame_idx,
            }
            for frame_idx in range(nii.shape[-1])
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


def render_window_to_rgb(render_window, vtk):
    from vtk.util.numpy_support import vtk_to_numpy

    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()

    vtk_image = window_to_image.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()
    image = vtk_to_numpy(scalars).reshape(height, width, 3)
    image = np.flipud(image)
    return image.astype(np.uint8)


def apply_camera_state(renderer, camera_state):
    if camera_state is None:
        renderer.ResetCamera()
        renderer.ResetCameraClippingRange()
        camera = renderer.GetActiveCamera()
        camera.Zoom(1.25)
        renderer.ResetCameraClippingRange()
        return {
            "position": camera.GetPosition(),
            "focal_point": camera.GetFocalPoint(),
            "view_up": camera.GetViewUp(),
            "clipping_range": camera.GetClippingRange(),
            "view_angle": camera.GetViewAngle(),
        }

    camera = renderer.GetActiveCamera()
    camera.SetPosition(*camera_state["position"])
    camera.SetFocalPoint(*camera_state["focal_point"])
    camera.SetViewUp(*camera_state["view_up"])
    camera.SetClippingRange(*camera_state["clipping_range"])
    camera.SetViewAngle(camera_state["view_angle"])
    renderer.ResetCameraClippingRange()
    return camera_state


def render_case_frame(sample, pred_contours, pred_labels, affines, camera_state=None, size=(1280, 720)):
    try:
        import vtk
    except ImportError as e:
        raise ImportError("vtk is required for infer_video.py. Install with: pip install vtk") from e

    images = sample["images"]
    bg = (0.96, 0.95, 0.88)
    color_map = {idx: CURVE_RGBA[idx].astype(np.float64) for idx in range(3)}

    renderer_pred = vtk.vtkRenderer()
    renderer_pred.SetViewport(0.0, 0.0, 1.0, 1.0)
    renderer_pred.SetBackground(*bg)
    renderer_pred.UseDepthPeelingOn()
    renderer_pred.SetMaximumNumberOfPeels(200)
    renderer_pred.SetOcclusionRatio(0.0)
    add_scene_lights(renderer_pred, vtk)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(*size)
    render_window.SetAlphaBitPlanes(1)
    render_window.SetMultiSamples(0)
    render_window.AddRenderer(renderer_pred)

    shared_camera = vtk.vtkCamera()
    renderer_pred.SetActiveCamera(shared_camera)

    add_predicted_vtk_actors(
        renderer_pred,
        {
            curve_idx: {
                slice_idx: sample_points
                for slice_idx, sample_points in build_curve_points(
                    pred_contours=pred_contours,
                    pred_labels=pred_labels,
                    affines=affines,
                    height=images.shape[1],
                    width=images.shape[2],
                    curve_idx=curve_idx,
                ).items()
            }
            for curve_idx in range(3)
        },
        color_map,
        vtk,
    )
    add_renderer_title(renderer_pred, "Predicted", vtk)

    render_window.Render()
    camera_state = apply_camera_state(renderer_pred, camera_state)
    render_window.Render()

    frame_rgb = render_window_to_rgb(render_window, vtk)
    return Image.fromarray(frame_rgb), camera_state


def build_curve_points(pred_contours, pred_labels, affines, height: int, width: int, curve_idx: int):
    from infer_vis3d import contour_ras_from_points

    contours_np = pred_contours.detach().cpu().numpy()
    labels_np = pred_labels.detach().cpu().numpy()
    affines_np = affines.detach().cpu().numpy()

    slice_to_points = {}
    for slice_idx in range(contours_np.shape[0]):
        if labels_np[slice_idx, curve_idx] <= 0.5:
            continue
        slice_to_points[slice_idx] = contour_ras_from_points(
            contours_np[slice_idx, curve_idx],
            affine=affines_np[slice_idx],
            height=height,
            width=width,
        )
    return slice_to_points


def interpolate_slice_dimension(pred_contours, cls_logits, affines, target_num_slices: int):
    num_slices = int(pred_contours.shape[0])
    if target_num_slices <= num_slices or num_slices < 2:
        return pred_contours, cls_logits, affines

    contours_interp = F.interpolate(
        pred_contours.permute(1, 2, 3, 0).reshape(1, -1, num_slices),
        size=target_num_slices,
        mode="linear",
        align_corners=True,
    ).reshape(pred_contours.shape[1], pred_contours.shape[2], pred_contours.shape[3], target_num_slices).permute(3, 0, 1, 2).contiguous()

    logits_interp = F.interpolate(
        cls_logits.transpose(0, 1).unsqueeze(0),
        size=target_num_slices,
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1).contiguous()

    affines_interp = F.interpolate(
        affines.reshape(num_slices, -1).transpose(0, 1).unsqueeze(0),
        size=target_num_slices,
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1).reshape(target_num_slices, 4, 4).contiguous()

    affines_interp[:, 3, :] = affines[0, 3, :]
    return contours_interp, logits_interp, affines_interp


def temporal_smooth_predictions(contours_list, logits_list, kernel):
    kernel = torch.as_tensor(kernel, dtype=contours_list[0].dtype)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    radius = int(kernel.numel() // 2)

    smoothed_contours = []
    smoothed_logits = []
    num_frames = len(contours_list)

    for frame_idx in range(num_frames):
        contour_accum = torch.zeros_like(contours_list[frame_idx])
        logit_accum = torch.zeros_like(logits_list[frame_idx])
        weight_accum = contours_list[frame_idx].new_zeros(())

        for kernel_idx, weight in enumerate(kernel):
            neighbor_idx = (frame_idx + kernel_idx - radius) % num_frames
            contour_accum = contour_accum + weight * contours_list[neighbor_idx]
            logit_accum = logit_accum + weight * logits_list[neighbor_idx]
            weight_accum = weight_accum + weight

        smoothed_contours.append(contour_accum / weight_accum)
        smoothed_logits.append(logit_accum / weight_accum)

    return smoothed_contours, smoothed_logits


def gaussian_kernel_1d(length: int = 11, sigma: float = 2.0):
    if length % 2 == 0:
        raise ValueError("Gaussian kernel length must be odd.")
    radius = length // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def run_inference(args):
    seed_everything(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CineNiftiDataset(args.cine_path, fourier_m=args.fourier_m)

    model = build_model(args.fourier_m, device, sampling_k=args.sampling_k)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    camera_state = None
    samples = []
    contours_by_frame = []
    logits_by_frame = []

    print(f"Rendering {args.cine_path} with {len(dataset)} frames")
    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            images = [sample["images"].to(device)]
            output = model(images)
            samples.append(sample)
            contours_by_frame.append(output["contours"][0].cpu())
            logits_by_frame.append(output["cls_logits"][0].cpu())

    if args.temporal_smooth:
        contours_by_frame, logits_by_frame = temporal_smooth_predictions(
            contours_by_frame,
            logits_by_frame,
            kernel=gaussian_kernel_1d(length=11, sigma=5.0),
        )

    for sample, pred_contours, cls_logits in zip(samples, contours_by_frame, logits_by_frame):
        render_affines = sample["affines"].cpu()
        pred_contours, cls_logits, render_affines = interpolate_slice_dimension(
            pred_contours,
            cls_logits,
            render_affines,
            target_num_slices=args.target_num_slices,
        )
        pred_labels = (torch.sigmoid(cls_logits) > args.cls_threshold).float().cpu()

        frame_img, camera_state = render_case_frame(
            sample=sample,
            pred_contours=pred_contours,
            pred_labels=pred_labels,
            affines=render_affines,
            camera_state=camera_state,
            size=(args.width, args.height),
        )
        frames.append(frame_img)

    if not frames:
        return

    stem = Path(args.cine_path).name.removesuffix(".nii.gz").removesuffix(".nii")
    output_path = output_dir / f"{stem}.{args.format}"
    if args.format == "gif":
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=args.duration_ms,
            loop=0,
        )
    else:
        fps = 1000.0 / max(float(args.duration_ms), 1.0)
        with imageio.get_writer(output_path, format="FFMPEG", fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render VTK predictions from a 4D cine NIfTI file.")
    parser.add_argument("--cine_path", default="data/ACDC/database/testing/patient101/patient101_4d.nii.gz", type=str)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cardiocontour_v1_pretrained_m6_last.pt")
    parser.add_argument("--fourier_m", type=int, default=6)
    parser.add_argument("--sampling-k", type=int, default=64)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="work_dirs/exp_m6/infer_video")
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-ms", type=int, default=60)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--temporal-smooth", action="store_true", default=True)
    parser.add_argument("--target-num-slices", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
