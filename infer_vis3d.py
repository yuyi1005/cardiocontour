import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.base_dataset import BaseShortAxisDataset
from train import seed_everything


CURVE_NAMES = ("LV", "MYO", "RV")
CURVE_RGBA = (
    np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32),
    np.array([0.0, 1.0, 0.0, 0.4], dtype=np.float32),
    np.array([0.0, 0.0, 1.0, 0.2], dtype=np.float32),
)


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
        case_id = self.nifti_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        self.samples = [{
            "case_id": case_id,
            "image_path": self.nifti_path,
            "gt_path": self.gt_path,
            "frame_idx": frame_idx,
        }]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._load_sample(
            sample["image_path"],
            sample["gt_path"],
            sample["case_id"],
            frame_idx=sample["frame_idx"],
        )


def contour_ras_from_points(contour, affine, height: int, width: int):
    contour = np.asarray(contour, dtype=np.float32)
    x_pix = np.clip(contour[:, 0] * (width - 1) + 0.5, 0.0, width - 1.0)
    y_pix = np.clip(contour[:, 1] * (height - 1) + 0.5, 0.0, height - 1.0)
    # NIfTI affine uses voxel coordinates in array-axis order: (row, col, slice).
    ijk = np.stack([y_pix, x_pix, np.zeros_like(x_pix)], axis=-1)
    linear = affine[:3, :3]
    offset = affine[:3, 3]
    return ijk @ linear.T + offset[None, :]


def build_gt_region_masks(mask_tensor: torch.Tensor):
    mask_np = mask_tensor.detach().cpu().numpy().astype(bool)
    return {
        "LV": mask_np[:, 0],
        "MYO": mask_np[:, 1] & ~mask_np[:, 0],
        "RV": mask_np[:, 2] & ~mask_np[:, 1],
    }


def mask_volume_vertices_faces(region_mask_shw: np.ndarray, base_affine: np.ndarray):
    from skimage import measure

    if region_mask_shw.ndim != 3 or not np.any(region_mask_shw):
        return None, None
    volume_hws = np.transpose(region_mask_shw.astype(np.float32), (1, 2, 0))
    verts_rcs, faces, _, _ = measure.marching_cubes(volume_hws, level=0.5)
    ijk = np.stack([verts_rcs[:, 0], verts_rcs[:, 1], verts_rcs[:, 2]], axis=1)
    ras = ijk @ base_affine[:3, :3].T + base_affine[:3, 3]
    return ras.astype(np.float64), faces.astype(np.int32)


def build_surface_ring_runs(slice_to_points: dict[int, np.ndarray]):
    slice_indices = sorted(slice_to_points.keys())
    if not slice_indices:
        return []

    runs = []
    current_run = [slice_indices[0]]
    for slice_idx in slice_indices[1:]:
        if slice_idx == current_run[-1] + 1:
            current_run.append(slice_idx)
        else:
            runs.append(current_run)
            current_run = [slice_idx]
    runs.append(current_run)

    ring_runs = []
    for run in runs:
        center_rings = [slice_to_points[slice_idx] for slice_idx in run]
        if len(center_rings) < 2:
            continue

        surface_rings = []
        first_step = center_rings[1] - center_rings[0]
        surface_rings.append(center_rings[0] - 0.5 * first_step)
        surface_rings.append(center_rings[0])

        for ring_a, ring_b in zip(center_rings[:-1], center_rings[1:]):
            surface_rings.append(0.5 * (ring_a + ring_b))
            surface_rings.append(ring_b)

        last_step = center_rings[-1] - center_rings[-2]
        surface_rings.append(center_rings[-1] + 0.5 * last_step)
        ring_runs.append(surface_rings)

    return ring_runs


def add_vtk_cap(renderer_points, renderer_polys, ring: np.ndarray, vtk, axial_offset: np.ndarray | None = None):
    ring_cap = ring.copy()
    if axial_offset is not None:
        ring_cap = ring_cap + axial_offset[None, :]

    cap_points = vtk.vtkPoints()
    poly_line = vtk.vtkPolyLine()
    num_points = ring_cap.shape[0]
    poly_line.GetPointIds().SetNumberOfIds(num_points + 1)

    for point_idx, p in enumerate(ring_cap):
        cap_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        poly_line.GetPointIds().SetId(point_idx, point_idx)
    poly_line.GetPointIds().SetId(num_points, 0)

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(poly_line)

    cap_poly = vtk.vtkPolyData()
    cap_poly.SetPoints(cap_points)
    cap_poly.SetLines(lines)

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(cap_poly)
    triangulator.Update()

    tri_output = triangulator.GetOutput()
    point_offset = renderer_points.GetNumberOfPoints()
    for point_idx in range(tri_output.GetNumberOfPoints()):
        renderer_points.InsertNextPoint(tri_output.GetPoint(point_idx))

    polys = tri_output.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() != 3:
            continue
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, point_offset + id_list.GetId(0))
        tri.GetPointIds().SetId(1, point_offset + id_list.GetId(1))
        tri.GetPointIds().SetId(2, point_offset + id_list.GetId(2))
        renderer_polys.InsertNextCell(tri)


def add_predicted_vtk_actors(renderer, all_curve_points, color_map, vtk):
    for curve_idx, slice_to_points in all_curve_points.items():
        rgba = color_map[curve_idx]
        cap_offset_sign = 1.0 if curve_idx == 0 else (-1.0 if curve_idx == 2 else 0.0)
        lid_offset_mm = 0.1

        for slice_idx in sorted(slice_to_points.keys()):
            ras = slice_to_points[slice_idx]
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            num_points = ras.shape[0]
            for p in ras:
                points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            point_ids = np.arange(num_points, dtype=np.int32)
            next_ids = np.roll(point_ids, -1)
            keep = (point_ids % 2) == 0
            for a, b in zip(point_ids[keep], next_ids[keep]):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, int(a))
                line.GetPointIds().SetId(1, int(b))
                lines.InsertNextCell(line)
            line_poly = vtk.vtkPolyData()
            line_poly.SetPoints(points)
            line_poly.SetLines(lines)
            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputData(line_poly)
            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetColor(float(rgba[0]), float(rgba[1]), float(rgba[2]))
            line_actor.GetProperty().SetOpacity(0.35)
            line_actor.GetProperty().SetLineWidth(2.0)
            renderer.AddActor(line_actor)

        if float(rgba[3]) <= 0.0:
            continue

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        offset_idx = 0
        any_surface = False
        for ring_run in build_surface_ring_runs(slice_to_points):
            first_dir = ring_run[1].mean(axis=0) - ring_run[0].mean(axis=0)
            last_dir = ring_run[-1].mean(axis=0) - ring_run[-2].mean(axis=0)
            first_dir_norm = np.linalg.norm(first_dir)
            last_dir_norm = np.linalg.norm(last_dir)
            first_offset = None
            last_offset = None
            if cap_offset_sign != 0.0 and first_dir_norm > 1e-6:
                first_offset = -cap_offset_sign * lid_offset_mm * (first_dir / first_dir_norm)
            if cap_offset_sign != 0.0 and last_dir_norm > 1e-6:
                last_offset = cap_offset_sign * lid_offset_mm * (last_dir / last_dir_norm)

            for pts_a, pts_b in zip(ring_run[:-1], ring_run[1:]):
                if pts_a.shape[0] != pts_b.shape[0]:
                    continue
                num_points = pts_a.shape[0]
                for p in np.concatenate([pts_a, pts_b], axis=0):
                    points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
                for point_idx in range(num_points):
                    next_idx = (point_idx + 1) % num_points
                    tri1 = vtk.vtkTriangle()
                    tri1.GetPointIds().SetId(0, offset_idx + point_idx)
                    tri1.GetPointIds().SetId(1, offset_idx + num_points + point_idx)
                    tri1.GetPointIds().SetId(2, offset_idx + num_points + next_idx)
                    polys.InsertNextCell(tri1)

                    tri2 = vtk.vtkTriangle()
                    tri2.GetPointIds().SetId(0, offset_idx + point_idx)
                    tri2.GetPointIds().SetId(1, offset_idx + num_points + next_idx)
                    tri2.GetPointIds().SetId(2, offset_idx + next_idx)
                    polys.InsertNextCell(tri2)
                offset_idx += 2 * num_points
                any_surface = True

            first_ring = ring_run[0]
            add_vtk_cap(points, polys, first_ring, vtk, axial_offset=first_offset)
            any_surface = True

            last_ring = ring_run[-1]
            add_vtk_cap(points, polys, last_ring, vtk, axial_offset=last_offset)
            any_surface = True

        if not any_surface:
            continue

        surface_poly = vtk.vtkPolyData()
        surface_poly.SetPoints(points)
        surface_poly.SetPolys(polys)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface_poly)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(float(rgba[0]), float(rgba[1]), float(rgba[2]))
        actor.GetProperty().SetOpacity(float(rgba[3]))
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().BackfaceCullingOff()
        actor.GetProperty().FrontfaceCullingOff()
        renderer.AddActor(actor)


def add_gt_vtk_actors(renderer, gt_region_masks, base_affine, color_map, vtk):
    for curve_idx, curve_name in enumerate(CURVE_NAMES):
        rgba = color_map[curve_idx]
        verts, faces = mask_volume_vertices_faces(gt_region_masks[curve_name], base_affine)
        if verts is None or faces is None:
            continue
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        for point in verts:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        for face in faces:
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(face[0]))
            tri.GetPointIds().SetId(1, int(face[1]))
            tri.GetPointIds().SetId(2, int(face[2]))
            polys.InsertNextCell(tri)
        surface_poly = vtk.vtkPolyData()
        surface_poly.SetPoints(points)
        surface_poly.SetPolys(polys)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface_poly)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(float(rgba[0]), float(rgba[1]), float(rgba[2]))
        actor.GetProperty().SetOpacity(float(rgba[3]))
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().BackfaceCullingOff()
        actor.GetProperty().FrontfaceCullingOff()
        renderer.AddActor(actor)


def add_renderer_title(renderer, text, vtk):
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(text)
    text_prop = text_actor.GetTextProperty()
    text_prop.SetFontSize(24)
    text_prop.SetBold(True)
    text_prop.SetColor(0.2, 0.2, 0.2)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.03, 0.03)
    renderer.AddActor2D(text_actor)


def add_scene_lights(renderer, vtk):
    return None


def build_model(fourier_m: int, device: torch.device, sampling_k: int = 64):
    from models.cardiocontour import CardioContour

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


def smooth_predictions_along_slices(pred_contours: torch.Tensor, cls_logits: torch.Tensor, sigma: float):
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"--smooth must be positive or None, got {sigma}")

    num_slices = int(pred_contours.shape[0])
    if num_slices < 2:
        return pred_contours, cls_logits

    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=pred_contours.dtype, device=pred_contours.device)
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)

    def smooth_tensor(tensor: torch.Tensor):
        flat = tensor.reshape(num_slices, -1).transpose(0, 1).unsqueeze(0)
        padded = F.pad(flat, (radius, radius), mode="replicate")
        smoothed = F.conv1d(padded, kernel.view(1, 1, kernel_size).expand(flat.shape[1], 1, kernel_size), groups=flat.shape[1])
        return smoothed.squeeze(0).transpose(0, 1).reshape_as(tensor)

    return smooth_tensor(pred_contours), smooth_tensor(cls_logits)


def visualize_case_3d_contours_vtk(
    pred_contours: torch.Tensor,
    pred_labels: torch.Tensor,
    affines: torch.Tensor,
    height: int,
    width: int,
    pred_volumes_ml: dict,
    gt_region_masks: dict | None = None,
    gt_base_affine: np.ndarray | None = None,
    gt_volumes_ml: dict | None = None,
    screenshot_path: str | Path | None = None,
):
    try:
        import vtk
    except ImportError as e:
        raise ImportError("vtk is required for --interactive-3d. Install with: pip install vtk") from e

    contours_np = pred_contours.detach().cpu().numpy()
    labels_np = pred_labels.detach().cpu().numpy()
    affines_np = affines.detach().cpu().numpy()

    color_map = {
        0: CURVE_RGBA[0].astype(np.float64),
        1: CURVE_RGBA[1].astype(np.float64),
        2: CURVE_RGBA[2].astype(np.float64),
    }

    all_curve_points = {curve_idx: {} for curve_idx in range(len(CURVE_NAMES))}

    for slice_idx in range(contours_np.shape[0]):
        affine = affines_np[slice_idx]
        for curve_idx in range(len(CURVE_NAMES) - 1, -1, -1):
            if labels_np[slice_idx, curve_idx] <= 0.5:
                continue

            ras = contour_ras_from_points(
                contours_np[slice_idx, curve_idx],
                affine=affine,
                height=height,
                width=width,
            )
            all_curve_points[curve_idx][slice_idx] = ras

    if not any(len(v) > 0 for v in all_curve_points.values()):
        print("No positive contours to show in VTK.")
        return

    bg = (0.96, 0.95, 0.88)
    has_gt = gt_region_masks is not None and gt_base_affine is not None
    viewports = [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)] if has_gt else [(0.0, 0.0, 1.0, 1.0)]

    renderer_pred = vtk.vtkRenderer()
    renderer_pred.SetViewport(*viewports[0])
    renderer_pred.SetBackground(*bg)
    renderer_pred.UseDepthPeelingOn()
    renderer_pred.SetMaximumNumberOfPeels(200)
    renderer_pred.SetOcclusionRatio(0.0)
    add_scene_lights(renderer_pred, vtk)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer_pred)
    render_window.SetSize(1280, 960)
    render_window.SetAlphaBitPlanes(1)
    render_window.SetMultiSamples(0)
    if screenshot_path is not None:
        render_window.SetOffScreenRendering(1)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    add_predicted_vtk_actors(renderer_pred, all_curve_points, color_map, vtk)
    add_renderer_title(renderer_pred, format_volume_label("Predicted", pred_volumes_ml), vtk)

    renderer_gt = None
    if has_gt:
        renderer_gt = vtk.vtkRenderer()
        renderer_gt.SetViewport(*viewports[1])
        renderer_gt.SetBackground(*bg)
        renderer_gt.UseDepthPeelingOn()
        renderer_gt.SetMaximumNumberOfPeels(200)
        renderer_gt.SetOcclusionRatio(0.0)
        add_scene_lights(renderer_gt, vtk)
        render_window.AddRenderer(renderer_gt)
        add_gt_vtk_actors(renderer_gt, gt_region_masks, gt_base_affine, color_map, vtk)
        add_renderer_title(renderer_gt, format_volume_label("GT", gt_volumes_ml), vtk)

    print("VTK coordinate frame meaning:")
    print("  +X = affine axis 0")
    print("  +Y = affine axis 1")
    print("  +Z = affine axis 2")
    print("Rotate/zoom the window to inspect the 3D contour stack.")

    shared_camera = vtk.vtkCamera()
    renderer_pred.SetActiveCamera(shared_camera)
    if renderer_gt is not None:
        renderer_gt.SetActiveCamera(shared_camera)

    renderer_pred.ResetCamera()
    if renderer_gt is not None:
        renderer_gt.ResetCameraClippingRange()
    if screenshot_path is not None:
        render_window.Render()
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()
        screenshot_path = Path(screenshot_path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(screenshot_path))
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        print(f"Saved screenshot: {screenshot_path}")
        return

    interactor.Initialize()
    render_window.Render()
    interactor.Start()


def regional_volumes_to_ml(volumes, images, affines):
    volume_np = volumes.detach().cpu().numpy().astype(np.float64)
    affine = affines.detach().cpu().numpy()[0]
    height, width = images.shape[-2:]
    pixel_area_mm2 = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    slice_spacing_mm = float(np.linalg.norm(affine[:3, 2]))
    scale = float(height * width) * pixel_area_mm2 * slice_spacing_mm / 1000.0
    return {name: float(value * scale) for name, value in zip(CURVE_NAMES, volume_np)}


def mask_volumes_to_ml(region_masks, affines):
    affines_np = affines.detach().cpu().numpy()
    volumes_mm3 = {name: 0.0 for name in CURVE_NAMES}
    for slice_idx in range(next(iter(region_masks.values())).shape[0]):
        affine = affines_np[slice_idx]
        pixel_area_mm2 = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
        slice_spacing_mm = float(np.linalg.norm(affine[:3, 2]))
        voxel_volume_mm3 = pixel_area_mm2 * slice_spacing_mm
        for name in CURVE_NAMES:
            volumes_mm3[name] += float(region_masks[name][slice_idx].sum()) * voxel_volume_mm3
    return {name: volumes_mm3[name] / 1000.0 for name in CURVE_NAMES}


def format_volume_label(title, volumes):
    return (
        f"{title}\n"
        f"LV={volumes['LV']:.2f} mL  MYO={volumes['MYO']:.2f} mL  RV={volumes['RV']:.2f} mL"
    )


def run_inference(args):
    seed_everything(args.seed)
    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NiftiSampleDataset(args.nifti_path, fourier_m=args.fourier_m, gt_path=args.gt_path, frame_idx=args.frame_idx)
    model = build_model(args.fourier_m, device, sampling_k=args.sampling_k)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    sample = dataset[0]
    images = [sample["images"].to(device)]
    affines = [sample["affines"].to(device)]
    gt_region_masks = build_gt_region_masks(sample["mask"]) if sample["mask"] is not None else None
    gt_base_affine = affines[0][0].detach().cpu().numpy() if gt_region_masks is not None else None
    gt_volumes_ml = mask_volumes_to_ml(gt_region_masks, affines[0]) if gt_region_masks is not None else None
    with torch.no_grad():
        output = model(images)

    pred_contours = output["contours"][0]
    cls_logits = output["cls_logits"][0]
    if args.smooth is not None:
        pred_contours, cls_logits = smooth_predictions_along_slices(pred_contours, cls_logits, sigma=args.smooth)
    pred_labels = (torch.sigmoid(cls_logits) > args.cls_threshold).float()
    pred_volumes_ml = regional_volumes_to_ml(output["volumes"][0], images[0], affines[0])
    volume_text = (
        f"pred LV={pred_volumes_ml['LV']:.2f} mL | "
        f"pred MYO={pred_volumes_ml['MYO']:.2f} mL | "
        f"pred RV={pred_volumes_ml['RV']:.2f} mL"
    )
    if gt_volumes_ml is not None:
        volume_text += (
            f" | gt LV={gt_volumes_ml['LV']:.2f} mL | "
            f"gt MYO={gt_volumes_ml['MYO']:.2f} mL | "
            f"gt RV={gt_volumes_ml['RV']:.2f} mL"
        )
    print(
        f"{sample['case_id']} | slices={images[0].shape[0]} | "
        f"positive_curves={int(pred_labels.sum().item())} | {volume_text}"
    )



    if args.interactive_3d:
        visualize_case_3d_contours_vtk(
            pred_contours=pred_contours.cpu(),
            pred_labels=pred_labels.cpu(),
            affines=affines[0].cpu(),
            height=images[0].shape[1],
            width=images[0].shape[2],
            pred_volumes_ml=pred_volumes_ml,
            gt_region_masks=gt_region_masks,
            gt_base_affine=gt_base_affine,
            gt_volumes_ml=gt_volumes_ml,
            screenshot_path=args.screenshot_path,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Render VTK predictions from one NIfTI file.")
    parser.add_argument("--nifti_path", default="data/ACDC/database/testing/patient101/patient101_frame01.nii.gz", type=str)
    parser.add_argument("--gt-path", default="data/ACDC/database/testing/patient101/patient101_frame01_gt.nii.gz", type=str)
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame to select when the NIfTI is 4D.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cardiocontour_v1_pretrained_m6_last.pt")
    parser.add_argument("--fourier_m", type=int, default=6)
    parser.add_argument("--sampling-k", type=int, default=64)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--smooth", type=float, default=1, help="Gaussian sigma in slices for smoothing contours and logits; use None/off to disable.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactive-3d", default=True, action="store_true")
    parser.add_argument("--screenshot-path", type=str, default=None, help="Render offscreen and save a PNG here instead of opening an interactive window.")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
