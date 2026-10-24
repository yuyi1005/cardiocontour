import numpy as np
from skimage import measure
from skimage.draw import polygon


def mask_to_contour(mask):
    """Extract the longest normalized contour from a binary mask."""
    mask = np.asarray(mask)
    if mask.size == 0 or not np.any(mask > 0):
        return np.zeros((0, 2), dtype=np.float32), 0.0

    contours = measure.find_contours(mask.astype(float), level=0.5)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32), 0.0

    contour = max(contours, key=lambda item: item.shape[0]).astype(np.float32, copy=False)
    if contour.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32), 0.0

    x = contour[:, 1] / float(mask.shape[1])
    y = contour[:, 0] / float(mask.shape[0])
    return np.stack([x, y], axis=-1).astype(np.float32, copy=False), 1.0


def contour_to_fourier(contour_xy, sampling_k=128, fourier_m=6):
    """Convert a normalized contour to Fourier coefficients."""
    contour_xy = np.asarray(contour_xy, dtype=np.float32)
    out_dim = 2 * fourier_m + 3
    empty_coeffs = np.zeros(out_dim, dtype=np.float32)
    if contour_xy.ndim != 2 or contour_xy.shape[1] != 2 or contour_xy.shape[0] < 3:
        return empty_coeffs, 0.0

    x = contour_xy[:, 0]
    y = contour_xy[:, 1]
    cx = (x.min() + x.max()) / 2.0
    cy = (y.min() + y.max()) / 2.0
    x = x - cx
    y = y - cy

    theta = np.mod(np.arctan2(y, x), 2 * np.pi)
    radius = np.sqrt(x**2 + y**2)
    order = np.argsort(theta)
    theta = theta[order]
    radius = radius[order]

    theta_round = np.round(theta, decimals=6)
    theta_unique = np.unique(theta_round)
    theta = np.asarray([theta[theta_round == value].mean() for value in theta_unique])
    radius = np.asarray([radius[theta_round == value].max() for value in theta_unique])

    theta_wrap = np.concatenate([theta, theta[:1] + 2 * np.pi])
    radius_wrap = np.concatenate([radius, radius[:1]])
    theta_uniform = np.linspace(0, 2 * np.pi, sampling_k, endpoint=False)
    radius_uniform = np.interp(theta_uniform, theta_wrap, radius_wrap)
    fft_radius = np.fft.fft(radius_uniform) / sampling_k

    coeffs = [cx, cy, fft_radius[0].real]
    for freq in range(1, fourier_m + 1):
        coeff = fft_radius[freq] / coeffs[2]
        coeffs.extend([coeff.real, coeff.imag])
    return np.asarray(coeffs, dtype=np.float32), 1.0


def fourier_to_contour(coeffs, sampling_k=128, fourier_m=6):
    """Reconstruct a normalized contour from Fourier coefficients."""
    coeffs = np.asarray(coeffs, dtype=np.float32)
    expected_dim = 2 * fourier_m + 3
    if coeffs.shape[-1] != expected_dim:
        raise ValueError(f"Expected coeff dim {expected_dim}, got {coeffs.shape[-1]}")

    fft_radius = np.zeros(sampling_k, dtype=np.complex64)
    cx, cy, r0 = coeffs[:3]
    fft_radius[0] = r0 + 0j

    idx = 3
    for freq in range(1, fourier_m + 1):
        real = coeffs[idx] * r0
        imag = coeffs[idx + 1] * r0
        fft_radius[freq] = real + 1j * imag
        fft_radius[-freq] = real - 1j * imag
        idx += 2

    radius = np.fft.ifft(fft_radius * sampling_k).real
    theta = np.linspace(0, 2 * np.pi, sampling_k, endpoint=False)
    x = radius * np.cos(theta) + cx
    y = radius * np.sin(theta) + cy
    return x, y


def contour_to_mask(contour, height, width):
    """Rasterize a normalized contour into a filled mask."""
    contour = np.asarray(contour, dtype=np.float32)
    x_pix = np.clip(contour[:, 0] * (width - 1) + 0.5, 0, width - 1)
    y_pix = np.clip(contour[:, 1] * (height - 1) + 0.5, 0, height - 1)
    mask = np.zeros((height, width), dtype=bool)
    rr, cc = polygon(y_pix, x_pix, shape=mask.shape)
    mask[rr, cc] = True
    return mask


def contour_iou(contour_a, contour_b, raster_size=256):
    """Compute IoU between two normalized contours on a shared raster."""
    mask_a = contour_to_mask(contour_a, raster_size, raster_size)
    mask_b = contour_to_mask(contour_b, raster_size, raster_size)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 1.0 if union == 0 else float(intersection / union)


def save_contour_visualization(mask, contour, reconstructed_contour, reconstructed_mask, save_path):
    """Save a compact visual check for the contour conversion pipeline."""
    import matplotlib.pyplot as plt

    save_path = str(save_path)
    height, width = mask.shape
    contour_px = contour * np.asarray([width, height], dtype=np.float32)
    reconstructed_px = reconstructed_contour * np.asarray([width, height], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask, cmap="gray")
    axes[0].plot(contour_px[:, 0], contour_px[:, 1], color="cyan", linewidth=1.5, label="GT")
    axes[0].plot(reconstructed_px[:, 0], reconstructed_px[:, 1], "r--", linewidth=1.5, label="Fourier")
    axes[0].set_title("Mask + contours")
    axes[0].legend()

    axes[1].imshow(reconstructed_mask, cmap="gray")
    axes[1].plot(reconstructed_px[:, 0], reconstructed_px[:, 1], "r--", linewidth=1.5)
    axes[1].set_title("Reconstructed mask")

    diff = reconstructed_mask.astype(np.int8) - (mask > 0).astype(np.int8)
    axes[2].imshow(diff, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("Recon - GT")

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run a visual smoke test for the contour utility pipeline."""
    import argparse
    from pathlib import Path

    import nibabel as nib

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", default="data/ACDC/database/testing/patient101/patient101_frame01_gt.nii.gz")
    parser.add_argument("--slice-idx", type=int, default=4)
    parser.add_argument("--label", type=int, default=None, help="Optional label value to isolate from a multi-class mask.")
    parser.add_argument("--fourier-m", type=int, default=6)
    parser.add_argument("--sampling-k", type=int, default=128)
    parser.add_argument("--output", type=str, default="contour_utils_visualization.pdf")
    args = parser.parse_args()

    nii = nib.load(args.mask_path)
    volume = nii.get_fdata()
    mask = volume[:, :, args.slice_idx] if volume.ndim >= 3 else volume
    mask = mask == args.label if args.label is not None else mask > 0

    contour, label = mask_to_contour(mask)
    if label <= 0.5:
        raise ValueError("No valid contour found in the selected mask slice.")

    coeffs, label = contour_to_fourier(contour, sampling_k=args.sampling_k, fourier_m=args.fourier_m)
    x, y = fourier_to_contour(coeffs, sampling_k=args.sampling_k, fourier_m=args.fourier_m)
    reconstructed_contour = np.stack([x, y], axis=-1).astype(np.float32, copy=False)
    reconstructed_mask = contour_to_mask(reconstructed_contour, *mask.shape)
    iou = contour_iou(contour, reconstructed_contour)

    save_path = Path(args.output)
    save_contour_visualization(mask, contour, reconstructed_contour, reconstructed_mask, save_path)
    print(f"label={label:.1f}")
    print(f"coeffs={coeffs}")
    print(f"reconstruction_iou={iou:.6f}")
    print(f"saved={save_path}")


if __name__ == "__main__":
    main()
