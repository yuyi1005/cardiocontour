import numpy as np
from scipy import ndimage as ndi


def zscore_nonzero(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).copy()
    mean = x.mean()
    std = x.std()
    if std < 1e-6:
        std = 1.0
    x = (x - mean) / std
    return x


def normalize_intensity(x: np.ndarray) -> np.ndarray:
    mean = 0.449
    std = 0.226
    vmin = x.min()
    vmax = x.max()
    if vmax - vmin < 1e-6:
        x = np.zeros_like(x)
    else:
        x = (x - vmin) / (vmax - vmin)
    x = (x - mean) / std
    return x


class Compose3D:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, gt, affine):
        for transform in self.transforms:
            image, gt, affine = transform(image, gt, affine)
        return image, gt, affine


class RandomGammaIntensity:
    """
    Apply a random gamma curve to the whole volume. This changes contrast in a
    way that is common across MRI scanners and sequences without altering the
    segmentation target.
    """
    def __init__(self, p=0.3, gamma_range=(0.7, 1.5)):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, image, gt, affine):
        if np.random.rand() > self.p:
            return image, gt, affine

        image = image.astype(np.float32, copy=False)
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max - image_min < 1e-6:
            return image, gt, affine

        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        image_norm = (image - image_min) / (image_max - image_min)
        image_aug = np.power(np.clip(image_norm, 0.0, 1.0), gamma)
        image_aug = image_aug * (image_max - image_min) + image_min
        return image_aug.astype(np.float32, copy=False), gt, affine


class RandomBiasField:
    """
    Simulate MRI intensity inhomogeneity with a smooth multiplicative bias
    field shared across slices.
    """
    def __init__(self, p=0.3, coeff_range=(0.2, 0.5), grid_shape=(3, 3)):
        self.p = p
        self.coeff_range = coeff_range
        self.grid_shape = grid_shape

    def __call__(self, image, gt, affine):
        if np.random.rand() > self.p:
            return image, gt, affine

        h, w, _ = image.shape
        gh, gw = self.grid_shape
        coarse = np.random.uniform(
            -self.coeff_range[1],
            self.coeff_range[1],
            size=(gh, gw),
        ).astype(np.float32)
        coarse -= coarse.mean()
        bias = ndi.zoom(
            coarse,
            zoom=(h / float(gh), w / float(gw)),
            order=3,
            mode="reflect",
            prefilter=True,
        )
        bias = bias[:h, :w]
        max_abs = float(np.max(np.abs(bias)))
        if max_abs > 1e-6:
            target_amp = np.random.uniform(self.coeff_range[0], self.coeff_range[1])
            bias = bias / max_abs * target_amp
        field = np.exp(bias).astype(np.float32)
        image_aug = image.astype(np.float32, copy=False) * field[..., None]
        return image_aug.astype(np.float32, copy=False), gt, affine


class RandomGaussianNoise:
    """
    Add low-amplitude Gaussian noise after geometric transforms to mimic MRI
    acquisition noise.
    """
    def __init__(self, p=0.25, sigma_range=(0.01, 0.05)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, image, gt, affine):
        if np.random.rand() > self.p:
            return image, gt, affine

        image = image.astype(np.float32, copy=False)
        image_std = float(image.std())
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1]) * max(image_std, 1e-6)
        noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        return (image + noise).astype(np.float32, copy=False), gt, affine


class RandomCenterCrop:
    """
    Randomly center-crop a 3D volume to simulate smaller input resolutions.

    The crop is centered in-plane and applied consistently to image, label,
    and affine. The output in-plane size is sampled independently for height
    and width within [ceil(original / 2), original].
    """
    def __init__(
        self,
        p=0.8,
        min_fraction=0.5,
        update_affine=True,
    ):
        self.p = p
        self.min_fraction = float(min_fraction)
        self.update_affine = update_affine

        if not (0.0 < self.min_fraction <= 1.0):
            raise ValueError(f"min_fraction must be in (0, 1], got {self.min_fraction}")

    def _sample_crop_size(self, h, w):
        crop_h_min = max(int(np.ceil(h * self.min_fraction)), 1)
        crop_h_max = h
        crop_w_min = max(int(np.ceil(w * self.min_fraction)), 1)
        crop_w_max = w

        crop_h = crop_h_max if crop_h_min == crop_h_max else np.random.randint(crop_h_min, crop_h_max + 1)
        crop_w = crop_w_max if crop_w_min == crop_w_max else np.random.randint(crop_w_min, crop_w_max + 1)
        return crop_h, crop_w

    def _update_affine_for_crop(self, affine, top, left):
        voxel_xform = np.eye(4, dtype=np.float32)
        voxel_xform[0, 3] = float(top)
        voxel_xform[1, 3] = float(left)
        return (affine @ voxel_xform).astype(np.float32)

    def __call__(self, image, gt, affine):
        if image.shape != gt.shape:
            raise ValueError(f"image and gt must have the same shape, got {image.shape} and {gt.shape}")
        if image.ndim != 3:
            raise ValueError(f"image and gt must be shaped (H, W, S), got {image.shape}")
        if affine.shape != (4, 4):
            raise ValueError(f"affine must be (4, 4), got {affine.shape}")

        if np.random.rand() > self.p:
            return image, gt, affine

        h, w, _ = image.shape
        crop_h, crop_w = self._sample_crop_size(h, w)
        top = max((h - crop_h) // 2, 0)
        left = max((w - crop_w) // 2, 0)

        image_t = image[top:top + crop_h, left:left + crop_w, :].astype(np.float32, copy=False)
        gt_t = gt[top:top + crop_h, left:left + crop_w, :].astype(np.float32, copy=False)

        if self.update_affine:
            affine = self._update_affine_for_crop(affine.astype(np.float32, copy=False), top, left)
        else:
            affine = affine.astype(np.float32, copy=False)

        return image_t, gt_t, affine


class RandomInPlaneAffine:
    """
    Apply the same random 2D affine transform to all slices of a 3D volume (H, W, S).

    Transform is defined in-plane only:
      - rotation
      - translation
      - isotropic scale

    The same transform is applied to:
      - image volume
      - gt volume
      - nifti affine (updated so world coords stay consistent)
    """
    def __init__(
        self,
        p=0.8,
        rotate_deg=15.0,
        translate_px=(10.0, 10.0),   # (tx_max, ty_max) in pixels
        scale_range=(0.9, 1.1),
        flip_lr_prob=0.0,
        flip_ud_prob=0.0,
        image_order=1,
        label_order=0,
        image_cval=0.0,
        label_cval=0.0,
        update_affine=True,
    ):
        self.p = p
        self.rotate_deg = rotate_deg
        self.translate_px = translate_px
        self.scale_range = scale_range
        self.flip_lr_prob = flip_lr_prob
        self.flip_ud_prob = flip_ud_prob
        self.image_order = image_order
        self.label_order = label_order
        self.image_cval = image_cval
        self.label_cval = label_cval
        self.update_affine = update_affine

    def _sample_params(self):
        angle_deg = np.random.uniform(-self.rotate_deg, self.rotate_deg)
        tx = np.random.uniform(-self.translate_px[0], self.translate_px[0])  # x = col
        ty = np.random.uniform(-self.translate_px[1], self.translate_px[1])  # y = row
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        flip_x = np.random.rand() < self.flip_lr_prob
        flip_y = np.random.rand() < self.flip_ud_prob
        return angle_deg, tx, ty, scale, flip_x, flip_y

    def _build_inverse_matrix_yx(self, H, W, angle_deg, tx, ty, scale, flip_x, flip_y):
        """
        Build output->input affine matrix in (y, x) coordinate order,
        suitable for scipy.ndimage.affine_transform.
        """
        angle = np.deg2rad(angle_deg)
        c, s = np.cos(angle), np.sin(angle)

        # Work in (x, y) first, then convert to (y, x)
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        T_center = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1],
        ], dtype=np.float32)

        T_uncenter = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        T_translate = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ], dtype=np.float32)

        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ], dtype=np.float32)

        S = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        F = np.array([
            [-1.0 if flip_x else 1.0, 0, 0],
            [0, -1.0 if flip_y else 1.0, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        # input(x,y) -> output(x,y)
        F_xy = T_translate @ T_uncenter @ R @ S @ F @ T_center

        # scipy affine_transform needs output -> input
        Finv_xy = np.linalg.inv(F_xy).astype(np.float32)

        # Convert between (x,y) and (y,x)
        P = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        Finv_yx = P @ Finv_xy @ P
        return Finv_yx

    def _apply_to_volume(self, vol, Finv_yx, order, cval):
        H, W, S = vol.shape
        out = np.empty_like(vol, dtype=np.float32)

        matrix = Finv_yx[:2, :2]
        offset = Finv_yx[:2, 2]

        for s in range(S):
            out[:, :, s] = ndi.affine_transform(
                vol[:, :, s],
                matrix=matrix,
                offset=offset,
                order=order,
                mode="constant",
                cval=cval,
                output_shape=(H, W),
                prefilter=(order > 1),
            )
        return out

    def _update_nifti_affine(self, affine, Finv_yx):
        """
        Update 4x4 voxel->world affine so that world coordinates remain correct
        after resampling.

        NIfTI voxel axes here are assumed to follow array axes:
          axis 0 = row (y)
          axis 1 = col (x)
          axis 2 = slice
        """
        voxel_xform = np.eye(4, dtype=np.float32)
        voxel_xform[:2, :2] = Finv_yx[:2, :2]
        voxel_xform[:2, 3] = Finv_yx[:2, 2]
        return (affine @ voxel_xform).astype(np.float32)

    def __call__(self, image, gt, affine):
        if np.random.rand() > self.p:
            return image, gt, affine

        H, W, S = image.shape
        angle_deg, tx, ty, scale, flip_x, flip_y = self._sample_params()
        Finv_yx = self._build_inverse_matrix_yx(H, W, angle_deg, tx, ty, scale, flip_x, flip_y)

        image_t = self._apply_to_volume(
            image, Finv_yx, order=self.image_order, cval=self.image_cval
        )
        gt_t = self._apply_to_volume(
            gt, Finv_yx, order=self.label_order, cval=self.label_cval
        )

        if self.update_affine:
            affine = self._update_nifti_affine(affine, Finv_yx)

        return image_t, gt_t, affine


class ResizeKeepAspect:
    """
    Resize MRI image / label volumes so the long edge does not exceed a target,
    while preserving aspect ratio.

    Rules:
      - if the long edge is larger than the target long edge, resize in-plane
        with aspect ratio preserved
      - update the affine when resizing so voxel-to-world coordinates remain
        correct for DICOM/NIfTI-style voxel-to-world affines
    """
    def __init__(
        self,
        max_h=256,
        max_w=256,
        image_order=1,
        label_order=0,
        image_cval=0.0,
        label_cval=0.0,
        update_affine=True,
    ):
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.image_order = image_order
        self.label_order = label_order
        self.image_cval = image_cval
        self.label_cval = label_cval
        self.update_affine = update_affine

        if self.max_h <= 0 or self.max_w <= 0:
            raise ValueError(f"max_h and max_w must be positive, got {self.max_h}, {self.max_w}")

    def _compute_output_size(self, h, w):
        scale = min(1.0, self.max_h / float(h), self.max_w / float(w))
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        return out_h, out_w, scale

    def _resize_volume(self, vol, out_h, out_w, order, cval):
        h, w, s = vol.shape
        if (h, w) == (out_h, out_w):
            return vol.astype(np.float32, copy=False)

        zoom = (out_h / float(h), out_w / float(w), 1.0)
        return ndi.zoom(
            vol,
            zoom=zoom,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=(order > 1),
        ).astype(np.float32, copy=False)

    def _update_affine_for_resize(self, affine, scale_y, scale_x):
        """
        Build new voxel->world affine after in-plane resizing.

        For output voxel coordinates [y_new, x_new, z, 1]^T, the corresponding
        input voxel coordinates are:
          y_old = y_new / scale_y
          x_old = x_new / scale_x

        Padding is added only on the right/bottom, so it does not shift the
        origin and therefore does not change the affine.
        """
        voxel_xform = np.eye(4, dtype=np.float32)
        voxel_xform[0, 0] = 1.0 / scale_y
        voxel_xform[1, 1] = 1.0 / scale_x
        return (affine @ voxel_xform).astype(np.float32)

    def __call__(self, image, gt, affine):
        if image.shape != gt.shape:
            raise ValueError(f"image and gt must have the same shape, got {image.shape} and {gt.shape}")
        if image.ndim != 3:
            raise ValueError(f"image and gt must be shaped (H, W, S), got {image.shape}")
        if affine.shape != (4, 4):
            raise ValueError(f"affine must be (4, 4), got {affine.shape}")

        h, w, s = image.shape
        out_h, out_w, scale = self._compute_output_size(h, w)

        image_t = self._resize_volume(image, out_h, out_w, order=self.image_order, cval=self.image_cval)
        gt_t = self._resize_volume(gt, out_h, out_w, order=self.label_order, cval=self.label_cval)

        if self.update_affine and scale != 1.0:
            affine = self._update_affine_for_resize(
                affine.astype(np.float32, copy=False),
                scale_y=out_h / float(h),
                scale_x=out_w / float(w),
            )
        else:
            affine = affine.astype(np.float32, copy=False)

        return image_t, gt_t, affine


class PadToShape:
    """
    Pad MRI image / label volumes to a fixed in-plane shape using right/bottom
    padding only.
    """
    def __init__(
        self,
        max_h=256,
        max_w=256,
        image_cval=0.0,
        label_cval=0.0,
    ):
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.image_cval = image_cval
        self.label_cval = label_cval

        if self.max_h <= 0 or self.max_w <= 0:
            raise ValueError(f"max_h and max_w must be positive, got {self.max_h}, {self.max_w}")

    def _pad_volume(self, vol, pad_h, pad_w, cval):
        return np.pad(
            vol,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=cval,
        )

    def __call__(self, image, gt, affine):
        if image.shape != gt.shape:
            raise ValueError(f"image and gt must have the same shape, got {image.shape} and {gt.shape}")
        if image.ndim != 3:
            raise ValueError(f"image and gt must be shaped (H, W, S), got {image.shape}")
        if affine.shape != (4, 4):
            raise ValueError(f"affine must be (4, 4), got {affine.shape}")

        h, w, _ = image.shape
        pad_h = self.max_h - h
        pad_w = self.max_w - w
        if pad_h < 0 or pad_w < 0:
            raise RuntimeError(f"Input shape {(h, w)} exceeds target {(self.max_h, self.max_w)}")

        image_t = self._pad_volume(image, pad_h, pad_w, cval=self.image_cval).astype(np.float32, copy=False)
        gt_t = self._pad_volume(gt, pad_h, pad_w, cval=self.label_cval).astype(np.float32, copy=False)
        return image_t, gt_t, affine.astype(np.float32, copy=False)
