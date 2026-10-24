import torch
import torch.nn as nn

from .detr import build_detr


class FourierContourHead(nn.Module):
    """
    Map query tokens to Fourier parameters.
    """

    def __init__(
        self,
        token_dim: int,
        fourier_m: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        cm_bound: float = 0.4,
        r0_scale: float = 0.05,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.fourier_m = fourier_m
        self.out_dim = 2 * fourier_m + 3
        self.cm_bound = cm_bound
        self.r0_scale = r0_scale

        self.head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(f"x must have at least 2 dims and end with token_dim, got {tuple(x.shape)}")
        if x.shape[-1] != self.token_dim:
            raise ValueError(
                f"Last dim mismatch: expected token_dim={self.token_dim}, got {x.shape[-1]}"
            )

        raw_coeffs = self.head(x)
        coeffs = raw_coeffs.clone()
        coeffs[..., 0:2] = torch.sigmoid(raw_coeffs[..., 0:2])
        coeffs[..., 2] = self.r0_scale * torch.exp(raw_coeffs[..., 2])
        coeffs[..., 3:] = self.cm_bound * (torch.sigmoid(raw_coeffs[..., 3:]) - 0.5)
        cls_logits = self.cls_head(x).squeeze(-1)
        return coeffs, cls_logits


class CrossSliceAttention(nn.Module):
    """
    Lightweight residual self-attention across slices for each curve-query stream.
    """

    def __init__(
        self,
        token_dim: int,
        num_queries: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        distance_bias: float = 0.25,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.distance_bias = float(distance_bias)
        self.norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def _distance_attention_mask(self, num_slices: int, device, dtype):
        slice_ids = torch.arange(num_slices, device=device, dtype=dtype)
        distance = torch.abs(slice_ids[:, None] - slice_ids[None, :])
        return -self.distance_bias * distance

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape (S, Q, C), got {tuple(tokens.shape)}")
        num_slices, num_queries, token_dim = tokens.shape
        if num_queries != self.num_queries or token_dim != self.token_dim:
            raise ValueError(
                f"Expected tokens shape (*, {self.num_queries}, {self.token_dim}), got {tuple(tokens.shape)}"
            )
        if num_slices <= 1:
            return tokens

        x = self.norm(tokens)
        attn_mask = self._distance_attention_mask(num_slices, tokens.device, tokens.dtype)
        context, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        context = context + self.ffn(context)
        return tokens + self.scale * context


class CardioContour(nn.Module):
    """
    CardioContour for variable-number multi-view input.
    """

    def __init__(
        self,
        token_dim=256,
        num_curve_queries=3,
        fourier_m=4,
        backbone_name="resnet50",
        pretrained_detr=True,
        sampling_k=64,
    ):
        super().__init__()
        self.num_curve_queries = num_curve_queries
        self.fourier_m = int(fourier_m)
        self.sampling_k = int(sampling_k)
        self.detr = build_detr(
            backbone_name=backbone_name,
            hidden_dim=token_dim,
            pretrained=pretrained_detr,
        )
        self.csa = CrossSliceAttention(
            token_dim=token_dim,
            num_queries=num_curve_queries,
            num_heads=8,
            distance_bias=0.25,
            init_scale=0.0,
        )
        self.fourier_head = FourierContourHead(token_dim=token_dim, fourier_m=self.fourier_m)

    def forward(self, images):
        """
        Args:
            images: list of tensors, each with shape (C, H, W), where C stacks slices

        Returns:
            dict with one list entry per input sample
        """
        if not isinstance(images, list):
            raise ValueError("Expected images to be a list")

        detr_images = []
        lengths = []
        for idx, image in enumerate(images):
            if image.ndim != 3:
                raise ValueError(f"Expected images[{idx}] with shape (C, H, W), got {tuple(image.shape)}")
            lengths.append(image.shape[0])
            detr_images.append(image[:, None].repeat(1, 3, 1, 1))

        detr_images = torch.cat(detr_images, dim=0)
        detr_out = self.detr(detr_images)
        slice_tokens = detr_out[-1, :, :self.num_curve_queries, :]

        slice_tokens_by_sample = []
        start = 0
        for num_slices in lengths:
            end = start + num_slices
            slice_tokens_by_sample.append(self.csa(slice_tokens[start:end]))
            start = end
        slice_tokens = torch.cat(slice_tokens_by_sample, dim=0)
        all_fourier_coeffs, all_cls_logits = self.fourier_head(slice_tokens)
        all_contours = self.coeffs_to_contours(all_fourier_coeffs)

        query_tokens = []
        fourier_coeffs = []
        cls_logits = []
        contours = []
        volumes = []
        start = 0
        for num_slices in lengths:
            end = start + num_slices
            query_tokens.append(slice_tokens[start:end])
            sample_coeffs = all_fourier_coeffs[start:end]
            sample_logits = all_cls_logits[start:end]
            fourier_coeffs.append(sample_coeffs)
            cls_logits.append(sample_logits)
            contours.append(all_contours[start:end])
            volumes.append(self.coeffs_to_regional_volume(sample_coeffs, sample_logits))
            start = end

        return {
            "query_tokens": query_tokens,
            "fourier_coeffs": fourier_coeffs,
            "cls_logits": cls_logits,
            "contours": contours,
            "volumes": volumes,
        }

    def coeffs_to_contours(self, coeffs: torch.Tensor) -> torch.Tensor:
        theta = torch.arange(self.sampling_k, device=coeffs.device, dtype=coeffs.dtype) * (2.0 * torch.pi / self.sampling_k)
        theta = theta.view(*([1] * (coeffs.ndim - 1)), self.sampling_k)
        cx = coeffs[..., 0:1]
        cy = coeffs[..., 1:2]
        r0 = coeffs[..., 2:3]
        r = r0.expand(*coeffs.shape[:-1], self.sampling_k)
        for k in range(1, self.fourier_m + 1):
            ck_real = coeffs[..., 2 * k + 1 : 2 * k + 2]
            ck_imag = coeffs[..., 2 * k + 2 : 2 * k + 3]
            k_theta = k * theta
            r = r + 2.0 * r0 * (ck_real * torch.cos(k_theta) - ck_imag * torch.sin(k_theta))
        x = r * torch.cos(theta) + cx
        y = r * torch.sin(theta) + cy
        return torch.stack([x, y], dim=-1)

    def coeffs_to_regional_volume(self, coeffs: torch.Tensor, cls_logits: torch.Tensor) -> torch.Tensor:
        r0 = coeffs[..., 2]
        ck = coeffs[..., 3:].reshape(*coeffs.shape[:-1], self.fourier_m, 2)
        area_factor = 1.0 + 2.0 * ck.square().sum(dim=(-1, -2))
        cumulative_area = torch.pi * r0.square() * area_factor
        cumulative_volume = (torch.sigmoid(cls_logits) * cumulative_area).sum(dim=0)
        if cumulative_volume.shape[0] != 3:
            return cumulative_volume
        return torch.stack([
            cumulative_volume[0],
            cumulative_volume[1] - cumulative_volume[0],
            cumulative_volume[2] - cumulative_volume[1],
        ])
