import torch
from torch import nn

from .backbone import build_backbone
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


class DETR(nn.Module):
    """DETR backbone/query module used by CardioContour."""

    def __init__(self, backbone, position_embedding, transformer, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = nn.Sequential(backbone)
        self.position_embedding = position_embedding

    def forward(self, images: torch.Tensor):
        features = self.backbone[0](images)
        src = list(features.values())[-1]
        pos = self.position_embedding(src).to(src.dtype)

        src = self.input_proj(src)
        hs = self.transformer(
            src.flatten(2).permute(2, 0, 1),
            self.query_embed.weight,
            pos.flatten(2).permute(2, 0, 1),
        )[0]
        return hs


_PRETRAINED_URLS = {
    "resnet50": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
    "resnet101": "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",
}


def _load_matching_weights(model, state_dict):
    model_state = model.state_dict()
    matching_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(matching_state, strict=False)


def build_detr(backbone_name="resnet50", hidden_dim=256, pretrained=True):
    if hidden_dim % 2 != 0:
        raise ValueError(f"hidden_dim must be even for sine position encoding, got {hidden_dim}")
    if hidden_dim % 8 != 0:
        raise ValueError(f"hidden_dim must be divisible by 8 for the default transformer heads, got {hidden_dim}")

    backbone = build_backbone(name=backbone_name, pretrained=pretrained)
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    model = DETR(backbone, position_embedding, transformer, num_queries=100)
    if pretrained:
        if backbone_name not in _PRETRAINED_URLS:
            raise ValueError(f"No pretrained DETR checkpoint configured for backbone_name={backbone_name!r}")
        checkpoint = torch.hub.load_state_dict_from_url(
            url=_PRETRAINED_URLS[backbone_name],
            map_location="cpu",
            check_hash=True,
        )
        _load_matching_weights(model, checkpoint["model"])
    return model
