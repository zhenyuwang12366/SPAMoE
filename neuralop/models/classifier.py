import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class TimmBackboneClassifier(nn.Module):
    def __init__(self, model_name: str, in_chans: int, num_classes: int, img_size: int = 224,
                 pretrained: bool = True, finetune_backbone: bool = True,
                 use_zscore_norm: bool = True, data_mean=None, data_std=None):
        super().__init__()
        self.use_norm = use_zscore_norm
        if self.use_norm:
            # Prefer passing dataset statistics (per-channel)
            assert data_mean is not None and data_std is not None
            mean = torch.tensor(data_mean, dtype=torch.float32)
            std  = torch.tensor(data_std,  dtype=torch.float32)
            if mean.ndim == 0: mean = mean.repeat(in_chans)
            if std.ndim  == 0: std  = std.repeat(in_chans)
            self.register_buffer("_mean", mean.view(1, -1, 1, 1))
            self.register_buffer("_std",  std.view(1, -1, 1, 1))
        else:
            self._mean = self._std = None

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # backbone features only (no classifier head)
            in_chans=in_chans,
            img_size=img_size,
            global_pool="token", # use [CLS] (or "avg" for mean over patches)
            # checkpoint_path="/data1/home/teacher/teacher_s/t108790/weights/vit_small_patch14_dinov2.lvd142m.safetensors"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, x):
        if self._mean is not None:
            x = (x - self._mean) / self._std
        feats = self.backbone(x)      # [B, D]
        logits = self.head(feats)     # [B, K]
        weights = F.softmax(logits, dim=1)
        return weights

def get_classifier(in_channels):
    model = TimmBackboneClassifier(
        model_name="vit_small_patch14_dinov2.lvd142m",
        in_chans=in_channels, num_classes=5, img_size=70,
        pretrained=False, finetune_backbone=True,   # set False if you only predict weights and do not fine-tune
        use_zscore_norm=False,
    )
    return model
