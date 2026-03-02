"""
VLMSlim — Model Definitions
============================
Load and wrap teacher (VLM + vision-only) and student models.
All teachers are frozen; students are trainable.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import TeacherConfig, StudentConfig


# ──────────────────────────────────────────────────────────
# Teacher loading
# ──────────────────────────────────────────────────────────

class TeacherWrapper(nn.Module):
    """Unified interface for all teacher types.

    Returns:
        logits:   (B, num_classes) raw classification logits
        features: (B, feature_dim) penultimate features
    """

    def __init__(self, model, feature_extractor, classifier, preprocess, cfg: TeacherConfig):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.preprocess = preprocess
        self.cfg = cfg
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


def load_teacher(cfg: TeacherConfig, num_classes: int, device: str = "cuda") -> TeacherWrapper:
    """Load a teacher model based on its config."""

    if cfg.model_type in ("clip", "openclip"):
        return _load_openclip_teacher(cfg, num_classes, device)
    elif cfg.model_type == "siglip":
        return _load_siglip_teacher(cfg, num_classes, device)
    elif cfg.model_type == "deit":
        return _load_deit_teacher(cfg, num_classes, device)
    elif cfg.model_type == "resnet50":
        return _load_resnet50_teacher(cfg, num_classes, device)
    else:
        raise ValueError(f"Unknown teacher type: {cfg.model_type}")


def _load_openclip_teacher(cfg: TeacherConfig, num_classes: int, device: str):
    import open_clip

    model_name, pretrained = cfg.checkpoint.split("/")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    class FeatureExtractor(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.visual = clip_model.visual

        @torch.no_grad()
        def forward(self, x):
            return self.visual(x)

    class ZeroShotClassifier(nn.Module):
        """Build a zero-shot classification head from text embeddings."""
        def __init__(self, clip_model, tokenizer, class_names, templates, device):
            super().__init__()
            self.weight = self._build_classifier(
                clip_model, tokenizer, class_names, templates, device
            )

        def _build_classifier(self, clip_model, tokenizer, class_names, templates, device):
            import open_clip as oc
            zeroshot_weights = []
            for classname in class_names:
                texts = [t.format(classname) for t in templates]
                texts = tokenizer(texts).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(texts)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.mean(dim=0)
                    text_features /= text_features.norm()
                zeroshot_weights.append(text_features)
            w = torch.stack(zeroshot_weights, dim=1).to(device)
            return nn.Parameter(w, requires_grad=False)

        @torch.no_grad()
        def forward(self, features):
            features = features / features.norm(dim=-1, keepdim=True)
            # logit_scale is typically 100.0 for CLIP
            return 100.0 * features @ self.weight

    # Placeholder: class names and templates are set during caching
    feat_ext = FeatureExtractor(model)
    # We return a partial wrapper; classifier is built per-dataset in cache_teachers.py
    return feat_ext, preprocess, model, tokenizer


def _load_siglip_teacher(cfg: TeacherConfig, num_classes: int, device: str):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(cfg.checkpoint).to(device)
    processor = AutoProcessor.from_pretrained(cfg.checkpoint)
    model.eval()

    class FeatureExtractor(nn.Module):
        def __init__(self, siglip_model):
            super().__init__()
            self.vision_model = siglip_model.vision_model

        @torch.no_grad()
        def forward(self, x):
            outputs = self.vision_model(pixel_values=x)
            return outputs.pooler_output

    return FeatureExtractor(model), processor, model, None


def _load_deit_teacher(cfg: TeacherConfig, num_classes: int, device: str):
    import timm

    model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=num_classes)
    model = model.to(device).eval()
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    class FeatureExtractor(nn.Module):
        def __init__(self, deit):
            super().__init__()
            self.deit = deit

        @torch.no_grad()
        def forward(self, x):
            return self.deit.forward_features(x)[:, 0]  # CLS token

    class Classifier(nn.Module):
        def __init__(self, deit):
            super().__init__()
            self.head = deit.head

        @torch.no_grad()
        def forward(self, features):
            return self.head(features)

    return TeacherWrapper(model, FeatureExtractor(model), Classifier(model), preprocess, cfg)


def _load_resnet50_teacher(cfg: TeacherConfig, num_classes: int, device: str):
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    class FeatureExtractor(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.features = nn.Sequential(*list(resnet.children())[:-1])

        @torch.no_grad()
        def forward(self, x):
            return self.features(x).flatten(1)

    class Classifier(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.fc = resnet.fc

        @torch.no_grad()
        def forward(self, features):
            return self.fc(features)

    return TeacherWrapper(
        model, FeatureExtractor(model), Classifier(model), preprocess, cfg
    )


# ──────────────────────────────────────────────────────────
# Student loading
# ──────────────────────────────────────────────────────────

class StudentModel(nn.Module):
    """Student network with separate feature extraction and classification head.

    Attributes:
        backbone:  everything except the final FC
        head:      the classification FC layer
        feature_dim: dimension of backbone output
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_dim = feature_dim

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features

    def get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a single flat vector (for anchor loss)."""
        return torch.cat([p.view(-1) for p in self.parameters()])


def load_student(cfg: StudentConfig, num_classes: int, input_size: int = 32) -> StudentModel:
    """Load a student model from scratch (no pretraining)."""

    if cfg.name == "resnet18":
        return _build_resnet18(num_classes, input_size)
    elif cfg.name == "mobilenetv2":
        return _build_mobilenetv2(num_classes, input_size)
    elif cfg.name == "efficientnet_b0":
        return _build_efficientnet_b0(num_classes, input_size)
    else:
        raise ValueError(f"Unknown student: {cfg.name}")


def _build_resnet18(num_classes: int, input_size: int) -> StudentModel:
    model = tv_models.resnet18(weights=None, num_classes=num_classes)

    # Adapt for 32×32 input (CIFAR): smaller initial conv, no maxpool
    if input_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    # Split backbone and head
    backbone = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
        model.avgpool, nn.Flatten()
    )
    head = model.fc
    return StudentModel(backbone, head, feature_dim=512)


def _build_mobilenetv2(num_classes: int, input_size: int) -> StudentModel:
    model = tv_models.mobilenet_v2(weights=None, num_classes=num_classes)

    backbone = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    head = model.classifier
    return StudentModel(backbone, head, feature_dim=1280)


def _build_efficientnet_b0(num_classes: int, input_size: int) -> StudentModel:
    model = tv_models.efficientnet_b0(weights=None, num_classes=num_classes)

    backbone = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten()
    )
    head = model.classifier
    return StudentModel(backbone, head, feature_dim=1280)


# ──────────────────────────────────────────────────────────
# Projection adaptor Φ
# ──────────────────────────────────────────────────────────

class ProjectionAdaptor(nn.Module):
    """Projects teacher features to student feature dimension.

    Architecture: Linear → BN → ReLU → Linear
    Trained with the student, discarded at inference.
    """

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(teacher_dim, student_dim),
            nn.BatchNorm1d(student_dim),
            nn.ReLU(inplace=True),
            nn.Linear(student_dim, student_dim),
        )

    def forward(self, x):
        return self.proj(x)
