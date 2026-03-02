"""
VLMSlim — Loss Functions
=========================
Total loss at phase k:
    L = α·L_CE + β·L_KD + γ·L_feat + λ·L_anchor

With simplified v2 hyperparameters:
    α = 0.1  [FIXED]
    β = 0.9  [FIXED]
    τ = 4.0  [FIXED]
    γ = auto [DERIVED at epoch 1]
    λ        [TUNED — only tuned hyperparameter]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class KDLoss(nn.Module):
    """Knowledge distillation loss: KL divergence on soft targets.

    L_KD = KL(softmax(teacher_logits / τ) || softmax(student_logits / τ)) × τ²

    The τ² scaling ensures gradients are comparable across temperatures.
    """

    def __init__(self, tau: float = 4.0):
        super().__init__()
        self.tau = tau

    def forward(self, student_logits: torch.Tensor, teacher_soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits:       (B, C) raw student logits
            teacher_soft_targets: (B, C) already softmax'd teacher target distribution
                                  (could be cumulative ensemble)
        """
        student_log_probs = F.log_softmax(student_logits / self.tau, dim=1)
        # teacher_soft_targets are pre-computed softmax — just use directly
        loss = F.kl_div(student_log_probs, teacher_soft_targets, reduction="batchmean")
        return loss * (self.tau ** 2)


class FeatureAlignmentLoss(nn.Module):
    """Feature alignment loss: MSE between projected teacher and student features.

    L_feat = MSE(Φ(f_teacher), f_student)
    """

    def forward(self, student_features: torch.Tensor,
                projected_teacher_features: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_features, projected_teacher_features)


class AnchorLoss(nn.Module):
    """Anchor regularization: L2 distance from phase snapshot.

    L_anchor = ||θ - θ̂_{k-1}||²

    Simple L2 penalty (no Fisher weighting for simplicity).
    """

    def __init__(self):
        super().__init__()
        self.snapshot = None  # Set at phase boundary

    def set_snapshot(self, model: nn.Module):
        """Take a snapshot of current model parameters."""
        self.snapshot = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def forward(self, model: nn.Module) -> torch.Tensor:
        if self.snapshot is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.snapshot:
                loss = loss + ((param - self.snapshot[name]) ** 2).sum()
        return loss


class CumulativeTargetBuilder:
    """Builds cumulative ensemble soft targets across phases.

    At phase k, the target is:
        q̂_k = Σ_{i=1}^{k} w_i · softmax(z_i / τ)

    where w_i = s_i / Σ_j s_j (score-weighted, normalized across ACTIVE teachers).
    """

    def __init__(self, teacher_scores: Dict[str, float], tau: float = 4.0):
        """
        Args:
            teacher_scores: {teacher_name: zero-shot accuracy} — used for weighting
            tau: temperature for softmax
        """
        self.teacher_scores = teacher_scores
        self.tau = tau
        self.active_teachers: List[str] = []

    def add_teacher(self, teacher_name: str):
        """Activate a new teacher (called at each phase boundary)."""
        if teacher_name not in self.active_teachers:
            self.active_teachers.append(teacher_name)

    def get_weights(self) -> Dict[str, float]:
        """Get normalized weights for currently active teachers."""
        total = sum(self.teacher_scores[t] for t in self.active_teachers)
        return {t: self.teacher_scores[t] / total for t in self.active_teachers}

    def compute_target(self, teacher_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted cumulative soft target.

        Args:
            teacher_logits: {teacher_name: (B, C) raw logits} for all available teachers

        Returns:
            (B, C) soft target distribution
        """
        weights = self.get_weights()
        target = None

        for teacher_name, w in weights.items():
            soft = F.softmax(teacher_logits[teacher_name] / self.tau, dim=1)
            if target is None:
                target = w * soft
            else:
                target = target + w * soft

        return target


class VLMSlimLoss(nn.Module):
    """Complete VLMSlim loss manager.

    Handles:
    - Loss computation for all four components
    - Automatic γ derivation at epoch 1
    - Phase transitions (snapshot + teacher activation)
    - Logging of individual loss components
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.9,
        tau: float = 4.0,
        lam: float = 0.1,
        teacher_scores: Dict[str, float] = None,
        use_cumulative: bool = True,
        use_anchor: bool = True,
        use_feature: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.lam = lam
        self.use_cumulative = use_cumulative
        self.use_anchor = use_anchor
        self.use_feature = use_feature

        # Loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDLoss(tau=tau)
        self.feat_loss = FeatureAlignmentLoss()
        self.anchor_loss = AnchorLoss()

        # Cumulative target builder
        self.target_builder = CumulativeTargetBuilder(teacher_scores or {}, tau=tau)

        # γ auto-derivation state
        self.gamma = None               # None = not yet derived
        self._gamma_kd_accum = []
        self._gamma_feat_accum = []
        self._gamma_calibration_batches = 100
        self._gamma_calibrated = False

    def begin_phase(self, teacher_name: str, model: nn.Module, phase_idx: int):
        """Called at the start of each training phase.

        - Activates the new teacher in the cumulative target
        - Takes anchor snapshot (except phase 0)
        """
        self.target_builder.add_teacher(teacher_name)

        if self.use_anchor and phase_idx > 0:
            self.anchor_loss.set_snapshot(model)
            print(f"  [Anchor] Snapshot taken at phase {phase_idx} boundary")

    def derive_gamma(self) -> float:
        """Derive γ from accumulated loss magnitudes. Called after calibration batches."""
        if not self._gamma_kd_accum or not self._gamma_feat_accum:
            print("  [γ] Warning: no calibration data, defaulting to γ=50.0")
            return 50.0

        mean_kd = sum(self._gamma_kd_accum) / len(self._gamma_kd_accum)
        mean_feat = sum(self._gamma_feat_accum) / len(self._gamma_feat_accum)

        if mean_feat < 1e-8:
            print("  [γ] Warning: feature loss near zero, defaulting to γ=50.0")
            return 50.0

        gamma = mean_kd / mean_feat
        print(f"  [γ] Derived: γ = {gamma:.4f} "
              f"(mean_KD={mean_kd:.6f}, mean_feat={mean_feat:.6f})")
        return gamma

    def forward(
        self,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Dict[str, torch.Tensor],
        projected_teacher_features: Optional[torch.Tensor],
        model: nn.Module,
        batch_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss and return component breakdown.

        Returns dict with keys: total, ce, kd, feat, anchor, and their float values.
        """
        device = student_logits.device
        losses = {}

        # ── 1. Cross-entropy loss ──
        l_ce = self.ce_loss(student_logits, labels)
        losses["ce"] = l_ce

        # ── 2. KD loss (cumulative or static target) ──
        if self.use_cumulative:
            soft_target = self.target_builder.compute_target(teacher_logits)
        else:
            # Static: average all available teacher logits
            all_soft = [F.softmax(t / self.tau, dim=1) for t in teacher_logits.values()]
            soft_target = torch.stack(all_soft).mean(dim=0)

        l_kd = self.kd_loss(student_logits, soft_target)
        losses["kd"] = l_kd

        # ── 3. Feature alignment loss ──
        l_feat = torch.tensor(0.0, device=device)
        if self.use_feature and projected_teacher_features is not None:
            l_feat = self.feat_loss(student_features, projected_teacher_features)
            losses["feat"] = l_feat

            # γ calibration: accumulate during first 100 batches
            if not self._gamma_calibrated:
                self._gamma_kd_accum.append(l_kd.item())
                self._gamma_feat_accum.append(l_feat.item())
                if len(self._gamma_kd_accum) >= self._gamma_calibration_batches:
                    self.gamma = self.derive_gamma()
                    self._gamma_calibrated = True
        else:
            losses["feat"] = l_feat

        # Use gamma (default to 50 if not yet calibrated)
        gamma = self.gamma if self.gamma is not None else 50.0

        # ── 4. Anchor loss ──
        l_anchor = torch.tensor(0.0, device=device)
        if self.use_anchor:
            l_anchor = self.anchor_loss(model)
            losses["anchor"] = l_anchor
        else:
            losses["anchor"] = l_anchor

        # ── Total loss ──
        total = (
            self.alpha * l_ce
            + self.beta * l_kd
            + gamma * l_feat
            + self.lam * l_anchor
        )
        losses["total"] = total
        losses["gamma_value"] = gamma

        return losses
