"""
KASPER — Composite training loss (Eq. 23, Section 3.2.4)

    L = L_Huber + lambda_s * sum_{p in Theta} |p|
              + lambda_c * L_contrastive
              + lambda_o * L_orth
              + lambda_b * L_balance

Lambda weights (Table 1):
    lambda_s = 0.001   (sparsity)
    lambda_c = 0.01    (contrastive)
    lambda_o = 0.01    (orthogonality)
    lambda_b = 0.05    (regime balance)

FIDELITY NOTE — read before using
----------------------------------
Four of the five terms have an explicit formula in the paper:
    - L_Huber:       standard Huber loss between y_hat and y  (named in Eq. 23's prose)
    - L1 sparsity:   sum |p| over all trainable parameters     (Eq. 23 itself)
    - L_contrastive: E[ ||z_i - z_j||^2 * y_ij ]                (Eq. 18)
    - L_orth:        || W W^T - I_R ||_F^2                      (Eq. 19)

L_balance has NO equation anywhere in the paper. It is described only in
prose as encouraging "balanced regime distribution across training
samples." Because of that, this module keeps it structurally separate:

    - composite_loss(..., include_balance=False)  [default]
      computes ONLY the four paper-verified terms. This is what "paper-
      verified formula" means here — nothing in the returned loss is
      invented.

    - composite_loss(..., include_balance=True)
      additionally adds unverified_regime_balance_penalty(), which is
      explicitly named and flagged as NOT from the paper — one reasonable
      way to implement "discourage regime collapse," not a reproduction
      of a stated formula. Its dict key is "balance_unverified" so it can
      never be silently mistaken for a verified term downstream.
"""

import torch
import torch.nn.functional as F

from regime_detection import RegimeDetectionLayer, contrastive_loss
from regime_forecasting import RegimeAdaptiveForecastingLayer

LAMBDA_SPARSITY = 0.001     # Table 1: L1 parameter sparsity penalty weight
LAMBDA_CONTRASTIVE = 0.01   # Table 1: contrastive loss weight
LAMBDA_ORTHOGONAL = 0.01    # Table 1: orthogonality regularization weight
LAMBDA_BALANCE = 0.05       # Table 1: regime balance penalty weight


def l1_sparsity(*modules: torch.nn.Module) -> torch.Tensor:
    """sum_{p in Theta} |p|, over every trainable parameter in the given modules. (Eq. 23)"""
    total = None
    for module in modules:
        for param in module.parameters():
            term = param.abs().sum()
            total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0)


def unverified_regime_balance_penalty(p: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation from uniform distribution using MSE (softer than KL).

    p: (batch, n_regimes) soft regime probabilities from KAN 1.
    """
    mean_probs = p.mean(dim=0)
    target_probs = torch.ones_like(mean_probs) / p.shape[1]
    return F.mse_loss(mean_probs, target_probs)


def composite_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    z: torch.Tensor,
    p: torch.Tensor,
    regime_ids: torch.Tensor,
    kan1: RegimeDetectionLayer,
    kan2: RegimeAdaptiveForecastingLayer,
    include_balance: bool = False,
    lambda_s: float = 0.001,
) -> dict:
    """
    Eq. 23. Returns every term individually (for logging) plus "total".

    Args:
        y_hat, y_true: (batch,) predicted vs. true returns.
        z:             (batch, hidden_dim) KAN 1 embedding, pre-softmax.
        p:             (batch, n_regimes) KAN 1 soft regime probabilities.
        regime_ids:    (batch,) hard regime assignment, e.g. p.argmax(-1).
        kan1, kan2:    the two KASPER layers, needed for their parameters
                        (sparsity) and orthogonality_loss().
        include_balance: if True, adds the explicitly-unverified L_balance
                        term on top of the four paper-verified ones.
        lambda_s:      dynamic weight for L1 parameter sparsity penalty.
    """
    l_huber = F.huber_loss(y_hat, y_true)
    # Eq. 23: L1 parameter sparsity term sums |p| over all trainable parameters Theta in both kan1 and kan2
    l_sparsity = l1_sparsity(kan1, kan2)
    l_contrastive = contrastive_loss(z, regime_ids)
    # Eq. 19: orthogonality on kan2.weights (W ∈ R^{R×F}, w^(r)_j = forecast weight for feature j in regime r)
    l_orth = kan2.orthogonality_loss()

    # Composite loss calculation (Eq. 23)
    total = (
        1.0 * l_huber
        + lambda_s * l_sparsity
        + LAMBDA_CONTRASTIVE * l_contrastive
        + LAMBDA_ORTHOGONAL * l_orth
    )

    terms = {
        "huber": l_huber,
        "sparsity_l1": l_sparsity,
        "contrastive": l_contrastive,
        "orthogonal": l_orth,
    }

    if include_balance:
        l_balance = unverified_regime_balance_penalty(p)
        total = total + LAMBDA_BALANCE * l_balance
        terms["balance_unverified"] = l_balance

    terms["total"] = total
    return terms


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, n_features, n_regimes = 32, 8, 3

    kan1 = RegimeDetectionLayer(n_features=n_features, hidden_dim=64, n_regimes=n_regimes)
    kan2 = RegimeAdaptiveForecastingLayer(n_features=n_features, n_regimes=n_regimes)

    phi_t = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size)

    z, p, logits = kan1(phi_t, tau=1.0)
    regime_ids = p.argmax(dim=-1)
    y_hat, forecast_per_regime, phi_per_regime = kan2(phi_t, p)

    print("--- Paper-verified terms only (default) ---")
    terms = composite_loss(y_hat, y_true, z, p, regime_ids, kan1, kan2)
    for k, v in terms.items():
        print(f"{k:>16s}: {v.item():.4f}")

    terms["total"].backward()
    print("\ngrad reached KAN1 spline:", kan1.spline.splines[0].w.grad is not None)
    print("grad reached KAN2 spline:", kan2.splines[0][0].beta.grad is not None)

    print("\n--- With the explicitly-unverified balance term opted in ---")
    terms2 = composite_loss(y_hat, y_true, z, p, regime_ids, kan1, kan2, include_balance=True)
    for k, v in terms2.items():
        print(f"{k:>20s}: {v.item():.4f}")
