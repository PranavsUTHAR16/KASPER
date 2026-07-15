import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (Equation 18 from the KASPER paper).
    Encourages latent embeddings of samples belonging to the same regime 
    to cluster together.
    
    L_contrastive = E[ ||z_i - z_j||^2 * y_ij ]
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, probs):
        """
        Args:
            embeddings (torch.Tensor): Latent embeddings, shape (B, D).
            probs (torch.Tensor): Regime routing probabilities, shape (B, R).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Calculate hard regime assignments
        assignments = torch.argmax(probs, dim=-1)  # Shape: (B,)

        # 2. Generate binary mask y_ij indicating if samples i and j share the same regime
        # Shape: (B, B)
        y_ij = (assignments.unsqueeze(0) == assignments.unsqueeze(1)).float()

        # 3. Vectorized pairwise squared Euclidean distance calculation:
        # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 * z_i^T * z_j
        norms = torch.sum(embeddings ** 2, dim=-1)                  # Shape: (B,)
        dot_products = torch.matmul(embeddings, embeddings.t())       # Shape: (B, B)
        
        # Broadcasting norms over rows and columns:
        distances_sq = norms.unsqueeze(1) + norms.unsqueeze(0) - 2 * dot_products
        
        # Clamp to prevent negative values due to floating point precision errors
        distances_sq = torch.clamp(distances_sq, min=0.0)

        # 4. Multiply by mask and return the mean
        loss = torch.mean(distances_sq * y_ij)
        return loss


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Loss (Equation 19 from the KASPER paper).
    Encourages the regime projection directions to remain orthogonal, 
    preventing degenerate routing where multiple indices cover the same space.
    
    L_orth = ||W * W^T - I||_F^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, W):
        """
        Args:
            W (torch.Tensor): Projection weight matrix, shape (num_regimes, hidden_dim).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Compute W * W^T
        # shape: (num_regimes, num_regimes)
        prod = torch.matmul(W, W.t())

        # 2. Subtract Identity matrix I
        I = torch.eye(prod.size(0), device=W.device)
        diff = prod - I

        # 3. Return the squared Frobenius norm (sum of squared elements)
        loss = torch.sum(diff ** 2)
        return loss


class KasperCompositeLoss(nn.Module):
    """
    Composite Loss Function for the KASPER framework (Equation 23).
    Combines:
    - Huber Loss for prediction accuracy
    - L1 Sparsity regularization on model parameters
    - Contrastive Loss for regime embedding clustering
    - Orthogonality Loss on regime projection weights
    - Regime Balance Loss to prevent routing collapse
    """

    def __init__(self, lambda_s=0.001, lambda_c=0.01, lambda_o=0.01, lambda_b=0.05):
        """
        Args:
            lambda_s (float): Weight for L1 parameter sparsity penalty.
            lambda_c (float): Weight for contrastive separation penalty.
            lambda_o (float): Weight for regime projection orthogonality penalty.
            lambda_b (float): Weight for regime probability balancing penalty.
        """
        super().__init__()
        self.lambda_s = lambda_s
        self.lambda_c = lambda_c
        self.lambda_o = lambda_o
        self.lambda_b = lambda_b

        # Instantiate sub-loss criteria
        self.huber_loss = nn.HuberLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.orthogonality_loss = OrthogonalityLoss()

    def forward(self, predictions, targets, embeddings, probs, model):
        """
        Args:
            predictions (torch.Tensor): Output forecast values, shape (B, 1).
            targets (torch.Tensor): Target values, shape (B, 1).
            embeddings (torch.Tensor): Stable latent representation, shape (B, D).
            probs (torch.Tensor): Regime probabilities, shape (B, R).
            model (nn.Module): The KASPER master model.
            
        Returns:
            total_loss (torch.Tensor): Combined scalar loss.
            loss_dict (dict): Dictionary of individual loss components.
        """
        # 1. Huber Loss (for forecasting accuracy)
        l_huber = self.huber_loss(predictions, targets)

        # 2. Sparsity Loss (L1 norm on Layer 2 parameters only)
        l_sparsity = sum(torch.abs(p).sum() for p in model.layer2.parameters())

        # 3. Contrastive Loss (regime clustering)
        l_contrastive = self.contrastive_loss(embeddings, probs)

        # 4. Orthogonality Loss (regime boundary separation)
        # Apply specifically to the final regime classification projection weights
        l_orth = self.orthogonality_loss(model.layer1.regime_proj.weight)

        # 5. Regime Balance Loss (prevents regime routing collapse)
        # Calculate the mean probability assigned to each regime across the batch
        mean_probs = probs.mean(dim=0)  # Shape: [num_regimes]
        # Target uniform distribution
        target_probs = torch.ones_like(mean_probs) / probs.size(-1)
        # Penalize deviation from uniform distribution using soft probabilities
        l_balance = torch.nn.functional.mse_loss(mean_probs, target_probs)

        # 6. Sum components using their respective weights
        total_loss = (
            l_huber + 
            self.lambda_s * l_sparsity + 
            self.lambda_c * l_contrastive + 
            self.lambda_o * l_orth + 
            self.lambda_b * l_balance
        )

        loss_dict = {
            "huber": l_huber.item(),
            "sparsity": l_sparsity.item(),
            "contrastive": l_contrastive.item(),
            "orth": l_orth.item(),
            "balance": l_balance.item()
        }
            
        return total_loss, loss_dict

