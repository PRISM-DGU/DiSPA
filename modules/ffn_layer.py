import torch
import torch.nn as nn

# =============================================================================
# [1] CELL LINE FEATURE ENCODER
# =============================================================================
class CelllineFFN(nn.Module):
    """
    Feed-Forward Network dedicated to Cell-line Gene Expression features.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout_rate=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass for Cell-line FFN.

        Args:
            x: [B, P, G]
        """
        B, P, G = x.shape

        # Flatten Batch and Pathway dimensions to apply FFN in parallel
        # [B, P, G] -> [B * P, G]
        x_flat = x.view(B * P, G)  # [B, P, G] -> [B*P, G]

        # Apply Sequential Network
        out = self.network(x_flat)  # [B*P, output_dim]

        # Restore original Batch and Pathway dimensions
        out = out.view(B, P, -1) # [B, P, output_dim]
        return out

    
# =============================================================================
# [2] DRUG FFN
# =============================================================================
class DrugFFN(nn.Module):
    """
    Feed-Forward Network for Drug embeddings.
    """
    def __init__(self, input_dim=768, output_dim=64, hidden_dim=1024, dropout_rate=0.5):
        super(DrugFFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass for Drug FFN.
        
        Args:
            x: [B, L, output_dim]
        """
        output = self.network(x)  # [B, L, output_dim]
        
        # Apply masking if provided (Zero-out padding tokens)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            output = output * mask
        
        return output