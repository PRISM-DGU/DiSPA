import torch.nn as nn

#  CELL LINE FFN
import torch
import torch.nn as nn

class CelllineFFN(nn.Module):
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
        # x: [B, P, G]
        B, P, G = x.shape
        x_flat = x.view(B * P, G)  # [B, P, G] -> [B*P, G]
        out = self.network(x_flat)  # [B*P, output_dim]
        out = out.view(B, P, -1) # [B, P, output_dim]
        return out

    
#  DRUG FFN
class DrugFFN(nn.Module):
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
        output = self.network(x)  # [B, L, output_dim]
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            output = output * mask
        
        return output