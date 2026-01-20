import torch
import torch.nn as nn
from typing import List, Optional, Union

# =============================================================================
# DEEP REGRESSOR HEAD
# =============================================================================
class DeepRegressorHead(nn.Module):
    """
    Input: [B, input_dim] -> Output: [B, out_dim]

    - hidden_dims: List of hidden layer widths (e.g., [512, 512, 256]).
    - dropout: Dropout rate (float) or list per layer.
    - norm: Normalization type ("batch" | "layer" | None). Can be a list.
    - act: Activation function ("relu" | "gelu" | "silu"). Can be a list.
    - residual_every: Apply residual connection every N layers (0 to disable).
    - residual_proj: Use 1x1 projection for dimension mismatch in residuals.
    - last_dropout: Whether to apply dropout after the last hidden block.
    - final_norm: Apply normalization before the final output layer.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: Union[float, List[float]] = 0.2,
        norm: Union[Optional[str], List[Optional[str]]] = "batch",
        act: Union[str, List[str]] = "gelu",
        residual_every: int = 2,
        residual_proj: bool = True,
        out_dim: int = 1,
        last_dropout: bool = False,
        final_norm: bool = False,
    ):
        super().__init__()
        assert len(hidden_dims) >= 1
        
        # [Configuration Standardization]
        # Convert single values to lists to ensure uniform handling per layer.
        if not isinstance(dropout, list):
            dropout = [dropout] * len(hidden_dims)
        if not isinstance(norm, list):
            norm = [norm] * len(hidden_dims)
        if not isinstance(act, list):
            act = [act] * len(hidden_dims)
        
        # [Validation] Ensure configuration lists match the network depth.
        assert len(dropout) == len(hidden_dims), f"dropout list length {len(dropout)} != hidden_dims length {len(hidden_dims)}"
        assert len(norm) == len(hidden_dims), f"norm list length {len(norm)} != hidden_dims length {len(hidden_dims)}"
        assert len(act) == len(hidden_dims), f"act list length {len(act)} != hidden_dims length {len(hidden_dims)}"

        # ---------------------------------------------------------------------
        # [Factory Methods] Component Initialization
        # ---------------------------------------------------------------------
        def make_activation(act_type):
            """Returns the requested activation module."""
            if act_type == "relu":
                return nn.ReLU
            elif act_type == "silu":
                return nn.SiLU
            else:
                return nn.GELU

        def make_norm(norm_type, d):
            """Returns the requested normalization module."""
            if norm_type == "batch":
                return nn.BatchNorm1d(d)
            elif norm_type == "layer":
                return nn.LayerNorm(d)
            else:
                return nn.Identity()

        self.residual_every = max(0, residual_every)
        self.residual_proj = residual_proj

        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList()
        self.residual_adapters = nn.ModuleDict()  # projection for residual when width changes

        # ---------------------------------------------------------------------
        # [Network Construction] Building Layers
        # ---------------------------------------------------------------------
        for i in range(1, len(dims)):
            in_d, out_d = dims[i-1], dims[i]
            layer_idx = i - 1  # hidden layer index

            # Layer-specific settings
            layer_act = act[layer_idx]
            layer_norm = norm[layer_idx]
            layer_dropout = dropout[layer_idx]
            
            
            Activation = make_activation(layer_act)

            # Construct the block: Linear -> Norm -> Activation -> Dropout
            block = nn.Sequential(
                nn.Linear(in_d, out_d),
                make_norm(layer_norm, out_d),
                Activation(),
                nn.Dropout(layer_dropout if (last_dropout or i < len(dims)-1) else 0.0),
            )
            self.blocks.append(block)

            # [Residual Connection Setup]
            if self.residual_every and (i % self.residual_every == 0) and (in_d != out_d) and self.residual_proj:
                self.residual_adapters[str(i)] = nn.Linear(in_d, out_d, bias=False)

        # Final Normalization (Optional)
        if final_norm:
            final_norm_type = norm[-1] if isinstance(norm, list) else norm
            self.final_norm = make_norm(final_norm_type, dims[-1])
        else:
            self.final_norm = nn.Identity()
            
        # Final Output Projection (Latent -> Target)
        self.out = nn.Linear(dims[-1], out_dim)

        # Initialize Weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier Uniform distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass with Residual Connections.
        """
        z = x
        for i, block in enumerate(self.blocks, start=1):
            h = block(z)
            if self.residual_every and (i % self.residual_every == 0):
                if h.shape[-1] == z.shape[-1]:
                    h = h + z
                elif self.residual_proj and str(i) in self.residual_adapters:
                    h = h + self.residual_adapters[str(i)](z)
            z = h
        z = self.final_norm(z)
        y = self.out(z)
        return y
