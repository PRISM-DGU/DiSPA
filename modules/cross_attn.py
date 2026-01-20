import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.rms_norm import RMSNorm

# =============================================================================
# [1] HELPER FUNCTION
# =============================================================================
def stable_softmax(x, dim, eps=1e-20):
    """
    Numerically stable Softmax function.
    """
    x = x.to(dtype=torch.float32)
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True) + eps
    return x_exp / x_exp_sum

# =============================================================================
# [2] [View 1] Path2Sub
# =============================================================================
class Path2SubCrossMHA(nn.Module):
    """
    [View 1] Path2Sub

    Pathway:   [B, P, E]
    Substructure: [B, L, E]  (multi-token substructure embeddings)
    Output:    [B, P, E]
    """
    def __init__(self, pathway_embed_dim: int, drug_embed_dim: int,
                 attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.pathway_embed_dim = pathway_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(pathway_embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(drug_embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(drug_embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, pathway_embed_dim, bias=False)

        self.norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                query_mask: torch.Tensor = None, key_mask: torch.Tensor = None):
        """
        Forward pass for Cross-Attention.

        Args:
            query(pathway):  [B, P, E]
            key(substructure): [B, L, E]
        Returns:
            out:          [B, P, E]
            attn_weights: [B, H, P, L]
        """
        B, P, E = query.shape
        _, L, _ = key.shape
        
        # ---------------------------------------------------------------------
        # [Step 1] Linear Projection & Multi-Head Splitting
        # ---------------------------------------------------------------------
        # Q: [B, P, H * D] -> [B, P, H, D] -> [B, H, P, D]
        Q = (self.q_proj(query)
             .view(B, P, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, P, head_dim]
        
        # K, V: [B, L, H * D] -> [B, L, H, D] -> [B, H, L, D]
        K = (self.k_proj(key)
             .view(B, L, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, L, head_dim]
        
        V = (self.v_proj(key)
             .view(B, L, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, L, head_dim]

        # ---------------------------------------------------------------------
        # [Step 2] Scaled Dot-Product Attention
        # ---------------------------------------------------------------------
        Q = Q * self.scaling

        # Scores: [B, H, P, D] @ [B, H, D, L] -> [B, H, P, L]
        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, P, L]

        # Apply Masking (if provided)
        if key_mask is not None:
            key_mask = key_mask.to(Q.device).bool()  # [B, L]
            # Broadcast mask: [B, 1, 1, L] -> Expand to [B, H, P, L]
            key_mask_exp = key_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, P, L)
            scores = scores.masked_fill(~key_mask_exp, -1e20)

        # Softmax: Normalize scores to probabilities
        attn_weights = stable_softmax(scores, dim=-1)  # [B, H, P, L]
        
        # ---------------------------------------------------------------------
        # [Step 3] Aggregation & Output Projection
        # ---------------------------------------------------------------------
        # Weighted Sum: [B, H, P, L] @ [B, H, L, D] -> [B, H, P, D]
        out = torch.matmul(attn_weights, V)  # [B, H, P, head_dim]

        # Apply RMSNorm
        out = self.norm(out)

        # Recombine Heads: [B, H, P, D] -> [B, P, H, D] -> [B, P, H * D]
        out = (out.permute(0, 2, 1, 3)
               .reshape(B, P, self.num_heads * self.head_dim))
        
        # Final Linear Projection -> [B, P, E]
        out = self.out_proj(out).view(B, P, E)

        return out, attn_weights

# =============================================================================
# [3] [View 2] Drug2Path
# =============================================================================
class Drug2PathCrossMHA(nn.Module):
    """
    [View 2] Drug2Path

    Drug:     [B, 1, E]
    Pathway:  [B, P, E]
    Output:   [B, 1, E]
    """
    def __init__(self, drug_embed_dim: int, pathway_embed_dim: int,
                 attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.drug_embed_dim = drug_embed_dim
        self.pathway_embed_dim = pathway_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(drug_embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(pathway_embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(pathway_embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, drug_embed_dim, bias=False)

        self.norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                query_mask: torch.Tensor = None, key_mask: torch.Tensor = None):
        """
        Forward pass for Cross-Attention.

        Args:
            query(drug):    [B, 1, E]
            key(pathway):   [B, P, E]

        Returns:
            out:          [B, 1, E]
            attn_weights: [B, H, 1, P]
        """
        B, _, E = query.shape
        _, P, _ = key.shape
        
        # ---------------------------------------------------------------------
        # [Step 1] Linear Projection & Multi-Head Splitting
        # ---------------------------------------------------------------------
        # Q (Drug): [B, 1, H, D] -> [B, H, 1, D]
        Q = (self.q_proj(query)
             .view(B, 1, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, 1, head_dim]
        
        # K, V (Pathway): [B, P, H, D] -> [B, H, P, D]
        K = (self.k_proj(key)
             .view(B, P, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, P, head_dim]
        V = (self.v_proj(key)
             .view(B, P, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, P, head_dim]

        # ---------------------------------------------------------------------
        # [Step 2] Scaled Dot-Product Attention
        # ---------------------------------------------------------------------
        Q = Q * self.scaling

        # Scores: [B, H, 1, D] @ [B, H, D, P] -> [B, H, 1, P]
        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, 1, P]

        if key_mask is not None:
            key_mask = key_mask.to(Q.device).bool()  # [B, P]
            key_mask_exp = key_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, 1, P)
            scores = scores.masked_fill(~key_mask_exp, -1e20)

        attn_weights = stable_softmax(scores, dim=-1)  # [B, H, 1, P]
        
        # ---------------------------------------------------------------------
        # [Step 3] Aggregation & Output Projection
        # ---------------------------------------------------------------------
        # Weighted Sum: [B, H, 1, P] @ [B, H, P, D] -> [B, H, 1, D]
        out = torch.matmul(attn_weights, V)  # [B, H, 1, head_dim]
        out = self.norm(out)

        # Recombine Heads
        out = (out.permute(0, 2, 1, 3)
               .reshape(B, 1, self.num_heads * self.head_dim))
        
        out = self.out_proj(out).view(B, 1, E)

        return out, attn_weights

