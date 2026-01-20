import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        """
        Computes the Root Mean Square (RMS) and normalizes the input.
        Formula: x * rsqrt(mean(x^2) + eps)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     output = self._norm(x.float()).type_as(x)
    #     if self.weight is not None:
    #         output = output * self.weight
    #     return output
    
    def forward(self, x):
        """
        Forward pass for RMSNorm.
        """
        output = self._norm(x) 
        if self.weight is not None:
            # Ensure the weight dtype matches the input dtype for mixed-precision compatibility
            output = output * self.weight.to(x.dtype)  

        return output

    def extra_repr(self) -> str:
        """String representation for printing the module."""
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    