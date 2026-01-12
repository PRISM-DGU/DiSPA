import torch
import torch.nn as nn
from typing import List, Optional, Union

class DeepRegressorHead(nn.Module):
    """
    입력: [B, input_dim], 출력: [B, out_dim]
    - hidden_dims: 각 층 너비 리스트 (예: [512, 512, 256, 256])
    - dropout: float 또는 각 층의 dropout 리스트
    - norm: str 또는 각 층의 normalization 리스트 ("batch" | "layer" | None)
    - act: str 또는 각 층의 activation 리스트 ("relu" | "gelu" | "silu")
    - residual_every: N>0 이면 N개 층마다 잔차 연결
    - residual_proj: 폭이 다를 때도 잔차를 쓰고 싶으면 True (1x1 Linear로 투영)
    - last_dropout: 마지막 블록 뒤 드롭아웃 적용 여부
    - final_norm: 출력층 전에 한 번 더 정규화 (회귀에서 과도하면 False 권장)
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
        
        # 리스트가 아닌 경우 리스트로 변환 (하위 호환성)
        if not isinstance(dropout, list):
            dropout = [dropout] * len(hidden_dims)
        if not isinstance(norm, list):
            norm = [norm] * len(hidden_dims)
        if not isinstance(act, list):
            act = [act] * len(hidden_dims)
        
        # 길이 검증
        assert len(dropout) == len(hidden_dims), f"dropout list length {len(dropout)} != hidden_dims length {len(hidden_dims)}"
        assert len(norm) == len(hidden_dims), f"norm list length {len(norm)} != hidden_dims length {len(hidden_dims)}"
        assert len(act) == len(hidden_dims), f"act list length {len(act)} != hidden_dims length {len(hidden_dims)}"

        # --- Activation factory ---
        def make_activation(act_type):
            if act_type == "relu":
                return nn.ReLU
            elif act_type == "silu":
                return nn.SiLU
            else:
                return nn.GELU

        # --- Norm factory ---
        def make_norm(norm_type, d):
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

        for i in range(1, len(dims)):
            in_d, out_d = dims[i-1], dims[i]
            layer_idx = i - 1  # hidden layer index

            # 레이어별 설정
            layer_act = act[layer_idx]
            layer_norm = norm[layer_idx]
            layer_dropout = dropout[layer_idx]
            
            # Activation 생성
            Activation = make_activation(layer_act)

            block = nn.Sequential(
                nn.Linear(in_d, out_d),
                make_norm(layer_norm, out_d),
                Activation(),
                nn.Dropout(layer_dropout if (last_dropout or i < len(dims)-1) else 0.0),
            )
            self.blocks.append(block)

            # 준비: 폭이 다른 구간에 대한 residual projection (N개 층마다)
            if self.residual_every and (i % self.residual_every == 0) and (in_d != out_d) and self.residual_proj:
                self.residual_adapters[str(i)] = nn.Linear(in_d, out_d, bias=False)

        # Final normalization
        if final_norm:
            # 마지막 레이어의 norm 타입 사용
            final_norm_type = norm[-1] if isinstance(norm, list) else norm
            self.final_norm = make_norm(final_norm_type, dims[-1])
        else:
            self.final_norm = nn.Identity()
            
        self.out = nn.Linear(dims[-1], out_dim)

        # --- Initialization ---
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
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
