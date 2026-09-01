# ---------------------------------------------------------------------------
# Bounded-Rank Attention — Quantum-inspired Quadratic Attention with
# Fourier-Domain Rank Control in Transformer Architectures
#
# This file is part of the research code associated with:
#   Mallick, M.T., Banerjee, S., Mattar, E.A., Bhattacharya, P., Pal, S.,
#   Kumar, A., Turdiev, J., Saha, H.N., & Chakrabarti, A. (2026).
#   "Quantum-inspired Quadratic Attention with Fourier-Domain Rank Control
#    in Transformer Architectures."  Scientific Reports (in press).
#   DOI: 10.1038/s41598-026-60978-w
#
# Original code by the authors above (AI-Lab group under Prof. Amlan
# Chakrabarti, A.K. Choudhury School of IT, University of Calcutta /
# Artificial Intelligence, IIT Kharagpur).
#
# This repository is maintained by Ravindra Mina (25AI60R02), M.Tech (AI),
# IIT Kharagpur, under Prof. Amlan Chakrabarti (MTP-2 supervisor) for
# ongoing extensions of this work.  See ATTRIBUTION.md.
# ---------------------------------------------------------------------------

# bounded_rank_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

class FourierFeatureEncoder(nn.Module):
    """
    Quantum-inspired fixed-dimensional feature encoding.
    Maps input embeddings to a bounded-rank feature space.
    """
    def __init__(self, input_dim: int, encoding_dim: int, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Fixed random projections (not learnable)
        torch.manual_seed(seed)
        self.W = torch.randn(encoding_dim, input_dim) * 0.1  # Scale for stability
        self.b = torch.rand(encoding_dim) * 2 * np.pi
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('weight', self.W)
        self.register_buffer('bias', self.b)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fixed-dimensional feature encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Encoded tensor of shape (batch_size, seq_len, encoding_dim)
        """
        # x: (B, N, d) -> (B, N, r)
        # Compute phi_m(x) = cos(2*pi*m/r * w_m^T x + b_m)
        m = torch.arange(1, self.encoding_dim + 1, device=x.device).float().view(1, 1, -1)
        scaling = (2 * np.pi / self.encoding_dim) * m
        
        # Project input: (B, N, r) * (r, d) -> (B, N, r)
        projection = torch.einsum('bnd,rd->bnr', x, self.weight)
        
        # Apply scaling and bias
        encoded = torch.cos(scaling * projection + self.bias.view(1, 1, -1))
        
        return encoded


class BoundedRankQuadraticAttention(nn.Module):
    """
    Bounded-rank quadratic attention with exact factorization.
    No softmax normalization - enables O(Nrd) complexity.
    """
    def __init__(
        self,
        embed_dim: int,
        encoding_dim: int = 32,
        dropout: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_dim = encoding_dim
        self.scale = 1.0 / math.sqrt(embed_dim)
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Fixed-dimensional feature encoder
        self.feature_encoder = FourierFeatureEncoder(embed_dim, encoding_dim, seed)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with exact factorized attention computation.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            return_attention: If True, return attention weights
        
        Returns:
            output: Contextualized representations
            attention_weights: Optional attention matrix
        """
        B, N, d = x.shape
        r = self.encoding_dim
        
        # Compute Q, K, V projections
        Q = self.W_q(x)  # (B, N, d)
        K = self.W_k(x)  # (B, N, d)
        V = self.W_v(x)  # (B, N, d)
        
        # Apply fixed-dimensional feature encoding
        # Q_enc, K_enc: (B, N, r)
        Q_enc = self.feature_encoder(Q)
        K_enc = self.feature_encoder(K)
        
        # Exact factorization: Z = Phi_Q @ (Phi_K^T @ V)
        # (B, r, N) @ (B, N, d) -> (B, r, d)
        KV = torch.einsum('bnr,bnd->brd', K_enc, V)
        
        # (B, N, r) @ (B, r, d) -> (B, N, d)
        Z = torch.einsum('bnr,brd->bnd', Q_enc, KV)
        
        # Optional: compute attention weights for analysis
        if return_attention:
            # A = Phi_Q @ Phi_K^T: (B, N, r) @ (B, r, N) -> (B, N, N)
            attention_weights = torch.einsum('bnr,bmr->bnm', Q_enc, K_enc)
            # Scale for numerical stability
            attention_weights = attention_weights / math.sqrt(r)
        else:
            attention_weights = None
        
        # Output projection
        output = self.W_o(Z)
        output = self.dropout(output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with bounded-rank quadratic attention.
    """
    def __init__(
        self,
        embed_dim: int,
        encoding_dim: int = 32,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        
        # Bounded-rank attention
        self.attention = BoundedRankQuadraticAttention(
            embed_dim=embed_dim,
            encoding_dim=encoding_dim,
            dropout=dropout,
            seed=seed
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual connection
        attn_out, _ = self.attention(self.norm1(x))
        x = x + attn_out
        
        # FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class BoundedRankVisionTransformer(nn.Module):
    """
    Vision Transformer with bounded-rank quadratic attention.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 768,
        encoding_dim: int = 32,
        depth: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.encoding_dim = encoding_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks with bounded-rank attention
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                encoding_dim=encoding_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                seed=seed + i
            )
            for i in range(depth)
        ])
        
        # Layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, N, d)
        x = self.patch_embed(x)  # (B, d, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, d)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classify using class token
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        logits = self.head(x)
        
        return logits


class StandardVisionTransformer(nn.Module):
    """
    Standard Vision Transformer with softmax attention for comparison.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)
        
        # Standard transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, 0]
        logits = self.head(x)
        
        return logits