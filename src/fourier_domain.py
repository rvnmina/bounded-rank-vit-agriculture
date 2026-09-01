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

# fourier_attention.py
import torch
import torch.nn as nn
import torch.fft
import math

class FourierBoundedRankAttention(nn.Module):
    """
    Fourier-domain representation of bounded-rank quadratic attention.
    Preserves rank, singular values, and condition number exactly.
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
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Feature encoder (same as spatial)
        self.feature_encoder = FourierFeatureEncoder(embed_dim, encoding_dim, seed)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, d = x.shape
        r = self.encoding_dim
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Encode features
        Q_enc = self.feature_encoder(Q)  # (B, N, r)
        K_enc = self.feature_encoder(K)  # (B, N, r)
        
        # Apply unitary Fourier transform along feature dimension
        # F: R^r -> C^r (unitary normalization)
        Q_hat = torch.fft.fft(Q_enc, dim=-1, norm='ortho')  # (B, N, r)
        K_hat = torch.fft.fft(K_enc, dim=-1, norm='ortho')  # (B, N, r)
        
        # Attention in Fourier domain: A = F^{-1}(Q_hat @ K_hat^*)
        # K_hat^*: complex conjugate
        K_hat_conj = torch.conj(K_hat)
        
        # Factorized computation in Fourier domain
        # (B, r, N) @ (B, N, d) -> (B, r, d)
        KV_hat = torch.einsum('bnr,bnd->brd', K_hat_conj, V)
        
        # (B, N, r) @ (B, r, d) -> (B, N, d)
        Z_hat = torch.einsum('bnr,brd->bnd', Q_hat, KV_hat)
        
        # Inverse Fourier transform back to spatial domain
        Z = torch.fft.ifft(Z_hat, dim=-1, norm='ortho').real
        
        # Optional: compute attention weights
        if return_attention:
            # A = F^{-1}(Q_hat @ K_hat^*) 
            A_hat = torch.einsum('bnr,bmr->bnm', Q_hat, K_hat_conj)
            A = torch.fft.ifft(A_hat, dim=-1, norm='ortho').real
            attention_weights = A / math.sqrt(r)
        else:
            attention_weights = None
        
        # Output projection
        output = self.W_o(Z)
        output = self.dropout(output)
        
        return output, attention_weights