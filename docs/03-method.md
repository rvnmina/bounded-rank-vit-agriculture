# 3. Method — bounded-rank quadratic attention

This file walks through the maths of the two attention variants in `src/`, using notation aligned with the code.

## 3.1 Standard scaled dot-product attention (baseline)

Given input `x ∈ ℝ^{B×N×d}` (batch, sequence length, embedding), project to `Q, K, V ∈ ℝ^{B×N×d}`. Attention is

$$
A = \mathrm{softmax}\!\left(\frac{QK^{\!\top}}{\sqrt{d}}\right) \in \mathbb{R}^{B\times N\times N}, \qquad Z = A\,V.
$$

Compute: `O(N² d)`. Memory: `O(N²)` for the attention matrix. Rank of `A`: unbounded up to `min(N, d)`. This is what `StandardVisionTransformer` in `src/core_model.py` uses via `nn.TransformerEncoderLayer`.

## 3.2 Bounded-rank quadratic attention (`BoundedRankQuadraticAttention`)

The idea is to replace the softmax kernel with a **fixed-dimensional feature encoder** `Φ : ℝ^d → ℝ^r` (with `r ≪ N, d`) such that the attention operator becomes

$$
A = \Phi(Q)\,\Phi(K)^{\!\top}, \qquad Z = A\,V = \Phi(Q)\left(\Phi(K)^{\!\top}V\right).
$$

Two things happen simultaneously:

1. **Rank bound.** By construction, `rank(A) ≤ r`.
2. **Exact factorisation.** No approximation: computing `KV = Φ(K)ᵀ V ∈ ℝ^{B×r×d}` first, then `Z = Φ(Q) · KV`, gives complexity `O(N r d)` — **linear in N** when `r` is held constant.

### 3.2.1 The encoder Φ

`FourierFeatureEncoder` (`src/core_model.py`, lines 9–48) uses random Fourier-style features:

$$
\Phi_m(x) = \cos\!\left(\frac{2\pi m}{r}\,w_m^{\!\top} x \;+\; b_m\right), \qquad m = 1, \dots, r,
$$

with `{w_m, b_m}` **frozen at initialisation** (`register_buffer`, not `nn.Parameter`). This is what puts the "quantum-inspired fixed-dimensional feature encoding" of the paper title into the code.

### 3.2.2 The forward pass

`BoundedRankQuadraticAttention.forward` (`src/core_model.py`, lines 86–135):

```python
Q_enc = self.feature_encoder(Q)          # (B, N, r)
K_enc = self.feature_encoder(K)          # (B, N, r)
KV    = torch.einsum('bnr,bnd->brd', K_enc, V)   # (B, r, d)
Z     = torch.einsum('bnr,brd->bnd', Q_enc, KV)  # (B, N, d)
output = self.W_o(Z)
```

The optional `return_attention=True` reconstructs the `(B, N, N)` attention matrix `A = Φ(Q)Φ(K)ᵀ / √r` for analysis, but the training/inference path never materialises it.

## 3.3 Fourier-domain formulation (`FourierBoundedRankAttention`)

Apply an orthonormal FFT along the feature axis of the encoded features:

$$
\hat{Q} = \mathcal{F}(\Phi(Q)), \qquad \hat{K} = \mathcal{F}(\Phi(K)).
$$

Because `F` is unitary, `‖\hat{Q}‖ = ‖Φ(Q)‖`, so **rank, singular values, and condition number of the operator are all preserved exactly**. Attention becomes

$$
\hat{A} = \hat{Q}\,\hat{K}^{*}, \qquad Z = \mathcal{F}^{-1}\!\left(\hat{Q}\,(\hat{K}^{*} V)\right).\;\text{real}
$$

where `K̂*` is the complex conjugate. Implemented in `src/fourier_domain.py` (lines 39–86) with `torch.fft.fft(..., norm='ortho')` and `torch.fft.ifft(..., norm='ortho')`. The `.real` at the end drops the numerically-zero imaginary component.

The Fourier variant is not cheaper than the spatial variant per se — it costs one FFT and one IFFT more — but it exposes a **spectral view** of the interaction, useful for both analysis (which frequencies of the kernel dominate?) and future extensions such as spectral pruning.

## 3.4 Vision Transformer integration

`BoundedRankVisionTransformer` (`src/core_model.py`, lines 185–274) is a standard ViT with:

- Conv-based patch embedding (`Conv2d`, patch size 16).
- Learnable position and CLS-token embeddings.
- `depth = 12` stacked `TransformerBlock`s, each with:
  - `LayerNorm → BoundedRankQuadraticAttention → residual`
  - `LayerNorm → FFN(GELU) → residual`
- Final `LayerNorm` + linear classification head on the CLS token.

`StandardVisionTransformer` is the matched baseline using `nn.TransformerEncoderLayer` for controlled ablations.

## 3.5 Complexity summary

| Variant                          | Compute        | Memory (attn matrix) | Rank of A |
|----------------------------------|----------------|----------------------|-----------|
| Softmax dot-product (baseline)   | `O(N² d)`      | `O(N²)`              | `≤ min(N,d)` |
| Linear attention                 | `O(N d²)`      | `O(N d)`             | `≤ d`     |
| **Bounded-rank (this repo)**     | `O(N r d)`     | `O(N r)`             | **`≤ r`** |
| **Fourier bounded-rank**         | `O(N r d + N r log r)` | `O(N r)`     | **`≤ r`** |

Default `r = 32` in every config in the codebase (`encoding_dim=32`).

## 3.6 What the paper reports on Tea Pest

- Accuracy: **84.5%** (bounded-rank ViT) vs. **82.1%** (softmax ViT), same experimental conditions.
- Attention wall-time / image: **14.8 ms → 5.6 ms** (2.6× measured, bandwidth-bound).
- Peak memory: **820 MB → 430 MB**.
- Robustness to additive noise: −2.9% accuracy vs. −3.1% for baseline (comparable).
