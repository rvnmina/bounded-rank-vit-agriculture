# P4 — Bounded-rank quadratic attention with Fourier-domain rank control (Mallick et al., 2026) — **this repo's code**

## Bibliographic entry
Mallick, M.T., Banerjee, S., Mattar, E.A., Bhattacharya, P., Pal, S., Kumar, A., Turdiev, J., Saha, H.N., & Chakrabarti, A. (2026). **Quantum-inspired Quadratic Attention with Fourier-Domain Rank Control in Transformer Architectures.** *Scientific Reports* (in press). DOI: [10.1038/s41598-026-60978-w](https://doi.org/10.1038/s41598-026-60978-w). Received 3 Apr 2026; accepted 1 Jul 2026; published online 8 Jul 2026.

## Abstract (verbatim)
> Quadratic attention enhances interaction capacity in Transformer models but leads to rapid growth in computational demands as attention rank increases. This paper presents a bounded-rank quadratic attention mechanism where fixed-dimensional feature encodings determine the interaction space and enforce a strict upper bound on attention rank. A Fourier-domain formulation offers a spectral view of the quadratic kernel via unitary transformation while maintaining exact attention computation. The proposed approach achieves bounded-rank attention with computational complexity that scales linearly with sequence length when the encoding dimension remains constant. Experimental validation on a real-world *Tea* pest image dataset yields **84.5% classification accuracy**, surpassing the **82.1%** achieved by a standard Vision Transformer under equivalent experimental conditions. Attention processing time per image declines from **14.8 ms to 5.6 ms**. Peak memory usage declines from **820 MB to 430 MB**. Although memory bandwidth and kernel launch overhead restrict the measured speedup to 2.6×, the accuracy loss under additive noise reaches **2.9%** compared to **3.1%** for the baseline, demonstrating comparable robustness. These findings indicate that explicit rank control in attention mechanisms can be realized through representational design, offering an efficient bounded-rank alternative to conventional full-rank attention.

## Key contributions
1. **Bounded-rank quadratic attention.** Fixed-dimensional (rank-r) feature encoder Φ constrains the effective attention rank ≤ r by construction.
2. **Exact factorisation** `Z = Φ(Q) · (Φ(K)ᵀ V)` giving `O(N r d)` compute — linear in sequence length when r is constant.
3. **Fourier-domain formulation** — a unitary FFT along the feature axis preserves rank, singular values, and the condition number of the attention operator, and exposes spectral structure of the interaction.
4. **Empirical validation** on a real-world Tea pest image dataset (5 classes), plus CIFAR-100, ImageNet-1K, and LRA (see `src/imagenet_lra.py`).
5. **Ablation** across softmax, linear, bounded-rank, and Fourier-bounded-rank attention (see `src/ablation.py`).

## Method
Given input `x ∈ ℝ^{B×N×d}`:
1. Project to Q, K, V.
2. Apply the fixed Fourier feature encoder Φ (see `FourierFeatureEncoder` in `src/core_model.py`): `Φ_m(x) = cos( 2πm/r · w_mᵀ x + b_m )`, `m = 1..r`, weights `{w_m, b_m}` are frozen at initialisation.
3. Compute `KV = Φ(K)ᵀ V` in `ℝ^{B×r×d}`, then `Z = Φ(Q) · KV` in `ℝ^{B×N×d}`. **No softmax** — the factorisation is exact.
4. (Optional) Move into the Fourier domain: `Q̂ = F(Φ(Q))`, `K̂ = F(Φ(K))`, same factorisation using `K̂*` (complex conjugate), then `Z = F⁻¹(Q̂ · K̂* V)`. This is what `src/fourier_domain.py` implements.

Complexity: `O(N r d)` vs. `O(N² d)` for softmax. Rank of the attention operator ≤ r.

## Results (from paper abstract)
- **Tea Pest classification: 84.5%** vs. **82.1%** for standard ViT under matched conditions.
- **Attention latency: 14.8 ms → 5.6 ms** per image.
- **Peak memory: 820 MB → 430 MB.**
- Measured wall-clock speedup: **2.6×** (bounded by memory bandwidth + kernel launch overhead, not by FLOPs).
- Robustness under additive noise: **2.9% accuracy drop** vs. **3.1%** for the baseline — comparable.

## My take
This is the primitive I want to build MTP-2 around, because (a) it is a real algorithmic advance rather than a system tweak, (b) it has a clean mathematical characterisation (rank / spectrum / condition number all preserved under the Fourier variant), and (c) it slots directly into the group's downstream systems work (P3, P5). The gap I see is: the measured 2.6× speedup is bandwidth-bound, not compute-bound, so **the accelerator work in P2/P3 is exactly what would unlock the theoretical linear-in-N gain**. That connection is the argument of `docs/04-research-gap.md`.

## Open questions
- Can the fixed random features Φ be **learned** without breaking the rank bound? (Probably not without introducing a rank-controlling constraint — could be a theorem worth proving.)
- How does the bounded-rank variant behave on **multimodal cross-attention** — the fusion layer of P3, and the chlorophyll-guided attention of P5?
- **Fourier-domain rank control** currently uses a full FFT along the feature axis; could a wavelet or Walsh–Hadamard replace it with equivalent guarantees and cheaper on-chip cost?
