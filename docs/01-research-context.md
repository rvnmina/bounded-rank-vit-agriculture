# 1. Research context

## 1.1 The problem

Roughly **20% of global crop losses** are attributable to pests and diseases, and in tea (*Camellia sinensis*) — the direct focus of the group's recent work — infestations of aphids, red spider mites, tea mosquito bugs, looper caterpillars, and thrips can reduce leaf quality by ~35% (Bermúdez et al., 2024; Chen et al., 2025, as cited in P3). Detection has historically depended on visual inspection by trained scouts, which does not scale to smallholder plantations.

Automated pest/disease detection is therefore a natural target for computer vision. The **hard constraints** in this specific setting are what make the problem non-trivial:

- **Field conditions.** Low light, high humidity, leaf occlusion, seasonal variation — models trained on controlled imagery degrade sharply in deployment (P3 reports 85.5% under perturbation vs. baseline drops to 77.9%).
- **Edge deployment.** Plantations often lack reliable network connectivity. Inference must run on-device on low-power hardware (P2 uses a Xilinx SoC / DPU; P3 uses custom dual-mode edge hardware; P5 uses federated learning over connectivity-limited nodes).
- **Latency & energy budget.** P3 targets 25 ms per inference at 0.12 J, and 60% lower energy than a Jetson Nano deployment. Transformer attention's `O(N²d)` cost is directly what breaks this budget on standard ViT.
- **Data scarcity + heterogeneity.** 1,520 field-collected images across 5 classes (P3) is realistic; models must generalise from tens-of-thousands, not millions, of examples, and must not leak private plantation data across federations (P5).

## 1.2 The group's programme

Prof. Amlan Chakrabarti's group (with Prof. Himadri Nath Saha and PhD student MD Tausif Mallick as recurring co-authors) has built up the following programme over roughly five years:

| # | Year | Focus                                                                    | Paper (see `02-paper-notes/`)                              |
|---|------|--------------------------------------------------------------------------|------------------------------------------------------------|
| P1 | 2022 | **Learning primitive on a crop** — CNN + transfer learning for mung bean, delivered on Android. | *Deep learning based automated disease detection and pest classification in Indian mung bean* — Multimedia Tools and Applications (Springer). |
| P2 | 2025 | **Hardware acceleration** — MobileNetV3 on a Xilinx DPU/SoC; 24× DPU speedup, 29% throughput, 19% power. | *High-speed system-on-chip based platform for real-time crop disease and pest detection* — Computers and Electrical Engineering (Elsevier). |
| P3 | 2026 | **Edge-optimised multimodal (image + env-sensor) for tea**, hybrid CNN–Transformer, custom dual-mode edge hardware, offline solar-powered inference. | *Transformative Energy-Efficient Edge-Optimised Multimodal DL Framework for Pest Management and Severity Analysis in Tea Plants* — ACM TECS. |
| P4 | 2026 | **This repo — the learning-primitive advance:** bounded-rank quadratic attention with Fourier-domain rank control, breaking `O(N²d)` while preserving rank guarantees. | *Quantum-inspired Quadratic Attention with Fourier-Domain Rank Control in Transformer Architectures* — Scientific Reports (in press). |
| P5 | 2026 | **Physiology-aware federated learning for tea** — RGB + chlorophyll + VOC + microclimate through a chlorophyll-guided cross-modal attention; FL for privacy. | *Physiology-Aware Chlorophyll-Guided Multimodal Transformer With Federated Learning for Early Pest-Risk Forecasting* — IEEE Access. |

The trajectory is legible: **crop-and-architecture (P1) → hardware acceleration of the same class of model (P2) → multimodal + edge co-design (P3) → learning-primitive advance that makes the transformer half of the hybrid cheap (P4) → shift from reactive detection to physiology-aware, privacy-preserving forecasting (P5)**.

## 1.3 Where "this code" sits

The code in `src/` is the reference implementation for **P4**. It is the primitive that is *complementary* to the systems work in P2/P3: instead of only accelerating a fixed model on custom hardware, P4 changes the model itself so that acceleration is easier — attention is factorised, memory is halved, and rank is bounded by construction, so a smaller, cheaper accelerator is enough.

The natural next step (and the one the MTP-2 roadmap in [`05-mtp2-roadmap.md`](05-mtp2-roadmap.md) targets) is to **plug P4's attention into the hybrid CNN–Transformer backbone of P3**, and — if that works — into the physiology-aware multimodal transformer of P5, to test whether bounded-rank attention pays off in a genuinely multimodal, federated, edge-deployed setting rather than only on a single-modality classification benchmark.
