# 5. MTP-2 roadmap

**Author:** Ravindra Mina (25AI60R02), M.Tech AI, IIT Kharagpur.
**Supervisor:** Prof. Amlan Chakrabarti.
**Collaborators:** MD Tausif Mallick (PhD), Prof. Himadri Nath Saha.
**Continuation of:** Summer research internship, end June 2026 → onward.
**Working hypothesis:** see [`04-research-gap.md`](04-research-gap.md).

## 5.1 Phase 0 — Foundation (done during summer internship)

- ✅ Read and summarised P1–P5 (see [`02-paper-notes/`](02-paper-notes/)).
- ✅ Received and studied the reference implementation of P4 (this repo's `src/`).
- ✅ Wrote up the group programme, the method, and the gap synthesis (`docs/01–04`).
- ✅ Set up this repository with proper attribution.
- ⬜ Reproduce Table X of P4 on Tea Pest — accuracy 84.5%, latency 5.6 ms — on my own hardware (Kanad / IIT KGP DGX). *In progress; expected week 1 of MTP-2.*

## 5.2 Phase 1 — E1: modality-agnostic drop-in (Aug–Sep 2026)

**Question.** Does replacing softmax attention with bounded-rank attention in P3's hybrid CNN–Transformer backbone preserve the 85.5% accuracy under combined perturbations?

**Deliverables.**
1. New file `src/hybrid_backbone.py` implementing P3's hybrid CNN–Transformer with a pluggable attention module (softmax / linear / bounded-rank / Fourier bounded-rank).
2. Retraining on the Tea Pest dataset with the P3 augmentation and perturbation protocol.
3. Latency + memory measurement on Jetson Nano (as a proxy for P3's custom accelerator).
4. Results table + analysis in `docs/experiments/E1.md`.

**Success criterion.** Bounded-rank variant within 1% accuracy of P3 baseline, at ≥ 1.5× lower latency on Jetson Nano.

**Risk.** The CNN branch, not the Transformer branch, might dominate compute on this dataset (patches are small). If so, E1 will show *no* significant latency win — the honest reporting move is to say so and pivot to E2.

## 5.3 Phase 2 — E2: physiology-aware bounded-rank cross-modal attention (Sep–Nov 2026)

**Question.** Can the fixed feature encoder Φ be *conditioned* on a physiological signal (chlorophyll index, VOC, microclimate) without breaking the rank bound, and does this recover P5's 11–16% F1 improvement over unimodal baselines?

**Deliverables.**
1. `src/physiological_encoder.py` — a Φ variant that takes a per-sample physiological vector `c ∈ ℝ^p` and modulates the frozen `{w_m, b_m}` in a rank-preserving way (candidate: `w_m ← w_m ⊙ f(c)` with `f` a small MLP whose output is a diagonal gate).
2. `src/multimodal_attention.py` — cross-modal attention layer that uses this encoder as a drop-in for P5's softmax cross-attention.
3. Training on a multimodal tea dataset (requires data from the group — coordinate with Tausif).
4. Results table + analysis in `docs/experiments/E2.md`.

**Success criterion.** F1 within 2% of P5's centralised baseline, with the bounded-rank guarantee empirically verified (measure operator rank of the trained attention block).

**Risk.** Conditioning could break the rank bound; the constraint has to be checked mathematically before running experiments. That check is the first deliverable of Phase 2.

## 5.4 Phase 3 — E3: federated round-cost measurement (Nov–Dec 2026)

**Question.** Does bounded-rank attention proportionally reduce federated-learning per-round transfer bytes without disproportionately hurting non-IID convergence?

**Deliverables.**
1. `src/federated_sim.py` — a Flower or FedML-based simulation of 4–8 plantation nodes with non-IID splits of the Tea Pest dataset.
2. Per-round transfer-size comparison: softmax vs. bounded-rank attention.
3. Convergence-curve comparison at matched compute budget.
4. `docs/experiments/E3.md`.

**Success criterion.** Bounded-rank FL round is ≥ 30% smaller in bytes than softmax FL round, at equal or better convergence within 2% F1.

## 5.5 Phase 4 — Write-up (Dec 2026 – Jan 2027)

- MTP-2 report to IIT KGP.
- If E2 or E3 produces publishable results, coordinate with the group on a follow-up paper naming Prof. Amlan Chakrabarti and MD Tausif Mallick as senior authors.
- Update this repository's `CITATION.cff` with the follow-up paper.

## 5.6 Non-goals (explicit)

- New hardware.
- Replacing the Fourier transform with wavelets.
- Theoretical convergence proofs before empirical validation.
- Public release of the raw Tea Pest images (they belong to the group and the plantations that provided them).

## 5.7 Weekly cadence

Meet Prof. Amlan Chakrabarti weekly; sync with Tausif and Prof. Saha bi-weekly; push notes into `docs/experiments/` after each meeting so the paper trail is legible.
