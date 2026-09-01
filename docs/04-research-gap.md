# 4. Research gap — my reading

*This is Ravindra's synthesis of the five reference papers into an open research direction. It is not endorsed by the authors of P1–P5; if it turns out to be the direction MTP-2 pursues, credit for the direction goes back to Prof. Amlan Chakrabarti and the group.*

## 4.1 What the group has demonstrated

Reading P1 → P5 as one trajectory (see [`01-research-context.md`](01-research-context.md)):

- **Application** is nailed down: pest/disease detection in Indian legumes (mung bean, mustard) and Indian tea, with a strong bias toward smallholder, edge-deployable systems.
- **Hardware acceleration path** exists: MobileNetV3 on Xilinx DPU (P2), and a custom dual-mode edge accelerator for CNN + Transformer (P3).
- **Multimodal fusion** is established: RGB + environmental sensors (P3), and RGB + chlorophyll + VOC + microclimate (P5).
- **Learning-primitive advance** exists: bounded-rank quadratic attention with Fourier-domain rank control (P4), evaluated on the same tea-pest dataset as P3.
- **Deployment/optimisation story** is federated (P5), edge-oriented (P3, P2), privacy-preserving (P5), robust to noise (P4, P3).

## 4.2 What is not yet joined up

Three specific gaps sit between the papers:

### Gap 1 — The bounded-rank primitive has not been dropped into the deployed multimodal pipeline

P4 evaluates bounded-rank attention on **single-modality** Tea Pest images and on generic vision benchmarks (CIFAR-100, ImageNet-1K, LRA). The multimodal fusion of P3 and the physiology-guided cross-modal attention of P5 are both currently **full-rank softmax**. Whether a bounded-rank substitution:

- preserves the 15% error reduction P3 gets from multimodal fusion,
- preserves the 11–16% F1 improvement P5 gets from physiology-guided fusion,
- and delivers the 2.6× latency / 2× memory saving that P4 shows in isolation,

is **an open experimental question**. This is a well-defined, tractable, and directly useful contribution.

### Gap 2 — P4's measured speedup is bandwidth-bound, not compute-bound

P4 reports `O(N r d)` theoretical complexity but a measured wall-clock speedup of only 2.6×, "restricted by memory bandwidth and kernel launch overhead". That is precisely the regime the custom dual-mode edge accelerator of P3 was designed for. **Nobody has measured bounded-rank attention on the P3 accelerator.** If the theoretical linear-in-N gain is realised on that hardware, the P3 25 ms / 0.12 J numbers should drop.

### Gap 3 — Federated learning cost is dominated by parameter transfer

P5's federated setup is within ~2% F1 of centralised training, but the bandwidth cost per round scales linearly with model size. Bounded-rank attention shrinks the Transformer half of a hybrid CNN–Transformer backbone (P3), which should compress each FL round roughly proportionally. **Nobody has measured this either.**

## 4.3 A single, well-defined MTP-2 question

> **Does a bounded-rank + Fourier-domain attention primitive (P4) preserve the accuracy and robustness of the multimodal, physiology-guided, federated tea-pest pipeline of P3 + P5, while delivering the compute/memory/energy gains it shows on single-modality benchmarks — and does the gain survive deployment on the custom edge accelerator of P3?**

Concretely, the answer to that question requires three experiments (see [`05-mtp2-roadmap.md`](05-mtp2-roadmap.md)):

1. **E1 — Modality-agnostic drop-in.** Replace the softmax attention in the P3 hybrid CNN–Transformer backbone with `BoundedRankQuadraticAttention`; retrain on the 1,520/7,600-sample tea-pest dataset; measure accuracy under P3's perturbation battery, plus latency and memory on a Jetson Nano (as a proxy for the custom accelerator).
2. **E2 — Physiology-aware bounded-rank cross-modal attention.** Extend `FourierFeatureEncoder` with an optional physiological conditioning input (chlorophyll index, VOC signature) and use it inside P5's cross-modal attention block. Measure F1 vs. P5's baseline.
3. **E3 — FL round-cost reduction.** Simulate P5's federated setup on 4–8 nodes with non-IID splits, compare per-round transfer cost between full-rank softmax and bounded-rank attention. Target: linear reduction in transfer bytes per round.

## 4.4 What is *not* in scope

- **Theoretical convergence proofs** for the physiologically-conditioned encoder — interesting, but a distraction for MTP-2 unless E2 works empirically first.
- **Fabricating new custom hardware** — beyond MTP-2. Use Jetson Nano + FPGA emulation as proxies.
- **Replacing the Fourier transform** with wavelet / Walsh–Hadamard — flagged in [`P4-...md#open-questions`](02-paper-notes/P4-quantum-fourier-attention-mallick-2026.md) but out-of-scope now.
