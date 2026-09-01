# Bounded-Rank Attention

**Quantum-inspired quadratic attention with Fourier-domain rank control in Transformer architectures — applied to agricultural pest/disease vision (Tea, Mung bean, Mustard) and standard vision benchmarks.**

> ⚠️ **Attribution first.** The research direction and original code in this repository were developed by the AI-Lab group under **Prof. Amlan Chakrabarti** (A. K. Choudhury School of Information Technology, University of Calcutta, and Artificial Intelligence, IIT Kharagpur). This repository is maintained by **Ravindra Mina (25AI60R02)**, M.Tech (AI), IIT Kharagpur, under Prof. Chakrabarti's supervision as part of the MTP-2 project, with the explicit approval of the guide. Full attribution: [ATTRIBUTION.md](./ATTRIBUTION.md). Citations: [CITATION.cff](./CITATION.cff).

---

## 1. What this repo is

A PyTorch implementation of the **Bounded-Rank Quadratic Attention** mechanism introduced in

> MD Tausif Mallick, Saptarshi Banerjee, Ebrahim Abdulla Mattar, Pronaya Bhattacharya, Sujan Pal, Avnish Kumar, Jon Turdiev, Himadri Nath Saha & Amlan Chakrabarti (2026). *Quantum-inspired Quadratic Attention with Fourier-Domain Rank Control in Transformer Architectures.* **Scientific Reports** (in press). DOI: [10.1038/s41598-026-60978-w](https://doi.org/10.1038/s41598-026-60978-w).

together with training, ablation and evaluation code for four settings:

| Setting        | Purpose                                                       | Script                       |
|----------------|---------------------------------------------------------------|------------------------------|
| **Tea Pest**   | Main real-world benchmark (5 classes, 1,520 → 7,600 augmented) | `src/main_experiment.py`     |
| **CIFAR-100**  | Small-image classification sanity check                       | `src/cifar100_eval.py`       |
| **ImageNet-1K**| Large-scale image classification                              | `src/imagenet_lra.py`        |
| **LRA**        | Long-range Arena sequence tasks                               | `src/imagenet_lra.py`        |
| **Ablation**   | Softmax vs. linear vs. bounded-rank vs. Fourier variants      | `src/ablation.py`            |

## 2. The idea in one paragraph

Standard softmax attention is `O(N²d)` and its interaction rank is effectively unbounded, which hurts both compute and memory on edge devices — the very setting agricultural IoT pipelines need. This work replaces the softmax kernel with a **fixed-dimensional (rank-r) feature encoder** built from Fourier-style random projections, which enables an **exact factorisation**
`Z = Φ(Q) · (Φ(K)ᵀ V)`  →  compute is `O(Nrd)` and attention rank ≤ r by construction. A **unitary Fourier-domain formulation** (`src/fourier_domain.py`) is mathematically equivalent (preserves rank, singular values, condition number) and exposes spectral structure. Reported gains on Tea Pest: **84.5% vs. 82.1% (ViT baseline), 14.8 ms → 5.6 ms, 820 MB → 430 MB memory.**

Detailed method walk-through: [`docs/03-method.md`](docs/03-method.md).

## 3. Where this fits — the research gap

The gap that the guide's group is filling is the interface between **agricultural pest/disease vision** (mung bean → mustard → tea, papers P1–P3 & P5) and **efficient deep learning on edge/IoT hardware** (SoC/DPU/FPGA, papers P2–P3). Bounded-rank attention (P4, this code) is the **learning primitive** that makes multimodal transformer pipelines like P3 and P5 realistically deployable on the dual-mode edge hardware in P3.

The five reference papers are summarised in [`docs/02-paper-notes/`](docs/02-paper-notes/); the synthesis and my own reading of the open gap is in [`docs/04-research-gap.md`](docs/04-research-gap.md); the MTP-2 plan that grows out of it is in [`docs/05-mtp2-roadmap.md`](docs/05-mtp2-roadmap.md).

## 4. Repository layout

```
bounded-rank-vit-agriculture/
├── README.md                     ← you are here
├── ATTRIBUTION.md                ← who did what
├── CITATION.cff                  ← how to cite the paper this repo implements
├── LICENSE                       ← placeholder (pending advisor decision)
├── requirements.txt
├── .gitignore
├── src/                          ← library code
│   ├── core_model.py             ← FourierFeatureEncoder, BoundedRankQuadraticAttention, ViT
│   ├── fourier_domain.py         ← unitary Fourier-domain variant
│   ├── training.py               ← TeaPestDataset, transforms, training loop, metrics
│   ├── ablation.py               ← 4-way ablation (softmax / linear / bounded-rank / Fourier)
│   ├── main_experiment.py        ← Tea Pest full experiment driver
│   ├── cifar100_eval.py          ← CIFAR-100 evaluation
│   ├── imagenet_lra.py           ← ImageNet-1K + LRA evaluation
│   └── __init__.py               ← re-exports
├── scripts/
│   └── run_all_experiments.py    ← CLI entry-point (--experiment tea_pest|ablation|cifar100|imagenet|lra|all)
├── notebooks/                    ← place for exploratory Jupyter notebooks
├── docs/
│   ├── 01-research-context.md    ← why this problem, who works on it
│   ├── 02-paper-notes/           ← one file per reference paper
│   ├── 03-method.md              ← the maths
│   ├── 04-research-gap.md        ← the synthesised gap
│   ├── 05-mtp2-roadmap.md        ← what I plan to add
│   ├── 06-team.md                ← mentors and collaborators
│   └── 07-reproducibility.md     ← seeds, hardware, expected runtimes
├── data/                         ← datasets go here (git-ignored, see data/README.md)
└── results/                      ← run artefacts (git-ignored)
```

## 5. Quick start

```bash
# 1. Clone
git clone https://github.com/<your-github-username>/bounded-rank-vit-agriculture.git
cd bounded-rank-vit-agriculture

# 2. Environment (Python 3.10+, CUDA 11.8+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Sanity: import the model
python -c "from src import BoundedRankVisionTransformer; \
           m = BoundedRankVisionTransformer(num_classes=5); \
           import torch; print(m(torch.randn(2,3,224,224)).shape)"
# expected: torch.Size([2, 5])

# 4. Full Tea Pest experiment (requires dataset — see data/README.md)
python scripts/run_all_experiments.py --experiment tea_pest \
       --data_dir ./data/tea_pest --epochs 100 --batch_size 32
```

Reproducibility notes (seeds, hardware, expected runtimes): [`docs/07-reproducibility.md`](docs/07-reproducibility.md).

## 6. Timeline

Summer research internship — end of June 2026 → ongoing (rolled into MTP-2). See [`docs/05-mtp2-roadmap.md`](docs/05-mtp2-roadmap.md) for the forward plan.

## 7. How to cite

If you use the method, cite the Scientific Reports paper (see [`CITATION.cff`](./CITATION.cff)); if you use this specific repository, additionally cite this repo — a `CITATION.cff` entry for it will be added when the paper appears in print and Amlan sir confirms a DOI.

## 8. Contact

**Ravindra Mina** — 25AI60R02, M.Tech (AI), IIT Kharagpur — repository maintainer.
Supervisor for MTP-2: **Prof. Amlan Chakrabarti** (see [`docs/06-team.md`](docs/06-team.md)).
