# 7. Reproducibility

## 7.1 Seeds

Every experiment script sets seeds explicitly via `set_seed(seed)` in `src/main_experiment.py`. The paper reports results averaged over 5 seeds: `[42, 123, 456, 789, 101112]` (see `main_experiment.py` config). For a single-run smoke test use `seed=42`.

`set_seed` covers `random`, `numpy`, `torch`, `torch.cuda`, and forces `torch.backends.cudnn.deterministic=True` / `benchmark=False` — expect a small throughput cost.

## 7.2 Hardware assumptions

The paper's reported numbers were produced on the group's hardware. Broadly-comparable settings:

| Component | Minimum      | Recommended                 |
|-----------|--------------|-----------------------------|
| GPU       | 1 × V100 16 GB | 1 × A100 40 GB or 1 × H100 80 GB |
| RAM       | 32 GB        | 64 GB+                      |
| Disk      | 50 GB free   | NVMe SSD                    |
| CUDA      | 11.8         | 12.1                        |
| Python    | 3.10         | 3.11                        |
| PyTorch   | 2.1          | 2.3+                        |

Jetson Nano / edge measurements (Phase 1 of the MTP-2 roadmap) use PyTorch's ARM build.

## 7.3 Expected runtimes (rough guide)

Numbers below are order-of-magnitude on a single A100. Scale up/down proportionally to your GPU.

| Experiment    | Command                                                  | Approx. runtime |
|---------------|----------------------------------------------------------|-----------------|
| Tea Pest      | `python scripts/run_all_experiments.py --experiment tea_pest --epochs 100` | ~4 h            |
| Ablation (4-way) | `python scripts/run_all_experiments.py --experiment ablation --epochs 100` | ~16 h        |
| CIFAR-100     | `python scripts/run_all_experiments.py --experiment cifar100 --epochs 100` | ~8 h            |
| ImageNet-1K   | `python scripts/run_all_experiments.py --experiment imagenet --epochs 100` | ~5–7 days       |
| LRA           | `python scripts/run_all_experiments.py --experiment lra`                  | ~1–2 days        |

## 7.4 Result artefacts

`run_all_experiments.py` writes `results/all_results_YYYYMMDD_HHMMSS.json`. Do **not** commit the `results/` directory (it is git-ignored except for `.gitkeep`).

## 7.5 If your numbers differ from the paper's

Expected sources of divergence, in order of likelihood:

1. **Dataset version.** The Tea Pest dataset comes from the group; make sure you have the same 1,520-image split that P4 used.
2. **Encoder dimension `r`.** The default is 32; the paper's tables report a sweep. Match the sweep in `src/ablation.py` before comparing.
3. **cuDNN benchmarking.** With `benchmark=True`, throughput is faster but numerics shift slightly across runs. `set_seed` disables it deliberately.
4. **Mixed precision.** The codebase runs in fp32 by default; enabling AMP will change results by a small but measurable margin.
5. **Batch size effects on BatchNorm.** The ViTs here use `LayerNorm` (not BN), so this is a minor factor — but the CNN branch of an E1-style hybrid backbone will be BN-sensitive.

If you cannot close the gap, open a GitHub issue with your hardware, seed, config, and the divergent numbers.
