# Notebooks

Place exploratory Jupyter notebooks here. Suggested files as MTP-2 progresses:

- `01-model-walkthrough.ipynb` — build a small `BoundedRankVisionTransformer` on random data; visualise the attention matrix at `return_attention=True`; confirm rank ≤ r empirically.
- `02-fourier-spectrum-analysis.ipynb` — take a trained model, plot the spectral distribution of `Φ(K)` and the attention operator in the Fourier variant.
- `03-ablation-table.ipynb` — post-process `results/all_results_*.json` from `src/ablation.py` into paper-ready tables.
- `04-hybrid-backbone-E1.ipynb` — Phase-1 experiment scratchpad (see `docs/05-mtp2-roadmap.md`).

Notebook conventions:
- **Do not commit outputs.** Clear all output cells before `git add` (or wire up `nbstripout`).
- Keep any bulky asset (checkpoints, dataset shards) in `results/` or `data/`, both of which are git-ignored.
- If a notebook produces a figure worth keeping, save it under `docs/figures/` as a small PNG.
