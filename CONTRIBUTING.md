# Contributing

This repository is currently maintained solo by **Ravindra Mina** as part of MTP-2 at IIT Kharagpur. External pull requests are welcome for typo fixes, documentation improvements, and reproducibility bug reports. Any change that touches the algorithm or the reference implementation in `src/` will be reviewed with Prof. Amlan Chakrabarti and MD Tausif Mallick before merge.

## Development setup

```bash
git clone https://github.com/<your-github-username>/bounded-rank-vit-agriculture.git
cd bounded-rank-vit-agriculture
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Branch model

- `main` — always green; matches the currently-published state of the repo.
- `mtp2/phase1-e1-*`, `mtp2/phase2-e2-*`, `mtp2/phase3-e3-*` — one branch per experiment in `docs/05-mtp2-roadmap.md`.
- `paper-notes/*` — small doc branches for updating individual paper notes.

## Commit message style

`<area>: <imperative summary>` — e.g. `docs: fix DOI in P4 paper note`, `src: add optional AMP flag to training loop`. Keep the body wrapped at 72 columns.

## Reporting a bug

Open a GitHub issue with:
1. hardware + CUDA + PyTorch version,
2. exact command run,
3. seed used,
4. the divergent number or the stack trace,
5. whether the same command works on `main` at a known-good SHA.
