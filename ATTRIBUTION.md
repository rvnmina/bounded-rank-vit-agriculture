# Attribution

This repository packages research and code that was developed by the **AI-Lab group under Prof. Amlan Chakrabarti** (A. K. Choudhury School of Information Technology, University of Calcutta, and Artificial Intelligence, IIT Kharagpur). It is published here with the explicit approval of the guide, for the purpose of supporting the MTP-2 project of Ravindra Mina (25AI60R02, M.Tech AI, IIT Kharagpur), and — subsequently — for continued research extensions built on top of the group's contribution.

## 1. Who created what

### The method and paper (P4)
> Mallick, M.T., Banerjee, S., Mattar, E.A., Bhattacharya, P., Pal, S., Kumar, A., Turdiev, J., Saha, H.N., & Chakrabarti, A. (2026). *Quantum-inspired Quadratic Attention with Fourier-Domain Rank Control in Transformer Architectures.* **Scientific Reports** (in press). DOI: 10.1038/s41598-026-60978-w.

**All algorithmic contributions** (bounded-rank quadratic attention, Fourier-domain formulation, rank-control theorems, ablation design, experimental protocol) are the authors' work.

### The reference-paper corpus (P1, P2, P3, P5)
Summarised in `docs/02-paper-notes/`. Author lists preserved verbatim there.

### The original code base (`src/*.py`, `scripts/run_all_experiments.py`)
Written by the authors of P4 and shared with Ravindra Mina during the Summer Research Internship (June 2026 – onward) for study and extension. The files as delivered are checked in unmodified except for:

1. an attribution header prepended to each file (see `_header` block at the top of every `.py`),
2. renaming of files from spaced titles (`Core Model Implementation.py`) to snake_case module names (`core_model.py`) for import cleanliness,
3. addition of `src/__init__.py` for package re-exports.

No functional or algorithmic change has been made to the original code as part of this initial commit. Any subsequent modification (from `git log` after the first commit) is Ravindra Mina's own work as MTP-2 extension, will be clearly documented in `docs/05-mtp2-roadmap.md`, and any research-worthy contribution arising from that work will be attributed to the full group per convention.

### This repository (README, docs, packaging, CITATION, license placeholder)
Assembled and written by **Ravindra Mina** with the guide's approval. The paper-note summaries are Ravindra's reading of the papers; any inaccuracy in those notes is Ravindra's, not the authors'.

## 2. Mentors and collaborators for MTP-2 continuation

- **Prof. Amlan Chakrabarti** — MTP-2 supervisor, IIT Kharagpur / University of Calcutta.
- **MD Tausif Mallick** (PhD) — collaborator; first author of the reference paper corpus.
- **Prof. Himadri Nath Saha** — Surendranath College / University of Calcutta; co-author of P1–P5.

Full team page: [`docs/06-team.md`](docs/06-team.md).

## 3. If you spot a mis-attribution

Please open a GitHub issue in this repository and tag @<maintainer-github-username>; the file will be corrected within the same day and the guide will be notified.
