# How to publish this repository to GitHub

Step-by-step for the first-time push. Do this after Prof. Amlan Chakrabarti has confirmed the public release.

## 0. Pre-publish checklist

- [ ] `ATTRIBUTION.md` — replace `<maintainer-github-username>` with your actual GitHub handle.
- [ ] `README.md` — replace `<your-github-username>` in the clone URL (two occurrences) and in "Contact".
- [ ] `CITATION.cff` — replace `<your-github-username>` in `repository-code`.
- [ ] `CONTRIBUTING.md` — replace `<your-github-username>`.
- [ ] `LICENSE` — leave as-is until the guide chooses a license; the placeholder text is intentional.
- [ ] Double-check no data files, checkpoints, or personal notes have crept into the repo (`git status --ignored`).

## 1. Create the empty GitHub repository

On https://github.com/new:
- **Repository name:** `bounded-rank-vit-agriculture` (primary). You said you liked all three of the candidate names — the other two (`fourier-rank-vit-tea-pest`, `mtp2-quantum-attention`) are set as **topics** on this repo below rather than as separate repositories, because maintaining three copies of the same code drifts fast.
- **Description:** *Quantum-inspired quadratic attention with Fourier-domain rank control in Transformer architectures. Reference implementation of Mallick et al., Sci. Rep. 2026. Maintained for MTP-2 (IIT Kharagpur).*
- **Visibility:** Public.
- **Initialise with:** none (no README, no .gitignore, no LICENSE — this repo already has them).

## 2. Initialise locally and push

From inside the `bounded-rank-vit-agriculture/` folder:

```bash
git init -b main
git add .
git status                                    # sanity-check what is staged
git commit -m "Initial commit — properly attributed reference implementation

Reference implementation of Mallick et al. (2026), Scientific Reports.
Original code by the AI-Lab group under Prof. Amlan Chakrabarti;
this repository is maintained by Ravindra Mina (25AI60R02, IIT KGP)
for MTP-2 with the guide's explicit approval. See ATTRIBUTION.md."

git remote add origin https://github.com/<your-github-username>/bounded-rank-vit-agriculture.git
git push -u origin main
```

## 3. Set repository topics (do this in the GitHub web UI, Settings → Topics)

Add every topic that recruiters or search will look for:

```
transformer, attention-mechanism, vision-transformer, pytorch,
edge-computing, iot, agriculture, pest-detection, tea-plantation,
mtp, iit-kharagpur, research, fourier-transform, low-rank,
mung-bean, mustard, mobilenet, federated-learning,
quantum-inspired
```

The three candidate repo names (`bounded-rank-vit-agriculture`, `fourier-rank-vit-tea-pest`, `mtp2-quantum-attention`) all become discoverable through this topic set.

## 4. Enable GitHub features

- **Settings → General → Features:** Issues on, Discussions optional.
- **Settings → Pages:** off unless you build a docs site later.
- **Settings → Actions → General:** allow all actions (needed for `.github/workflows/ci.yml`).
- **Insights → Community Standards:** should show green for README, LICENSE, CONTRIBUTING, CITATION. (LICENSE will show "custom / not detected" — that is fine, it's a placeholder pending Amlan sir's decision.)

## 5. Pin the repo on your profile

`https://github.com/<your-github-username>` → Customise your pins → pin `bounded-rank-vit-agriculture`. Add a short profile bio mentioning IIT Kharagpur M.Tech AI, MTP under Prof. Amlan Chakrabarti.

## 6. Link from your CV / CDC profile

Where you list this internship on your CV, add the repo URL. In interviews, the specific paragraph a recruiter will read is the **first block of `README.md`** — that block is deliberately written to make the attribution and the scope obvious in ten seconds.

## 7. If you later publish a follow-up paper from MTP-2

Update `CITATION.cff` — add a second `preferred-citation`-style entry, and bump the `version:` field to `0.2.0`.
