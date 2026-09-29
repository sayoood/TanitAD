# HiCAP paper (LaTeX)

**HiCAP — Hierarchical Contrastive Action Priors on a Frozen Vision–Language Backbone for Elastic, Imagination-Aware Driving.**
Design and pre-registration paper, working draft 2026-09-29. **No HiCAP model has been trained.**

## Build
```
cd Paper/HiCAP
latexmk -pdf -interaction=nonstopmode hicap.tex     # pdflatex + bibtex; needs texlive (latex-extra, science, pictures)
```
Files: `hicap.tex` (main) · `sec_intro/related/problem/method/imag/elastic/theory/eff/prelim/protocol/diff/limits.tex` · `fig_arch.tex` (TikZ) · `app_vocab.tex` · `refs.bib`.

## Evidence convention (programme standard)
Every number carries a tag: `[M]` measured by us with an artifact in the repo · `[P]` published, cited · `[I]` inherited from a programme document, not re-verified · `[E]` closed-form estimate (no timing) · `[H]` hypothesis.

## Companion documents
- Pre-registration: `Project Steering/PREREG_HICAP.md` (extends `PREREG_REFF_CONTRASTIVE_SELECTOR.md`).
- Research streams and measurement scripts: `TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/`.

## Known gaps
- Bibliography: arXiv is egress-blocked in the build environment; ids were confirmed from earlier repo citation checks and web search, not from the arXiv pages. Re-check `refs.bib` with unrestricted access before submission.
- §9.4 (toy worlds for hierarchical retrieval, imagination, camera gating) is filled in from the R-B / R-C / R-D streams.
