# DiffusionDriveV2 — authoritative account and transfer analysis for REF-C

**status: IN PROGRESS — done: prior audit read; V2 paper (17/17 pp incl. suppl.) extracted to `raw/2512.07745v1_text.txt`; V2 repo read IN FULL at `1cd12a1` (`_model_rl.py` 52,504 B, `_model_sel.py` 72,863 B, both agents, both configs, yamls, docs — sha256s in `raw/ddv2_fetched_sha256.txt`, Apache-2.0 copies in `raw/ddv2_src/`); exploration-noise magnitudes computed (`raw/ddv2_noise_magnitudes.{py,json}`); sibling deliverables located (`2026-08-29-rl-posttrain-library`, `2026-08-29-refc-origins-successors`). next: write §1 V1→V2 differences (paper + code, incl. the four paper-absent facts the code settles), §2 results with table numbers, §3 transfer under our constraints extending H-DDA-1..4 / E-DDA-1..5, §4 one-page PI brief; then KNOWLEDGE_BASE lines, register rows, kb_add cited-by + GTRS bank, `--verify`.**

**Stream:** Research Lab · Arch+Inference · **Date:** 2026-09-05 · **Branch:** `agent/arch-inf-20260803` · **Zero GPU.**
**PI question (verbatim):** *"analyze again the successor of the original paper: DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving and also github https://github.com/hustvl/DiffusionDriveV2 to check any differences, benefit from their results, consistency of implementation in the common parts, potential improvements."*

**Starts from what is banked, does not redo it:** the implementation audit at `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refc-vs-diffusiondrive-audit/RESULT.md` (its verdicts are register rows D-REFC-DDAUDIT-1..6; its pre-registered experiments are H-DDA-1..4 / E-DDA-1..5). Our code is NOT re-audited here.

(sections follow — being written)
