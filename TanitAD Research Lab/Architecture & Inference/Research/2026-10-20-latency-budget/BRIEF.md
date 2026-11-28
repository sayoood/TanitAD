# BRIEF: LATENCY-BUDGET, a decision memo for the PI (Master Mind, 2026-10-20 ~08:45Z)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- Work dir `C:/Users/Admin/par18/latbudget/`. No GPU, no pod, no Thor access; CPU only, reading and writing.
- Never commit or push.

## Why
`BACKLOG LATENCY-BUDGET` (OPEN ARCHITECTURE DECISION): no absolute planning-latency budget exists for the deployed stack (`MM_RULING_C2AMEND.md` DQ4). So no search or planner can be called deployable, and C2's pass will read "deployability UNMEASURED". C1 just read O1_CONFIRMED (+0.56 PDMS for the CEM search over P2's pick, which costs 14-38 scorer calls per token), so the question is now live.

## Do (≤ 2 pages of memo; a decision for the PI, not a survey)
1. **What we already hold (repo, read from the mirror tip):** the Thor deployment profile (`.../Implementation/incoming/2026-08-02-thor-deployment-profile/THOR_PROFILE.md`), the MPC planner design, `SPEC_P6_PAPER_FAITHFUL.md`'s latency text, the C2 amendment's §7 protocol, the dev-box CPU reference in `raw/2026-10-20-search-thor-latency/`, and any MEASURED Thor trunk/engine ms in `MODEL_REGISTRY.md`. Quote each with its evidence class, artifact path and tier.
2. **What the literature says (PUBLISHED, banked):** the planning cycle of deployed / reference stacks and of the NAVSIM-class planners we compare with (e.g. the reference paper's own inference latency, Hydra-MDP, DiffusionDrive, Apollo / Autoware cycle rates). ⛔ Bank every primary you cite with `python tools/kb_add.py ...` (research banking rule); a number from an aggregator is PUBLISHED-SECONDARY and must be labelled so.
3. **The proposal:** 2-3 budget options (e.g. the cycle rate, the end-to-end budget, and the per-stage share: trunk / planner-scorer / search / margin), each with what it implies for the CEM search (the max scorer calls per cycle at the measured or estimated per-call cost) and for P6. Mark every estimate ESTIMATED with its arithmetic. Recommend one and say what would change the recommendation.
4. Name exactly what must be MEASURED on Thor to close it (the `SEARCH-THOR-LATENCY` instrument covers the scorer and search; say what it does not cover).

## Library mechanics
Run `kb_add.py` inside an EXPORT of the tip under your work dir (`tools/` + `TanitAD Research Lab/Library/`, repo layout), never in E: or D:. In the manifest, list the new PDFs plus `library.json` / `LIBRARY.md` with `"base=<the tip's blob>"` (the files `--reindex` regenerates). Run `kb_add.py --verify` before you hand back.

## Deliverable
- `LATENCY_BUDGET_PROPOSAL.md`, plus a ≤ 6-line `PI_DECISION` block at the top: the question, the options, the recommendation.
- Manifest `latbudget/out/paths.json` (`[repo path, abs src, "raw"]`), repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-latency-budget/`, plus the library entries `kb_add.py` writes (list them).
- `HANDOFF.md` (≤ 400 tokens).

## Hand-back (≤ 200 words)
The options, the recommendation, the measured vs estimated split, and the manifest path.
