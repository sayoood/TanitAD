# BRIEF: C2 readiness, the navtest amendment draft + the pipeline's licence guard (Master Mind, 2026-10-20 ~03:20)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/c2amend/`.
- CPU only. Write SMALL files only: C: has about 0.1 GB free and D: has 0.
  - Put pytest basetemp under your work dir; never write .pt files or bundles.
  - If C: drops below 40 MB, stop and write HANDOFF.md.
- Never commit or push. No GPU, no pod.
- Never read any navtest label, score or search output, or any B-REST output. None exists yet; keep it that way.

## Why
The search on P2's own scorer (`H-SEARCH-P2-1`) is the programme's largest measured lever. Its navtest look (C2) now has a new licence: sealed `C:/Users/Admin/par18/spc/MM_RULING_C2_LICENCE.md`.
- C2 is licensed iff C1 reads O1 or O2, AND every safety point is ≥ −0.25, AND the moved share is ≥ 5 %.
- The EP-ADMISSION gate no longer licenses it.
- C2 still needs its OWN amendment, sealed before any navtest token. Write it now, so it exists before C1 reads.

## Inputs
- `C:/Users/Admin/par18/spc/`: `MM_RULING_SPC.md` (DQ3 = C2's committed bar shape; DQ4; DQ8), `PREREG_SEARCH_P2_CONFIRM.md` (sealed), `MM_RULING_C2_LICENCE.md`.
- The built pipeline in `C:/Users/Admin/par18/navpipe/`: `DESIGN.md`, `HANDOFF.md`, `COST.md`, `code/navpipe_chain.py --mode navtest|c2`, and 119 tests.
  - The fixture has NOT yet read IDENTICAL; it is stopped on RAM/disk.
  - Read `logs/finish*.log` and `raw/finish_status.json` if they exist, and do not re-run anything heavy.

## Deliverables
1. **`c2amend/AMENDMENT_C2_DRAFT.md`** (DRAFT; the MM seals). It states:
   - **The object:** the sha-pinned search, identical to C1's arm, on navtest's tokens (which ones, how many, and the token-list sha).
   - **The endpoint and tier:** the T1 route from DESIGN §3.
   - **The estimator:** paired, the cluster unit, B, the seed.
   - **The decision rule, verbatim from DQ3**, with the numbers. They may not be loosened.
   - **The four families, reported beside it per family** (STRATEGIC PAUSED).
   - **The licence check:** C1's `RESULT_SPC.json` sha and the three licence terms, read by code before any navtest token.
   - **Inference cost per token on the deployment target (Jetson Thor):** how it would be MEASURED (script, inputs, what is timed). Or name it UNMEASURED, with the reason and what it would take.
   - **The fixture precondition:** carve B reads IDENTICAL.
   - **Cost** from `COST.md`.
   - **Every gate with its RED arm.**
   - **Outcomes and their committed next arms:** pass; fail on lower CI; fail on +0.10; fail on seed range; a safety miss.
   - Put the decisions to me as DQ1..DQn with your recommendation; choose none.
2. **The pipeline's licence guard.** `navpipe_chain.py --mode c2` may currently require an EP-ADMISSION admission record. Find out from source and quote `file:line`.
   - Change it to require the C1 licence instead: `RESULT_SPC.json` pinned by sha256, outcome in {O1_CONFIRMED, O2_DIRECTIONAL}, each safety point ≥ −0.25, moved share ≥ 5 %. Plus the sealed C2 amendment sha.
   - Add tests: each failing term must REFUSE, and so must a missing or edited file.
   - Do a mutation check: delete each guard line and the suite must go RED.
   - Run the FULL navpipe suite and report the count.
   - Do NOT touch the s0bank or fixture modes beyond what the change needs.
3. **`c2amend/HANDOFF.md`** (1.5k tokens or fewer) and `c2amend/out/paths.json`: the manifest of `[repo path, abs src, "raw"]`.
   - Repo dir for the draft: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-p2-confirm/c2/`.
   - Repo dir for the code: wherever navpipe's code lands (read `navpipe/out/paths.json` if it exists, else propose one).

## Hand-back (300 words or fewer)
- The DQs.
- The guard change (`file:line` before and after).
- Test counts and mutation results.
- The Thor cost plan.
- The manifest path.
