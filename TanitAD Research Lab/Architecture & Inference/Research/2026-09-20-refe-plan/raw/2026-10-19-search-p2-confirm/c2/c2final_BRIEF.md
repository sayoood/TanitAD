# BRIEF: finish C2's code to seal-ready (Master Mind, 2026-10-20 ~04:15)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/c2final/`.
- CPU only. C: has about 0.4 GB free: small files only, pytest basetemp under your work dir, and delete each test's own tmp output (`c2amend/code/conftest.py` shows a pattern). If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit or push. No GPU, no pod, no fixture run.
- Never read a navtest label (`pdms` / `sub` / `valid`) or any B-REST output.

## Start from the LANDED code
Tip `6a0cfae` on `agent/arch-inf-20260803` (mirror `C:/Users/Admin/tanitad-push`; run `git fetch`, then pin the tip once).
- Export `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-navtest-pipe/` and `.../2026-10-19-search-p2-confirm/c2/` into your work dir, keeping the repo layout.
- Read `c2/MM_RULING_C2AMEND.md` (THE RULING, binding), `c2/AMENDMENT_C2_DRAFT.md`, `c2/HANDOFF.md`, `c2/LICENCE_GUARD.md`.
- Baseline: the full navpipe suite is 201 passed.

## Deliverables (each with tests, RED arms, and mutation coverage via `mutation_check.py`)
1. **DQ6 (b):** C2 RE-SCORES every P2 pick through the identical T1 route as the search outputs. The paired delta is same-route by construction.
   - The banked-CSV planted identity stays as a cross-check. A mismatch STOPS the chain with a named marker.
   - Update the cost: the ESTIMATED extra harness time.
2. **DQ7 (b):** the fixture chain writes a fixture record (verdict, the sha256 of every code file it ran, its token-list sha). `--mode c2` refuses unless the record reads IDENTICAL and its code shas equal the sealed pins.
3. **Config selection** (`MM_RULING_C2AMEND.md` ADDITION):
   - C2 accepts exactly C8 (`stepB_config.json`) or O4 (`stepB_config_O4.json`, sha256 `1ebd7196…`).
   - It must equal what `c2_config(...)` returns. That function lives in `C:/Users/Admin/par18/doseamend/code/dose_chain.py:135`, which another agent is finishing now. Import it BY PATH with a sha256 pin filled at seal. Do NOT re-implement the table.
   - When O4 is selected, bind the dose result like C1's: a typed sha256, plus the check that it is the dose read on B-REST.
   - Add a ONE-RUN marker: a second C2 run of any config refuses.
4. **Portability:** `navpipe_chain.py:31-32` default `SPC_STEPB_CODE` / `L1_CODE` to dev-box scratch paths. Default them to the REPO-RELATIVE sibling packages; the env var still overrides. Verify that those packages exist on the tip, and quote the paths.
5. **DQ9:** `seal_c2_amendment.py`, modelled on `C:/Users/Admin/par18/spc/code/seal_spc.py` (or wherever `seal_spc.py` lives; find it). It fills the `@@SEAL@@` pins, writes the sidecar sha256 and `SEALED_AT`, and runs the chain's own seal check.
   - RED arm: a placeholder left unfilled must refuse.
   - Do NOT run it for real; the MM seals.
6. Fold the ruling into `AMENDMENT_C2_DRAFT.md`: retitle it "FINAL DRAFT", and mark §14 "RULED (`MM_RULING_C2AMEND.md` sha256 `fe4d9ce6…`)". The terms must match the ruling exactly.
7. Run the full navpipe suite and the mutation check, and report both counts.
8. Write `c2final/HANDOFF.md` (1.5k tokens or fewer) and `c2final/out/paths.json` (`[repo path, abs src, "raw"]`, same repo dirs).

## Hand-back (250 words or fewer)
- Each deliverable's `file:line`.
- The test and mutation counts.
- Anything in the ruling that the code shows to be wrong or impossible.
- The manifest.
