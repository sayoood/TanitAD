# BRIEF: prepare the land of the P7 navtest read (evidence + register + registry rows). Land-ready, NOT landed. Agent: Sonnet. MM, 2026-11-16

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/land/`.
- **Mirror:** `C:/Users/Admin/tanitad-push`. Run `git fetch -q origin agent/arch-inf-20260803` and read with FETCH_HEAD (`core.autocrlf=true`: read blobs with `git show` / `git cat-file blob`).

## Inputs (all MEASURED, 2026-11-16)
- **Read work dir:** `E:/Projects/TanitAD-artifacts/p7_full_read/` (`CHAIN.log`, `chain_stdout.log`, `markers/`, `logs/`, `stage3/`, `families/`, `command_adherence_p7_full.json`, `fam_ci_P6_vs_P7.json`).
- **Stage-2 out:** `F:/TanitAD_p7read_tip/<PK>/raw/2026-10-05-sft-stage2/p7_full/` (`result.json`, `score_status.json`, `seam.log`, `*_families.log`, `score_1.log`), and `.../p7_full_rep/`. Skip the `*.npz` seam files there; they get banked on HF by the MM.
- **The verdict, as the MM read it:** `E:/Projects/TanitAD-artifacts/dagland/LOOP_STATE_P6_2026-11-06.md`, the entry "2026-11-16 22:10Z MM: P7 READ COMPLETE". ⛔ Read only; never write to `dagland/`.

## Deliverable
1. **Raw copies.** Copy the read's text artifacts (json, log, md; no npz, nothing above ~5 MB) into `land/raw/` under `PK/raw/2026-11-16-p7-read/`, with a `MANIFEST.md`: file → source path → sha256.
   - Scan every file for IPv4-looking strings and for `hf_` tokens. Report the counts and mask nothing silently.
2. **`P7_READ_RESULT.md`** (`PK/raw/2026-11-16-p7-read/`). It holds:
   - the endpoint: PDMS and its sub-scores vs P6, vs P2 if stage3 has it, and vs the deployed system, each with its CI, estimator and n;
   - the SPEC_P7 §5 verdict (NOT PROVEN) and the GOAL status;
   - the determinism result;
   - the four families from `fam_ci_P6_vs_P7.json`, primary frame `executed_tracked`, per family and never pooled, listing every separated difference with its CI, plus a count of the others;
   - **STRATEGIC: "not computed: the instrument still stamps PAUSED, but the PI lifted the pause 2026-11-16 ⇒ work item"**;
   - the tier stamp T1;
   - the "answers another draw of logs only" caveat.
3. **Register rows** (as text files for the MM; do not edit the registers):
   - a `GOALS_AND_CLAIMS.md` row `D-REFE-P7-READ-1`, one line in the 5-column table format, to be inserted after the row the MM names;
   - a `Project Steering/MODEL_REGISTRY.md` result block for P7, matching that file's format. Read how P6's navtest result is recorded there. State both `full_set` and the estimator.
4. **`HANDOFF.md`** (≤ 2k tokens): the file list with repo paths (≤ 200 chars each) and any doubt about a number. **Every number cites its artifact path.**

## ⛔ Do not
- Commit, land or push.
- Write into `dagland/`.
- Edit `GOALS_AND_CLAIMS.md`, `MODEL_REGISTRY.md` or any frozen file in place.
- Delete anything outside your own scratch.
- Exceed about 120k context.

**Report:** at most 300 words, plus a manifest.
