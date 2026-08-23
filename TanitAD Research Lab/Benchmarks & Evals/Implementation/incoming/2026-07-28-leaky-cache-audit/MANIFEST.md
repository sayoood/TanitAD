# Deliverable manifest — `leaky-cache-audit`, 2026-07-28

**Agent:** `leaky-cache-audit` · **Repo HEAD at start:** `31134bb` · **Branch:** `agent/benchmarks-eval-20260721`
**Operating rules:** STAGE, NEVER PUSH. Everything below is `git add`ed into the working tree.
**Nothing lives in only one place** — no pod-only artifact, no worktree.

## Artifacts

| # | path | what | lives |
|---|---|---|---|
| 1 | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-leaky-cache-audit/LEAKY_CACHE_AUDIT.md` | the audit: coverage, every hit with claim + verdict, mass quantification, the registry correction, the plain withdrawal statement | `repo:` — staged |
| 2 | `…/2026-07-28-leaky-cache-audit/hits_inventory.json` | machine-readable inventory: A = results scored on it (11 entries), B = ruled out, C = documents/registry rows, D = every `episode_id`-founded disjointness claim flagged, E = the withdrawals | `repo:` — staged |
| 3 | `…/2026-07-28-leaky-cache-audit/leak_mass_by_result_set.json` | per-result-set leaked mass: clips, windows, % , plus the subset that is in the **scoring head's own** train clips, plus each set's content-clean tags | `repo:` — staged |
| 4 | `…/2026-07-28-leaky-cache-audit/search_coverage.json` | the 10 probes, their scope and their return — so *"absence at one location is not absence"* is checkable rather than asserted | `repo:` — staged |
| 5 | `…/2026-07-28-leaky-cache-audit/MANIFEST.md` | this file | `repo:` — staged |
| 6 | `Project Steering/MODEL_REGISTRY.md` | **MODIFIED** — four corrections (§10.1 Leakage bullet, §10.1 provenance line, §10.1 results table, §7 R8) + one inline flag (§2.2) | `repo:` — staged |

## Inputs consumed (read-only, not modified)

| path | role |
|---|---|
| `…/incoming/2026-07-28-parity-leak-check/raw/KNOWNPOSITIVE_f1b378_x_train.json` | the 62 leaked tags + per-pair detail |
| `…/incoming/2026-07-28-parity-leak-check/raw/hashes_fp_val_f1b378f295ae_full.json` | 80 episodes × 7 hash families + `T`, `T_frames` |
| `…/incoming/2026-07-28-parity-leak-check/raw/hashes_fp_val_0c5f_deployed40_full.json` | the deployed clean 40 — used for the 8-shared-episode cross-check |
| `…/incoming/2026-07-28-parity-leak-check/raw/hashes_fp_train_e4387_full.json` | 2,376 train fingerprints — used to recompute the 62 independently |
| `…/incoming/2026-07-26-s3-decision-grade/disjointness_result.json` | the FIRST content measurement (62/80 by poses sha256, 2026-07-26) |
| `Project Steering/RETRACTION_LOG.md` C43 · `Project Steering/AGENT_OPERATING_STANDARD.md` | the preamble and the standing rule |

## Pod access (read-only; pod1 and pod2 were never contacted)

| host | what was read | why |
|---|---|---|
| `tanitad-pod3` (idle, ours) | `ls -d /workspace/pai_epcache/*val*`; `/workspace/tmp/branchb_eval/val_rig_table.json`; `/workspace/tmp/idm/rig_table.json` | to recover WHICH episodes each result set used — the result JSONs do not record it. Computed on-pod; only counts and cache-local `ep_XXXXX` indices were returned. |
| `tanitad-eval` | not contacted | not needed |
| **`tanitad-pod` (pod1, TRAINING 23,000/30,000)** | ⛔ **NEVER CONTACTED** | |
| **`tanitad-pod2` (small validation, arm B_wide)** | ⛔ **NEVER CONTACTED** | |

⛔ No cache was written to, moved, deduplicated or re-selected. **Parity is untouched.**
🔒 No clip UUID and no `episode_id` value appears in any artifact — counts, hashes, window masses and
cache-local `ep_XXXXX` file tags only.

## Test suites

Run on the dev box, `C:/Users/Admin/venvs/tanitad`.

| suite | briefed | measured | |
|---|---|---|---|
| `stack/` | 1576 / 12 skipped | **1,576 passed, 12 skipped** (128 s) | ✅ exactly as briefed |
| `taniteval/` | 726 / 0 | **1 failed, 725 passed, 0 skipped** (111 s) | ⚠️ see below |

**Zero new skips. This audit touched no code** — its only non-new file is
`Project Steering/MODEL_REGISTRY.md` (prose), which no test reads.

⚠️ **The `taniteval` failure is NOT this audit's doing, and the cause is PROVEN, not guessed.**
`tests/test_control.py::test_a_pure_offset_moves_placement_and_NOT_heading` fails. The working tree
carries **uncommitted sibling edits to `taniteval/taniteval/control.py` and
`taniteval/taniteval/pseudosim.py`** (`git status --short`). Substituting **HEAD's** versions of exactly
those two files into an otherwise identical tree and re-running the same file gives **34 passed, 0
failed**; with the working-tree versions, 1 fails. ⇒ the failure belongs to the concurrent closed-loop
control/recovery workstream (the `31134bb` area), which owes either the fix or an updated assertion.
*(This is the same pattern `PARITY_LEAK_CHECK.md` §8.4 reported yesterday for `stack`'s
`test_heldout_gate.py` — a `taniteval` edit breaking a test outside its own suite. Second occurrence.)*

## Escalations that need an owner (full text in `LEAKY_CACHE_AUDIT.md` §8)

1. 🟥 **The inverted remedy** — *"re-evaluate on the **clean** `f1b378` val"* — is still live in
   `taniteval/registry.py:85-87`, `Paper/TANITAD_PAPER.md` §7.2 and
   `…/2026-07-26-doc-correction-sweep/DOC_CORRECTION_SWEEP.md`. I fixed the two registry copies; the
   other three belong to the code and paper owners. R4 F6 flagged this on 2026-07-25.
2. 🟥 **`idm_head_v1_card.json` needs a leak block** — its `provenance` says *"val metrics below are
   held-out"* and its `per_number_verdict` says `val_parityval` yaw R² is *"NOT STALE"*. Both read as a
   clean bill of health for a number that is 80 % train-contaminated. Same for
   `…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` §5.
3. ⚠️ **One probe closes the Branch-B question** — hash Branch B's own 2,466 training clips against the
   already-staged `f1b378` fingerprints.
4. ⚠️ **`physicalai-val-heldout-79d4e3d2d4c6` is content-verified at the POSES level only** and carries
   E1a's headline, E1b and E1c. One `fingerprint_pai_cache.py --mode full` run settles the sensor level.
5. ⚠️ **`v1-encoder-char` and `youtube-idm-pilot` downstream have no committed result JSON** — which
   numbers they published is UNVERIFIED. Stranding, not only leakage.
6. ⚠️ **`run_idm_*` still does not stamp its val cache into its result JSON.** `VAL_PARITY_REPORT.md`
   §4.4 shipped that fix for `taniteval`; this audit had to reconstruct four episode sets from pod rig
   tables because the IDM family never got it. **That is the reason this audit was expensive.**
