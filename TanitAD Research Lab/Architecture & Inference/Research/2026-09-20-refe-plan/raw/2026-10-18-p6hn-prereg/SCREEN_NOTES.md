# P6-HN-S screen: notes, open questions, and the implementation checklist

Written 2026-10-18 on tip `e9b9f3db`. The prereg is `PREREG_P6HN_SCREEN.md`, and its sha256 is in the `.sha256` file beside it.
It is registrable as written. Landing and hashing it are the Master Mind's job. The landed draft `PREREG_P6HN.draft.md`
(`4a9ea208…`) was not edited.

## Questions (each one blocks only this screen)

**Q-S1. PI: does a SCREEN-GO count as the draft's precondition 3?**
- The draft's precondition 3 is "the PI has said go on the compute".
- The ruling of 2026-10-18 says the full arm runs "only if the screen's pre-registered outcome says so".
- The prereg reads GO as meeting that condition. It does not read GO as booking the 6.8-7.65 A40-days. Confirm or override.

**Q-S2. Master Mind: the draft needs an amendment before it is registered (a design finding, not a result).**
- The draft says hard negatives are "appended to that sample's scorer candidates" AND that "the fan term is unchanged bit for bit".
- In REFe the scoring decoder self-attends across the candidate set:
  - `refe/model.py:429`/`:435` (`CrossBlock.self_attn`);
  - `score_trajectories`' docstring at `:699-705`: "a trajectory is scored IN ITS SET".
- So a plain append moves every fan row's score. It also trains the head on 72-candidate sets that are never deployed.
- The draft's own I3 and G-LIVE identity ("loss bit-identical to P6's") would then fail for a reason unrelated to the lever.
- **The screen commits the masked two-call form** (prereg §2.4):
  - the fan call is unchanged;
  - a second call runs over `[fan ; hn]` in which each hard negative attends to the 64 fan rows and to itself, and only the
    hard-negative rows are kept.
- **The draft should adopt the same form by a dated amendment.** It then carries one extra scorer forward per sample, which
  bears on its G-LIVE-MEM row.
- The term lives in ONE function (`refe/hn_bank.py`), imported by both trainers. That is what lets a screen answer carry over at
  all.

**Q-S3. Master Mind: the run box.**
- The default is the dev box, behind `devbox_gpu.lock`, shared with SR-TTC S0. S2 runs in units of 30 min or less with the lock
  released between units.
- Box 2 (a second RTX 4060) could take half of the S2 units, which would cut the ~7-9 h cache build roughly in half.
- Box 2 is the PI's personal machine, so assigning it is your call.

**Q-S4. Master Mind: staging and disk.**
- **Volume:** ~35 GiB is read off the pod's volume (14.5 GiB of jpgs and ≤ 20.4 GiB of nuPlan DBs, ESTIMATED from P2HO's measured
  file sizes).
- **Timing:** before P6 launches, or under P6's overlap rule.
- **Disk:** the dev box needs ~105 GiB in all, of which 69.6 GiB is the bf16 context cache. The free space on D: (exFAT,
  external) is UNVERIFIED.
- **If space is short:** put the cache on an internal drive. That also raises the I/O rate S3 depends on.

**Q-S5. Master Mind: RL-val's base exposure.**
- u400 touched at most 25,600 SFT sets, and the draw cannot be recovered. Up to ~26 % of RL-val tokens may therefore have been
  seen by the base scorer (ESTIMATED, from `PREREG_RL_S1.md` OPEN-BASE).
- This shifts both arms equally, and B − A is unaffected. RL-S1 accepted the same property.
- Confirm you accept it here too.

**Q-S6. Master Mind: the eval set of the frozen-P6 re-screen.**
- The re-screen runs only on a NO-GO/HEAD-BLIND with P6's ZT4 at HEAD-MOVES.
- It needs a navtrain set that P6 never trained on.
- Whether P6's data manifest excludes the 24 SFT held-out logs is UNVERIFIED.
- If it does not, the re-screen has no clean eval set short of navtest. Its prereg amendment must say so before it runs.

## Design notes (none of these is a result)

**Why zeros, not ΔPDMS.**
- On RL-val one paired seed resolves ΔPDMS to a half-width of about 0.58 (R0 − C0 +1.16 [+0.59, +1.74], INHERITED). The only
  published effect is +0.22.
- The zero-pick count resolves a net cut of about 12 picks (ESTIMATED, prereg §12). A DA-WAM-sized cut is about 14-25 picks.
- So the screen can be decisive where the full arm, on its own endpoint, probably cannot.

**Why u400 and not P2.**
- P2 has no clean navtrain pool: it trained on 99,495 of 101,583 eligible tokens.
- P2 has no RL-val fan, and it is NOT REPLICATED.
- For u400, every rig piece already exists.

**Why the banked pre-head features are not used.** They are P2's, they are stale after one update, and they do not exist for
offsets. The full table is also navtest. Prereg §3 gives the details.

**What is cached instead.** The frozen trunk's `visual_ctx` (3.75 MiB per scene, bf16) for 19,000 scenes.
- One forward pass serves all 6 arm-runs (9 if rule E fires), every eval and the fan reproduction check.
- Without the cache: ~72 dev-box GPU-h. With it: ~8-12 GPU-h.

**The inherited regime.** Continued BCE on RL-train degrades the base: C0 sat −0.88 [−1.47, −0.14] below u400 on navtest
(INHERITED).
- The screen keeps the recipe. B − A is a matched contrast inside that regime.
- It also keeps the C0 cross-check (A_0 against C0, report-only, with a RIG-DRIFT flag) and the registered `refe_rls1` profile
  lineage.

**Seeds and logs.**
- Logs: one joint log draw per resample across all seeds and arms.
- Seeds: a sign rule (3 of 3), and a floor F_z equal to the range of the three no-lever controls.
- Rule E: one pre-committed extension to 5 seeds, used only when the log condition holds and the seed conditions do not.
- Strict determinism makes a same-seed rerun bit-identical, so the seeds are the replicate.

**Rule Zero continuation.**
- Every NO-GO row names a next lever, and most of them are 0-GPU reads on the dumps the screen produces.
- **S-DAC is pre-registered inside the prereg** (row 2), so an UNSAFE verdict continues at once without a new prereg cycle.

## Implementation checklist (for the next agent, Sonnet; stage, never commit)

**Order, with stop points:**
1. S0 stage
2. S1 build + SY + I1 + I2 → **stop if SPARSE or INVALID**
3. S2 cache + I6 + G15 (GPU) → **stop if INVALID**
4. mint the `refe_hnscreen` token
5. S3 passes for seeds 0-2
6. S4 read
7. verdict
8. same run: the next lever of the verdict's row

Every stage longer than ~4 min: write a HANDOFF and end (token budget rule).

| # | file (PK-relative) | what | tests that must pass (each with a deliberate-regression arm that must go RED) |
|---|---|---|---|
| 1 | `refe/hn_bank.py` (new) | **Builder.** Offsets (left-normal from yaw, else a central-difference tangent; δ grid; STATIONARY rule < 2.0 m). Labels by calling `onpolicy_relabel_pdm.label_log` itself, never a copy. Hard-negative rule and sub-types. A bank file keyed `(log, token, step, rank)` + `ckpt_step`, with a manifest: partition id, label_version, δ grid, fan fingerprint, sha256. The SY yield report. **Loss.** `hn_term(...)`, the masked second call plus the BCE sum / (B·64·6), shared with `train.py` later. | `refe/test_hn_bank.py`: I2 geometry (straight and curved, median error ≤ 0.01 m, time and heading kept); no map API in the builder's call graph, with a control that must find one in the labeller; the analytic duplicate-row identity; the pad-perturbation identity; mutations (the normal's sign, a dropped STATIONARY rule, a wrong label_version, a copied labeller body) |
| 2 | `refe/ctx_cache.py` (new) | Builds bf16 `visual_ctx` shards for the 19,000 scenes. It captures them with a forward hook on the scene projection's visual output (as `p2ho_dump.py` hooks `score_head`). It writes a manifest binding the trunk fingerprint, scene-list sha and calib. It checks the fan against `raw/rl_fan` dumps within the `atol_calibration` tolerance. It takes the GPU only through the landed `take_lock` / `wait_idle.py` helpers, in per-log units, and resumes. | `refe/test_ctx_cache.py` (CPU fixture): cache == live; a shuffled-row control; a stale-trunk refusal; a resume that is bit-identical |
| 3 | `refe/scorer_finetune.py` | `--hn-bank` (arm B only), `--ctx-cache` (it serves the cached `sctx`; arm A via `model.score_trajectories`, the same function forward uses). The `hn` and `ctx_cache` blocks in the data event. Preflight gates **G14** and **G15**. The refusals of prereg §4.4. `--dump-scores` unchanged. | `refe/test_hn_screen_wiring.py`: I3 (the pre-edit `main()` against the new one, bit-identical with both flags absent); B ≡ A with the bank absent and with it empty; the §4.3 gradient table (non-zero, plus the exact-zero lane row and arm A); the 7 mutations of §4.3 |
| 4 | `refe/gate_checks.py`, `refe/launch_gate.py` | `DVB_KINDS_SCORER += {hn_bank: data, ctx_cache: data}`; `SCORER_HN_KEYS`, `SCORER_CTX_KEYS`; `judge_g14`, `judge_g15`; `SCORER_GATES += G14, G15`; profile `refe_hnscreen` (base argv = `rls1b_arms.json` C1, with the allowed differences of prereg §4.2) | `refe/test_launch_gate.py`: fixtures plus the 12 regression arms of prereg §4.4; `refe_rls1*` arms unchanged |
| 5 | `eval/hn_screen_offsets_score.py` (new) | M3: scores the RL-val offsets of each final checkpoint from the cache as 65th candidates (the shared masked call) | `eval/test_hn_screen_read.py` (below) |
| 6 | `eval/hn_screen_families.py` (new) | Adapts the families instrument (the one that produced `sft4_B_u400_families.json`) to RL-val, with the logged human future as GT. Families that cannot be computed are reported with their reason and n. STRATEGIC is PAUSED. | covered by the test below |
| 7 | `eval/hn_screen_read.py` (new) | Reuses `rl_val_eval`'s lookup for base / A_s / B_s. Joint log draws across seeds (10,000, seed 20260927). Z1-Z3 and rule E; ΔPDMS̄; the co-gate; M1-M3 (M2 = `zt4_baseline.twin_pairs` unchanged); IC (an arm minus itself = 0, CI [0, 0]); A_0 against C0; the outcome classifier of prereg §11 (first matching row) | `eval/test_hn_screen_read.py`: synthetic fixtures that reach **every** §11 row, plus the boundary cases (Δz̄ exactly at F_z; 2-of-3 adverse seeds); IC; mutations (the per-seed draws de-synchronised; tied picks broken by the wrong index) |
| 8 | `refe/hnscreen_devbox.py` (new) | The chain: stage order, a token check before every trainer start, DONE markers, a HANDOFF per stage, the lock. It never removes another job's lock. | `refe/test_hnscreen_chain.py`: deleting the token check or the lock acquisition turns it RED |

Run every test that names an edited file, in full form, before hand-back. G-SUITE counts 0 regressions.

## Proposed register text (for the Master Mind to land with the prereg; not written into any register by this agent)

**GAC row `H-REFE-HNSCREEN-1`** (registered on landing).
- **The hypothesis:** with u400's trunk frozen, the offset hard-negative bank on scorer arm B cuts RL-val zero-PDMS picks against
  the paired BCE-only arm A, robustly across 3 training seeds (Z1-Z3), without an NC/TTC co-gate failure.
- **What its outcomes decide:** GO / NO-GO on the 6.8-7.65 A40-day P6-HN arm.
- **Tier and estimator:** selection tier (RL-val lookup, not T1); paired log-cluster bootstrap over 75 logs, 10,000 resamples,
  seed 20260927, with joint draws across seeds.
- **Cost:** 0 A40-h; about 8-12 dev-box GPU-h (ESTIMATED).
- **Prereg:** `raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md`.
- **Status:** OPEN, not started.

**BACKLOG P6-HN, append:** "Screen prereg written 2026-10-18 (`PREREG_P6HN_SCREEN.md`): u400 frozen; RL-val zero-pick endpoint;
seeds 0-2 plus a pre-committed extension; S-DAC pre-registered; the masked set-scoring correction for the draft (Q-S2). NEXT:
implementation checklist items 1-8, then S0."

## Nothing was changed in the repo

- No existing repo file was edited, so no base blob sha applies.
- Only read-only `git show` / `ls-tree` was run on the mirror at `e9b9f3db`.
- No pod, GPU, navtest table or screen data was touched.
