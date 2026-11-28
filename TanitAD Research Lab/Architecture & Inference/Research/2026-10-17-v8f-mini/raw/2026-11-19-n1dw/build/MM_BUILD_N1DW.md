# MM build record: N1DW (PREREG_V8F_N1DW, sha256 59f2b7f9, with Amendment 1), 2026-11-19

**Code of record:** commit `134f5da1ee50a3c2fa242bb340a26dd6e4072ade` on branch `agent/v8f-n1dw-20261119`, a child of N1D's `0c3b374f`. It changes 10 files; the patch is `b1/n1dw.patch` (sha256 049e60d0).

## B1: dev-box build and CPU proofs (`b1/`)
- **Records:** `REPORT_B1.json` (sha256 dc212041) and `HANDOFF_B1.md`. Mutants (a) to (g) all go RED (`mutation_table_n1dw.json`).
- **The step-0 properties the card binds, proven on CPU** (`REPORT_B1.json`, `controls`):
  - **Z0:** at w = 1 the tree is bit-identical to the tip. All 114 state-dict values and all 17 loss terms are equal, and the total is equal.
  - **Z1:** at w = 4, no existing term or output differs from w = 1. The extra term is `rb_traj`. The total moves by `w_traj x 3 x rb_traj`, to a relative error of 1.9e-6.
  - **Z2:** the term's contribution equals `w_traj x (w - 1) x term`.
  - **Z_ID:** the per-sample gradient ratio on roundabout samples is exactly w (3.9999995 to 4.0000005). Non-roundabout samples are bit-equal.
- **Sampler p** on N1D_s0's argv: 0.0672 over the first 3,000 batches, 0.0678 over the epoch. K-SHARE: 0.126 at w = 2, 0.224 at w = 4.
- **Dev-box regression comparison against 0c3b374f** (`b1/regress_base`, `b1/regress_new`, `b1/regress_new2`). These files fail the same tests on both trees:
  - `test_text_encoding_is_explicit` (1);
  - `test_v8f_oacc` (2);
  - `test_v8f_flat` (1);
  - `test_v8f_launch_intent` (1);
  - `test_v8f_ego_dagger_compose` (1, from the `t2_bank_*` summary keys of ae02d340, present in both trees).

## B2: pod build (`b2/`; pod records in `b2/pod/`)
- **Pod tree** `/root/strat1/n1dw/tree`: 3,429 of 3,429 files equal their 134f5da1 blobs.
- **Pod G-SUITE against 0c3b374f** (`b2/pod/gsuite_result_n1dw.json`): PASS, with 0 regressions, 0 new-test failures, 0 missing, and 94 new-only tests. Base 15,121 tests, new 15,215.
  - Failures plus errors went from 635 to 640. The 5 extra are tests that SKIPPED in the base junit and FAIL in the new run: `test_v8f_wc` x3 and `test_refcv7_vis1` x2 (`b2/pod/junit_skip_to_fail.txt`).
  - **Cause, MEASURED:** both pod trees hold one stray file, `TanitAD Research Lab/.../2026-10-17-v8f-mini/raw/eval_grid_windows.json`. It was added to the base tree at 21:02Z on 11-18, after the base junit ran at 19:01Z. With it present, these tests take the tree for a full checkout and fail on files it lacks.
  - Rerun on the base tree at about 16:05Z on 11-19, the same 5 fail identically (5 failed, 24 passed, 3 skipped). They are environmental, not N1DW.
- **K-CLIP:** clipping was active on 60 of 60 logged steps for N1D_s0 (median grad_norm 15.2) and N1D_s1 (14.1), at clip 1.0 (`b2/kclip_N1D_s*.json`). This is printed beside every N1DW label, as the card says.
- **Argv:** four files in `b2/argv/`, each equal to N1D_s{0,1}'s banked argv plus `--rb-window-w 4|2` and its own `--out`. The pairing check PASSES, and its three negative controls are REFUSED.

## Launch-gate tokens (all four read PASS on every gate)
| arm | token record (`b2/pod/`) | sha256 (first 16) | argv sha256 (first 8) | gates |
|---|---|---|---|---|
| N1DW4_s0 | `N1DW4_s0_gatetok.json` | a1f14f7c5acfdedd | dabcbfb9 | G-CKPT, G-CLOCK, G-DVB, G-EVAL, G-HYG, G-LIVE, G-SUITE, frozen-argvs: all `pass: true` |
| N1DW4_s1 | `N1DW4_s1_gatetok.json` | 0428daa404657d6b | ea857750 | all `pass: true` |
| N1DW2_s0 | `N1DW2_s0_gatetok.json` | 5f3472b83747e6d4 | 26b98d66 | all `pass: true` |
| N1DW2_s1 | `N1DW2_s1_gatetok.json` | b179a07af62056a3 | adc6ac85 | all `pass: true` |

- All four were minted by `strat1_check.py mint` at commit 134f5da1, between 16:12Z and 16:20Z.
- The word "FAIL" inside a token's G-CLOCK note refers to that check's deliberate-regression arm, which must fail.

## The R2 flag, and why it is dismissed
- **Raised by:** `b2/r2_compare.py`, written for card R2 ("read every GPU check against a same-code repeat"). Output: `b2/pod/r2_compare.out`, `b2/pod/r2_result.json`.
- **What it compares:** the `live.terms` of the G-LIVE selfcheck (`gate/selfcheck.json`) for each N1DW arm against two same-code N1D repeats. It flagged every existing term of every arm.
- **Why that is not a defect.** G-LIVE runs `steps` optimizer steps and records the LAST step's terms (`v8f_mini_train.py` at 134f5da1, the live loop: `for s in range(steps): ... T["total"].backward(); ... opt.step()`, `terms_seen` overwritten each step). So a lever that changes the gradient moves every term after step 0, by design.
- **Consistent with that:**
  - the repeat is bit-identical (difference 0.0);
  - at s0, the ADE term moves 0.0623 at w = 2 and 0.1795 at w = 4, a ratio of 2.9, close to the ratio of (w - 1), 3;
  - the s1 arms differ from the repeat by about the same amount at both weights (ckpt term 0.4475 and 0.4478), which a seed difference against the repeat's argv would produce.
- **The binding step-0 evidence is B1's CPU proof above** (Z0, Z1, Z_ID). Card R2 itself placed the Z checks on CPU because GPU runs are not bit-reproducible.
- **Lesson for later comparators:** compare step-0 terms only, on the same seed as the repeat.

## Launch
- The chain `b2/n1dw_chain.sh` was started by the MM at 16:26:31Z on the pod, under `flock /root/gpu_queue.lock`. Arms run in the card's order.
- The GPU-priority marker is released by `n1dw_marker_release.sh` after the 4th arm trains.
- **Deviation, to close before any verdict:** the chain runs the D1-HO dumps on pod CPU but not the eval-141 guard dumps (card section 5.2). In NAVRBD those were a dev-box step. They run on the dev box after another session's read frees its memory.
