# AMEND_ACTSENS_1 (Master Mind, 2026-11-15): C3 reads the first G-LIVE step, against a reference gate of R35's argv on the same tree

Card `PREREG_V8F_ACTSENS_DRAFT.md` (frozen, sha256 `b62227d9`). Written **before any ACTSENS arm has trained and before any window is scored**. Doses, arms, run order, the deciding read, outcomes and floors are unchanged. Only the instrument of control C3 changes, and one gate is added.

## What happened (MEASURED)
* The chain started at 02:29 on 2026-11-15 (code of record `6b2bca1f`). It stopped at 02:38 at the pilot job's gate of IDM_s0, as designed (fail closed, nothing trained): `C3: 16 unweighted live terms differ from R35_s0's (first: wm/pred/L1 0.2695418894290924 vs 0.2549646496772766)`; `level/L1 difference 101.49755096435547 != 900 x idm/L1 101.86743512749672`.
* **Cause.** The gate's `live.terms` is the record of the **last** of G-LIVE's three AdamW steps. `wm_only_live_check` (trainer, `6b2bca1f`) overwrites `terms_seen` at every step. By step 3 the IDM arm's weights have moved under a ten-fold level-1 IDM gradient, so its terms cannot equal R35's, at any level. The card's C3 is about the same init and the same first batch, so the checker read the wrong step.
* **Reference gate on box 2** (diagnostic, `--selfcheck-only`, nothing trained): R35_s0's exact argv through the shipped ACTSENS tree. Its `live.terms` equal R35_s0's banked `selfcheck.json` terms **25 of 25, bit for bit**. That is the real-data form of C1 for chassis s0, measured for the first time. IDM_s0's gate matches that reference on 6 of 25 terms, the six counts.
* **CPU** (synthetic, `F:/TanitAD-scratch/actsens/mmcheck/c3_mech.json`): at step 0 all 9 unweighted terms are equal and the 900 x identity holds. By step 2 one term has moved.

## Amendment
* **A1, C3's instrument = the first G-LIVE step.** On a lever-named launch the trainer records `live_first`: the terms of G-LIVE's step 0, taken before any update. C3 compares IDM_s{k}'s `live_first` with REF_s{k}'s `live_first`. The unweighted terms must be equal bit for bit, `level/L2` and `level/L3` equal, and `level/L1` difference = 900 x `idm/L1` (1e-5 relative). The card's wording of C3 stands; only the instrument changes.
* **A2, gate REF_s{k}** (gate only, never trains, no token). The argv is R35_s{k}'s with `--wm-idm-gamma 100 100 100` added: the default dose, named so that `live_first` is recorded. It runs on the shipped tree inside the chain, before IDM_s{k}'s gate. Its `live.terms` must equal R35_s{k}'s banked terms bit for bit on every key, which is C1 on real data and also shows that the named default equals the absent flag. Its built gamma must read [100, 100, 100].
* **Deliberate regressions** (each must make the gate refuse):
  * C3 reading `live.terms` (step 3) instead of `live_first`, which is what happened at 02:38;
  * a REF record with one term changed;
  * an IDM record whose `live_first` comes from a different batch.
* **Cost:** two more `--selfcheck-only` gates on box 2, minutes each. The ~21 h estimate is unchanged.
