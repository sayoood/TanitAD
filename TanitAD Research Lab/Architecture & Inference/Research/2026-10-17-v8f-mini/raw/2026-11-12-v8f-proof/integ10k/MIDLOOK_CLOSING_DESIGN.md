# MIDLOOK-CLOSING: later looks of the rule for RRc and RRh (design note)

Generator `make_midlook_closing.py` writes `v8f_t1_eval_reader_rule_rzgfhqc.py` from the wrapper of record `v8f_t1_eval_reader_rule_rzgfhq.py` (sha256 prefix 92a8034cc0f7b0d2; the generator refuses any other parent). Parent line numbers below are those of the parent file. Eight asserted edits (each old text exactly once, 33 lines added net).

## What changes (and nothing else)
1. Refusal (parent 1174-1175): `--rule-every` below the replan interval was accepted for RLb, RLc, RR only. Now also RRc and RRh. RRd keeps being refused, with its own message that names RRd (its closing speed is a gap rate over an earlier read, `gap_rate_closing` 236, `prev_read` 219; a later look carries no such read).
2. Reader branch of `mid_looks` (parent 583): `if form == "RR"` becomes `if form in ("RR", "RRc", "RRh")`. Presence and gap come from `_reader_at(st, rd, q)` (524-539) and `feed(...)` exactly as for RR.
3. Closing at a later look (parent 586 sets `hc` to zeros): RR keeps zeros; RRh takes `c_read`, from the new helper `_reader_closing_at(st, rd, q)`; RRc keeps zeros as the reader value.
4. Mid-look log: new key `c_read` (added to `MID_FIELDS`, parent 508; NaN in the label forms RLb, RLc). No existing key is changed.

## Exact semantics at a later look q (frame j + q)
- RRh: `hc = reader_closing(rd, tokens)` on the tokens of cache steps j+q-4 .. j+q of the window's own clip (`s.d["feats"][e-hist+1:e+1]`, e = j + q), float32, times `CLOS_SCALE_MS` = 10 (parent 206-216). This is the token window, row rule (`e-(hist-1) >= 0 and e < len(feats)`) and scaling of the gap read at j + q (parent 531-534) and of the tick-level RRh feed (253, 271: `c if form == "RRh"`). Rows without a window: NaN, and presence needs the same window (p is NaN), so no row is fed a NaN. Mix branch `mix["closing"] == "head"` (parent 602; RRh = "head", parent 1246) then gives `c_src = hc` (the label is used only where a present lead has no reader closing, which cannot happen on a window row). `v_L = max(0, v_l + c_src)` as at the tick.
- RRc: nothing in the closing code changes. Parent 1246 sets RRc closing = "label"; the branch at parent 600-601 reads `c_src = where(present & ~lab_c & head_c, hc, lc)` with `lc` from `label.read(ci, j+q)` (parent 579-582, label at the clip and frame j + q), so a later look feeds the label closing at j + q; where the label has none and a lead is present, `hc` = 0 (the reader value of the form at the tick is `c_fed` = zeros, parent 252-271). Verified from the code, no change made there; the refusal was the only obstacle.
- RR: closing 0 at every later look, as in the parent (mix closing "zero", parent 605). `c_read` is logged but not fed.
- RLc, RLb: unchanged (label branch, parent 591-593).

## Assumptions
- `s.d["feats"]` at the window state is the array the batch tokens are gathered from (stated in the parent's own `_reader_at` text, 524-527); the check reads the cache file of the clip instead and so tests that premise.
- The reader closing is read on tokens only: no model weight, no label (parent 206-208).
- With `--rule-every` at or above the replan interval no later look runs and `save_mid` returns before writing (parent 636): arrays equal the parent.

## What the pod check covers (`check_midlook_closing.py`, 12 tiny rollouts + 1 refusal)
A: every=10, forms RR, RRc, RRh: dump, tick-file (not the run-name array) and rule-log arrays equal the parent, difference 0. B: every=5, forms RR and RLc: the same plus the mid-look log, its reader side file and the first-plan file; new keys are listed. C: every=5 reach: RRc `c_src` equals the label closing at (clip, j+q) recomputed from the join (1e-9, not 0 on one row at least); RRh `c_src` equals the closing recomputed from the cache tokens by an independent write-out of the head (1e-6 default, `--tol-reader`; not 0 on one row; equals the logged `c_read` exactly); RR control exactly 0; `c_read` of all three reader forms equals the recomputed value; a planted arm (RR log read as RRh) is caught. D: RRd with every=5 refused (return code and text).
Not covered: (i) the check recomputes from the cache file one row at a time, so the 1e-6 bar also absorbs batch-size float differences, and a failure at 1e-6 with a difference of order 1e-6 would call for `--tol-reader`, not a code change; (ii) only present rows are tested (not-present rows carry `c_src` of no consequence); (iii) outcomes of the rule on the metrics are not examined; (iv) RRd later looks are not built.

## Rejected
- Feeding the later look the RRd-style gap rate: needs an earlier read per look; a larger build, and RRd stays refused.
- Changing `_reader_at` to return the closing: it would change the arrays RR logs through the `rd` side file or its return shape; a separate helper leaves every parent path untouched.
- Writing `c_read` into the `_rd` side file: that file has a fixed key set read by other instruments; the mid-look log is the place the brief names.
- A patched copy of the pinned tool: not needed; the wrapper alone carries the change.
