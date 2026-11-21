# SPEC_WPB_LADDER — AMENDMENT A4 (DRAFT for the Master Mind to register; written 2026-10-12 ~09:40 Europe/Berlin)

**Status: DRAFT.** It becomes part of the registered text only when the Master Mind registers it and records its sha256.
Every clause of `SPEC_WPB_LADDER.md` (sha256 `e1b6aff7…`), A1 (`cbd64fa2…`), A2 (`8c7c73bb…`) and A3 (`41094eca…`)
applies unchanged, except ONE control row of ONE bar (below).

## Protocol fields (the SPEC_REFCV8 §12.2 form)
* **Time:** written 2026-10-12 ~09:40 Europe/Berlin (~07:40Z). The ladder chain is RUNNING on Thor: V0 and V-R8 have
  finished and been evaluated; V-R8d is training (chunk 2 at 07:16Z).
* **Author:** a WP-B engineer agent, briefed by the Master Mind. Registration, and the decision to run the post-chain
  step, are the Master Mind's.
* **refcv8 / ladder data read before writing: YES — disclosed in full.**
  * **The Master Mind** read a DESCRIPTIVE comparison of V-R8 against V0 BEFORE this amendment was requested. It
    included the V-R8 eval rows **vmax_off** and **vmax_shuf**. Because vmax_shuf is byte-identical to base (below),
    that reading showed the quantity "known speed vs unknown speed" (bar 3's comparison), whatever it was labelled.
  * **The author** read only input-side arrays and equality booleans from the V-R8 passes. These were `win_sha12`,
    `v_fed` and `v_valid` of `base_s{0,1}.npz`, and the per-array equality of `vmax_shuf` and `vmax_off` against
    `base`. The author also read the chain log's progress lines (including its `SIZE_K 100.0` instrument line), N = 3000
    and V-R8's argv flags. **The author read no outcome metric**: no trajectory, no speed / ADE / route / tactical number,
    on any row or arm.
  * The descriptive reading is not a scored rung. L4 is not run (A3 item 4), so no L4 verdict exists to protect or to
    move.
* **What changes:** the control row of L4 bar 2 (`B-X3`, second bullet: "VMAX-SHUF − VMAX-OFF is not separated
  better"). Its treatment row VMAX-SHUF is replaced by **VMAX-XSHUF**: each window is fed a DIFFERENT CLIP's speed input.
  The same substitution applies wherever this ladder reads SPEC_REFCV8 §8.3 (c)(3) off V-R8's rows (A3 item 6). Two
  guards come with it:
  * a must-read-known-value control (item 3 below);
  * bar 2 is UNDECIDABLE, never a pass, when its comparison is missing. The registered code passed it vacuously.
* **What does NOT change:** every bar literal (bar 2 stays "not separated better" on speed MAE 2–6 s, at BOTH sampler
  seeds), every other bar and rung, every arm, the 11-arm scope (A3), N, the eval139 window grid and its digest, the
  sampler seeds (0, 1) and their per-batch rule `seed · 100003 + j`, the estimator (the paired episode-cluster
  bootstrap, B 2000, the shared seed-0 draws), the floor F, the reading order, and every existing eval row. The chain is
  not touched: `ladder_chain.sh`, `ladder_eval.py` and `ladder_score.py` are byte-identical on Thor to the repo blobs
  (md5 `f71a25fd…`, `87d4dfa2…`, `7c85d214…`, MEASURED 2026-10-12). Its `LADDER_SCORE.json` is written as registered.
* **Why admissible:** this is an INSTRUMENT SCOPE FIX, not a goalpost move. The registered control row is proven blind
  by construction (byte identity, below), so it cannot answer the bar's question at any data value. The fix:
  * registers the replacement row BEFORE its first number exists (no VMAX-XSHUF pass has run anywhere);
  * keeps the bar literal, the estimator, the seeds and the windows unchanged.

  Under SPEC_REFCV8 §12.4, a control that fails after data makes its row VOID, never PASS. The existing VMAX-SHUF passes
  are therefore VOID as a leak control (item 5 below), and they stay on the record as such.

## Why (MEASURED, Thor, 2026-10-12)
1. **The grid puts one episode in every batch.** `ladder_eval.py` `window_set()` takes exactly K = 8 windows per
   episode, with episodes in sha order. The DataLoader runs `shuffle=False` and the chain passes `--batch 8`. On V-R8's
   passes:
   * the grid has 1,112 windows over 139 episodes;
   * `win_sha12` is constant inside every block of 8;
   * the fed speed `v_fed` is also constant inside every block of 8;
   * all 1,112 windows are known (`v_valid` sum 1112).

   Probe: `/home/nvidia/refcv8_ladder/mm_a4/probe1.py` on `W/eval/V-R8/base_s0.npz`.
2. **The speed input is per CLIP.** `--r8-speed-input label` feeds the road-type step from
   `refcv6_speed_max_v8_eval.jsonl` by clip id (`refc_v3_train.py` `__getitem__`, the `max_speed_enabled` block). On
   the grid, the 139 episodes read {30: 47, 50: 49, 100: 36, 120: 7} km/h.
3. **So the VMAX-SHUF roll is the identity.** `refcv8_train.speed_input_treatment` implements "shuf" as
   `torch.roll(·, 1, 0)` inside the batch. Every window therefore receives its own clip's value.
   * MEASURED by the Master Mind: `W/eval/V-R8/vmax_shuf_s0.npz` equals `base_s0.npz` on EVERY array, `v_fed`
     included.
   * Re-measured by the author on both seeds: all 31 arrays are identical for s0 and for s1
     (`mm_a4/probe2.json`).
   * `vmax_off_s0.npz` differs from base, as it must.
4. **Consequence.** Bar 2 as coded (`ladder_score.py` L645–647) compares **base** with VMAX-OFF. It asks "does the
   KNOWN speed beat unknown?", not "does ANOTHER clip's speed beat unknown?". It can neither detect a leak nor clear one.
   Pinned on literal worlds by `test_ladder_xshuf.py::test_DEFECT_…__false_pass_and_false_fail`:
   * it CLEARS a world where another clip's speed beats unknown;
   * it FAILS a world where another clip's speed reads exactly like unknown.

   This is the CLAUDE.md class *"a check that shares the defect it checks for"*, here in both directions.

## The amendment
1. **New eval row VMAX-XSHUF** (`code/ladder/ladder_eval_xshuf.py` = `ladder_eval.py` plus this one row). The rule is
   fixed in code (`XSHUF_RULE`) and copied into every record:
   * window j of episode e is fed the RAW (`v_max_ms`, `v_max_valid`) of window j of the **donor** episode;
   * the donor is the next episode after e in the grid's sha order (cyclic) whose FED speed differs from e's and that
     is known on every window;
   * when no donor exists, e keeps its own value and is COUNTED;
   * the substitution happens in the batch dict BEFORE `compute_losses_v3`;
   * the model's `_r8_speed_eval` stays `None`, as for `base`;
   * batch j keeps its sampler seed `seed · 100003 + j`, so the row pairs window-for-window with `base`.
2. **Its own instrument checks.** The plan is built from the dataset's per-clip map without decoding an image.
   * Every batch ASSERTS that the loader's own (`v_max_ms`, `v_max_valid`) equal the plan's before anything is
     substituted; a misaligned plan refuses.
   * The record carries the donor-map sha256, `n_changed`, `n_kept_same`, `n_episodes_no_donor`, the per-transition
     counts and a self-check (fed speed vs own value).
   * **Pre-registered expectation (MEASURED from the base grid's input arrays, both seeds identical):**
     `n_changed` 1,112 / 1,112, `n_kept_same` 0, `n_episodes_no_donor` 0, donor map `7e3b86b8179e7fed…`.
     The wrong clip's limit is HIGHER on 552 windows and LOWER on 560. The transitions are, in km/h:
     30→50 184, 30→100 184, 30→120 8, 50→30 240, 50→100 104, 50→120 48, 100→30 120, 100→50 144, 100→120 24,
     120→30 8, 120→50 24, 120→100 24.
     The post-chain step refuses a pass whose donor map differs from this.
3. **L4 bar 2 reads VMAX-XSHUF − VMAX-OFF** (`code/ladder/ladder_score_a4.py`; the bar literal is unchanged). Its
   MUST-READ-KNOWN-VALUE control has two conditions, at each sampler seed:
   * the xshuf row's `v_fed` differs from V-R8 base's on at least the `n_changed` windows its own record names;
   * `n_changed` ≥ 1.

   If either fails, **L4 is INSTRUMENT-BLIND (never a pass) and bar 2 is UNDECIDABLE**. Bar 2 is also UNDECIDABLE when
   a per-seed comparison is missing; the registered code passed that case vacuously.
4. **VMAX-SHUF is reported as `STRUCTURAL-IDENTITY (A4)`.** Its byte identity with base is re-checked at score time.
   If a future grid makes it NOT an identity, it is reported as `NOT-IDENTICAL` and must be re-read before it is
   quoted. It is never scored.
5. **What is VOID.** Any reading of "VMAX-SHUF − VMAX-OFF" from the existing V-R8 passes is VOID as a leak control: it
   is base − VMAX-OFF under another name. The passes themselves stay on disk and on the record. They are a valid **base**
   replicate, and nothing else.
6. **Scope, stated so that nothing is over-read.** L4 needs V-VSHUF, which is NOT RUN (A3 item 4; chain log
   `ZZPI-SKIP V-VSHUF`). L4 therefore stays INCOMPLETE in `LADDER_SCORE.json` and in `LADDER_SCORE_A4.json`. Bar 2's
   quantity still matters, because A3 item 6 reads V-R8's X3 rows "as in SPEC_REFCV8 §8.3", and (c)(3) is that bar.
   `ladder_score_a4.py` writes it to an `a4.x3_c3_reported` block as a **REPORTED** reading
   (NOT-SEPARATED-BETTER / SEPARATED-BETTER / UNDECIDABLE). No rung verdict is taken from that block.
7. **Where the results go.** The chain is not touched. `post_chain_a4.sh` (Thor):
   * waits for the chain's all-done marker after its last start;
   * runs VMAX-XSHUF on V-R8 at seeds 0 and 1 under the chain's own lock, MemAvailable guard, timeout, env and flags;
     the one forced difference is `--ckpt`, because the chain stripped `ckpt.pt` to `model_final.pt`, which holds the
     same `model` tensors and step;
   * asserts on the artifacts;
   * scores into `mm_a4/LADDER_SCORE_A4.json` and asserts that the chain's `LADDER_SCORE.json` md5 is unchanged.

## Pre-registered reading (both outcomes, committed now)
* **Control INSTRUMENT-BLIND** ⇒ no leak reading exists. The next step is to trace the speed path, from the batch
  dict through `r8_before_forward` to the forward kwargs. It is NOT a pass.
* **(c)(3) NOT-SEPARATED-BETTER at both seeds** ⇒ on the ladder rig, another clip's road-type limit does not beat
  "unknown" on speed MAE 2–6 s. **Stamp:** tiny rung, open-loop, one training seed, R > 1 false-pass caveat as base
  SPEC §10.1.
* **(c)(3) SEPARATED-BETTER at either seed** ⇒ a wrong clip's limit improves the plan over no limit. That is the
  signature of the channel carrying information beyond the limit itself (a leak or a prior). It is reported as such,
  and the next lever is the D4 LEAK instrument on the fed channel (SPEC_REFCV8 §8.3 (a)(1)). The PI ruled that LEAK is
  reported, not barred, so this reading informs and does not stop refcv8.

## Validation (binding: false, SPEC_REFCV8 §12.6 — instrument checks, not numbers)
1. **CPU pins.** On the dev box, against a `git archive` of `agent/arch-inf-20260803` at `a8e5ab1`: 21/21 new pins
   and 20/20 existing `test_ladder_score.py` pins pass. Five code mutations each turn the suite RED.
2. **Real-data smoke on Thor, CPU only.** Conditions:
   * the real loader and tree `1df35031db`;
   * the V-R8 `model_final.pt` (md5 `1b12e8eb…`, an md5-verified copy in `mm_a4/ckpt/V-R8/`);
   * the first 24 grid windows (3 episodes), sampler seed 0;
   * `CUDA_VISIBLE_DEVICES` empty, nice 19, 2 threads.

   Results:
   * The `base` row from `ladder_eval.py` and from `ladder_eval_xshuf.py` is **identical on all 31 arrays** and in its
     record (compute times aside) (`mm_a4/smoke/probe4.json`).
   * `vmax_xshuf` changed 24/24 windows and its self-check reads PASS. The plan came from the per-clip map in 0.0 s.
   * `v_lim` differs from base on 24/24 windows, so the ceiling follows the donor's limit.
   * Every input array (`gt`, `v0`, `nav`, `rc_raw_ds`, `lat_v9`, the window list) is identical to base. The window
     list and the fed speed equal the chain's own `base_s0.npz` first 24 (`mm_a4/smoke/probe3.json`).
   * Cost: wall 614 s for the two passes plus 385 s for the chain harness's base.
   * Memory was read by hand, a few samples only. The watchdog's own recorder read 0, because `/proc/*/children` is
     absent on Thor. VmHWM of the python process was 10.98 GB in a mid-run sample, so the true peak is at least that.
     The lowest MemAvailable sampled was 66.7 GB, against the watchdog's 50 GB floor. The chain kept training
     throughout.
3. **`post_chain_a4.sh` dry test on Thor** (`mm_a4/drytest/dry.log`). Conditions: a fake W and A4, a private lock
   file, and fixture passes, so the GPU step is never reached.
   * The fixture `chain.log` reads start → STOP → start. The script WAITED, and continued only after the all-done line
     was appended after the last start.
   * **GREEN** fixture ⇒ status `DONE`. The fixture's `LADDER_SCORE.json` md5 was unchanged.
   * **BLIND** fixture (xshuf `v_fed` = base) ⇒ status `EVAL-ARTIFACT-BAD`. It never reached the scorer.

## Launch (the Master Mind's decision; NOT done)
`ssh -n tanitad-thor-wifi 'cd /home/nvidia/refcv8_ladder/mm_a4/code && setsid nohup bash post_chain_a4.sh > ../post_chain.out 2>&1 < /dev/null &'`

To read the outcome, `cat /home/nvidia/refcv8_ladder/mm_a4/A4_STATUS`. It holds one token:
* while running: `WAITING`, `PREFLIGHT`, `EVAL-RUNNING` or `SCORING`;
* finished: `DONE` or `DONE-INSTRUMENT-BLIND`;
* stopped: `BLOCKED-…`, `EVAL-ARTIFACT-BAD`, `SCORE-ARTIFACT-BAD`, `CHAIN-SCORE-CHANGED` or `TIMEOUT-WAITING`.

The summary is in `mm_a4/A4_SUMMARY.txt`. The code on Thor (`mm_a4/code/`) is md5-pinned by `A4_MD5SUMS`; a changed
file refuses with `BLOCKED-CODE-MD5`.

## Cost (ESTIMATED from the chain's own MEASURED timings)
One VMAX-XSHUF row costs about the same as each of V-R8's six chain rows: 10.9–11.0 min for both seeds on the GPU
(`chain.log`, e.g. vmax_shuf 06:45:20Z → 06:56:20Z). The plan adds no image decode. The scorer is CPU only.

## Code and pins (staged as `hold_ladder_a4`)
* `code/ladder/ladder_eval_xshuf.py`: `ladder_eval.py` plus the row. Its unified diff against `ladder_eval.py`
  (both LF) is −1 / +148. The one removed line is the `ROWS` tuple, which comes back extended.
* `code/ladder/ladder_score_a4.py`: `ladder_score.py` with the bar-2 row, the control, `--extra-eval` / `--out` and
  the `a4` block. Its unified diff is −12 / +104. The 12 removed lines are exactly:
  * L4's row need;
  * the `score()` signature and its pass-loading loop;
  * bar 2;
  * the L4 verdict line;
  * `main()`'s three output lines.
* `code/ladder/test_ladder_xshuf.py`: 21 CPU pins, all literal expectations, each reading with a RED arm. The pins
  cover:
  * the identity reproduced on the real `speed_input_treatment`;
  * the donor rule;
  * pairing;
  * all six old rows identical between the chain harness and the A4 copy;
  * INSTRUMENT-BLIND on an identity xshuf row;
  * every other rung identical under the A4 scorer.

  Five code mutations each turn the suite RED (blind gate off, no-op substitution, an old row drifting, bar 2 reading
  vmax_shuf again, a shifted sampler seed).
* `code/ladder/post_chain_a4.sh`: not launched; the Master Mind launches it.
