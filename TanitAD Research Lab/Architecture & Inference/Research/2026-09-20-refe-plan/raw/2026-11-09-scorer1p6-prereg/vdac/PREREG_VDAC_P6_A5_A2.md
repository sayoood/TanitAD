# PREREG_VDAC_P6_A5_A2: HC-s0's post-training gate is told the 2 m head count, and how the arm is completed WITHOUT retraining

**Status: FROZEN by the Master Mind (MM) 2026-11-13 (sha256 in `PREREG_VDAC_P6_A5_A2.sha256`, written beside this file). Frozen before HC-s0 finished training, before the by-hand completion ran, and before any HC-s0 output was opened by a reader.**
This amends `PREREG_VDAC_P6_A5.md` (sha256 `a51b0349c01f...`) and follows `PREREG_VDAC_P6_A5_A1.md` (sha256 `2b40a672f287...`). It changes **execution only**. No lever, seed, update count, bar, arm, argv, data pin or outcome row of either card changes.
Evidence classes: MEASURED (artifact) / INHERITED / ESTIMATED / HYPOTHESIS. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## 1. The defect (MEASURED on the pod's shipped tree, `stage_hc-s0/*.ok` tree `f17ce364...`, before HC-s0 finished)

* `refe/vdac_chain.sh:350` runs the post-training gate as `vdac_postgate.py "$OUT" --updates $HAUPD`, with **no `--n-params`**.
* `refe/vdac_postgate.py:13` sets `N_PARAMS_CARD = 2440450`, and `:94` makes that the `--n-params` default. That is the 2 m head (cell 2.0).
* HC's head is the 1 m head (cell 1.0, 80 x 64 queries). Its BUILT count is **3,423,490**, from `gate_checks.AUX_HEAD_PARAMS_BY_CELL`.
* So the gate compares HC's Adam state against the wrong count and must print `ZZVDAC_POSTGATE_FAIL`. `vdac_chain.sh:353` then logs `ZZVDAC_CHAIN_FAIL HC-s0 (post-training gate)` and exits 1.
* Consequences:
  * no export and no egress of HC-s0;
  * by the card's stop rule 1, HC-SHUF is NOT run in the `A5HC` pause, and P7 resumes.
* The defect is in the CHECK, not in the training. The gate refuses a correct arm. Every other stage of the chain (preflight, launch gate, run check) is unaffected.
* **Class:** "check the DEFAULT, not just the flag" (CLAUDE.md, *built, tested and unreachable*, rule 4), and "a derived constant silently changes the experiment when its input changes" (`TRAPS.md`).
  * The 2 m count was correct for every arm before HC (HA, HB).
  * The 1 m cell is the first arm whose head size differs.
  * Nothing passed the arm's built count to the gate.
  * `test_vdac_gate.py` ran the chain only on 2 m arms.
* **When it was found:** by the MM while verifying the VDAC A6 (HBC) hold, with HC-s0 at about update 1,750 of 10,000. The MM decided **not** to edit the running pinned tree (CLAUDE.md invariant). HC-s0 trains to completion unchanged.

## 2. The fix (in the A6 hold `vdac_a6_hbc`, landed with this amendment's GAC row)

1. **`refe/vdac_chain.sh`:**
   * the arm table sets `HEADN`: default `2440450`; `HEADN=3423490` for HC-s0, HC-SHUF, HBC-s0 and HBC-SHUF;
   * the gate line becomes `vdac_postgate.py "$OUT" --updates $HAUPD --n-params $HEADN`.
2. **`refe/vdac_pause_p7.sh`:** a new stage `A5HCSHUF`, which runs `HC-SHUF` alone (timeout 7,200 s, unchanged from `A5HC`).
3. **`refe/test_vdac_hbc.py`:**
   * asserts the arm table and the gate line (`HEADN=2440450` once, `HEADN=3423490` four times);
   * asserts the `A5HCSHUF` stage;
   * asserts that the post-training gate REFUSES a 1 m Adam state at the 2 m default, and passes it at 3,423,490 (the deliberate-regression arm).

## 3. Completing HC-s0 WITHOUT retraining (the one execution change; the precedent is A1 section 5(d))

The chain cannot resume mid-arm. The MM therefore runs **by hand** exactly the steps the chain would have run after its run check. The script is `a5_a2/hc_s0_complete.sh` (sha256 `3fb48a03346294d482ae8c5a1a65b5f87338ba1be26ecbc255011b1d6a300a2a`, shipped to the pod and verified by sha256 and `bash -n` before HC-s0 finished).

1. **Preconditions; it REFUSES unless all hold:**
   * `run_check.log` carries `ZZVDAC_RUN_OK`. That is the chain's own check that `ZZSFT_DONE` updates equal 10,000, the head was saved, the H48 forward count equals the audit's excluded frames, and the held-out prediction file and both model files exist.
   * The chain log carries `ZZVDAC_CHAIN_FAIL HC-s0 (post-training gate)`.
   * The chain's `postgate.log` is a `ZZVDAC_POSTGATE_FAIL`.
   * No `postgate_a2.log` exists yet (one run only).
2. **The post-training gate** with HC's built count: `vdac_postgate.py hc-s0 --updates 10000 --n-params 3423490 > postgate_a2.log`. It must print `ZZVDAC_POSTGATE_OK`, or HC-s0 is NOT RUN (section 5).
3. **The egress:**
   * The SAME file list as `vdac_chain.sh:374`, under the SAME prefix `vdac/hc-s0` and tag `arm_vdac_hc-s0`. The list includes the chain's FAILED `postgate.log`, kept unaltered as evidence.
   * Added: `postgate_a2.log`, `chain_hc-s0.log`, and `a2_local_sha256.txt` (the pod sha256 of every banked file, the two model files, the scorer states and the H48 export).
   * Verified by the `ZZEGRESS_OK` line, i.e. by the remote listing, not by the push call.
4. **Hub LFS sha256** of `aux_drv.pt` and of `vdac/export/h48_proposals_subscores.npz`, compared with the pod files (`a2_lfs_sha.json`). The export was banked by an earlier arm on this pod. HC-s0's chain never reached its export stage, so the reader's input is checked rather than re-pushed.
5. **What is NOT done:**
   * No chain marker, no `ZZSFT_DONE`, no `ZZVDAC_CHAIN_DONE`, and no stage file is written by hand.
   * No launch token is touched.
   * Nothing trains and the GPU is not used. The script needs CPU and network only while P7 trains, and the gate reads one small checkpoint.

## 4. HC-SHUF

* HC-SHUF runs in its own P7 pause, stage `A5HCSHUF`, on the **fixed** tree: the A6 hold landed and re-shipped.
* Its chain mints its own launch token, exactly as the card's section 6 sets it. A re-ship voids tokens bound to the old tree, as intended.
* Same flags, seed and update count as the card (`HC's flags + --aux-drv-shuffle-seed 0`, 2,000 updates).
* Cost:
  * the card: 0.86 L40-h plus chain overhead;
  * ESTIMATED from HC-s0's measured 1.07 s/update: 2,000 x 1.07 s = 0.6 h of training, plus about 0.5 h of chain (preflight, G-SUITE, gate, export check) = **about 1.1 L40-h**.
* The PI approved HC on 2026-11-13 over the cap (verbatim: *"ok to exceed cap"*). This pause is part of that approved HC pair, not new spend.

## 5. Outcome rows (Rule Zero: every row names its successor)

| result | next |
|---|---|
| `postgate_a2.log` `ZZVDAC_POSTGATE_OK`, egress `ZZEGRESS_OK`, LFS sha equal | HC-s0 is complete. Pull, sha256-verify against `a2_local_sha256.txt`, write `manifest_a5_hc.json` BEFORE opening. The reader runs HC-s0 when HC-SHUF has finished (or with HC-SHUF's row NOT RUN if its chain fails). |
| `postgate_a2.log` FAILS at 3,423,490 | A real post-training defect (a head parameter with no Adam state or a zero first moment). HC-s0 is **NOT RUN**: report which parameters. A retrain is a PI decision: about 3.0 h of training (10,000 x 1.07 s/update, the MEASURED rate) plus chain overhead. HC-SHUF is not run. |
| a precondition refuses (`run_check` not OK, or the chain failed elsewhere) | HC-s0 did not reach the post-training gate. The chain's own failure stands; this amendment does not apply; the card's stop rule 1 rules. |
| egress unverified | HC-s0 lives on the pod only. Re-push the same list (idempotent); no read until `ZZEGRESS_OK`. |
| HC-SHUF chain fails | HC-s0 stays complete; HC-SHUF's row is NOT RUN, and C-SHUF for HC is unread. HC-s0 can only be reported as **controls incomplete**, never as X-LIFT. |

## 6. Disclosure (binding)

* While checking progress, the MM read HC-s0's `sft.log` train lines at update 7,050: `bceA`, `lossB`, `listnet`, `aux_bce` and `aux_huber`. These are training losses, not evaluation numbers and not the card's read.
* The MM has read no HC-s0 eval line, prediction file or S1 number.
* Nothing in this amendment depends on, or was chosen after, those lines. No lever, seed, update count, bar, arm or outcome row changes.

## 7. MM rulings at freeze

1. The by-hand completion of section 3 is accepted on the A1 section 5(d) precedent.
2. Stage `A5HCSHUF` is accepted (one extra P7 pause; the card's `A5HC` would have run HC-SHUF in the same pause).
3. **No K-ledger entry** (as the card: unkeyed S1 navtrain reads, section 10).
4. The A6 hold's other contents (the HBC arms, their launch-gate profiles and the DRAFT HBC card) land as **code and a DRAFT only**. HBC's pod time is a separate PI decision, asked after the HC read.
