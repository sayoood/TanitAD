# PREREG_DATASCALE AMENDMENT 1: the pre-read plumbing check (MM, 2026-11-28)

**Status.** Dated amendment to the FROZEN card (`PREREG_DATASCALE.md`, sha256 `4e52b00e...`). It changes **only the pre-read plumbing check in section 10**, a procedure. No bar, measure, control, outcome or next arm changes.
- Made before any DATA-SCALE fit on the 185-log index; that index does not exist yet.
- Made before any DATA-SCALE navtest number exists.

**What section 10 said.** *"Before the decision read, the read runs once on the parent EP-HEAD run1 heads under `--lever ds_ep` and must reproduce `READ_EPHEAD.json`'s M1 to 1e-9."*

**Why it cannot run as written (MEASURED 2026-11-28, `datascale/run1/plumb/read.log`).**
- `ds_read.py read --lever ds_ep` on `l2next/run1/eph_{s0,s1,red}` stops in its CTRL-CV(e) recheck with `KeyError: 'per_log'`.
- The parent heads were selected by `cv5`, whose `cv_select.json` has no per-log out-of-fold table. CTRL-CV(e) recomputes `cv5m` from that table, so a `cv5` head can never pass it.
- That refusal is correct by construction. The failure is loud (rc 1, no READ file) and cannot produce a number.

**What ds_read adds over the proven read.**
- `ds_read.py` IMPORTS the frozen `l2next_read.py` and rebinds exactly two functions: `_run_head` (the head-select check wants `cv5m`) and `_ctrl_cv` (CTRL-CV, including the (e) recheck).
- The pick and score arithmetic (4.1, 4.2, 4.4), the bars, the remaining controls and the families are `l2next_read.py`'s, unchanged. They were proven end to end on real data by EP-HEAD's read (`PK/raw/2026-11-27-l2next/run1/read_eph/READ_EPHEAD.json`, every control PASS).
- So the plumbing check must exercise the two rebound functions on REAL `cv5m` heads.

**The replacement check (binding, run before the decision read):**
1. Train lever B's three arms with the card's recipe and `--select cv5m` on the **parent's 35-log sealed index**, each under its own launch-gate PASS token. The profiles are `refe_ds_ep_{red,s0,s1}`; the outputs go to `datascale/run1/plumb35/dep35_{red,s0,s1}`.
2. Run `ds_read.py read --lever ds_ep` on them with `--train-index` set to that index.
3. **Pass condition:**
   - the read completes (rc 0, a READ file);
   - every control PASSES, including CTRL-CV (a)-(e) on the real per-log table, CTRL-RED and CTRL-ID;
   - the outcome is **NULL-CV**, as M6 at n = 35 already measured: `cv5m` chooses step 0 there, `run1/m6_35/lc/lc.json`;
   - Delta0 = 0 exactly (the picks are C1's).
   Any other result is a defect in `ds_read` and blocks the decision read.
4. These heads are a plumbing fixture. They are never a DATA-SCALE arm, and no number from them is a DATA-SCALE result.

**Not changed:** `ds_read.py` itself. A `cv5` head under `--lever ds_ep` stays a loud crash, not a reported control FAIL; that is recorded here as known behaviour.
