# PREREG_V8F_PROOF_ADD152 — HEADCLOSE10K: does the vision lead-speed head's closing, fed as the closing input of the every-step reader-form rule, lower the unsafe rate on the candidate (RH under readout K)?

Frozen before any tool or run of this card and before any value of it exists. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance. PI, 2026-11-28, verbatim: "Card, build and run it" (box 2 is allocated to the v8 programme).

## 1. Why this card

Row 217 (card ADD150 with amendment 1, `LEADSPEEDALL10K-SPEED-ONLY`): a temporal head on camera tokens, trained on all 3,984 fit clips, reads the lead's closing speed at r 0.426 (training seed 148) and 0.445 (seed 149) on the eval-141 clips, above its 0.30 bar, and not the acceleration (r about 0.17). Its committed consequence: the head's closing is the closing input of a reader-form rule arm, without the acceleration. Today's reader form is row 211's `H_RQ50KE1` (form RR, rule every step, readout K): presence and gap from the token reader through the q50 map, closing input zero, so the lead speed is v_L = own logged speed. Row 213 is the same form at age (pre-roll 30, `H_RQ50KE1_P30`).

Stated guess before data: `HEADCLOSE10K-U-NOT-SEPARATED` (the head's closing r of about 0.44 leaves most of the closing variance unread), contact C not separated, families not separated.

## 2. The new form and the closing table

* **Form**: RRe with `--closing-table` of the existing wrapper `shufclose/v8f_t1_eval_reader_rule_rzgfhqet.py` (sha256 02a58e655e5c1979; card ADD83's tool, mutation-tested there; used unchanged). Presence and gap are those of form RR (token reader TK16, q50 map); at every look of the rule (the tick at frame j and every later look at frame j + q, main phase and pre-roll) the closing input is the table value at (clip index of the window, frame of the look), so v_L = max(0, v(f) + c(f)) with v the token cache's own logged speed (the speed RR already uses). Where the table has no row or a value that is not finite, the closing input is exactly 0.0 (form RR's value).
* **Head**: the MEAN of the two row-217 heads' closing per frame, c(f) = (c_148(f) + c_149(f)) / 2. Checkpoints `D:/tanitad/leadspeed/all/head_30000.pt` (sha256 83faaa6d355562f7) and `D:/tanitad/leadspeed/all149/head_30000.pt` (4ae87b88ab53ec39), `make_head(1024, 64, temporal=True)`, read by `head_outputs` of `train_full_box2.py` (0433cde4195e7fa0) on the eval tokens `D:/tanitad/cache/eval` (fp8 4x10 tokens, read-only memmap by `tokstore.py`). The choice is made here: two seeds that both clear the bar, averaged, carry less training-seed noise than either.
* **Frames before the head's first window** (f < 9: the head needs the 10 frames f - 9 .. f) and frames past the clip's end: no value, closing input 0.0 (the RR value). A frame is never filled from a later or an earlier frame.
* **Clip index**: the position of the clip in `pins/join_eval141.npz` `clip` (the s.ci the rollout uses). Table arrays `clip_index`, `frame`, `est` (m/s), one row per (clip, frame) of every eval-141 clip, NaN where the head has no window. Beside (not fed): `c148`, `c149`, `v`.
* **Vision only at inference**: the closing comes from camera tokens only; v is the measured own speed already used by form RR. No label, no lead file, no ego state reaches the head.

## 3. Runs (box 2, RH_s0..s2, readout K, reader seeds 0, 1, 0, heading bound 0.2, rule every 1, eval-141)

| run | grid | pre-roll | closing table | partner |
|---|---|---|---|---|
| `H_RQ50KVz_s0` (control t2) | grid141, 2,399 windows | none | zero table (0.0 on every row) | `H_RQ50KE1_s0` |
| `H_RQ50KVz_P30_s0` (control t3) | grid_p30, 1,271 windows | 30 steps | zero table | `H_RQ50KE1_P30_s0` |
| `H_RQ50KV1_s0..s2` | grid141 | none | head table | `H_RQ50KE1_s0..s2` (row 211) |
| `H_RQ50KV1_P30_s0..s2` | grid_p30 | 30 steps | head table | `H_RQ50KE1_P30_s0..s2` (row 213) |

Argv of each new run equals that of its partner except: the wrapper (rzgfhqet in place of rzgfh / rzgfhq), `--form RRe`, `--closing-table <table>`, and the names. Order: t2, t3 first (a failed one stops the queue before any arm), then m = 0, 1, 2 with V1 then V1_P30.

**Conditional arm (Rule Zero, committed now)**: if the primary name is `-U-NOT-SEPARATED` or `-U-HIGHER`, the label-closing ceiling `H_RQ50KC1_s0..s2` (form RRc of the same wrapper: reader presence and gap, closing input from the labels, rule every 1, grid141; a privileged diagnostic, never a deployable form) runs next on box 2 within this card, if D: on box 2 has at least 1.75 GB free; else D: space on box 2 is the named blocker. Its read: `H_RQ50KC1 - H_RQ50KE1` (does any closing input lower U in the reader form) and `H_RQ50KV1 - H_RQ50KC1` (how far the head sits below it), same estimator.

## 4. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* **t1 table**: the head columns reproduce row 217's eval-141 closing r through `read_quantities` with the labels of `train_full_box2.py` (`lab_eval`): S of `c148` 0.4256 and of `c149` 0.4446, each within 0.002, n_S 3,430 for both; both checkpoints' sha256 equal the values above; the table's `v` equals the clip file's `v`; every clip of the join has its rows; `est` is NaN exactly at frames 0 .. 8 and finite at frames 9 .. T - 1. The S of the mean column is printed beside (not judged).
* **t2, t3 zero tables**: every dump file of the zero-table run equals the partner's dump file array for array (largest difference 0 on every array the partner has), and the same for the rule log and the later-look log on their common keys; the run line reports the table's sha256.
* **t4 runs**: done files, 141 dump files with finite plans, rule logs, later-look logs with q = 1 .. 9, the run line `est looks ...` with the head table's sha256; argv equal to the queue row and to the partner's argv apart from the items of section 3; partner folders on box 2 equal the pulled copies by sha256.
* **t5 fed values**: on every rule-log and later-look row with `est_ok`, `c_est` equals the head table at (ci, frame of the look) within 1e-9; `est_ok` is True exactly where the table holds a finite value; the share of present looks with `est_ok` is reported.
* the landed readers' controls (headway, families, route goal), as in cards ADD145 and ADD147.

## 5. Reads and names (landed readers unchanged; paired episode-cluster bootstrap pooled over the three model seeds; it answers another draw of episodes, not another training or inference run)

* **V1** (primary): `H_RQ50KV1 - H_RQ50KE1` on grid141 with the read of card ADD145 (`read_rulefc10k.py` engines), bar: spread of `H_RQ50KE1` over the seeds. Headway U (primary), contact C, F, and the four families (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC with route goal), each with tier, estimator and interval. Name `HEADCLOSE10K-U-LOWER` if the pooled interval of U lies below zero, `-U-HIGHER` if above, `-U-NOT-SEPARATED` otherwise.
* **V1P30**: `H_RQ50KV1_P30 - H_RQ50KE1_P30` with the age read of card ADD147 (`read_rhk10k_am2.py` engines), bar: spread of `H_RQ50KE1_P30`. Name `HEADCLOSE10K-P30-U-LOWER / -U-HIGHER / -U-NOT-SEPARATED`; C, F and the four families beside.
* **Consequence committed**:
  * V1 U LOWER, no family WORSE in V1 or V1P30, C not HIGHER in either, and V1P30 U not HIGHER: the head closing (mean of the two seeds) is the closing input of the candidate's reader form (specification section 1, the rule line, via the other session).
  * V1 U LOWER with a family WORSE or C HIGHER in either pair, or V1P30 U HIGHER: to the PI with its cost.
  * V1 U NOT-SEPARATED or HIGHER: the reader form keeps closing zero; the conditional ceiling arm of section 3 runs; if the ceiling lowers U and the head does not, the head's closing accuracy is the named lever (next: a longer-history or wider head on the same 3,984 clips); if the ceiling does not lower U either, the closing input is not a lever of the reader form at this presence and gap, and the reader's gap and presence are the named next lever.
* **Beside clause** (row 218 carries it): row 217's data gain in closing r is 2.52x the 1,500-clip seed spread on held-out clips but 0.91x on eval-141 (spread 0.129) (`TIMES_NOTE.md` 2026-11-25T23:23Z).

## 6. Cost (stated before running)

Box 2: the table, 141 clips x 2 heads, T0, about 5 to 10 min; 8 rollouts (2 controls + 6 arms; cards ADD145 and ADD147 took 17 and 18 min for 3 runs each), about 50 min; the conditional ceiling 3 rollouts, about 17 min. Total about 1.0 to 1.5 h of box 2, below the 12 h line. Disk on box 2: about 280 MB of run folders (37 MB per grid141 run, 20 MB per P30 run) against 1.93 GB free and the queue's 1.5 GB floor; the zero-control runs are deleted from box 2 after their control passes (re-creatable). No training, no encode, no pod.
