# STOPPERC amendment 3: arm A's open readings, fixed by the Master Mind before any arm-A data (2026-11-12)

**Status.** Written 2026-11-12, after A's code was built and BEFORE any arm-A run, smoke or number exists. The card (`PREREG_V8F_STOPPERC.md`, sha256 `70c75176…8877`) is unchanged. It names A's head, its per-token cell BCE and a presence BCE (§5), and is silent on the points below. The build agent fixed ten readings (`stage1_A/RESULTS.md`, "Readings fixed here"). This file accepts nine of them as written, replaces reading 4 (the presence negatives), and fixes the camera table that reading 7 needs.

## 1. Presence negatives: only moving windows with no rest point (replaces reading 4)

**Agent's reading.** Every label-bearing window without an in-view rest point was a presence-0 negative. That pool included:
* windows whose rest point exists but falls outside the frame (behind the camera, past ±60°, or below the frame: the last metres of every stop);
* windows with no rest point because the ego is at HOLD or CREEP;
* windows whose lon class is unknown.

**Why it is replaced.**
* The head reads ONE frame. A car holding at a red light, or a stop line passing under the bonnet, looks like an approach to a stop.
* Labelling those frames "no rest point in view" asks the head to separate scenes that the frame cannot separate. The gradient then flows into the shared adapter, the only thing A is meant to shape.
* The size of the pool, MEASURED on the DAG-OACC training windows (`stopbal/prevalence_stopbal.json`, 306,566 windows):

| lon class (v9 id) | windows | under the agent's reading | under this reading |
|---|---|---|---|
| STOP (2), rest point in the frame | ~0.9 × 13,116 (ESTIMATED share, below) | presence 1 + cell | presence 1 + cell |
| STOP (2), rest point outside the frame | ~0.1 × 13,116 (ESTIMATED) | presence 0 | **left out** |
| HOLD (0) | 5,678 | presence 0 | **left out** |
| CREEP (1) | 4,997 | presence 0 | **left out** |
| unknown (`lon_allowed` 0) | 3,826 | presence 0 | **left out** |
| moving, no stop in the horizon (3, 4, 5, 6) | 51,563 + 52,585 + 69,396 + 86,285 | presence 0 | presence 0 |

* The agent's reading thus had about 15,800 conflicting negatives (14,501 MEASURED, plus about 1,300 ESTIMATED out-of-frame) against about 11,900 positives (0.91 × 13,116, ESTIMATED).
* The in-frame share of STOP rest points is **ESTIMATED at 0.91**, from the local 4,369-clip v9 copy (`E:/Projects/TanitAD-artifacts/v9labels`, a different release from the 4,572-clip training join) through one median camera. It is not the trainer's number. The G-LIVE probe and the selfcheck record the real masks per run.

**The rule now** (`stack/tanitad/data/v8f_stopaux.py` `stop_aux_labels`, `NEG_LON_IDS = (3, 4, 5, 6)`):
* `stop_pos` = label row AND camera row AND a finite rest point that projects into the frame (unchanged).
* `stop_has` (the presence-BCE rows) = `stop_pos` OR (label row AND camera row AND no rest point AND lon class in 3..6).
* The per-token cell term is unchanged.
* Pinned by `test_label_masks`: an out-of-frame rest point, and a no-rest-point window at HOLD, CREEP or IGNORE, are in neither mask.

## 2. The camera table: 4,572 clips, built from PhysicalAI-AV's own calibration (fixes reading 7's dependency)

* **The repo's `rig_extrinsics_train2400.json` covers 187 of the 4,512 training clips** (MEASURED against box 2's `D:\tanitad\runs\v8f_mini_full\join_train.npz` clip array: 4,572 clips, sorted-id sha256 `e9bb9ef8…cdd18c`; train = minus `--exclude 540:600` by sha12 = 4,512). The trainer's 0.99 coverage gate would refuse.
* `stage1_A/build_stopaux_extr_table.py`:
  * reuses `build_rig_extrinsics_table.py`'s row reader and output format;
  * takes the clip → chunk map from `clip_index.parquet` (306,152 clips); the builder's own selection tables hold only 261 of the join's clips;
  * refuses unless every clip resolves.
* Output `stopaux_extr_join4572.json`:
  * **4,572 / 4,572 clips** from 1,387 chunk parquets, table sha256 `77b6e8dc63d07d90548d09182fd5b8073e81079e224e8ce672dda11d2c3ae361`;
  * mount height 1.203-1.687 m (921 distinct), forward offset 1.690-2.163 m.
* Checks (MEASURED):
  * the 190 clips the old table shares are identical (max |diff| 0.0);
  * coverage of the 4,512 training clips through A's own `make_stop_aux_arrays` is 1.0.
* Launch argv: `--stop-aux-extr <box-2 path of this file>`; the gate binds its sha256.

## 3. Readings accepted as written (`stage1_A/RESULTS.md` 1-3, 5-10)
* Head: two MLP(d, d, 1); per-token cell logits on the 40 NOW-frame tokens, presence on their mean. Loss 1:1. Label from the NOW row.
* No offset (the rest point is the ego pose origin at z = 0). Cell = floor(px / 64), 4x10 only.
* W = goal / stop_aux at step 0 from the G-LIVE probe. Checkpoint cfg written only when on. The probe refuses a batch with no in-view rest point.

## 4. The NULL reuse test reads model tensors, not the file sha256 (added 2026-11-12, before any stage-1 run)
* The card (§5) reuses BAL-NULL_s1 and BAL-NULL_rep if NULL_s0's checkpoint sha256 equals BAL-NULL_s0's.
* The whole file cannot be equal by construction. `ckpt_last.pt` embeds `argv_sha`, and the argv holds `--out`. NULL_s0 cannot share BAL-NULL_s0's `--out`: the trainer would resume from BAL-NULL_s0's checkpoint in that folder.
* **Reading:** the reuse test is **bit-identity of every model tensor**, compared against BAL-NULL_s0's banked checkpoint. That checkpoint is pinned by its file sha256 `5169ea9554bf74004f801b2d594358355363ba47fe07aa8cac052d1da9aa2fdf` (`C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/BANK_MANIFEST.json`).
* Both file sha256s are recorded beside the result (`stage1_A/box2/stopperc_check.py reuse`). EQUAL ⇒ reuse BAL-NULL_s1 / _rep. DIFFERS ⇒ the chain runs NULL_s1 and NULL_rep (+80 min ESTIMATED).
* This keeps the card's intent (the same training run, bit for bit) and changes only the object compared.
