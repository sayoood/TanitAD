# PREREG_V8F_DAG_REREAD: on the adopted DAG base, (A) does v8F dominate the kinematic floor P10 at T1-kin on all four families, and (B) does it beat its capacity-matched FLAT control?

Written 2026-10-27, Master Mind. Claim row `H-V8F-DAG-REREAD-1`. Zero training. Dev-box CPU only (`CUDA_VISIBLE_DEVICES=-1`).

**Why this read exists.** `PREREG_V8F_DAGGER.md` §7 read **ROBUST** on 2026-10-27 (`raw/2026-10-27-dagger-run/read/DAG_READ.json`, sha256
`3393fdd5…`). Its committed next, verbatim: *"adopt DAG as the base. Re-read T1 dominance over P10 and FLAT's committed repeat on it. If the
crawl beside rows (M1, L2) still fail, next is DAgger iteration 2 (DAG's own states), under its own read"*. This file registers those
re-reads. Each was earlier read on another arm; neither has a registered **dominance** bar (`RESULT_V8F_T1_FAMILIES.md` §4: *no better-bar was
registered*). This file sets one, before the data.

## 0. What has been seen, and what has not

* **Seen** (all DAG − DAG-0 or within-run, in `DAG_READ.json`): R-TLAT, R-TLON, R-ADE, R-T0, the tick-0 head F1, the goal-head F1, the
  per-run crawl fraction of the head reader, the off-route counts, and **S-POS@30 / 50 / 80 against CTRV within each run**. DAG's S-POS
  deltas against CTRV are separated-better at 30, 50 and 80 m at both seeds. ⚠ They enter §2's STRATEGIC rows below. They are disclosed as
  seen, and the read counts them as decided before registration.
* **Not seen by anyone:** every DAG − P10 row, every DAG − FLAT row, and every FLAT T1-kin rollout (none exists).
  * ⚠ The T1 tool writes a `paired_model_minus_floor_t1` block into each run JSON (`t1v8f_DAG*_s*_m10_causal_heads.json`). It is a
    by-product of the rollout. Nobody has read it for DAG or DAG-0, and it does not decide anything here: §2's rows come from the readers.

## 1. Arms, tier, estimator

* **DAG** = `M2b_s{0,1}_dag` (`E:/Projects/TanitAD-artifacts/v8f-dagger/`, sha256 in each `SHA256SUMS.txt`), the adopted base. Its
  decisive T1-kin dumps are banked: `v8f-dagger-t1/dump_DAG_s{0,1}_m10_causal_heads/`.
* **P10**: the kinematic floor that every v8F T1-kin read used (`fl` in the T1 tool). It reads no model, no camera and no bucket, so its
  tracks do not depend on the arm. Control C3 checks that.
* **FLAT** = `M2b_s{0,1}_flat.pt` (`E:/Projects/TanitAD-artifacts/v8f-mini-flat/`, sha256 `8496934a…` / `6fdfc9b0…`): M2b with
  `--o-interface cut` (`PREREG_V8F_FLAT.md`).
* **DAG-0** = `M2b_s{0,1}_dag0`, beside only. It has DAG's argv with `--dagger-p 0`: M2b plus the same 3,000 fine-tune steps without DAgger.
* **Tier.**
  * A and B-2 are **T1-kin**: m = 10, `--nav deadreckon --scene logged --limit causal --schedule periodic`. These are DAG's decisive
    settings (`t1v8f` block of the run JSONs). T1-kin is a self-action OPEN loop: ⛔ never "closed loop", never "drives". The lead follows
    its logged future.
  * B-1 is FLAT's registered tier: T1-class open loop on eval-141 (box 2's standard eval npz).
* **Estimator, every row:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, per seed, over the 141 episodes of eval-141
  (2,399 windows). It answers *"would another draw of EPISODES say this?"*. The training-seed question is answered by the two seeds and by F.
  The planner is deterministic, so there is no inference-seed question.
* **F** = 0.1207 m (`MODEL_REGISTRY.md` §17, M2b's two seeds); **2F = 0.2415 m**.
* **Sign:** arm − comparator; for an error row, negative = arm better.

## 2. Read A: dominance over P10 (DAG − P10, T1-kin)

**Readers, unchanged except for the arm's paths:**
* `raw/2026-10-26-t1kin-hacc/t1k_read.py` (`comps`: the canonical T1-kin rows);
* `raw/2026-10-26-t1-latcond/t1klc_read.py` (`lat_rows`: L1, L2, L3);
* `raw/2026-10-26-t1-families/t1kf_read_a1.py` (TACTICAL and DISTANCE KEEPING behind the lead, with Amendment 1's C6).
STRATEGIC comes from the T1-HEADS reducer through the DAGGER reader (`S-POS vs CTRV`, within run). P10 has no strategic layer, so CTRV is the
kinematic floor for S.

| family | rows (each must be NOT separated-worse) |
|---|---|
| PRIMARY | **ADE6** |
| LONGITUDINAL | speed MAE, along-track MAE, accel MAE, DK-VIOL, DK-HW, DK-TG |
| LATERAL | heading MAE, yaw-rate MAE, curvature MAE, cross-track MAE, L1 lateral-accel MAE, **L2** curvature MAE at GT ≥ 5.2 m/s |
| TACTICAL | TAC-LAT, TAC-LON (the 3-way agreement of the executed track with the human's) |
| STRATEGIC | S-POS@30, @50, @80 against CTRV (disclosed in §0) |

**DOMINANT at seed s** iff all three hold at s:
* **D1:** ADE6 is separated-better, with point ≤ −2F = **−0.2415 m**.
* **D2:** no row of the table is separated-worse.
* **D3:** each of the four families has at least one row that is separated-better.

**The crawl rows** (the DAGGER clause). They are reported for each seed:
* **L2** is already in D2.
* **M1** is the arm-crawl fraction: the GT-fast pairs (GT ≥ 5.2 m/s) on which the arm's own pair speed is < 5.2 m/s, as defined in
  `PREREG_V8F_HACC_MC.md` (P10's M1 is 0.0137). It is read as a paired per-window share, DAG − P10, with the same estimator.
* The crawl rows **fail** iff L2 or M1 is separated-worse at ≥ 1 seed.

**Beside (never decisive):**
* the same rows in STRICT mode (`--scene frozen --limit causal`), from 2 new DAG rollouts, about 5 min each;
* DAG-0 − P10 on the same rows;
* DAG − HACC on ADE6 and L2, if HACC's banked dumps share the windows;
* M2 (the arm's L2 share on the crawl pairs).

## 3. Read B: FLAT's committed repeat on DAG

* **B-1, FLAT's registered tier.** Reader `code/v8f_flat_read.py`, with only the arm swapped from M2b to DAG. PRIMARY: ADE6, DAG − FLAT.
  **ESTABLISHED** iff the point is ≤ −2F and the row is separated at both seeds. Its identity row (FLAT's GT-code oracle − own) must read
  exactly 0. Beside: DAG-0 − FLAT, and the banked M2b − FLAT.
  * ⚠ Prediction, stated before the data: DAG − DAG-0 at T0 reads +0.002 / +0.048 m (R-T0). B-1 is therefore expected near M2b − FLAT
    (−0.036 / −0.003 m, NOT ESTABLISHED). A pass at B-1 would be a surprise worth checking against R-T0.
* **B-2, new: T1-kin.** FLAT is rolled out with DAG's exact settings: the same tool tree (`dagt1_tree_b43c7a9a`, digest `6a4b47c8…`), the
  same inputs and the same flags, with `--ckpt` and `--tag` the only changes. That is 2 rollouts, about 5 min each. PRIMARY: T1-kin ADE6,
  DAG − FLAT. **ESTABLISHED** iff the point is ≤ −2F and the row is separated at both seeds. Beside: DAG-0 − FLAT, and the T1-kin families of
  §2 for DAG − FLAT.
* ⛔ **The named confound.** DAG received an own-state (DAgger) fine-tune that FLAT never did. B-2 ESTABLISHED therefore says *"the
  DAgger-fine-tuned hierarchy beats the flat control at T1-kin"*. It does not say *"the hierarchy beats flat"*. Attribution needs **FLAT-DAG**
  (FLAT plus its own bank, the same recipe, both seeds). DAG-0 − FLAT, beside, shows how much of B-2 the extra 3,000 steps alone carry.

## 4. Controls (each must pass, or that read is REFUSED; fix and re-run)

* **C1, reproduction:** each reader reproduces its banked numbers on its banked arms before it reads DAG, ± 1e-4:
  * `t1k_read.py`, HACC − P10 ADE6, causal bucket: −0.7680 / −0.7336;
  * `t1kf_read_a1.py`, HACC − P10 TAC-LAT: +0.136 / +0.092;
  * `t1klc_read.py`, HACC − P10 L2: +0.0210 / +0.0082;
  * `v8f_flat_read.py`, M2b − FLAT: −0.036 / −0.003.
* **C2, identity:** (DAG, DAG) reads +0 [0, 0] on every row of every reader.
* **C3, the floor:** P10's tracks paired with DAG equal the banked floor, max |Δ| = 0.0 m at both seeds.
* **C4, same windows:** every pair is scored on the same 2,399 windows / 141 episodes, or on the row's own registered mask with its n.
* **C5, FLAT's rollouts:**
  * the argv differs from DAG's in `--ckpt` and `--tag` only;
  * the checkpoint sha256 equals the banked one before and after the rollout;
  * the tool's `model_keys_read` shows the cut interface: O reads no S or T code.

## 5. Outcomes, committed now

**Read A**

| outcome | condition | committed next |
|---|---|---|
| **DOMINANT** | D1, D2 and D3 at both seeds; crawl rows pass | the claim *"at T1-kin, v8F (DAG) is no worse than the kinematic floor on any row of the four families, better on each family, and ≥ 2F better on ADE6"* is registered. Signal the PI with B's outcome, and ask for scale compute only if B-2 is also ESTABLISHED. Evidence videos are re-rendered on DAG |
| **DOMINANT, CRAWL OPEN** | D1–D3 at both seeds; M1 separated-worse at ≥ 1 seed | as DOMINANT, and **DAgger iteration 2** (DAG's own states), under its own PREREG (the DAGGER clause) |
| **MARGIN, NOT DOMINANT** | D1 at both seeds; D2 fails at ≥ 1 seed | the failing rows are named. If they are crawl rows (L2, curvature, yaw-rate, M1): **DAgger iteration 2**. Otherwise the failing family's lever, under its own PREREG |
| **NO MARGIN** | D1 fails at ≥ 1 seed | DAG − HACC (beside) decides whether the post-map carried HACC's margin. Next: the HACC bake applied to DAG (zero training) under its own read, then iteration 2 |
| **REFUSED** | a control of §4 fails | fix and re-run |

D3 fails with D1 and D2 held means no row is worse, but some family has no row that is better. That reads **MARGIN, NOT DOMINANT**, with the
family named.

**Read B**

| outcome | condition | committed next |
|---|---|---|
| **B-2 ESTABLISHED** | B-2 PRIMARY ≤ −2F, separated at both seeds | **FLAT-DAG** on box 2, under its own PREREG (it needs the PI's go): it attributes B-2 to the hierarchy or to DAgger |
| **B-2 NOT ESTABLISHED** | otherwise | even with the DAgger advantage, the hierarchy shows no 2F value over flat at T1-kin. The interface remains the defect (`RESULT_V8F_FLAT.md`). Next: MIXO's committed T-side arm *"on the kind KIND ranks first"*, under its own PREREG |
| B-1 | reported as ESTABLISHED / NOT ESTABLISHED at its registered tier | it decides nothing beyond its own row |
| **REFUSED** | a control of §4 fails | fix and re-run |

## 6. Reporting

* Each row is reported per seed as delta [lo, hi], separated or not, the arm means and n (windows / episodes), with its tier stamp.
* A family row that cannot be computed is reported with its reason and n. It never silently counts as passing.
* STRATEGIC is reported (not paused for v8F, PI 2026-10-17, R73).
* Package: `raw/2026-10-27-dag-reread/`. Result: `RESULT_V8F_DAG_REREAD.md`. The register row is updated in the same landing.

## Amendment 1 (`A1-2026-10-27`): control C5 read for what it is meant to show. POST-DATA, CONTROL-ONLY

**When, exactly.** The reads ran on 2026-10-27, and every row of Read A, B-1 and B-2 was computed and seen before this amendment. As written,
C5 does not pass:
* **C5-1:** the FLAT rollouts' argv differs from DAG's in `--ckpt` and `--tag`, as registered, and also in `--out-dir` and `--dump-ticks`. Those
  two were pointed into the re-read package, so that nothing would be written into DAG's banked directory (`raw/C5.json`, `argv/`).
* **C5-3** cannot be executed as written. The T1 tool's `model_keys_read` lists the INPUT keys the tool hands the model. That list is
  identical for FLAT and DAG, because the cut is inside O, not in the inputs. It never could have shown the cut.

So B-2, as written, is **REFUSED**. Read A and B-1 do not depend on C5, and their outcomes stand as computed.

**What changes, C5 only.**
* **C5-1** reads: *"every setting flag is byte-equal to DAG's. Only `--ckpt`, `--tag` and the two output-location flags `--out-dir` /
  `--dump-ticks` differ."* MEASURED: `tool_settings_equal_DAG: all equal` at both seeds. The output flags choose where files are written and
  enter no computation.
* **C5-3** reads: *"the rolled-out checkpoint is the cut model"*, shown by two facts on the same checkpoint bytes (C5-2: sha256 `8496934a…` /
  `6fdfc9b0…`, unchanged before and after):
  * its saved config has `o_interface: cut`, against `full` for DAG (`raw/C5.json`);
  * B-1's identity row, FLAT's GT-code oracle − own, reads **exactly +0.0000 [0, 0]** at both seeds on that checkpoint's eval. So O's
    output does not move when S's and T's codes are replaced by the ground truth's.

**What does not change.** Every row, bar, estimator, every other control, and the outcome tables of §5. B-2 is read once more against the same
rows, with C5 as amended.

**Disclosure.** This amendment is written after B-2's rows were seen, and they meet the bar. It is admissible only because:
* it touches a control and no row;
* both changed clauses test the same property as before (same computation, cut model), with evidence that existed before the read;
* the read as written is reported REFUSED beside the amended outcome.
