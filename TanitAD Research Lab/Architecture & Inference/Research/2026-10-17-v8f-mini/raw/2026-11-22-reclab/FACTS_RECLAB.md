# FACTS_RECLAB: what a swapped DAgger sample carries, and what the recovery label needs

Evidence class: MEASURED from source (file:line at the code of record) and from one timing run on real bank-table records. No GPU. Nothing staged, committed or pushed.

## 0. Base commit (MM: read this first)

* The brief says "export the tip tree" (arch-inf tip `6cb0d6bf0615f59d3875c1cb5e5b7629492550ad`). **That tree does not carry the arms' code.** The RH arms' argv uses `--dagger-bank-readout causal_g1x`, `--dagger-bank-kappa 0.2`, `--detach-upper`; on the arch-inf tip `grep -c dagger-bank-readout stack/scripts/v8f_mini_train.py` = **0** (control on 97fe0a56: 1).
* `97fe0a56767320dbc5f4b823d67fb07a3209a745` is NOT an ancestor of the arch-inf tip; it is the remote tip of `agent/v8f-rollhist-cdb-20261120` (`git ls-remote origin`), the code the pod runs.
* **All line numbers below, and the whole build, are on 97fe0a56** (`F:/TanitAD-scratch/reclab/tree97/stack`). The superseded arch-inf export is listed in DELETIONS.log.

## 1. Every label tensor a swapped sample carries (item built at `v8f_mini_data.py:676-685`; the swap touches only `REBUILT` + `lab_s.rc`, `v8f_dagger.py:541-585`)

| tensor | file:line | derived from the future waypoints? | rule under `recovery_v1` |
|---|---|---|---|
| `gt` [30,2] ego-frame future waypoints | `v8f_mini_data.py:681` (built `:404 gt_future`, `:597`); loss `v8f_mini.py:1181-1183` (ade, fde), `:1190` (acc, second difference of `gt`) | **YES** | **replaced by the recovery label** (own frame) |
| `s_h` [30] human arc length NOW -> knot i | `:682` (built `:417 human_arc`, `:598`); read `v8f_mini.py:632` (LEAD gap) and `:1215` (`gt_constraints_t`, `--consistency`) | **YES** (cumulative norm of the same waypoints) | **re-derived from the recovery label**: cumsum of step norms from the origin; float32, same as `human_arc` |
| `gt_ok` bool | `:681` | validity of the 30-step future (`t + 30 <= T - 1`) | unchanged: the labeller's own condition `r + 30 <= T - 1` is the same predicate, so every record that can be labelled has `gt_ok` True |
| `lead_gap_h` [30] | `:683` (join column `lead_gap`) | no: an obstacle-join quantity at the logged rows, not a function of the ego waypoints | unchanged. Caveat: it is the gap seen from the HUMAN's position; `v8f_mini.py:632` already adds `s_h - s_pred`, so the re-derived `s_h` keeps that formula consistent |
| `lab`, `lab_t`, `lab_s` (goal_y/w, lat/lon class + allowed, `cons`, `rc`, `lead_valid`) | `:685`, `_lab_pack :660` | no: offline situation labels from the join rows | unchanged; `lab_s.rc` keeps its existing frame change `v8f_dagger.py:580` |
| `cons` targets (v_hi, d_reach, t_reach, turn times) | inside `lab` | no (join) | unchanged |
| `limit`, `limit_ok` | `:679` | no | unchanged (KEPT) |
| `launch` mask | trainer `launch_stage` `v8f_mini_train.py` (called `:3581`, BEFORE the swap) | yes, from LOGGED gt | **OFF in the RH argv** (no `--launch-intent-w`); when on it is computed from the logged label before any swap and stays so. Reported, not changed |
| `level_y`, `lead_y/has/near` | `:3583`, `:3585` | logged-state labels, BEFORE the swap | OFF in the RH argv; unchanged |
| `aux_ok` | `v8f_mini.py:1189` | T2-bank only | absent |
| band terms (`band_metrics`) | `v8f_mini.py:1186` | yes (from `gt`) | OFF in RH argv (`w_op`, `w_tac_traj` 0); would follow `gt` automatically |

**STOP condition: not triggered.** Every waypoint-derived label (`gt`, `s_h`) is a pure function of the 30 waypoints and is re-derived from the recovery label by one code path. Nothing needs approximating.

Where the swapped plan label is written: today **nowhere** (the batch keeps the human's `gt` at r' unchanged; `out.update(upd)` `v8f_dagger.py:720` carries only REBUILT + `lab_s`). The own state is re-anchored by the rigid motion that maps own NOW onto the logged NOW at r' (`reanchor_rows :163`, called `:565`), so **the sample's ego frame IS the own frame and `label_own` applies as `gt` with no further transform.** This is the line the mechanism adds.

## 2. The matcher and where eligibility is computed

* `match_route` `v8f_dagger.py:125-160`; cut order **off_lat** (`e_lat > 3.0`, `:153`) -> **off_head** (`|dpsi| > 30 deg`, `:155`) -> **end_edge** (`r > T - 31`, `:157`) -> ok. So an off-route record that is also past the edge reads off_lat/off_head, not end_edge (the labeller then returns `no_logged_future`).
* The bank stores the reason per record (`RECORD_FIELDS` `:197`). Eligibility is computed **once, at load**: `DaggerBank.__init__` `:271-275` (`ok = np.nonzero(reason == OK)`, sorted key `ci * 2^24 + r'`), read by `candidates :291`, `pick :301`, `coverage :313` and `DaggerSwap.eligible`.
* Bank record fields: `ci t0 tick j p10f pose arc kin3 r e_lat dpsi reason`.

## 3. Labeller inputs and whether the record carries them

| labeller input | source | in the record? |
|---|---|---|
| own pose (x, y, yaw), world | `rec["pose"][:, -1, :3]` float64 | yes |
| own speed | `rec["p10f"][:, -1, 0]` (the registered V1 run used exactly this, `bankio.py`; `kin3[:,0]` differs by <= 1.9e-6, `bank_table_s0.check.json`) | yes |
| matched row r' | `rec["r"]` | yes |
| reason | `rec["reason"]` | yes |
| logged grid [T,3] and logged speed [T] of the clip | NOT in the record: `ClipStore.get_kin(clip)["pose"]` columns 0-2 and column 3 (`v8f_mini_data.py:458`, mmap, no feature read; `check_track_matches_grid` compares all four columns to the native track) | **no: needs the store at bank load** (precedent: `RecoverDevTable.from_dagger_bank`, `v8f_recover.py:141`) |

## 4. Labeller wall (MEASURED, `F:/TanitAD-scratch/reclab/tmp/time_lab.py`, research V1 labeller, dev box CPU, real bank-table s0 records of the earlier DAgger bank, seed 0 sample of 1,000 each)

* **ok records: 0.281 s per 1,000** (916 emitted, 73 refused_infeasible, 11 path_exhausted).
* **off_lat / off_head records: 0.506 s per 1,000** (198 emitted, 518 refused_infeasible, 239 no_logged_future, 40 degenerate, 5 path_exhausted).
* A bank of 613,132 records has 503,380 ok+off records: **about 163 s of labeller time** (114.6 + 48.5). Clip-grid reads (4,512 clips, mmap) are additional and not measured here (pod disk); the earlier whole-bank labelling run took 57-69 s per seed for the off-route set. **Load-time cost is well under 10 min: no sidecar is needed.** Not measured: the full load on the pod's disk with the real clip store.
* Yield: of off-route records about 20 % emit; of ok records about 92 % emit (the ~8 % refused keep today's human label).
