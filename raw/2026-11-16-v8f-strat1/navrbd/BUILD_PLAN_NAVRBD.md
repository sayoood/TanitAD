# BUILD_PLAN_NAVRBD (2026-11-18): what to change, gate, test and run for the RB-DIST arm

Design only. Nothing under `E:/Projects/TanitAD/stack` was written, `launch_gate.py` untouched, no pod write, no GPU. All line numbers are in the NAVRB2 work tree `F:/TanitAD-scratch/strat1/navrb2/tree/stack/` (= `4dedac45` + navrb2.patch). Card: `PREREG_V8F_NAVRBD_DRAFT.md`. Facts: `FEASIBILITY_NAVRBD.md`.
Evidence: line numbers MEASURED (read); every cost and every unrun step ESTIMATED or UNVERIFIED.

## 1. Design decision: a width lever, not a new mode value
`--nav-rb on` stays the switch (about 12 `== "on"` sites in the trainer and gates keep working). A new `--nav-rb-feat state|exitdist` (default `state`) sets the width of `e_nav_rb`: `NAV_RB_DIM = {"state": 1, "exitdist": 15}`. With `state` everything is the N1R path bit for bit (Z0). The alternative (a third mode `exitdist` in `NAV_RB_MODES`) touches every `== "on"` site; rejected.

## 2. Files to change (all new work goes in a study tree built from `e71782a3`; the MM lands it)

| # | file | change |
|---|---|---|
| 1 | NEW `stack/scripts/build_navtext_nav_rbd.py` | Writes `{split}_nav_rb` [N, 15] f32 and `{split}_clip/off/n`. Reads `cls_id, flag, dist_m, text_i, age_s, vocab` of a navtok table, `texts.json` (parse with `navtok.parse_full`, `exit_n`) and, for c13, the native tracks (the `clip_nav` arc rule in `build_navtext_nav.py`, sign -1). Columns as card section 2. Own G-CLOCK-RBD with the 1-s-ahead regression arm. Accepts a table whose split names are `train` / `eval` (the 10k goes in as `eval`) |
| 2 | NEW `stack/scripts/build_navtok_table_x.py` | Persists the navtok table for another join: wraps the unedited `navtok/build_table.build` (as `probe/count_rb10k.py` does in memory) over a join and the parquet rows of its clips; writes `{train,eval}_{clip,off,n,cls_id,dist_m,flag,text_i,age_s}` + `vocab` + a report with sha256. Needed for the 10k (`join_X.npz`, `navigation.parquet`) |
| 3 | `tanitad/models/v8f_mini.py` | `NAV_RB_FEATS = ("state","exitdist")`, `NAV_RB_DIM`; cfg field `nav_rb_feat: str = "state"` beside `nav_rb` (`:266`), validated beside `:307-308`; `Linear(NAV_RB_DIM[cfg.nav_rb_feat], d, bias=False)` at `:912`; `built_levers()` adds `nav_rb_feat` only when `exitdist` (the key is absent for `state`, so every earlier arm's levers are the tip's; `:1395-1396`). `_nav_tok` (`:918-921`) is unchanged (a Linear of any width) |
| 4 | `tanitad/data/v8f_mini_data.py` | `NAV_RB_VARIANTS` gains `shuffle` (`:372`). `nav_rb_arrays(..., feat)` (`:375-391`): expected width from the feat, not `(N, 1)`; values check {0,1} only for `state`; for `exitdist` check c0 in {0,1}, one-hot blocks in {0,1} with at most one 1 per block, c13 in [0, 3], c14 in {0,1}, and **every column is 0 where c0 is 0**. `shuffle` = a seeded clip-level derangement (rows resampled to the receiving clip's length), record `n_on`, the donor map sha256 |
| 5 | `scripts/v8f_mini_train.py` | flag `--nav-rb-feat` (`:218-225`), flag-class entry `"nav_rb_feat": "lever:nav_rb_feat"` (`:122-124`), argparse default `state`; `make_config` passes it (`:470`); `bad_navrb` (`:1773-1815`): declared == built width, `--nav-rb off` with a non-default feat refused (`:1810`), `info.nav_rb.feat`; the line that loads the array passes the feat (`:3488-3489`); `navrb_probe` (`:1820-1860`): the planted-ones gradient plus **per-column-group gradients** on the real batch (c0, c1-6, c7-12, c13, c14) with the batch's own non-zero counts, and the zero-control exactly 0.0 per group; the ckpt blob and cfg round-trip carry the feat (`:3967-3968`); `declared_dead` rule (`:2027-2028`) unchanged |
| 6 | `scripts/strat1_check.py` | kinds `N1D` (base `N1`) and `N1Dx` (base `N1`, variant shuffle) in `KINDS`, `RB_BASE` (`:35-37`); `rb_gate_problems` (`:147`): the six flags named for D kinds, `levers.nav_rb_feat == exitdist`, `info.nav_rb.feat`, the per-group gradient clauses, `n_on` for each group > 0 on the real batch; an R kind (N1R, S1R) refuses `--nav-rb-feat exitdist` and a D kind refuses a missing one; `rb_report_problems` (`:427`) for the new report (regression arm FAILS); `DATA_FLAGS_RB` (`:304`) unchanged (`--nav-rb-npz`); the N1D / N1R pairing rule: argv equals N1R's plus exactly the new flag and a new npz and sha |
| 7 | NEW `stack/scripts/v8f_t1_eval_rbd.py` | Study wrapper over `v8f_t1_eval_rb.py` (`:66-100`): `RBWindowState` subclass that holds c0-c12, c14 from t0 and, per tick, sets `c13 = clip(c13_t0 - arc[j]/100, 0, 3)` using the base class's own `self.arc` (`v8f_t1_eval.py:175,215-219`); flag off = nothing patched; `--nav-rb-feat` refused unless equal to the checkpoint cfg; `--nav-rb-variant follow|real|zero|shuffle`. The pinned `v8f_t1_eval.py` (sha `180044eb`) and `v8f_mini_eval.py` (sha `008f48b7`) stay byte-untouched |
| 8 | NEW `stack/scripts/v8f_mini_eval_rbd.py` | T0 counterpart (pattern `v8f_mini_eval_rb.py`) for the STRATEGIC guard cells |
| 9 | study readers (NOT stack): `F:/TanitAD-scratch/strat1/navrbd/read/` | `navrbd_read.py` reusing the pinned STRAT1 reader family unedited (as `navrb2_lib.py`), with (a) a held-out mode (join, grid, labels, tracks from the 10k inputs), (b) `ho_floor.py`: computes F_HO from the two banked N1R dumps and writes `HO_FLOOR.json` + sha256, (c) K1 / K-CONST / K-RE / Z-USE, (d) the outcome table of card section 7 |
| 10 | NEW held-out inputs (pod, study dir only) | `navtok_table_x.npz` (file 2), `navtext_nav_x.npz` (the text-nav eval-split layout via `build_navtext_nav.py` over the 10k table and `native_tracks_datascale10k.pt`, as the N1 arm's eval needs `--nav-text-npz`), `navtext_nav_rbd_x.npz` (file 1), `eval_grid_x.json` (RB grid windows: stride 4, `t = 3 mod 4`, state at NOW, label present = 7,594 windows / 545 episodes, order = sorted clip id) |

## 3. What the MM must add to the gate (not edited by this agent)
`launch_gate.py` has no v8f-mini profile (DSCALE10K card section 0 item 6); the study checker `strat1_check.py` IS the gate for this family. So the MM-side additions are:
1. **G-DVB entry** for `--nav-rb-feat`: lever `nav_rb_feat` in the flag-class table (`v8f_mini_train.py:122-124`) and a clause in `bad_navrb`: declared equals built width, equals the loaded array's width, equals the ckpt cfg. A new trainer flag without a G-DVB entry is refused by the gate itself (CLAUDE.md launch-gate rule 3); this closes it.
2. **G-LIVE** record fields `info.nav_rb.feat`, `navrb_probe.group_grads` (five groups), `group_n_on`.
3. **G-EVAL**: the evaluator builds the identical model for a 15-column checkpoint; a cfg that drops the feat is REFUSED (test, as NAVRB2 `geval`).
4. **G-CLOCK-RBD**: read `navtext_nav_rbd_report.json` (regression arm must FAIL on both splits, held-out counts must equal 7,594 / 545).
5. **G-CKPT**: round-trip of `e_nav_rb.weight` shape `[d, 15]` and the cfg field.
6. **G-SUITE**: pod full suite for the new commit, 0 regressions vs the tip (`navrb2_gsuite.sh` pattern); `test_launch_gate*`-reading tests re-run if any pin or profile fingerprint moves (memory: a pin edit moves 6 fingerprints in 3 tests).
7. **Token**: `nav_rb_report_sha256` for D kinds; the HO floor's sha256 is bound in the card, not in the token.
8. **Zero-grad post gate** (`zerograd_gate.py`, `--control 'e_nav_rb\.weight'`, as NAVRB2 `zerograd_N1R_s0.log`: `ok (1 trained matches)`), plus the **per-column check** from the Adam state: the gate reads parameter-level entries only, so the per-column proof comes from `navrb_probe.group_grads` at G-LIVE and from the post-run weight: each column group's weight norm moved off 0.0 (a group whose column was never non-zero in training must read exactly 0.0 and is named, not failed, in the report).

## 4. Tests and mutation arms (CPU, `CUDA_VISIBLE_DEVICES=-1`; run the WHOLE test file, not `-k`)
New `test_v8f_navrbd.py`, `test_strat1_navrbd_check.py`, `test_v8f_navrbd_eval.py`; every one asserts `tanitad.__file__` is the study tree.
* **Z0** snapshot: `state` feat vs `e71782a3` identical (every key, value, loss term, output). **Z1** step-0 identity of the exitdist build. **Z2** gradient per group, zero control exactly 0.0.
* Builder: columns recomputed independently; **future-row test** (rows after j replaced by garbage leave c0..c14 unchanged); exit-last is backward only; c13 dead-reckoning equals the `clip_nav` rule on a fixture; every column 0 where c0 is 0.
* Loader: shape, one-hot, range refusals; `shuffle` derangement has no fixed point and equal clip lengths preserved.
* Evaluator wrapper: **G-EVAL identity** at batch 32 / 64 / 256 (a model with `e_nav_rb.weight` exactly 0.0 gives bit-equal `forward_infer` to the channel-off weights), plus `navrb_geval_identity.py pair|compare` on a real N1D checkpoint through the real evaluators (the NAVRB2 plan's step 3b, run by the MM before any read; NAVRB2's own pair was run only after the chain). A test shows c13 falls by `arc/100` per tick and c0..c12 stay at t0.
* **Mutation arms (each must turn a named test RED):** delete the `e_nav_rb` term in `_nav_tok`; delete the join line (`:3488-3489`); drop `nav_rb` from the window item (`:798-799`); read row `j + 5`; delete the exit-last read; delete the c13 arc subtraction (train and eval separately); `--nav-rb-feat` default flipped to `exitdist`; feat omitted from `make_config`; feat dropped from the ckpt cfg; checker without the group-gradient clause; checker without the regression-arm clause; the shuffle variant replaced by identity.
* Regression: `test_v8f_navrb2*`, `test_strat1_*`, `test_v8f_strat1`, `test_v8f_dagger*`, every `test_v8f_mini*` and every launch_gate-reading test, serially with the commit-headroom wait (memory: one pytest at a time; commit headroom >= 5.5 GB), and the pod G-SUITE as the binding full-suite comparison.
* Run `test_text_encoding_is_explicit` before landing (memory: `text=True` without `encoding=` fails the Thor precheck); keep repo paths <= 200 chars; no file named `aux`.

## 5. Order of operations (what runs when)
0. **Now, no build needed (CPU, zero-training):** (a) pull the banked N1R_s0 / N1R_s1 checkpoints from HF; (b) build the 10k table, text-nav and grid (files 1, 2, 10; items 2 and 10 need only the table builders and `build_navtext_nav.py` unedited); (c) dump both N1R checkpoints on the 7,594 held-out windows on the pod, CPU, one thread count (note: the N1R checkpoints need the NAVRB2 `v8f_t1_eval_rb.py`, whose 1-column `nav_rb` table for the 10k is the existing builder over the 10k table); (d) compute `F_HO` and write `HO_FLOOR.json` + sha256. This also gives the first powered read of how N1R's TAC-LAT behaves on RB windows (no new lever, no PI cost).
1. Freeze the card with `F_HO` in it. 2. Build, tests, mutation arms, G-SUITE. 3. Mint tokens, chain.

## 6. Pod chain shape (LoRA pod; staged, not launched; no address or port in any file)
```
cd /root/strat1/navrbd || exit 1            # first command, study root
exec flock /root/gpu_queue.lock bash /root/strat1/navrbd/navrbd_chain.sh
```
`navrbd_chain.sh` (pattern `navrb2/pod/navrb2_chain.sh`): `COMMIT=__MM_FILLS__` refused until set; per arm: `strat1_check.py mint` (token) -> `run` (refuses without the token) -> `post` -> `zerograd_gate.py` -> `ZZ ARM <tag> DONE ZZ`; arm order N1D_s0, N1D_s1, N1Dx_s0; a STOP file at `/root/strat1/chains/STOP` ends it between arms; ends `ZZ STRAT1 NAVRBD DONE|FAIL ZZ`. Queue script waits for the previous chain's DONE / FAIL line (`queue_navrbd.sh` pattern) and `exec`s under the flock. Before launch: the pod tree is `cp -a` of the base tree + the new files, `sha256sum -c` against a tree list; inputs (the train-side `navtext_nav_rbd.npz`, the held-out inputs) sha-checked; a GPU-idle watcher on the pod. Do not touch `/root/dscale10k/blockum` or any running job.

Held-out dumps (CPU, pod, after the chain, one thread count for all five dumps incl. the two N1R floor dumps): `v8f_t1_eval_rbd.py` per checkpoint with `--eval-cache /root/data/cache_x_dscale/train`, `--eval-grid eval_grid_x.json`, `--eval-join /root/dscale10k/merged/join_X.npz`, `--tracks native_tracks_datascale10k.pt`, `--nav-text-npz navtext_nav_x.npz`, `--nav-rb-npz`, `--dump-ticks --dump-heads`. Memory headroom: the pod container is capped (memory: 50 GB cgroup on another pod; read `memory.max` on this one before launching); run one dump at a time.

## 7. Cost
| item | amount | class |
|---|---|---|
| 3 training arms (N1D_s0, N1D_s1, N1Dx_s0) | **about 0.75 pod-h** at 0.25 pod-h per arm measured on this pod for NAVRB2's four arms (12-16 min wall each including mint, post and gate) | MEASURED rate, ESTIMATED total |
| same, in the card's unit | NAVRB2 quoted 0.16 L40-h per arm (0.19 s/step x 3,000 steps, INHERITED from `strat1_chain2.sh:2`); 3 arms = about 0.48 L40-h on that basis | ESTIMATED |
| conversion used | none measured between this pod's RTX 4000 Ada and an L40. Quote **pod-hours as measured** and the card's 0.16 L40-h per arm as the inherited unit; do not multiply one into the other. The only A40 to L40 factor on file is 0.863 (A40-h x 0.863, P6 rate, ESTIMATED) and it is not a Ada factor | stated |
| held-out dumps (5 arms x 7,594 windows) | about 1.5 h CPU on the pod at the dev-box rate of 0.14 s per window (2 threads); less with more threads (U) | ESTIMATED |
| eval-141 guard dumps (2 N1D arms, dev box, 5.5 min each) | about 0.2 h CPU | ESTIMATED from STRAT1 |
| table and input builds | minutes, CPU | ESTIMATED |
| conditional: S1D (4 arms) | about 1.0 pod-h | ESTIMATED, needs a PI go |
| G-SUITE | CPU, as `navrb2_gsuite.sh` | ESTIMATED |

## 8. Deliverable manifest (all single copies on F:, nothing staged, nothing in git)
| artifact | where |
|---|---|
| card draft | `F:/TanitAD-scratch/strat1/navrbd/PREREG_V8F_NAVRBD_DRAFT.md` |
| feasibility facts | `F:/TanitAD-scratch/strat1/navrbd/FEASIBILITY_NAVRBD.md` |
| this plan | `F:/TanitAD-scratch/strat1/navrbd/BUILD_PLAN_NAVRBD.md` |
| count probe + outputs | `.../navrbd/probe/count_rb10k.py`, `.../navrbd/out/RB10K_COUNTS.json`, `TRAIN_CONTROL_COUNTS.json` |
| pulled inputs (temp, delete refused, section 9) | `.../navrbd/in/` |

## 9. Deletions (own temp, literal paths)
`F:/TanitAD-scratch/strat1/navrbd/in/join_X.npz` (73,149,031 bytes), `.../in/navigation.parquet` (67,084,481 bytes), `.../in/merge_record.json` (4,915 bytes): pulled copies whose source (pod `/root/dscale10k/merged/`, `/root/datascale/meta/navigation/`) still exists; re-pullable. The delete was refused by the permission layer in this session, so the three files STAY in place (140,238,427 bytes in total, re-pullable); the MM or the PI removes `F:/TanitAD-scratch/strat1/navrbd/in/` when the card needs no re-count. Not routed around.
