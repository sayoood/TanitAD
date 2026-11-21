# FINDING 3: the T1-kin rollout path, and where a read-out substitution would hook in (PREREG_V8F_SPEEDSUB section 6, infeasible-as-written item 1)

Agent: Sonnet, 2026-10-28. Read-only: nothing was run, nothing implemented. Sources: `E:/Projects/TanitAD-artifacts/dagorc_t1/` (`.cmd`, `.sh`, `dagorct1_lib.py`, `argv/*.tmpl`, `PINS.txt`, `NOTES.md`), `E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/` (the banked CTRL rollouts), `F:/TanitAD-scratch/dagorc_t1_out/` (the DAG-ORC chain's outputs), and the instrument. **The T1-kin instrument at the tip is `stack/scripts/v8f_t1_eval.py`, sha256 `61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397`, byte-identical (MEASURED) to the copy in the pinned tool tree `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a` (digest `6a4b47c8…`), which rolled DAG-OACC, FLATDAG, DAG2 and DAG-ORC out.** All `v8f_t1_eval.py:N` references below are to that file.

## 1. The exact rollout command (one process = one checkpoint x one limit lane x all 2,399 windows)

Rendered argv, DAG-OACC s0 (banked, `v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv`; the DAG-ORC chain renders the same template, `dagorc_t1/argv/rollout_causal.tmpl`, proven equal byte for byte on startup, `NOTES.md` "Rollout argv: NOT retyped"):

```
C:/Users/Admin/venvs/tanitad/Scripts/python.exe
E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack/scripts/v8f_t1_eval.py run
  --ckpt E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt      # sha256 67265ec1…  (s1: …/M2b_s1_dagoacc, 90b01474…)
  --tag DAGOACC_s0_m10_causal_heads
  --tracks E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt        # md5 189f4e68913c0163912a509203963f8d
  --eval-cache E:/Projects/TanitAD-artifacts/sfresh_inputs/cache                      # the eval split; the SAME cache + tracks SPEEDSUB's T0 evals read ($SF in run_speedread.sh)
  --eval-grid "<tree>/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"   # md5 34ed8ac56365eeff72e42581bde2f5c4, 2,399 windows
  --eval-join "<tree>/…/raw/join_eval141.npz"                                         # md5 532fcd644c0e0835e847b2f1cee011ad, 141 episodes
  --out-dir E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1                              # DAG-ORC: F:/TanitAD-scratch/dagorc_t1_out
  --replan-steps 10  --nav deadreckon  --scene logged  --schedule periodic  --limit causal
  --dump-ticks <out>/ticks  --dump-heads  --device cpu
```
Environment (`dagorct1_chain.sh:117`): `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=<tree>/stack` (the comment there: "OMP/MKL 8 = DAG-OACC's rollouts (same reduction order)"). The other lane is the same command with `--limit logged` (the DAG-ORC chain's K4 lane; `rollout_logged.tmpl`; the only argv difference between lanes, `NOTES.md`).

* **m = 10** (`--replan-steps 10`; must divide H = 30, `v8f_t1_eval.py:404`), so **3 ticks per window** at steps 0, 10, 20; 2,399 windows x 3 = **7,197 tick rows** (MEASURED: `ticks_DAGOACC_s0_m10_causal_heads.npz` has 7,197 rows, `tick` in {0, 1, 2}).
* **Causal bucket** = `--limit causal`: `limit` is replaced by `speed_max_bin(max finite measured ego speed over rows [j-20, j])`, logged up to t0 and own-executed after (`build_item` `:237-258`, `causal_bucket` `:261-272`, `CAUSAL_STEPS = 20` `:69`); `limit_ok` false when there is no finite past.
* **Dead-reckoned nav** = `--nav deadreckon`: the nav rows after t0 are NaN, and each executed row's route progress is `clip(nav0[3] - arc/100, 0, 3)` with token and validity held from t0 (`WindowState._finish` `:216-220`).
* **Scene** `logged`: the camera tokens are the human's frames (not re-rendered; this is why it is T1, not T2). **Ages** `periodic`: `a_t = step % 3`, `a_s = step % 6` (`ages_for` `:136-143`), i.e. (0,0), (1,4), (2,2).
* The model is built from the checkpoint's own cfg and REFUSES a non-yaw checkpoint (`cmd_run` `:505-512`); chunks of `--batch 32` windows (75 chunks); per chunk the tool rolls the model arm and the P10-floor arm (`:530-540`), so one process also contains the floor's rollout.
* Output: `t1v8f_<tag>.json`, `dump_<tag>/ep*.npz` (141 per-episode dumps with arms `cl`, `ol`, `fl`, `ha0`), `ticks/ticks_<tag>.npz` (7,197 rows incl. `t_cons_pred` [N, 26], the T head's physical cells per tick, `v0`, `age_t/age_s`, `tick`), and the chain's `done_<tag>.txt` (`wall_s`, ckpt sha before and after, tool digest).

## 2. Per-rollout time (MEASURED, `wall_s` in the done records; CPU, OMP/MKL 8, one process at a time)

| chain | wall per rollout | files |
|---|---|---|
| FLATDAG (4 rollouts) | **258, 260, 261, 261 s** (the "about 260 s" in SPEEDSUB section 6 is this) | `v8f-flatdag-t1/done_FLATDAG*_m10_causal_heads.txt` |
| DAG-OACC s0, s1 | **305, 305 s** | `v8f-dagoacc-t1/done_DAGOACC_s*_m10_causal_heads.txt` |
| DAG2 / DAG20 (8 rollouts, causal and logged) | 261 to 356 s | `v8f-dagger2-t1/done_*` |
| DAG-ORC A / REP (8 rollouts) | causal 225, 325 (A), 144, 286 (REP); logged 156, 299 (A), 143, 210 (REP) | `F:/TanitAD-scratch/dagorc_t1_out/done_DAGOR*_m10_*_heads.txt` |

A substitution rollout costs the same plus a 22->256->256->5 MLP per tick (about 75 k parameters, negligible). **Budget for a T1-kin SPEEDSUB on the two DAG-OACC bases:** `identity`, `ego`, `egoshuf` = 3 rollouts x 2 seeds x ~305 s = ~31 min at 8 threads (ESTIMATED), plus 2 more rollouts if `own` is re-run rather than reused; adding the two DAG-ORC REP bases doubles it (~1 h). At the 4-thread cap used on this box while another job runs, ESTIMATE up to 2x. ⚠️ The chains pin OMP/MKL 8 "for the same reduction order": a 4-thread `identity` run need not equal the banked `own` to the bit, and a plan difference of 1e-7 is carried through 3 ticks of a closed feedback. The S1/S5-type control (`identity` equals banked `own`) must therefore be read at the banked thread count, or carry a stated tolerance measured first.

## 3. How the five speed cells reach O inside a rollout (call chain, file:line)

`rollout()` `:399-425`, per tick n: `ages_for` -> `build_item` for every window (`:411`) -> `D.collate` (`:412`) -> `batch = {k: full[k].to(device) for k in MODEL_KEYS}` (`:413`; `full` still holds the labels, the filter drops them) -> `plan = planner(batch)` (`:414`).
`planner` = `model_planner(...)` -> `plan(batch)` `:285-299` -> **`model.forward_infer(batch)` `:287`** -> `V8FMini.forward_infer` (`v8f_mini.py:773-781`) -> `_forward(g, 0.0, False)` (`:726`):
1. `run_s` (S's checkpoints), then **`run_t`** (`:683-696`): T reads `p10f`, `nav`, `limit`, `limit_ok`, `ck_t`, the scene; `cs = h_cons(h)` is the 26 standardised constraint cells (`cons_std`), `cons_pred = cs * cons_sd + cons_mu`.
2. `code = self._code_pred(tout)` (`:653-655`) = `cat[goal_prob 22, softmax(lat) 5, softmax(lon) 7, cons_std 26]` = 60 dims. **The five speed cells are `code[:, [46, 56, 57, 58, 59]]`**, T's STANDARDISED cells (`34 + CI`; physical = `cons_pred` at cons indices 12, 22, 23, 24, 25).
3. `code_override` is None at inference (`:756-759` is the `forward_diag`-only branch); `out["code"] = code` (`:762`).
4. **`run_o(…, code)`** (`:698-712`): `self.e_code(code)` (`:704`, `nn.Linear(60, d)`, `:619`) is one conditioning token of O; O's plan is `traj60`, `traj30`.
The tool then commits the first m = 10 knots of the plan as executed motion (`commit_plan`, `:181-193`), rebuilds `p10f`, `base`, `pose`, `nav` for the new rows (`_finish` `:201-221`), and the next tick's T reads them.

## 4. Where a read-out substitution hooks in (describe only)

* **Seam A (smallest, in the tool): `model_planner.plan`, `:285-299`, replacing line `:287`.** Compute `own = model.forward_infer(batch)["code"]`; the read-out's cells `phys = readout.predict_ego_batch(batch)` from `ego_vectors(batch)` (`v8f_mini_eval_speedread.py`; it reads `EGO_READ_KEYS` = `p10f, nav, limit, limit_ok, frame_ok, a_t`, all inside `MODEL_KEYS` = `INFERENCE_INPUT_KEYS - {act, pose}`, `v8f_t1_eval.py:61-62`); `new = substitute(own, phys_to_std(phys, model.cons_mu[ci], model.cons_sd[ci]), [46, 56, 57, 58, 59])`; then `out = model.forward_diag(batch, new)` (`v8f_mini.py:828-837`, "inference exactly as `forward_infer` except O reads `code_override`", HACC post-map applied equally, `_keys_read` and the head tensors present, so `--dump-heads` keeps working). `forward_diag` costs a second forward; inlining `_forward(g, 0.0, False, code_override=new)` avoids it.
* **Seam B (no edit of the pinned tool, SPEEDSUB's style): a wrapper that patches `V8FMini.forward_infer` at class level and calls the T1 tool's `main()`**, exactly as SPEEDSUB's hook patched `V8FMini.forward_diag` (`v8f_mini_eval_speedavg.py:155-159`) and ran `v8f_mini_eval.main`. ⛔ SPEEDSUB's `SpeedReadHook` is installed on `forward_diag` and is **not reachable from the T1 tool**, which never calls `forward_diag`: it needs a `forward_infer` variant (its `__call__` logic `v8f_mini_eval_speedread.py:479-515` carries over: own code, substitute, hand O the new code). The floor arm (`floor_planner` `:301-310`) never calls the model, so a class-level patch cannot touch it.
* **Seam C (for a `gt` ceiling or control only): `:412-413`.** `full["lab_t"]["cons"]` is available there (collated labels, filtered out at `:413`); it is the logged human's cells at the item's step index, not conditioned on the rollout state. ⚠️ Which step index (`j` or `j - a_t`) `lab_t` is packed at (`v8f_mini_data.py:516`) was not verified here.
* What the hook must also do, from SPEEDSUB's design: record the used cells per (window, tick) beside `--dump-heads` (`TickRecorder` `:324-397` records T's OWN `cons_pred`, which the override does not change); an `egoshuf` source array indexed by (window, tick), made from the `ego` run's recorded cells; the `identity` control; `--max-windows N` (`:488`, evenly spaced subset, the tool's own smoke) as the **real-window reachability test that the T0 harness lacked**: ~200 windows is ~25 s per rollout (ESTIMATED from 305 s x 200/2399).

## 5. What the read-out sees at each tick (the part the PREREG has to name)

| tick | `p10f`, `nav` | `limit` | `a_t` |
|---|---|---|---|
| 0 | logged up to t0: **identical to SPEEDSUB's T0 input** (row `t0 - a_t`) | `causal`: differs from the fit; `logged`: identical | 0 (T0 cycle covers it) |
| 1, 2 | the model's OWN executed motion (rebuilt via `fk.kin_features`, dead-reckoned route) | as above | 1, 2 |

* At tick 0 the only difference from SPEEDSUB's input is the limit lane, so `--limit logged` + tick 0 is a direct bridge back to SPEEDSUB's T0 numbers (the `ol` arm of the dumps is the tick-0 plan). Ticks 1 and 2 are where the loop closes: the substituted cells change the plan, the plan changes the executed speed, the executed speed is the next read-out input. This feedback is new relative to every SPEEDSUB row.
* `p10f` carries `v` and `a` of the model's own executed motion; for a slow base (the MM's diagnosis for REP s0, T cells ~1 m/s slow) the read-out's `v`-driven part follows the slow executed speed, which is the honest T1 reading but changes what "ego" means: it is not the human's speed any more.
* The MLP was fitted on ages `a_t` in {0, 1, 2} and logged features; ticks 1-2 inputs are out of the fit's distribution by construction (not measured here; the cheapest measurement is the smoke rollout of section 4 with a recorder on `ego_vectors`).

## 6. Blockers and unknowns, named

* No blocker to writing the PREREG. Building the hook (Seam B variant, ~60 lines plus tests) and the T1 reader are work items; the readers `dagoacc_read.py` / `dag_orc_read.py` / `flatdag_read.py` read banked CTRL / A / REP dumps and are the model for a paired `ego - own` T1-kin reader (paired episode-cluster bootstrap, `taniteval/tools/t1_eval.py::analyze` via `analyze_dumps` `:460-467`).
* `own` for CTRL is banked (dumps under `E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s*_m10_causal_heads/`, ticks under `…/ticks/`); REP / A under `F:/TanitAD-scratch/dagorc_t1_out/`. E: has 2.0 GB free (MEASURED `df`); a new rollout writes ~20 MB of ticks + 141 dumps, so run outputs go to F:.
* Not verified: whether `lab_t` in the tick items is packed at `j` or `j - a_t`; thread-count bit-reproducibility of a rollout; that the DAG-ORC `K-RC 0 diff` controls (INHERITED from the MM's state note) hold at 4 threads.
