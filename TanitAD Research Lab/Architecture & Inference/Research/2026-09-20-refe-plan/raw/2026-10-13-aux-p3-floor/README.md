# Amendment A2's P3 floor in the S1 readout harness (2026-10-13)

Spec: `eval/PREREG_AUX_DRIVABLE_S1.md`, "Amendment A2" (A2.1 floor, A2.2 P1 floor, A2.4 floor measured before S1). Code: `eval/aux_s1_readout.py`; tests: `eval/test_aux_s1_readout.py` (**`ZZTEST_PASS 63`**, was 48; the output is `ZZTEST_aux_s1_readout.log` in this directory). Nothing in `aux_navtest_forward.py`, `refe/*` or any PREREG file was edited.

## What was built

1. **`aux_s1_readout.py perception --pred-init ARM=<head_pred npz of that arm's INITIAL head>`** (one per head arm).
   - **P3 floor.** `p3_read(..., init_sdf=)` reads the init head on the same proposals / tokens / clusters, reports its pooled AUC (`floor.init`) and the paired difference (trained - init) as `floor.trained_minus_init`. It reuses `auc_diff_read` with the **same `W`** (`boot_weights(n_logs, 10,000, 20260927)`) as head-minus-scorer. Also reported: `floor.init_minus_scorer`.
   - **Decision** (`p3_decision(auc_head, auc_scorer, floor_lo)`): thresholds unchanged (0.90 / scorer + 0.05; head < scorer => `H2_KILLED`). `H2_EVALUATED` / `H2_PROVISIONAL` additionally need the 95 % **lower bound > 0**, else `P3_AT_FLOOR`; no init prediction => `FLOOR_UNREAD`. The output also carries `decision_without_floor` and `floor_lower_bound`.
   - **R3 rename.** `UNCOMMITTED_BAND` is now `H2_PROVISIONAL` (the band is pinned by `chk_p3_decision` and the `p3_band_label_reverted` mutation).
   - **P1 floor** (A2.2). `arms.<ARM>.floor`: the init head's P1 (every band x split), P2, and the paired trained-minus-init (`head_fit_diff`). No decision; K2's 0.80 unchanged.
   - **Guard.** `align_init_pred`: the init npz must cover exactly the trained npz's `(log, token)` keys, else `FloorRefused` => `ZZAUXS1_REFUSED <ARM>: ...`, exit 3. It re-orders the init rows to the trained order. A `--pred-init` for an N arm, or for an arm not in `--pred`, exits 4.
2. **`make_init_heads.py`** (this directory). Writes `D:/Projects/TanitAD-artifacts/aux_navtest_forward/init_heads/seed{0,1,2}/aux_drv.pt` (+ `init_heads_manifest.json`) and asserts: feed on / feed off build bit-identical heads per seed; the file round-trips through `load_aux_head`; the 3 seeds differ; **the seed-0 file equals `fixture_fresh_head_seed0/aux_drv.pt` bit-exactly** (the sha256 is identical: `af19121c...`). Output ends `ZZINITHEADS_OK 3`.
3. **`upgrade_smoke32_cache.py`**: ADDITIVE. `smoke32_ctx/` was written by an older tool layout (32 ctx files + one `scorer_logits.npz`); the current `cache_complete` also wants `logits_<token>.npy`. This wrote those 32 small files from the npz and nothing else. The forward tool's identity check against the banked C3 table then read **32 of 32 bit-identical, max |d| 0.0**, so the files are right.

## Judgement calls (the spec was silent)

- `H2_KILLED` is reported as KILLED whether or not the floor was read (A2: "unchanged"). `FLOOR_UNREAD` replaces only the two outcomes that would promote H2.
- The floor test is `floor_lo > 0` strictly; a NaN bound (unreadable) gives `P3_AT_FLOOR`.
- Two pre-existing assertions had to change because they encode the pre-A2 semantics: `chk_p3_integration` (no init => now `FLOOR_UNREAD`, with `decision_without_floor == H2_EVALUATED`) and `chk_p3_decision` (label rename; thresholds identical). The other 46 are untouched and green.

## New tests (15) and the mutations that must go RED

Checks: `chk_p3_floor_rule` (pure rule), `chk_p3_floor_decisions` ((a) trained beats init => EVALUATED, (b) trained == init => `P3_AT_FLOOR` at AUC 1.0, (c) band => PROVISIONAL only if the floor holds, (d) missing init => `FLOOR_UNREAD`, kill unchanged, shape mismatch refused), `chk_p3_floor_same_draws` (the init head's scores = the scorer's => the two paired differences are identical, interval included), `chk_p3_floor_ci_bound` (est +0.083, lower bound -0.083 => at floor), `chk_floor_key_guard` ((e) subset / superset / other keys refused, permutation aligned), `chk_cli_pred_init_refusals`, plus the real-store `real_store_a2_floor_cli`.
Mutations (all RED): `a2_floor_condition_dropped`, `a2_floor_unread_is_a_pass`, **`a2_floor_diff_unpaired_weights`**, `a2_floor_bound_is_point_estimate`, `a2_floor_read_on_the_trained_head`, `a2_floor_key_guard_off`, `a2_floor_arm_check_off`, `p3_band_label_reverted`.
Lesson logged by the suite: on a regular fixture the unpaired-weights mutation stayed GREEN, because discrete AUC differences land on the same percentile atoms under two different draws. The same-draws fixture is therefore deliberately irregular (14 logs of 1-5 tokens, continuous offsets).

## Plumbing check (32 C3 tokens, ONE log: **the numbers mean nothing**; CIs are degenerate)

Head-only on CPU (`--device cpu --no-gpu-lock`, 18 s for 2 heads), init seeds 0 and 1 from `smoke32_ctx`; then `perception` end to end with seed 1's init as a stand-in "trained" H2-s0 and seed 0's init as its floor. Artifacts: `D:/Projects/TanitAD-artifacts/aux_navtest_forward/p3_floor_plumbing32/` (`perception_floor_plumbing32.json`, `control_*.json`, logs).

| INSTRUMENT reading, 32 tokens, 2,048 proposals (1,443 pass / 605 fail) | pooled P3 AUC |
|---|---|
| u400 scorer | 0.8005 |
| init head, seed 0 | **0.8245** (reproduces the A2 trigger figure) |
| init head, seed 1 | **0.6959** |
| stand-in "trained" (seed 1) minus floor (seed 0) | -0.1286, CI [-0.1286, -0.1286] (degenerate) => decision `H2_KILLED` (head < scorer) |

Controls through the same CLI: trained == init => `P3_AT_FLOOR` (diff exactly 0.0, base `H2_PROVISIONAL`); no `--pred-init` => `FLOOR_UNREAD`. CPU seed-0 vs the banked GPU `smoke32/head_pred_H2-s0.npz`: max |d logit| 1.3e-6, max |d sdf| 6.3e-6 (same tokens), so a GPU floor and a CPU floor are not bit-identical: **read the trained head and its floor on the same device.**

## Full floor, once `c3_full_ctx` holds all 1,058 tokens (`ls .../c3_full_ctx | grep -c ^ctx_` = 1058; `c3_full/progress.json`)

```
PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
ART=D:/Projects/TanitAD-artifacts/aux_navtest_forward
cd "$PKG"
# 1. head-only passes on the cached fp32 ctx (no trunk). ~0.3 s per token per head on CPU (MEASURED on 32 tokens): ~15 min for 3 heads. RUN IN THE BACKGROUND, do not wait on it.
$PY eval/aux_navtest_forward.py --out $ART/p3_floor_full/heads --ctx-cache $ART/c3_full_ctx --device cpu --no-gpu-lock \
    --heads H2-s0=$ART/init_heads/seed0/aux_drv.pt H2-s1=$ART/init_heads/seed1/aux_drv.pt H2-s2=$ART/init_heads/seed2/aux_drv.pt
#    -> head_pred_H2-s{0,1,2}.npz, identity.json (must be PASS over all 1,058)
# 2. A2.4: the floor as an INSTRUMENT PROPERTY (the init heads read as heads; the DECISION field is meaningless here, read pooled.head and its CI)
$PY eval/aux_s1_readout.py perception --labels-set navtest --dac-csv raw/p2_r2_instrument/baseline/u400_navtest_full.csv \
    --c3-table raw/2026-10-11-c3-selection-headroom/table.npz --c3-tokens raw/2026-10-11-c3-selection-headroom/tokens.json \
    --pred H2-s0=$ART/p3_floor_full/heads/head_pred_H2-s0.npz H2-s1=$ART/p3_floor_full/heads/head_pred_H2-s1.npz H2-s2=$ART/p3_floor_full/heads/head_pred_H2-s2.npz \
    --out $ART/p3_floor_full/floor_instrument.json
# 3. At S1 time, per arm: the trained head's npz as --pred, the SAME seed's init npz as --pred-init (one init file serves H1 and H2 of a seed; same device as the trained pass)
$PY eval/aux_s1_readout.py perception ... --pred H2-s0=<trained H2-s0 npz> H1-s0=<trained H1-s0 npz> \
    --pred-init H2-s0=$ART/p3_floor_full/heads/head_pred_H2-s0.npz H1-s0=$ART/p3_floor_full/heads/head_pred_H2-s0.npz --out p.json
```
Read in the JSON: `arms.<ARM>.p3.decision.decision` (+ `decision_without_floor`, `floor_lower_bound`), `p3.floor.{init, trained_minus_init}`, `arms.<ARM>.floor.p1` (the init head's P1 beside `arms.<ARM>.p1`).

## Manifest

| artifact | where | copies |
|---|---|---|
| `eval/aux_s1_readout.py`, `eval/test_aux_s1_readout.py` (edited, LF) | repo working tree, `...2026-09-20-refe-plan/eval/` | 1 (unstaged) |
| `make_init_heads.py`, `upgrade_smoke32_cache.py`, `README.md`, `ZZTEST_aux_s1_readout.log` | repo working tree, `...refe-plan/raw/2026-10-13-aux-p3-floor/` | 1 (unstaged) |
| init heads seeds 0/1/2 + `init_heads_manifest.json` | dev box only: `D:/Projects/TanitAD-artifacts/aux_navtest_forward/init_heads/` (regenerable by `make_init_heads.py`, deterministic) | 1 |
| plumbing outputs | dev box only: `D:/Projects/TanitAD-artifacts/aux_navtest_forward/p3_floor_plumbing32/` | 1 |
| 32 added `logits_<token>.npy` | dev box only: `.../aux_navtest_forward/smoke32_ctx/` (regenerable by `upgrade_smoke32_cache.py`) | 1 |
