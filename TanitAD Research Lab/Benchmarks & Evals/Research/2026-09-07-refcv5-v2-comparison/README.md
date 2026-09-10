# refcv5-v2 vs refcv4b — the ONE-COMMAND comparison, proven on a control BEFORE the checkpoint exists

**Owner:** Benchmarks & Evals · **Date:** 2026-09-07 · **Tier: T1 on every number**

`refcv5-v2` is training on the A40 (step ~2,600 of 40,284 at the time of writing,
4.0 s/step ⇒ landing ≈ Monday evening Berlin). ⛔ **Nothing here touched that GPU.**
Every roll below ran on the **dev-box RTX 4060**.

---

## THE ONE COMMAND

```sh
bash "C:/Users/Admin/refcv5cmp/compare.sh" \
     --ckpt   C:/Users/Admin/refcv5v2_final/ckpt_40284_FINAL.pt \
     --config C:/Users/Admin/refcv5v2_final/config.json \
     --tag    refcv5-v2
```

That is the whole operation. It (1) re-syncs the code off the G: Drive and writes an
md5 manifest of what it synced, (2) rolls the checkpoint on the RTX 4060 over the
**141-episode / 4,823-window B1 v7.2 EVAL grid** (~30 min), (3) pairs it against the
banked refcv4b baseline dump and the banked refcv3 @40,284 prior, and (4) prints the
four-family panel and a machine-written **VERDICT** against a bar registered *before*
the checkpoint existed.

⛔ The **canonical source** of every file it runs lives in this directory's `scripts/`;
`C:/Users/Admin/refcv5cmp/` is the off-Drive working copy, because **G: cannot RUN the
stack** (Errno 22 mid-import).

Re-analysis after a landed roll costs **zero GPU**:

```sh
bash "C:/Users/Admin/refcv5cmp/compare.sh" --ckpt <same> --tag refcv5-v2 --skip-roll
```

---

## THE BAR — pre-registered, in the harness, not in a report afterwards

`refcv5_compare.py::BAR_PRIMARY`:

> **refcv5-v2 must BEAT the echo control `ha0_ext` on ADE, SEPARATED** — the paired
> delta below zero *and* the whole episode-cluster interval below zero — **and clear the
> pre-registered 10 % relative margin on every horizon slot** (`echo_gate.py`'s own
> criterion: a separated CI on a 0.3 % margin is a real but useless difference).

**Why that and not "beats refcv4b":** refcv4b @40,284 only **TIES** the do-nothing
baselines — `os − ha` −0.0021 [−0.0178, +0.0154] and `os − ha0_ext` +0.0101
[−0.0050, +0.0273], neither separated. An arm that ties a plan holding its own t0
action and curvature has not learned to drive either, whatever it does against another
model. The refcv5-v2-vs-refcv4b delta is emitted as `BAR-REFCV5V2-3`,
**INFORMATIONAL**, and is explicitly *not* a pass.

This is the same criterion the sibling compose agent pre-registered in
`…/Architecture & Inference/Research/2026-09-07-refcv5-v2-compose/README.md` §5.5 —
converged independently, and the harness now enforces §5.5's margin requirement and
§5.8's vacuity gate as well.

---

## What the harness emits

| block | what it is |
|---|---|
| `[0] INSTRUMENT` | the constant-only arm must read the **no-information value EXACTLY** — `ha0` paired against itself is zero-width, its tactical κ is exactly 0, and its curvature bias equals its MAE (κ ≡ 0). A failed check invalidates the panel. |
| `[0b] ANTI-ECHO GATE 1` | `stack/tanitad/eval/echo_gate.py` per horizon slot; **refuses** a panel missing `ha` or `ha0_ext`. ⚠️ GATE 2 / GATE 2b need a live model — named as WORK ITEMS, not skipped silently. |
| `[0c] VACUITY GATE` | per-class **recall and support** beside every decision number. One zero-violation result was MEASURED to have been bought by a `turn_left` recall of exactly 0.0000. |
| FOUR FAMILIES | LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC per arm, **never pooled**, each with its estimator and n. ADE is printed as *one row of four*, never as the result. |
| PAIRED MARGINS | `taniteval.ci.paired_episode_cluster_bootstrap` only. ⛔ `overlapping_holdout_se` is used nowhere. |
| VERDICT | PASS / FAIL / **NOT YET MEASURED**, written by the code. |

### LATERAL is read on curvature, with the floor beside it

Every curvature row prints the **straight-line floor** (`ha0`, whose κ ≡ 0 so its MAE
*is* mean|κ_gt|), the echo control's κ, and the ratio. ⛔ The estimator is **masked**:
`κ = dh / ds_mid` is singular as speed → 0, and `excluded_below_min_ds` / `min_ds_m`
are printed with every number. Dropping that mask let 43 stopped windows (4.9 %,
v0 = 0.000) publish "84× worse than a never-steer plan", which reverses to **0.64× —
better, separated** once masked. `taniteval/tests/test_four_families_curvature_analytic.py`
pins the estimator against an analytic circle (1/R) and a line (exactly 0);
**14/14 pass on the dev box**.

### Every separated cell says which variance question it answered

```
episode draw  : ANSWERED — paired episode-cluster bootstrap, cluster = episode
training run  : ⛔ NOT MEASURED — one seed per arm (H-ESTIM-SEED-1 OPEN)
inference run : CLOSED BY CONSTRUCTION (refcv4b) / ⛔ NOT MEASURED (refcv5-v2)
```

⭐ The harness **reads `--sampler` out of the run's own `config.json`**. refcv4b has no
`--sampler`, so its decoder noise is zeroed at eval (`refc.py:1599`) and it is
deterministic. **refcv5-v2 carries `--sampler ddim --w-u0 0.5`, so it is STOCHASTIC AT
INFERENCE** — a re-roll of the *same* checkpoint can move every number, and the
inference-seed question that is closed for refcv4b is **OPEN** for refcv5-v2. The
harness stamps that automatically.

A cell that is separated but whose interval reaches back to within 25 % of the point
estimate is stamped **`⛔ NOT YET MEASURED`** and never "a direction". The panel also
prints its separated fraction against the **measured 6/42 = 14.3 %** false-positive
rate of a zero-lever replicate.

### ORACLE-NAV, stamped

Both arms consume the v7.2 nav command and all 4,719 v7.2 nav records are `ego-future`.
Fair — refcv4b shares the identical input — and a first-class route input under the PI
ruling of 2026-09-04, but noiseless and perfectly timed where a real router is coarse.
⛔ Neither arm's nav may be read as a production command.
