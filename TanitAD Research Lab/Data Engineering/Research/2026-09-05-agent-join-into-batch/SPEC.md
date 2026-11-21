# SPEC — wire `obstacle.offline` into the batch so `E-AGT-HEAD` can train

**Stream:** DataFlyWheel · **Date:** 2026-09-05 · **Branch:** `agent/arch-inf-20260803`
**Closes:** refcv5 build-stream escalations **#3** (train agent density unmeasured) and
**#7** (`V3Dataset` emits no `agent_box`, so `--agents head` refuses).

## The correction this starts from

refcv5's escalation #7 said the train-corpus join *did not exist* and that only val40 was
available. **That absence claim was wrong**, and it is `CLAUDE.md`'s *"absence found at ONE
location is not absence"* in its most expensive form: the join was built **2026-08-17**, banked to
**HF `Sayood/tanitad-ph0-aug120 → joins/train2400_agents.jsonl.xz`**, and documented at
`TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-17-train-obstacle-join/`.

⚠️ Framed as the **class**, not as a criticism of the stream. Escalating was the right action:
the stream probed its own box, found only val40, and said so. The record simply had a second home
it could not see from where it stood. The durable fix is the one this package applies — **name the
second location in the code that needs it** (the `--agent-join` help text now carries the HF path
and the md5), so the next reader does not have to know.

## Deliverables, in priority order

| P | deliverable | done when |
|---|---|---|
| **P1** | the join pulled and verified | md5 `24cbdca8c3b23aafc2fb17e6bf99cf76` **and** a CONTENT verify through the real consumer (`train_p8_occupancy.JoinFileReader`): `n_records 433040 · n_clips 2308 · has_occlusion_flags True` |
| **P2** | `V3Dataset` emits the batch keys `refc_agents`/`compute_losses_v3` already consume | every key present at the consumer's dtype/shape; pre- and post-filter `n` reported |
| **P3** | the TRAIN agent density measured | agents/frame distribution + observable fraction; a verdict on whether `--agent-queries 32` is supported on train, with the measured max |
| **P4** | `E-AGT-HEAD` proven trainable | a tiny-rig run reaching a checkpoint with a detector loss that is **non-zero and moves**, plus a deliberate-regression control in which the labels are withheld and the run **REFUSES** |
| **P5** | the correction banked | `GOALS_AND_CLAIMS.md` records the class |

## Binding constraints carried into the work

* ⛔ **Never touch `tanitad-refcv3` (refcv4b, live) or Thor (refav1 Stage B).** Neither was
  contacted; all compute was dev-box **CPU** (the RTX 4060 showed foreign python compute at
  100 % util, so the GPU was left alone). `OMP_NUM_THREADS=6` on every job.
* ⛔ The stack cannot be *run* from the G: mount (`Errno 22` mid-import). Everything ran from the
  off-Drive mirror with `PYTHONPATH=C:/Users/Admin/tanitad-wt/stack;…` — verified by
  `import tanitad; print(tanitad.__file__)` on every run. **Both trees were patched**, and each
  patch verified its own write by marker plus a same-breath control.
* ⛔ `Keys.txt` read in place, token never printed, written to args, or committed.
* ⛔ The visibility filter stays where the refcv5 stream put it (`refc_agents.agent_losses`,
  default ON), **not** in the dataset — filtering at the dataset would delete the
  `filter_visible=False` deliberate-regression arm.
* ⚠️ The projection is **CYLINDRICAL** (`256×640`, `f_ref` 305.577, column linear in azimuth,
  HFOV **120°**). Nothing here re-derives a pixel; `rig_projection` / `refc_agents`' own
  `FOV_HALF_ANGLE_RAD` are used as they stand. The pinhole formula gives 92.6° and looks plausible.

## Controls committed BEFORE the numbers existed

| id | control | why |
|---|---|---|
| **C0** | the re-stated cut predicates must reproduce the **banked val40** `infield_bevbox` table exactly | the cuts are function-local upstream and had to be re-stated; a re-statement with no control is a second definition |
| **C1** | `occ` recomputed from `(cx, cy)` at hfov 120 must agree on every box | if the flag does not mean what the doc says, every in-field number is void |
| **C2** | the three headline counts re-derived, never copied from the meta | a container census passes on an empty file |
| **C4** | drop fraction at `n_queries = max` must be **exactly 0.0** | a table that does not read 0 there is computing something else |
| **K1** | every batch key present at the CONSUMER's dtype/shape | the contract is read off the call site, never invented |
| **K2** | the in-field fraction of the emitted targets must land near the corpus's own `visible_frac` **0.4106** | two independent measurements of one physical fact |
| **K4** | `agent_losses` must return a finite, non-zero total with non-zero matched `n` | a block that cannot produce a loss is not a wiring |
| **K5** | **DELIBERATE REGRESSION** — labels withheld ⇒ the run REFUSES, it does not quietly score | the exact failure the refcv5 stream caught: an arm that trained, converged and stamped `w_agent: 1.0` with the detector never supervised |

## Both outcomes committed in advance for P3

`--agent-queries 32` was chosen on **val40**, where the in-field ∩ decode-box per-frame count has
**max 24**, so 32 drops zero. The train corpus is 59× larger and its distribution was never seen.

* **If the train max ≤ 32** — 32 is confirmed and escalation #3 closes with "the val40 number
  generalises".
* **If the train max > 32** — 32 is **not** a zero-drop covering N on train, the stream's own
  sizing rule (*a covering N with ZERO drop on the FILTERED set*, explicitly chosen over "use the
  p99") is violated, and the finding is the measured drop **and the range of the nearest
  sacrificed target** — because a target dropped at 40 m is clutter and one dropped at 13 m is
  inside the braking envelope.
