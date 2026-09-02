# MM-E19 — the k=8 clip-0.5 CONTROL: what actually caused the action ratio to fall

**Date:** 2026-09-01 · **Tier:** `T0-DIAGNOSTIC` (world-model probe — ⛔ **not** a driving
claim, and not comparable to any T1 number) · **Prereg:** `Project Steering/PREREG_MM_E19_K60_HORIZON.md`

## 1. The question this control answers

MM-E19 relaunched the arm as `k60clip05p30k`, which changed **two things at once** versus the
banked incumbent `postrain30k`: the rollout horizon (`o5_k` 8 → 60) **and** the gradient clip
(1.0 → 0.5), plus it started from scratch (`init_from` null) where the incumbent started from
`distill_init.pt`. Its h1 action ratio **fell**. A fall from a three-variable change is not
attributable to any one of them — exactly the `--v2` conflation failure the programme has
already paid for once.

This control holds the horizon at the incumbent's **k=8** while keeping **clip 0.5** and
**`init_from` null**, so the two arms differ in one variable.

## 2. The one-variable proof — MEASURED, not asserted

Diffing the two arms' `config.json` (`raw/k8_config.json` vs the k60 pull) over every key:

| key | k8 control | k60 arm | note |
|---|---|---|---|
| `args.o5_k` | **8** | **60** | ⭐ the one launched difference |
| `max_horizon` | 20 | 60 | derived from `o5_k` |
| `o4.o4_n` | 415002 | 319002 | derived (window count follows the horizon) |
| `o4.o4_span_steps` | 14 | 66 | derived |
| `o4.o4_saliency_quantiles` | … | … | derived |
| `o4.o4_weight_max_over_min` | 30.98 | 15.54 | derived |

**Every non-`o5_k` entry is a downstream consequence of `o5_k`**, not an independent lever.
`clip`, `init_from`, `seed`, `cond_param`, the EMA schedule, `o13_*`, `o14_*` and the corpus are
**identical** between the two. ⇒ this is a true one-variable comparison; the incumbent
comparison never was.

⛔⛔ **AMENDED 2026-09-02 — "DERIVED" IS NOT THE SAME AS "NOT A CONFOUND", AND THIS TABLE
ORIGINALLY CONFLATED THEM.** The `o4_n` row above is labelled *"derived (window count follows
the horizon)"* and was treated as harmless for that reason. It is not. `415,002 → 319,002` is a
shortfall of **exactly 96,000 = 40 × 2,400** — the horizon delta times the corpus — so the k=60
arm trained on **76.9 % of the k=8 arm's windows, 23.1 % fewer and temporally truncated**.
⇒ the **intervention** remains one-variable and *"the fall is caused by `o5_k`"* stands; what is
**NOT** established is *which consequence of `o5_k`* produced it — the longer rollout, or the
smaller/truncated training set it forces. **This control does not separate them**: it differs on
the window-set axis too, in the same direction. Every use of the 1.71× scene rise carries that
clause until the axis is controlled (backlog **L-10**). Caught by the Research Lab
(`…/2026-09-02-scene-matched-action-criterion/COMMS.md` §E3), not by me.

## 3. Results — h1 action-divergence ratio (the prereg's PRIMARY instrument)

| arm | `o5_k` | clip | `init_from` | h1 ratio | vs incumbent |
|---|---|---|---|---|---|
| `postrain30k` (banked incumbent) | 8 | 1.0 | distill | **0.005947** | — |
| **`k8clip05p30k` (this control)** | **8** | **0.5** | null | **0.006165** | **1.04× — flat** |
| `k60clip05p30k` | 60 | 0.5 | null | *(re-measurement in flight — §6)* | |

Per-horizon detail for the control (`raw/MISLABELLED_INSIDE_AS_k60__actdiv_local.log`):

```
[k8 control]  h=1  action 0.00132  scene 0.21391  RATIO 0.0062  C0 PASS
              h=2  action 0.00001  scene 0.37561  RATIO 0.0     C0 PASS
              h=4  action 0.00001  scene 0.37561  RATIO 0.0     C0 PASS
[postrain30k] h=1  action 0.00139  scene 0.23434  RATIO 0.0059  C0 PASS
              h=2  action 0.00001  scene 0.39906  RATIO 0.0     C0 PASS
              h=4  action 0.00001  scene 0.37561→0.39906  RATIO 0.0  C0 PASS
```

⚠️ **Both arms lose the action entirely past h=1** (ratio 0.0 at h=2 and h=4, action term
1e-05). That is not a difference between arms; it is a property of both, and it is the more
important observation in this table.

### The controls that make the panel admissible

| control | required value | read |
|---|---|---|
| `constant` (no-information) | exactly 0.0000 | **+0.0000** ✅ |
| `z_t` drift (positive control) | large | **+0.6674 / t 146.85** ✅ |
| C0 (instrument fault gate) | PASS, scene_spread ≠ 0 | **PASS**, scene 0.214–0.399 ✅ |
| **incumbent reproduction** | prereg banks **0.00595** | freshly computed **0.005947** ✅ |

⭐ The last row is what makes the cross-read comparison legitimate: the incumbent was
**recomputed from scratch in this run** and landed on the banked reference exactly, so the
runner is deterministic and the two reads are on the same instrument. Without it, a difference
between reads could have been runner drift.

## 4. Verdict

**The gradient clip and the scratch init are NOT the cause.** Changing clip 1.0 → 0.5 and
dropping the distill init, with the horizon held at 8, moves the h1 ratio **1.04×** — flat.

⇒ Whatever moved the k60 arm's ratio is attributable to **`o5_k` = 60** — the setting, ⚠️ **not
necessarily "the horizon itself"** (see the amendment in §2: `o5_k` also cuts the training-window
set by 23.1 %, and this control does not separate the two consequences). This is the attribution
MM-E19 could not make on its own.

⚠️ **Direction matters and must not be smoothed over.** The prereg committed three outcomes —
`HORIZON-WORKS` (≥10× rise), `HORIZON-PARTIAL` (rise <10×), `HORIZON-INERT` (unchanged within
noise). A **fall** is off that table. The prereg did not anticipate the horizon making action
sensitivity *worse*, so the honest reading is that the committed outcome set was incomplete,
not that the result maps onto one of its rows. Recording that as a prereg defect, not as a
verdict.

## 5. Evidence classes

| claim | class |
|---|---|
| control h1 ratio 0.006165 | **MEASURED** (ours, dev-box CUDA; `raw/`) |
| the one-variable config diff | **MEASURED** (both `config.json` compared key-by-key) |
| ckpt identity (md5 `2d744d6d…`) | **MEASURED** (md5 on the pod source AND the local pull) |
| incumbent 0.005947 | **MEASURED** here; also the prereg's banked reference |
| k60 0.002983 | ⛔ **INHERITED and currently UNCITABLE** — see §6 |

## 6. ⛔ An incident this package must carry: the k60 number was nearly lost, and the k8 read was MISLABELLED

Two defects, both now fixed in `mm-e19-probes/mm_e19_read.py`.

**(a) The runner labelled this control `k60clip05p30k`.** `--arm-name` **defaulted** to
`"k60clip05p30k"`, so passing a different checkpoint via `--ckpt` produced a complete,
plausible, entirely k60-stamped result set. Every table, log and JSON in `raw/` still says
`k60clip05p30k` — which is why they are prefixed `MISLABELLED_INSIDE_AS_k60__`. Only an
**md5 of the installed checkpoint** (`2d744d6d…` = Thor's `k8clip05p30k/ckpt.pt`, versus
`7e3c776d…` for k60) proved which arm the numbers belonged to. The numbers were right and the
name was wrong — the worse failure, because it reads as a k60 re-read that moved 2.07×.
⇒ `--arm-name` is now **required**.

**(b) Installing the control DESTROYED the k60 arm's assets and its raw logs.** The install
overwrote `<assets>/v7tiny_k60clip05p30k/ckpt.pt`, and the runner writes to a **fixed** output
directory, so all four `latentmotion` logs and `actdiv_local.log` from the k60 read were
overwritten too. The banked k60 JSON stores only `cmd`/`rc`/`seconds`/`log` — **no numbers** —
so its ratio survived nowhere in the repo. It was recoverable only because a copy of the
checkpoint happened to remain in the pull directory.
⇒ the install now **refuses** when the target holds a different checkpoint (`--replace-arm` to
override), and the k60 read is being **re-measured** to restore a citable number.

**Root-cause class:** *an identifier that can silently name the wrong artifact is not an
identifier* — the same family as a filename without its corpus, or a count without its
operator. Banking a JSON that **references** external logs by path is not banking the raw data.

## 7. Deliverable manifest

| artifact | location |
|---|---|
| this result | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-01-mm-e19-k8-attribution/RESULT.md` |
| k8 raw (mislabelled inside) | `…/2026-09-01-mm-e19-k8-attribution/raw/MISLABELLED_INSIDE_AS_k60__*` |
| k8 launch config | `…/raw/k8_config.json` |
| merged read JSON | `…/raw/mm_e19_read_step30000_K8.json` |
| k8 checkpoint | Thor `/home/nvidia/v7tiny/k8clip05p30k/ckpt.pt` (md5 `2d744d6d…`); local pull `C:\Users\Admin\k8-pull\` |
| k60 checkpoint | Thor `/home/nvidia/v7tiny/k60clip05p30k/ckpt.pt` (md5 `7e3c776d…`); local `C:\Users\Admin\mm-e19-pull\` |
| runner fixes | `mm-e19-probes/mm_e19_read.py` (required `--arm-name`; install refusal) |
