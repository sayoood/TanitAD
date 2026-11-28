# TanitAD program report — 2026-08-16 ~01:30 UTC (03:30 Europe/Berlin)

**Branch `agent/arch-inf-20260803`, HEAD `545c98d`. Seven commits this cycle.
Suite MEASURED at this tree: `stack` 2,996 passed / 0 failed / 17 skipped / 2 xfailed ·
`taniteval` 1,042 passed.**

---

## 1. ⭐ THE PI'S QUESTION, ANSWERED: Thor vs the A40

| | marginal s/step | window |
|---|---|---|
| **Thor** (Jetson Thor) | **27.18** | 100 steps, 6300 → 6400, 3 logged points |
| **A40** | **20.46** | its own matched 100-step band |
| **ratio** | **1.329× — Thor is 32.9 % SLOWER** | |

**Remaining 23,600 steps: Thor 7.42 days · the A40 would have needed 5.59.**

The rate is **stable and that is measured**: marginal over 50 steps **27.21**, over 100 steps
**27.18**, cumulative including startup **27.32** — three statistics with different startup exposure
agreeing to **0.5 %**.

**Three traps had to be cleared to get a correct number, and each would have produced a
confident wrong one:**
1. `step_s` is a **cumulative mean**, not a per-step time (`train_v6_staged.py:1406`).
2. The A40's own lifetime figure (17.46) **understates its end-of-run rate** (20.46) because the
   `o5` rollout grows — marginal-vs-cumulative would have reported **1.56×** instead of 1.33×.
3. ⛔ Step numbers **overlap across machines** (banked A40 ends 6300, Thor resumes 6250), so a
   `step > 6250` filter computes a rate *across two machines*. Logged as **C68**.

**The resume is sound** — Thor's step-6300 loss (2.4341) sits inside the A40's own step-to-step
spread (2.145–3.296), and `o5_loss`, the slowest-varying term, is continuous. Different batches, so
this is evidence against corruption, not bit-identity; the bit-level claim is the separate strict
load (573 tensors).

### 🟥 ONE PI COMMAND — 25 % of Thor's memory bandwidth is unused

`nvpmodel` mode **1 = "120W"** (MAXN unused) · GPU core **already maxed** at 1386/1386 MHz · **EMC
3200 of 4266 MHz**, sampled 5× and pinned · **33 W of a 120 W budget** at **58 °C** · CPU 99.4 % idle.
The core is at its ceiling, so the only headroom is **memory bandwidth** — and a 336 M transformer at
batch 8 on 20 SMs with `--grad-checkpoint` ON is exactly the workload where EMC binds.

`sudo` requires a password, so I cannot run it. *(Checked before attempting — a blind `sudo` inside a
piped ssh hangs invisibly.)*

```bash
ssh tanitad-thor-wifi 'sudo jetson_clocks --store ~/jetson_clocks.before.conf && sudo jetson_clocks && cat /sys/class/devfreq/bwmgr/cur_freq'
```

⚠️ Until this runs, **27.18 s/step is "Thor as configured", not "Thor"** — quoting it as capability
would record our own power mode as the hardware's limit (**C14**'s family). Clean before-window is
banked; the straddling interval must be discarded. Script ready in `code/`.
⛔ `nvpmodel -m 0` (MAXN) **not** recommended unattended — it can prompt for a reboot.

## 2. SIX LADDER DEFECTS IN TWO DAYS — every one found by EXECUTING, none by reading

| edge | defect | state |
|---|---|---|
| S-S → S-T (backward) | S-S retrains the goal a FROZEN S-T selector consumes; **no gate existed** | fixed `dc50dbc` |
| S-W → S-T (arms) | `"goal"` had no capacity control ⇒ a win unattributable (C6 confound) | fixed `b12c190` |
| S-W → S-T (forward) | `--init-from` `strict=True` **refused the designed introduction — BOTH arms unlaunchable** | fixed `8e215b3` |
| `--resume auto` across stages | **NO STAGE CHECK AT ALL**; the only barrier was `torch.optim` for an unrelated reason | fixed `5725d95` |
| `--init-from` + `--resume` | **provenance lie** — `config.json` kept the init's md5 while the resume overwrote the model | fixed `5725d95` |
| `--init-from <fp16 snapshot>` | refused with a **400-key "geometry mismatch"** for a container never opened | fixed `5725d95` |

**Clean and now pinned:** S-T→S-S, S-S→S-J, freeze×init (with a *vacuity control* — every group
proved reachable when unfrozen), X3 isolation after `--init-from`.

⇒ **New class C70: a guard that fires for a reason unrelated to what it checks.** Distinct from C13
(*a guard that cannot fail*): C13 produces confident **silence**, C70 confident **noise** — it looks
like positive evidence the check works. The cross-stage barrier held **solely** because trainable
counts differ (S-W 240 · S-T 80 · S-S 54 · S-J 374) and was **skipped entirely** for the fp16
handover artifact.

## 3. Instruments corrected

- **κ ladder single-sourced.** TWO bare `κ ≥ 0.2` thresholds in `hierarchy.py` against a published
  **0.1/0.4**; 0.2 was published nowhere. `verdict_stable` flips **true → false** over the full
  sweep — but **SUBSTANTIAL at both 0.15 and 0.10, the gates actually used, so no live decision
  moves.** Now one constant, pinned by object identity, with an AST-based drift detector.
- **A2 count INVERTED.** *"Only 255 of 4,729 trajectory rows carry a metric"* → **REFUTED**: two
  disjoint JSON schema variants; **4,474 rows (94.6 %) carry `ade_vs_gt_m`** with zero nulls and 64
  waypoints. Usable trajectories exist for 4,474 clips, not 255.
- **Distance-keeping was reporting UNAVAILABLE while the data was present.** Two lead-block
  *containers* (`.pt` vs `.npz`); the consumer called `torch.load` unconditionally — a hard stop, not
  a degraded read. Verified reproducing `670f614` exactly (headway 30.5717 m, n 228/881).
- **DIR_YAW re-read:** v2corpus "decorative" **survives unconditionally**; v1's "weak" is **not
  established** and the word was deleted from the paper.
- **P8 ported to v6** with the geometry mismatch quantified: **7.682 % of target cells lie outside
  the camera**, rising to **51.2 % of the near band under 9.09 m** — exactly where distance-keeping
  lives. ⭐ It **pre-dates v6**, which makes it ~3.6× smaller (27.682 % on the legacy square frame).

## 4. My own errors this cycle, corrected in place

- **C68** — three attempts to measure s/step from a shared append-only log without a producer
  discriminator; one md5-of-`torch.save` identity claim (retracted, re-done per-tensor: 405 keys, 0
  differing).
- **C69** — I wrote *"no banked latents exist"* from `find -maxdepth 4` when the files sit at
  **depth 6**, then called it "two independent probes". **Two probes that share a blind spot are one
  probe.** The cost verdict was withdrawn; the REF-C route needs **no GPU at all**.
- **I commissioned duplicate work** — `obstacle.offline` was already built; I briefed it from a
  stale INTAKE that still said "blocked". That agent found a real defect instead.
- **CLAUDE.md's feature count has rotted three times** (2 → 4 → **5**). Root cause named in the file:
  a count in prose with no test pinning it to source.
- **`git ls-files --cached` is not a staging check for a MODIFIED tracked file** — it reports
  "staged" for a pre-edit blob. Rule sharpened to a blob comparison.

## 5. Streams live now (5)

| stream | state |
|---|---|
| v6F S-W on Thor | training, step ~6,400/30,000, pid 25477, GPU 97 % |
| stale-blocker INTAKE sweep + feature-count pinning test | running |
| E-WC2 σ\* — locate val40 poses, backfill, run | running |
| `bev_raster` consumer audit (2+ unaudited consumers) | running |
| v6 stage chain S-W→S-T→S-S→S-J, validated by execution | running |

## 6. Decisions for the PI

1. 🟥 **`jetson_clocks` on Thor** — one command, 25 % memory-bandwidth headroom on a 7.42-day run.
   *Default if you say nothing: it stays unrun and 27.18 stands as "as configured".*
2. 🟥 **Licensing collision** — `TANITDATASET_V1_STRATEGY.md:61` firewalls the *source dataset*
   (no-derivatives, enforced by a `PermissionError`) while `ALPAMAYO2_SUPER_ANALYSIS.md` cites
   OpenMDW-1.1 on the *model weights*. The A2 augmentation set is exactly the collision — a
   derivative of the restricted dataset produced by the permissive model, **already published**.
   Internal use unaffected under both readings.
3. 🟥 **115-clip SAM3 hole** (§11.2) needs ~30 GPU-min + a re-fuse. Must not contend with training —
   schedule at the S-W→S-T boundary.
4. ⚠️ **`fused_w120val`'s 4 silently-empty records** — correcting them re-baselines the published
   175/41/56. Flagged, not silently redone.
5. ⚠️ **S-T must run `"goal"` and `"mlp"` as an ARM PAIR.** If S-T runs `"goal"` alone the result is
   unattributable and the run cannot be re-used to answer the question afterwards.

## 7. Durability

⚠️ `--save-every 250` from 6250 puts the first new checkpoint at step **6500**. Until then every step
past 6250 exists in **GPU memory only**; 6250 itself is safe on HF. The ops loop (pid 25824) is
pushing and **verifying from the far side by size**, not from the push log.
