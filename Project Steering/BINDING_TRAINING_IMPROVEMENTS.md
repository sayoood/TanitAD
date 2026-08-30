# BINDING IMPROVEMENTS FOR ALL FUTURE MODEL TRAINING — the proven set

**Written** 2026-08-30 (Master Mind), at the PI's request to *"derive together proven
binding improvements we prepare for the training of all our future models."*

⛔ **ADMISSION RULE FOR THIS DOCUMENT.** A line enters as **BINDING** only if it is
MEASURED (ours, with the artifact) or PUBLISHED (primary, banked). Anything weaker goes
to §B/§C as a CANDIDATE with the experiment that would promote it. **Nothing is admitted
because it sounds right** — the whole value of this list is that a future launch can
cite it without re-deriving it.

---

## §A — BINDING NOW (evidence in hand)

### A1. Step count is chosen by CONVERGENCE, never by an epoch target

The v6 sampler draws **i.i.d. with replacement** (`train_v6_staged.py:5217-5218` to
`train_v58f_unicycle_head.make_sampler:357-378`): random episode, then random window
within it. **There is no epoch.** `steps x batch / n_windows` is an expected-draws
ratio, not a coverage fraction.

Measured coverage (simulation of the real sampler; window coverage tracks the
coupon-collector analytic to 0.3 pp):

| corpus / config | EPISODE | WINDOW | FRAME mean | FRAME p1 |
|---|---|---|---|---|
| parity, batch 8 | **100.00 %** | 44.00 % | 98.18 % | 93.84 % |
| B1, batch 8 | **100.00 %** | 25.43 % | 96.41 % | 86.93 % |
| B1, batch 16 | **100.00 %** | 44.26 % | 98.19 % | 93.47 % |

⇒ **Raising steps "to reach one epoch" is forbidden** — the premise is false and the
cost is GPU-days. *(Origin: MM-C8, my own retracted "0.29 epochs".)*

### A2. Every coverage number carries its UNIT

Episode / window / frame differ by **~4x** on stride-1 windows, and a naive division
reports the **least meaningful** one: 25 % window coverage coexists with 96 % frame
coverage, because consecutive windows share all but one frame. **A coverage claim
without its unit is inadmissible.**

### A3. Batch size is NOT a free coverage lever on Thor

Thor's throughput is flat at **12.3-14.1 windows/s across a 6x batch range** (the 20 SMs
saturate at batch 8). The unit is *windows/s*, verified two ways — `rdw8p30k` ran 30,000
steps at batch 8 in 29,758 s = 0.992 s/step = 8.1 windows/s, i.e. windows/s is *defined*
as batch / step_time.

⇒ Flat windows/s means **step time scales linearly with batch**: batch 16 is **2x
wall-clock**, and **at fixed GPU-time batch size does not change coverage at all**
(coverage depends on `steps x batch`, which flat throughput pins to GPU-time). The
honest framing of any batch increase is *"2x the GPU-time for +X pp tail coverage"*,
never *"a free win"*.

⚠️ The CLAUDE.md wording that misled me — *"a bigger batch buys nothing and only costs
memory"* — reads as time-neutral and is exactly wrong under flat windows/s. It is true
if you already know the unit and misleading if you do not. **Proposed correction to
CLAUDE.md, for the PI: say "costs proportional wall-clock, and memory on top."**

### A4. The O4-weighted sampler does NOT starve episodes

`InteractionSampler` draws **episodes uniformly** (`v6.py:786`, verbatim *"so no episode
is starved"*) with `floor=0.25` keeping every window reachable (`v6.py:759`); measured
weight spread only **2.98x** max/min. Weighted vs uniform: window coverage within
**1.4 pp**, frame mean within **0.6 pp**, p1 down at most **2.2 pp**.

⇒ Weighting is safe to keep. ⚠️ Quote **p1, not MIN** — MIN is a single-sample extreme
over thousands of episodes and is volatile (B1 b8 read 64.8 % uniform vs 73.6 %
weighted; that ordering is noise).

### A5. ⛔ No number from a TRAINER LOG is quotable — only eval output

*("v1.6 is best-in-program" came from a trainer log and was ~10 % optimistic against
`eval_*.py`.)* This already binds; it is restated because §B1 is the fix for the gap it
leaves.

### A6. ⛔ Every arm carries the nav-command input at all three layers

PI directive 2026-08-30. Mandatory at inference and in tests. A missing nav token
**raises**; it never defaults. *(Spec: `SPEC_NAV_CONDITIONING_ALL_LAYERS.md`.)*

And because our nav is `provenance: ego-future` on **4,719 of 4,719** — an oracle — **no
capability claim from a nav arm is admissible without the shuffled-nav control reported
beside it.** Actions carry a hold-action floor; nav carries the twin.

⚠️ Nav is damped **three times** at init (a pre-existing global FiLM zero-init, plus the
conditioner's zero-init projection and 0.1 gate). ⇒ **An INERT verdict may not be
declared without reading the LEARNED GATE per layer at every checkpoint** — a gate that
grew and still shows no shuffled-nav degradation is a real finding; a gate still at its
init is an *engagement failure*, and the answer is a warmup arm, not a conclusion.

---

## §B — THE THREE GAPS THAT SHOULD BECOME BINDING (highest value first)

### B1. ⛔ There is NO val-side eval inside training

Confirmed at file:line — the save-interval block (`train_v6_staged.py:6131`) contains
only a default-off seam dump and `_save_ckpt`; the only `.eval()` calls are the VLM
(`:957`) and the EMA target deepcopy (`:1063`); the val-dataset flag is **refused at
startup**. ⇒ **We cannot see a checkpoint degrade during a run, and we select
checkpoints blind.** Every real read (drift, meanpred, absorption, T1) is a post-hoc
script on a pulled checkpoint.

**Design constraint, from A5:** the in-training eval must emit **the same quantities the
post-hoc probes emit**, or it becomes a second scale nobody can compare against — which
is precisely how the v1.6 error happened.

⭐ **And it must not be ADE-only.** The first T1 read showed the closed-loop-vs-
hold-action gap is only **~1 %** on distance metrics (ade +1.37 % / +1.25 %, fde
+0.64 % / +0.94 %) — an *action-insensitivity* signature that a distance metric barely
registers. A cheap action-sensitivity probe may be worth more per save-interval than ADE.

### B2. ⛔ Latency is reported as a MEAN; the measured risk is the TAIL

Published: under real latency distributions a system lost **-24.62 %** driving score to a
long-tail submodule (>150 ms per invocation) that fired **exactly when obstacles appeared
ahead**; **deleting it raised the score 16.94 %**. Our own numbers are means.

⇒ **Report p99 tick, not mean.** Cheap; no training required.

*(Context: our 10 Hz costs only -5.0 % DS vs 20 FPS, -7.6 % vs the 24 FPS optimum, and
buys **36 % better comfort**. The cadence is fine; the tail is unmeasured.)*

### B3. ⛔ No checkpoint SELECTION criterion exists

`--save-every 2500` on a 30k run = 12 checkpoints, and nothing but "final" chooses among
them. **B1 is the prerequisite** — selection needs a val-side signal to select on.

---

## §C — INPUT-PIPELINE ITEMS THAT BELONG IN THE SAME LIST (measured, not yet actioned)

| # | item | evidence | cost |
|---|---|---|---|
| **C1** | **Ego-motion compensate the 3-frame stack** | BEVDet4D Tab. 3: concatenating an *unaligned* previous frame is **worse than no temporal input** (NDS 39.2 to 37.6, mAVE **+70 %**); translation align mAVE -23.2 %, +rotation a further -13.6 % | shares a homography with C2 |
| **C2** | **Recovery / trajectory perturbation augmentation** | ChauffeurNet Fig. 7: the unperturbed baseline recovers **0 of 20** closed-loop situations; perturbed **20 of 20**. PilotNet's recipe is single-camera, no 3-D scene, **2 s label-correction horizon** — our exact setting | ~1-2 days, zero new GPU |
| **C3** | **`dt` as an explicit input + staleness jitter** | Zoox Tab. 1: 100 ms staleness — **exactly our frame spacing** — costs -56 % / -31 % / -78 % F1, and is **almost fully recovered** by dt-as-feature + jitter, at ~zero cost on clean data | best cost/benefit in the review |
| **C4** | **Persist the per-clip validity mask** | 73.25 % of the corpus is rig B with **8.89 %** hard-zero pixels; the mask **is computed then thrown away** (`calib.py:995`) and reaches no consumer. A fixed-location black region is a rig-identifying signal for a from-scratch ViT | sidecar; forecloses nothing |
| **C5** | Photometric augmentation **drawn once per WINDOW** | All three E2E stacks apply it; CVRL Tab. 9: breaking temporal consistency costs **11.5 points**. Our 3 frames share one patch-embed conv, so per-frame jitter injects fake illumination change *inside one token* | low |

⚠️ **C1-C3 are the ranked three.** All are input-side, all are measured, none needs a new
corpus. **C2 addresses the failure our T1 floor actually exhibits.**

⚠️ **Clarification the PI asked for, recorded because I stated it badly once:** we DO
feed three camera frames. MEASURED from a real epcache record — `n_stack = 3`, 201 stored
frames per episode, decoded into `[T, 9, H, W]` where 9 = 3 frames x RGB — plus ego data
(`poses (201,4)`, `actions (201,2)`) and the `--cond-param` channel. What is absent is a
temporal **mechanism**: those 9 channels enter a single patch-embed `Conv2d`, so the
encoder has no time axis, no temporal attention and no recurrence. Temporal structure
exists at the *predictor* level (`--window 6` timesteps). **Input: yes. Architecture: no.**
C1 is why the distinction matters.

---

## §D — WHAT THIS LIST DOES NOT CLAIM

* No entry in §C is validated **on our data** — it is PUBLISHED evidence transferred by
  argument. Each needs its own pre-registered arm before entering a scaled recipe.
* ⚠️ **A de-confounding intervention is EXPECTED to make T0 worse.** Banked primary: an
  arm **10.7x worse** on open-loop next-action MSE was **2.3x better** closed-loop,
  because history helps only once the shortcut is closed. ⇒ **Pre-register that as
  acceptable, or our own T0 gate will reject the fix.**
* The training-setup review is **1 of 5 questions answered**. Q2 (per-host throughput),
  Q3 (in-training eval), Q4 (LR/schedule) and Q5 (checkpoint policy) are open; §B is
  written from the *existence* answers, not from their designs.
