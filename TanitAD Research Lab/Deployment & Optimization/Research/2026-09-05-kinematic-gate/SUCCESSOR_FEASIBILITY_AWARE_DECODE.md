# SUCCESSOR (scoped, ⛔ NOT LAUNCHED) — a FEASIBILITY-AWARE DECODE

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-kinematic-gate/SUCCESSOR_FEASIBILITY_AWARE_DECODE.md`
Architecture & Inference FlyWheel · 2026-09-05 · one page + one experiment.
⛔ **This document scopes work. It does not start it.** No arm here has been run; the only
measurement quoted as new is the displacement frontier in §4, which is a **zero-GPU readout of
already-banked tensors** and exists to make the design decision-grade rather than speculative.

---

## 1. The defect, stated as a number and not as an intuition

MEASURED (`…/veto-only-fan-safety/raw/bank_vs_fan_feasibility.json`, 240 windows × 128
candidates, one `fan_safety.score_paths` definition throughout):

| | frozen anchor vocabulary | emitted fan | delta |
|---|---|---|---|
| `peak_g` | **0.4808 g** | **4.1131 g** | **+3.6323 g = 8.56×** |
| `envelope` violation | **1.6 %** | **88.8 %** | +87.2 pp |
| `off_reach` | 77.8 % | 10.8 % | **−67.1 pp** |
| mean \|emitted − bank\| per waypoint (2 s) | — | — | **9.29 m** |

⭐ **Read the `off_reach` row before drawing the wrong conclusion.** The vocabulary is drivable
but *wrong for the window* — it is a fixed path set that mostly does not match the ego's speed,
and 77.8 % of it is out of reach. Displacing it is the offset head's **job**. The defect is not
that the decode displaces; it is that **the displacement is paid for entirely in the friction
envelope**, at 8.56×, with no term anywhere in the decode that knows the envelope exists.

⛔ **And the two obvious fixes are already priced and already refuted:**

1. **"Add / raise a feasibility reward."** MEASURED: `feasibility ×4` moves ρ(reward, envelope)
   by **0.001**; deleting `progress` moves ρ(reward, `peak_g`) by **0.27** — a **270×** worse
   lever. The RL arm that made feasibility *worse* was carrying `feasibility: 0.5`.
2. **"Post-train the decoder with a constraint channel."** MEASURED: the veto-only arm moved
   `fan_peak_g_mean` by **−0.098 g** against a **+3.63 g** blow-up = **≈2.7 %** of the gap, and
   its T1 `ade_m` regressed **+0.0362 m, separated**. Right surface, wrong size of instrument.

⇒ The lever has to be **inside the thing that moves the waypoints**, not a scalar added after it.

## 2. Where it must go — from source, three stages not one

```
refc.py:1590   bank = self.roll_bank(...)      # refcv3: anchor_v0_cond False -> anchors[None].expand
refc.py:1640   conf0, offset = self._decode(kv, cond, x0, 0);   x = bank + offset     # classifier pass
refc.py:1676   for i in range(steps):  x_in = x + noise;  _, off = self._decode(...);  x = x_in + off
```

⚠️ `out["offset"]` is the **classifier-pass offset only** — assigned once at `:1640` and never
reassigned in the refinement loop, which adds `off`. So a penalty on `out["offset"]` reaches
**one of three** waypoint-moving stages, which is a further reason the RL stage's authority was
smaller than its trainable-parameter count suggested.

⇒ **Any feasibility mechanism must apply at EVERY pass that moves a waypoint, or it is applied
to a third of the decode.**

## 3. Three designs, in decreasing preference

### ⭐ A — CONTROL-SPACE REPARAMETERISATION (preferred)

Predict a **control residual** `(a_long, kappa | a_lat)` instead of a metre-space waypoint
offset, clamp it to the (a, κ) box **before** rolling, and integrate. The path is then
envelope-feasible **by construction**: there is no representable output that violates the
envelope, so no loss term has to be tuned against `progress`.

⭐ **The machinery already exists and is already declared.** `refc_sampler.roll_controls(u,
v_roll, horizons, control_units=…, tick=…, alat_v_floor=…)` rolls a normalised control sequence
into waypoints, and `anchor_meta.py` writes `control_units` / `kappa_cap` / `alat_v_floor` /
`ref_speed_ms` **into the anchor file** — the contract added on 2026-09-05 after the
396 g/0.31 g units incident. refcv4b's anchors already carry `control_units = alat`.
So this is a **head-shape change plus a clamp**, not new infrastructure.

- **Cost:** the decoder head's output changes from `[N, 8, 2]` waypoint offsets to `[N, S, 2]`
  control residuals; trunk, selector, tactical and route heads untouched. Trainable in the
  existing refcv3/refcv4 recipe.
- **⚠️ The failure mode to design against, named in advance:** a clamp that is too tight
  re-creates the vocabulary's own problem — the bank is drivable *and* 77.8 % off-reach. The
  clamp must therefore be **speed-dependent** (a Kamm circle at the window's own `v0`), which
  `roll_controls` already supports through `alat_v_floor`. ⛔ The arm must report `off_reach`
  beside `envelope`, or it will trade one failure for the other and call it a win.

### B — AN ENVELOPE BARRIER AT EVERY PASS (cheaper, weaker)

Apply `rewards._kinematic_feasibility`'s exceedance `ex = a_ex + k_ex` as a differentiable
barrier at `:1640` **and** at each `:1676` iteration, rather than to the final output.

- **Cost:** a loss term; no architecture change.
- **⚠️ Why it is second, not first:** it is a *soft* constraint, i.e. the same family as the
  feasibility weight that already failed. The distinction that might rescue it is real — that
  failure composed feasibility **with `progress`, the one term positively correlated with
  violation** (ρ(progress, `peak_g`) **+0.2900**, ρ(progress, `ttc_below`) **+0.4697**) — so a
  barrier with no competing term is genuinely a different object. But it is still tunable-away,
  and A is not.

### C — POST-DECODE PROJECTION (rejected as the successor)

Project each emitted path onto the feasible set before selection. Zero training. ⛔ **Rejected:
it is the gate again, one size larger.** It changes the geometry the selector ranked, so the
selection and the projection fight each other, and it leaves the decode still manufacturing 4 g
paths. It is a mitigation, and the programme already has one.

## 4. ⭐ THE CHEAPEST DISCRIMINATING EXPERIMENT — and it costs ZERO GPU

**The question A and B both rest on:** is the decode displacing **more** than the ADE it earns
requires? If yes, feasibility is recoverable inside the same decode and A is worth building. If
ADE degrades in lockstep with displacement, the friction cost is **intrinsic** to matching the
human on this vocabulary, and the successor must change the **vocabulary** instead.

**The instrument:** sweep the one quantity that separates bank from fan —

```
path(lambda) = bank + lambda * (fan - bank),    lambda in [0, 1]
```

and read `envelope` / `peak_g` / `off_reach` against `oracle_ade` and `sel_ade` at each λ. It
needs **one already-banked forward** and the checkpoint's frozen `core.decoder.anchors`, so it
is `raw/displacement_frontier.py` and it runs in seconds.

⛔ **Two controls, and the second is the one RETRACTION #30 demands.**
- **C1 (arithmetic):** λ = 1 must reproduce the banked fan exactly.
- **C2 (OBJECT):** λ = 0 must reproduce `bank_vs_fan_feasibility.json`'s **bank** rates —
  measured by a *different script* on a *different window draw*. ⚠️ C1 alone is **blind by
  construction**: λ = 1 is the emitted fan whatever the left operand is, which is exactly how
  #30's control passed while interpolating a tensor that exists at no point in the decode. The
  bank is a *fixed* path set, so `envelope` / `kamm_over` / `peak_g` are window-independent and
  must match across draws; `off_reach` is not, and is deliberately not asserted.

**The decision rule, committed here:**

| reading | verdict |
|---|---|
| `sel_ade` degrades **sub-linearly** in λ while `peak_g`/`envelope` fall steeply — i.e. a knee exists | ⭐ **BUILD A.** The decode over-shoots; a constrained parameterisation buys feasibility at a priced ADE cost |
| `sel_ade` and `peak_g` fall/rise **together with no knee** | ⛔ **DO NOT build A on this vocabulary.** The friction cost is intrinsic; the work item becomes a **v0-conditioned / larger anchor vocabulary**, so the decode has less distance to cover |
| `off_reach` rises as fast as `envelope` falls | ⛔ neither: the two failure modes are a single trade on this bank, and the vocabulary is the binding constraint |

## 4b. ⭐⭐ THE EXPERIMENT WAS RUN (0 GPU) — AND IT REDIRECTS THE DESIGN

MEASURED 2026-09-05, `raw/displacement_frontier.json`, **400 EVAL windows × 128 candidates**,
`core.decoder.anchors` read straight out of the checkpoint, one already-banked forward.

⭐ **Both controls pass, and C2 is the one that matters.** C1 (arithmetic) PASS. **C2 (OBJECT)
PASS**: at λ = 0 this script reads `peak_g` **0.480782** and `envelope` **0.015625** against
`bank_vs_fan_feasibility.json`'s **0.480782** and **0.015625** — a *different script*, on a
*different window draw* (240 vs 400). The bank is a fixed path set, so those three flags are
window-independent and must match; `off_reach` is not and is deliberately not asserted.

| λ | mean disp (m) | `fan_envelope` | `fan_peak_g` (g) | `fan_off_reach` | `sel_ade` (m) |
|---|---|---|---|---|---|
| 0.00 | 0.000 | **0.0156** | **0.4808** | **0.7822** | **3.9824** |
| 0.30 | 2.065 | 0.6782 | 0.9564 | 0.6984 | 2.8110 |
| 0.50 | 3.441 | 0.7871 | 1.5234 | 0.5863 | 2.0420 |
| 0.70 | 4.818 | 0.8417 | 2.5966 | 0.4035 | 1.2985 |
| 0.90 | 6.195 | 0.8756 | 3.9007 | 0.1838 | 0.6347 |
| 1.00 | 6.883 | 0.8879 | 4.1789 | 0.1094 | **0.4486** |

⛔ **THERE IS NO KNEE.** ADE and feasibility trade monotonically and steeply across the whole
range. Even the gentlest step off the shipped fan — λ = 0.9, giving back 10 % of the
displacement — costs `sel_ade` **0.4486 → 0.6347 (+41 %)** to buy `fan_peak_g`
**4.179 → 3.901 (−7 %)**. And `off_reach` rises as fast as `envelope` falls
(**0.109 → 0.782** against **0.888 → 0.016**).

⇒ **The committed decision rule fires on rows 2 AND 3 of §4's table:**

> *"`sel_ade` and `peak_g` fall/rise together with no knee ⇒ DO NOT build A on this vocabulary
> … the work item becomes a v0-conditioned / larger anchor vocabulary"*, and
> *"`off_reach` rises as fast as `envelope` falls ⇒ the two failure modes are a single trade on
> this bank, and the vocabulary is the binding constraint."*

⚠️ **STATE THE FUNCTION CLASS, OR THE NEGATIVE OVERREACHES.** This sweep explores ONE
one-dimensional family — the straight line between bank and fan. A trained control-space decode
is **not** confined to that segment and could in principle find paths that are both feasible and
close. The admissible conclusion is therefore: **the linear displacement family offers no free
lunch on this vocabulary, and the reason is visible in the `off_reach` column** — the bank is
**speed-blind**, so shrinking toward it is shrinking toward paths the ego cannot reach. Design A
would be fighting that, not the envelope.

⭐⭐ **⇒ THE DESIGN CHANGES: CONDITION THE VOCABULARY FIRST, THEN CONSTRAIN THE DECODE.** If the
bank were rolled at the window's own `v0`, λ → 0 would be drivable **and** on-reach, and the
trade might acquire the knee it does not have today. ⭐ **A sibling stream measured exactly this
lever from the other side, independently**: `H-EGO-LIT-4` reads T0 oracle-in-vocabulary
**4.6762 m (fixed bank) → 2.7873 m (own predicted-speed bank)**, paired
**−1.8889 [−2.7307, −1.1304] SEPARATED** — ≈ 56 % of the way to the true-`v0` leak bound
(1.2841 m) — and concludes *"the binding constraint at this rung is the SELECTION step, not the
bank."* Two streams, two instruments, one conclusion: **the v0-conditioned bank is the
prerequisite, and design A is its successor, not its substitute.**

⇒ **REVISED ORDER OF WORK:** (1) v0-conditioned anchor vocabulary (already instrumented by
`H-EGO-LIT-4`'s `--withheld-bank` switch and `roll_bank`); (2) re-run this λ frontier on the
conditioned bank and look for the knee; (3) only then design A's control-space reparameterisation,
with `off_reach` reported beside `envelope` in every arm.

## 4c. RULE ZERO's next arms — both run, one admissible, and the control that stopped the other

`CLAUDE.md`'s RULE ZERO (2026-09-05) requires that a refuted arm leave behind the NEXT arm.
§4b's diagnosis was *"the bank is speed-blind"*, so the next arm is **condition the vocabulary**.
Two forms of that were run here at **zero GPU**.

### ARM 1 — similarity scaling `bank_v0 = bank * (v0 / v_ref)` — ADMISSIBLE, and NEGATIVE

MEASURED (`raw/v0_conditioned_bank.json`, 400 windows × 128 candidates, `v_ref` = 10.0 m/s from
`refc.py:361`; C1 and C2-OBJECT both pass). At λ = 0:

| | fixed bank | scaled bank | delta |
|---|---|---|---|
| `off_reach` | 0.7822 | 0.6770 | **−0.1052** |
| `envelope` | **0.0156** | **0.2421** | **+0.2265 (15.5×)** |
| `peak_g` | 0.4808 g | 0.5498 g | +0.0690 |
| `oracle_ade` | 1.0985 m | 1.1571 m | +0.0586 |

⛔ **A similarity scale is the WRONG OPERATION, and the arithmetic says why before the data
does.** Under a spatial scale `s` on a fixed time grid, speed scales by `s` and curvature by
`1/s`, so **`lat_acc = v²κ` scales by `s`** — and this corpus needs scales up to **3.63×**, so
the friction load is multiplied by up to 3.63. It buys a little reach and pays for it in exactly
the currency the successor exists to save.

### ARM 2 — a control-space roll (constant-speed re-integration) — ⛔ INADMISSIBLE, NOT QUOTED

The right operation is `roll_bank`'s: re-integrate the anchor's own control content at the
window's speed, which changes **arc length** without multiplying the friction load. A stand-in
was implemented (`raw/v0_roll_bank_probe.py`) and it produced a **spectacular-looking** result —
at λ = 0, `off_reach` 0.7822 → 0.0000, `envelope` 0.8879 → 0.0009, `oracle_ade` 1.0985 → 0.4715.

⛔ **It is not reported as a finding, because its known-answer control FAILED.** `C3b` rolls a
**synthetic constant-speed arc**, whose answer is known exactly, and the round-trip error is
**3.17e-02 m** against a 1e-4 m bar: the integrator takes the heading of the segment containing
the current arc position and steps a full `v·dt` along it, which is first-order wrong on a curve.
`C3a` (the near-constant-speed anchor subset, 34 of 128) reads 0.75 m and also fails.
⭐ **That column would have been the headline of this stream.** Two earlier grid errors were
caught the same way and fixed (`ARM_HORIZONS` is uniform at 0.5 s only over the first four slots,
and anchor slot 0 is at t = 0.5 s rather than the origin — both are the `dt`/scope family). The
control is what separates "the vocabulary is the answer" from "my integrator is first-order".

⇒ **What would settle it, and what blocks it:** the repo's own tested integrator,
`refc_sampler.roll_controls` (present in the repo tree, **absent from the frozen clone this
stream ran in**), applied to anchors that carry a **control sequence**. refcv3's
`core.decoder.anchors` are WAYPOINTS; **refcv4b's anchors declare `control_units` via
`anchor_meta.py`**. ⇒ **The conditioning experiment belongs on refcv4b, not on refcv3**, and it
needs the un-frozen tree. That is the blocker, named, and it is not a compute blocker.

## 4d. ⛔ ESCALATION — a feasibility-aware decode ALREADY EXISTS AND IS NOT IN THE REPO

MEASURED 2026-09-05 17:15Z, by a positive assertion with a same-breath control:

```
git cat-file -e HEAD:stack/tanitad/refs/feasible_decode.py  -> does not exist in HEAD
git cat-file -e HEAD:stack/tanitad/refs/refc.py             -> PRESENT   (control)
```

while the frozen clone `C:/Users/Admin/refcv4b_repo` holds
**`stack/tanitad/refs/feasible_decode.py` (13,672 B)** and
**`stack/tests/test_feasible_decode.py` (7,791 B)**, both written minutes earlier. Its own
docstring opens on the same 8.56× measurement this document is built on and states its thesis
as *"after `project_feasible`, an envelope-violating or Kamm-violating path is not merely
penalised, it is UNREPRESENTABLE — a structural zero, not an estimate"*.

⭐ **So the successor is already being BUILT by a sibling stream, and this document's job
changes from proposing it to supplying its evidence.** Concretely, §4b and §4c hand that module
three things it needs and does not yet have:

1. **The frontier it must beat** — `raw/displacement_frontier.json`: on the *shipped* vocabulary
   there is no knee, so a projection that only shrinks displacement cannot win.
2. **The operation NOT to use** — a similarity scale multiplies `lat_acc` by `s`; it must roll,
   not scale.
3. **The guard its arms must carry** — `off_reach` reported beside `envelope`, because on this
   bank they are one trade, and a projection that fixes the envelope by shortening paths will
   read as a win while making the fan unreachable.

⛔ **This is an INTEGRATION ESCALATION, not a merge request written into a README** (the failure
mode that cost 10 days once already). The two files live on **one disk** and are absent from
git. They are **not** committed by this package, because they are a live sibling's in-progress
work and sweeping them under this stream's message is the documented git-hygiene failure. ⇒ The
owning stream must stage them, and until it does they are a single-disk deliverable.

## 5. What this successor is NOT

1. ⛔ **It is not the gate, and the gate does not substitute for it.** MEASURED: a selection
   rule cannot move a fan-level metric at all — that is a **structural zero**, not a small
   effect. The gate improves which member of a bad fan is driven; the successor is about the
   fan.
2. It is **not** a closed-loop claim. Everything here is T0/T1 open loop (PI ruling 2026-09-02).
3. It is **not** a reward-design project. §1 prices the reward lever at 270× worse than the term
   that actually rewards violation, and P1's rank probe already showed the composed reward ranks
   violating candidates **lower** (ρ = −0.5367 [−0.5581, −0.5138]) — the reward is not the defect.
4. It is **not** started by this document. The first admissible step is §4's readout, then a
   pre-registered SPEC for A with the `off_reach` guard written **before** the arm runs.
