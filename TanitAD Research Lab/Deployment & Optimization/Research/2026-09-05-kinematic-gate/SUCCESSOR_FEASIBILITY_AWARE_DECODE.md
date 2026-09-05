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
