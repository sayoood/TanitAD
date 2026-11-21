# PREREG — REF-C v4: the MEASURED ego state at t0, and the anti-echo guard that must earn it

*Architecture & Inference FlyWheel · 2026-09-03/04 · registered BEFORE any full
run. Skill: `TanitAD_ValidateAIDesign` §1. Sibling literature package:
`../2026-09-03-refc-ego-inputs-and-anti-echo/RESULT.md` (E-REFC-EGO-1).
⛔ **NO FULL RUN IS AUTHORISED BY THIS DOCUMENT.** The PI's sequence is refcv3's
eval → video → push first; the launch line in §10 waits for the Master Mind's go.*

---

## 0. The mandate, and the one sentence that unblocked it

**PI, 2026-09-03, verbatim:**

> *"I think we need to fix refcv3 and redesign to allow it to use the measured
> ego state (measured current speed, measured current acceleration, measured
> current yaw rate) and not the future one, let call it refcv4. … explore
> methods to avoid echoing the ego dynamics in refc like architectures,
> implement them, validate them."*

> *"The vision only rule is actually saying that the semantic understanding and
> the trajectory planning must be based on image frames and should avoid **echo
> of the ego dynamics** or predicting the ego trajectory blindly without
> considering the environment."*

**PI, 2026-09-02, verbatim:** *"It is allowed to use the velocity as initial
measured state at its cycle time. It is not allowed to use the future dynamic
information from the ground truth."*

⇒ The binding line is **TIME + ANTI-ECHO, not MODALITY.** Measured ego at `t0`
is admissible everywhere, including the goal and planner heads. What is refused
is (a) any **future** ego quantity and (b) a planner that **reproduces its own
dynamics** instead of reading the scene. This pre-registration exists because
(b) is not enforceable by a config flag — it has to be measured, and the
measurement has to be able to FAIL.

---

## 1. Hypotheses (registered in `Project Steering/GOALS_AND_CLAIMS.md`)

| id | hypothesis | status at registration |
|---|---|---|
| **H-ECHO-1** | The measured `t0` ego state (`v0`, `a_long`, `yaw_rate`, `curvature`) reaches the REF-C goal/tactical path (E11′) and the arm still **reads the scene** — i.e. it clears `ha0_ext` on the four families and its plan degrades when handed the wrong scene | **OPEN** |
| **H-ECHO-2** | The X15 validity bit + per-sample withholding removes the zero-collision, so a withheld ego block is distinguishable from a genuine standstill and the `brake_stop` branch becomes reachable at eval | **OPEN** |
| **H-ECHO-3** | E14 (echo-quotiented goal supervision: `g_tac = ha0_ext + zero-init residual`) makes echoing **visible from step 1** as `echo_ratio`, without itself preventing it | **OPEN** |
| **H-ECHO-4** *(the gate's own hypothesis — the one the tiny rig must settle first)* | The anti-echo gate **FAILS** an arm that echoes by construction. If it does not, every PASS it issues is vacuous | **OPEN** |

⚠️ **H-ECHO-4 is tested FIRST and gates the others.** A guard that cannot fail
is decoration (C108: `goal_admissibility` sat with zero call sites for 12 days
because nothing ever raised).

---

## 2. The one variable, arm by arm

⛔ **`TanitAD_ValidateAIDesign` §1 refuses arms differing in more than one
variable.** The panel is therefore a **LADDER**, each rung one lever:

| arm | = previous rung + | what it isolates |
|---|---|---|
| **A `v3`** | — (the incumbent at the rig rung) | the honest vision-only bound; the `state3` analogue of PlanTF Tab. II |
| **B `v4_noguard`** | `ego_state_inject=True` (+ its precondition `core.ego_valid_channel=True`), `ego_dropout 0.0` | **does the ego state help at all?** |
| **C `v4_drop`** | `ego_dropout 0.5` | **what does per-sample withholding cost/buy?** (PlanTF's rate sweep; DRAMA's 0.5) |
| **D `v4_full`** | `echo_base=True` (E14) | **does the echo quotient change the solution or only the readout?** |
| **E `regress`** | `--ablate-frames` | ⛔ **THE DELIBERATE-REGRESSION ARM.** Same v4 config, image replaced by a constant ⇒ an echo BY CONSTRUCTION. **The gate MUST fail it** |

⚠️ **`core.ego_valid_channel` travels with `ego_state_inject` and that is
declared, not hidden.** It is a **precondition** of the lever, exactly as
`graft_target_latent` is of `hier` in refcv3's own registered delta: admitting
four channels whose withheld state is indistinguishable from a real physical
state would multiply the X15 defect by four (§4.2). The pair is pinned as
`REGISTERED_DELTA_KEYS_V4` and asserted from the dataclasses by
`tests/test_refc_v4.py::test_registered_delta_is_pinned` — a **derived** claim,
never a hand-written list of intentions.

### 2.1 Held constant across every arm

`corpus` · `episode set` · `window (8)` · `horizons (V3_HORIZONS, 8 slots)` ·
`steps` · `batch` · `lr` · `warmup` · `seed` · `--size` · `--arm hier` ·
`--mode diffusion` · anchors (128) · decoder geometry · `tac_vocab_version` ·
`nav source` · every loss weight.

⛔ **Diff the launch commands, not the intent** (2026-08-22: a row-bank arm
changed `n` and silently multiplied the effective λ; two sweeps were
invalidated). The panel script emits each arm's full argv into its own log.

---

## 3. THE EDGE-LIST DELTA vs refcv3 — exactly what E11 becomes

REF-C v3's conditioning graph is `REFC_V3_DESIGN.md` §4, edges **E1–E12** (plus
**E13**, `nav → {z_tac, ctx}`, added by the PI's 2026-09-02 ruling). v4 changes
**three** things and nothing else.

| edge | v3 | **v4** | why |
|---|---|---|---|
| E1–E10, E12 | — | **UNCHANGED** | the trunk still never sees `v0`; see §3.2 |
| **E13** — `nav_cmd → {z_tac, ctx}` (PI 2026-09-02, zero-init, 50,176 params) | present | **UNCHANGED, AND NAMED AS A KNOWN-BROKEN INPUT** | ⚠️ **the shipped refcv3's route head is nav-INDEPENDENT at exactly +0.0000** (true minus shuffled accuracy). It beats the majority rate, so it reads *something* — but nothing from the route token. **This is a second dead input sitting beside the missing ego state, and it gets its own row so a v4 gain cannot be silently credited to "the conditioning is fine now".** ⛔ It is explicitly OUT of scope for this package: fixing two conditioning edges in one arm would make the result non-attributable, which is the `--v2` conflation failure. `ego_injected` and `nav_injected` are both logged per step, so an inert edge is visible rather than inferred |
| **E11** | ⛔ **`v0 ↛ {z_tac, g_str, ĝ_tac}`** — a REFUSED edge, *"goal path stays vision-pure"* (`refc_v3.py:213`, `REFC_V3_DESIGN.md:174`), pinned by intervention audit | ⭐ **E11′ — `ego_state@t0 → {z_tac, ctx→g_str, ĝ_tac}` is a REQUIRED LIVE edge.** `ego_state = (v0, a_long, yaw_rate, curvature, keep)` read at the **last OBSERVED frame** | the refusal cited the 2026-08-03 vision-only rule and drew the line at **modality**; the PI has since narrowed it twice to **time + anti-echo**. E11 was also **stricter than every planner in the published comparator set** — TCP, VAD, VADv2, Hydra-MDP, PARA-Drive and NAVSIM-TransFuser all feed ego status to the planner and **none** withholds it from a goal head by design (E-REFC-EGO-1 §5) |
| **— (new)** | — | ⛔ **E11″ — `future_poses` / `future_actions` ↛ ANY goal node. REFUSED, and this is now the PINNED NEGATIVE EDGE** | a provenance audit whose only refused edge just became a positive edge has no teeth. The replacement is the edge the PI's constraint actually names. ⚠️ **The mechanically-obvious wrong implementation of this feature is `a0 = (future[:,0,3] − v0)/dt`** — identical in a config diff, satisfies any *"does the goal use acceleration?"* check, produces a **better-looking** result, and is a future read |
| **— (new)** | — | ⭐ **E14 — echo-quotiented goal supervision**: `ĝ_tac = ha0_ext(v0,a0,κ0) + Δ`, `Δ` from a **zero-init** head, `ha0_ext` multiplied by `keep` | the trivial solution is handed to the model for free, so gradient descent has no incentive to re-derive it and every learned parameter is spent on what the ego state cannot explain. It also makes the echo **legible in the training log** (`echo_ratio = \|Δ\| / \|ha0_ext\|`) |
| **— (new)** | `core.ego_valid_channel = False` | ⭐ **X15 — `core.ego_valid_channel = True`, and `keep` is drawn ONCE per sample and shared** by the goal path and the measurement encoder | §4.2 |

**Registered config delta, derived and pinned:**
`config_delta(refc_v4_config(s), refc_v3_sized_config(s)) ==
{"ego_state_inject", "echo_base", "core.ego_valid_channel"}` — exactly, no more.

⭐ **E11′ enters through its OWN embedding, never through `tactical_speed_input`.**
`core.tactical_speed_input` stays **False** in v4, exactly as in v3. Two
consequences, both deliberate:

* the ego reaches the hierarchy's factored `lat_head_tac` / `lon_head_tac`
  (they read `z_tac`, which E11′ moves) but **not** the core's own pooled-based
  `lat/lon` aux heads — so the **core aux surface remains an ego-free control
  inside the same arm**, and a within-arm comparison of the two decision
  surfaces is available for free;
* the registered delta stays **three keys**. Routing through
  `tactical_speed_input` would have carried only `v0` (a scalar concatenated to
  `pooled`), not the four channels and the presence bit, and would have widened
  a second head's input as a side effect.

### 3.1 What v4 does NOT relax

* ⛔ the **situation classifier's output** into any goal node — still refused
  (PI 2026-08-03, UNCHANGED, and the goal/situation disjointness argument is
  untouched by this package);
* ⛔ the **LAN corridor** at inference — still label-only (E12, UNCHANGED);
* ⛔ any **future** ego quantity — now the pinned negative edge (E11″).

### 3.2 Why the placement is the safe one, from the primaries

The literature's structural line is at the **scene encoder**, not at the
planner. The only closed-loop measurement of harm is **CARLA-TransFuser Tab. 10:
DS 56.68 → 45.35** when the velocity is summed into the backbone at all four
stages (*"a sharp drop in DS, which cannot be recovered"*, lib `2205.15997`).
refcv4's placement — a **separate ego embedding → zero-init projections into
`z_tac` / `ctx`**, plus the existing measurement-encoder → decoder condition —
is the site where PARA-Drive calls the same information *"marginal"* and safe,
and where NAVSIM's TransFuser **gains 1.5–2.6 PDMS** (libs `paradrive-cvpr2024`,
`2406.15349`). **AdaptiveAD (lib `2511.13079`)** names *premature fusion of ego
status in the upstream BEV encoder* as the root cause and keeps a
deliberately ego-free scene branch — which is what refcv4 already has and keeps.

⇒ **The design is not the risk. The corpus and the metric are (§5, §6).**

---

## 4. The ego channels — derivation, and why nothing is differentiated

### 4.1 The derivation, at source

⭐ **Neither derived channel needs a finite difference**, which removes the
brief's worst-case hazard (a noisy derived channel correlated with the target is
an echo vector, not a feature). MEASURED at source,
`stack/tanitad/data/physicalai.py:597-632`:

```
actions[:, 1] = ax                       the corpus's OWN longitudinal accel
                                         (its docstring at :13 says explicitly
                                         "NOT d/dt(v), which differentiates
                                         interpolation noise and lags")
actions[:, 0] = atan(WHEELBASE * curvature)   an INVERTIBLE encoding of the
                                              corpus's OWN curvature column
```

and `stack/tanitad/data/_contract.py:130` already returns `ep.actions[t:t+w]` in
every window, `:137` returns `pose_last = ep.poses[t+w-1]`. So:

| channel | derivation | dt | filtering | source field |
|---|---|---|---|---|
| `v0` | `pose_last[:, 3]` | — | none (corpus value) | `poses` |
| `a_long` | `actions[:, -1, 1]` | — | none (corpus `ax`) | `actions` |
| `curvature` | `tan(actions[:, -1, 0]) / 2.9` | — | none (exact inverse) | `actions` |
| `yaw_rate` | `v0 * curvature` | — | none (algebraic) | derived |
| `keep` | the withholding draw (X15) | — | — | model |

**`t0 = W−1`, the LAST OBSERVED frame, for every channel.** No future index
exists in `ego_state_at_t0` and none may be added — that is what E11″ pins
interventionally rather than in a docstring (including this one).

⚠️ **The wheelbase cancels, and that is not an accident we may assume.**
`WHEELBASE = 2.9` is wrong for 98.2 % of clips, but the cache **stored**
`atan(2.9 · κ)`, so `tan(steer)/2.9` recovers κ **exactly** — *provided* the
cache is the legacy `const2p9` regime, which the parity corpus is
(`physicalai.py:82`). A `per_clip_v1` cache is **REFUSED with a raise**, not
silently mis-inverted: a wrong `L` scales every curvature by `L_true/2.9` and
reads exactly like a working channel.

⚠️ **`curvature` is fed ALONGSIDE `yaw_rate`, not instead of it.** `yaw_rate =
v0·κ` vanishes at standstill and **16.50 % of frames have `v < 0.5 m/s`**. A
stopped car with the wheel turned has a real path geometry and a zero yaw rate.
The two are algebraically dependent given `v0`; that is **disclosed, not hidden**.

### 4.1b The two independent cross-checks (absence/validity needs two probes)

MEASURED 2026-09-04 on the val epcache `physicalai-val-bb543bdf7836`,
**40 episodes / 7,963 frames**, instrument
`tanitad.eval.echo_gate.corpus_echo_report`, artifact
`raw/census_val40.json`:

| check | reads | verdict |
|---|---|---|
| **`yaw_rate` vs `d/dt(unwrap(yaw))`** — an entirely different corpus field (the orientation quaternion) | **r = +0.9430** (mean over 40 episodes) | ⭐ confirms the **sign** and the **wheelbase inversion**. A sign error would read ≈ −0.94 and a wheelbase error would read ≈ 0 — both look exactly like a working channel without this probe |
| **`a_long` vs `d/dt(v)`** | **r = +0.5624** | ⚠️ **EXPECTED to be well below 1, and it is the reason `ax` is used.** `physicalai.py:13` states the finite difference *"differentiates interpolation noise and lags the true signal"*. A high correlation here would have meant `ax` was itself a finite difference |

**Channel statistics on the same sample** (they set `EGO_SCALE_*`, so the block
enters at ~unit magnitude rather than at a guessed scale):

| channel | mean | std | range / notes |
|---|---|---|---|
| `v0` | 5.2449 | **3.6714** | 16.501 % of frames `< 0.5 m/s` |
| `a_long` | −0.1607 | **0.9312** | [−4.128, +4.407]; non-zero on **100.0 %** of frames |
| `yaw_rate` | — | **0.1593** | \|max\| 0.8695 |
| `curvature` | — | **0.0546** | non-zero on **99.598 %** of frames |

**Noise floor, stated:** these are the corpus's own recorded channels, not
estimates — there is no differentiation step and therefore no differentiation
noise to bound. The residual uncertainty is the corpus's own interpolation,
which is the same uncertainty already carried by `v0` in refcv3.

### 4.2 X15 — the zero-collision, and why the validity bit is a PRECONDITION

MEASURED, n = **781,635 windows / 4,572 clips**
(`../2026-09-03-ego-zero-collision/`):

* `v0` is **exactly 0.0** on **4.4531 %** of windows — a hard atom (the `[0, 0.1)`
  bin holds **13×** the next);
* with `ego_dropout 0.5`, **P(input reads 0) = 52.227 %**, of which **95.737 %
  withheld** vs **4.263 % genuine → 22.5 : 1**;
* dropout is training-only ⇒ at eval **100 % are genuine**. **The same token means
  something 23.5× different between train and eval.**
* Consequence, predicted from source and then MEASURED: the stop branch needs
  `v0 ≥ 1.0`, so `brake_stop` is unreachable at `v0 = 0` — **0 of 34,807** at
  eval against **16.0 % brake mass at train**. This is very likely behind the
  shipped model's `brake_stop` recall **0.3106**.

⇒ Admitting **four** channels whose withheld state is indistinguishable from a
real physical state would multiply the defect by four. **`ego_valid_channel` is
therefore a precondition of E11′, and the code raises if it is off.** The
mechanism is different per channel and each half is MEASURED
(`raw/census_val40.json`), so it is stated per channel rather than asserted once:

| channel | exactly 0.0 in the corpus | why a withheld 0 is still a lie |
|---|---|---|
| `v0` | **4.4531 %** of windows — a hard atom | indistinguishable from a genuine standstill; **22.5 : 1** withheld-vs-genuine at train |
| `curvature` | **0.402 %** of frames | indistinguishable from a genuinely straight wheel |
| `a_long` | ⚠️ **0.000 % — never exactly zero** (0 of 7,963 frames) | so a withheld 0 is *out of distribution* as an exact value, yet it sits at the **centre of the density** (mean −0.1607, std 0.9312) and reads as an ordinary cruise/mild-decel. The model cannot tell "no reading" from "not accelerating" |
| `yaw_rate` | follows `v0 · κ` | vanishes at standstill by construction, which is why `curvature` is fed beside it |

⚠️ **CORRECTED HERE.** An earlier draft of this section — and the raise message
in `refc_v3.py` — said *"`a_long = 0` is the MODE of the distribution"*. The
census refutes that: `a_long` is **non-zero on 100.0 %** of the sample. The X15
argument survives intact and is *sharper* stated per channel; the claim that did
not survive was an inherited plausibility, not a measurement. Class: an
INHERITED claim asserted where a measurement was available.

⭐ **SECOND PROBE, DIFFERENT SAMPLE AND DIFFERENT PATH BINDING**
(`raw/zero_probe_val40.json`, **100 val episodes / 19,900 frames**). The
4.4531 % figure the X15 precondition rests on is MEASURED by a sibling package
over **781,635 TRAIN windows** — for this package it is INHERITED, and it
decides a design constraint, so it gets a second probe:

| quantity | sibling (781,635 **train windows**) | this probe (19,900 **val frames**) |
|---|---|---|
| `v0` exactly 0.0 | **4.4531 %** | **11.00 %** |
| the `[0, 0.1)` atom vs the next bin | **13×** | **12.63×** |
| `curvature` exactly 0.0 | (0.402 % @ 40 ep, stride 7) | **0.387 %** |
| `a_long` exactly 0.0 | — | **0.000 %** |
| `yaw_rate` exactly 0.0 | — | **11.34 %** |

⚠️ **The two `v0` rates differ by 2.5× and that is NOT a contradiction — they are
different units on different splits** (per-window at t0 over train vs per-frame
over val). Reporting them as agreeing would be the `df`/`step_s` scope error.
**What reproduces is the load-bearing structural fact: the hard atom at exactly
zero, 13× vs 12.63× the neighbouring bin.** That is the property that makes a
withheld zero indistinguishable from a genuine standstill, and it now rests on
two independent samples.

⭐ **The fix bit already existed.** `keep` is computed on every forward and
passed to the decoder as `ego_keep` for the S2 reachability band
(`refc.py:2157`, honoured `:1354`/`:1494`); it was withheld from the
**measurement encoder** — the only route by which ego speed reaches the
trajectory decoder — solely by `ego_valid_channel = False` (`refc.py:585`).
**A wiring change, not new machinery.**

⭐ **ONE DRAW, ONE OWNER.** v4 draws `keep` once per sample in
`RefCV3Model.forward` and hands the *same* vector to both consumers (the goal
path via `ego_state[:, 4]`, the measurement encoder via the new `ego_keep`
seam). `refc.py`'s own comment warns that a second, unsynchronised dropout is
the wrong fix, and it is right. ⛔ The alternative — pre-multiplying `v0`
outside — is **worse than wrong**: `keep` is derived from `v0 is not None`, so a
pre-zeroed `v0` arrives with `keep = 1` and *asserts* "this really is a
stationary car", which is exactly the collision this seam removes.

⚠️ **THE RATE IS UNVALIDATED AND OUR DOCSTRING LAUNDERED IT.** `refc.py:10-11`
says the TCP-C stack is kept verbatim *including "the measurement encoder with
per-sample ego-dropout"*. The **encoder** is TCP's (FC-128 × 2, bit-for-bit);
the **dropout is not** — "dropout" occurs **zero times** in TCP (two probes,
E-REFC-EGO-1 §2). `ego_dropout = 0.5` is a TanitAD invention nobody has swept.
The two papers that did sweep it landed on **0.5** (DRAMA, NAVSIM PDMS
0.835 → 0.848) and **0.75** (PlanTF, nuPlan closed-loop). **This is a
documentation correction owed in the same commit as the code.**

---

## 5. The anti-echo mechanism — what was chosen, and why not the alternatives

**Chosen (three parts, all shipping together):**

1. ⭐ **Per-sample ego withholding with a PRESENCE BIT (X15).** Best-evidenced
   mechanism in the field: **three independent adoptions** (PlanTF's State
   Dropout Encoder, PLUTO reusing it unchanged for **+2.60 nuPlan**, DRAMA on
   **NAVSIM PDMS 0.835 → 0.848**), a **published rate sweep**, and **half of it
   was already computed in our code**. The presence bit's primary is **GRU-D**
   (lib `1606.01865`), whose argument is the PI's verbatim: imputation *"cannot
   distinguish whether missing values are imputed or truly observed"* —
   ablation, MIMIC-III AUC: mask-only **0.8367** vs mean-impute **0.8192**.
   Cost: ~2.6 × 10⁴ params, **zero inference cost**.
2. ⭐ **Keep the ego OUT of the scene encoder** — already true, now *stated with
   its citation* instead of as a blanket refusal (§3.2).
3. ⭐ **E14, the echo quotient.** The residual head is zero-init, so the arm
   **starts exactly at `ha0_ext`** and every learned parameter is spent on what
   the ego state cannot explain. ⚠️ **Stated honestly: E14 does NOT make echoing
   impossible** — a lazy model outputs `Δ ≈ 0` and scores exactly `ha0_ext`.
   What E14 buys is that this outcome is **visible from step 1 and
   attributable**, via `echo_ratio` in the training log. The thing that
   **catches** echoing is the gate (§6). Both ship.

**Refused, with the evidence:**

| candidate | verdict |
|---|---|
| **Residual-over-CV as the OUTPUT PARAMETERISATION** | ⛔ **REFUSED as designed.** Not found as a published driving method (two probes), and the nearest measurement in **our own architecture class** points the other way: DiffusionDrive Tab. 8 — learned anchors **88.1**, trained-on-extrapolated **84.7**, extrapolated-at-inference **81.3**. ⇒ use the extrapolation as a **REFERENCE**, never as the vocabulary. *(E14 is not this: it is an additive base under a zero-init residual on the GOAL head, with the 128-anchor trajectory vocabulary untouched.)* |
| **Adversarial / GRL stripping of ego from the visual feature** | ⛔ **no driving precedent found** (two differently-worded probes); high risk; and it fights a design we want — we *want* the ego in the measurement encoder |
| **Scene-grounding auxiliary losses** | ⚠️ deferred. The literature is genuinely conflicted (PRIX **+17.4 PDMS** from grounding; SSR's Tab. 4 shows map/obstacle supervision **worsening** L2 0.75 → 0.81–0.86; BEV-Planner+Map moved L2 **0.55 → 0.96** while CCR **improved** 4.26 → 2.60). It resolves only under manoeuvre stratification, which this package builds first |

---

## 6. Success and failure, committed in advance

### 6.1 ⛔ THE BAR IS `ha`, NOT `ha0` — and `ha0_ext` sharpens it

MEASURED on refcv3, **4,823 windows / 141 episodes**, paired episode-cluster
bootstrap (`taniteval/ci.py`), tier **T1 — and OPEN LOOP by the 2026-09-02
ruling**. Artifacts: `taniteval/results/openloop-suite-refcv3-30k-ckpt30000.json`
and `taniteval/results/RESULT-refcv3-40284-openloop.md`:

```
step 30,000:  ha 0.2996 m  <  os 0.4799 m  <  ha0 0.6723 m
   os − ha0 = −0.1924 [−0.2525, −0.1379]   separated (the model beats the floor)
   os − ha  = +0.1803 [+0.1563, +0.2047]   separated THE WRONG WAY

step 40,284:  ha 0.2996 m  <  os 0.4419 m  <  ha0 0.6723 m
   os − ha0 = −0.2304                       (the floor margin improved)
   os − ha  = +0.1423 [+0.1187, +0.1658]    STILL separated the wrong way
```

⚠️ **Both reads are quoted because a newer one exists and the conclusion is the
same** — quoting only the 30k number would have been the *stale-source* error,
and quoting only the 40,284 number would have hidden that 10,284 further steps
closed just **21 %** of the gap. `ha` and `ha0` are checkpoint-independent and
read **bit-identically at both steps on the same 4,823 windows**, which is the
internal control that the two reads sit on one surface.

⇒ **a refcv4 that merely beats `ha0` proves nothing refcv3 does not already
do.** ⚠️ And `ha` is itself an ego-dynamics extrapolation, so **beating `ha` by
echoing harder is not success either** — which is why `ha0_ext` (the same
control built on the corpus's own `ax` rather than a finite difference of speed,
hence the **stronger** form) sits beside it. **Clearing `ha` while TYING
`ha0_ext` means the margin came from a better extrapolation of the ego state,
not from reading the scene**, and the gate reads it that way mechanically.

**The corpus-kinematic size of the echo**, MEASURED 2026-09-04 on the val
epcache (40 ep / 7,963 frames, stride 7), artifact `raw/census_val40.json`:

| horizon | `ha0` (const v) | `ha0_ext` (const v, a, κ) | ext beats ha0 | n windows |
|---|---|---|---|---|
| 2.0 s | 0.7040 m | **0.4449 m** | **+36.80 %** | 1,041 |
| 6.0 s | 5.3547 m | **4.4652 m** | **+16.61 %** | 801 |

**0.4449 m at 2 s from ZERO PIXELS** is the same league as flagship v1's
deployed 0.452 m (`MODEL_REGISTRY.md`, `flagship4b-speedjerk-30k`). An arm
handed these channels can reproduce the programme's headline 2 s number without
ever reading the image.

⚠️ **SCOPE, STATED.** These are a corpus-kinematic property of **those** windows
— the design input and the gate's construction. They are **NOT decision
numbers**, and they are **NOT a driving tier**. `echo_gate` recomputes both
controls on the SAME windows as the model it scores, paired. Quoting 0.4449
against a differently-windowed arm would be the `df`/`step_s` scope error in a
new costume.

⚠️ **CORRECTION LOGGED IN THE SAME BREATH.** The first draft of
`echo_gate.py`/`refc_v3.py` recorded **0.4476 / 4.5912** (+36.42 % / +14.26 %)
for these two cells. Re-running the *current* instrument on the *same* 40
episodes and the *same* window counts (1,041 / 801) reproduces `ha0` exactly
(0.70402) but reads `ha0_ext` **0.44492 / 4.46515**. The mechanism is almost
certainly the **stop clamp** (`τ` clamped at `−v0/a0` under deceleration, so the
vehicle stops rather than reversing) being added *after* that measurement — it
can only reduce the error, and it did, in both cells. **The banked JSON is the
quotable number; the docstrings are corrected to match.** Class: a derived
constant re-measured after its input changed.

### 6.2 SUCCESS — every clause, with its CI, committed now

⛔ **The gate is NOT scored on ADE.** Three independent primaries say a working
anti-echo guard *lowers* displacement error: PlanTF **OLS 88.55 → 87.07 (−1.48)
while R-CLS 74.79 → 80.59 (+5.80)**; CADET measures that suppressing genuinely
spurious reliance moves open-loop metrics by **< 0.1 m** (*"displacement error
does not register a change in causal reliance"*); BEV-Planner's map aux moved L2
**0.55 → 0.96** while CCR improved **4.26 → 2.60**. **An ADE-only gate will
reject the working guard.**

| # | clause | criterion (committed in advance) |
|---|---|---|
| **S1** | **GATE 1 — beat the echo, per family** | on the **6.0 s** primary slot, **paired episode-cluster bootstrap over episodes**, the arm beats **BOTH** `ha` **and** `ha0_ext` with the CI excluding zero **AND** a **relative point margin ≥ 0.10**. Reported **per metric family** (longitudinal / lateral / tactical / strategic), never pooled |
| **S2** | **GATE 2 — structural, both directions** | `ego_intervention_test` returns `READS_BOTH`: perturbing `ego_state` moves every goal node **AND** replacing `frames` moves every goal node, with a passing determinism check. `FUTURE_LEAK`, `ECHOING`, `EGO_DEAD` and `UNPOWERED` all FAIL |
| **S3** | **GATE 2b — functional, and it is the one that matters** | `source_ablation_test` returns `READS_BOTH`: the **wrong scene** raises ADE by **≥ 5 % relative with a separated paired CI**, and so does the **wrong ego** |
| **S4** | **the X15 clause** | at eval, `brake_stop` is **reachable**: > 0 of the eval windows predict it (against the MEASURED **0 of 34,807** today), and `config.json` **stamps** the full ego configuration |
| **S5** | **the echo-harder trap, mechanised** | clearing `ha` while **tying** `ha0_ext` (margin < 0.10 or CI straddling zero) is a **FAIL**, not a pass |
| **S6** | **manoeuvre stratification** | S1 is additionally reported on the **non-straight** subset. ⚠️ BEV-Planner App. Tab. 4: aggregate CCR moved 3.82 → 3.81 (*nothing*) while CCR-ST went 9.13 → 17.6 and CCR-LR 3.05 → 1.69 — **the aggregate hid a total behavioural collapse** |

### 6.3 FAILURE — committed with equal force

| # | outcome | reading |
|---|---|---|
| **F1** | the arm ties `ha0_ext` on the primary slot (margin < 0.10 or CI straddling zero) | **E11′ bought an echo.** The ego channels are reproducing the dynamics; refcv4 is not an improvement and must not ship |
| **F2** | `source_ablation_test` says `ECHOING` (wrong scene does not hurt) | same verdict as F1, arrived at without needing a control's ADE |
| **F3** | `source_ablation_test` says `IGNORES_EGO`, or GATE 2 says `EGO_DEAD` | **the wiring is dead** — the arm is v3 wearing a v4 config, and a null result from it is about the plumbing, not the hypothesis. Fix the wiring, re-run; do **not** report "ego does not help" |
| **F4** | GATE 2 says `FUTURE_LEAK` | ⛔ the PI's *"not the future one"* is violated. **Stop; the arm is inadmissible regardless of its score** |
| **F5** | `echo_ratio` stays ≈ 0 to the end of training | the model learned nothing the ego state did not already say. E14 made this legible; it is a **FAIL of H-ECHO-3's usefulness clause**, not of E14's honesty clause |
| **F6** | ⛔⛔ **the gate does NOT fail arm E** | **THE PANEL IS VOID.** Not the model — the *instrument*. No PASS from it is admissible until the gate is repaired (`TanitAD_ValidateAIDesign` §2) |

### 6.4 ⛔ What may NOT be cited as support

* the `metrics.jsonl` lateral-vs-longitudinal **5.3×** gap. `tactical_speed_input
  = False` means those heads never saw the speed, and **no CI is computable from
  that file**. It is not evidence for the zero-collision and must not be quoted
  as such.
* any **tiny-rig** number as a capability claim (§8).
* `echo_ratio` as a verdict. It is a ratio of magnitudes: a large `Δ` can still
  be wrong and a small one can still be right if `ha0_ext` is already near
  optimal. It says whether to keep paying for the run.

---

## 7. Controls — the three the skill requires, plus the two the field supplies

| control | must read | why it is here |
|---|---|---|
| **`constant_only`** | **exactly** the no-information value (`R² = 0.0` to machine precision; zero degradation under *either* derangement) | 2026-08-22: three of four estimator failures were caught **only** because a control read the same value as the thing being measured |
| **`raw_input_floor`** | ridge from downsampled pixels → target, with **`n` and `d` PRINTED** | a learned representation that does not beat raw input has added nothing. `n ≪ d` is underpowered **by construction**, not a negative |
| **`deliberate_regression`** (arm E) | ⛔ **the gate must FAIL it** | if the gate does not fail an arm that echoes by construction, a PASS on the real arm means nothing |
| `ha0` (const-v) | the programme's old floor | the `ha0 → ha0_ext` gap is itself the quantity that says how much of a margin is echo |
| `ha` / `ha0_ext` | the deciders (§6.1) | published precedent: NAVSIM's own CV calibration practice |

⛔ **Hyper-parameter discipline:** every λ and PCA basis in every probe is fit on
the **FIT split only**. Selecting λ on the scored split picks maximal
regularisation, shrinks the ridge to the constant predictor, and scores a
beautiful, meaningless 0.0000 with a zero-width CI (2026-08-22 failure #3).

### 7.1 ⛔⛔ The structural probe is NOT sufficient — MEASURED, and it voided the gate's first version

The deliberate-regression arm **PASSED** the interventional `scene → goal`
probe. Of course it did: the encoder is a live function of its input, so
`frames → goal` is a live forward path in **any** model with a non-degenerate
trunk — *including one that has learned to ignore the scene completely*. **The
intervention measures WIRING; echoing is about USE.** That is the same family as
the finding `goal_provenance` was built on (*"a detached wire carries the full
signal and zero gradient"*), read in the converse: **a live wire can carry zero
useful signal.**

⇒ `assert_not_echoing(gate2b=None)` returns a verdict explicitly marked
**`STRUCTURAL_ONLY`**. It is **not** an anti-echo pass and must not be reported
as one.

---

## 8. Splits — and nothing is tuned on the scored split

| split | what it is | what may touch it |
|---|---|---|
| **FIT** | episodes of the train epcache | training only |
| **VAL** | carved from FIT **only** | any hyper-parameter selection (λ, probe basis). ⭐ **In this panel: NOTHING is selected.** Every hyper-parameter (lr, warmup, batch, steps, seed, dropout rate) is **fixed a priori and identical across arms**, so the VAL split is unused by construction rather than by promise |
| **TEST** | the **episode-disjoint** val corpus | **scored, never tuned on.** Paired episode-cluster bootstrap over episodes; the arms are scored on the **same windows**, which is what makes the pairing valid and strictly more powerful than combining two intervals in quadrature |

⚠️ **The tiny rig runs on a NON-PARITY corpus and says so out loud.** The
trainer prints `[parity] ⚠ NON-PARITY corpus`, and `config.json` stamps
`size: "tiny"`, `rig_rung: true`. ⛔ **Nothing measured at the rig rung is a
model claim or may enter `MODEL_REGISTRY.md`** — it validates the DESIGN (does
the wiring fire, does the gate fail what it must fail), never the architecture's
quality. Precedent and its limits: **H-SCALE-2** — *"screen architecture on tiny
arms, but never quote a tiny-arm number as the programme's capability."*

---

## 9. What this package does NOT fix (named, not hidden)

1. **The route head is nav-independent at exactly +0.0000** (true minus shuffled
   accuracy). It beats the majority rate, so it reads *something* — but nothing
   from the route token. **A second broken input beside the missing ego state,
   and v4 does not touch it.** It gets its own edge in a later delta; naming it
   here is what stops a v4 result being read as "the conditioning is fine now".
2. **The E-ECHO-2 corpus census** — *what fraction of our val split a
   constant-velocity / constant-curvature agent already solves* — is the
   highest-leverage item in the sibling package (NAVSIM moved the CV agent
   **PDMS 79 → 22 by filtering alone**, larger than any architectural fix here).
   §6.1's table is the two-point version; the full distribution and the `hard`
   split are **not** in this package.
3. **The `ego_dropout` rate is not swept here.** The panel fixes 0.0 vs 0.5;
   PlanTF's 0.75 and the OLS/CLS trade it implies are a follow-up arm.
4. **No T2.** Every number here is open-loop; per the 2026-09-02 ruling a planner
   feeding its own predictor is still open loop, and "drives" requires T2.

---

## 10. The launch line (⛔ DO NOT FIRE — the PI's sequence is refcv3 eval → video → push first)

⭐⭐ **THE LINE IS THE INCUMBENT'S OWN COMMAND WITH THREE FLAGS ADDED, AND THAT IS
THE POINT.** It was read from `HF_CARD_tanitad-refc-v3.md` §"Exact command (read
from live `ps` on the training pod)" — same rung (`--size base`), same corpus
(`--v2-cache`), same geometry (`--image-hw 256 640`), same nav source
(`--nav-from-v7`), same optimiser, same seed, same steps. **Anything else would
make refcv4-vs-refcv3 a bundle instead of a lever.** ⚠️ An earlier draft of this
section named `--size small` and `--data-root`; the incumbent is `base` on the
v2 cache, and a comparison across those would have been the C6 confound.

**Preflight (free, CPU, ~2 min — it exits non-zero on any refusal). VERIFIED
2026-09-04 at the real rung:**

```
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py --preflight \
  --arm hier --size base --image-hw 256 640 --synth-episodes 4 \
  --out /workspace/experiments/refcv4-pf \
  --ego-state-inject --echo-base --ego-dropout 0.5 --device cpu
```

reads: `arm=hier params={... 'ego_inject': 25536 ... 'total': 107053435}` ·
`freeze-history gate: pass: True` · **`E11' OK: ego reaches the goal path,
frames still move g_str/z_tac, keep=0 withholds cleanly`** · one end-to-end loss
step with `echo_ratio 0.0` / `echo_base_absmean 3.111` (the zero-init residual,
exactly as designed) / `ego_keep_frac 0.5` · `✅ PASS`.

**The run:**

```
OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack \
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --out /workspace/experiments/refcv4-b1-v72-40k \
  --ego-state-inject --echo-base --ego-dropout 0.5
```

* `--ego-state-inject` **implies** `--ego-valid-channel` — the build **raises**
  without it (X15 is a precondition, not an option).
* `--echo-base` turns on E14.
* ⛔ `--ablate-frames` is the rig's deliberate-regression lever and **must never
  appear on a real run**. A run carrying it is stamped `ablate_frames: true` in
  its own `config.json` and is a GATE CONTROL, never a model.
* **Expected parameter count with `--v7-labels`: 107,058,565** = the incumbent's
  pinned 107,032,901 + **25,664**. *(The preflight above reads 107,053,435
  because a synthetic corpus carries `kin3` labels and the vocab follows the
  data — a 5,130-param head difference, not a config difference.)*

⭐ **The v2 cache carries what E11′ needs, verified at source**:
`tanitad/data/v2_dataset.py` stores `.actions` per clip and
`tanitad/data/_contract.py:132` returns `ep.actions[t:t+w]` — the *same* field
the raw-epcache path uses. **No cache rebuild, no new dataset field.** And if a
cache ever lacked it, `ego_state_from_batch` **REFUSES with a message** rather
than substituting zeros, which would train a dead channel and read as *"ego does
not help"*.

**Where it may run:** ⛔ **not Thor** (training refav1), ⛔ **not
`tanitad-refcv3` while the final eval is on it**. A free A40/A6000 pod, or
`tanitad-refcv3` once its eval has filed.

---

## 11. Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| this pre-registration | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refc-v4-design/PREREG_REFC_V4.md` | no |
| corpus census (the §4.1b / §6.1 numbers) | `repo:.../2026-09-03-refc-v4-design/raw/census_val40.json` | no |
| tiny-rig panel result | `repo:.../2026-09-03-refc-v4-design/RESULT.md` + `raw/` | no |
| E11′ + E14 + X15 implementation | `repo:stack/tanitad/refs/refc_v3.py`, `repo:stack/tanitad/refs/refc.py` | no |
| the anti-echo gate | `repo:stack/tanitad/eval/echo_gate.py` | no |
| trainer flags + stamps + rig rung | `repo:stack/scripts/refc_v3_train.py` | no |
| tests | `repo:stack/tests/test_refc_v4.py` | no |
| sibling literature package | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refc-ego-inputs-and-anti-echo/` | no |
