# V5 → V5-FLAGSHIP — deep review and redesign

**Written 2026-08-02 (PI directive: act as reviewer + designer; confirm each intent point; redesign
if necessary; decide the restart at the next checkpoint).** v5 is at **step 5,300 / 30,000** as of
12:39 UTC, alive on pod2. Every claim below carries its evidence class; the verdicts use the
programme's own measurements, not intentions.

**The one-line summary: the running v5 implements roughly HALF of the PI's stated design. The
half that is present is good and confirmed by config; the half that is absent — imagination
conditioning and a *proven* hierarchy — is exactly where our instruments already show the
programme's deepest defects. The redesign closes the gap; the restart is justified; three
decisions are owed.**

---

## 0. What v5 actually is today — MEASURED, from the run's own config.json

| axis | value |
|---|---|
| arch | `flagship-v4` — *"joint WM + operative planner; λ_plan curriculum"*, **FROM SCRATCH** (random-init trunk, no v1 warm-start) |
| corpus | **parity** `physicalai-train-e438721ae894` (2,376 eps), `require_parity` ✅; val `physicalai-val-0c5f7dac3b11` |
| geometry | 256×640 **cylindrical, 120° HFOV**, subframe **176×624** |
| head | **anchored-diffusion decoder** — `n_anchors 256`, `diffusion_steps 2`, `noise_std 0.1`, d=384 ×4 layers, 20 horizons |
| conditioning | `cond_states ✅` · `cond_vtarget ✅` · `cond_route ✅` · ⛔ **`cond_imagination = False` (hard-wired, `train_flagship_v4.py:1242`)** |
| goal handling | v2.1 route labels (`route_target_v21`, explicit `ROUTE_UNKNOWN`), VTARGET 23-band set-speed, ReZero gates init 0.1, **goal_dropout 0.5, ego_dropout 0.5** |
| loss | `w_lat .05 · w_lon .05 · w_dist .05 · w_jerk .02 · w_curv .01 · w_strat .05` |
| schedule | phase_a 2000, phase_b 8000, gate 10000; milestones [5k, 10k, 15k, 20k, 30k]; λ_plan `sched` |
| ops | gateless (C65/C66: the in-loop held-out gate killed it twice via a stale-path force-load; C67: my kill cost 3.5 days) — gate runs OFFLINE on checkpoints |

---

## 1. The PI's points, confirmed or not — one verdict each

### P1 — "v5 = v1 flagship + REF-C diffusion planner, its prediction conditioned on predictions, with anchor-selection improvements" → **PARTIAL: the planner is there, the conditioning-on-predictions is NOT**

✅ **Confirmed half (MEASURED, config):** the head is the REF-C lineage made flagship —
anchored diffusion (`diffusion_steps 2`) over a **256-anchor** vocabulary with confidence + offset
selection, state-conditioned anchors (E-V5-4), goal tokens. The world model is jointly trained and
grounds the operative rollout (`g_op_fwd_ade_m` is live in the log at ~0.15–0.17 trainer-internal).

⛔ **Falsified half (MEASURED, source):** *"its prediction conditioned on predictions"* — *that is
`cond_imagination`, and it is OFF and hard-wired off.* `flagship_v15.py:20-40` calls it verbatim
**"THE NOVEL PART"**: probe action sequences are rolled through the frozen predictor and the
imagined latents fed as conditioning tokens, so *"the decoder sees the CONSEQUENCES of candidate
controls before it denoises."* With it false, `n_imag_tokens = 0` (`flagship_v15.py:342-343`) —
zero, not fewer. No phase turns it on; `train_flagship_v4.py:1242` fixes it inside the trainer
("imagination off per real_smoke" — an attributability choice from a smoke experiment that silently
became the flagship default).

⇒ **The running v5 is a reactive planner with a world-model-grounded operative layer.** It never
evaluates a candidate control by rolling it forward. The PI's sentence describes a model we are
not currently training.

⚠️ Also for precision: v5 is **not weight-level** "v1 + REF-C" — it is from-scratch at a new
geometry (v1's 256×256 encoder does not transfer to 176×624). The combination is architectural.
That is the right call and just needs saying.

### P2 — "prove the dominance of operative+tactical+strategic; tactical goals → operative, strategic goals → tactical" → **WIRING PARTIAL, PROOF ABSENT — and currently the evidence points the OTHER way**

The wiring that exists: `cond_vtarget` (tactical set-speed) and `cond_route` (strategic route) feed
the operative head as gated goal tokens; `strategic: "full"`; `w_strat` in the loss.

⚠️ **The wiring the PI describes does not exist in v5's head**: the described cascade is
strategic → tactical → operative with a tactical layer *"continuously evaluating different tactical
manoeuvres"*. In v5, **both** goal tokens feed the operative head **directly**; the "tactical
evaluation" is implicit in 256-anchor confidence selection. That may be better or worse — but it is
a different design than stated, and it must not be *proven* by pointing at the 4b cascade.

⛔ **The proof is absent and the current measurements are adverse (all MEASURED):**

| instrument | v1 | v2corpus | reading |
|---|---|---|---|
| hierarchy seams beneficial | **0/3** | **0/3** | top-down conditioning is NOT measurably load-bearing in either arm |
| manoeuvre κ (declared vs driven) | 0.253 WEAK | **0.0072 DECORATIVE** | the `--v2` pack destroyed the coupling |
| vision route vs majority-straight | 0.9474 vs 0.9474 = **no skill** | 0.5088 vs 0.9474 = **collapse** | neither arm demonstrably does route work |

⭐ **HYPOTHESIS (testable, not established): the seam failure and the imagination gap are the same
defect.** A planner that cannot see the consequences of its options has no mechanism by which
top-down goals *could* become load-bearing — a goal token can only bias anchor choice, not select
among predicted futures. If true, P1's fix is P2's fix. The v5-flagship gate is designed to decide
this.

⛔ **Instrument gap that blocks P2's proof today:** `run_hierarchy` supports only
`tactical_policy + strategic_policy` arms (its own G1 guard says so and SKIPs otherwise). v5's
head has neither attribute — **the tactical/strategic families cannot currently be scored on v5 at
all.** Generalising the instrument to the anchor head (declared anchor/manoeuvre vs driven path;
route under nav/follow/zeronav) is a **prerequisite for calling anything "v5-flagship"** — a
binding-rule work item, not an optional extra.

### P3 — "all smoothness (jerk) improvements + significant longitudinal/speed improvement" → **CONFIRMED IN DESIGN, unproven in numbers**

✅ **In the loss, MEASURED:** `w_jerk 0.02` (v5 carries the jerk term **natively** — unlike v1arch,
where it is unreachable without `--v2`; unlike v1, whose loss is unbuildable from HEAD at all —
`V1_RECONSTRUCTION_RISK_RESOLVED.md`). `w_lon`/`w_lat` **separated** — the exact defect of the 5-way
mixed softmax, addressed at the loss level. `w_dist` (distance), `w_curv` (curvature — the axis
RR-20 damaged), and `cond_vtarget` = the measured **88.7 %** longitudinal lever as an input.

⚠️ Unproven: no v5 checkpoint has been through the four families yet. At step 5,300 the trainer
shows `oracle_ade` 0.27–0.34 and `g_op_fwd_ade_m` 0.15–0.17 — **trainer-internal, ~10 % optimistic,
not quotable**. The proof point is the checkpoint gate below.

### P4 — "train v5 on the new corpus if the tactical/strategic labels are better" → **NOT YET — the premise is unestablished and the evidence we have leans against**

- **The labels:** v5 **already carries the repaired v2.1 route labels** (`route_target_v21`,
  explicit `ROUTE_UNKNOWN`, `net_dyaw`) — the label improvements are not tied to the v2bal corpus.
  ⚠️ The *situation* labels remain ego-derived (`situations.py:29` — the sitclf finding), so
  "tactical labels are better" is **not yet true** for situations; that is the L0 gold-set item.
- **The corpus:** the only completed v2bal arm is **worse** — 0.575 vs 0.393 ADE, κ 0.0072, route
  collapse — but that is **confounded** with the ten-lever pack (the prereg amendment). The clean
  corpus contrast (`v1arch`, no `--v2`) is at ~5k/30k and **unfinished**; C64 leaves v2bal without
  a trustworthy val surface until option B is built (feasibility measured: junction stratum has
  6.77× headroom; column semantics unresolved).

⇒ **Recommendation: v5-flagship trains on the PARITY corpus.** Revisit v2bal only after (a) the
v1arch contrast lands and (b) a clean v2-line val exists. Anything else re-runs the C64 mistake.

### P5 — "include the findings from RR-20" → **INCLUDE AS PHASE-2, with the trade managed — not baked in**

**MEASURED (n=881, 40 eps, paired episode-cluster bootstrap B=2000):** RR-20 vs RR-CTL, one flag
apart: ADE 0.424→**0.348** (ΔCI [0.0613, 0.0906], separated); **speed bias +0.9397 → −0.0092 m/s**
(erased); speed MAE −27 %. **Cost:** curvature MAE **2.2×** worse (0.0218→0.0483), yaw-rate worse,
miss@2m 0.043→0.056, tms 0.099→0.269.

Design consequences:
1. ⛔ **Do not bake k=20 into the 30k run.** The only from-scratch high-k datapoint (v2corpus,
   k=12) is bad-but-confounded; the from-scratch k=20 arm never launched (credit). The measured
   win is a **fine-tune** recipe (~7 GPU-h).
2. ⭐ **Phase-2 rollout-recovery is a standard stage of v5-flagship:** after 30k, fine-tune
   ~2,000 steps at elevated rollout-k, then gate on the four families.
3. ⭐ **Mitigation HYPOTHESIS, pre-registered here:** RR-20's curvature cost arose under **v1's
   loss, which has no curvature term**. v5's loss carries `w_curv` — an RR phase on v5 keeps the
   curvature penalty active during recovery training and may buy the speed fix at lower lateral
   cost. Outcomes: (a) speed fix retained, curvature cost < 2.2× ⇒ mitigation real; (b) same
   trade ⇒ the cost is intrinsic to high-k training; both are informative.
4. The **k-sweep (k=8 first)** finds the operating point before committing the flagship to k=20.

### P6 — "any other discovered effect must be included" → the list, so nothing silently drops

| finding | consequence for v5-flagship |
|---|---|
| E-CR **H-COMPOUND** (CR 3.50→80.77, teacher-forced flat) | the mechanism justifying any rollout-recovery phase — compounding is real, not task difficulty |
| C65/C66 (in-loop gate force-loaded a stale tree, killed v5 twice) + C67 | **gate runs OFFLINE on checkpoints, never in-loop**; sync verified by real import before every launch |
| goal_dropout 0.5 route-collapse risk (v2corpus: nav-route 1.0→0.535 under nav-dropout 0.5) | the **pre-registered 5k guard** (`PREREG_v5_cheapest_guard.md`) decides keep-0.5 vs lower-to-0.25 **before** the relaunch |
| four-family binding rule | the flagship gate reports LON/LAT/TAC/STR + ADE at every milestone; **ADE alone is not a result** |
| DATA-vs-ARCH preflight (`_preflight_banner`) | the relaunch prints its axes; a repeat of the `--v2-cache`/`--v2` conflation is structurally visible |
| efficiency: v1-class plan step p50 ≈ 138 ms = **140 % of the 100 ms budget** at 276.9 M params | the efficiency family gates v5-flagship too; 256 anchors × diffusion at 176×624 needs its own p50 measured at the 10k milestone, not assumed |
| two-rig cy (rig A cy≈543 / rig B cy≈755; centre crop ~215 px wrong for rig B) | v5's per-clip geometry cache already handles this — **verify at relaunch preflight**, do not assume |
| speed sign flip (v1 +1.465 too fast; v2corpus −1.260 too slow) | signed `speed_bias` is a first-class gate metric — an arm can pass MAE while flipping sign |
| E-DPSI null below 12° | no anticipation claim enters the gate below that threshold |

---

## 2. THE REDESIGN — v5-flagship spec (delta from running v5)

1. ⭐ **`cond_imagination = True`** — expose `--cond-imagination` as a trainer flag (un-hard-wire
   `:1242`), default ON for the flagship. This is the PI's P1 sentence made real, and the
   three-planner directive ("each planner predicts via imagination") applied. Cost: +32 tokens
   (8 probes × 4 read horizons), each probe a 20-step predictor roll — **step-time hit ESTIMATED
   1.2–1.5×, must be MEASURED over 200 steps before committing to 30k** (the RR-20 rule: measure
   `step_s`, then commit).
2. **Keep:** 256-anchor diffusion head, `cond_vtarget`, `cond_route`, separated
   `w_lat/w_lon/w_dist/w_jerk/w_curv/w_strat`, v2.1 labels, 176×624@120° cylindrical, parity
   corpus, from-scratch, offline gate, preflight banner.
3. **`goal_dropout`: decided by the 5k guard**, not by taste — ≥0.9 route retention ⇒ keep 0.5;
   ≈0.5 ⇒ lower to 0.25 or ramp 0→0.5 over phase_b.
4. **Phase-2 rollout-recovery** (P5) with `w_curv` active; k from the k-sweep.
5. ⭐ **Generalise `hierarchy.py` to the anchor head** so TAC/STR are scorable on v5 — without
   this, "flagship" status cannot be claimed under the binding rule. (0-GPU implementation.)
6. **Gate at every milestone [5k, 10k, 15k, 20k, 30k]:** four families + efficiency (p50 vs
   100 ms) + `speed_bias` sign + κ + route-vs-majority, offline, both outcomes pre-registered per
   milestone.

## 3. THE RESTART PROTOCOL — ordered, so nothing burns compute blind

| step | what | cost | blocks |
|---|---|---|---|
| R1 | **Run the pre-registered 5k guard** on `v5_modelonly.pt` (hierarchy pass; needs the instrument generalisation of #5 for TAC/STR, LON/LAT run regardless) | ~3 min GPU | a GPU being up |
| R2 | **Implement** `--cond-imagination` + hierarchy generalisation + sha256 pin update (deliberate) | 0-GPU | — |
| R3 | **Measure imagination step-time** over 200 steps | ~1 GPU-h | R2 |
| R4 | **Relaunch as `flagship-v5f-w120-30k`** with the spec above; preflight banner on | ~3.5–5 GPU-d | R1–R3 + PI decisions |
| R5 | Milestone gates; **Phase-2 RR** after 30k; k-sweep in parallel if a second GPU exists | — | R4 |

⚠️ **Fleet reality:** pod2 still runs v5 (5,300 @ 12 s/step, 33 % GPU just now — I/O phase);
newpod still runs v1arch. Credit was $3.61 this morning. **R4 needs the PI to fund/start a pod.**
Continuing the current v5 to 30k as-is costs ~3.5 days and produces a model that P1 and P2 already
say is not the flagship design — **the review's recommendation is restart, not continue.**

## 4. DECISIONS OWED BY THE PI

| # | decision | recommendation |
|---|---|---|
| D1 | **Restart v5 as v5-flagship with `cond_imagination=True`** (discards 5,300 steps ≈ 17 h) | ⭐ **YES** — the gap is the design's core, and 25k steps of the wrong design cost more than 17 h of the right one. Keep pod2's v5 running only until R1 verdicts land, then kill by explicit PID. |
| D2 | **Corpus for v5-flagship** | **parity** now; v2bal only after the v1arch contrast + clean val (P4) |
| D3 | **goal_dropout** | let the R1 guard decide (pre-registered) |
| D4 | Fund a pod for R3+R4 (~4–5 GPU-days total) | required for any restart |

## Evidence class

| claim | class |
|---|---|
| every v5 config value | **MEASURED** — the run's own `config.json`, rescued 2026-08-02 |
| `cond_imagination` hard-wired off; 0 tokens | **MEASURED** — `flagship_v15.py:342`, `train_flagship_v4.py:1242` |
| 0/3 seams; κ 0.253 / 0.0072; route collapse | **MEASURED** — `hier_*.json`, 19 leak-free eps, n=418 |
| RR-20 verdict + trade | **MEASURED** — n=881, paired episode-cluster bootstrap B=2000 |
| step 5,300 alive | **MEASURED** — pod log 12:39 UTC |
| "seam failure = imagination gap" | **HYPOTHESIS** — the gate is designed to decide it |
| imagination step-time 1.2–1.5× | **ESTIMATED** — R3 measures it |
| "RR curvature cost mitigated by active `w_curv`" | **HYPOTHESIS** — pre-registered, both outcomes stated |
