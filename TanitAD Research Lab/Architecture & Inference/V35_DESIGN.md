# Flagship v3.5 — Design: the synthesis of every proven asset

> 🟥 **SUPERSEDED 2026-07-21 by `V4_FLAGSHIP_DESIGN.md`. Retained as provenance only — do NOT quote
> this document as current.** Six of its positions were overturned by measurements taken after it was
> written (§1 of the v4 doc lists them with the killing number each): the encoder-widening
> recommendation (§7.1), Branch 1 (v1.6 ran and failed), alternative ② (the second-encoder KV source),
> the S2 FiLM seam, the ≈345–365 M budget, and the imagination-latency/CEM concern (§7.5). A seventh —
> `--aux-accel` / `--jerk-weight` as part of "the v1 recipe verbatim" — was retired by the v3enc
> post-mortem. What survives is carried forward verbatim into v4 §1.

**Date:** 2026-07-20 · **Status:** 🟥 SUPERSEDED (was: DESIGN — awaiting Sayed's line-by-line gate approval, §11)
**Charter (Sayed):** *"combine all our proven assets… this could be the real flagship v3, let's call it v3.5."* Push results to the best possible BEFORE new research directions.
**Evidence bar:** every "proven" below cites a raw eval JSON, a run `config.json`, `Project Steering/MODEL_REGISTRY.md` (§-refs), or a commit body whose numbers were re-checked against the raw JSON this session. Live numbers were re-read from `tanitad-eval:/root/taniteval/results/*.json` on 2026-07-20. Anything not so grounded is marked **ASSUMPTION**.
**This is a design + evidence document, not a training order.** Nothing here launches until the gates in §11 are approved.

---

## 0. Executive summary

**v3.5 = the flagship v1 world model and REF-C's proposal machinery fused into one hierarchically-wired ≤400 M stack, reached by the cheapest gated path: finish the LP-FT ladder that v1.5 started (v1.6 = unfreeze the trunk on the `ab` arm), and only if that fails to buy REF-C-grade proposals, run one joint from-scratch consolidation train (v3.5-J) that keeps the v1 recipe verbatim and adds exactly the levers that are individually proven: the anchored-diffusion tactical decoder, imagination conditioning, v2.1 labels + fixed VTARGET data, speed-input, jerk/aux-accel, milestone+atomic ckpt ops, and an H26 norm cap on every conditioning seam.**

**Two constraints from Sayed (2026-07-20) reshape this design and are now binding:**

1. **The hierarchical wiring is a REQUIREMENT, not an option** — strategic goals → tactical planner → operative predictor, **all trained jointly**. §2A is the full spec (wiring diagram, per-seam mechanism, hard norm caps, goal-dropout, the causality gate, and the S3 circularity decision). Consequently **the "goals are inert" finding is RE-OPENED, not carried forward** (§2A.1): v1.5 tested goals on a *frozen* trunk, at a seam measured **2.80× swamped**, for 8 k steps, mid-label-repair — that is a broken test, not a null result.
2. **Parameter cap raised 300 → 400 M** — §7 re-prices everything. Headline: **decoder width is NOT the lever** (v1.5 runs ~1.3× REF-C's decoder and proposes 1.87× worse), so the extra ~140 M goes to the **encoder** (REF-C spends 79 % of its budget there vs flagship's 33 %, and owns the better fan). Recommended build ≈ **345–365 M**. Alternative ② is re-priced honestly: viable **only** with REF-C-*base*'s 90.5 M encoder (≈348 M ✓); the XL variant is ≈457 M and **still over cap**.

**Inference efficiency is now a first-class constraint** (§7.5): target **≤50 ms tick**, and the design's own best lever — imagination conditioning — is also its dominant latency term (20 sequential batched predictor steps), so probe count is an explicit accuracy/latency knob.

**The architecture answer is CONDITIONAL on v1.6, which is now LIVE** (launched 20:01 UTC on pod2, verified by live probe — trunk-LR 1e-5 vs head 1e-4, `--unfreeze-enc-blocks 4 --unfreeze-predictor`, 6 k steps). §2.2 specifies the branch where the v1.5 lineage becomes v3.5's skeleton; §2.3 specifies, per failure diagnosis, which alternative fires (joint-from-scratch ① · REF-C-base-encoder-as-KV ② · fan distillation ③). ⚠️ **Its first val row is a red flag** (§11 G-A note).

The two parents and what each contributes:

| Parent | Proven strength (cited in §1) | What v3.5 takes |
|---|---|---|
| flagship v1 (0.4522) | best mean, best tail (miss@2m 0.060), causal vision proof, closed-loop-able rollout, low/med-speed dominance | the trunk: encoder + operative predictor + grounding + h15 imagination |
| REF-C-XL (0.458) | best proposal set in the program (oracle-in-fan 0.1640), wins high-speed outright (0.324 final vs 0.551) | the decoder: anchored truncated-diffusion head as the tactical brain, its FPS anchor vocabulary, its loss recipe |
| v1.5 (bridge, failed its gates) | the ONLY proven conditioning mechanism: imagination tokens −0.1355 m, CI-significant | the `ab` conditioning pattern and the LP-FT ladder |

### ⚠️ Escalations / open decisions (headline)

1. **v1.6 is LIVE and its first val row is a RED FLAG (fail-loud).** Timeline verified by probe: pod2 was idle at my first check (17:17 UTC, no trainer); v1.6 launched **20:01 UTC** and is now at step ~950/6000. **Step-500 val: `oracle_ade@2s` 1.27144, `ade@2s` 1.62093, miss@2m 0.593** — against v1.5-`ab`'s 0.3073 / 0.5437 / 0.1643. The fan got **4× worse**, not better. Mitigating context: trunk-LR was still ~0 at that eval (ramp starts at 500), head LR 1e-4 is re-shaping a head that was already converged, and 500 steps is very early. **The WM-integrity canary is healthy and already instrumented in the trainer** (`canary_ade@2s 0.4215, canary_vs_base −0.00015` — the trunk is intact), which is exactly the guard §11 specifies. **Decision 1: read v1.6's 1.5–3 k rows before committing to Branch 1; if the fan does not return below v1.5-`ab`'s 0.3073 by ~3 k, §2.3's alternative ① fires.**
2. **Decision 2 — skeleton choice is gated, not made:** if v1.6 passes (G-A, §11) the v1.5 lineage IS v3.5 (v3.5-A, ~0.5 GPU-day spent total); if it fails, the alternative is specified per failure mode in §2.3 (joint-from-scratch v3.5-J ~4 A40-days is the recommendation; REF-C-KV and fan-distillation are the named runners-up). §2 gives the full decision table.
3. **Decision 3 — v3enc is DOWNGRADED to unproven-trending-weak; its levers are OUT of v3.5 by default.** Measured this session from the live log (`flagship4b-v3enc-30k/train_log.jsonl`, step 5,050): on the **canonical 1.5k-window the v2 diagnostic itself used**, v3enc's `g_op_fwd_ade_m` power-law exponent is **−0.503 (R² 0.225, n=72)** — i.e. sitting on v2's **−0.50**, not v1's **−0.84**. Adoption now requires **two** things, not one: a passed 10 k ADE gate AND an exponent that separates from −0.50. See the fit-stability caveat in §1.2/N7 — I do not claim v3enc is refuted, only that it cannot currently be called an asset.
4. **Decision 4 — REF-C base (104.2 M) lands tonight** (step 3,800/30,000 at last report). It is the first real scale rung. What it would change is stated in §3.6: it can only *re-price* v3.5's decoder, never block it — but a base-beats-XL result would cut the decoder budget and is worth waiting one day for before committing to G-C's geometry.
5. **Registry defect found while verifying (fail-loud):** MODEL_REGISTRY §4.1's FINAL REF-C-XL row carries **stale 16 k strata** ("low 0.708 / med 0.586 / high 0.521, straight 0.523"). The raw final JSON (`refc-xl-30k.json`, read live) says **high 0.3243 / low 0.5912 / med 0.4989**. The brief's "0.330 vs 0.551" is the step-28k provisional pair; the final pair is **0.3243 vs 0.5513** (REF-C's win got *bigger*). Registry fix flagged as follow-up (§12).

---

## 1. The verified asset inventory

Each row was re-verified this session. ✅ = number re-read from the primary artifact named in "Source".

### 1.1 IN — positive, proven

| # | Asset | The number | Source |
|---|---|---|---|
| P1 | **flagship v1 trunk** | ADE@2s **0.4522 ± 0.0312** heldout / 0.4271 full-set; **miss@2m 0.0602** (best in program); FDE 0.9437 ✅ | `results/flagship-30k.json` (re-read live); REGISTRY §1.2 |
| P2 | v1 stratum profile | low **0.3594** / med **0.3704** / high **0.5513**; straight 0.3931 / gentle 0.5158 / sharp 0.5128 ✅ | `flagship-30k.json` `by_speed`/`by_curvature` (read live) |
| P3 | v1 failure signature | 89 % of 2 s squared error longitudinal; high-speed speed over-predict +0.66 m/s | REGISTRY §1.2 (pathspeed) |
| P4 | **v1.5 imagination conditioning** — the only proven conditioning mechanism | `a`→`ab` **−0.1355 m (−19.9 %), CI [0.038, 0.233] — SIGNIFICANT**; 0.6792 ± 0.0729 → 0.5437 ± 0.0653; drives miss@2m 0.2359→0.1643 ✅ | raw `flagship-v15-{a,ab}-ckpt.json` (in-repo, `…incoming/2026-07-20-vtarget-validation/`); commit `fc2c484` |
| P5 | **REF-C-XL** | ADE@2s **0.4577 ± 0.057** heldout / 0.4714 full-set (final 29,999) ✅ | `results/refc-xl-30k.json` (re-read live); REGISTRY §4.1 |
| P6 | REF-C proposal set — best in program | **oracle-in-fan 0.1640** corpus (n=881); v1.5's frozen-trunk fan: `ab` 0.3073 / `abc` 0.3377 ✅ | REGISTRY §4.1; commit `fc2c484` table |
| P7 | REF-C **high-speed win** | final **0.3243 vs flagship 0.5513** (CV 0.6468, n=294) — the only stratum flagship loses ✅ | `refc-xl-30k.json` `by_speed` (read live); 28k provisional was 0.3301 (`2026-07-20-refb-refc-final-eval.md` §3) |
| P8 | REF-C decoder economics | anchored-diffusion decoder **22,702,345 params**, 256 FPS anchors, 2 truncated denoise steps, noise_std 0.1 — *smaller* than v1.5's ~32 M head ✅ | REGISTRY §4.1 params + run config; `flagship_v15.py` docstring |
| P9 | Anchored tactical decoder works inside a from-scratch stack | REF-B v2 (`--arch-v2`, time-anchored decoder): 0.5921 ± 0.0685 final, **first REF-B to beat CV in every speed stratum** (@20 k); flagship-v2's same lever had healthy per-lever telemetry when the run died of *simultaneity* | REGISTRY §3.5, §1.3 diagnosis, D-031 |
| P10 | **v2.1 route labels** | coverage 26.0→**81.9 %**; unlearnable genuine turns 63.1→**8.9 %**; roundabout wrapped-`net_dyaw` sign-flip fixed; ≥45° referee: zero false-straights | commit `a0645a7` (numbers table); REGISTRY §4.3 coverage cross-check |
| P11 | v2.1 effect location | improves the **proposal set** (oracle −0.0583, `abc_legacy`→`abc`) more than end-to-end ADE (−0.025 heldout, **n.s.**) ✅ | commit `fc2c484` decomposition table; raw `flagship-v15-abc{,_legacy}-ckpt.json` |
| P12 | **Fixed VTARGET mint** (data/cost input ONLY) | defect: `VT_LOOK_LO=100` defined-never-used (realized lookahead 3–19 s; only 53.5 % had ≥10 s); fix = smooth-before-gate+percentile, **5 s floor, explicit `valid` mask, DROPPED embedding row**; **82.91 % of windows trustworthy** | commit `c4f75d6`; `vtarget_validation_train.json` (in-repo, `mint_constants` block) ✅ |
| P13 | **speed-input (`v0` 3rd action channel)** — biggest fix in program history | REF-A fwd-ADE **3.73 → 0.83**, speed-R² 0.61 → 0.965 (isolated); causal pair: flagship no-speed 2.918 vs speed 0.4522, paired **+2.21 m CI [2.04, 2.39]** | REGISTRY D-A3, §1.1 vs §1.2 |
| P14 | Ops: milestone ckpts (5k/15k/20k/30k) + atomic ckpt_io + RAM guard | the REF-A ceiling-vs-overfit verdict was only possible because of D-032; v3enc's first attempt died in a non-atomic ckpt write on a 98 %-full overlay | REGISTRY D-032, §1.4 |
| P15 | **H26 norm-parity monitoring** | v1 intent seam: `intent_proj` norm 31.4 swamps act-emb 28.3 (harmful, cos −0.238); v1.5 `abc`: **ROUTE seam 2.80× measurement norm despite rt_gate 0.10 — the monitor FIRED** | HYPOTHESIS_LEDGER 07-18 H26 entries; commit `3d41bd0` flag 2 |
| P16 | LP-FT ordering | v1.5 = LP (head on frozen trunk) trained stably to 0.5437 in 8 k steps; v2 = everything-at-once diverged (P-N6) | commits `c4f75d6`/`fc2c484` vs REGISTRY §1.3 |

### 1.2 OUT — negative, structurally excluded (each with its number)

| # | Excluded | The killing number | Source |
|---|---|---|---|
| N1 | Post-hoc selection / re-ranking of a fixed fan | ≤ **8.4 %** of the oracle gap learnable (47 trained arms; ~92 % aleatoric); hand-written cost **0.0 %** (λ=0 optimal; pure cost −171 %) | REGISTRY §4.1 (v1.0/v1.2 table); commits `7227537`, `f7dec15`, `e185ee6` |
| N2 | Ranking on unsupervised denoise-time confidence | refined-pass confidence selection **1.36593 = 2.9× WORSE** than baseline. **Rule: if v3.5 ranks on post-denoise confidence, the conf head MUST be supervised at denoise timesteps** (v1.5 already does this: ranking CE on `sel_score`, the argmax quantity) | REGISTRY §4.1; commit `c4f75d6` |
| N3 | VTARGET as a 2 s ranking/selection signal | GT-perfect speed-matcher **1.1236 vs baseline 0.4714** (worse); VTARGET vs v@+2s: MAE 1.632, bias +1.464 m/s — it is a 10–20 s aspiration | REGISTRY §4.1; `vtarget_validation_train.json` `agreement_with_achieved_future_speed` ✅ |
| N4 | ~~Goal conditioning~~ → **RE-OPENED, NOT EXCLUDED (see §2A.1)** | The measured null stands but is **narrow**: `ab`→`abc` **+0.0106 m**, CI [−0.094, +0.072] — *goal tokens bolted onto a **frozen** fan at a seam measured at **2.80× swamped**, after only 8 k head-only steps, mid-label-repair*. That is not a test of the hierarchy thesis. **v3.5 wires goals into a jointly-trained, norm-capped stack (§2A) and re-tests.** What stays binding: goals do not rescue a frozen fan | commit `fc2c484`; raw v15 JSONs ✅; swamping from `3d41bd0` flag 2 |
| N5 | Frozen-generic-encoder line; supervised head as the decider | REF-A ceiling **2.92** (monotone milestone curve = capability limit, not overfit; H4 closed); v1 tactical head **3.1501** vs the same model's rollout 0.4522 — heads are lossy readouts | REGISTRY D-A5, §2.3; `planner_p2_flagship-30k.json` `tactical_head` ✅ |
| N6 | Simultaneous encoder-grounding levers | v2 killed: power-law exponent **−0.50 vs v1 −0.84**; v1 reached v2's 7.5 k value at step ~250; projected ~9× worse per A40-day | REGISTRY §1.3, D-031 |
| N7 | v3enc's staged levers | **UNPROVEN, TRENDING WEAK** — zero evaluated checkpoints; and on the canonical **1.5k–5.05k** window (the window the v2 diagnostic used, §140 of that note) its exponent fits **−0.503, R² 0.225, n=72** — on top of v2's **−0.50**, nowhere near v1's **−0.84**. Level check: `g_op_fwd_ade_m` 0.649 (1.5–2 k) → 0.456 (≥4.5 k): improving, slowly. ⚠️ **Fit-stability caveat, stated rather than hidden: this exponent is window-unstable and poorly determined** — same log, same key, different windows: −0.379 (all ≤5 k, R² 0.573), −0.563 (≥1 k, R² 0.396), −0.774 (≥2 k, R² 0.294), −0.82 (last-60 ≤5 k, R² 0.308). Every R² is 0.22–0.57; step-to-step noise is large (last 3 rows: 0.295 / 0.480 / 0.414). **Honest verdict: the exponent cannot presently discriminate v1-like from v2-like; only the canonical-window fit is comparable to the −0.84/−0.50 pins, and it says v2-like.** Post-5 k (rollout-k 4→8) has only just begun and is the remaining test | live log fit this session (`train_log.jsonl`, step 5,050) ✅; window convention from `2026-07-19-flagshipv2-6k-diagnostic.md` §"Power-law fit (1.5k–7.5k)"; REGISTRY §1.4 |

### 1.3 CONTEXT — regime facts the design must respect

| # | Fact | Number | Source |
|---|---|---|---|
| C1 | Open-loop does not predict closed-loop | v1: 0.452 open → **1.685 ± 0.098 closed**, divergence >5 m **22.2 %** | REGISTRY §1.2 |
| C2 | **Planning-over-WM: what is ACTUALLY measured** (framing corrected per the 360° review) | P2 CEM over frozen v1 beats **the head** (open-loop 0.8929 vs head 3.1501) and wins **closed-loop**: 1.0377 ± 0.2022 vs 1.685 ± 0.098 (38 % less drift), divergence **8.7 % vs 22.2 %**. ⚠️ **It does NOT beat CV open-loop — 0.8929 vs CV 0.8248 is a LOSS, not a tie.** The honest claim is "planning beats the supervised head, and reduces closed-loop drift" — **not** "planning-over-WM is proven superior in general" | `planner_p2_flagship-30k.json` ✅; REGISTRY §5 |
| C3 | P2's residual is 66 % lateral (no lateral cost term by design) | curved-stratum planner 2.114 vs true-action 0.484 | REGISTRY §5 / D-A10 |
| C4 | REF-C's deficit vs v1 is curves + tail, not the mean | curve Δ −0.15/−0.22 m; straights Δ 0.0000 with REF-C median *better* (0.219 vs 0.347); miss@2m 0.146 vs 0.060 | `2026-07-20-refb-refc-final-eval.md` §3–4 |
| C5 | VLM labels: **595 records banked** (verified: exactly 595 JSONs in `tanitad-eval:/root/vlm_pilot/bulk/out/`, +38 quarantined) ✅ | direction = chance (52.7 %; replicated 57.1 %, CI [0.400,0.745] — MODEL bias, not prompt: enum-order probe 07-21); turn RECALL **~78 %** (77–81 % across three independent measurements — 76.8 / 80.6 / 78.6; the old "89.3 %" was AGREEMENT on a 74 %-straight corpus AND does not reproduce; `2026-07-21-cosmos-reason2-production-semantic-labeling.md` §1); LONMODE render-stability 62 %; lead present 277 / HEADWAY minted 268 of 522 measured keys | live count this session; commits `a0645a7` Part 2, `547c8ec`; `enrich.log` |
| C6 | OOD is the open generalization gap (not addressed by v3.5; owned by v3enc/v3) | comma2k19 0.849 vs floor 0.372 (win 17.5 %) | REGISTRY §1.2 OOD panel |

---

## 2. Q1 — Architecture: one integrated model, reached by a gated two-path plan

**Recommendation: v3.5 is ONE integrated model — the flagship 4-brain trunk with REF-C's anchored-diffusion decoder installed as the tactical brain, conditioned v1.5-`ab`-style (states + imagination). There are two ways to get there; which one is v3.5 is decided by the v1.6 gate, not by taste.**

### 2.1 Path A — v3.5-A: the v1.5 lineage completes (LP-FT; cheapest; **run first**)

v1.6 = unfreeze the v1 trunk under the `ab` head (LP-FT step 2; the exact lever `fc2c484` names: *"the lever is unfreezing the trunk, more head steps, or richer anchor/offset capacity — NOT more scoring"*). **v1.6 has NOT started training** (Escalation 1 — verified by live probe, and confirmed by the coordinator: its agent died before launch). It is Gate A of §11, and the architecture answer below is **conditional on its outcome, both branches specified**.

### 2.2 Branch 1 — v1.6 closes the oracle gap ⇒ **the v1.5 lineage IS v3.5's skeleton**

Trigger: oracle-in-fan moves decisively from **0.3073** toward REF-C's **0.1640** (gate: ≤0.22) **and** G1/G2/G3 pass (beat 0.458, beat 0.4522, miss@2m ≤0.10) **and** the WM-integrity gate holds (§11).

Then v3.5 = frozen-lineage-descended integrated model: v1 trunk (now fine-tuned) + REF-C anchored-diffusion decoder as the tactical brain + `ab` conditioning. Total spend ~0.5–1 GPU-day, no 30 k run needed, and the deployed v1 remains the trunk's provenance. v3.5-J drops to an optional consolidation (S4 polish, §4).

### 2.3 Branch 2 — v1.6 does NOT close the gap ⇒ the specified alternatives, in recommended order

The diagnosis determines which alternative fires — these are not interchangeable:

| If v1.6 fails like this | Diagnosis | Alternative that fires | Cost | Evidence / caveat |
|---|---|---|---|---|
| Fan improves but **stalls >0.25**; trunk healthy | Proposal quality is bought **during** full training, not bolted on: REF-C's 0.164 comes from a 200 M conv encoder trained end-to-end for 30 k steps *with the decoder attached* (REGISTRY §4.1) | **① v3.5-J joint from-scratch** (§2.4) — **the recommendation** | ~4 A40-days | Strongest evidence base; also the only path that lets decoder gradients shape the encoder from step 0 |
| Fan barely moves **and** trunk degrades under decoder gradients | The v1 latent geometry cannot host a REF-C-grade fan; more gradient makes it worse, not better | **② REF-C encoder as a second KV source** — keep the WM trunk for rollout/closed-loop, cross-attend REF-C's conv feature map for proposals | ~1–2 A40-days (decoder-only train, both trunks frozen) | **Re-priced at the 400 M cap (§7.4): viable ONLY with REF-C-*base*'s 90.5 M encoder (≈348 M total ✓). The XL variant is ≈457 M and remains OVER CAP even at 400 M.** Gated on tonight's REF-C-base result; costs a second encoder pass per tick (§7.5) |
| Fan barely moves, trunk healthy, **and** GPU time is the binding constraint | Cheapest way to import fan quality without importing the encoder | **③ Distillation from REF-C's fan** — offline top-K trajectories from `refc-xl-30k` as auxiliary WTA targets | ~0.3 A40-day | ⚠️ **ASSUMPTION: unproven in this program.** Nearest evidence is *negative*: REF-A's feature-distill converged on train (cos 0.9998) but FAILED to generalize (held cos 0.60) and never beat a plain linear read (HYPOTHESIS_LEDGER 07-17). Capped at teacher quality (0.164) by construction. Fallback, not a plan |
| Trunk breaks specifically when the **encoder** unfreezes | Decoder gradients are corrupting perception, not dynamics | **retry once**: encoder frozen, predictor unfrozen (§11 G-A fallback), then re-enter this table | +0.5 A40-day | LP-FT ordering (P16) |

### 2.4 Path B — v3.5-J: one joint from-scratch consolidation run (alternative ① above)

`flagship4b` trunk trained with the **v1 recipe verbatim** (P13: speed-input, jerk 0.02, aux-accel, SIGReg free-dims 64, rollout-k 4, AdamW 3e-4/wd 0.05/warmup 2000) **plus only individually-proven decode-side additions**:

1. **Anchored-diffusion tactical decoder replaces `tactical_policy`'s unimodal `wp_heads`** — proven inside from-scratch stacks by REF-B v2 (P9) and healthy-per-telemetry in flagship-v2 (D-031); geometry from REF-C-XL (256 FPS anchors, d512, 2 truncated denoise steps, P8).
2. **`ab` conditioning**: decoder KV = readout state tokens + imagined future latents from the (co-training) operative predictor under the 8-probe vocabulary — mechanism proven at −0.1355 m (P4). **ASSUMPTION flagged loudly: P4 was measured on a FROZEN 30 k trunk; that the same conditioning helps while the trunk is still forming is plausible but unproven — this is exactly what the 5 k/15 k milestone gates in §11 test, with the v2-style rate-of-learning kill-switch (N6).**
3. **No goal tokens in the decoder** (N4). VTARGET/ROUTE enter as *labels and cost inputs* only (§8).
4. **H26 norm cap** on the imagination-conditioning seam: per-seam contribution-norm ratio logged every step, hard cap ≤ 1.0× measurement norm (P15 — the 2.80× ROUTE swamp partially confounded `abc`; do not reuse the pattern uncapped).
5. Encoder-grounding levers (decorr, fa-dropout, invdyn-gradscale, rollout-k 12): **OFF — default, and the default hardened this session.** v3enc's canonical-window exponent is −0.503, sitting on the arm we killed (N7). Turning any of them on requires a passed 10 k ADE gate **and** exponent separation from −0.50; absent both, v3.5-J trains the v1 recipe untouched (P1/P13).

### 2.5 Alternatives considered and rejected outright (not branch-dependent)

| Alternative | Why not |
|---|---|
| Two-model system as the **default** (flagship trunk + REF-C-**XL** as a permanent proposal server) | ≈457 M — **over cap even at 400 M** (§7.4) — and doubles the deploy tick (§7.5); P9 shows the decoder works inside one stack. (The REF-C-**base**-encoder form fits at ≈348 M and is retained as the *conditional* alternative ② in §2.3) |
| Adopt REF-C-XL as the base and add a WM to it | Throws away the tail king (miss 0.060 vs 0.146, C4), the causal vision proof, closed-loop-ability (REF-C has no operative rollout, `step_readout=None`), and the P2 machinery |
| Wait for v3enc and build v3.5 on it | v3enc is unproven (N7) and answers a different axis (OOD/encoder-grounding, C6). v3.5 must not take a dependency on a pending result — it takes its *outcome* as an optional lever |

---

## 2A. THE HIERARCHICAL WIRING — a REQUIREMENT, not an option (Sayed, 2026-07-20)

> **Sayed, verbatim intent:** *"we need operative predictor getting actions, inputs, strategic AND tactical goals. The tactical layer gets strategic goals. All of them are trained TOGETHER."*

This is **H26 — hierarchical cross-alignment — the program's declared core-goal edge** (HYPOTHESIS_LEDGER 07-18: *"the CORE-GOAL proof that hierarchical E2E world-understanding+planning is efficient & dominant"*). It is now a **design requirement of v3.5**, and it supersedes the earlier draft's `ab`-only conditioning.

### 2A.1 Why "goals are inert" must NOT be carried forward as a settled negative

The v1.5 `ab`→`abc` null (**+0.0106 m**, CI [−0.094, +0.072]) was a **weak test of the hierarchy thesis, not a fair one**. Four independent reasons, each documented:

| # | Why v1.5 could not test this | Evidence |
|---|---|---|
| 1 | **The trunk was FROZEN.** Goals could not shape features at all — they could only re-weight a fixed fan. The thesis is that goals shape *representation*; v1.5 structurally forbade that | `flagship_v15.py` docstring: *"Nothing in the trunk receives a gradient; nothing in the trunk is even in the optimizer"* |
| 2 | **Only 8 k head-only steps.** The one seam that ever flipped to load-bearing in this program did so **between 19 k and 30 k of training** (`ctx→tactical`, content_matters false→true) — duration alone moved it | HYPOTHESIS_LEDGER 07-18 (H26 @30 k: seams 0/3 → 1/3) |
| 3 | **The H26 swamping monitor FIRED** — the ROUTE seam's contribution norm was **2.80×** the measurement norm despite `rt_gate` 0.10. That is a **mis-scaled implementation**, i.e. a broken seam, not a null result. The program has seen this exact bug class corrupt a seam before (v1's `intent_proj` norm 31.4 vs act-emb 28.3 → cos −0.238, *harmful*) | commit `3d41bd0` flag 2; HYPOTHESIS_LEDGER 07-18 |
| 4 | **Labels were mid-repair.** v2.1 landed alongside those arms; `abc` used v2.1 but no arm isolated *wiring* from *labels* — and labels demonstrably move the fan (oracle −0.058) | commit `fc2c484` decomposition; P10/P11 |

**Ruling for this design: N4 is RE-OPENED, not excluded.** "Goal conditioning is inert" is plausibly an artefact of *frozen trunk + swamped seam + 8 k steps*. The claim that survives v1.5 is narrower and stays binding: **goal tokens bolted onto a frozen fan, at a mis-scaled seam, do not improve ADE.** v3.5 tests the actual thesis — goals wired into a **jointly trained** stack at **norm-capped** seams — for the first time.

### 2A.2 The wiring diagram

```
            nav / ego / scene
                    │
            ┌───────▼────────┐
            │   STRATEGIC    │  (d384×4, existing)
            │  goal module   │  emits G_s = ⟨ROUTE, VTARGET-band, mission/ODD envelope⟩
            └───────┬────────┘
                    │ G_s  ── S1: KV tokens ──┐        ── S2: FiLM ──┐
                    │                         │                      │
            ┌───────▼─────────────────────────▼──┐                   │
            │   TACTICAL = THE ANCHORED-DIFFUSION │                  │
            │   PLANNER ITSELF                    │                  │
            │   256 anchors → fan → select        │                  │
            │   the SELECTED PLAN *is* the        │                  │
            │   tactical goal  G_t                │                  │
            └───────┬─────────────────────────────┘                  │
                    │ G_t (selected plan)                            │
                    │ ── S3: inverse-dynamics → future-action seq ──┐ │
                    │                                              │ │
            ┌───────▼──────────────────────────────────────────────▼─▼──┐
            │   OPERATIVE PREDICTOR                                      │
            │   inputs: window states + actions + v0  (existing)         │
            │         + G_s (strategic)  + G_t (selected plan)  ← NEW    │
            │   → latent rollout → grounded metric Δpose                 │
            └────────────────────────────────────────────────────────────┘
                         ALL THREE TRAINED JOINTLY (the key delta vs every v1.5 arm)
```

### 2A.3 Seam specification — mechanism, why, and the norm discipline

| Seam | From → To | **Mechanism** | Why this mechanism | Norm discipline |
|---|---|---|---|---|
| **S1** | strategic G_s → tactical planner | **Cross-attention KV tokens** (2–4 tokens appended to the decoder's existing heterogeneous KV set) | The decoder is *already* a cross-attn stack over a mixed token set (v1.5 proved this plumbing works with state + imagination tokens). Appending tokens is the minimum-surface change, and per-anchor queries stay free to attend or not — attention is itself a learned soft gate | ReZero scalar per token-group, **init 0.1**; hard cap: token-group contribution norm ≤ **1.0×** the measurement-token norm |
| **S2** | strategic G_s → operative | **FiLM on the predictor's conditioning vector** | The operative's conditioning path is a *vector*, not a token set — this is v1's native seam type (`strategic ctx --FiLM--> tactical --intent FiLM--> operative`). Do not invent a second interface | **Norm-matched projection**: LayerNorm the cond + scale to the action-embedding norm (this is the literal H26 fix: v1 failed at 31.4-vs-28.3). ReZero init 0.1, cap ≤1.0× |
| **S3** | tactical plan G_t → operative | **Inverse-dynamics → future-action sequence, entering the predictor's EXISTING action channel** | ⭐ The operative predictor is *already action-conditioned* (`action_dim 3`). A selected plan is a future trajectory; converting it to an action sequence means **the plan enters through a proven port instead of a new seam** — no new swamping surface, and it makes training identical to what P2's CEM already does at inference (roll the WM under a candidate action sequence). The program already owns the inverse-dynamics head (`inv_dyn 0.5` in the v1 loss) | No new gate needed — the action channel's scale is already calibrated (SPEED_SCALE contract). Still monitored: log the plan-action norm vs the observed-action norm |

**Every seam carries goal-dropout ≥ 0.5** (the H25/H26 anti-shortcut rule; v1.5 already ran 0.5 — `flagship_v15.py`: *"without it the head reads the goal and ignores vision"*).

**Norm caps are a HARD GATE, not a log line** (the explicit ask). Implementation: each seam computes `ratio = ‖contribution‖ / ‖measurement‖` every step; the contribution is **rescaled in-graph** whenever `ratio > 1.0`, and a `ratio > 1.5` event **fails the run loud** (it means the rescale is being fought). The 2.80× ROUTE event that confounded `abc` would, under this rule, have been clamped at 1.0 and alarmed.

### 2A.4 The circularity / leakage problem in S3 — and the decision

S3 feeds the operative a plan **selected from the model's own fan**. Three real hazards, each with a decision:

1. **Degenerate co-adaptation.** If the operative's loss can flow back into the decoder through S3, the decoder can minimise operative loss by proposing *easy-to-predict* plans rather than *good* plans. → **DECISION: stop-gradient on G_t where it enters the operative.** The plan is a conditioning input, never a differentiable path back into the tactical decoder. The decoder is trained by its own anchor-CE + L1 only.
2. **Train/inference mismatch vs early-training garbage.** Teacher-forcing the GT-assigned anchor is stable but never occurs at inference; own-selected plans match inference but are noise for the first few thousand steps. → **DECISION: scheduled sampling.** Teacher-forced (the GT-nearest anchor the anchor-CE already computes) for the first ⅓ of training, then ramp linearly to own-selected by ⅔, own-selected thereafter. This is standard DAgger-style annealing. **ASSUMPTION: the ⅓/⅔ schedule is chosen by convention, not measured in this program** — the 5 k/15 k milestone gates read it.
3. **GT leakage.** During teacher-forcing, G_t is *derived from the future GT* while the operative is scored on predicting that same future. → **DECISION, two-part:** (a) goal-dropout ≥0.5 applies to S3 as well, so half of all steps train the plan-free path; (b) **the WM-integrity canary is evaluated with S1/S2/S3 ZEROED** — the intent-free rollout, exactly as v1's deployed rollout is intent-free by design. Any headline ADE must be reported **both** plan-conditioned and plan-free, and **no leaderboard number may come from a GT-derived plan.**

### 2A.5 The causality gate — the seam may not be claimed live without it

A seam that exists is not a seam that works; H26's whole lesson is that v1's seams *cohered but did not drive*. Before any "the hierarchy is load-bearing" claim:

| Test | Requirement |
|---|---|
| **Sensitivity** | Swap a goal token (ROUTE left↔right; VTARGET band ±k) ⇒ the conditioned output changes, **CI-separated** |
| **Correctness** | The change is in the **right direction** (route-right swap moves the plan's lateral offset right; VTARGET-up raises predicted speed) — sensitivity alone is satisfiable by noise |
| **Per-window content** | Use **`per_window_content_helps`** (real per-window goal vs a *constant/mean* goal), **not** `helps_vs_none`. v1's H26 showed `helps_vs_none` flips sign by encoder while content stayed inert — a constant offset can look like "help" | 
| **Effect-size floor** | The existing panel gates: CI-separated **and** ≥0.02 acc / 0.05 m / 0.01 cos |
| **Norm parity** | Every seam's ratio ≤1.0 across the whole run (§2A.3) |

This is **Gate H** in §11 — a first-class gate, run at 5 k and every milestone, not a post-hoc panel.

---

## 3. Q2 — The proposal engine: how v3.5 gets REF-C-quality proposals

The measured gap: REF-C fan oracle **0.1640**; v1.5 frozen-trunk fan **0.3073** (`ab`) — and every v1.5 arm's own oracle would clear G1/G2 if ranking were free (P6, `fc2c484`). Ranking is ~aleatoric-bounded (N1), so **the fan itself is the lever**.

**Recommendation (ordered by evidence):**

1. **Primary: buy the fan with trunk gradients** — Path A (unfreeze) then Path B (joint). Direct evidence: REF-C's oracle comes from end-to-end training (P5/P6); v1.5's LP-only fan is 1.87× worse; `fc2c484` names unfreezing as the lever. ⛔ **NOT by widening the planner — that option is removed entirely (§7.1): v1.5 already runs ~1.3× REF-C's decoder capacity and proposes 1.87× worse.** Capacity goes to the encoder (§7.2).
2. **Keep the REF-C anchor vocabulary verbatim**: reuse `refc_anchors_full.pt` (256 FPS anchors over real GT trajectories; FPS-not-kmeans because the corpus is ~74 % straight — REGISTRY §4). Zero-cost, removes one degree of freedom from the comparison.
3. **Keep v1.5's supervised-selection fix**: ranking CE applied to `sel_score` (the argmax quantity), confidences carried through the denoise pass (N2 rule).
4. **Fallback (only if Path B's 15 k fan gate fails, or per §2.3 alternative ③): REF-C fan distillation** — offline top-K trajectories from the finished `refc-xl-30k` ckpt as auxiliary WTA targets for the decoder. **ASSUMPTION: proposal distillation is unproven in this program** (nearest evidence: REF-A's feature-distill FAILED to generalize, held cos 0.60 — HYPOTHESIS_LEDGER 07-17; different substrate, weak transfer). Capped at teacher quality (0.164 oracle) by construction. Cheap (~0.3 GPU-day) but strictly a fallback.
5. **REF-C conv encoder as a second KV source — conditional, not default** (§2.3 alternative ②): +199 M breaks the single-model budget (§7), and C4 says REF-C's *decoder + end-to-end training* is the asset while its curve behaviour is the liability. Fires only on the specific diagnosis "v1 latent geometry cannot host the fan".

### 3.6 What REF-C-base (104.2 M) changes when it lands — it re-prices, it cannot block

REF-C-base is at **step 3,800/30,000** (coordinator, this session); it is 2.42× smaller than XL, uses **128 FPS anchors that are a strict prefix of XL's 256**, and carries the **v2.1-labels confound** (XL trained v1 labels — REGISTRY §4.3 ⚠️, so scale and labels are entangled in any base-vs-XL delta).

| REF-C-base outcome | What v3.5 changes | What it does NOT change |
|---|---|---|
| base ≈ XL (≈0.458) or better | Two things: (i) **the decoder can be trimmed** (base geometry d384/4-layer/128-anchor → ~9–15 M), freeing capacity for the encoder where §7.1 says it belongs; (ii) ⭐ **it validates REF-C-base's 90.5 M encoder as the second-KV candidate that makes alternative ② fit under 400 M (§7.4)** | The architecture. The decoder still installs as the tactical brain |
| base clearly worse than XL | Keep the **XL geometry** (256 anchors, d512, 6–8 layers) as specified in §7 — the current default | Nothing; this is the assumed case |
| base worse **and** its fan oracle ≫ 0.164 | Read as "fan quality scales with capacity" → argues *against* trimming the decoder and *for* Path B's full-capacity joint train | The gates |

**Recommendation: it is worth waiting ~1 day for this before committing G-C's decoder geometry** — G-A (v1.6) is independent and should start immediately regardless, so nothing is idled by the wait. A 128-anchor fan is half XL's width, so a worse oracle is partly pure coverage and must not be read as a decoder-quality verdict (REGISTRY §4.3 eval-plan note).

---

## 4. Q3 — Training curriculum

**Recommendation: staged, maximum checkpoint reuse (LP-FT proven P16; simultaneity proven-bad N6; v1 recipe proven P1/P13).**

| Stage | What trains | Init / reuse | Proven basis |
|---|---|---|---|
| S0 | nothing — mint v2.1 labels + fixed VTARGET caches for train+val | v1.5's pod2 caches exist (`/tmp/labels_*` logs, `label_set: v21` in the v15 run JSONs) | P10–P12 |
| S1 (=Path A LP) | head only, trunk frozen | **reuse `flagship-v15-ab/ckpt.pt`** — already trained, 0.5437 | P4, P16 |
| S2 (=v1.6, Gate A) | unfreeze trunk under the `ab` head, LP-FT | S1 ckpt + v1 trunk | P16; `fc2c484` lever |
| S3 (=Path B, only if S2 fails gates) | everything from scratch, decode-side levers on from step 0, encoder-grounding levers off (Escalation 3) | fresh; anchors + labels reused | P1 recipe, P9, N6 |
| S4 (optional polish, post-gate) | LP-FT pass of the S3 winner's head on its own frozen trunk (cheap sharpening) | S3 ckpt | P16 |

**Trunk-LR rule for S2 (ASSUMPTION, standard practice, not program-measured): trunk at 0.1× head LR with its own warmup.** The WM-integrity gate (§11) is the empirical guard; if rollout ADE degrades, first retry is predictor-unfrozen/encoder-frozen.

---

## 5. Q4 — The closed-loop planner's role at inference (P2 machinery)

The two regimes must not be conflated (C1/C2, N1):

- **Open-loop scoring / leaderboard**: v3.5 outputs the decoder's argmax-confidence trajectory. **No cost re-ranking of the fan** (N1: 0.0 % hand cost, ≤8.4 % learned; N3: no VTARGET term at 2 s).
- **Closed-loop / deployment**: **P2 CEM over v3.5's own operative WM, warm-started from the decoder's top-K proposals** instead of P2's 16-seed constant-action grid. This is the v3 design's M3/M6 pattern (V3_HIERARCHICAL_PLANNING_DESIGN §8) instantiated on v3.5. **Framing corrected (C2): what P2 proved is that planning beats the supervised HEAD (0.893 vs 3.150) and cuts closed-loop drift (1.038 vs 1.685; divergence 8.7 % vs 22.2 %) — it did NOT beat CV open-loop (0.893 vs 0.825, a loss).** So the claim v3.5 rests on is the *closed-loop drift* result, not open-loop superiority. The warm-start upgrade is where proposal quality should compound (P2 planned from constant-action seeds; v3.5 would hand it a 0.16–0.31-oracle fan). **ASSUMPTION: the compounding is unmeasured — Gate D (§11), eval-only.**
- VTARGET's only inference-time role is inside the P2/P3 cost at its proven timescale (10–20 s aspiration, N3/P12); the gap/TTC barrier stays out until lead-state labels exist at coverage (C5: 51–53 % on a 595-record pilot is not coverage).
- The P2 code is uncommitted (REGISTRY R3) — vendoring it is a precondition for Gate D and already flagged repo-wide.

---

## 6. Q5 — Losses: what survives

| Loss / aux | In v3.5? | Evidence |
|---|---|---|
| JEPA latent prediction + SIGReg (free-dims 64) + inv-dyn 0.5 | **YES** — the v1 trunk recipe | P1; REGISTRY §1 preset |
| speed-input `v0` channel (+SPEED_SCALE=10 contract) | **YES** | P13 |
| jerk 0.02 + aux-accel | **YES** | P13/P1 (v1 flags) |
| h15 imagination loss (mask 0.5, w 0.5) | **YES** — the imagination is what conditioning reads (P4) | P1/P4 |
| grounded step-readout (metric Δpose) | **YES** — the closed-loop substrate (C2), and the H18 dominance evidence | REGISTRY §6 reading 2 |
| anchor-classification CE (1.0) + L1-from-assigned-anchor (1.0) + truncated denoise (2 steps) | **YES** — the REF-C/v1.5 decoder recipe | P5/P8; `flagship_v15.py` LOSS block |
| ranking CE on `sel_score` incl. denoise pass (supervised conf at denoise timesteps) | **YES** — mandatory if ranking touches refined confidence | N2 |
| **Hierarchical goal conditioning S1/S2/S3** (strategic→planner, strategic→operative, plan→operative) | **YES — REQUIRED (§2A).** Not a loss but a wiring; trained jointly, norm-capped, goal-dropout ≥0.5, stop-grad on S3, gated by G-H | Sayed's requirement; N4 re-opened (§2A.1) |
| Strategic goal-CE (ROUTE/VTARGET band) as the strategic module's own supervision | **YES** — the strategic head must *produce* goals at inference, so it must be trained to | V3 design M2; §8 |
| VTARGET strategic prediction head (CE on minted bands, label not input) | **YES, small** — v3-design M2's key head; VTARGET as *data* (P12), zero inference leakage | V3_HIERARCHICAL_PLANNING_DESIGN §3(1); **ASSUMPTION: its training value is unmeasured — it is a labeled aux, gated at 15 k like everything else** |
| REF-C LAW aux | **NO** — LAW is REF-C's world-model *substitute*; v3.5 has the real operative WM. Redundant by construction. **ASSUMPTION (structural argument, not a measurement)** | REGISTRY §4 (LAW description) |
| encoder-ego decorr + fa-dropout + invdyn-gradscale + rollout-k 12 | **NO (default off, hardened)** — v3enc's canonical-window exponent −0.503 sits on killed-v2's −0.50; needs a passed 10 k ADE gate **and** exponent separation to re-enter | N6/N7, Escalation 3 |
| H26 seam norm cap (≤1.0×) + per-seam norm telemetry | **YES** (monitor + cap, not a loss) | P15 |

---

## 7. Q6 — Parameter budget — **cap raised to 400 M** (Sayed, 2026-07-20)

> *"If the combination of v1 and refc requires larger model, this is not a problem — my cap is at 400 M currently."*

### 7.1 Where the extra ~140 M should go — and where it must NOT

**The single most important capacity finding: decoder width is NOT the lever. Remove "bigger planner" from every branch.**

| Arm | Decoder params | Fan quality (oracle-in-fan) |
|---|---|---|
| REF-C-XL | **22,702,345** | **0.1640** |
| flagship v1.5 (`ab`) | **~29.9–31.0 M = ~1.3× REF-C's** | **0.3073 = 1.87× WORSE** |

v1.5 already runs ~**130 %** of REF-C's decoder capacity and proposes **1.87× worse**. Capacity is not the binding constraint on the fan — coupling to a trained encoder is. (Arithmetic verified from `c4f75d6`'s stated head size and REGISTRY §4.1's measured decoder params; ratios 30.45/22.70 = 1.34 and 0.3073/0.1640 = 1.87.)

**Where the evidence points instead — encoder allocation:**

| Arm | Encoder | % of total | Fan oracle | High-speed ADE@2s |
|---|---|---|---|---|
| REF-C-XL | 199,496,532 | **79.2 %** | **0.1640** | **0.3243** |
| flagship v1 | 87,121,280 | **33.1 %** | 0.3073 (v1.5 fan) | 0.5513 |

REF-C spends **79 %** of its budget on the encoder; flagship spends **33 %** — and REF-C owns the better fan and the high-speed win. ⚠️ **This is SUGGESTIVE, NOT PROVEN: the comparison is confounded** (conv vs ViT, direct-head vs world-model, different objectives). It is the best allocation evidence available, and it says: **spend the extra capacity on the encoder, not the planner.**

**Recommended allocation of the ~140 M headroom, in priority order:**
1. **Encoder capacity (+60–100 M)** — d768×12 → d1024×16 (or d768×24). Evidence: the allocation table above; the fan gap traces to encoder/coupling, not decoder width.
2. **Hierarchical seams (+1–3 M)** — essentially free (see 7.3). The hierarchy costs *optimization risk*, not parameters.
3. **Anchor/probe breadth (+0–5 M)** — more anchors widens coverage cheaply (embedding-table growth), unlike decoder depth.
4. **NOT the decoder.** Hold at REF-C-XL geometry (~22–30 M).

### 7.2 Revised budget table at 400 M

| Module | v3.5-J (baseline) | v3.5-J+ (recommended, wider encoder) | Source |
|---|---|---|---|
| ViT encoder | 87,121,280 | **~170–190 M** (d1024×16) | v1 config ✅ / **ASSUMPTION (scaling est.)** |
| Operative predictor (d768×10, action_dim 3) | 96,609,283 | 96,609,283 | v1 config ✅ |
| h15 imagination | 22,055,683 | 22,055,683 | v1 config ✅ |
| grounding heads | 13,432,338 | 13,432,338 | v1 config ✅ |
| aux-accel | 528,897 | 528,897 | v1 config ✅ |
| strategic_policy + VTARGET band head | ~8.5 M | ~8.5 M | v1 config ✅ + est. |
| Anchored-diffusion tactical decoder | ~30 M | ~30 M (**held — 7.1**) | `c4f75d6` / REGISTRY §4.1 ✅ |
| **Hierarchical seams S1+S2+S3** | **~1–3 M** | ~1–3 M | 7.3 |
| REMOVED: v1 `tactical_policy` (−22.7 M) + `tactical_pred` (−26.5 M) | — | — | replaced by the planner |
| **TOTAL** | **≈ 260 M** ✓ | **≈ 345–365 M** ✓ (headroom to 400 M) | |

### 7.3 Seam parameter cost (why the hierarchy is nearly free)

| Seam | Params | Composition |
|---|---|---|
| S1 (KV tokens) | ~0.3–0.8 M | ROUTE 5 rows + VTARGET 24 rows embeddings at d512 + token projections + ReZero scalars |
| S2 (FiLM to operative) | ~0.4–1.2 M | d_ctx → 2×d768 γ/β projection + LayerNorm |
| S3 (plan → actions) | **~0 new** | reuses the existing inverse-dynamics head and the predictor's action channel (§2A.3) |
| **Total** | **~1–3 M** | |

### 7.4 Alternative ② re-priced honestly at 400 M — it fits ONLY in its trimmed form

The earlier draft rejected "REF-C encoder as a second KV source" at ~460 M. Re-pricing against the new cap:

| Variant | Arithmetic | Verdict at 400 M |
|---|---|---|
| WM trunk (no tactical: 87.1+96.6+22.1+13.4+8.5 = **227.7 M**) + **REF-C-XL** encoder 199.5 M + decoder 30 M | **≈ 457 M** | ❌ **STILL OVER CAP** — the honest answer is that raising 300→400 M does *not* rescue the XL variant |
| Same trunk + **REF-C-BASE** encoder **90,458,632** + decoder 30 M | **≈ 348 M** | ✅ **FITS** — and this is the genuinely new option the raised cap unlocks |

**So the re-priced answer is specific: alternative ② becomes viable only with REF-C-*base*'s 90.5 M encoder, not XL's 199.5 M — and whether that encoder is good enough is exactly what tonight's REF-C-base result measures (§3.6).** That converts §3.6 from a nice-to-have into a decision input for ②. ⚠️ Still second-choice: two encoders means two forward passes per tick (§7.5), and C4 says REF-C's curve behaviour is its liability.

### 7.5 Inference efficiency — now a first-class constraint, not an afterthought

> Sayed's ask; a sibling agent is benchmarking v1 vs REF-C now. **A 400 M model that cannot run at 10 Hz is not a driving stack.**

Measured anchor: **v1 deploy tick 11.16 ms / 89.6 Hz (fp16, 1.59× vs fp32); predictor CUDA-graph 2.57×** (REGISTRY §1.2) ✅. Everything below is **ASSUMPTION / estimate pending the sibling benchmark** — stated so the design can be falsified on latency, not so it can claim a number.

| Component | Expected tick cost | Note |
|---|---|---|
| Encoder (widened to ~180 M) | ~1.6–2.2× v1's encoder share | roughly linear in params at fixed resolution |
| Operative rollout (grounded, 20 steps) | as v1 | CUDA-graph path already proven 2.57× |
| Anchored decoder (256 anchors, 2 truncated denoise steps) | small — one batched cross-attn stack ×2 | denoise steps are the whole point of *truncated* diffusion |
| ⚠️ **Imagination conditioning (the `ab` asset)** | **8 probes × 20 steps = 160 predictor evaluations per tick** | **This is the efficiency risk of the design's best-proven lever.** Probes batch (8-wide) so it is **20 sequential** batched predictor steps, not 160 — but it is still the dominant term |
| Hierarchical seams | negligible | ~1–3 M params, no extra passes |
| Second encoder (alternative ② only) | **+a full encoder pass** | a real argument against ② beyond params |

**Design constraints adopted:**
- **Open-loop deploy tick ≤ 50 ms (≥20 Hz)** — half the 10 Hz control budget, leaving headroom for perception I/O.
- **Imagination probe count is a tunable knob** (8 → 4 → 2) and must be **ablated for the accuracy/latency curve** at G-E; if the tick misses budget, probe count is the first thing cut, not the encoder.
- **Closed-loop planning (P2 CEM) is budgeted separately** — N=48×2 iterations is a planning-time cost at replan cadence, not part of the perception tick.
- **Latency is reported in every gate table from G-A onward**, alongside ADE.

### 7.6 Legacy ≤300 M framing (superseded, retained for provenance)

Measured components (run configs, REGISTRY §1.2/§4.1; v1.5 head from `c4f75d6`):

| Module | Params | Source |
|---|---|---|
| ViT encoder (9ch/256/p16, d768×12) | 87,121,280 | v1 config ✅ |
| Operative predictor (d768×10, action_dim 3) | 96,609,283 | v1 config ✅ |
| h15 imagination | 22,055,683 | v1 config ✅ |
| grounding heads (incl. step-readout) | 13,432,338 | v1 config ✅ |
| aux-accel head | 528,897 | v1 config ✅ |
| strategic_policy (d384×4) + VTARGET band head | 8,385,027 + ~0.1 M | v1 config ✅ + **ASSUMPTION (head est.)** |
| **Anchored-diffusion tactical decoder** (256 anchors, d512, v1.5-class incl. conditioning projections) | ~30–32 M | `c4f75d6` "29.9–31.0 M"; REF-C's bare decoder is 22.7 M ✅ |
| REMOVED: v1 `tactical_policy` unimodal wp_heads (−22,736,141) and `tactical_pred` (−26,535,424) | — | replaced by the anchored decoder; **tactical_pred removal is a design choice: its 2–8 s coarse-rollout role returns in v3 (M4b), not v3.5 — ASSUMPTION that dropping it is safe is gated by the WM-integrity + miss@2m gates** |
| **v3.5-J total** | **≈ 258–260 M** | comfortably ≤ 300 M ✓ |
| v3.5-A total (v1 263.4 M + 32 M head, tactical_policy retained frozen) | ≈ 295 M | ≤ 300 M ✓ (tight); drop the dead `tactical_policy` at export → ~273 M |

Rejected on budget: dual-encoder variants (+90–200 M → 350–460 M ✗, §3.5).

---

## 8. Q7 — Data & labels from step 0

- **Corpus**: the strict-parity 2,376-ep PhysicalAI set (`physicalai-train-e438721ae894`, skip `f09e44db`) — every cross-arm comparison depends on it (REGISTRY §0.1, D-A2).
- **v2.1 route labels from step 0** (P10/P11) — first *trained-from-scratch* deployment; XL/base's scale-vs-label confound (REGISTRY §4.3 ⚠️) does not apply because v3.5-J is single-arm with fixed labels.
- **Fixed VTARGET mint from step 0** (P12): 5 s floor, valid mask, DROPPED row; enters as (a) strategic head *target* and (b) P2 cost input at 10–20 s. Never a decoder token (N4), never a 2 s ranking term (N3).
- **VLM 595 records (C5): eval-only in v3.5.** Grounds: coverage is 595 records vs 406,099 train windows (~0.15 %); direction is chance-level; LONMODE is not appearance-invariant (62 % render-stability). Their proven uses: (i) eval stratification/scenario panels, (ii) label-QA masking, (iii) the WHY sidecars (signs/lead/scene/RISK) for failure forensics.
  > ⚠️ **CORRECTED 2026-07-21 — "IS-A-TURN 89.3 %" is retracted, twice over.** (a) It does **not
  > reproduce**: three independent measurements give **76.8 / 80.6 / 78.6 %**. (b) More importantly it
  > was *agreement*, not *recall* — agreement counts straight-straight matches, and on a **~74 %
  > straight** corpus a model answering "straight" every time scores ~74 % while detecting **zero**
  > turns. It was never evidence of "a good event detector". Quotable: **turn recall ≈ 78 %**
  > (**77–81 %** across the three: eval pod 200 win · pod3 banked 400 win · enum-order probe 200
  > held-out win — `2026-07-21-cosmos-reason2-production-semantic-labeling.md` §1),
  > direction = chance (**57.1 %**, CI [0.400, 0.745]).
  > **And the direction failure is now attributed:** the enum-order probe (2026-07-21) shows the left
  > bias is **the model's, not our prompt's** — listing `right` first moved left-share only
  > 74.5 → 66.7 % on a 48.2 %-left corpus, and **recall on true right turns was bit-identical
  > (0.2069) in both arms**. Swapping the enum did not buy one extra right turn. ROUTE stays
  > kinematic on model evidence, not suspicion. Training use requires the Cosmos-Reason1-7B bulk pass over the full corpus — that is TanitDataSet scope, not v3.5. **Provenance split is binding: kinematics own VTARGET/LONMODE/LATMANEUVER/DYN/HEADWAY; the VLM owns the WHY** (`547c8ec`).

---

## 9. Q8 — Naming & registry hygiene

Three "v3"-family objects now exist with different meanings; the registry must keep them orthogonal:

| Name | Axis | Status rule |
|---|---|---|
| **v3enc** (`flagship4b-v3enc-30k`) | encoder-grounding levers, staged (OOD axis) | stays a §1.x flagship row; its gate result is an *input* to v3.5-J's lever set, nothing more |
| **v3.5** (this doc) | consolidation of proven assets (performance axis) | new registry section **"1.6 flagship-v3.5"** with an explicit `lineage:` field: `v1-trunk × REF-C-decoder × v1.5-ab-conditioning`. Run names: `flagship-v35-a-unfreeze` (=v1.6; register the eval key as **both** `flagship-v16` and `v35-a` from day one so no later rename rewrites history), `flagship-v35-joint` |
| **v3** (V3_HIERARCHICAL_PLANNING_DESIGN, frozen vocab) | hierarchical planning-over-WM (planning axis; D-033) | unchanged; v3.5 feeds it a better substrate (better fan → better M3 warm starts) and steals nothing from its scope (no goal-conditioned M4b, no Rulebooks lattice in v3.5) |
| v1.5 / v1.6 | LP-FT ladder on the frozen v1 trunk | v1.6 is retro-labeled v3.5-A **only if** it passes G-A; a failed v1.6 stays v1.6 (a diagnostic, like v1.5) |

Registry contract additions: the v3.5 row must carry the **v1.6-outcome decision** (which path fired and why, with the gate numbers) so the lineage stays reconstructible; and per the maintenance contract, strata lines must name their step (the §4.1 stale-strata defect, Escalation 4, is exactly this rule violated).

---

## 10. Q9 — Risk register (top 5)

| # | Failure mode | Early-warning metric (when) | Fallback |
|---|---|---|---|
| R1 | **Unfreezing/joint training breaks the WM** — decoder gradients reshape the trunk, rollout + imagination degrade, closed-loop substrate lost | **WM-integrity gate at every milestone**: operative-rollout ADE@2s ≤ 0.452 + CI on the canonical 881; h15 imagination panel non-degraded | S2: freeze encoder, unfreeze predictor only, trunk-LR 0.1×; S3: raise grounded-rollout loss weight; ultimate: Path A ships, Path B abandoned |
| R2 | **Simultaneity trap reappears in v3.5-J** (the v2 death, N6) | `g_op_fwd_ade_m` power-law exponent vs v1's −0.84, read at 2 k/5 k. **Protocol pinned after this session's finding (N7): fit on the canonical 1.5k-onward window ONLY, report R², and report the ≥2 k sensitivity fit alongside — a single window is not decisive** (v3enc reads −0.503 or −0.774 depending on start point). Kill threshold: canonical fit ≤ −0.60 **and** the level at 5 k not above v1's same-step value | drop to the S1/S2 lineage (Path A result stands); re-introduce decoder at 10 k on a v1-recipe warm start (LP-FT it, P16) |
| R3 | **High-speed win doesn't transfer** — REF-C's 0.324 may live in its conv encoder / direct-head training, not the decoder v3.5 borrows | high-speed stratum ADE@2s at 5 k/15 k/20 k/30 k vs three pins: v1 0.5513 / REF-C 0.3243 / CV 0.6468 | longitudinal up-weight + speed-stratified sampling (the v2.1-lever plan: measure-first at milestones); accept partial (≤0.45) if mean+tail gates pass |
| R4 | **Conditioning-seam swamping** (H26 fired at 2.80× on ROUTE; now THREE new seams S1/S2/S3 carry the required hierarchy) | per-seam contribution-norm ratio every step — **hard in-graph clamp at 1.0×, fail-loud at 1.5×** (§2A.3), plus G-H causality at every milestone | ReZero re-init / norm-matched projection (the literal H26 fix); if a capped seam stops helping, that is a *finding* about the mechanism — report it, do not uncap it |
| R6 | **The hierarchy is decorative** — seams pass norm parity but carry no per-window content (v1's measured failure mode: seams cohered but did not drive; `per_window_content_helps` ≈ 0 on both arms) | **G-H per-window-content test** at 5 k (not 30 k) — the earliest honest read | Change mechanism, not scale: S1 tokens → FiLM, or make the operative attend to the plan explicitly. If content stays inert at 15 k, report H26 as **falsified in this architecture** rather than shipping decorative wiring |
| R7 | **Latency blowout** — a 345–365 M model with 20 sequential imagination-probe rollouts misses the control budget | deploy-tick reported in **every** gate from G-A onward; target ≤50 ms (§7.5) | cut probe count 8→4→2 (ablate the accuracy/latency curve) **before** touching encoder width; fp16 + CUDA-graph paths already proven (1.59× / 2.57×) |
| R5 | **Tail import** — REF-C's heavy tail (miss 0.146, C4) arrives with its decoder and costs the program its tail king | **miss@2m ≤ 0.10 as a HARD gate at every milestone** (G3); straight-stratum median AND mean tracked (C4's median/tail split) | deploy contract: the shipped output stays v1-style grounded rollout unless the decoder path beats it on **both** mean and miss; decoder then serves proposals-only (still feeds Gate D closed-loop) |

(Ops risk is standing, not top-5: pod2 overlay/quota deaths — mitigated by P14 milestone+atomic ckpt + guard, `supervise_run.sh` auto-resume.)

---

## 11. Q10 — Staged experiment plan (cheapest first, gate-by-gate)

> Approval requested per line. GPU-day figures: measured paces where they exist (v1 30 k ≈ 53 h A40 = REGISTRY §1.2 `wallclock_s`; v1.5 8 k head-only ≈ 3–5 h/arm on pod2 = daily 2026-07-20 §2); joint-run figures assume v1's pace + ~15 % decoder overhead — **ASSUMPTION**.

| Gate | Experiment | Reuses (no rebuild) | Cost | Pass ⇒ | Fail ⇒ |
|---|---|---|---|---|---|
| **G-A** 🟢 **RUNNING** (`flagship-v16-ab-ft`, pod2, launched 20:01 UTC, 6 k steps) | **v1.6**: unfreeze 4 encoder blocks + predictor under the `ab` head, head-LR 1e-4 / trunk-LR 1e-5 | `flagship-v15-ab/ckpt.pt` (warm head) + v1 trunk + v2.1 labels + `anchors256.pt` + `probes8.pt` | **~0.5 A40-day** (in flight) | oracle ≤0.22 AND G1 <0.458 AND G2 <0.4522 AND G3 miss ≤0.10 AND WM-integrity canary flat ⇒ **Branch 1** (§2.2); go G-D/G-E | **§2.3 decision table.** ⚠️ **Step-500 val already shows oracle 1.271 (4× worse than the 0.3073 it must beat)** — decision point at 1.5–3 k |
| **G-H** (NEW — first-class, per Sayed's wiring requirement) | **Hierarchy causality gate** (§2A.5): sensitivity + direction-correctness + `per_window_content_helps` + effect-size floor + norm-parity ≤1.0 on every seam | TanitEval hierarchy panel (exists, auto-reruns) | ~0.1 eval-day per milestone | the wiring may be **claimed live**; H26 core-goal proof advances | seam is decorative ⇒ fix scaling/mechanism before any hierarchy claim; **no silent carry-forward** |
| **G-B0** (free, ~1 day) | REF-C-base 30 k completes + canonical eval ⇒ fixes G-C's decoder geometry (§3.6) | running run | 0 (scheduled eval) | adopt base geometry if base ≈ XL (frees ~15–20 M) | keep XL geometry (the assumed default) |
| **G-B1** (parallel, free) | v3enc 10 k gate (pre-registered, REGISTRY §1.4) — **now a two-part gate** | running run | 0 (scheduled eval) | ADE gate passed **AND** canonical-window exponent separates from −0.50 ⇒ staged encoder levers may enter G-C | either part fails ⇒ **G-C runs the pure v1 recipe** (current default, N7) |
| **G-C** (only if §2.3 selects alternative ①) | **`flagship-v35-joint` 30 k** on the parity corpus: v1 recipe + anchored tactical decoder + **the full hierarchical wiring S1/S2/S3 (§2A), jointly trained** + widened encoder (§7.2) + v2.1/VTARGET data + **hard** norm caps; milestones 5 k/15 k/20 k/30 k, full panel each (mean, miss, strata, oracle-in-fan, frac_2x, **plan-free** WM-integrity canary, seam norms, **G-H causality**, **deploy-tick latency**, R2 exponent at 2 k/5 k) | anchors, labels, caches, TanitEval, milestone/atomic ops | **~4–5 A40-days** (ASSUMPTION; +overhead for the wider encoder) | ADE ≤0.43 AND miss ≤0.10 AND high-speed ≤0.45 AND oracle ≤0.20 AND **G-H passes** AND **tick ≤50 ms** @30 k | R2 kill at 2–5 k ⇒ Path A stands as v3.5; partial pass ⇒ S4 LP-FT polish then re-gate |
| **G-D** (eval-only) | **Closed-loop compounding**: P2 CEM over the winning v3.5 WM, warm-started from its decoder top-K (§5) | P2 (`planner_p2.py`, needs vendoring — R3), winner ckpt | ~0.2 eval-pod-day | closed ADE <1.038 AND divergence <8.7 % ⇒ v3.5 becomes the deployed closed-loop substrate and v3's P3/P4 base | if no gain over constant-seed P2: fan-quality claim is open-loop-only; v3 proceeds on v1 unchanged |
| **G-E** (final) | Full panel on the winner: strata, OOD triple (comma/cosmos), genuine-prediction causal panel, hierarchy panel re-run, HF push + registry row | TanitEval panels (all exist) | ~0.3 eval-pod-day | registry §1.6 row + paper §7 update | — |

**Decision economics:** G-A risks half a GPU-day to possibly save four; that ordering is the same de-risk pattern P2 proved for v3 (zero-cost gate before training commitment, D-033).

---

## 12. Follow-ups this doc does NOT execute (flagged, not silently done)

1. **Paper**: not updated — nothing in TANITAD_PAPER.md §7/§9 is invalidated by a design doc, and inserting v3.5 before Sayed approves the gates would put speculation in the paper. Follow-up: add v3.5 to §9/roadmap after G-A has a verdict.
2. **MODEL_REGISTRY §4.1 stale-strata fix** (Escalation 4): replace the 16 k strata line on the FINAL row with the `refc-xl-30k.json` values (high 0.3243 / low 0.5912 / med 0.4989) and step-label every strata quote.
3. **HYPOTHESIS_LEDGER**: the v1.5 double-dissociation + `ab` attribution entries are still commit-body-only (daily 2026-07-20 §4 names this gap); the ledger write should land with or before G-A so the next agent reads it.
4. **Vendor P2** (`planner_p2.py`) before G-D (registry R3, one-pod-loss-from-gone).
5. **595 VLM records live pod-only** (`tanitad-eval:/root/vlm_pilot/bulk/out/`) — same reconstruction-risk class; pull into the lake or HF before the eval pod is recycled.
6. **The v3enc exponent should be re-fit with a pinned protocol and written into the v3enc row**, not left as a passed-around scalar: this session found the same log yields −0.379 / −0.503 / −0.774 / −0.82 depending on the fit window (R² 0.22–0.57). Whoever owns the 10 k gate should publish window + R² + n with the number, or the gate will be argued rather than decided.

---

*Verification method note: pods were read-only this session (ssh probes to tanitad-pod, tanitad-pod2, tanitad-eval; `ls`/`ps`/`nvidia-smi`/JSON reads only). No training was launched, no checkpoint touched, nothing committed or pushed — this file is staged only.*
