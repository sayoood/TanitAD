# TanitAD program report — 2026-08-28 morning (D-025)

*Master Mind · times Europe/Berlin (pods/Thor log UTC) · branch `agent/arch-inf-20260803` ·
every number carries arm + tier + evidence class; sources are the registry, raw JSON, or the
named commit.*

---

## 0. Headlines

1. **The O14 absorption read — the decision on R2's place in v7r — is IN:**
   **ABSORBED** — the incumbent's pixel-marginal (+0.0096, t +5.11) is GONE on the O14
   arm (−0.0047, t −3.09), gates holding. **R2 is adopted into the v7r recipe.**
   ⚠️ Drift itself is UNCHANGED — O14 is the anti-displacement lever, not anti-drift.
2. **The P0 EMA bake-off ran itself overnight**: the chain launched both `ema2k` arms after
   `o14fut30k` vacated and both completed (08:03 / 08:35). Config-diff verification: **both
   arms CLEAN** — exactly the prereg's admissible set. Reads in §3.
3. ⭐⭐ **PI directive registered (D-LABEL-GT, `063972648`)**: *"the only ground truths of the
   labels is what we created together here"* — the v7 label set (4,719 clips / 26.2 h) is THE
   canonical ground truth, blob-pinned with its three producers and four traveling caveats;
   all v6.1 label sets and both non-v7 g_tac producers are superseded as label sources.
4. **Both stranded worktrees drained** (tharp 32 files `937afe3da`, kepler 31 of 32
   `9e61f1a7c`) — zero worktrees hold unintegrated work for the first time since 08-22.
5. **The vocab mandate's full blast radius found and repaired**: 46 tests across 16 suites
   (not MM-C1's 9/2) — full suite **4812 passed / 0 failed** restored.

## 1. Fleet (MEASURED this morning)

| resource | state |
|---|---|
| Thor | **idle** since 08:35 — `o14fut30k` done (30k steps, 0.979 s/step, `done: true`, 19.89M params); P0 chain complete; EMA trainer installed (backup kept) |
| dev box 4060 | absorption sequence DONE (§2); now running the P0 probes (sequential, `OMP_NUM_THREADS=6`) |
| Research Lab agent | **running** (daily pass, respawned 08:40 after the first spawn died with the CLI restart; four domains, literature-first, GPU embargo till 12:00) |
| FlyWheel sessions | DataFlyWheel active overnight (coordinated, 5 exchanges); EvalFlyWheel quiet since v2.3.0 |

## 2. The O14-fut absorption read (E-DEC-67, T0-DIAGNOSTIC, MEASURED)

**Design recap** (PREREG_O14_FUTURE_OBS.md + Amendment B): the incumbent trained-encoder arm
carries a pixel-borne marginal of **+0.0096 (t 5.11)** the token field actively displaces
(E-DEC-63). R2 adds a small head predicting 32×80 grey pixels at t+4 (`--w-o14 1.0
--o14-mode fut`), one variable on the verbatim `postrain30k` line. The tiny ladder passed all
gates and showed the marginal is a 30k-scale phenomenon (baseline-null at 2k), so the
absorption PRIMARY moved to this 30k arm. The o14 loss plateaued by ~10k — the
"aux not converged" escape hatch is closed in advance.

**RESULT — inserted after probes:**

| read | o14fut30k | incumbent (rdw8p30k / postrain30k) | verdict input |
|---|---|---|---|
| pixel-marginal (slim) | **−0.0047, t −3.09** | +0.0096, t 5.11 | **the positive marginal is GONE** (sign-flipped; residual = estimation cost of 2,560 now-uninformative dims — z_t+pixels 0.6685 < z_t 0.6732) |
| latentmotion drift | 0.6709 (t 142.96) | 0.669 | HOLDS (+0.3 % rel, inside band) |
| meanpred nrmse | 0.8288 | 0.8115 | HOLDS (+2.1 % < 10 %); cos_ctr 0.6043 vs 0.6395 (−5.5 %, stated) |

**Verdict: ABSORBED** (rig valid: constant 0 exactly, deliberate-regression failed as required,
shuffles ~0, A1 in band; ckpt md5 `3e4a7443` verified both sides). Consequence executed same
turn: **R2 adopted into `V7_RECIPE_AND_SCALEUP.md` §5.1** (`--w-o14 1.0 --o14-mode fut
--o14-k 4`), E-DEC-67 + registry updated, raws banked in the ladder package. **The two facts
that travel:** absorption did NOT reduce drift (0.6709 ≈ 0.669) — E-DEC-66's “O14 is the
only live anti-drift lever” is corrected: **no measured lever moves drift on the trainable
line**; the drift attractor is now cleanly separated from displacement and stays the open
front (P0/EMA next). And prediction mildly paid (nrmse +2.1 %, cos −5.5 %).

## 3. The P0 EMA-teacher bake-off (MM-E1, T0, first reads)

Chain-run overnight: `ema2k_s0` + `ema2k_s1` (the o14base2k line verbatim + `--o5-target ema`
[+ seed 1]), incumbent = `o14base2k` reused (bit-comparable by the pinned defaults).
**Config-diff: CLEAN on both arms** (only `o5_target` ABSENT→ema, `tac_vocab_version`
ABSENT→v6.0 ≡ identity by the loader property, `seed`, `out`). 2k scale-facts bind: reads are
vs-base RELATIVE, cos the most sensitive axis. Probe reads follow the absorption sequence on
the 4060 (never concurrent) — **§3 numbers land in the midday report**; the committed
outcomes (EMA-BETTER → 30k pair; EMA-NEUTRAL → drop as redundant complexity; EMA-WORSE →
drop; MIXED → numbers, no verdict) are in PREREG_P0_EMA_BAKEOFF.md.

## 4. The night's other results (all committed)

| item | commit | substance |
|---|---|---|
| D-LABEL-GT | `063972648` | the PI's ground-truth directive, blob-pinned + 4 caveats (87.2 % CoT untimed — **supervision by untimed tokens is an OPEN PI decision**; turn det. P 70.6 %/R 61.3 %; 9/52 tokens NOT_YET_EXTRACTABLE must be masked; grounding ~13 %, confirms-never-refutes) |
| tharp integration | `937afe3da` | LAN-preflight arm (+ refc_v3_train merge), sam3 audit, alpamayo-mapping, s2-abstain-v3, 4 product SPECs, 2 suites — Hub→Lab re-pathed |
| vocab blast radius | `937afe3da` | 37 more broken tests found by the full-suite gate; frame-pinned v6.0; the mandate's measured cost: +5,130 params (refc) / +5,775 (production), head-only; RETRACTION_LOG MM-C1 addendum: *the regression set for a default change is the FULL suite* |
| kepler integration | `9e61f1a7c` | the g_tac geometry floor (g_tac_lon **92.28 % coverage on parity**, n=69,447; LAT abstains) + the ungated-Alpamayo library path in `tac_str_labels` (removes the 25.6-T4-day VLM gate); 1 file dropped (v6.1 builder, superseded) |
| D-DATA-GTAC-b | `fe3a3032c`+`a6b5ec0ee` | two blind g_tac producers at HEAD; **CORRIDOR_OFFSET contested 2-vs-1** (built-and-measured-inadmissible × v7-emitter-declines vs the v6.1 module's raw ≥1.0 m gate — the refuted quantity, uncalibrated); ⛔ no supervision from those labels; blast radius import-verified: v7 set carries ZERO |
| MM-E2 prereg | `4005ed829` | the three-armed cross-agreement experiment, kappa-only + chance baseline (C136 guard), REFEREE-SPLIT outcome = shared bias; CORRIDOR_OFFSET not decidable without a human-audit reference (N=50) |
| P0 prereg | `dc261b461` | written BEFORE launch; o14base2k as matched incumbent |
| DataFlyWheel 25 | `2c4b8d2bb` | the v7 label set integrated (now D-LABEL-GT's canonical set) + vocab freeze corrected to 7 strategic actions (PI verbatim) |

New traps pinned this night: **CRLF false-divergence** (blob-vs-worktree comparisons must
normalize or every file reads changed — tharp's find); **name-collision grep lies** (the v7
emitter's local `tactical_goals()`); narrative-clock drift ×2 (settled by Thor `date`).

## 5. Ladder position (L0–L5 of V7_RECIPE_AND_SCALEUP §8)

- **L0 rig validity** ✅ (controls + DR arms clean throughout)
- **L1 collapse** ✅ solved at parity (participation 25.58), do-not-relitigate
- **L2 representation** 🔶 content proven (`splitp30k` n_agents +0.3881 > frozen DINOv3) but
  target-specific; **today's absorption verdict is the L2→L3 lever test**
- **L3 content+prediction in ONE arm** ⛔ the open blocker (the dissociation)
- **L4 T1 driving** ⛔ not attempted for v7 (first parity T1: both arms at the drift
  predictor's face, ADE ~14.5 m vs CV floor 0.5352 — the honest baseline)
- **L5 hierarchy dominance** — gated on L4
- Supporting decisions closed this week: D1 encoder TRAINABLE, D2 omega ADOPTED, teacher =
  DINOv3 (9/9), o5_k INERT on the trainable line (interaction), vocab v7 wired end-to-end.

## 6. Ordered next steps

1. **Absorption verdict → prereg OUTCOME + E-DEC-67 row + registry + v7r §R2** (this morning,
   same turn as the probes).
2. **P0 reads on the 4060** (sequential after absorption): drift/nrmse/participation vs base →
   prereg outcome (midday report).
3. **MM-E2 hand-off to the DataFlyWheel** (0-GPU; design pinned) + the v7-emitter local
   function rename.
4. Research Lab pass lands (~afternoon): four domain packages, integration triage.
5. NavSim: EvalFlyWheel's criteria v2.3.0 gates are in; **provisioning decision below**.
6. B1 production: awaiting the PI's signal (quantisation validation is its gate; today's
   Deploy&Opt research pass feeds it).

## 7. Decisions for Sayed (each with a default)

| # | decision | default (acts unless overridden) |
|---|---|---|
| 1 | **NavSim data**: which split to provision. `navhard_two_stage` = **31 GB** (v2 EPDMS, the current-gen headline); `navtest` = 223 GB (v1 PDMS); `navtrain` = 445 GB (training only). | **Provision `navhard_two_stage` only** (31 GB, fits dev-box disk, the v2 lane where Drive-JEPA's 89.0 front-only @307M sits); defer navtest/navtrain until an entry is actually planned. No download starts without your go (spend/disk rule). |
| 2 | **Untimed CoT tokens** (87.2 % of CoT-derived labels): may they supervise time-banded heads? | **No default** — you flagged it as your decision (D-LABEL-GT). Options: (a) exclude from supervision, train on geometry+timed only; (b) supervise with a discount factor; (c) supervise the STRATEGIC band only (untimed ≈ episode-level). The Lab's Data-Eng pass today surveys what the literature does. |
| 3 | **B1 production start** | Waits for your signal, per your instruction. Quantisation-validation prep is in today's research pass. |
| 4 | **CORRIDOR_OFFSET resolution path**: fund the N=50 human-audit reference (your ~30 min, one-time) or leave the token dead? | **Leave dead for now** (v7 set carries zero; nothing blocks on it); the audit becomes worth it only if MM-E2's other axes show the derivers are otherwise trustworthy. |
| 5 | **"Benchmarks & Eval**(s)**"** naming (singular vs plural directories) — still split across docs. | Default: **plural** ("Benchmarks & Evals"), matching your rename directive; the remaining singular citations get a lint-driven sweep. |

## 8. Incidents, honestly

- The Claude Code process exited ~07:50 and took the first Research Lab spawn and two
  watch tasks with it. The P0 chain (detached on Thor) was unaffected and completed both
  arms; the Lab agent was respawned 08:40 with no work lost (it had produced nothing yet).
- My tick-report headers ran up to ~75 min ahead of wall-clock twice (compaction artifact);
  arithmetic against Thor's clock caught it both times. Mitigation: timestamps in reports now
  come only from a measured `date`.
- `test_v6_*` suite damage (46 tests) was MY unfinished blast-radius from the vocab default
  flip — found by the full-suite gate, fixed and pinned the same night (MM-C1 addendum).
