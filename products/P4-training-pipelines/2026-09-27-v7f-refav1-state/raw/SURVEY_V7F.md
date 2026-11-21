<!-- Verbatim final report of a read-only survey sub-agent, extracted programmatically from its transcript on 2026-09-27; snapshot b3f7ea6f. Numbers are as the survey reported them; the coordinator's spot-checks are listed in ../STATE_AND_GAPS.md. -->

Both coordinator items are settled from source, and every other thread is closed or explicitly marked unverified. Delivering the report.

# v7F readiness survey (snapshot `b3f7ea6f`)

## Executive summary

1. **v7F has never been trained.** There is no run, checkpoint or registry row: `Project Steering/MODEL_REGISTRY.md` has 0 hits for "v7f" against 238 for the control "flagship" [MEASURED by me, 2026-09-27]; see also `Project Steering/GOALS_AND_CLAIMS.md:7243`. It exists only as a pre-registration (`Project Steering/PREREG_V7F.md`) for `stack/scripts/train_v6_staged.py` at stage S-W.
2. **Two binding gates block any launch.**
   - The PI hold of 2026-08-31 (`Project Steering/V7_LAUNCH_GATE.md:3-4`) has never been lifted, and no PI-queue item asks for it (0 hits; control "refcv6" 58).
   - The 2026-09-26 launch-gate rule (`CLAUDE.md:1286-1310`) requires a PASS token, but no `launch_gate*.py` exists at `b3f7ea6f`. The G-DVB check (argv lever vs built model) is imported only by `refc_v3_train.py:128`.
3. **The §9 launch line fails four ways.** I ran it through the real argparse and preflight on CPU [MEASURED by me]:
   - the LDAD flags are unknown (the loss exists nowhere in `stack/`);
   - the `--horizons` default is refused;
   - `--w-s2-goal 1.0` is refused at stage S-W;
   - `--obs-monitor-every` is missing, so the anchor check refuses.

   The 09-05 dry run reported only two of these. Three flag fixes plus dropping LDAD give a clean preflight.
4. **The pre-registered run cannot answer its own success criterion.** S-W trains only the world-model groups, and with O1 at 0 the metric readout stays at random init. Yet success requires a four-family T1 read (the closed-loop tier). The metric-decode refusal guard that the register calls "shipped" does not exist at the ref.
5. **Your item 1 is confirmed.** The DINOv3-seeded trunk receives [0,1] frames with no ImageNet mean/std anywhere in the v7F path, and its position table is left at random init.
6. **Your item 2:** all seven flags of the 14.3 % replicate panel exist only in `refc_v3_train.py`, so that floor is REF-C's, not v7's. The v7 line has two T0 seed pairs and no training-seed floor on any statistic v7F's gates read.
7. **All v7 evidence comes from v7-tiny** (~19 M parameters, planner objectives at 0) and is T0 (world-model diagnostic):
   - non-collapse holds relative to a reference arm;
   - L3 fails on all four arms;
   - the predictor ignores its actions;
   - the only T1 read is a false result (decoded through a random-init readout), which the registry still prints.
8. **Positives to build on:**
   - v7F measures 245.6 M parameters, under 300 M, so the prereg's 336.5 M escalation is stale;
   - a frozen trunk's latent displacement decodes longitudinal action at R² 0.33 against a near-zero pixel floor;
   - 245 v7F-pinning tests pass today.
9. **Missing before a launch can be justified, in rank order:** the PI hold and slot; a launch gate for `train_v6_staged.py`; build or strike LDAD; make the driving gate readable; register `H-V7F-1`; repair the action-sensitivity instrument; the R0/S0/G-RANK measurements; a tactical loss and max speed in the v7 trainer.
10. **Cheapest route to a positive pre-registered result:** the 0-GPU seed checks S0 and R0 on the B/16 seed, then a predictor-side LDAD pair plus a replicate on v7-tiny, then the first admissible v7 T1 read on an O1-trained checkpoint.

**Escalations (rule 3). These need the Master Mind or the PI:**
- (a) The 08-31 hold needs a PI decision, and no queue item exists for it.
- (b) A launch gate is needed for `train_v6_staged.py`.
- (c) `MODEL_REGISTRY.md:4724-4742` needs correcting (the void T1 table).
- (d) The register row `D-METRIC-DECODE-REFUSAL` (`GOALS_AND_CLAIMS.md:8946`) says a guard shipped that does not exist.
- (e) `CLAUDE.md:180-188` attributes the replicate floor to the wrong rig.

Evidence tags used below:
- **[M-me]** measured by me on 2026-09-27 at `b3f7ea6f`, by source read or CPU execution.
- **[M-reg]** measured, quoted from `MODEL_REGISTRY.md`.
- **[M-raw]** measured, read from raw JSON at the ref.
- **[M-inh]** another agent's measurement, quoted from the register or a RESULT file; inherited, not re-verified by me.
- **[EST]** estimated · **[PUB]** published · **[HYP]** hypothesis.

Tiers are T0 (world-model diagnostic), T1 (closed-loop on the model's own actions) or "none" (instrument or probe).

---

## 0. Method, and the snapshot is partial

- The snapshot holds 3,796 of the ref's 17,531 files [M-me].
  - Complete: `stack/`, `Project Steering/`, `products/P4`, `taniteval/tools`.
  - About 20 % coverage of `Research Lab/Architecture & Inference`. Absent entirely: the EvalFlyWheel, `taniteval/results`, `taniteval/tests` and the Library.
- For missing paths I read the same commit with `git show b3f7ea6f:<path>`. These are cited below as `ref:<path>`.
- I executed two things, both CPU-only, with the GPU hidden, bytecode writing off, and nothing written into the snapshot:
  - **Launch-line probe.** The §9 line through `build_parser()` + `preflight()` + `assert_trunk_anchor_preflight()`, after asserting that `tanitad` imported from the snapshot and not from the venv's editable install, which points at G:.
  - **12 targeted test files** against the snapshot.

---

## 1. Your two items

### 1a. Trunk input normalisation: CONFIRMED [M-me]

| link in the v7F path | source |
|---|---|
| uint8 frames become float [0,1] and nothing else | `stack/tanitad/data/_contract.py:51-56` (`x.float().div(255.0)`) |
| the window dataset applies it to `frames` and `future_frames` | `_contract.py:131-133`; the v2 loader defers to it (`stack/tanitad/data/v2_dataset.py:162`) |
| the stack passes the raw window to the encoder | `stack/tanitad/models/v6.py:5531-5533` |
| the encoder applies no input affine: patch conv, then the learned position table | `stack/tanitad/models/encoder.py:152-163` |
| the converter copies the DINOv3 patch conv as `identity`, with no ImageNet fold | `stack/scripts/dinov3_seed_checkpoint.py:344-349` |
| normalisation is absent from the declared losses | `dinov3_seed_checkpoint.py:568-578` (only POSITIONAL and TOKENS are declared) |
| the B/16 seed stamp has no normalisation field | `ref:…/2026-09-18-dinov3-b16-seed/raw/dinov3_vitb16_seed.json`: `left_at_init_keys ['pos']`, `layer_scale_folded true`, no `imagenet_affine_folded` |
| ImageNet mean/std appear only in the O7 teacher | `stack/scripts/train_v6_staged.py:1007-1010`; v7F sets `--w-o7-distill 0` (`Project Steering/PREREG_V7F.md:350`) |

- A case-insensitive search for `imagenet`, `0.485` and `0.229` across `encoder.py`, `v6.py`, `v2_dataset.py`, `_contract.py` and the converter returns zero input-side hits. The remaining "normalis*" hits are feature L2 norms, RMSNorm and loss weights.
- **Nothing else in the v7F launch path normalises.**
- Measured consequence, at ViT-L/16 because B/16 was absent then (E-SEED-2, [M-inh], tier none), from `TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-v7-seed-and-external-target/RESULT.md`:
  - The seed as wired is not separable from random init on speed: +0.1671 [−0.1742, +0.6124] (`:65`).
  - Normalisation alone is also not separated: +0.0992 [−0.1100, +0.2910] (`:67`, `:21`).
  - On scene content the seed loses a separated share of DINOv3: `n_agents` +0.2050 [+0.0896, +0.3233] (`:109`).
- So the missing normalisation is real, but it is not measured to be the dominant loss. Fixing it is free and is pre-registered, but not yet run (`Project Steering/PREREG_V7_SEED_POS.md:83-92`, S0 at `:124`).

### 1b. The replicate floor

**Which trainer owns the panel's flags [M-me].** Every panel flag (`--arm`, `--size`, `--withheld-bank`, `--ego-dropout`, `--ego-state-inject`, `--echo-base`, `--data-root`) appears once in `stack/scripts/refc_v3_train.py` and zero times in `train_v6_staged.py`. Conversely, `--o5-k` and `--stage` appear only in `train_v6_staged.py`.
- The panel spec uses those flags (`ref:…/2026-09-05-withheld-bank-panel/SPEC.md:14,20,55`).
- ⇒ The 14.3 % floor (6 of 42 cells, `ref:…/RESULT.md:26,37`) belongs to the REF-C tiny rig. The 09-15 status report says the same (`Project Steering/Reports/2026-09-15-1445-refcv6-refav1-v7-status.md:211`).
- `CLAUDE.md:180-188` and the panel's own title (`ref:…/RESULT.md:1,12`) still call it "the v7-tiny rig".

**What seed spread the v7 line actually has (all T0):**

| pair | result | source | caveat |
|---|---|---|---|
| `postrain30k` vs `postrain30k_seed1` (differ only in `--seed`) | drift 0.6674 vs 0.6741, "~1.0 %" [M-reg] | `MODEL_REGISTRY.md:4758-4759,4765-4766,4777` | the drift metric was later ruled arithmetic (`GOALS_AND_CLAIMS.md:7749`) |
| `ema2k_s0` vs `ema2k_s1` (2k steps) | drift 0.4002/0.3644 · cos 0.1336/0.0979 · nrmse 0.9912/0.9954 [M-reg] | `MODEL_REGISTRY.md:4744` | 2k steps only |
| "3.5 % 2-seed band" | nrmse, pair not named | `MODEL_REGISTRY.md:4711` | source pair UNVERIFIED |
| `enc3f_2k` vs `_s1` action ratio | 0.000119 / 0.000035 [M-inh] | `…/2026-09-05-v7f-six-problems/STATUS_BOARD.md:125` | Thor-only artifact |

**Answer:** the v7 line has **no training-seed floor** for anything v7F's gates read:
- the anchored `rel_pinned` statistic (G-ACT);
- the Observer-Effect ratio;
- L3;
- the T1 families.

It has none at v7F geometry either. The v7 world model's forward pass is deterministic (identity control exactly 0.0, `STATUS_BOARD.md:125`), so T0 probes need no inference-seed floor. Whether v7F's T1 planner samples is UNVERIFIED.

---

## 2. What v7F is

No registry row exists, so none of the following is registry-quotable (`CLAUDE.md:139-141`). The sources are the prereg plus measured research packages.

| attribute | value | source |
|---|---|---|
| hypothesis | `H-V7F-1`: a DINOv3-initialised trainable trunk, a distillation anchor and an encoder-side LDAD term on `Δz → (a, κ)` make the predictor action-sensitive at a committed bar without destroying trunk decodability | `PREREG_V7F.md:37-47` |
| trainer, stage | `stack/scripts/train_v6_staged.py` (11,057 lines), `--stage S-W`, which trains `encoder, readout, predictor_op, aux` | `PREREG_V7F.md:313-314`; `stack/tanitad/models/v6.py:4567-4573` |
| trunk | `ViTEncoder` 768×12×12, patch 16, 3-channel newest frame, 256×640 cylindrical, 120° field of view; 86,138,112 parameters [M-raw] | `PREREG_V7F.md:329-331`; seed stamp `target_param_count` |
| trunk policy | trainable; LR ×0.1 with a 2,000-step warmup; MSE anchor to a frozen copy of the seed, weight 1.0 | `PREREG_V7F.md:332-333,511-521` |
| objectives | O5 L1 rollout 1.0, O6 SIGReg 0.1, O14 future-observation 1.0 (k 4), EMA target τ 0.996 fixed, `o5_k` 60 (6 s), BPTT truncate 4; LDAD is a placeholder | `PREREG_V7F.md:336-346` |
| conditioning | `omega_accel_v` = measured ego state [yaw rate, a_long, v/10], predictor only | `PREREG_V7F.md:474-478`; `…/2026-09-05-v7f-state/RESULT.md:317-319` |
| nav | `--nav-labels` + `--nav-cond`, mandatory at all three layers | `PREREG_V7F.md:325`; `train_v6_staged.py:9207-9210` |
| labels | v7.2 `s2-geom-v7`, md5 `0ff90213…` (train) / `aa12c948…` (eval) | `GOALS_AND_CLAIMS.md:5525` |
| what the labels actually do at S-W | nothing but nav input: strategic supervision is refused at S-W, and no tactical loss exists | `train_v6_staged.py:10791-10800,2410-2415` |
| corpus | B1 `physicalai-b1-w120-256x640cyl`, 4,713 clips = 4,572 train + 141 eval with pixels; eval clips excluded by default (`--exclude-eval-clips auto`) [M-reg] | `MODEL_REGISTRY.md:2714`; `train_v6_staged.py:10001` |
| parity key | `episode_uid_sha256 e8bfb98e06eb…`, derived from the manifest and never checked against cache bytes | `GOALS_AND_CLAIMS.md:5525`; `…/v7f-state/RESULT.md:279-285` |
| size | total 245,588,568; declared trainable 143,949,315; effective trainable 140,192,128 (97.4 %) [M-inh; the census argv was read by me] | `…/2026-09-06-v7f-budget/RESULT.md:157`; `GOALS_AND_CLAIMS.md:8552` |

- The size census argv omits the anchor, nav and label flags (`…/v7f-budget/raw/v7f_budget_census.py:32-54`).
- The anchor teacher is built outside the stack (`train_v6_staged.py:9028-9035`), and `param_report` counts only stack parameters (`v6.py:6310-6347`). Whether the nav and label flags add parameters is UNVERIFIED.

---

## 3. Where it is in the launch sequence

### Launch gate, item by item

**GREEN:**

| item | evidence |
|---|---|
| B1 corpus built and trained on by refcv3 | `MODEL_REGISTRY.md:2714` |
| eval-clip exclusion implemented, default on | `GOALS_AND_CLAIMS.md:5189`; test passes |
| encoder-seed path (`--init-encoder-from`, alias `--enc-init-from`), trunk LR flags, anchor, Observer-Effect monitor | all present in argparse [M-me] |
| DINOv3 B/16 seed built 2026-09-18 | `PI_DECISION_QUEUE.md:1273-1319`; lives only at `D:/Projects/TanitAD-artifacts/dinov3-seeds/` |
| EMA teacher removed from the optimizer; `--refuse-unreached` / `--allow-unreached`; declared-freeze preflight | `GOALS_AND_CLAIMS.md:8552,8607` |
| collapse gate now rules on `participation_ratio` | `v6.py`, [M-me] present |
| τ-ramp closed NEUTRAL | `MODEL_REGISTRY.md:4722` |
| 12 test files pass: 245 passed / 3 skipped | [M-me]; the one guard skip is below |

**RED:**

| item | evidence |
|---|---|
| PI hold of 2026-08-31 | `V7_LAUNCH_GATE.md:3-4`; only the PI closes items (`:12-15`) |
| 2026-09-26 launch gate absent for this trainer | `CLAUDE.md:1297-1310`; no gate file, and no PASS-token code in `stack/` or `taniteval/` [M-me, two probes] |
| launch line: 4 refusals, plus a silent 4×4 readout | table below |
| LDAD unimplemented | argparse and substring search [M-me] |
| stage S-W cannot yield the G-DRIVE read | `v6.py:4567-4573`; `PREREG_V7F.md:391-399` |
| metric-decode guard missing | `stack/tests/test_metric_decode_refusal.py:34-58` (a module-level skip); 0 implementation files [M-me] |
| `H-V7F-1` not in the register (prereg precondition 8, SPEC refusal 1) | 0 hits against 57 for the control `H-ESTIM-SEED-1` [M-me] |
| `actdiv_anchored.py` has no pinned denominator or bootstrap (precondition 9) | 0 hits against 32 for the control `F_sep` [M-me] |
| `m_t` slot not reserved (precondition 6) | `stack/tanitad/models/predictor.py:358-360` returns only `out[k]` [M-me] |
| rungs R0–R3 and the G-RANK reference unrun (preconditions 3–5) | `Project Steering/BACKLOG.md:148` |
| no tactical loss or max-speed learning, against PI directive 2026-09-15(c) | `Decisions/2026-09-15-pi-directives.md:37`; 0 max-speed hits in the trainer [M-me] |
| frozen-trunk advisory unanswered for v7f | `GOALS_AND_CLAIMS.md:15808` |

**UNKNOWN:**
- Compute target and slot. The §9 paths are Thor paths (`PREREG_V7F.md:317`). The PI's 09-15 directive puts preparation on the dev box and heavy training on a pod he will provide (`Decisions/2026-09-15-pi-directives.md:39`). The pod comes only after dev-box proof (`PI_DECISION_QUEUE.md:1311-1313`) and is "later" (`:1388`). The order of work is refcv6 → refav1 → v7 (`pi-directives:40`). refcv7, registered 2026-09-26, is planned for Thor (`Project Steering/SPEC_REFCV7.md:66-70`).
- Stability at v7F width was never measured (`STATUS_BOARD.md:65-66`).
- Wall-clock time and memory are unmeasured.
- Steps for one epoch: about 76,000 at batch 8 [EST: 132.92 windows/episode × 4,572 ÷ 8, from `…/v7f-state/RESULT.md:355-366`]. Per `GOALS_AND_CLAIMS.md:5711` this must be recomputed at launch.

### The launch line, executed [M-me]

Source: `PREREG_V7F.md:312-360`, run through the real argparse and preflight.

| variant | result |
|---|---|
| §9 as written | argparse: `unrecognized arguments: --w-ldad 1.0 --ldad-target a_kappa --ldad-form delta_z` |
| minus LDAD | preflight refusals: `--horizons (1,2,4)` (`train_v6_staged.py:9266`); `--w-s2-goal 1.0 in S-W` (`:10791-10800`, present since `ee02ff71`, 2026-08-16); a label path absent on this box (environment only). The anchor preflight also refuses: `--w-trunk-anchor > 0 REQUIRES --obs-monitor-every > 0` (`:8978-8986`) |
| `--horizons 1`, drop `--w-s2-goal`/`--s2-labels`, add `--obs-monitor-every 250` | preflight 0 problems; anchor preflight PASS |

- The 2026-09-05 dry run (`…/v7f-state/raw/v7f_launch_line_dryrun.json:6-19`) could not have seen the anchor refusal. `main()` runs `preflight()` first (`:11024`), and the anchor check runs only inside `dry_run`/`train` (`:5875`, `:6495`).
- Silent defect: no `--readout-grid-w`, so the readout falls back to 4×4 (`:9254-9255`; also noted in `Reports/2026-09-15…:168`).

---

## 4. Results that exist

v7F itself: **none.** Everything below is v7-tiny or a probe.

| measurement | number | class · tier | source |
|---|---|---|---|
| non-collapse, participation σ² val-side (n 1,440, d 2,048) | `emao14_30k_tauramp` 24.975, `emao14_30k` 24.659, `o14fut30k` 22.969, `postrain30k_freeze` 6.357; reference `rdw8p30k` 25.582 reproduces banked 25.583 | M-inh · T0 | `GOALS_AND_CLAIMS.md:7743` |
| absolute participation bar | none admissible (8.56 not reproducible) | M-reg · T0 | `MODEL_REGISTRY.md:5154`; `GOALS:7744` |
| L2, the latent carries the environment | `splitp30k` `n_agents` +0.3881 (t 28.21) vs frozen DINOv3 +0.2754; wins 1 of 5 targets; lead range −0.1611 | M-reg · T0 | `MODEL_REGISTRY.md:4898`; `STATUS_BOARD.md:165` |
| L3, does the predictor add over `z_t` | max \|t\| 1.08 across 9 cells vs a null bar of 2.9; **no arm passes** | M-inh · T0 | `GOALS:7750` |
| action sensitivity, h1 action/scene ratio | k60 0.002983 vs `postrain30k` 0.00595, i.e. 0.50× against a ≥10× bar | M-raw · T0 | `ref:…/2026-08-31-mm-e19-k60-horizon/raw/mm_e19_read_step30000.json` |
| share of response from the speed channel | 97 % | M-inh · T0 | `GOALS:7619` |
| arms whose nrmse moves >0.1 % when actions are noised | 0 of 32 | M-reg · T0 | `MODEL_REGISTRY.md:4955-4957` |
| predictor cos at h=1 | `rdw8p30k` 0.6224, `splitp30k` 0.5481 | M-reg · T0 | `MODEL_REGISTRY.md:4848,4896` |
| stability | k60 at clip 1.0 diverged (gnorm 2.1e9); max gnorm 1.931e10; 0 NaN rows across 43 run dirs | M-inh · T0 telemetry | `GOALS:7611` |
| drift | effect null once its control is subtracted | M-inh · T0 | `GOALS:7749` |
| T1 read (3 arms) | ADE ~14 m, heading ~95°, tactical κ≈0, strategic unavailable. **FALSE RESULT**: decoded through a bit-identical, untrained readout; no paired CI | M-inh · T1, VOID | `GOALS:8945,7609,8992` |
| seed as wired (ViT-L/16) | see §1a | M-inh · none | E-SEED RESULT `:63-71,105-109` |
| frozen refcv5-v2 trunk, Δz → longitudinal accel | R² 0.3297 [0.2384, 0.3957], +0.3279 over the pixel floor, controls pass; lateral VOID-LEAK | M-inh · none | `ref:…/2026-09-18-ldad-displacement-probe/RESULT.md:17-21` |

Four metric families: the only T1 read is void. LON was inadmissible, LAT heading at chance, TAC at chance, and STRATEGIC unavailable; STRATEGIC is now PAUSED by PI ruling (`CLAUDE.md:1275-1283`).

---

## 5. How v7F is supposed to be proven

**The ladder.** Each rung has a deliberate-regression arm that must fail (`PREREG_V7F.md:289-295`; `…/2026-09-03-v7f-design/SPEC.md:124-132`):

| rung | question | regression arm | status |
|---|---|---|---|
| R0 | is LDAD tautological at step 0? | constant control / pixel floor | not run |
| R1 | BPTT truncation depth ∈ {0,2,4,15} | truncate 0 must diverge | not run |
| R2 | LDAD weight λ ∈ {0,1,10,50} | λ 0 | not run |
| R3 | trunk policy ∈ {frozen, full, anchored, lastk} | `full` must trip the monitor; `frozen` | not run |
| R4 | the composition | v1.7's 0.0430 T1 collapse must be reproduced | not run |

**The gates, in order:**
- **G-RANK** reports only, until a matched frozen-DINOv3 reference exists (`:180-193`).
- **G-DECODE (a)** content must beat both pixel and constant, with `frzrand` as regression arm (`:199-203`).
- **G-DECODE (b)** Observer-Effect ratio ≥ 0.70 × step 0, with the CI excluding 0.70 (`:212`).
- **G-ACT** verdict SENSITIVE and `rel_pinned(2σ) ≥ 0.0595` with its CI excluding the bar; the denominator is pinned to the R3 scratch arm (`:227-229`). A pre-committed G-ACT-WORSE branch reads below 0.0087 (`:231`).
- **G-DRIVE** T1, closed-loop minus hold-action ADE separated below 0 by paired bootstrap, `trivial_profile_fraction` < 1, `baseline_won_frac` < 0.7571, all four families (`:244-252`).
- Committed failure branches: `:81-87`, `:554-570`. Checkpoint selection rule: `:257-276`.

**What has passed:** nothing, on v7F. On proxies, G-ACT-equivalent readings of 0.0032–0.0181 (`:234`, T0) all sit below the bar, and L3 fails. The recipe's own ladder L0–L5 (`Project Steering/V7_RECIPE_AND_SCALEUP.md:261-270`) never reached L4.

**Defects in the proof itself:**
- R0's rule uses "floor" in two opposite senses (`PREREG_V7F.md:291`); it must be pinned before running.
- G-DRIVE cannot be read from an S-W run, as §3 shows.
- `baseline_won_frac` presupposes a planner with a baseline; v7F plans through a latent goal head (`GOALS:7259`). Whether the instrument applies is UNVERIFIED.

---

## 6. Contradictions and staleness

The registry wins by rule. Each row names what needs fixing.

| conflict | detail |
|---|---|
| registry T1 table vs register | `MODEL_REGISTRY.md:4724-4742` presents the T1 read as a measured floor; `GOALS:8945,8996` call it a FALSE RESULT; 0 correction hits in the registry [M-me] |
| register "guard shipped" vs code | `GOALS:8946` says wired at five sites; the symbols are absent and the test skips; `train_v6_staged.py:5584-5585` references it in prose only |
| replicate-rig attribution | `CLAUDE.md:180-188` says v7-tiny; it was the REF-C rig (§1b) |
| parameter budget | 336.5 M INHERITED (`PREREG_V7F.md:24,491`) vs 245,588,568 measured (`Reports/…:162`) |
| τ-ramp "never ran" | `…/v7f-state/RESULT.md:61` vs ran and read NEUTRAL (`GOALS:7433`; `MODEL_REGISTRY.md:4722`) |
| "exactly two launch defects" | dry-run JSON and `GOALS:7255,7275` vs four measured here |
| BACKLOG rows R8, R15, R23, R24 open | `BACKLOG.md:133,146,165,166` vs all done |
| `V7_LAUNCH_GATE.md` | quotes the void T1 read (P1 `:23-27`); lists as unrun an arm that ran (`GOALS:7603`); calls the O11 arm abandoned when it was adjudicated degenerate (`:7613`); quotes withdrawn h2/h4 numbers (`:7615`); its "nearest decision" tail (`:738-743`) predates its own MM-E19 answer (`:284-300`) |
| `V7_RECIPE_AND_SCALEUP.md` | "Collapse SOLVED" (`:77`, contradicted by `GOALS:7605`); `--o5-k 8` and parity corpus (`:132,145`); nav path "not emitting" (`:200`, closed `V7_LAUNCH_GATE.md:376`); step checkpoints absent (`:201`); 8.56 bar (`:266`) |
| prereg says existing flags "do not exist" | `PREREG_V7F.md:308-309,415-417`; they exist now |
| P4 plans | `products/P4-training-pipelines/V7_FULLSCALE_*` (Aug 23–25) superseded by D1 = trainable encoder (`V7_RECIPE:174`) |
| launch-gate tool name | `CLAUDE.md:1297` names `launch_gate.py`; `SPEC_REFCV7.md:39` names `launch_gate_refcv7.py` and says it "landed" (`:76`); neither exists at the ref |
| parity invariant | `CLAUDE.md:944` names the old parity corpus; v7F uses B1 by PI decision (`V7_LAUNCH_GATE.md:676-679`) |

---

## 7. Cheapest experiments toward a positive, pre-registered result

| # | experiment | compute · cost | pre-registered bar |
|---|---|---|---|
| 1 | Make §9 preflight-clean (the three flag fixes, explicit readout grid, fill `--steps`, seed path on the host) and register `H-V7F-1` | none · minutes | enables everything below |
| 2 | S0: E-SEED-2 at the launch geometry B/16, with the ImageNet fold and zeroed `pos` | dev-box 4060 · ~15 min of forward passes [M-inh] | `PREREG_V7_SEED_POS.md:47-61`; scene half failed at L/16, so a positive is uncertain |
| 3 | R0 on the B/16 seed, after pinning the "floor" wording | 4060 · minutes | `PREREG_V7F.md:291`; the frozen-trunk longitudinal R² 0.33 suggests the longitudinal half clears |
| 4 | G-RANK matched reference, `actdiv_anchored` pinned denominator + clip bootstrap, `m_t` slot | CPU · hours | preconditions 3, 6, 9 |
| 5 | Predictor-side LDAD on v7-tiny: two arms plus a replicate, λ = 0 control | 4060 or Thor · a small code change + ~3 tiny arms | ratio ≥ 3× at equal-or-better O5, and the replicate does not reproduce the rise (`ref:…/ldad-displacement-probe/RESULT.md:37`). This is the largest candidate effect on the programme's binding P2 statistic, with published jumps of 0.002 → 0.59–0.76 (PUB, `PREREG_V7F.md:233`) |
| 6 | First admissible v7 T1 read, on an O1-trained checkpoint (e.g. `o1ctrl30k`) or a re-roll with latent dumping | GPU · ~11 min per arm [M-inh] | makes P1 readable; `ref:…/2026-09-06-metric-decode-hazard/RESULT.md:258-275` |
| 7 | A1 command-channel arm (a_lon, a_lat) | ~8.6 h ×2 [EST] | committed at `GOALS:7261`; weaker prior because the corpus holds no command in its bytes (`V7_LAUNCH_GATE.md:90-101`) |

Items 2–5 fit the PI's "prove on the dev box first" ruling. Item 6 needs a GPU slot and ~11 min per arm.

---

## Deliverable manifest

| artifact | where | single location? |
|---|---|---|
| this report | reply to the coordinator | yes; the coordinator banks it |
| `v7f_preflight_probe.py` + `.out.json` (the measured launch-line refusals) | scratchpad `…\ff19d6ac…\scratchpad\` | yes; worth banking beside the report |
| `pytest_v7f.out.txt` (165 passed / 2 skipped; the second batch, 80 passed / 1 skipped, went to console only) | scratchpad | yes |
| `guard_import.py`, plus extracted copies `b16seed.json`, `mme19.json`, `tree_ref.txt`, `tree_snap.txt` | scratchpad | yes; ref content, no need to bank |

Nothing was staged: this was a read-only survey by instruction, the D: tree is stale, and G: is off-limits.

## Not read or not verified

- Thor-only artifacts (T1 dumps, v7-tiny run dirs, `actdiv_2k.json`) and current Thor/pod occupancy.
- Whether a launch gate landed after `b3f7ea6f`.
- v7F step time, memory and stability.
- Whether the nav and label flags add parameters.
- The source pair of the "3.5 %" band.
- Which argv produced the `--dry-run` exit 0 in `GOALS:8607`.
- The applicability of `baseline_won_frac` to v7F.
- The seed `.pt` on D: (only its stamp was read).
- The step-checkpoint code (`PREREG_V7F.md:275`), not re-verified.
- The full suite; only 12 targeted files were run.
- Library PDFs, the EvalFlyWheel, `taniteval/results` and `taniteval/tests` (outside the snapshot).
- A full-history pickaxe for the missing guard was stopped; two other probes agree that it is absent.