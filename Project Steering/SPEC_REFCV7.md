# refcv7 — refcv6 + DrivoR's proven planning tricks ("DrivoR-T")

**Status: DRAFT for PI review (2026-09-19).** Requested by the PI 2026-09-19: *"Build the DriveR-T
refc variant, combine all our advantages with their proven tricks, keep the hierarchy etc.,
prepare it to be trained with our last data corpus, review the implementation and check its
quality, we will call it refcv7."*

⭐ **refcv7 = refcv6 exactly as specified and PI-ruled (`SPEC_REFCV6_V2.md` §10–§12, 416 × 1024)
PLUS three planning components from DrivoR/TOAD.** Nothing refcv6 carries is removed. Every
new component sits behind its own flag, is OFF in a refcv6 build, and is bit-inert when off.

Evidence base: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-19-drivor-deep-analysis/RESULT.md`
(DrivoR `2601.05083`, TOAD `2606.07170`, CLOVER `2605.15120`, DriveZero `2609.06055`, all read in
full from banked PDFs).

---

## 1. What refcv7 keeps from refcv6 — our advantages (unchanged, binding)

| ours | kept as |
|---|---|
| ImageNet-pretrained timm trunk (DrivoR's single largest lever: random 70.1 → pretrained 90.0 PDMS) | `resnet101` primary / `resnet34` comparison, exactly §10.2 |
| **K = 3 frame history** (DrivoR has none; its own failure case) + **past-only ego history** | §10.3 |
| **416 × 1024** cylindrical geometry, nothing hard-coded | §12 |
| BEV lift + **SAM3 map head** + **3-D box head**, all backpropagating into the trunk | §6, R2/R3 |
| **Tactical layer**: DETR behaviour decoder over agents + BEV (map), valid-behaviour set | §4, R2 |
| **Nav MANDATORY** in condition, tactical layer and selection; nav → turn allowed | §5, §10.5 |
| **Max speed = 4-value INPUT** {30, 50, 100, 120} km/h, containing-window rule | §5, §10.4 |
| DD-faithful diffusion decoder F1–F9 over the **117 v0-rolled anchors** | §3 |
| strategic layer **deactivated** (PI); flags remain, OFF | §1 |
| gradient-conflict detector, per-head gradient-reach reporting | R2 cost 1 |
| vision-only inference; labels may use anything | PI 2026-08-03 |

## 2. What refcv7 adds — DrivoR / TOAD, each with the measurement that earns it

| # | component | module | the evidence (primary, single-run unless stated) |
|---|---|---|---|
| **D1** | **WTA proposal decoder**: 64 learned queries, ONE token per whole trajectory, winner-takes-all L1 on the 8 V3 slots; condition (nav, max speed, v0, ego history) added to every query | `refcv7_heads.WTAProposalDecoder`, `wta_loss` | DrivoR Tab. 5: 1 → 64 queries **80.1 → 90.0**; Tab. 12: per-pose → single token **83.9 → 90.0** |
| **D2** | **Disentangled scorer**: its own 4-layer decoder over the shared scene tokens; candidates enter ONLY as re-embedded waypoints behind a **stop-gradient**; one head per **oracle sub-score**; condition-aware (nav must shape selection) | `refcv7_heads.DisentangledScorer`, `scorer_loss` | DrivoR Tab. 6: shared **84.7** → separate 86.8 → disentangled **90.0**; one score 88.2 vs sub-scores 90.0 |
| **D3** | **PhysicalAI oracle** the scorer learns from (LABEL-SIDE ONLY): NC + TTC (`obstacle.offline` boxes), DAC (SAM3 drivable, unseen = abstain), EP (vs the human path), COMF (NAVSIM limits, read from source), SPD (set-speed compliance) | `refcv7_oracle` | replaces NAVSIM's PDM oracle, which needs an HD map we do not have |
| **D4** | **Off-proposal scorer training**: the scorer is also shown control-space perturbations of the candidates | trainer | TOAD Tab. 3: a vocabulary-fit scorer as a search reward **34.7 → 23.9**, DrivoR's **→ 49.8** |
| **D5** | **Selection by the aggregated scorer**: PDMS-shaped (gates NC · DAC · SPD × weighted TTC/EP/COMF), fused with refcv6's nav-compliance term, speed mask and tactical validity prior; **behaviour profile** weights settable at inference | `refcv7_oracle.aggregate` + trainer | DrivoR §3.4, Fig. 7 (safety-oriented agent) |
| **D6** | **TOAD test-time search** (eval / inference only): CEM on REF-C's own slot controls through `roll_controls` (the vocabulary's integrator), warm start = selected plan, trust-region anchor + NAVSIM-comfort penalty, never returns a plan scored below the base, explicit seed | `refcv7_toad` | TOAD: DrivoR **54.6 → 56.3** navhard; **+1.9 ms** at K 5 / M 64 when the scorer shares the backbone |

### 2.1 Deliberate deviations from DrivoR (each stated, none accidental)

1. **No ego pose/velocity/acceleration at inference.** Only v0 (PI 2026-09-02) + past ego history
   (§10.3) + nav + max speed. DrivoR never ablates its ego status; we cannot use it.
2. **Scorer self-attention OFF** (a flag restores it). A test-time search reward must be a
   function of the trajectory alone; with self-attention a candidate's score depends on its
   companions. `test_refcv7_heads.py` proves the coupling exists with it ON and is absent OFF.
3. **Our oracle, not NAVSIM's.** Six sub-scores, not PDMS's; NC/TTC by a conservative circle
   cover of the boxes; DAC abstains (never passes) on unseen ground.
4. **DrivoR's "accelerated second target" is OFF**: it helped NAVSIM-v1 (+0.6) and hurt the
   out-of-distribution v2 warm-up (−1.6, Tab. 7).
5. **No register compression in refcv7.** DrivoR's registers live INSIDE a ViT; refcv6's trunk
   is a PI-ruled ResNet, and the CNN equivalent (a query-decoder compressor) is the variant
   DrivoR measured **worst** (89.3 vs pooling 89.7). The ViT + registers trunk is the separate
   arm **DR-4** (`LAB_BACKLOG.md`), not part of refcv7.
6. **Candidates = anchors' diffusion fan ∪ 64 WTA proposals.** The anchor path is our existing
   investment and stays; the WTA set removes the vocabulary as the only source of candidates.

## 3. Model at a glance

```
3 frames 416x1024 ─► pretrained ResNet (shared weights) ─► stride-16 / stride-32 maps
   ├─► BEV lift ─► BEV tokens ─┬─► SAM3 MAP head   (label: SAM3)
   │                           ├─► 3-D BOX head    (label: obstacle.offline)
   │                           └─► TACTICAL behaviour decoder (+ agent slots) ─► valid behaviours, lat/lon prior
   └─► image tokens
 nav · max-speed · v0 · ego history ─► CONDITION ─► every decoder / query
 SCENE = {image tokens, BEV tokens, agent slots}
   ├─► DIFFUSION DECODER over 117 anchors (F1–F9) ────────────┐
   ├─► [D1] WTA DECODER, 64 queries ──────────────────────────┤ candidates (detached)
   │                                              + [D4] control perturbations (train only)
   └─► [D2] DISENTANGLED SCORER ◄─────────────────────────────┘  ─► 6 sub-score logits
                 ▲ BCE vs [D3] ORACLE (labels only)
 SELECTION [D5]: aggregate(sub-scores) + nav-compliance + speed mask + tactical prior ─► plan
 [D6] TOAD (inference): CEM around the plan, reward = scorer ─► final plan
```

## 4. Training on the latest corpus

**The corpus is `Sayood/tanitad-v7-training-corpus` (private HF)** — the one the PI ordered
augmented on 2026-09-15. refcv7 consumes four of its layers, and every one is already used by
refcv6 except the future-agent block:

| layer | what refcv7 uses it for | state (2026-09-20) |
|---|---|---|
| camera clips (4,719) | the 416 x 1024 cylindrical v2 cache | ⛔ **the train cache is NOT built** (10.6 h, 386.5 GB; eval-139 and eval-124 exist) |
| **v8.1 labels** (`s2_labels_v8_train.jsonl.gz`) | nav (`--nav-from-v7`), the 22-token tactical set, lat/lon | ✅ on D: |
| **SAM3 semantic maps** | the map head AND refcv7's **DAC** oracle | ⏳ production on Thor (ETA ~22 Sep); 135/139 eval maps on D: |
| **`obstacle.offline` 3-D join** | the box head AND refcv7's **NC/TTC** oracle (now read at every waypoint slot) | ✅ eval join on D:; ⭐ MEASURED: the join labels **every frame** (spacing 1), so the future slots resolve |

### 4.1 Order of operations — nothing here may be reordered

1. **SAM3 production finishes** (the DAC oracle and the map head both wait on it).
2. **Build the 416 x 1024 train cache** — check the HF quota BEFORE the push (+112.9 GB over the
   256-wide plan, a hard ceiling).
3. ⭐ **Derive the nav tolerance ON THE TRAIN SPLIT**:
   `python stack/scripts/refcv7_derive_nav_tau.py --v2-cache <train> --labels <v8 train> --json tau.json`
   and pass its `tau` as `--r7-nav-tau-rad`. ⛔ Never the eval split.
4. **Run the refcv6 pipeline validation** unchanged, then the refcv7 smoke (§4.3) on the eval
   clips, on CPU or a free GPU — never on a card that is training.
5. **Launch the ladder** A0 → A0b → B → C, then re-evaluate C with `--r7-toad` as D
   (`PREREG_REFCV7.md` §1).

### 4.2 The launch command (refcv6's, plus refcv7)

```
--refcv7 --w-r7-wta 1.0 --w-r7-scorer 1.0 --r7-nav-tau-rad <from step 3> --r7-n-perturb 32 [--r7-no-select]            # arm B
[--r7-toad]                 # arm D, EVAL ONLY
```
Everything else is the refcv6 arm verbatim (trunk, geometry, K=3 history, ego history, nav,
max-speed input, the two perception heads, the tactical decoder, the conflict detector).

### 4.3 What has been validated on the dev box (MEASURED 2026-09-19/20)

* the refcv7 modules and their wiring: **60 tests** (oracle, heads, TOAD, model, trainer);
* an end-to-end trainer run on **real eval-139 data at 416 x 1024** (CPU, resnet34): both refcv7
  terms report `TRAINS / graph=yes` in the trainer's own effective-weights table, the run writes
  `config.json` with the two ledger lines (`refcv7_wta` **4,493,840**, `refcv7_scorer`
  **3,429,127**) and every `r7_*` telemetry key in `metrics.jsonl`;
* ⛔ **NOT validated:** anything at corpus scale, on GPU, or about quality. No refcv7 arm has
  trained. Every number above is a wiring fact.

## 5. Acceptance — pre-registered before any GPU hour

`PREREG_REFCV7.md`: arms A0 / A0b / B / C / D, the controls that must read known values, the
four-family gates, and the refusals the launch checker enforces.

## 6. Cost

| | params | note |
|---|---|---|
| refcv7 WTA decoder | **4,493,840** | MEASURED at d_model 256, 4 layers, 64 queries |
| refcv7 scorer | **3,429,127** | MEASURED, 7 sub-score heads |
| total added | **7.92 M** | on a refcv6 arm of ~42.5 M (resnet101) / ~21.3 M (resnet34) trunk |

⚠️ Training cost per step and eval-time TOAD cost are **not yet measured on GPU** (`PREREG` G5).
