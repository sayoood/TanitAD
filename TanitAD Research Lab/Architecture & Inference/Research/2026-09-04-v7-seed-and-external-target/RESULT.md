<title>RESULT — E-SEED-2: the DINOv3 seed loses a separated fraction of DINOv3's SCENE content and no cheap repair recovers it; the self-generated-target problem is narrowed, not closed</title>

# RESULT — E-SEED-1/2/2c: **the DINOv3→`ViTEncoder` seed loses a large, SEPARATED fraction of DINOv3's SCENE content, and neither free wiring repair recovers it** — the repair that looked decisive is an EGO-axis artefact of one probe, and my own follow-up measurement is what refuted it

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-v7-seed-and-external-target/`
**Compute** dev-box RTX 4060 (8,187 MiB) only. ⛔ Thor and pod `tanitad-refcv3` were never contacted. No training was launched.
**Evidence classes** `MEASURED (ours + artifact path)` · `PUBLISHED` · `INHERITED (a register row / brief, NOT re-verified here)` · `UNVERIFIED`.
**Tier** ⛔ **none of these is a driving number.** Representation / decodability diagnostics only.
**Builds on** the killed 2026-09-03 run's `LIT.md` (banked, staged) and its `E-SEED-1` panel, whose raw JSON is re-read here rather than inherited. Its draft `RESULT` was written at 00:11 and its own disambiguating control landed at 00:17; **this file reconciles them and retracts one of its four escalations.**

---

## 0. ⛔ ESCALATE FIRST — seven items that are decisions, not notes

| # | escalation |
|---|---|
| **X1** ⛔⛔ | **`PREREG_V7F.md`'s premise fails its first held-out measurement in the geometry v7f will actually use.** At 256×640 (the v7 geometry), the seeded trunk exactly as `V6Stack.encode_window` feeds it is **not separable from a randomly-initialised encoder** (`seed_asis − scratch` = **+0.1671 [−0.1742, +0.6124]**, spatial speed), while published DINOv3 **is** separated from that same floor (**+0.5021 [+0.0779, +1.0576]**) and from raw pixels (**+0.9049 [+0.4426, +1.4878]**). ⇒ the Observer-Effect ρ 0.91 argument (§6.2b), the `anchored` arm and `--w-trunk-anchor` all reference a trunk whose function had never been measured. |
| **X2** ⛔⛔ **CORRECTED BY MY OWN FOLLOW-UP — READ THIS BEFORE THE TABLES BELOW** | An earlier draft of this file said the two free wiring repairs (ImageNet normalisation + `pos` zeroed) *"close the gap"*. **They do — on the EGO axis, under a linear ridge scored by pooled R². They do NOT on the ENVIRONMENT axis under the register's own instrument, and the environment axis is the one that matters.** E-SEED-2c (§2.4) runs the SAME cached features through `rangeprobe_rff.rff_fold` + `within_clip_r` + `panel_kfold.kfold_clip_scores` — all IMPORTED from the register's modules — and reads: `seed_imnet_pos0 − seed_asis` on `n_agents` **+0.0295 [−0.0690, +0.1246] NOT separated**, and the repaired seed does **not** beat the raw-pixel floor (**+0.0061 [−0.1236, +0.1318]**). ⇒ **the repair is free, correct hygiene and is NOT a content fix.** Its rank in §4 is downgraded accordingly. |
| **X2b** ⭐⭐ **THE FINDING THAT SURVIVES, AND IT IS THE STRONGER ONE** | **The seed loses a large, SEPARATED fraction of DINOv3's SCENE content, and no cheap repair recovers it.** Under the register's own instrument, `n_agents`: `dino_hf` **+0.2401 [+0.116, +0.365]** (which reproduces the register's frozen-DINOv3 **+0.2754** inside its interval — an independent cross-check that the instrument is the register's) vs `seed_asis` **+0.0351 [−0.017, +0.089]** and `seed_imnet_pos0` **+0.0646 [−0.023, +0.147]**, both CIs containing zero. `dino_hf − seed_asis` **+0.2050 [+0.0896, +0.3233] SEPARATED**; `dino_hf − seed_imnet_pos0` **+0.1755 [+0.0525, +0.2959] SEPARATED**. ⇒ **this is direct evidence for `PREREG_V7F` §10 D1 option A** (wrap the real `DINOv3ViTModel` so RoPE comes with the weights) **and against the current converter route**, and it means the trunk anchor — whose teacher IS this seed — anchors toward a field that does not beat raw pixels on scene content. |
| **X2c** ⛔ **AND THE TWO PROBES DISAGREE FOR A NAMED REASON — A DIFFERENT ESTIMAND, NOT A BUG** | The register's environment numbers are a **within-clip Pearson r** (`rangeprobe_rff.within_clip_r`, whose constant control returns EXACTLY 0.0); my first pass scored **pooled R² against the scored-set mean**. On a target whose variance is mostly BETWEEN clips these can disagree completely, and they did: the same frozen DINOv3 reads **−0.0769** (pooled R²) and **+0.2401** (within-clip r) on the same rows. ⛔ **No number of one kind may be compared with a number of the other**, and §2.1's environment column stays VOID *as an R² read* for exactly that reason. |
| **X3** ⛔ | **RETRACTION of the killed run's own X2.** Its draft says *"the dominant cause is a two-line, exactly-foldable bug"* — the ImageNet normalisation — and calls it *"the single highest-return change available to v7 tonight"*. **That mechanism claim is withdrawn by two independent measurements**: (a) its own `eseed1_v3` one-variable control, pre-committed, shows **real DINOv3 retains 101.5 %** of its speed R² under `[0,1]` input (0.4498 → 0.4564), whose own written criterion reads *"normalisation is NOT the cause and (A) architecture carries the loss"*; and (b) at the v7 geometry normalisation **alone** is **+0.0992 [−0.1100, +0.2910], NOT separated.** The fix is still correct and still free — it is simply not the mechanism, and it does not carry the effect by itself. |
| **X4** ⛔ | **The v7f launch line, as published, REFUSES on three counts** (MEASURED at HEAD by argparse dump + two independent greps): `--enc-init-from` **does not exist** (the implemented spelling is `--init-encoder-from`, and the trainer's own docstring at `train_v6_staged.py:7927` says so); `--w-ldad / --ldad-target / --ldad-form` **do not exist anywhere in `stack/`** (`grep -ril ldad` → **0 files**); and the line pairs a **ViT-B/16** trunk (`--enc-dim 768 --enc-depth 12 --enc-heads 12`, `--trunk-anchor-model …vitb16…`) with the **only seed that exists**, which is **ViT-L/16** — a pairing `build_trunk_anchor` refuses by design (`train_v6_staged.py`: *"disagrees with the SEED's own provenance stamp"*). `dinov3-vitb16` is **not in the local HF cache** (two probes). ⚠️ §9's *"flags that do NOT exist"* list is otherwise **stale** — `--init-encoder-from`, `--trunk-lr-scale`, `--trunk-lr-warmup-steps`, `--w-trunk-anchor`, `--trunk-anchor-model`, `--exclude-eval-clips` all exist and the anchor is **wired**. |
| **X5** ⛔ | **A LATENT ZERO-COLLISION IN THE v7 NAV CHANNEL, of exactly the family the PI flagged tonight.** `apply_nav_control(..., "none")` returns `torch.zeros_like(token_id)`, and `NAV_COMMAND_TOKENS[0]` is **`NAV_FOLLOW_ROAD`** — the modal command on a driving corpus. ⇒ **the `none` ablation is not an ablation; it is a constant-FOLLOW_ROAD arm**, and it would manufacture the conclusion *"nav buys nothing"*. `channel_active=False` is returned but `NavConditioner.embed` has only **3** slots, so the model is still told "follow road". ⭐ **It has never been used** (two probes: no caller anywhere in `stack/` or `taniteval/`; no `--nav-control` argparse entry), so this is a defect to fix **before** it is quoted, not a retraction. **Fix: a 4th `NAV_NONE` slot, or a validity flag concatenated to the embedding.** ⭐ `PUBLISHED-PRIMARY` support (`LIT_SOTA.md` S5): an explicit missingness INDICATOR is the field's standard fix and is beneficial **even under MCAR** (`lib:2407.19804`), which names our exact defect — a constant imputation **becomes** the missingness code; and `lib:2511.13079` prices **zeroed ego velocity at an 8.7x planning degradation**. |

---

## 1. THE FIVE-ISSUE AUDIT (the PI's mandate: drift · encoder quality · prediction · representation · decodability)

Each row: what is MEASURED today, the best-supported hypothesis, and the **cheapest discriminating experiment**. Where two of our own documents disagree, both are named.

| # | issue | MEASURED state today | best-supported hypothesis | cheapest discriminating experiment |
|---|---|---|---|---|
| **1** | **DRIFT** | Our "drift" is `r` of a linear probe predicting `Δz` from `z_t` = **0.669–0.6709** across every trainable arm. **`INHERITED`**: `E-DEC-67` — *"no measured lever moves drift on the trainable line"*; O14-fut absorbed the pixel-marginal (+0.0096 t 5.11 → −0.0047 t −3.09) and left drift at 0.6709. **⛔ TWO DOCUMENTS DISAGREE ON WHAT "DRIFT" MEANS**: `E-DEC-66` called O14 *"the only live anti-drift lever"*; `E-DEC-67` corrects it in place. And `2026-09-03-sota-drift/RESULT.md` §0.1 establishes that **the field's drift is a rollout-vs-encoded-future divergence** (`JEPA-x`: `E‖ẑ−z‖²/E‖z_t−z‖²`) and **no primary defines it as increment predictability** ⇒ **no published number may be quoted against our 0.669.** | Our "drift" statistic is **misnamed**, and the quantity that matters operationally — rollout divergence — **has no instrument here** (backlog L-17). The refcv3 open-loop result (`INHERITED` from the orchestrator brief: hold-action **0.2996** < model **0.4799** < constant-velocity **0.6723** on 4,823 windows) is the divergence question in its deployed form and it is **lost**. ⚠️ **AND IT MAY BE A STATEMENT ABOUT THE VAL MIX, NOT THE MODEL** (`LIT_SOTA.md` S4, `lib:2312.03031` BEV-Planner): **73.9 % of driving is straight (87 % of eval samples)**, and a model that loses its input adopts a conservative straight policy that then **wins the pooled average**. There is **no field standard** for trivial controls (0/121 corpus texts, two arXiv probes). ⇒ **the hold-action comparison must be stratified by manoeuvre before it decides anything.** | Build L-17 (the divergence instrument) and read it on the **already-banked** `rdw8p30k` / `splitp30k` / `o14fut30k` checkpoints — **0 new GPU-training**. Both outcomes: if divergence tracks our 0.669 across arms, the rename is cosmetic; if it does not, every "drift" conclusion in the register is about a different quantity than the one that loses to hold-action. |
| **2** | **ENCODER QUALITY** | ⭐⭐ **MEASURED TONIGHT, ON BOTH INSTRUMENTS (§2.4 is the binding one): the seed loses a SEPARATED fraction of DINOv3's SCENE content (`dino_hf − seed_asis` +0.2050 [+0.0896, +0.3233]) and no cheap repair recovers it (+0.0295 [−0.0690, +0.1246], not separated; and the repaired seed does not beat raw pixels).** Ego-axis detail, linear read: The v7f trunk at step 0, 256×640, 24 clips, 2,372 rows: `dino_hf` spatial speed **+0.5072 [+0.078, +0.622]**; `seed_asis` **+0.1722 [−0.273, +0.308]**; `scratch` **+0.0050 [−0.613, +0.384]**; `pixel` **−0.3977**. `seed_asis − scratch` **NOT separated**. `seed_imnet_pos0 − seed_asis` **+0.2165 [+0.0243, +0.4567] SEPARATED**; `dino_hf − seed_imnet_pos0` **+0.1185 [−0.096, +0.263] NOT separated.** Independently, `INHERITED` `E-DEC-14b-R`: at parity a **frozen distilled** encoder reads `n_agents` **+0.3881** vs the trainable **−0.0180**. | ⛔ **The loss is ARCHITECTURE, not wiring** — which is the opposite of what an earlier draft of this file concluded, and my own §2.4 is what refuted it. Wiring (normalisation + `pos`) moves the EGO axis under one probe and leaves the SCENE axis where it was. What remains is the declared architectural loss: DINOv3 is RoPE-only, so no `pos` value can restore a positional operator the transplanted `Wq`/`Wk` were co-adapted to (`lib:2403.13298` §3 Eq. 16). Consistent with `lib:2008.11687`: the transferable benefit is concentrated near the INPUT, which is where `pos` lives. | **DONE tonight** (§2). The follow-on is the ENV replication (§3), which this rig could not deliver. |
| **3** | **PREDICTION** | `INHERITED`, `H-PROOF-5`: **every arm predicts at h=1 and is indistinguishable from noise at h≥2** — `rdw8` z 3.99/0.80/0.84, `o7w1p0` 4.10/0.87/−1.86 — *including the arm whose latent improved most*. `E-DEC-19`: at parity, predictor cos h=1 **0.6224 (z 30.55)**, an 11.5× rise over the tiny arm. `E-DEC-63`: our predictor is **at the achievable ceiling of this latent to within noise** (predictor marginal over drift −0.0023, t −1.84, null) while **raw pixels retain +0.0096 (t 5.11)**. | Prediction beyond h=1 is a **separate defect from the representation** (H-PROOF-5 is explicit), and the h≥2 heads **are never trained** under the two-term recipe (`E-DEC-67`'s vestigial-head note). ⇒ *"buy predictor capacity"* is refuted; *"supervise h≥2"* is untested. | A **one-variable** tiny arm on `--o5-k` with the h≥2 heads actually in the loss vs the incumbent, reading the composed-h1 rollout (H-PROOF-7) as the control. ⛔ Its deliberate-regression arm is the incumbent, which **must** reproduce z≈0 at h≥2. |
| **4** | **REPRESENTATION** | `INHERITED`: participation at parity **25.58 / 26.96** (`rdw8p30k`) — the collapse the campaign fought is **largely absent at scale**; the frozen-distilled arm is **6.38 / 7.63** and *wins* the capability axes. ⛔ **The 8.56 floor is RETIRED** and the same encoder reads **5.756 / 20.228** on two corpora, so participation measures sample diversity as much as representation. `C131`: the highest rank ever measured here belonged to an arm with no environment interpretation. | **Rank is settled as necessary-not-sufficient and is no longer the binding constraint.** The binding constraint is CONTENT: `E-DEC-17` — a frozen **random** field gives predictor cos **0.0016** while a frozen **distilled** field gives **0.1872**; *"freezing is necessary but not sufficient"*. | ⛔ **None is worth buying.** The register already carries the discriminating result (E-DEC-17 + E-DEC-14b-R). The work item is to **stop gating on rank**: G-RANK must be demoted to *report-only* until its commissioned matched reference exists (`PREREG_V7F` §6.1 already says this — it is not yet done). |
| **5** | **DECODABILITY** | ⭐ **RESOLVED TONIGHT, IN TWO STEPS — AND THE FIRST STEP'S VERDICT WAS WRONG.** Pass 1 (linear ridge, pooled R²) was VOID: On all three `n_agents` definitions (register-canonical PSG sum, in-FOV count, all-cuboid count) **every arm including frozen DINOv3 reads BELOW the constant control**, and `dino_hf − scratch` = **+0.1387 [−0.3232, +0.5740] NOT separated** ⇒ instrument validity fails and **no seed conclusion may be drawn from that column.** ⛔ **THIS CONTRADICTS THE REGISTER'S +0.2754 FOR THE SAME ENCODER ON THE SAME TARGET NAME**, and the difference is the *instrument*: the register's panels use an **RFF (nonlinear)** probe (`rangeprobe_rff.py`, `panel_kfold.py::rff_fold`); mine is a **linear ridge on PCA-256** at n=24 clips. | `CLAUDE.md`'s own rule applies to me: **a negative from a LINEAR probe is not a negative about learnability.** The environment axis needs the nonlinear probe and ≥130 clips; at 24 clips with clip-disjoint folds the count target is underpowered by construction. | Re-run **this exact arm set** through `rff_fold` + `kfold_clip_scores` on the **130-clip** `physicalai-val130-heldout` bank (both already on this box). ⛔ Pre-committed: **if `dino_hf − scratch` is still not separated, the register's own environment numbers need re-deriving, not my panel.** |

---

## 2. E-SEED-2 — the measurement (MEASURED, `raw/eseed2_*.json`)

**Corpus** the 24 clips of `physicalai-val-w120-256x640cyl`, UUID-joined to `val40_agents.jsonl` (|val24 ∩ val40| = **24**, MEASURED). Newest frame, PNG-decoded (`codec` field read, not the `jpeg_buf` name), 256×640 cylindrical, stride 2 ⇒ **n = 2,372 rows / 24 clips**. Frame bank content-asserted (mean **63.099**, non-zero fraction **0.8879**).
**Splits / estimator** 4-fold **clip-disjoint** out-of-fold prediction; PCA basis **and** ridge λ fitted inside each fold's training clips only (λ by an inner 4-fold `GroupKFold` over those clips); pooled R² against the scored-set mean; **clip-cluster bootstrap, 2,000 draws**, and the **paired** bootstrap for every contrast. Nothing is selected on a row it scores.
**d** ambient 1,024 (global) / 16,384 (spatial 4×4); probe d = 256.

### 2.1 The panel — spatial 4×4 (the read the prereg's monitor uses)

| arm | speed (EGO) | `n_agents_psg` (ENV) |
|---|---|---|
| `dino_hf` — published DINOv3 ViT-L/16 | **+0.5072 [+0.078, +0.622]** | −0.0769 [−0.627, +0.235] |
| `seed_asis` — **the trunk as `V6Stack` feeds it** | +0.1722 [−0.273, +0.308] | −0.1405 [−0.403, −0.021] |
| `seed_imnet` — + ImageNet normalisation | +0.2714 [−0.142, +0.388] | −0.1144 [−0.528, +0.056] |
| **`seed_imnet_pos0` — + `pos` zeroed** | **+0.3886 [+0.038, +0.504]** | −0.1311 [−0.729, +0.133] |
| `scratch` — random init ⛔ **deliberate-regression arm** | +0.0050 [−0.613, +0.384] | −0.2156 [−0.638, +0.018] |
| `pixel` — raw-input floor | −0.3977 [−1.189, −0.056] | −0.0014 [−0.205, +0.077] |
| **constant-only control** | **0.0000 (exact, by construction)** | **0.0000 (exact)** |

### 2.2 The contrasts that decide it — paired clip-cluster bootstrap, spatial 4×4, **speed**

| contrast | question | Δ | CI95 | separated |
|---|---|---|---|---|
| `dino_hf − scratch` | **INSTRUMENT VALIDITY** | **+0.5021** | [+0.0779, +1.0576] | **YES** ✅ |
| `dino_hf − pixel` | does the reference beat raw input? | **+0.9049** | [+0.4426, +1.4878] | **YES** ✅ |
| `seed_asis − scratch` | **is the seed AS WIRED better than random?** | +0.1671 | [−0.1742, +0.6124] | ⛔ **no** |
| `seed_imnet − scratch` | is normalisation alone enough? | +0.2663 | [−0.1120, +0.7496] | ⛔ **no** |
| `seed_imnet − seed_asis` | does normalisation alone help? | +0.0992 | [−0.1100, +0.2910] | ⛔ **no** |
| **`seed_imnet_pos0 − seed_asis`** | **does the FULL repair help?** | **+0.2165** | **[+0.0243, +0.4567]** | ⭐ **YES** |
| `seed_imnet_pos0 − pixel` | does the repaired seed beat raw input? | **+0.7863** | [+0.3578, +1.4657] | **YES** |
| **`dino_hf − seed_imnet_pos0`** | **how much does the repaired seed still lose?** | **+0.1185** | **[−0.0960, +0.2628]** | ⭐ **no — not separable from DINOv3** |
| `dino_hf − seed_asis` | how much does the as-wired seed lose? | **+0.3350** | [+0.0902, +0.5541] | **YES** |
| `seed_imnet_pos0 − scratch` | does the repaired seed beat random? | +0.3836 | [−0.0320, +0.9359] | no (2.3× the as-wired Δ) |

**The same two conclusions hold on the `global` read** — `seed_imnet_pos0 − seed_asis` **+0.1730 [+0.0136, +0.3803] SEPARATED**; `dino_hf − seed_imnet_pos0` **+0.1281 [−0.1933, +0.2830] NOT separated** — so the finding is not an artefact of one pooling.

### 2.3 ⛔ THE BOUNDS, STATED RATHER THAN ROUNDED AWAY

1. ⛔ **This is measured on `speed`, an EGO target, and `E-DEC-17` binds: *"no ego number may be cited as evidence that an objective worked"*** — a frozen **random** encoder read the best speed of three arms (+0.3552). ⇒ this panel measures **TRANSFER FIDELITY** (does the seeded trunk behave like DINOv3?), which is a legitimate and different question from **scene content**. It does **not** establish that the repaired seed carries DINOv3's scene content.
2. ⛔ **The ENVIRONMENT column is VOID** (§1 row 5): its own reference arm fails the instrument-validity control. Every ENV cell above is reported for completeness and **none of it is quotable**.
3. ⚠️ **`seed_imnet_pos0 − scratch` is NOT separated.** The strongest admissible claim is *"the repair moves the seed from separated-below-DINOv3 to not-separated-from-DINOv3, and is separated from the unrepaired seed"* — **not** *"the repaired seed beats a random encoder"*.
4. ⚠️ **Measured on ViT-L/16** (the only seed that exists, the only DINOv3 in the local cache). `PREREG_V7F`'s launch line specifies **ViT-B/16**. The *mechanism* (no positional tensor; CLS/registers dropped) is identical by construction — the converter declares it — but the **magnitude** is an L/16 number.
5. ⚠️ **E-SEED-1's magnitudes do not replicate at this geometry and must not be quoted.** At 256×256 it read `seed_asis` **+0.0034** against `scratch` **+0.0266**; at 256×640 the same arms read **+0.1722** and **+0.0050**. The **ordering and the non-separation replicate**; the *"~1 % of the function"* framing does not.


### 2.4 ⭐⭐ E-SEED-2c — THE SAME FEATURES THROUGH THE REGISTER'S OWN INSTRUMENT

⛔ **Why this run exists.** §2.1's environment column came back VOID. Two causes were possible with opposite consequences, and both were **committed in advance** in `code/e_seed2_rff.py`'s docstring before it ran: **(I)** the instrument — the register's environment panels use `rangeprobe_rff.rff_fold` (PCA-96 → random Fourier features D=1024, RBF, median-heuristic bandwidth → ridge, λ on a **clip-disjoint** inner split) scored by `within_clip_r`, a **within-clip Pearson r**, not pooled R²; or **(P)** power — 24 clips is simply too few.

**The verdict is (I).** The features are the **byte-identical cached arrays**; only the probe changed. Every module is **imported** from the register's own code, never re-implemented.

| arm | `n_agents` (within-clip r) | speed | row-shuffled control |
|---|---|---|---|
| `dino_hf` | **+0.2401 [+0.116, +0.365]** — reproduces the register's frozen-DINOv3 **+0.2754** inside its interval | +0.1547 [−0.003, +0.299] | +0.0051 [−0.027, +0.039] ✅ |
| `seed_asis` | +0.0351 [−0.017, +0.089] | +0.1196 [+0.030, +0.211] | −0.0525 [−0.090, −0.014] |
| `seed_imnet` | +0.0730 [−0.017, +0.163] | +0.1533 [+0.081, +0.226] | +0.0269 [−0.010, +0.066] |
| `seed_imnet_pos0` | +0.0646 [−0.023, +0.147] | +0.0630 [−0.036, +0.171] | +0.0132 [−0.037, +0.070] |
| `scratch` ⛔ **deliberate regression** | **−0.1100 [−0.257, +0.040]** | +0.0845 [−0.031, +0.192] | +0.0134 [−0.031, +0.058] |
| `pixel` raw-input floor | +0.0585 [−0.097, +0.212] | +0.0642 [−0.121, +0.244] | −0.0061 [−0.043, +0.032] |
| **constant-only** | **0.0000 exactly** (`within_clip_r` returns exactly 0.0 for a zero-variance prediction) | ″ | ″ |

**Paired clip bootstrap, `n_agents`:**

| contrast | Δ | CI95 | separated |
|---|---|---|---|
| `dino_hf − scratch` — **INSTRUMENT VALIDITY** | **+0.3501** | [+0.1822, +0.5228] | **YES** ✅ |
| `dino_hf − pixel` — beats the raw-input floor | **+0.1816** | [+0.0104, +0.3484] | **YES** ✅ |
| `seed_asis − scratch` | +0.1451 | [+0.0130, +0.2718] | **YES** — the seed does carry *something* |
| `seed_imnet_pos0 − scratch` | +0.1746 | [+0.0224, +0.3249] | **YES** |
| ⛔ **`dino_hf − seed_asis`** | **+0.2050** | [+0.0896, +0.3233] | **YES — the seed loses scene content** |
| ⛔ **`dino_hf − seed_imnet_pos0`** | **+0.1755** | [+0.0525, +0.2959] | **YES — and the repair does not close it** |
| ⛔ **`seed_imnet_pos0 − seed_asis`** — does the repair help the SCENE? | +0.0295 | [−0.0690, +0.1246] | **no** |
| ⛔ **`seed_imnet_pos0 − pixel`** — does the repaired seed beat raw input? | +0.0061 | [−0.1236, +0.1318] | **no** |

⚠️ **And on `speed`, under THIS instrument, NOT ONE contrast is separated** — including `dino_hf − scratch` (+0.0702 [−0.1154, +0.2475]). ⇒ **the ego-axis result of §2.2 is a property of the linear/pooled-R² read and does not survive the register's instrument.** Two probes disagreeing on the ego axis while agreeing that the seed loses scene content is exactly why the environment axis is the one quoted.

⭐ **What this buys, stated plainly.** The panel is **no longer VOID**: it has a valid instrument (`dino_hf − scratch` separated), a raw-input floor DINOv3 clears, a deliberate-regression arm that reads **negative**, a row-shuffled control at the constant value, and an independent cross-check (DINOv3 reproducing the register's own number). **On that panel the answer to "does the seed carry DINOv3's scene content" is: a little, separated from random, and separated well below DINOv3 — and not above raw pixels.**

---

## 3. ⭐ IS THE SELF-GENERATED-TARGET PROBLEM CLOSED BY THE CURRENT v7 RECIPE?

**No. It is narrowed. And the one term that MEASURABLY closed it is explicitly switched off.**

The test is E-DEC-7's own: *can this term be satisfied by "ego motion + noise" with zero scene content?* Applied term by term to `PREREG_V7F` §9's launch line, read from that line and from the trainer's source:

| term in v7f | where its target comes from | satisfiable by ego + noise? | evidence |
|---|---|---|---|
| **O5** `--o5-target ema --ema-decay 0.996` | the model's **own EMA** | ⛔ **YES** | `E-DEC-13` REFUTED the EMA route in **both** published forms: *"a slow copy of our own encoder is external in the formal sense and still produces nothing"* (`n_agents` → **−4.36**, and −2.97 for the DMT-JEPA variant) |
| **O6** SIGReg | isotropy only | ⛔ **YES — exactly** | `E-DEC-7`: noise satisfies isotropy; `2607.27017`: the regulariser *"constrains the DISTRIBUTION of embeddings, not their CONTENT"* |
| **O14-fut** `--w-o14 1.0 --o14-mode fut` | a future observation aux | partly no | `E-DEC-67`: it **ABSORBS** the pixel-predictive marginal (+0.0096 → −0.0047) — a real gain — but **does not reduce drift** and costs prediction (cos −5.5 % rel) |
| **LDAD** (proposed, `--w-ldad`) | the action labels `(a, κ)` | ⛔⛔ **YES, and this is the sharpest finding of the audit** | `E-DEC-57` (MEASURED, r **0.9988**): **our action channel is not a command — it is the ego's measured motion in different units.** ⇒ LDAD's "external" target **is** the ego motion that E-DEC-7's degenerate optimum already encodes. It is a self-generated target wearing a label's clothes. ⚠️ And `E-SEED-1` MEASURED that at init `Δz → (a, κ)` reads ≈ 0 on **every** arm *including real DINOv3*, with the endpoint-shuffled control equal — so there is no head-start either |
| **trunk anchor** `--w-trunk-anchor 1.0` | a **frozen deepcopy of the model's OWN SEED** (`build_trunk_anchor(… stack.encoder …)`, taken after `apply_encoder_seed`) | it **PRESERVES**, it cannot **SUPPLY** | `E-DEC-17`: immovability is necessary, not sufficient — *"a frozen empty field is an empty field"*. **And §2 measures what this particular field contains**: as wired, not separable from random. ⛔ **The anchor's teacher inherits the random `pos` verbatim**, so the `pos` repair must land in the converter, before the deepcopy |
| **O7** `--w-o7-distill 0` | frozen DINOv3 — genuinely external **content** | ⛔ **NO** | `E-DEC-8`: pure distillation moved `n_agents` **−1.0407 → +0.3274** (t 12.63, 24/24) with **no cost to ego** — the only term that ever put the scene in. **v7f sets it to zero.** |
| **s2 / nav labels** | external goal/route labels | no — but they are **goal**, not scene | PI binding: the goal path must stay information-disjoint from the situation classifier. And `E-DEC-18b/c`: a supervised state head **destroyed the predictor even when no gradient reached it** |

⭐ **THE ANSWER, PRECISELY.** v7f replaces an external target whose content is the **scene** (O7) with an external target whose content is **the model's own initialisation** (the trunk anchor). That is a change of *kind*, not of *strength*: an anchor bounds drift-from-init, it cannot add information the init lacks. The recipe therefore closes the self-generated-target problem **only to the extent that the seed itself carries scene content** — which was the load-bearing unmeasured premise, and which §2 measures on the ego axis and **cannot** measure on the environment axis.

**Is the external target strong enough to dominate? The honest answer is that the question is not yet well-posed for v7f, and the one time we DID pose it the answer was no.** `E-DEC-9` MEASURED, at w = 1.0, that adding O7 to O5+O6 delivers `n_agents` **−0.2553** where O7 **alone** delivers **+0.3274** — *"the self-referential terms are still steering the latent toward the ego+noise optimum and partially winning."* ⚠️ **That is a 2k-step, 130-clip measurement and `E-DEC-19` / `H-SCALE-2` re-scope it**: tiny-arm screening is valid for architecture, invalid for capability levels. The balance question is architectural, so the *direction* survives; the *level* does not.

⇒ ⭐⭐ **AND TONIGHT'S LITERATURE PASS SETTLES IT FROM THE FIELD, ON OUR EXACT OBJECTIVE SHAPE.** `LIT_SOTA.md` (sibling package, this date) closes the named empty search the killed run had left open. **`lib:2509.10156` (LayerLock) Table 3** adds a **self-generated-target latent loss** to an **external-target (pixel) loss as a weighted sum** and measures **SSv2 50.1 → 3.7**; its own caption reads *"Adding latent losses to the MAE paradigm without freezing leads to representation collapse"*. ⛔ **BOTH weight schedules tried — constant AND cosine — collapsed (3.7 and 5.6).** Only **staging with FREEZING** worked. ⇒ **E-DEC-9 is the published failure mode, and the published verdict is that a `w`-sweep is not the fix.** Our own register already holds the matching positive: `E-DEC-12` (distil first, then self-supervise) **+0.1327** where co-training gave **−0.2553**, and `E-DEC-14b-R` (freeze the distilled trunk) **+0.3881** where the trainable trunk gave **−0.0180**.

⛔⛔ **RETRACTION OF THE KILLED RUN'S ESCALATION #3 — GradNorm is NOT the instrument.** `…/2026-09-03-v7-issue-audit/LIT.md` escalation #3 and §3 item 5 recommend replacing the O7 `w`-sweep with GradNorm (`lib:1711.02257`). **On the two banked tables whose objective has our shape — an external supervised term co-trained with self-supervised auxiliaries — GradNorm LOSES to the sweep it was proposed to replace**: `lib:2010.08244` Table 1, CIFAR-10 test error GradNorm **14.07** vs grid search **13.76**, SVHN **7.68 vs 6.07**; `lib:2110.14048` Table 4, GradNorm **81.03** vs a fixed-weight baseline **81.35**. GradNorm's own supporting result is **NYUv2, three SUPERVISED tasks** — no self-generated-target term anywhere. It also does not remove a sweep; it swaps K weights for one α whose good value differs **12×** between its own two settings.

⭐ **The correct instrument is `lib:2505.08170` (MoKD) §3's TWO scalars, not one norm**: **Gradient Conflict** `⟨g_ext, g_self⟩` and **Gradient Dominance** `‖g_ext‖/‖g_self‖`, measured there at **10⁻³–10⁰** in a real distillation run — the distillation gradient up to **1000× smaller** than the task gradient. And it names the reason a weight is the wrong control, which is **not** GradNorm's reason: the norm ratio is **non-stationary** (*"the norm of gradients varies throughout optimization"*), so any fixed `w` is correct at one step only. **A forward/backward hook on the already-banked `o7w1p0` arm, 0 GPU-training.** That is V7A-2 in §5's pre-registration, now with the right statistic.

⚠️ **And the tempting third option is measured to be a trap.** MoKD Table 3 ablates separation against balancing inside one hybrid and separation wins 4.3× (**+1.3 AP** vs **+0.3**) — but `lib:2609.03150` measured **near-disjoint expert routing (J̄ ≈ 0.056) with +1128 perplexity of negative transfer at `cos = −0.002 ± 0.003`**: **interference WITHOUT conflict**, invisible to a PCGrad-style conflict test. ⇒ **a naive "O7 head vs O5 head" split is not the experiment**; output-level separation does not separate the parameters.

---

## 4. ⭐ RANKED RECIPE CHANGES

Ranked by (measured effect) ÷ (cost), with the honest evidence class on each.

| rank | change | cost | evidence | status |
|---|---|---|---|---|
| **1** | ⭐⭐ **ADOPT `PREREG_V7F` §10 D1 OPTION A — wrap the real `DINOv3ViTModel` as the trunk instead of seeding `ViTEncoder`.** The converter route loses a **separated** fraction of DINOv3's scene content that **no cheap repair recovers**, and the trunk anchor inherits exactly that field. | a wrapper class + its loader | **MEASURED tonight** (§2.4): `dino_hf − seed_asis` **+0.2050 [+0.0896, +0.3233]**, `dino_hf − seed_imnet_pos0` **+0.1755 [+0.0525, +0.2959]**, both separated; the repaired seed does not beat raw pixels (+0.0061 [−0.1236, +0.1318]) | **the decision this package exists to inform** |
| **2** | ⚠️ **Seed `pos` to ZERO and fold the ImageNet affine into the patch embed — as free HYGIENE, not as a content fix.** ⛔ **DOWNGRADED by §2.4**: separated on **ego** under a linear/pooled-R² read (+0.2165 [+0.0243, +0.4567]) and **not** separated on **scene** under the register's own instrument (+0.0295 [−0.0690, +0.1246]). Still correct, still free, and it must land in the CONVERTER because `build_trunk_anchor` deepcopies after `apply_encoder_seed`. | **2 lines, 0 GPU** | **MEASURED tonight**, both readings | **PRE-REGISTERED in §5, with the ego/scene split written into its success criterion** |
| **3** | ⛔ **Do not run v7f's launch line as published.** Fix `--enc-init-from` → `--init-encoder-from`; either build a ViT-B/16 seed or move the line to ViT-L/16 (the anchor cross-check refuses the current pair); delete the `--param-budget` correction note, which names a flag the block does not contain. | doc + 1 flag rename | **MEASURED** (argparse dump + 2 greps + source of `build_trunk_anchor`) | **work item, X4** |
| **4** | ⭐⭐ **STOP CO-TRAINING THE EXTERNAL AND SELF-GENERATED TARGETS; STAGE THEM WITH A FREEZE.** Our own `E-DEC-12` (+0.1327 vs co-trained −0.2553) and `E-DEC-14b-R` (+0.3881 vs trainable −0.0180) already measured it; `lib:2509.10156` Table 3 is the published form (**SSv2 50.1 → 3.7**, and **both** weight schedules collapsed). ⚠️ `lib:2302.14138` Table 1: **the ORDER matters more than the staging** — right order **+8.9** over joint, wrong order **15.1 below** joint. | recipe change, no new code | **MEASURED (ours) + PUBLISHED-PRIMARY** | **the leading design change** |
| **5** | ⭐ **Measure the balance instead of sweeping the weight** (V7A-2): log MoKD's **conflict** `⟨g_ext, g_self⟩` **and dominance** `‖g_ext‖/‖g_self‖` at the encoder output of the banked `o7w1p0` arm. ⛔ **NOT GradNorm** — refuted on our objective shape (see §3). | **0 GPU-training**, a hook | `PUBLISHED` `lib:2505.08170` §3 + `MEASURED` `E-DEC-9` | **PRE-REGISTERED in §5** |
| **6** | ⛔ **Reconsider `--w-o7-distill 0`.** The anchor cannot supply what the seed lacks; O7 is the only term that ever moved the environment axis. ⚠️ **But re-enable it STAGED, not co-trained** (rank 3) — running it back into the weighted sum reproduces the LayerLock collapse. | 1 tiny arm pair | `MEASURED` `E-DEC-8/9`, re-scoped by `E-DEC-19` | **decision for the Master Mind** |
| **7** | ⛔ **Give `nav` a 4th `NAV_NONE` slot (or a validity flag).** Today `none` = constant `NAV_FOLLOW_ROAD`. | 1 vocab slot | **MEASURED** from source, **never yet used** | **work item, X5** |
| **8** | ⚠️ **Demote G-RANK to report-only** until its commissioned matched-corpus/matched-`d` DINOv3 reference exists. `PREREG_V7F` §6.1 already commits to this and the reference has not been measured. | 1 `spectrum_report` run | `C131`, `C132`, `C135` | **work item** |
| **9** | ⛔ **Do not adopt LDAD on the `(a, κ)` channel without first discharging R0** — `E-DEC-57` makes its target the ego's own motion, i.e. the degenerate optimum's content. The prereg's R0 rung already refuses the family if the read is at the floor at init; **E-SEED-1 measured that it IS ≈0 at init on every arm**, which is the *other* horn: nothing to start from. | 0 GPU (already measured) | `MEASURED` + `E-DEC-57` | **escalate to the prereg's owner** |

---

## 5. THE PRE-REGISTRATION

`Project Steering/PREREG_V7_SEED_POS.md` — **one variable (`pos` initialisation), both outcomes committed in advance, three controls, and a deliberate-regression arm the gate must FAIL.** It is a separate file so it can be executed without re-opening `PREREG_V7F.md`.

---

## 6. WHAT THIS PACKAGE DOES **NOT** ESTABLISH

1. ⛔ **No v7-tiny TRAINING arm was run.** The gate reported here is the **step-0 tier on two instruments**, each with its own deliberate-regression arm, and both fail it (`scratch` reads +0.0050 [−0.613, +0.384] pooled-R² on speed and **−0.1100** within-clip-r on `n_agents`). The mission's compute rule forbade a full run; the tiny ladder was not started because the environment-axis instrument failed validity and a training arm scored on a void axis would have been uninterpretable. **The gate reported here is the step-0 tier**, with its deliberate-regression arm (`scratch`) correctly failing.
2. ⛔ **Nothing here says the repaired seed carries scene content.** See §2.3.1–2.
3. ⛔ **Nothing here measures the trunk anchor in training.** The anchor's *teacher* is characterised; its *effect* is not.
4. ⚠️ The refcv3 open-loop numbers in §1 row 1 are `INHERITED` from the orchestrator's brief; the in-repo `RESULT-refcv3-40284-openloop.md` reads **PENDING** in its own §0 and does not carry them.

---

## 7. EVIDENCE-CLASS LEDGER

| claim | class | source |
|---|---|---|
| every number in §2.1–2.3 | **MEASURED (ours)** | `raw/eseed2_eseed2_panel.json`, `raw/eseed2_eseed2b_panel.json`, `raw/eseed2_eseed2b_extra_contrasts.json`, `code/e_seed2_*.py`, dev-box RTX 4060 |
| every number in §2.4 (**the binding panel**) | **MEASURED (ours)** | `raw/eseed2_eseed2c_rff.json`, `code/e_seed2_rff.py`; probe modules IMPORTED from `rangeprobe_rff.py` / `panel_kfold.py` in `C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901\` — ⚠️ **those two modules live on the dev box only and are NOT in the repo**; they are the register's environment instrument and that is a stranding risk in its own right |
| E-SEED-1's 256×256 numbers, `dino_hf_01` retained fraction 1.0148 | **MEASURED (ours)** | `raw/eseed1_eseed1_v2.json`, `raw/eseed1_eseed1_v3.json` — re-read from the JSON, not from the killed run's prose |
| `STAGE_GROUPS["S-S"] == ("layer_str",)`, `S-W == ("encoder","readout","predictor_op","aux")` | **MEASURED** | `stack/tanitad/models/v6.py:4374-4380` |
| `--w-ldad` absent (2 probes); `--init-encoder-from` etc. present | **MEASURED** | repo-wide `grep -ril ldad` → 0 files; argparse dump of 228 flags; repo and mirror byte-size identical (611,966) |
| the trunk anchor's teacher is a deepcopy of the live seeded encoder | **MEASURED** | `stack/scripts/train_v6_staged.py::build_trunk_anchor` |
| `NAV_COMMAND_TOKENS[0] == "NAV_FOLLOW_ROAD"`; `apply_nav_control("none")` zeroes ids; `NavConditioner.embed` has 3 slots; no caller | **MEASURED** | `stack/tanitad/models/vocab_v7.py:331-333`, `stack/tanitad/models/nav_conditioning.py:220-241`; 2 probes for the absence of a caller |
| `n_agents` = sum of the PSG count channel | **MEASURED** (imported, not re-implemented) | `stack/tanitad/data/psg_targets.py` module docstring + `frame_target` |
| the seed's 292 mapped / 1 left-at-init / 3 skipped, ViT-L/16 | **MEASURED** | `…/2026-09-03-v7f-design/raw/dinov3_vitl16_seed_stamp.json` |
| E-DEC-7/8/9/12/13/14b-R/17/19/57/59/63/67, H-PROOF-5, H-SCALE-2, C131/132/135 | **INHERITED** | `Project Steering/GOALS_AND_CLAIMS.md`; **not re-measured here** |
| the four SOTA theme verdicts (drift / encoder / action / decodability) | **INHERITED** | the four sibling `2026-09-03-sota-*/RESULT.md`; their primaries are banked, their numbers not re-verified here |
| GradNorm's claim; the 0/83 corpus grep | **PUBLISHED-PRIMARY / MEASURED** | `lib:1711.02257`, banked by the killed run; grep recorded in `…/2026-09-03-v7-issue-audit/LIT.md` §3.5 |
| refcv3 hold-action 0.2996 / model 0.4799 / ha0 0.6723 | **INHERITED** | orchestrator brief; the in-repo result file is PENDING |

---

## 8. DELIVERABLE MANIFEST

| artifact | location | state |
|---|---|---|
| this file | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-v7-seed-and-external-target/RESULT.md` | staged + committed |
| the pre-registration | `repo:Project Steering/PREREG_V7_SEED_POS.md` | staged + committed |
| E-SEED-2 raw JSON (panel, relabelled panel, extra contrasts, bank meta) | `repo:…/2026-09-04-v7-seed-and-external-target/raw/` | staged + committed |
| **E-SEED-2c raw JSON (the binding RFF panel)** | `repo:…/2026-09-04-v7-seed-and-external-target/raw/eseed2_eseed2c_rff.json` | staged + committed |
| ⚠️ `rangeprobe_rff.py` + `panel_kfold.py` — **the register's environment instrument** | `C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901\` | ⛔ **ONE PLACE, and it is not the repo.** Every environment number the campaign quotes runs through these two files. **ESCALATED: they belong in `stack/` or `taniteval/`.** |
| E-SEED-1 raw JSON (v2 panel, v3 one-variable control, paired) | `repo:…/2026-09-04-v7-seed-and-external-target/raw/` | staged + committed — **rescued from the killed run's scratchpad, where it lived in ONE place** |
| the code for both experiments | `repo:…/2026-09-04-v7-seed-and-external-target/code/` | staged + committed |
| the killed run's literature pass | `repo:…/2026-09-03-v7-issue-audit/LIT.md` | staged + committed (was uncommitted) |
| feature banks (`feat*.npy`, `frames_u8.npy`, ~5 GB) | scratchpad only | ⚠️ **ONE PLACE** — deliberately not banked (regenerable in ~15 min from `code/`; the inputs are on this box) |
| the DINOv3 ViT-L/16 seed checkpoint (1.2 GB) | scratchpad only | ⚠️ **ONE PLACE** — regenerable from `stack/scripts/dinov3_seed_checkpoint.py` + the HF cache; its **stamp** is already in the repo |
