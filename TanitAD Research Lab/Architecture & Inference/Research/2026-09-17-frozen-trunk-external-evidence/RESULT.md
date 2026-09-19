<title>The frozen trunk gets its first same-backbone external datapoint — and WA-JEPA names our predictor's objective as the defect</title>

# AI13-2 is supported from outside for the first time (frozen > fine-tuned on the SAME 14B backbone), while WA-JEPA attaches **+1.0 EPDMS** to replacing deterministic regression with flow matching

**2026-09-17 · Research Lab (LAB-RUN-014) · Architecture & Inference · discharges the A3 dedicated-DEEP debt (pre-committed rotation item 1, scan-only for two passes) · touches AI13-2, row 9, row 5, P-2, H-RANK-16**
⛔ **Tier: none of ours.** Every external number below is the source's own tier and is stamped as such. ⛔ No training was run; the 4060 was **not available** (four `python.exe` in `nvidia-smi`'s compute-apps list, 100 % util).

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **AI13-2 has its first SAME-BACKBONE, matched external comparison — and frozen wins on every metric.** FROST-Drive, Waymo Open E2E, Table 3: **VLM (14B) Frozen — RFS 8.17 ±0.015, ADE@3s 1.04, ADE@5s 1.88**; **VLM (14B) FinetuneOnly — RFS 8.13 ±0.014, ADE@3s 1.47, ADE@5s 2.19**; ViT FinetuneOnly 7.79 / 1.20 / 2.15. The authors state plainly: *"the fine-tuned 14B VLM performs worse than its frozen counterpart."* ⇒ freezing is not merely a compute concession here; it **beat** fine-tuning at equal backbone. | PUBLISHED lib `2601.03460` Tab. 3 |
| **F2** | ⭐⭐ **And the frozen thing is 300 M — our own scale, not a 14 B one.** FROST-Drive's `InternVL3-1B`/`-14B` rows both freeze a **300 M vision encoder** (the 38B/78B rows freeze 6 B). ⚠️ **V-5 unit trap, and it is the one most likely to corrupt a quote:** *"14B frozen"* names the **VLM**, while what is frozen and doing the perception is **300 M**. Any sentence of the form *"they froze a 14 B encoder"* is wrong. | PUBLISHED lib `2601.03460` |
| **F3** | ⚠️⭐ **The same paper's ablation runs straight into our measured readout-geometry ceiling.** Table 4, frozen 38B, embedding dimensionality: **256 → RFS 7.68**, 2202 → 7.57, 4147 → 7.81, **5120 → 8.17** — *"high-dimensional features are critical for this task"*, a **+0.49 RFS** spread from width alone. Our v6 readout pools to **4 azimuth bins over 120°** (measured), i.e. we compress hard at exactly the stage they find is load-bearing. | PUBLISHED lib `2601.03460` Tab. 4 + INHERITED (our readout-geometry measurement) |
| **F4** | ⛔⭐⭐⭐ **WA-JEPA indicts the objective our predictor uses, by name, and attaches a number to the fix.** Verbatim: *"V-JEPA is built around random-mask completion and **deterministic regression**, making it **fundamentally ill-suited** for autonomous driving planning that demands future-directed prediction tightly coupled with action."* Its ablation separates the two levers: masking (full 91.3 · patch 91.0 · both **91.7**, i.e. **+0.4**) and **flow matching vs regression: +1.0 EPDMS** — the larger of the two by 2.5×. | PUBLISHED lib `2608.20974` Tab. 4(b-c) |
| **F5** | ⛔⭐⭐ **WA-JEPA's headline is `navtest`, and must never meet our navhard numbers.** Table 1 caption: *"Comparison with state-of-the-art methods on NAVSIM-v2 **navtest**"* — WA-JEPA **91.7**, Discrete-WAM 90.4, SparseDriveV2 90.1. Our stamp table is **navhard** (48.3–56.6). Closed-loop HUGSIM HD-Score: WA-JEPA **0.4462**, DrivoR 0.3252, UniAD 0.3124, LTF 0.2310, VAD 0.1393. *(Banked in full in today's B&E package.)* | PUBLISHED lib `2608.20974` Tab. 1–2 |
| **F6** | ⚠️ **Both papers' headline evidence is OPEN-LOOP, and one says so about itself.** FROST-Drive: *"open-loop setting on the validation set"*, 4,021 scenes, H = 20 (5 s), primary metric **RFS** (a rater-feedback score with speed-aware tolerance), 3rd on the Waymo E2E leaderboard. ⇒ under `EVAL_DOCTRINE` these are **T0-class** for us and cannot carry a driving-capability claim. WA-JEPA's HUGSIM row is the one genuinely closed-loop number in today's set. | PUBLISHED, both |
| **F7** | ⭐ **FROST-Drive's concession, which is the sentence that costs the author something:** *"While the ADE is higher than some other methods, this is an expected trade-off resulting from our model's explicit optimization of the custom RFS loss function."* ⇒ their win is **metric-aligned**: they optimised the scoring function. A frozen-encoder claim read off RFS alone would inherit that alignment. | PUBLISHED lib `2601.03460` |

---

## 1 · What this does and does not do to AI13-2

| | |
|---|---|
| ✅ **It supports the freeze** | F1 is the matched comparison the decision never had: same backbone, same data, same protocol, frozen ahead on all three reported metrics. Until today the freeze rested on *our* compute constraint plus an inference from the frozen-encoder literature. |
| ⛔ **It does not license "frozen is better"** | Open-loop only (F6), one dataset, one metric family the authors optimised for (F7), and a 300 M encoder inside a 14 B VLM whose language half may be doing work our 300 M-total programme has no analogue for (F2). |
| ⚠️ **It sharpens the REAL question** | AI13-2 is not *"freeze or fine-tune"* in the abstract. F3 says the **width of what the frozen encoder hands downstream** is worth **+0.49 RFS**, and our readout throws most of it away. ⇒ **the readout, not the freeze, is where our version of their lever sits.** |

## 2 · Five-dimension analysis

**(i) Frozen encoders — FROST-Drive `2601.03460` (full text)**

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | **AI13-2 is live.** V-4 ranks this first: it is the only item today that touches a decision currently open. Also row 10 (H-RANK-16) and the readout-geometry ceiling. |
| **CONSEQUENCE** | The trunk-freeze decision may cite an external same-backbone result instead of resting on our constraint. ⛔ Stamped **T0-class / open-loop**, so it informs the decision and does not settle it. |
| **COMBINATION** | ⭐ Three of our own lines now point the same way: the frozen trunk **ties pixels on distance** and **beats them on place identity** (today's Deployment package, three reads), the BEV head's `main − pixel` tie (09-13), and the 09-13 MVV datapoint that a **frozen DINOv2 matched a 9 B VLM on signs (0.484 vs 0.467)**. ⇒ *representation quality is not tracking parameter count* — the same pattern in a fourth place. ⚠️ Against that: F3 says **width matters a lot**, and we are narrow where they are wide. |
| **CHANCES / RISKS** | **Upside:** the freeze stops being a concession and becomes a defensible position with an external number — which is also strategic guideline S-2's shape. **Risks:** (a) F7's metric alignment; (b) open-loop only; (c) their frozen encoder sits under a 14 B language model, ours under nothing comparable — the transfer is to the *encoder-freezing* claim, **not** to the architecture. |
| **EXPERIMENT** | **`E-AI-WIDTH-1` (proposed AI17-1, v7-tiny ladder):** sweep the readout width our frozen trunk exposes to the predictor at matched predictor params — the current pooled geometry vs 2× and 4× azimuth bins. **Committed in advance:** if quality is flat across a ≥ 4× width change, F3 does **not** transfer and our pooling is exonerated; if it rises monotonically, the readout is a measured bottleneck and outranks every encoder-side item on the AI13-2 list. **Controls:** a constant-readout arm (must read the no-information value) and a raw-pixel floor at matched width. Report `n` and the participation of each arm. |

**(ii) The predictor's objective — WA-JEPA `2608.20974`**

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Row 5 (v7 head geometry), row 9 (anti-collapse), **P-2** (the predictor does not use its actions), and the whole O5 objective. |
| **CONSEQUENCE** | ⭐⭐ **It supplies a mechanism for our most stubborn measured defect.** Our h=1 action/scene ratio is **0.004**: the predictor barely conditions on action. A **deterministic regressor** trained against a stochastic future is driven to the **conditional mean**, and a conditional mean is action-insensitive *by construction* — the action cannot move an expectation that averages over it. WA-JEPA's **+1.0 EPDMS** for flow matching over regression is the first external number attached to that mechanism. ⛔ It does not prove our defect has that cause. |
| **COMBINATION** | ⭐⭐ This is the **third independent line converging on "the deterministic latent regression is the problem"**: WA-JEPA (F4), Sub-JEPA's frozen-projection concession read on 09-15 (*"learned projections can align with directions that reduce the effective strength of the regularizer"*), and our own P-1/P-2 proposals. ⚠️ And it **cuts against** today's other headline: Fast-WAM shows future *generation* is unnecessary **at test time** while being essential **during training** — which is exactly compatible, because WA-JEPA's flow matching is a **training objective**. ⇒ **The two combine rather than conflict: model the future stochastically while training, do not roll it at inference.** That is a coherent design position and it is not the one v7 currently holds. |
| **CHANCES / RISKS** | **Upside:** a training-objective change that needs no inference-time cost, and it targets the defect four arms have failed to move. **Risks:** (a) WA-JEPA pre-trains on nuPlan and fine-tunes on NAVSIM — a different corpus and a supervised planning head, so the +1.0 is not portable as a number; (b) **navtest, not navhard** (F5); (c) no parameter counts are published, so "matched params" cannot be verified against them; (d) no limitations section — the paper concedes nothing, which by charter §7.2 is itself worth noting. |
| **EXPERIMENT** | **`E-AI-FLOW-1` (proposed AI17-2, v7-tiny ladder, 2 arms):** replace the O5 predictor's deterministic latent regression with **conditional flow matching over the latent future** at matched predictor params and matched step budget. **Committed in advance:** the arm succeeds only if the **h=1 action/scene ratio rises ≥ 3×** (0.004 → ≥ 0.012) **at equal-or-better o5 loss**; a ratio rise bought by a quality loss is **refuted**, and a quality gain with a flat ratio means the lever is real but is **not** fixing P-2 and must be re-attributed. ⭐ **Read numerator and denominator separately** per MM-1 — a ratio that rises only because scene spread fell is a denominator effect, not action sensitivity. **Controls:** constant-predictor (must read the no-information value), raw-pixel floor, and a **replicate arm** (same flags, same seed) so the effect is read against the rig's own noise floor per `H-ESTIM-SEED-1`. |

## 3 · What this changes (≤3)

1. ⭐⭐⭐ **AI13-2 may now cite an external same-backbone frozen > fine-tuned result (F1)** — stamped open-loop/T0-class, and with F2's correction that the frozen encoder is **300 M**, not 14 B.
2. ⭐⭐ **Re-aim the AI13-2 conversation at the READOUT.** F3 prices feature width at **+0.49 RFS** while our readout pools to 4 azimuth bins. `E-AI-WIDTH-1` is the cheap test and it outranks further encoder-side items.
3. ⭐⭐ **Record the combined position from today's two clusters: stochastic future modelling as a TRAINING objective, no future rollout at INFERENCE** (WA-JEPA F4 + Fast-WAM). Both halves are external, both are ablation-backed, and together they contradict the current v7 arrangement in *both* directions.

## 4 · Stopping condition (Rule Zero)

**(3b) — every remaining lever on this thread needs compute the Lab may not spend today.** `E-AI-WIDTH-1` and `E-AI-FLOW-1` are **v7-tiny ladder arms**, i.e. GPU, and the 4060 was occupied all session by another process (verified, not assumed). They are pre-registered here with both outcomes committed so a FlyWheel can run them without re-deriving the design. ⛔ **No Thor, no pod, no spend** — per the charter. The 0-GPU half of this thread *was* executed: the A3 debt is discharged, F5's split error is caught before it could enter a table, and F2's unit error is corrected before it could be quoted.

## 5 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-frozen-trunk-external-evidence/RESULT.md` |
| Search log | `repo:…/raw/search_log.md` |
| Primaries banked today | lib `2608.20974` (WA-JEPA), `2608.07409` (UniJEPA); `2601.03460` (FROST-Drive) and `2606.31232` (Delta-JEPA) were **already banked — cited-by updated** (V-1 re-find) |
