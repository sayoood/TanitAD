# CITATION TABLE — imagination/perception manifold mismatch

**Companion to** `MANIFOLD_MISMATCH_RESEARCH.md`. **Date:** 2026-07-27.

## How to read the verification column — this is the honest part

| tag | meaning |
|---|---|
| ⭐ **PRIMARY-FETCHED** | I fetched the paper page or full text **in this session** and the quoted text came from that fetch. Quotable. |
| **LISTING-VERIFIED** | The arXiv ID / venue was confirmed against ≥ 2 independent listings (arXiv index, OpenReview, venue page, author site) in this session, but I did **not** extract the quoted mechanism from the full text. The *existence and identity* are solid; a mechanism claim attributed to it is `INHERITED`. |
| ⚠️ **SEARCH-EXTRACT** | The quoted sentence came from a **search-engine extract of the source page**, not from my own fetch of the paper. Treat the wording as approximate and re-verify before it decides anything. |
| ⛔ **ID-UNVERIFIED** | Well-known work cited from established knowledge; the identifier was **not** re-fetched in this session. **Do not quote a number from it.** |

⚠️ **No claim in the main report that decides a GPU-day rests on anything below PRIMARY-FETCHED.** The
decision-grade material in that report is **`MEASURED` (ours)**, not `PUBLISHED`.

---

## A. The posterior/prior decode crux (§4 of the main report)

| # | work | identifier | verification | what it is cited FOR | DEMONSTRATED or ASSERTED |
|---|---|---|---|---|---|
| **1** | Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, Davidson — **"Learning Latent Dynamics for Planning from Pixels" (PlaNet)**, ICML 2019 | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) · [PMLR v97](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) | ⭐ **PRIMARY-FETCHED** (abstract + title + authors + venue confirmed; *"a multi-step variational inference objective that we name latent overshooting"* quoted from the fetched abstract) | **latent overshooting** — multi-step priors trained toward filtered posteriors, with posterior gradients stopped for d > 1. Family (A) of §4.1 | **DEMONSTRATED** as an ablated design choice improving long-horizon consistency. ⚠️ The detailed mechanism wording (*"compute KL divergence between multi-step predicted latent distributions and the corresponding filtered posteriors"*) is ⚠️ **SEARCH-EXTRACT** from the paper page, not my own full-text fetch |
| **2** | Hafner, Lillicrap, Norouzi, Ba — **"Mastering Atari with Discrete World Models" (DreamerV2)**, ICLR 2021 | [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.02193) | ⭐ **PRIMARY-FETCHED** (ar5iv full text) | ⭐ the three load-bearing quotes: **KL balancing** — *"we minimize the KL loss faster with respect to the prior than the representations by using different learning rates, α=0.8 for the prior and 1−α for the approximate posterior"*, which *"encourages learning an accurate prior over increasing posterior entropy, so that the prior better approximates the aggregate posterior"*; **decoder on posteriors** — *"From the posterior model state, we reconstruct the current image x̂ₜ and predict the reward rₜ and discount factor γₜ"*; **imagination seeded from posteriors** — *"The trajectories start from posterior states computed during model training and predict forward by sampling actions from the actor network"* | **DEMONSTRATED** (KL balancing is an ablated hyperparameter); the posterior-seeding and posterior-decoding are **architectural facts**, quoted verbatim |
| **3** | Hafner, Pasukonis, Ba, Lillicrap — **"Mastering Diverse Domains through World Models" (DreamerV3)**, 2023 | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) | ⭐ **PRIMARY-FETCHED** (title/authors/year confirmed) | identity only — DreamerV3 inherits the DreamerV2 RSSM prior/posterior structure. **The inheritance claim itself is `INHERITED`, not re-verified against V3's text** | — |
| **4** | Hansen, Su, Wang — **"TD-MPC2: Scalable, Robust World Models for Continuous Control"**, ICLR 2024 | [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.16828) | ⭐ **PRIMARY-FETCHED** (ar5iv full text; equation number confirmed) | ⭐⭐ **the single most important citation in the report.** **Eq. 3**: `Σ λᵗ( ‖z′_t − sg(h(s′_t))‖²₂ + CE(r̂_t,r_t) + CE(q̂_t,q_t) )` — the predicted latent is regressed **onto a stop-gradient of the encoder output**; heads are applied to **rolled** latents `z_{t+1} = d(z_t,a_t), z_0 = h(s_0)`; **decoder-free**; default horizon **H = 3** (Table 8) | **DEMONSTRATED** — a named, load-bearing loss term with an equation number and a hyperparameter table. **This is our configuration WITH the term we lack** |
| **5** | Micheli, Alonso, Fleuret — **"Transformers are Sample-Efficient World Models" (IRIS)**, ICLR 2023 (notable top-5 %) | [arXiv:2209.00588](https://arxiv.org/abs/2209.00588) · [OpenReview](https://openreview.net/pdf?id=WIimAcYcZ5U) · [code](https://github.com/eloialonso/iris) | **LISTING-VERIFIED** (3 independent listings) | the **shared discrete codebook** argument — an imagined token is by construction an element of the same codebook as a real token, so the decoder cannot be handed an off-manifold input. Family (B) of §4.1 | **Architectural.** ⚠️ The paper does **not** measure an imagined-vs-real decode gap; the safety is structural. Do not cite it as evidence of a measured gap |
| **6** | Alonso et al. — **"Efficient World Models with Context-Aware Tokenization" (Δ-IRIS)** | [arXiv:2406.19320](https://arxiv.org/abs/2406.19320) | **LISTING-VERIFIED** | tokenises **stochastic deltas between timesteps** — relevant because our defect lives in a pair *difference* | **Architectural**; `INHERITED` |
| **7** | Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret — **"Diffusion for World Modeling: Visual Details Matter in Atari" (DIAMOND)**, NeurIPS 2024 spotlight | [arXiv:2405.12399](https://arxiv.org/abs/2405.12399) | ⭐ **PRIMARY-FETCHED** (title, authors, venue confirmed) | operates in **pixel space** — conditioning is always a valid image, so there is no separate latent manifold to fall off. ⚠️ **I could NOT extract the conditioning details from the abstract page; that row of §4 is `INHERITED`** | the paper's own measured claim is that visual detail improves agent performance |
| **8** | **DINO-WM — "DINO World Models"** | [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) | ⭐ **PRIMARY-FETCHED** | ⭐ **decoder-free for control** — *"model[s] visual dynamics without reconstructing the visual world"*, predicting future DINOv2 patch features; planning entirely in feature space. Family (C)/(D) of §4.1 | **DEMONSTRATED** (verified by fetch: the method has no control decoder) |
| **9** | **V-JEPA 2 / V-JEPA 2-AC** | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | **LISTING-VERIFIED** for identity; ⛔ **UNVERIFIED for mechanism** | cited only as *"a latent action-conditioned world model … planning with image goals"* (quoted from the fetched abstract). ⛔ **I could NOT reach the energy/cost-function text.** The claim that its planning cost is a predicted-vs-encoded feature distance is **`HYPOTHESIS`** and is labelled so in §4 | — |

## B. Exposure bias / train-test gap — the pathology's names (§3)

| # | work | identifier | verification | cited FOR | DEMONSTRATED or ASSERTED |
|---|---|---|---|---|---|
| **10** | Bengio, Vinyals, Jaitly, Shazeer — **"Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"**, NeurIPS 2015 | [arXiv:1506.03099](https://arxiv.org/abs/1506.03099) | ⭐ **PRIMARY-FETCHED** | the canonical statement of the gap: *"At inference, the unknown previous token is then replaced by a token generated by the model itself. This discrepancy between training and inference can yield errors that can accumulate quickly along the generated sequence"*; the fix is a curriculum *"from a fully guided scheme using the true previous token, towards a less guided scheme which mostly uses the generated token instead"* | **DEMONSTRATED** in the original sequence setting. ⛔ **Points the OPPOSITE way to our defect** — see §5 row C4 |
| **11** | Huang, Li, He, Zhou, Shechtman — **"Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"**, NeurIPS 2025 | [arXiv:2506.08009](https://arxiv.org/abs/2506.08009) · [OpenReview](https://openreview.net/forum?id=mSiN7i0BYH) · [project](https://self-forcing.github.io/) | ⭐ **PRIMARY-FETCHED** | the modern restatement: *"exposure bias, where models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs during inference"*; the fix conditions *"each frame's generation on previously self-generated outputs by performing autoregressive rollout … during training"* | ⚠️ **ASSERTED quality parity**, not a numeric table — the fetched abstract reports *"matching or even surpassing"* quality with sub-second latency and gives **no numeric metric**. ⛔ Same wrong direction as #10 |
| **12** | Lamb, Goyal, Zhang, Zhang, Courville, Bengio — **"Professor Forcing: A New Algorithm for Training Recurrent Networks"**, NeurIPS 2016 | arXiv:1610.09038 | ⛔ **ID-UNVERIFIED** (not re-fetched this session) | named only, in the §3.1 vocabulary table. **Nothing in the report depends on it** | — |
| **13** | Ross, Gordon, Bagnell — **DAgger**, AISTATS 2011 | arXiv:1011.0686 | ⛔ **ID-UNVERIFIED** (not re-fetched this session) | named only, as the control-theory framing (covariate shift). **Nothing in the report depends on it** | — |

## C. The information ceiling — can a monocular latent pair carry metric ego-motion? (§5.1, §6/X1)

| # | work | identifier | verification | cited FOR | DEMONSTRATED or ASSERTED |
|---|---|---|---|---|---|
| **14** | El Banani et al. — **"Probing the 3D Awareness of Visual Foundation Models"**, CVPR 2024 | [CVPR 2024 supplemental](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Banani_Probing_the_3D_CVPR_2024_supplemental.pdf) | **LISTING-VERIFIED** (CVPR open-access page) | ⭐ **the closest published analogue of our X1**: linear probes on frozen foundation-model features for tasks **including relative camera pose**; DINOv2 shows *"reasonable object-level 3D awareness"* and outperforms other encoders, **but** *"strong semantic models like DINOv2 may achieve multiview correspondence through semantic matching"* — i.e. a probe can look 3D-aware for non-geometric reasons | ⚠️ **SEARCH-EXTRACT** for the quoted wording. **DEMONSTRATED** that the probe protocol works; ⚠️ **it does NOT report metric-scale ego-motion in metres**, which is what we need — so it bounds the method, not the answer |
| **15** | Marsal, Chapoutot, Xu, Filliat — **"A Simple yet Effective Test-Time Adaptation for Zero-Shot Monocular Metric Depth Estimation"** | [arXiv:2412.14103](https://arxiv.org/abs/2412.14103) | ⭐ **PRIMARY-FETCHED** (title/authors confirmed) | the fixed-calibration dependence of learned metric depth: the fetched abstract states training datasets *"must contain images captured by the camera that will be used at test time"* and that fine-tuning to a new camera *"can be costly and time-consuming"*. ⚠️ **The stronger wording used in an earlier draft** (*"using them with another camera leads to ill-scaled predictions"*) is ⚠️ **SEARCH-EXTRACT** and was NOT found in my fetch — the verified, weaker form is what §5.1 should stand on | **DEMONSTRATED** — the paper exists to fix this problem, so the problem is its premise |
| **16** | Zhan, Garg, Weerasekera, Li, Agarwal, Reid — **"Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction"**, CVPR 2018 | [CVPR 2018 open access](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhan_Unsupervised_Learning_of_CVPR_2018_paper.pdf) | **LISTING-VERIFIED** | monocular scale ambiguity in learned VO, and stereo/geometric priors as the standard remedy | ⚠️ **SEARCH-EXTRACT** for wording |
| **17** | monocular VO scale-prior literature (camera height / ground-plane / known-object-size) — e.g. *"Visual Attention-based Self-supervised Absolute Depth Estimation using Geometric Priors in Autonomous Driving"* | [arXiv:2205.08780](https://arxiv.org/pdf/2205.08780) | **LISTING-VERIFIED** | the standard statement that absolute scale needs an external prior — *"a single RGB image can only constrain depth up to an unknown affine transformation"*; scale recovered from *"the camera height or size of known objects"* | ⚠️ **SEARCH-EXTRACT**. This is a **textbook-level** fact in monocular VO, cited here for form rather than as a contested claim |

## D. ⛔ What I searched for and did NOT find

Stated because a clean negative is a result, and because `CLAUDE.md` rule 2 requires the search strategy to
be visible rather than the conclusion asserted.

| I searched for | search strategies used | outcome |
|---|---|---|
| **a paper naming or measuring our DIRECTION** (a head trained on imagined latents that fails on encoded ones) | "inverted/reverse exposure bias"; "decoder trained on imagined latents fails on encoded"; "readout trained on rollouts fails on observations"; "model-generated-data over-adaptation"; "prior latent decode worse than posterior" | ⛔ **none found.** `UNVERIFIED` as an exhaustive negative — but §4 gives a *structural* reason it should be rare (almost nobody builds such a head) |
| **a quantitative posterior-vs-prior DECODE gap** in any world-model paper | "prior vs posterior reconstruction gap"; "open-loop imagination fidelity RSSM"; "teacher-forced vs free-running evaluation" | ⛔ **no numeric table found.** The prior/posterior gap is universally treated as a thing to *prevent*, never as a thing to *measure* |
| **a frozen-trunk repair** (re-fit a small head to close such a gap) | "frozen encoder head refit distribution gap"; "post-hoc decoder calibration world model" | ⛔ **no precedent found** — this is falsifier **F1**, and it fired only partly: the route is **unattempted**, not refuted |
| **driving-specific decode gaps** (GAIA, Vista, GenAD, DriveDreamer, occupancy/Gaussian lines) | delegated to a parallel search — see §8 of the main report | see §8 |
| **horizon-mismatch literature** (readout trained at k, applied at K ≫ k; direct vs iterated multi-step) | delegated to a parallel search — see §7 of the main report | see §7 |

## E. Our own primary sources (`MEASURED`) — the ones the report's decisions actually rest on

| # | artifact | what it establishes |
|---|---|---|
| **P1** | `taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl` (620 rows, final step 29999) | ⭐⭐ `g_op_mid_de_m` (REAL-pair decode) vs `g_op_fwd_ade_m` (IMAGINED decode) on the **same training batch** — the F4 result, §2.2 |
| **P2** | `taniteval/results/trainlogs/nospeed-phase0_train_log.jsonl` + `nospeed-phase0_config.json` vs `v1-speedjerk_config.json` | ⭐⭐ the attributing ablation, §2.3, and the field-by-field config diff proving only 3 fields differ |
| **P3** | `stack/tanitad/models/metric_dynamics.py` (`grounding_losses`, `rollout_transitions`, `rollout_decode`, `StepDisplacementReadout`, `MetricInverseDynamics`) | what the two grounding terms are and which latent source each consumes, §2.1 |
| **P4** | `stack/tanitad/train/train_worldmodel.py::_rollout_loss` | that our latent tie is **TD-MPC-shaped** (predicted latent regressed onto encoded latent, K = 4, weight 0.5), §4.1 |
| **P5** | `stack/tanitad/train/flagship_losses.py::horizon_plan` + `stack/scripts/train_flagship4b.py:513, 530-536` | the `op/tac/str` horizon plan, and that the grounding log is a **training-batch** quantity |
| **P6** | `…/Implementation/incoming/2026-07-26-blind-imagination/` (`BLIND_IMAGINATION.md`, `artifacts/horizon_curve.json`, `perwindow/bi_perwindow_compact.pt`) | the 9.4× finding itself; the CV floor; the ego-speed distribution used for the mean-speed baseline |
| **P7** | `…/Implementation/incoming/2026-07-26-tblind-ladder/`, `…/2026-07-26-tblind-rung1/` | the horizon-swap lever and its v4 canary movement (`INHERITED` here, not re-verified) |

---

## F. ⚠️ DELEGATED SEARCH — §7 (horizon mismatch) and §8 (driving) of the main report

**Provenance, declared not laundered.** §7 and §8 were produced by **two parallel delegated search agents**
run by this stream, each briefed with the same evidence discipline (verify every identifier by fetching;
separate DEMONSTRATED from ASSERTED; mark UNVERIFIED rather than guess). **Per `CLAUDE.md` operating rule 1
this material is `INHERITED` and MUST NOT decide a GPU-day.** It is used in the report to *frame* and
*rank*, never to justify. Their verification tags are preserved below verbatim.

**What I re-verified myself:** ⭐ **Valdi** ([arXiv:2607.00917](https://arxiv.org/abs/2607.00917)) — I
fetched it and independently confirmed **title, authors (Lindenberg & Chitta), submission date 2026-07-01,
and that the experiments are CarRacing** (*"In preliminary experiments on the CarRacing environment…"*),
which matches the delegated report exactly. ⛔ Its **App. G.2 quotes are `INHERITED`** — I could not reach
the full text. ⛔ **I²-World's Table 1** — I fetched the paper page and **could NOT reach the table**; the
numbers in §8.3 are `INHERITED` from the delegated agent's three-probe cross-check and are labelled so.

### F.1 Horizon mismatch (§7)

| work | URL | delegated verification tag |
|---|---|---|
| **APEBench** — train-unroll × test-rollout sweep; *"11% increased error at the first step"* | https://arxiv.org/html/2411.00180v1 | **DEMONSTRATED, verbatim, verified** |
| **GraphCast** (Science 2023), suppl. Fig. 30 | https://arxiv.org/pdf/2212.12794 | **DEMONSTRATED** (sentences confirmed 3 ways incl. an independent restatement in FuXi). ⚠️ **numeric values in Fig. 30 UNREAD** — PDF over the fetch size limit, supplement absent from ar5iv |
| **FuXi** — 3-model cascade; Z500 9.25 → 10.5 d, T2M 10 → 14.5 d | https://ar5iv.labs.arxiv.org/html/2306.12873 | **DEMONSTRATED** |
| **Pangu-Weather** (Nature 619:533) — four lead-time models | https://arxiv.org/pdf/2211.02556 | **DEMONSTRATED** |
| **Benechehab et al.**, multi-timestep MBRL models | https://arxiv.org/abs/2310.05672 · https://arxiv.org/abs/2402.03146 | **DEMONSTRATED** (verbatim quote verified); ⚠️ **Fig. 4 R² values read off a plot — approximate** |
| ⚠️ **PlaNet Fig. 7 — the counter-evidence** | https://ar5iv.labs.arxiv.org/html/1811.04551 | **DEMONSTRATED (negative)** — latent overshooting *"slightly reduces performance of our RSSM"* |
| **Marcellino, Stock & Watson (2006)** — iterated beats direct on 170 macro series | https://econpapers.repec.org/RePEc:eee:econom:v:135:y:2006:i:1-2:p:499-526 | **DEMONSTRATED, abstract-level only.** ⛔ per-horizon MSE tables unreachable (ScienceDirect 403) |
| **Chevillon (2007)** survey, DOI 10.1111/j.1467-6419.2007.00518.x | https://www.ofce.sciences-po.fr/pdf/dtravail/WP2005-10.pdf | **ASSERTED-with-survey-support** |
| **Ing (2003)** *Econometric Theory* 19(2):254–279 · **Ing (2004)** *Ann. Statist.* 32(2):693–722 | https://doi.org/10.1017/S0266466603192031 · https://arxiv.org/abs/math/0406433 | **DEMONSTRATED (theory)** |
| **Bhansali (1997)**, *Statistica Sinica* 7:425–449 | https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n210.pdf | ⛔ **UNVERIFIED at source** — PDF unrenderable. **The strongest pro-DMS theory claim encountered; verify before it decides anything** |
| **Taieb et al. (2012)**, NN5 benchmark | https://arxiv.org/abs/1108.3259 | **DEMONSTRATED** |
| **Brandstetter et al.**, pushforward trick | https://arxiv.org/abs/2202.03376 | **DEMONSTRATED** |
| **Diffusion Forcing** | https://arxiv.org/abs/2407.01392 | **DEMONSTRATED** |
| **Foster et al.**, "Is Behavior Cloning All You Need?" — the H² compounding bound is **not** tight | https://arxiv.org/abs/2407.15007 | **DEMONSTRATED (theory)** |
| **Lambert et al.**, objective mismatch | https://arxiv.org/abs/2002.04523 | **DEMONSTRATED** |
| **Myers, Ji & Eysenbach**, "Horizon Generalization" | https://arxiv.org/abs/2501.02709 | **DEMONSTRATED** |
| **Wen et al.**, MQ-RNN (horizon-specific decoders, forking-sequences) | https://arxiv.org/abs/1711.11053 | **LISTING-VERIFIED** |
| Latent-WM readout head, Δ≤50 → 35.0 % vs full-horizon → 97.5 % | https://arxiv.org/html/2605.22164v1 | ⚠️ **DEMONSTRATED but a recent unrefereed preprint from a small group — suggestive, not decision-grade** |
| Unrolled training in neural physics simulators, "38% on average" | https://arxiv.org/pdf/2402.12971 | ⛔ **UNVERIFIED — snippet-only, PDF unparsed** |
| ⛔ **FALSE LEAD, logged** — "Closing the Train-Test Gap in World Models for Gradient-Based Planning" | https://arxiv.org/abs/2512.09929 | ⛔ **DO NOT CITE for horizon mismatch.** Its gap is *next-state-prediction objective vs test-time action estimation*. A first fetch appeared to confirm the horizon reading — **that was the summarising model echoing a leading question.** Root-cause class: *leading-prompt confirmation on a summarising tool* |

### F.2 Driving world models (§8)

| work | URL | note |
|---|---|---|
| ⭐ **Valdi** — the only paper that trains decoders on its own rollout latents | https://arxiv.org/abs/2607.00917 | identity **PRIMARY-FETCHED by me**; App. G.2 quotes `INHERITED`. **ASSERTED, never measured** |
| ⭐ **I²-World** — recon-vs-forecast table (2.0–3.9×) | https://arxiv.org/abs/2507.09144 | ⛔ **table `INHERITED`** (I fetched, could not reach it). Cross-checked 3 ways by the delegated agent. ⚠️ **confounded by a time gap — not comparable to our matched-timestep 9.4×** |
| ⭐ **NeuRAD** Tab. 3 — off-distribution render FID with a no-shift baseline | https://arxiv.org/abs/2311.15260 | **DEMONSTRATED, table.** Closest analogue in kind; **2–4×**. ⚠️ FID vs ADE not commensurable |
| **UniSim** — same protocol, *"the gap is more significant in extrapolation settings"* | https://arxiv.org/abs/2308.01898 | **DEMONSTRATED** |
| ⭐ **MILE** Fig. 4 — "driving in imagination" degradation | https://arxiv.org/abs/2210.07729 | ⚠️ **values read off a plot.** Opposite polarity (decoders on the posterior) |
| ⭐ **LAW** — `L_latent = Σ‖p−v‖₂`; **trajectory head reads the OBSERVED latent** | https://arxiv.org/abs/2406.08481 | **DEMONSTRATED** (L2 0.26/0.57/1.01, avg 0.61) |
| ⭐ **Epona** — names the pathology verbatim; TrajDiT reads the observation-derived latent | https://arxiv.org/abs/2506.24113 | **DEMONSTRATED** (prose quote) |
| **HorizonDrive** — names "exposure bias"; fixes the **dynamics model**, not the decoder | https://arxiv.org/abs/2605.11596 | **ASSERTED**, no GT-vs-generated ablation table |
| **GAIA-1** / **GAIA-2** / **Vista** | https://arxiv.org/abs/2309.17080 · https://arxiv.org/abs/2503.20523 · https://arxiv.org/abs/2405.17398 | decoder on **encoded real** frames; ⛔ **no gap measured.** GAIA-1 has **no quantitative table at all**; GAIA-2 has **no reconstruction/oracle row**; Vista's VAE frozen-vs-finetuned **not stated** |
| **GenAD** | https://arxiv.org/abs/2403.09630 | ⛔ **UNVERIFIED — all four questions.** ar5iv conversion error; `/html/` 404 at v1/v2/none; PDF over size limit; CVPR OA 403; Semantic Scholar 404. Abstract only |
| **DriveDreamer** / **-2** / **Drive-WM** / **MUVO** / **Copilot4D** / **iVideoGPT** | https://arxiv.org/abs/2309.09777 · https://arxiv.org/abs/2403.06845 · https://arxiv.org/abs/2311.17918 · https://arxiv.org/abs/2311.11762 · https://arxiv.org/abs/2311.01017 · https://arxiv.org/abs/2405.15223 | ⛔ **no gap measured.** ⚠️ MUVO **does not specify** which latent its decoders see in training — a real source ambiguity. ⚠️ A DriveDreamer "0.29 m" action-L2 figure surfaced on a single probe — **do not quote.** ⚠️ **iVideoGPT is NOT a driving model** (robotic manipulation) |
| **DrivingGaussian** / **S-NeRF++** / **MARS** | https://arxiv.org/abs/2312.07920 · https://arxiv.org/abs/2402.02112 · https://arxiv.org/abs/2307.15058 | ⛔ **all test on the RECORDED trajectory** — off-distribution decode never measured. S-NeRF++ reports **no** PSNR/SSIM/LPIPS/FID at all |
| **OccWorld** / **OccSora** / **DOME** / **OccLLaMA** | https://arxiv.org/abs/2311.16038 · https://arxiv.org/abs/2405.20337 · https://arxiv.org/abs/2410.10429 · https://arxiv.org/abs/2409.03272 | the occupancy family behind I²-World's table. **None attributes the gap, mentions exposure bias, or ablates it** — an open lane |
| ⚠️ *"Reconstruction or Semantics?"* — the only Enc-vs-WM **readout-head** table found | https://arxiv.org/abs/2605.06388 | ⛔ **robotics (BridgeV2), not driving**, and **two transcription probes disagreed on the Enc/WM column alignment.** Prose direction only; `PROVISIONAL` |
| Leads verified in identity but **not** found to contain a decode gap | https://arxiv.org/abs/2606.01935 (Unified Driving Tokens — **the best unexplored lead**, pairs a *"lightweight planning readout"* with a tokenizer) · https://arxiv.org/abs/2607.13410 (discusses a *"prior–posterior bottleneck"* and *"bounded imagination drift"*, **reports no comparison**) · https://arxiv.org/abs/2606.32026 (AdaJEPA — loss curves only) · https://arxiv.org/abs/2605.18137 · https://arxiv.org/abs/2606.18208 | recorded so they are not re-searched |
