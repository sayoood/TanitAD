# Citations — with retrieval depth marked per row

**Date:** 2026-07-27 · companion to `LATENT_ACTION_RESEARCH.md`.

## Depth legend

| tag | meaning | admissible for? |
|---|---|---|
| **FULL-TEXT** | retrieved from the paper's own HTML rendering (arXiv `/html/…` or ar5iv) and the extracted facts include section-level detail, tables or verbatim quotes | quotation, load-bearing claims |
| **ABSTRACT** | the paper's own abstract, retrieved verbatim from arXiv `/abs/…` | quotation *of the abstract only*; load-bearing only for claims the abstract itself makes |
| **SUMMARY** | reached through a lossy hop — a PDF→summarisation pass, a search-result snippet, a blog or vendor page | ⚠️ **NOT admissible as a quotation. Nothing load-bearing may rest on it.** |
| **UNVERIFIED** | attempted and failed to retrieve | nothing |

⚠️ **This legend is not boilerplate.** In this session a PDF→summarisation hop **fabricated a verbatim
quotation and a section heading** for arXiv:2605.20223, and **three numbers** for arXiv:2602.03668.
See `LATENT_ACTION_RESEARCH.md` §8. Rows below marked SUMMARY are quarantined for that reason.

---

## Thread 1 — latent action models

| # | paper | id | depth | what it is cited for | load-bearing? |
|---|---|---|---|---|---|
| 1 | **Genie: Generative Interactive Environments** (Bruce et al., ICML 2024) | arXiv:2402.15391 | **FULL-TEXT** | LAM = VQ-VAE, \|A\|=8, embed 32, 300 M; encoder sees x₁:ₜ **and** xₜ₊₁; "apart from the VQ codebook, the entire LAM is discarded at inference time"; **200 expert samples** to match the oracle on CoinRun; 30,000 h / 6.8 M clips from 55 M; ΔₜPSNR 1.91, FVD 40.1; limitations (16-frame memory, ~1 FPS) | ✅ yes |
| 2 | **LAPO — Learning to Act without Actions** (Schmidt & Jiang, ICLR 2024 spotlight) | arXiv:2312.10812 | **FULL-TEXT** | joint IDM/FDM; **VQ is the anti-leakage bottleneck** — "the IDM learns to encode only the difference between o_{t+1} and o_t"; **"a decoder trained on less than 256 labeled transitions matches the performance of a policy trained from scratch for 4M steps"**; Procgen, expert recovery in 4M frames; **delayed-effect limitation** ("models the visible effects of an action, not the action itself"); stochasticity limitation | ✅ yes |
| 2b | *(same paper — codebook shape)* | " | ⚠️ **UNVERIFIED** | extraction returned two inconsistent shapes ("8 discrete latents, 16-d" vs "2 codebooks × 4 discrete latents, 64 embeddings of 16 dims") | ❌ no |
| 3 | **LAPA — Latent Action Pretraining from Videos** (Ye et al.) | arXiv:2410.11758 | **FULL-TEXT** | 3 stages; **8⁴** default latent space; pretraining 970 k Open-X + 220 k SSv2 + 60 k Bridge traj; **150 traj/task** finetuning; 60.38 % vs OpenVLA 54.16 %; **272 H100-h vs 21,500 A100-h**; **failure cases**: pick-and-place **50 % vs 66.67 %**, cross-env transfer **33.6 %/29.6 % vs 64.8 %/54.0 %** | ✅ yes |
| 4 | **UniVLA — Learning to Act Anywhere with Task-centric Latent Actions** | arXiv:2505.06111 | **FULL-TEXT** | latent actions "capture task-irrelevant dynamics, such as movements of non-ego agents or unpredictable camera shifts"; two-stage codebook + DINOv2 features; **\|C\|=16, N=4** vs OpenVLA 256⁷; decoder 10.8 M; **Table III: 88.7 % vs 56.5 %; LIBERO-Long 79.4 % vs 0.2 %** | ✅ yes — this is the strongest contradicting evidence in the report |
| 5 | **Why Latent Actions Fail, and How to Prevent It** (Lee, Cho, Zhao, Lee — 13 May 2026) | arXiv:2605.20223 | **ABSTRACT** (verbatim) + **FULL-TEXT** (partial) | abstract, verbatim: "minimizing the standard reconstruction objective produces latent actions that encode exogenous information from future observation"; full-text pass confirms failure modes = **future leakage** + **exogenous-noise sensitivity**, diagnostic = linear-probe **NMSE**, fixes = ℒX-exo + ℒξ-robust | ✅ for the abstract quote only |
| 5b | ⚠️ *(same paper — the FABRICATED rows)* | " | 🚫 **SUMMARY — REJECTED** | a PDF hop produced a "Mode 3: Metric/Scale Information Loss", a quoted "latent actions cannot inherently recover metric/scale information", and a "when NOT to use latent actions" section. **All three returned NOT FOUND at full-text/abstract depth.** | 🚫 **excluded from the report** |
| 6 | **MVP-LAM — Action-Centric Latent Action via Cross-Viewpoint Reconstruction** | arXiv:2602.03668 | **FULL-TEXT** | "Viewpoint changes introduce camera movements and perspective shifts, entangling visual transitions with the agent's action"; metric = **mutual information I(Z;A)** (KSG) ≈ **1.1 bits** vs UniVLA ≈ **0.5**; NMSE ~0.73 vs ~0.91; needs synchronized multi-view | ✅ for the problem statement; numbers are read off figures (approximate) |
| 6b | ⚠️ *(same paper — the FABRICATED rows)* | " | 🚫 **SUMMARY — REJECTED** | PDF hop gave "codebook 512 / latent dim 32", "~15–20 % alignment improvement", "~100–500 annotated sequences". **All NOT FOUND at full-text depth.** | 🚫 **excluded** |

## Thread 2 — V-JEPA 2 / V-JEPA 2-AC

| # | paper | id | depth | cited for | load-bearing? |
|---|---|---|---|---|---|
| 7 | **V-JEPA 2 / V-JEPA 2-AC** (Meta) | arXiv:2506.09985 | **FULL-TEXT** | **"we freeze the video encoder"**; predictor ~300 M / 24 L / 16 H / 1024; **block-causal** attention over action + end-effector state + patch features; action = **7-d delta EE**; **<62 h / 23k Droid trajectories**; **L1** teacher-forcing + **L1** rollout (T=2); CEM **800 samples / 10 refinements / 16 s per action**, image goals; pick-place cup **80 %** vs Octo 15 %; limitations: must "implicitly infer the action coordinate axis from the monocular RGB camera input", authors "manually tried different camera positions" | ✅ yes |
| 7b | *(same paper — §11.4 camera-pose sensitivity numbers; AC-head data-scaling ablation)* | " | ⚠️ **UNVERIFIED** | fetched twice, appendix not returned. Report says the paper states **no** data-scaling ablation exists. | ❌ no |

## Thread 3 — VPT

| # | paper | id | depth | cited for | load-bearing? |
|---|---|---|---|---|---|
| 8 | **VPT — Video PreTraining** (Baker et al., NeurIPS 2022) | arXiv:2206.11795 (ar5iv) | **FULL-TEXT** | **non-causality verbatim**: "the IDM can be non-causal, meaning its prediction for a_t can be a function of both past and future events"; 128-frame context, temporal kernel width 5 (t−2…t+2); **1,962 h** contractor data; **90.6 %** keypress, **R² 0.97** mouse; **"two orders of magnitude more data efficient than a BC model"**; 270,000 h → **~70,000 h** after an SVM filter trained on **8,800** labeled frames; **13 independent binary keys + separate 11-bin foveated camera-X/Y**; IDM gains plateau after 100 h | ✅ yes |

## Thread 4 — discrete vs continuous action/value heads

| # | paper | id | depth | cited for | load-bearing? |
|---|---|---|---|---|---|
| 9 | **Stop Regressing: Training Value Functions via Classification** (Farebrother et al., ICML 2024) | arXiv:2403.03950 | **FULL-TEXT** | HL-Gauss construction; **σ/ς = 0.75** default, ~6 neighbouring bins; bins ∈ {21,51,101,201}, **optimal σ/ς independent of bin count**; +~30 % Atari MoE IQM, +45 % vs C51 multi-game, +40 % Wordle, +70 % chess gap, +67 % robotics; **⚠️ Figure 15: advantage disappears under high environment stochasticity**; **⚠️ softmax+MSE gives no gain — the CE loss is essential**; **⚠️ Two-Hot underperforms MSE in online RL** | ✅ yes — including the three counter-findings |
| 10 | **DreamerV3** (Hafner et al.) | arXiv:2301.04104 | **FULL-TEXT** | symlog = `sign(x)·ln(\|x\|+1)`; **41 symexp-spaced bins**, two-hot; verbatim mechanism: the loss "only depends on the probabilities assigned to the bins but not on the continuous values associated with the bin locations, **decoupling the size of the gradients from the size of the targets**"; **no direct numerical symlog+twohot vs MSE ablation is given** | ✅ for the mechanism quote; ❌ for an effect size |
| 11 | **RT-1: Robotics Transformer** | arXiv:2212.06817 | ⚠️ **SUMMARY** (search snippet) | 11 action dimensions each into 256 uniform bins independently; 7 arm + 3 base + **1 separate mode variable** | ❌ **not load-bearing** — corroborates VPT's factorisation rule, does not carry it |

## Thread 5 — action-conditioned driving world models

| # | paper | id | depth | cited for | load-bearing? |
|---|---|---|---|---|---|
| 12 | **Vista** | arXiv:2405.17398 | **FULL-TEXT** | 4 action modalities (angle+speed, trajectory, command, goal point) as Fourier embeddings via cross-attention; **OpenDV-YouTube 2000 h unlabeled trained collaboratively with nuScenes "with the action conditions for OpenDV-YouTube set to zero"**; **no pseudo-labeling, no latent action**; FID **6.9**, FVD **89.4**; two-phase training with LoRA at high res | ✅ yes — this is the A6 null-arm proposal |
| 13 | **GenAD / OpenDV-2K** (CVPR 2024 Highlight) | arXiv:2403.09630 | ⚠️ **SUMMARY** | ~2000 h of driving video incl. YouTube; two stages (image-domain transfer → video-prediction pretraining); adaptable to action-conditioned prediction or planning | ❌ **not load-bearing**; HTML 404 ×2 and PDF over the fetch size limit |
| 14 | **DriveVA: Video Action Models are Zero-Shot Drivers** (2026 preprint) | arXiv:2604.04198 | **FULL-TEXT** | explicitly **not** a latent-action model — supervised, paired video+action; action = **3-D (x, y, yaw)** in metric trajectory space; flow matching on Wan2.2-TI2V-5B; NAVSIM **90.9 PDMS**; nuScenes zero-shot **0.84 m L2 / 0.06 %**; Bench2Drive **1.33 m / 1.79 %** | ✅ yes |
| 15 | **LFG — Learning to Drive is a Free Gift** (Strong et al., Feb 2026) | arXiv:2602.22091 | **ABSTRACT** (verbatim) + **PROJECT PAGE** (authors' own, `lfg-ai.github.io`) | label-free teacher-guided pretraining from **unposed YouTube driving video**; pseudo-labels = **point maps, camera poses, semantic segmentation, motion masks**, "without poses, labels, or LiDAR"; teachers = **π3** (depth/point maps) + **SegFormer** (semantics); NAVSIM single-camera **PDMS 85.2 @ 100 % labels vs 81.4 @ 10 %**, NC 98.2 / DAC 93.7 / TTC 94.4 | ✅ yes for the architecture pattern and the 10 %-label result (**this is A3b, and it is what proves the unlabeled/labeled asymmetry is exploitable ON DRIVING**) |
| 15b | *(same paper — hours of video; metric-scale mechanism)* | " | ⚠️ **UNVERIFIED** | project page: scale "mechanism not disclosed"; hours not stated; PDF exceeded the fetch size limit | ❌ **no** — A3b's experiment measures this rather than assuming it |
| 15c | **π3** (feedforward point-map/pose backbone, used as LFG's geometry teacher) | — | ⚠️ **UNVERIFIED** | named on LFG's project page only; not independently retrieved. Whether its pose output is metric or scale-free **on our corpus** is the first quantity A3b measures. | ❌ no |

## Thread 6 — geometric metric-scale grounding (added; not in the brief but decisive for the crux)

| # | paper | id | depth | cited for | load-bearing? |
|---|---|---|---|---|---|
| 16 | **StableCamH — "Camera Height Doesn't Change: Unsupervised Monocular Scale-Aware Road-Scene Depth Estimation"** | arXiv:2312.04530 | **FULL-TEXT** | camera height from road-plane normals + reprojected 3D points; the invariance "the camera height does not change in the sequence" formulated as an optimisation; **camera height is NOT measured a priori — it is jointly optimised**, with a learned vehicle-size prior for absolute scale; KITTI **AbsRel 0.108 / RMSE 4.740**, beating **VADepth 0.120 which uses GT camera height** and Monodepth2 0.968; **⚠️ no camera-height-error ablation, no hill/pitch evaluation**, needs road+instance segmentation | ✅ yes — this is A3 |

## Thread 7 — flow matching / diffusion action heads

| # | source | id | depth | cited for | load-bearing? |
|---|---|---|---|---|---|
| 17 | **π0 / π0-FAST / FAST tokenization** (Physical Intelligence) | arXiv:2501.09747 + HF blog + pi.website | ⚠️ **SUMMARY** | FAST = DCT + BPE discrete action tokens, ~10× compression; **5× faster training / 3× fewer steps** vs the diffusion variant; but **~750 ms** inference per chunk vs π0 flow matching's ~100 ms | ❌ **not load-bearing** — A7 recommends **no change**, so nothing rests on it |

---

## Our own measurements cited in the report (`MEASURED`, primary artifacts)

| claim | artifact path |
|---|---|
| `long_accel` R² −0.240 pooled / −0.298 pai / −0.254 cm; label r = 0.434 with dv/dt ⇒ ceiling **R² 0.188**; comma frozen-latent probe −0.095 | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-idm-v2/IDM_V2_RESULTS.md` §3.3, §5 |
| `yaw_rate` PhysicalAI R² **0.9035**; 9 impossible windows move pooled 0.105 → 0.497 | *ibid.* §3.2, §0 |
| `speed` R² 0.8651; **oracle clip-level ceiling 0.942**; shrinkage **gain 0.830**; per-corpus constant `cam_h` calibration is a **NO-OP** | *ibid.* §3.1, §5, §5.2 |
| `steer` R² **0.742** (A0) | *ibid.* §3.4 |
| three inconsistent `cam_h` values (1.5 / 1.43 / 1.22) | *ibid.* §5 item 5 |
| the head we would modify: `self.scalar_head = nn.Linear(d_model, n_scalars)` | `…/2026-07-26-idm-v2/idm2_v2.py:66` |
| `SCALAR_NAMES = ("speed", "yaw_rate", "steer", "long_accel")` | `stack/scripts/idm_head.py:37` |
| zeroing `v0` degrades the imagined decode **×93.73 (v1) / ×39.43 (v4)**, perceived decode bit-exactly unchanged; both action channels deleted with `v0` kept costs **×1.32 / ×1.07** ⇒ **71× / 37×** separation | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-v4-instrument/V4_INSTRUMENT.md` (headline + line 243, 254) |
| encoded latents available for the A1/A3/A4 experiments (104 episodes, regenerable in 102 s) | `tanitad-eval:/root/idm2/lat/` via `…/2026-07-26-idm-v2/idm2_encode.py` — **NOT touched by this agent** |
| two camera rigs (cy≈543 rig A / cy≈755 rig B) | program record (`MEMORY`) — ⚠️ **INHERITED**, not re-verified here |

### Sibling `idm-v3` measurements that CHANGED this report

Read from the **git index** (staged, not committed) at `TanitAD Research Hub/Architecture &
Inference/Implementation/incoming/2026-07-27-idm-v3/`. **No file there was modified and no pod was
touched.** Depth: **PRIMARY (a sibling agent's own pre-registration + its cited script output)** —
i.e. `MEASURED (theirs)`, one hop from the raw artifact, **not re-run by me**.

| their claim | where | what it did to this report |
|---|---|---|
| ground-plane scaling `v̂·h/h̄` makes speed **significantly WORSE** (MAE 2.960 → 3.236, CI [+0.051, +0.551]); **shuffled heights as good as real**; oracle-`k` vs height **r = −0.466**, partial **−0.352**, opposite sign | `PRE_REGISTRATION_IDMV3.md` §4 E4, from `idm3_geomtest.py`, n = 40 held-out PhysicalAI clips | 🔴 **withdrew my A3** |
| oracle per-clip scale headroom: MAE 2.960 → **1.607**, CI [−1.869, −0.881] | *ibid.* | kept §2.3's decomposition alive; the scalar is real, its *source* is open |
| camera height is **per-clip 1.2450–1.6066 m**, 37 distinct values in 40 clips; all three circulating constants wrong | *ibid.* §2 (from PhysicalAI's own gated calibration; 40/40 join verified) | closed `IDM_V2_RESULTS.md` §5 item 5; corroborated StableCamH's "never assume the height" principle while refuting its application |
| `v = (f·h)·Φ` vs `ω = (du/dt)/f`; `f_eff` already canonicalised ≈266 px ⇒ "the yaw channel is already geometry-matched and the speed channel is not" | *ibid.* §3 | **independent physics derivation of §2.1's rotation-vs-translation split** — raised §2.2 from HYPOTHESIS toward CONFIRMED |
| **E5 = `long_accel` as a 21-bin softmax-expectation decode**, bar R² > 0, bins over the longitudinal axis alone | *ibid.* §4 E5 | ⭐ **turned my A1 from a proposal into 3 upgrades on an arm already in flight**, and made Farebrother's "Two-Hot underperforms MSE" the highest-value row in this report |
