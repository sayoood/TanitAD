# Architecture & Inference — 2026-07-14 — D1–D3 gate runner + JEPA-WM / decoding / MoE / quant deltas

> Note on dating: this is the first substantive Architecture & Inference weekly run (STATE seeded
> `never` at kickoff). Dated 2026-07-14 (Wednesday of W29) to stay chronological after the Monday
> Tools&DevEnv (07-13) and Tuesday DataEng (07-07) notes.

**Run:** weekly Architecture & Inference agent (Wednesday). **Loop:** iteration 1 of 3.
**Budget used:** 6 web searches + 1 fetch, ≈1.5 h wall-clock — well under caps (25 searches / 2 h / 3 iters).
**Consumed:** Monday's Tools&DevEnv note (2026-07-13, front-cam RGB sim arm), Tuesday's DataEng note
(2026-07-07, comma2k19 validation + A8), DECISIONS D-004/D-008/D-009/D-010/D-011, Phase 0 Plan §2/§4,
INITIAL_RESEARCH_SYNTHESIS baseline. Every claim carries a source link or a repo-path reference (G-A).

**Mid-session repo advance (P8 note — my standing "re-check git" memory earned its keep).** During this
run Sayed pushed `a731481` (JEPA generalization theory + HiT-JEPA analyzed; **D-012**, **D-013**) and
`121177a` (DATA_STRATEGY v1.0). These changed my own agent file, `_common-protocol.md`, the ledger and
DECISIONS *after* I had read them. Consumed on re-read: (i) **D-013** upgrades the SEARCH step (systematic
arXiv sweeps + citation-graph walks + a `Ressources/` inbox rule) and gives Architecture a standing
**theory-watch duty** (Balestriero/LeCun, Klindt, HaoChen spectral SSL, PKU Yisen Wang group); (ii) my
backlog gained a new **top item #0 `p0-spectral-sizing`** (see §1b) — so this run delivers *both* #0 and
the already-committed #1; (iii) Sayed's note `Research/2026-07-06-jepa-generalization-theory-and-hit-jepa.md`
(arXiv 2606.27014 / 2507.00028) is now the anchor for the theory watch — I build on it rather than
re-deriving it. The two `Ressources/` PDFs flagged at session start are already analyzed there → inbox clear.

---

## 1. Implementation increment — D1–D3 gate runner (backlog #1)

Delivered as an intake package (D-011, hub/MVP separation — **not** written into `stack/`):
`Architecture & Inference/Implementation/incoming/2026-07-14-gate-runner-d1-d3/`
(`tanitad_gates.py` + `tests/test_gates.py` + `INTAKE.md`). Proposed target: `stack/tanitad/eval/gates.py`.

**What it is.** The falsification harness for the three Phase-0 *decode* gates (Phase 0 Plan §4):

| Gate | Claim | Threshold (encoded) | Ablation |
|---|---|---|---|
| D1 | encoder state decodable | frozen-probe ADE@1s < 0.5 m (BEV) / < 1.0 m (camera); I2, I3 pass | vs global-pool (A7) |
| D2 | imagination usable for selection | direction acc > 0.7; imag-rel < 0.8; I1 sanity first | vs persistence |
| D3 | trajectory decode from imagination | imagined-ADE@2s ≤ 1.5× oracle-decode ADE@2s | probe_real vs probe_imag (A3) |

**The load-bearing idea — instrument doctrine made mechanical (D-004 → G-AI1).** Each gate assembles its
I1–I4 rows **first** (protocol §6) and is `admissible` only if those rows clear their bars; a gate can be
`passed` *only if* admissible. A gate whose instruments fail is reported **BLOCKED**, never FAIL — the
number is not a claim, the harness is not trusted. This is the exact substrate for "no gate, no change":
the runner refuses to emit a passing decode number when I2 (batch-consistency, the ALPS-4B BatchNorm
incident) or I3 (episode-split leakage) is broken. Composes the *existing*
`tanitad.instruments.checks` (I2/I3/I4) + frozen `RidgeProbe` — no reimplementation. Ships standard
`ade_fde`, an I3-correct `split_by_episode`, both ablations, a protocol-§6 `gates_metrics_json`
(instruments-first + PASS/FAIL/BLOCKED summary), and an `extra_metrics` seam for **Thursday's** custom
suite (LAL/TMS/OKRI/CNCE/LOPS) so the two halves compose without editing the runner. — impact: WP6 / D1–D3
/ D-004 — repo-path `.../2026-07-14-gate-runner-d1-d3/`.

**Tests:** 13 passed / 1.58 s, no simulator, no trained model; proves BOTH the PASS path (controlled-linear
data actually clears thresholds; grid readout 0.0002 m vs global-pool 3.97 m, A7) **and** the BLOCKED path
(bad/missing I2 → BLOCKED; garbage predictor → D2 BLOCKED on I4); end-to-end run of a real
`WorldModel(smoke_config)` whose batch-free-norm encoder genuinely passes I2. Full stack suite unaffected
(65 passed, 1 skipped). G-E met (verifiable increment + passing tests).

## 1b. Implementation increment — p0-spectral-sizing (backlog #0, L2 / theory-watch)

Second intake package (D-011): `Architecture & Inference/Implementation/incoming/2026-07-14-spectral-sizing-p0/`
(`tanitad_spectral_sizing.py` + tests + INTAKE). Proposed target `stack/tanitad/eval/spectral.py`.

**What & why.** The new backlog #0 operationalizes leverage action **L2** from Sayed's JEPA-theory note:
arXiv 2606.27014 proves the optimal latent dim k sits at the **knee of the singular spectrum of the
action-conditioned transition operator** (approximation error = spectral tail Σ_{i>k}σᵢ², ↓k; sample error
~O(k²), ↑k; Thms 4.3–4.6). The tool fits a linear operator `(z_t, a_t) → z_{t+1}` with the tested
`RidgeProbe`, SVDs the state-transition block, and reports σ decay, entropy effective-rank (the *offline
twin of the live `erank` collapse row*), the 99%-energy knee, a trade-off-optimal `k*`, the spectral tail
at candidate dims, and an OVER-/UNDER-provisioning verdict **against the current 2048-dim readout** (D-008
= grid 4×4 × d_readout 128). — impact: H3 / WP3 / D-008 sizing — repo-path.

**Tests:** 8 passed / 1.52 s. Recovers a KNOWN rank r=5-in-32 (knee==5, effective-rank≈5, tail beyond r≈0,
sharp σ gap); OVER-/UNDER-provisioning flags fire correctly; `k*` grows toward the knee with N; end-to-end
`WorldModel` latent path. **Honest scope (P8):** a decision-grade comma2k19 spectrum needs a *trained*
checkpoint — untrained/collapsed latents (p0-sB00) are near-isotropic → degenerate spectrum. The real
sizing run is queued behind the A40 Stage-0 checkpoint; this is the tool it calls. No sizing claim on
untrained latents. Full stack suite unaffected (65 passed, 1 skipped).

## 2. Research delta — latent world models for driving (H1/H3)

- **The finding that reshapes our gate reading (arXiv [2512.24497](https://arxiv.org/abs/2512.24497),
  "What Drives Success in Physical Planning with JEPWMs?", large-scale ablation).** Models that unroll
  faithfully do **not** necessarily plan well: *"even with models which are able to faithfully unroll a
  large number of actions, success at the planning task is not an immediate consequence."* → **Decode /
  probe quality is necessary but not sufficient for driving competence.** Direct consequence for us: D1–D3
  are *instrument* gates on the representation; closed-loop **D4–D6 remain the arbiters** of the hierarchy
  edge. Baked into the gate runner (labelled necessary-not-sufficient) and into G-AI1 below. Other
  concrete ablation results: **AdaLN action-conditioning wins** (feature/sequence conditioning worse) —
  our FiLM predictor is AdaLN-family, **confirmed**; adding **RoPE** to the conditioning is a cheap
  bake-off lever. **Multistep rollout loss** (2-step in sim, up to 6-step on real DROID) acts as
  "data-augmentation against compounding error" — we currently train multi-horizon *heads* {1,2,4} from a
  single window but do **not** feed predictions back during training; a K-step rollout loss is a concrete
  predictor increment (H5 link, ties into imagine-and-select which already rolls). **ViT-L encoder +
  depth-12 predictor optimal for complex real dynamics** — validates base250 (d768×14 enc, ×12 pred).
  Caveat for H4: **DINO > V-JEPA encoders** for planning in their setup — a supporting data point for arm
  B (frozen DINOv3) and for our Pareto framing (not absolute SOTA). — impact: D1–D3 / H1 / H4 / H5.
- **V-JEPA 2 AC** ([arXiv 2506.09985](https://arxiv.org/html/2506.09985v1)): a **300 M** block-causal,
  action-conditioned latent world model that autoregressively predicts the next-frame representation from
  action + previous states. Same family and **same envelope as our 261 M** operative path — external
  confirmation that action-conditioned latent prediction at our scale is the right size class, not a
  compromise. — impact: H1 / D-008 scale sanity.
- **LeWM / stable latent WMs** ([overview](https://medium.com/@adnanmasood/leworldmodel-and-the-case-for-stable-latent-world-models-0e4c33ca0f3c)):
  end-to-end action-conditioned JEPA from raw pixels with **two loss terms, no EMA/stop-grad**, avoiding
  representation collapse. Directly **supports our LeJEPA/SIGReg-only, crutch-free anti-collapse choice
  (H3, D-003)** — the field is converging on "regularize, don't stop-grad". — impact: H3.

## 3. Efficient decoding (H5)

- **MTP draft-then-verify** is now standard vocabulary (multiple 2026 refs, e.g.
  [FlexDraft](https://arxiv.org/pdf/2605.20022)): lightweight draft heads propose k future steps, verified
  in one forward pass. Our multi-horizon heads {1,2,4} are exactly draft heads — the free-speedup framing
  (H5) holds; the *rollout-loss* result in §2 says training them with fed-back predictions also improves
  quality, not just speed. — impact: H5.
- **Revisable trajectory decoding** ([ReflectDrive-2](https://arxiv.org/html/2605.04647v1), masked discrete
  diffusion): any subset of trajectory tokens can be re-masked and rewritten, unlike token-by-token VLA
  planners whose latency scales with trajectory length. This is the alternative to our **discrete tactical
  vocabulary + imagine-and-select** (which is K batched predictor passes, milliseconds, no diffusion, no
  CEM). Our approach stays cheaper on the efficiency moat; diffusion planners are a Phase-1 **comparison**
  target, not an adoption — consistent with Phase 0 Plan's explicit rejection of CEM/diffusion planners. —
  impact: H5 / tactical WP4.

## 4. MoE modality/skill steering (H2/H8)

- **DriveMoE** (CVPR 2026, [arXiv 2505.16278](https://arxiv.org/abs/2505.16278),
  [code](https://github.com/Thinklab-SJTU/DriveMoE)): Scene-Specialized **Vision MoE** (a learned router
  prioritizes camera *views* by driving context) + Skill-Specialized **Action MoE** (flow-matching
  planner, experts for merge/overtake/brake/yield). **GEMINUS** ([arXiv 2507.14456](https://arxiv.org/abs/2507.14456)):
  dual-aware global + scene-adaptive MoE. Both route on a **learned scene signal**. Our differentiator
  (H15↔H2 link, D-008): route the tactical/sensor MoE on the **imagination's epistemic σ** — power a sensor
  down / gate an expert only where the imagination's uncertainty in its field of view is low. That is a
  *principled* gate (self-monitoring substrate), not a learned black-box router. Actionable for WP4's MoE
  upgrade: the router's input includes `ImaginationField.logvar` per sector. — impact: H2 / H8 / WP4.

## 5. Deployment / quantization (H5 / CNCE / efficiency moat)

- **Native TensorRT ViT INT8 is a known trap** ([SqueezeBits](https://blog.squeezebits.com/how-to-quantize-transformerbased-model-for-tensorrt-deployment-55802)):
  MHA (and RoPE) block automatic INT8 kernel fusion; native TRT quantization is "incompatible with the ViT
  architecture". Tooling that *does* work: OwLite (30 % latency cut, 0.7 % acc drop on ViT INT8);
  [DFQ-ViT](https://arxiv.org/pdf/2507.14481) (data-free, no fine-tuning); NVIDIA **ModelOpt** PTQ
  (FP8/INT4/FP4/INT8, [guide](https://www.spheron.network/blog/tensorrt-model-optimizer-modelopt-quantization-guide/)).
- **Batch-free LayerNorm is the deployment ally, not just the I2 guarantee.** TensorRT-LLM's fused
  reduce-norm (ResidualAdd+LayerNorm in one kernel) is recommended precisely for **small batch** —
  i.e. our **batch-1 streaming** inference path. Our LayerNorm-only, batch-free-norm encoder (I2) is
  exactly the architecture that keeps the small-batch fused-norm path open. Actionable, no-cost-now: keep
  encoder input static `[6,256,256]` (Monday's rec #3 — no ONNX-shape divergence between sim-eval and
  deploy), and when Phase-1 quantization starts, plan the **OwLite/ModelOpt** path, not the native-TRT one.
  — impact: H5 / CNCE / Phase-1 deploy — labelled **estimate** (no measured latency yet; G-AI2).

## 6. Actionable recommendations (each names its gate + bake-off — G-AI1)

1. **[WP6, D1–D3 — ready for triage]** Integrate the gate runner → the Stage-0 bake-off (Phase 0 Plan §4,
   W2) can finally emit instrument-gated D1–D3 numbers on comma2k19 held-out routes. Falsifier: if a bake-
   off lever moves ADE but I2/I3 fail, the runner BLOCKS it — that is the isolation. Owner: MVP orchestrator.
2. **[Predictor, H5 — bake-off lever]** Add a **K-step rollout loss** (feed predictions back, K=2 to start)
   alongside the multi-horizon heads. Isolated by a one-lever run vs current single-window training; gate
   it on D2/D3 (imag-rel, imagined-ADE ratio). Evidence: 2512.24497 multistep-as-augmentation. Falsifier:
   no D2/D3 improvement at matched steps → drop it.
3. **[Predictor, H5 — cheap lever]** Try **RoPE** in the FiLM/AdaLN conditioning block; isolate vs current
   learned positional embedding; gate on D1 (decodability) and D3. Evidence: 2512.24497 "AdaLN+RoPE best".
4. **[Tactical MoE, H2/H8/WP4]** When the tactical MoE upgrade lands, route on **`ImaginationField.logvar`**
   (epistemic σ), not a learned scene router — our principled differentiator vs DriveMoE/GEMINUS. Gate:
   the modality-steering exit demo (G0.7 Pareto) + must not regress D2. Falsifier: σ-gating Pareto ≤ learned
   router at matched active-FLOPs.
5. **[Deploy, H5 — do NOT native-TRT]** Record now: Phase-1 ViT INT8 goes via **OwLite/ModelOpt**, keep
   LayerNorm-only + static `[6,256,256]` input. No resource-ledger entry (no GPU spend this run).
6. **[Do not adopt]** Diffusion/masked-discrete-diffusion trajectory planners (ReflectDrive-2) and
   DriveMoE's flow-matching planner stay Phase-1 **comparison** targets — heavier than imagine-and-select.

## 7. Self-critique (quality gates)

- **G-A** sources: every §-claim carries a link or repo path. ✅  **G-B**: 6 actionable recs, each tied to a
  hypothesis/WP. ✅  **G-C**: KB updated (deltas, newest first). ✅  **G-D**: HYPOTHESIS_LEDGER updated —
  H3/H5 gain external support, H4 gains a DINO>V-JEPA data point, H2 gains the σ-gating differentiator (no
  status *upgrade* — all external evidence, none measured on our stack; honest). ✅  **G-E**: 13-test
  passing standalone increment + measurable next step (Stage-0 bake-off through the runner). ✅
- **G-AI1**: every recommendation names the gate that would falsify it and the one-lever bake-off that
  isolates it. ✅  **G-AI2**: the only efficiency numbers cited (OwLite 30 %/0.7 %, fused reduce-norm) are
  labelled **external / estimate** — no measured TanitAD latency claimed this run. ✅
- **Honesty (P8)**: the headline research finding *contradicts* a convenient reading of our own gates
  (decode≠planning) and is recorded as first-class, shaping the runner rather than being buried. The gate
  runner's BLOCKED path is tested, not just asserted.
- **Gap (recorded)**: D1–D3 have not yet been *run on real comma2k19 latents* — that needs the A40 Stage-0
  run (Sayed, RUNPOD_RUNBOOK). The runner is validated on controlled + smoke-model data only; no gate is
  claimed PASS on real data this run.
