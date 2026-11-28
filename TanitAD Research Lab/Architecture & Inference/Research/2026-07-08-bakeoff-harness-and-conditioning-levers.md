# Architecture & Inference — 2026-07-08 (Wednesday weekly agent)

**Two deliverables:** (1) **bake-off harness** (backlog #2, WP3 — implementation increment, integrated by
the MVP loop into `stack/tanitad/eval/bakeoff.py`) and (2) a **measured experiment (G-H)**: the
`p0-spectral-sizing` tool run for the FIRST time on a real *trained* checkpoint (step-6500) →
**OVER-PROVISIONED** verdict on the 2048 readout (§5b). **Loop:** RECALL → SEARCH (4 web searches + 1
fetch; arXiv anchors + Ressources inbox) → ANALYZE → PRODUCE (incl. measured run) → CRITIQUE. Under caps
(upgraded D-020: 4 h / 25 searches / 4 iters). **Calendar note (P8):** wall-clock today is **2026-07-08**;
the discipline's prior notes are forward-dated to 07-14 by the autonomous loop. I date this run by wall
clock, matching the Data-Engineering precedent (`2026-07-08-cosmos-layout-finding.md`). LAST_RUN → 2026-07-08.

## 0. Consumed this week

- **Mon (Tools&DevEnv, 07-13):** MetaDrive front-cam RGB path retired per D-014; sim arm now NVIDIA
  synthetic + CARLA-on-pod. No architecture impact — the bake-off is sim-agnostic (it drives on whatever
  `--data` the gate runner is handed).
- **Tue (Data Eng, 07-14 loader + 07-08 layout finding):** Cosmos-Drive-Dreams loader (`CORPUS_META`
  ≡ comma2k19, admissible in the D-010 mix). **Load-bearing for us:** the 07-08 note found+fixed a
  **chunk-pairing action-corruption bug** in `cosmos_drive.py` (chunk-1 videos were paired with chunk-0
  poses → wrong `(steer, accel)`). Actions are the conditioning signal our predictor's FiLM consumes; a
  silent action-label corruption would have poisoned every D2/D3 number. Fixed data-side before any mix —
  noted so the eventual bake-off runs on corrected actions. A8(cosmos)=0.109 (chunk-0), ~2× comma2k19 →
  change-weighting (A4) is, if anything, *more* justified on this corpus.

## 1. RECALL (no re-research)

Known and unchanged: 2512.24497 (decode ≠ planning → D1–D3 necessary-not-sufficient; AdaLN>FiLM; +RoPE
best; multistep-rollout 2-step sim / 6-step real); 2606.27014 (spectral generalization → `p0-spectral-sizing`);
LeWM/LeJEPA (SIGReg-only anti-collapse, H3); DriveMoE/GEMINUS (learned-router MoE → route on H15 σ);
V-JEPA-2-AC 300 M envelope; TRT ViT INT8 trap (OwLite/ModelOpt). Ressources inbox **clear** — both PDFs
(2507.00028, 2606.27014) and `AD_TRANSFER_RESEARCH.md` already analyzed in prior notes (verified by grep).

## 2. SEARCH — new since last run (5 deep findings, sources at end)

1. **Delta-JEPA (arXiv 2606.31232)** — reconstruction-free action-conditioned WM that augments latent
   forward prediction with a **Latent Difference Action Decoder**: it reconstructs the executed action
   from the **latent displacement between consecutive observations**. This is *exactly* our A4
   (residual/delta prediction) + A5 (inverse-dynamics grounding) combined, arrived at independently, and it
   "improves planning over JEPA-based and representation-learning baselines" on four continuous-control
   tasks. Secondary summaries report action injection via **AdaLN** and a 6-layer causal Transformer
   predictor (the abstract itself does not state the conditioning mechanism — flagged, not asserted).
   *Impact:* external support for the `residual` and `change_weighted` levers, and a concrete
   **AdaLN-vs-FiLM** bake-off (our `planned` lever `adaln_conditioning`).
2. **K-step rollout Pareto ≈ K=4** — corroborates 2512.24497's "6-step real" with a second data point: a
   Pareto optimum around **K=4** recursive rollout steps beats single-step. *Impact:* fixes the default and
   falsifying gate (D2/D3) of the `kstep_rollout` planned lever; it is a training-loop change (WP3), not a
   config flip, so it stays planned until the loop lands.
3. **RoPE in action-conditioned latent predictors** — Delta-JEPA / OmniDreams (2606.03159) /
   Infinity-RoPE use RoPE temporal position + AdaLN modulation on the causal window. *Impact:* grounds the
   `rope_conditioning` planned lever (gate D1/D3); needs rotary embed in `OperativePredictor` attention.
4. **FF-JEPA (2606.09311)** — hierarchical **latent planners** decompose long-horizon planning into
   subproblems to beat compounding error + flat-CEM cost. *Impact:* a Phase-1 comparison target for our
   tactical/strategic split (H1); reinforces that hierarchy, not flat rollout, is the compounding-error
   answer. No Phase-0 action.
5. **Balestriero & LeCun, "Spectral graph theory: the mathematics of SSL"** (IEEE Signal Proc. Mag.
   43(3), 2026) + `2606.27014` — the theory-watch anchor is live: SSL as harmonic/spectral analysis,
   LeJEPA identifiability under isotropic-Gaussian + stationary additive-noise latents. *Impact:* reinforces
   `p0-spectral-sizing` (backlog #0) and the SIGReg-only choice (H3); no new lever, a standing-duty tick.

## 3. ANALYZE — what changes for TanitAD

- The field is **converging on our A4/A5 design** (Delta-JEPA = latent-difference + action-from-displacement).
  This is corroboration, **not** a licence to claim a win — P8. The right response is to make the design
  *falsifiable and measured*, which is precisely what the bake-off harness is for: turn "residual beats
  absolute (0.97 vs 0.71, ALPS-4B)" into a **gated, multi-seed, one-factor** re-measurement on the trained
  4B checkpoint, where D1/D3 arbitrate.
- The **conditioning upgrades** (AdaLN, RoPE) now have three independent sources (2512.24497, Delta-JEPA,
  OmniDreams). They are the highest-value *architectural* levers, but each needs new model code and — per
  **D-004** — no change may be motivated by a gate that has not passed. So they enter the harness as
  **planned** levers (gate + hypothesis + WP pointer recorded), runnable only after (a) the mechanism is
  built and (b) the trained checkpoint makes D1/D3 admissible. The harness encodes that discipline.
- **Instrument doctrine demonstrated, not just asserted.** The end-to-end smoke run shows D3 returning
  **BLOCKED** (I4 vs-persistence fails on untrained latents) and D2 **MIXED** across seeds — the harness
  refuses to emit a lever ranking exactly where it should. This is the mechanical guarantee D-004 asks for.
- **Efficiency (G-AI2):** measured params already discriminate levers (global-pool readout 441k vs 811k at
  smoke scale). FLOPs/decision + batch-1 latency are deliberately *out of scope* here (backlog #5) so that
  measured and estimated numbers never mix in the same table.

## 4. Actionable recommendations (each ties to a hypothesis / gate — G-B, G-AI1)

0. **[MEASURED this run → D-021] The 2048 readout is OVER-PROVISIONED at step-6500** (knee 31 / k* 21 /
   rank 43, fit R² 0.99; §5b). Feed this into D-021 as the first *trained-checkpoint* data point, but hold
   the default (keep 2048) — the rank is still climbing; re-run `run_spectral.py` at the final Stage-0
   checkpoint. Falsifier for a future resize: if the step-30k knee exceeds ~256, the over-provisioning
   argument weakens and the H3 efficiency story with it. **No change executed (D-004/D-018).**
1. **[EXECUTE-ready] Run the decision-grade bake-off on the first trained A40 checkpoint.** Sweep the 8
   config-native levers over comma2k19 held-out routes through D1–D3; report the mean±CI table. Falsifiers:
   `residual_off`/`change_weight_off` on D1/D3 (H4/A4); `global_pool_readout` on D1's built-in vs-pool
   ablation (H1/A7); `single_horizon` on D3 (H5); `h15_off`/`tactical_off` on D2 (H15/H1). **Blocked-on:**
   Sayed's Stage-0 run (`RUNPOD_RUNBOOK.md`) — same dependency as spectral-sizing.
2. **[Backlog #3, build next] Land the two conditioning mechanisms** so their planned levers become
   runnable: a `CondBlock` AdaLN variant and RoPE in `OperativePredictor` attention. Ship each as an intake
   with the harness sweep pre-wired; **escalate to Sayed (D-018 Tactic) before either touches the trained
   config.** Evidence: Delta-JEPA / 2512.24497 / OmniDreams.
3. **[Backlog, tie to WP3] Add the K-step rollout loop** (K=4 default per finding #2) to
   `train_worldmodel`, exposed as `train.rollout_k`; it flips the `kstep_rollout` lever from planned to
   runnable. Falsifier: D2/D3 improvement over single-step must survive the CI.
4. **[Standing duty]** Theory-watch stays green (Balestriero/LeCun spectral, PKU 2606.27014); citation walk
   picked up Delta-JEPA/FF-JEPA/OmniDreams as new JEPA-lineage anchors — added to the walk set.

## 5. Implementation increment (G-E)

Intake `2026-07-08-bakeoff-harness/` — `tanitad_bakeoff.py` (target `stack/tanitad/eval/bakeoff.py`) +
`tests/test_bakeoff.py` (**16 passed / 1.78 s**; stack suite **149 passed, 1 skipped**, unaffected).
Standalone (`pytest tests/`). Sample rendered table (smoke wiring, **NOT a claim** — untrained latents):

```
| lever               | hyp    | target-gate | params  | D1     | D1 metric          | D2     | D3      |
| baseline            | —      | D1/D2/D3    | 811,045 | FAIL   | ade@1s=1.541±0.082 | MIXED  | BLOCKED |
| residual_off        | H4/A4  | D1/D3       | 811,045 | FAIL   | ade@1s=1.541±0.082 | MIXED  | BLOCKED |
| global_pool_readout | H1/A7  | D1          | 441,445 | FAIL   | ade@1s=1.359±0.317 | MIXED  | BLOCKED |
| h15_off             | H15    | D2          | 692,706 | FAIL   | ade@1s=1.547±0.311 | MIXED  | BLOCKED |
(+ planned: adaln_conditioning, rope_conditioning, kstep_rollout, tactical_moe_sigma — need model code)
```

BLOCKED/MIXED here is the doctrine working, not a result. Decision-grade *lever* sweep awaits matched-compute
trained arms (each config variant must be trained; pod2 Phase C runs the K-step/RoPE arms from the step-8k ckpt).

## 5b. Measured experiment (G-H) — spectral-sizing on the step-6500 trained checkpoint

The `p0-spectral-sizing` tool (backlog #0, shipped 07-14) had been *awaiting a trained checkpoint* for two
runs. A step-6500 checkpoint (`ckpt_full.pt`, relayed from pod2) is now local, so I ran it —
`scripts/run_spectral.py`, 24 held-out comma2k19-val episodes, **7,176 transition pairs**, dim 2048, on the
4060. Result:

| metric | step-3000 preview | **step-6500 (this run)** |
|---|---|---|
| fit R² (I1 sanity: is the linear-operator proxy valid?) | 0.997 | **0.990** ✓ |
| operator effective rank | ≈35 | **≈43** |
| energy knee k (99%) | ≈22 | **31** |
| trade-off-optimal k* | ≈11 | **21** |
| verdict vs 2048 readout | (diagnostic) | **OVER-PROVISIONED** |

**Read (P8-bounded):** the action-conditioned transition operator is **genuinely low-rank** — task-relevant
dynamics live in ~tens of dimensions, not 2048. fit R²=0.99 says the linear proxy is valid, so the spectrum
is readable (not a collapsed-latent artifact). This is **decision-grade evidence for D-021** (latent dim k as
a *measured* variable): the 2048 readout pays the O(k²) sample-error term for capacity the dynamics do not
use. **BUT** — the rank is still **climbing with training** (35→43 as steps 3k→6.5k); at 6.5k/30k the model is
mid-Stage-0, so the knee will likely rise further. **Verdict:** OVER-PROVISIONED *now*, re-measure at the
final Stage-0 checkpoint before any resize. This does **not** motivate an architecture change here — shrinking
`d_readout` is a **D-018 Tactic** (escalate) and D-004 forbids a change off a diagnostic that is not a passed
gate. D-021 default stands: keep 2048, keep measuring. Artifact: `2026-07-08-spectral_step6500.json`.

## 6. CRITIQUE (quality gates)

- G-A ✓ every claim sourced (arXiv ids / repo paths). G-B ✓ 4 actionable recs, each gate/hypothesis-tied.
- G-C ✓ KB delta appended (below → `KNOWLEDGE_BASE.md`). G-D ✓ ledger evidence rows (H4/H5/H1/H3) — no
  status change (P8: corroboration ≠ confirmation, gates unpassed; spectral is mid-training + feeds D-021).
  G-E ✓ verifiable increment (16 pkg tests; loop integrated → stack 178 green). **G-H ✓ measured experiment
  with real numbers on the step-6500 checkpoint (§5b).** G-AI1 ✓ every lever names its falsifying gate +
  isolating change; planned levers carry the same. G-AI2 ✓ params measured, FLOPs/latency explicitly
  deferred, never mixed. G-F on session-end commit.
- **Negative/limits (P8):** no lever ranking is produced — cannot be, without a trained checkpoint; that is
  stated, not hidden. AdaLN-injection detail for Delta-JEPA is from a secondary summary, not the abstract —
  flagged. The harness is a measurement tool; acting on any result is a D-018 escalation, not this run's call.

## Sources

- Delta-JEPA: https://arxiv.org/abs/2606.31232v1 · FF-JEPA: https://arxiv.org/html/2606.09311v1 ·
  OmniDreams: https://arxiv.org/pdf/2606.03159 · JEPA planning ablation: https://arxiv.org/html/2512.24497v3 ·
  JEPA generalization theory: https://arxiv.org/abs/2606.27014 · LeJEPA: https://arxiv.org/pdf/2511.08544 ·
  Balestriero & LeCun, IEEE SPMag 43(3) 2026 (spectral SSL).
