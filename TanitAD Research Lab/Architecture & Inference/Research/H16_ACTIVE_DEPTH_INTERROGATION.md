# H16 — Tactical-Commanded Active Depth Interrogation (Sayed's idea, 2026-07-11)

**Status:** open (idea → hypothesis). **Origin:** Sayed, verbatim intent: let the model decide
inherently *"the situation is complex, I need to know the distance to a certain critical part of the
environment, to confirm assumption, imagination or planning scenario"* — the tactical layer picks the
right camera AND an ROI within it ("zoom"), runs a ZipDepth-class lightweight depth net ONLY there,
on demand. Matches the tactical-steered MoE architecture (H2/H8).

## Why this is architecturally interesting (assessment)

It composes three assets we already have into an **active-perception** capability none of the
profiled competitors has:

1. **The trigger already exists:** H15's ImaginationField emits per-sector epistemic σ. "Situation is
   complex / assumption unconfirmed" = high σ on a critical sector ∧ that sector matters for the
   currently imagined plans (imagine-and-select already scores plans — the gradient of plan-value
   w.r.t. sector uncertainty identifies WHERE confirmation pays).
2. **The scheduler is H2:** attention-based camera steering extends naturally from "which camera to
   encode" to "which camera + which ROI to interrogate with a specialist tool". In MoE terms the
   depth tool is **one more expert with a routing cost** — the router learns WHEN a query is worth
   its latency/energy, via the same load-balancing/cost-penalty machinery as H8.
3. **The specialist is commodity:** ZipDepth-class = 6.1M params, ~5 ms-class on an ROI crop,
   ~hundreds of mJ full-frame on Orin NX — an ROI query at tactical cadence (1–2 Hz, only in complex
   scenes) is energy-negligible vs. running full-frame depth on every camera continuously (what the
   lidar-less competitors effectively do).

**The under-appreciated bonus — resolution recovery:** our encoder consumes downsampled frames
(224–256 px). An ROI crop taken at NATIVE sensor resolution recovers detail the encoder threw away.
The query is not only cheaper than full-frame depth — it is *better informed* at the point of
interest than anything the encoder saw. "Zoom" is digital (crop), no hardware implication.

**Safety narrative (inherent-safety edge):** the model operationalizes its own epistemic humility —
it KNOWS what it doesn't know (σ, D9-calibrated) and has a cheap measurement tool to resolve exactly
that. "Confirm imagination with measurement before acting on it" is runtime verification of world-model
beliefs; maps cleanly onto UN-ADS risk-confirmation language. Fallback layer gets the depth answer as
an envelope/veto input (independent channel), the tactical layer optionally as a measurement token.

## Honest caveats (pre-registered risks)

- **Metric scale is THE technical risk:** affine-invariant mono-depth on a crop is scale-free.
  Candidate anchors, in order of preference: (a) ego-speed from CAN + two consecutive queries →
  SfM-style metric scale (we have real actions — H7 asset); (b) ground-plane intersection inside the
  crop; (c) known-size priors (lane width, vehicle class width). If none survives contact with data,
  H16 degrades to *ordinal* confirmation ("A closer than B") — still useful for plan ranking,
  weaker for the envelope.
- **Quality artifacts** (Sayed's own caveat on ZipDepth): quantify artifact rate on our val routes
  BEFORE any safety-envelope claim (Prod backlog 3b step b). Envelope/veto use only until then.
- **Latency budget:** a query (~5–15 ms) fits the tactical 1–2 Hz cadence, NOT the operative
  10–20 Hz path. H16 confirms plans/imaginations, never reflexes — by design.
- **Trigger learning:** Phase-1 baseline = hand σ-threshold (zero training risk); learned router
  with query-cost penalty is the H8-style upgrade, only after the baseline shows queries land on
  genuinely depth-ambiguous events.

## Falsifiable predictions (what would make H16 real, or kill it)

- **F1:** σ-triggered queries concentrate on genuinely depth-ambiguous/critical events (occlusion
  onsets, cut-ins, work-zone closures — SC-01/SC-04 telemetry gives ground truth). Falsifier:
  queries scatter uniformly → trigger carries no information → H16 refuted at the trigger.
- **F2:** imagine-and-select WITH confirm-queries beats no-query at matched decision quality with
  ≥5× lower perception energy than always-on full-frame depth (CNCE-style ledger row). Falsifier:
  always-on wins or energy gap <2× → the scheduling premise fails.
- **F3:** metric anchor (a) or (b) achieves usable range error (<10–15 % at 10–40 m) on val routes.
  Falsifier: none does → ordinal-only downgrade, envelope claim dropped.

## Sequencing (no Phase-0 impact)

Phase 1 second half (~Sep), AFTER: multi-cam (mid-Aug), 30k D9 σ-calibration read, H2 scheduler
skeleton. Cheap pre-work that CAN happen earlier on the 4060 (Prod backlog 3b): ZipDepth ROI latency
curve (crop size vs ms), artifact assessment, plus one offline probe — replay SC-01 telemetry, ask
"would a σ-threshold trigger have fired before the reveal?" (pure logs, no new training).

**Cross-refs:** `../../2026-07-11-sayed-papers-screening.md` (ZipDepth), `ENCODER_MULTICAM_OPTIMIZATION.md`
(H2 scheduling), `Project Steering/REFERENCE_ARCHITECTURES.md` (imagination assets), HYPOTHESIS_LEDGER H16 row.
