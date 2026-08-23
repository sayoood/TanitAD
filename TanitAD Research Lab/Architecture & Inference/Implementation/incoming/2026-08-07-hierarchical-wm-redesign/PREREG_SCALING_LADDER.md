# PRE-REGISTRATION SCAFFOLD — the scaling ladder: data, distribution, resolution, encoder

**Drafted 2026-08-10 ~21:15Z (PI question: "do we need further analysis regarding the
required amount of data and its distribution, resolution, encoder etc to boost the
performance of our flagship and prove the dominance of 4-brain TanitAD?"). Answer: YES — and
none of it exists yet as a measured curve. This scaffold names the arms; each PHASE gets its
own full prereg (gates + both outcomes bound) before it runs. GPU-day spends are PI
decisions; this doc makes them decidable.**

## The honest baseline

v5f-30k trains on **2,376 episodes ≈ 13 h** of the canonical corpus (parity f09e44db).
PhysicalAI-AV offers **306,152 clips ≈ 1,701 h** — we currently consume **<1 %**. No
measured data-volume curve exists for ANY arm of the programme; E6 (committed prereg) tests
hierarchy-vs-monolith data EFFICIENCY at fixed small scale, not absolute scaling. Resolution
and encoder width have single points (176×624-of-256×640 cyl; w120). "Dominance" claims need
curves, not points.

## S1 — data volume (the first and cheapest lever to measure)

Arms: {1×, 3×, 10×} the canonical corpus (episode-count nested, selection by the SAME
distribution machinery as the aug manifest — road-class stratified so the added data is
diverse, not more highway). Fixed architecture (v5.8f), fixed steps-per-episode budget vs
fixed wall budget as two stated variants. Metric: T0 oracle + T1 ADE + S-rate + four
families. Gate sketch: log-log slope of T1 ADE vs data; the decision rule for "build the 30×
corpus" binds to the measured 1×→10× slope (no extrapolation past 2× of the fitted range —
the learning-curve rule in CLAUDE.md applies to DATA curves too). Cost: corpus builds (~1-2
pod-days I/O) + 2 extra trainings (~2× 30k-run cost each at 3×/10×). ⚠ corpus building must
NOT re-select episodes of the parity corpus — additive supersets only, new parity hashes
registered.

## S2 — distribution (what data, not how much)

Same total hours, three mixes: {as-is, intersection/turn-enriched (road-class + VLM domain
labels when PH1 lands), longitudinal-event-enriched (E4.1 LON severity mining)}. Metric:
tail metrics per family (p90 per-window errors) + S-rate + manoeuvre-class recall, NOT means
only — distribution moves tails first. This is where the VLM labeling (17987be) pays into
the flagship directly.

## S3 — resolution & context

{176×624 (current), 256×640 full frame, +longer temporal window 8→16} at matched compute
(width trade stated per arm). Gate: does resolution buy LATERAL family error at junctions
(the hypothesis: sign/lane detail) or nothing the families can see?

## S4 — encoder capacity & pretraining

{w120 (current), w160/deeper at matched data} ± initialisation from a pretrained encoder
(the Alpamayo-teacher distillation option E4.5 generalised: distill encoder features on our
corpus, contamination caveat carried). The sub-300M budget is the binding envelope for ALL
arms — "dominance of 4-brain TanitAD" = beating bigger published baselines at ≤300M total,
so every S4 arm reports total param count against the envelope.

## Sequencing & the dominance claim

S1 first (cheapest, most decision-relevant), after v5.8f §1.14 + T1 rows exist as the fixed
yardstick. S2 needs VLM PH1 labels (sequenced). S3/S4 after S1's slope says whether data or
capacity is binding. The dominance claim itself additionally requires the external yardstick
work: T1-comparable published baselines (nuPlan/NAVSIM-style metrics mapped onto our
families) — a separate memo when S1 lands.

- [ ] S1 full prereg · [ ] S1 corpus supersets built · [ ] S1 run · [ ] S2 · [ ] S3 · [ ] S4
