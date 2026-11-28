# WP-A COMMS — decisions asked / made, handovers, integration status

## 2026-10-04 — Stage 1 handover to the Master Mind (SPEC review gate)

* **Delivered:** `SPEC.md` (v9 definitions + the echo/leak study design + every validation bar, committed before any
  build), two S1 probes (`code/s1_probe_*.py` → `raw/s1_*.json`), the Hidden-Biases citation recorded in the Library
  (`tools/kb_add.py 2306.07957 … --cited-by SPEC.md`, then `--verify`: 605 entries, 0 problems).
* **Nothing was built**; no Thor job was started in Stage 1.
* **Asked (SPEC §13):** D-WPA-1 … D-WPA-8. None blocks Stage 2 (the echo/leak study) — it uses only §5/§6 fields and
  ships every variant.
* **Waiting on:** the Master Mind's go on SPEC.md (Stage 2 starts on it).

## Integration notes for WP-B (preview; the binding contract is Stage 4's `INTEGRATION.md`)
* Inputs WP-B will receive: `nav_token`, `nav_side_next`, `nav_d_next_m`, `nav_d_end_m`, `nav_dyaw_next_deg`,
  `nav_args_valid`, `nav_lookahead_m`; the chosen RC variant's `(x, y, ψ)` + `rc_valid`; optionally N2/N3.
* ⛔ Never inputs: `nav_t_next_s`, `nav_token_ttime`, `RC-H*`, any goal / action field, any VLM token.
* Targets: `lat_cls_{a,b}` / `lon_cls` (+ `*_allowed` partial-label masks, + v7-id encodings), the constraint fields,
  the 22-token goal `y/w` bits per frame, the continuous SPEED goal.
* EvalFlyWheel (NavSim): RC from the scene route centreline is a privileged-route arm; report RC-ON and RC-OFF (SPEC §6.4).

---

<!-- Master Mind 2026-10-04: the Stage-1 handover above is kept verbatim (it landed in 2b8216e); the Stage-4 handover follows. -->

# WP-A COMMS — decisions asked / made, handovers, integration status

## 2026-10-04 — Stage 1 (SPEC review gate)
* Delivered `SPEC.md` (v9 definitions, the echo/leak study design, every validation bar — before any build), two S1
  probes (`code/s1_probe_*.py` → `raw/s1_*.json`), the Hidden-Biases citation recorded in the Library (`tools/kb_add.py
  2306.07957 … --cited-by SPEC.md`; `--verify` 605 entries, 0 problems).
* Master Mind ACCEPTED it with D-WPA-1..8 at my defaults: variant a; ship both id sets (WP-B uses v7); A7 clip-level
  suppression within the builder's 35-s view; the spatial nav token with R8-2's bar reported three ways; ship N2/N3; VLM
  lateral tokens without the NUDGE gate + per-cell evidence; absence on ≥ 6 s gated by V7.
* D-WPA-7 became two dropouts (Master Mind): RC dropout ≥ 0.3; nav-args dropout keeping the token with an "unknown"
  flag (NavSim's command form). The NavSim mapping and the two reporting rows (privileged / legal) bind INTEGRATION.md.

## 2026-10-04 — Stage 2 (echo / leak)
* Reported: as registered, no RC variant admissible (E2 FAIL vs NAV under trees); diagnosis: road geometry. Failed
  controls reported as worded (E1 scrambled-vs-CV unsatisfiable; O1 0.28–0.38 vs ≥ 0.40; trees uncertified).
* **Master Mind ruling (provisional; PI confirmation pending):** road-geometry (curvature-ahead) speed information is
  ADMISSIBLE in a route input; lane-level ego choice is not; the leak reference is the road-level route.
* `SPEC_ADDENDUM_S2A2.md` (E2′/E3′) written by WP-A, REGISTERED by the Master Mind (hash a44bd428…, 12:59:38Z), run
  after Stages 3–4 as instructed: **RC-A50 noised CONFIRMED** (certified instrument; Δ −0.035; E3′ PASS).

## 2026-10-04 — Stages 3–4
* Releases built and validated (RESULT §S3); LC ships never-positive (V5 FAIL); `H_ABS_MIN = 8.0` (V7 FAIL); reversing
  mask added post-hoc (disclosed). Master Mind accepted the V1 / V2 / V7 / V8 readings as reported.
* v7_labels module-state fix (added to Stage 4 by the Master Mind) delivered in `code/fix/` (86f0c46e → abb1f64c).
* Corpus manifest: neutral `drop_list` + "PI decision 8; MM default = keep all clips, apply the mask only" (accepted).

## Open items (owner)
* PI: confirm the road-geometry ruling (the RC confirmation rests on it); decision 8 (the drop list).
* WP-B: wire INTEGRATION.md; the trainer-side test for the v7_labels policy (INTEGRATION §6); the RC and nav-args
  dropouts and the RC training noise (load-bearing, E2′).
* EvalFlyWheel: the NavSim bridge's privileged row (our announced token + route args + RC from the route centreline)
  and the legal row (bare command).
* Levers WP-A named but did not run: a single-line lane-crossing detector to lift LC recall (V5); a relative tolerance
  for nav d_next (V3), if the Master Mind wants it registered; per-contested-turn suppression (D-WPA-3 alternative).
