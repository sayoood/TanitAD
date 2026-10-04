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
