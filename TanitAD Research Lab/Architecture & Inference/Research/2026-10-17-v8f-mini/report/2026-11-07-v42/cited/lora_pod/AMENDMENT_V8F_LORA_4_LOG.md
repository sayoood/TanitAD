# PREREG_V8F_LORA Amendment 4 (frozen `626aa1e5`), dated log beside it

* 2026-11-06 23:22 (dev-box clock), Master Mind. Item 5 names the gate wrongly; this entry fixes the gate route before any B-prime or LoRA training step.
  * Item 5 says each run starts "with a `launch_gate.py` PASS token". `stack/scripts/launch_gate.py` has no profile for `v8f_mini_train.py`: every profile binds `refc_v3_train.py` (AGENT, `lp2`, MEASURED).
  * The programme's ruling for this trainer is `E7_DECISION.md` (Master Mind, 2026-10-26). The launch gate for `v8f_mini_train.py` is the trainer's own in-process gate (`--selfcheck-only`: G-LIVE gradients, identity, every loss term finite). The chain writes the PASS token from that gate's output, and the token binds the argv sha256.
  * DAG-OACC and DAGDOSE launched by that route. MEASURED there: their `launch_gate_token.json` reads `"hmac": "none: written by the box-2 chain, not by launch_gate.py"`, with gate kind `v8f_mini_train.py --selfcheck-only`.
  * B-prime and L use that route. The token is written by the chain from the gate's own PASS output, never by hand, and the sha256 of `/root/ops/DECISION_INPROCESS_TOKEN.txt` enters both tokens.
  * Nothing else in Amendment 4 changes: argv, bars, estimator, the identity rule.
  * What had been seen: no B-prime or LoRA training step. The early real-GPU smoke readings on the partial cache are the chain's preflight: training-path identity rel-L2 0.0, 1.2 s/step, detach mutant RED in G-LIVE (`HANDOFF_LORAPOD2.md`).
