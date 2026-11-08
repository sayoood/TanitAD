# HANDOFF - bundle 2 (closed-loop behaviour-suite candidates) - FINISHED, 2026-11-08

STATE: bundle built and independently verified. Nothing blocking. Awaiting the Master Mind's rulings on AMBIGUITIES.md, then the path goes to the simulator session (not messaged by this agent).

Bundle (private nuPlan/OpenScene data, never copy into a repo): `F:/TanitAD_tsf_bundle2_20261107/bundle/` (496 MB)
Code, results, pins: `E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/`. Frozen rule: BEHAVIOUR_SCORECARD.md s3.1, sha256 fac6c7f1... (unedited).

## Counts
27 tokens, 25 logs, 216 t0 JPEGs + 216 `tm2s` JPEGs (8 cams each), 783 files in MANIFEST.json (incl. 12 sky-prescreen frames x 27, calib rows, 25 meta pkl, a0 jsonl).

## What passed (MEASURED, `code/verify_bundle_independent.py` -> `VERIFY_BUNDLE_RESULT.json`; does not import build_bundle.py)
1. sha256 + byte size of all 783 files equal MANIFEST.json; no file on disk outside the manifest.
2. 27 tokens x 8 cams, t0 and tm2s: all 432 JPEGs decode, all 1920x1080.
3. tm2s = frame index-4 in the OpenScene meta pkl (path-checked per cam against the manifest); dt to t0 = 1.9989 .. 2.0008 s (rule 2.0 +- 0.05): no outlier; equals the manifest's dt values.
4. Byte equality t0 + tm2s vs source tgz (camera_24.tgz) for 3 tokens (1a1fbb25, 86cad912, 87751a07): 48/48 equal.
5. a0_model_inputs.jsonl: 27 rows, one per token, log matches the meta, ego7 len 7, goal len 4, 0 null picks. Fields = the cruise bundle's 5 fields plus 2 extra (`device`, `forward`); none missing. Pick is a CPU fp32 forward (the cruise bundle's came from GPU): a near-tie pick may differ.
6. Selection reproducibility: `select_suite_tokens.py` rerun from scratch (~3 min, out on F:) gives byte-identical SUITE_CANDIDATES.json (3d669b74...) and SUITE_CANDIDATES_ALT.json (5633414d...); suite_features.csv.gz decompresses identical (the .gz bytes differ only by header). Script sha256 359c571059d8... equals SUITE_PIN.json.
7. HAND_VERIFY.txt (21:32) post-dates the final selection run (run2, 21:29), whose JSON hashes equal the final files; it covers the rank-1 LITERAL candidate of all 5 slots (86cad912, 024d89a3, 9164913b, 3984225e, 963d98ea) against the raw DB, and these are rank 1 in the final list. Not redone. One flag in it: stop_go 3984225e shows main vs independent gap max diff 1.924 m over the 148 ticks with a lead (median 0.000, same lead id 171/171); other slots max 0.000 m.

## Shortfalls per slot
None: every slot has its 5 candidates, PRIMARY and ALT (shortfall 0). Thin margin: cruise has only 11 passing logs (22 tokens), so a tier-2 cruise slot of 6 leaves little margin (AMB-13). The bundle holds the 5 per slot; ranks > 5 are in SUITE_CANDIDATES.json but have no images. f233f263 is both lead_braking #5 and stop_go #5 (AMB-14).

## Open ambiguities (NOT ruled on here): see `AMBIGUITIES.md`
Biggest: AMB-1 (free-road reading; literal text gives 35 vs 22 cruise tokens), AMB-7/8/9 (event may lie before t0: 2/5 steady lead, 3/5 braking without the event at t0), AMB-10 (2/5 merge candidates are 90-degree crossing traffic), AMB-12/14. ALT list = secondary, not the frozen rule; adopting it needs an AMENDMENT.

## Previous agent's scratch, NOT deleted (for the Master Mind to decide)
On F:/TanitAD_tsf_bundle2_20261107/ (du -sk): `a0_capture` 21 KB, `a0_forward` 21 KB, `a0_forward.out` 12 KB, `a0_fwd_cruise5` 9 KB, `a0_fwd_cruise5.out` 4 KB, `a0_fwd_test` 3 KB, `a0_test` 21 KB, `work` 480 KB, `work_build_all.log` 0, `work_build_chain.log` 4 KB (all tiny).
On E: `run1/` 11 MB (superseded first selection run), `run2/` 14 MB (final run, source of the pinned files; keep), `SKY_METHOD_CHECK.png` 3.6 MB.

## Temp of this agent
`F:/b2fin_tmp_agent/` (~11 MB: verifier copy, outputs, selection rerun) - deletion was refused by the safety check; verifier script and result are copied into the E: folder. Needs the user/MM to delete `F:/b2fin_tmp_agent` (literal path).
