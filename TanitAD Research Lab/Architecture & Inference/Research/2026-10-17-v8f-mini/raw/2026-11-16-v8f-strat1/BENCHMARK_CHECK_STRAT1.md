# BENCHMARK_CHECK_STRAT1

Tool: `python tools/criteria_check.py F:/TanitAD-scratch/strat1/read/out/RESULT_STRAT1.json` (E:\Projects\TanitAD, registry `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.10.3), run 2026-11-16.

Tool output (verbatim):

```
=== RESULT_STRAT1.json ===
UNKNOWN_SCOPE: matches no in-scope or out-of-scope marker
the driving criteria do not apply to this artifact — not scored.
```

An UNKNOWN is never compliant. The registry has no in_scope marker for the STRAT1 schema (`families.<FAMILY>.cells`, `route_following`, `detected_info`). Work item W1: add the marker and a per-cell mapping, with a deliberate-regression arm in `tools/tests/test_criteria_check.py`. I did not edit the registry (not my file).

## Manual diff against the registry family criteria (MEASURED by reading RESULT_STRAT1.json; not the tool's verdict)

| criterion | state | where / reason |
|---|---|---|
| long.target_speed | PRESENT | speed_mae_mps, operative and tactical |
| long.along_track_error | PRESENT | along_mae_m, both bands |
| long.progress | ABSENT | no cell |
| long.distance_keeping | PARTIAL | dk_hw_abs_err_m, dk_tg_abs_err_s present; TTC and DK-VIOL (listed in card 4.1) absent |
| lat.cross_track / heading / curvature / yaw_rate | PRESENT | 8 cells, both bands |
| tac.manoeuvre_decision | PARTIAL | TAC-LAT, TAC-LON accuracy present; T decision F1 absent |
| tac.confusion | ABSENT | no cell |
| tac.goal_selection | ABSENT | goal macro-F1 not in the read |
| strat.decision | PARTIAL | route following S_CKPT (real, donor, zero, best constant) |
| strat.route_goal | PRESENT | RC-A route setting, 6 cells per seed |
| strat.nav_compliance + shuffle + zero controls | PARTIAL | STRATEGIC donor and zero rows present; TACTICAL nav-echo split (card 4.3) absent |
| tier stamp | PARTIAL | "T1-kin" / "T0" strings per family, not a registry T-value path |
| estimator named | PRESENT | paired episode-cluster bootstrap, B 2,000, seed 0 |
| leak guards | UNVERIFIED | not diffed here |

Every ABSENT or PARTIAL row is a work item and is listed in `RESULT_STRAT1.md` section 4. Whether the dumps under `read/t1/` already hold the absent quantities is UNVERIFIED.
