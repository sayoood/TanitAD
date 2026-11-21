# P6 implementation: HANDOFF (what is NOT done), 2026-10-11

State: SPEC_P6 section 2 deltas D0-D6, D8-D10 are built, reachable from `train.run`, and covered by `refe/test_p6_levers.py` (mutations RED). Details, file:line and evidence: `eval/P6_IMPLEMENTATION.md`. Everything is STAGED, nothing committed.

## Remaining, in priority order

1. **`refe/p6_pod.sh` (SPEC D10) is NOT written.** The P2 chain (`p2r_pod.sh`) is the pattern: define `ARGS` once, run `launch_gate.py run --profile refe_p6 --args "$ARGS" ... --gate-lab-json --gate-mem-metrics --gate-egr-json`, and call `token_ok` (`launch_gate.py verify`) before EVERY `train.py` launch. Add the mutation set that `test_p2r_chain.py` uses for P2 (delete the check before train.py, make `token_ok` constant, define ARGS after the gate...). The profile `refe_p6` already carries the gate's side.
2. **P6 smoke bank.** `gate_fixture.carve_gate_bank` carves from on-policy sets and does not fit `--labeller online`. Write `carve_p6_gate_bank(targets, metric_cache, dst, n)` (rows whose `<log>/<token>.pkl.xz` exists) and make `launch_gate.derived_argvs` use it for `refe_p6`; the SMOKE argv also needs `--metric-cache` pointing at a cache that covers the carved rows.
3. **G-EGR: first `ckpt_last.pt` push size.** `gate_checks.check_egr(egr_json, ckpt_sizes=(local, remote))` is implemented and tested (`egr_wrong_size` arm); `run_checks` does not yet pass `ckpt_sizes` (needs the pod chain to write the local size and the listing size to a JSON, as `hf_egress.push` already records `sizes`).
4. **PI ruling P6-Y** -> set `refe_p6.required_flags["--yaw-wrap"]` in `launch_gate.py` to `"on"` or `"off"` (currently `None` = must be named, value open).
5. **D7 lane-rank twins.** `build_metric_cache.py build --lane-rank K` needs the DriveRL fork's `route_rank.apply` (pod). Run `verify` on one lane-rank token: PDM-Closed's first centreline lane must equal the twin's lane; if PDM-Closed does not honour the patch, switch the twins to the declared fallback (trajectory loss only; the labeller already masks a twin whose parts are missing).
6. **D6 command source.** Decide between running `annotate_commands.py` on the pod (needs a `{log/token: driving_command}` table from the navtrain metadata) and accepting the 0.90 goal rule for all rows. Re-measure the rule on navtrain goals with `refe/measure_command_rule.py`-style code.
7. **Pod measurements to bank:** G-LIVE-LAB at 12+ workers (`diag_p6_labeller_identity.py`), G-LIVE-MEM over >= 200 micro-batches (peak <= 44 GB, else micro-batch 2 x accum 128), a ViT-L real-data smoke with `--compile --amp bf16 --tf32`.
8. **Regenerate `SHIP_MANIFEST.json`** after landing (`launch_gate.py ship-manifest --commit <40-hex>`); the shipped set now includes the new `.py` files.
9. **Planner/runner (pod):** the NAVSIM runner must set `planner.driving_command` per scenario for a command-mode checkpoint; `REFePlanner.infer` in command mode has no dev-box end-to-end test. The owner of the SFT-4 / Amendment 8-9 planner hunks must `git add refe/planner.py` to complete the staging (see `P6_IMPLEMENTATION.md` section 8).

## Do not
* Re-run the ViT-L training on the dev box (RTX 4060, 8 GB).
* Edit `P6_GO_SIGNAL.md` / `PREREG_P6.md`; proposed edits are in `P6_IMPLEMENTATION.md` section 6.
* Treat `--label-backend synthetic` as anything but a control: the profile forbids it.
