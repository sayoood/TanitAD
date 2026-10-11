#!/bin/bash
# Build the stage tree (NEW repo paths only) + UUID scan + md5 manifest.
set -u
H=/c/Users/Admin/qland/hold_clockref203
D="$H/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/data_coverage_2026-10-12/clockref_add203"
rm -rf "$H/stage"; mkdir -p "$D/raw" "$D/code"
cp "$H/raw/q4c_grid_vs_egolog_ADD203.json" "$D/q4c_grid_vs_egolog_ADD203.json"
[ -f "$H/raw/q4c_grid_vs_egolog_ADD203_slowregime.json" ] && cp "$H/raw/q4c_grid_vs_egolog_ADD203_slowregime.json" "$D/q4c_grid_vs_egolog_ADD203_slowregime.json"
for f in run_q4c.py run_q4c_slow.py mk_ctx.py run_gate_clock.sh xcheck_q4c_vs_ts33.py cmp_ctl.py read_gate_evidence.py unverified_breakdown.py probe_stationary.py cross_slow_vs_ts.py stage_build.sh; do [ -f "$H/code/$f" ] && cp "$H/code/$f" "$D/code/$f"; done
cp "$H/code/psutil.py" "$D/code/thor_shim_psutil.py"
for f in ctl.log add.log all.log slow.log gate_A.log gate_B.log gate_C.log gate_D.log gate_E.log gate_A_try1.log gate_B_try1.log q4c_CTL.json q4c_ADD203_rawmain.json probe_stationary.json xcheck_q4c_vs_ts33.json unverified_55_breakdown.json q4c_slowregime_raw.json cross_slow_vs_ts.json verdict_summary.json ctx_A.json ctx_B.json ctx_C.json ctx_D.json ctx_E.json; do
  for src in "$H/raw/$f" "$H/work/$f"; do [ -f "$src" ] && cp "$src" "$D/raw/$f" && break; done; done
for L in A B C D E; do mkdir -p "$D/raw/gate_evidence_$L"; cp "$H/raw/gate_out_$L/evidence/G-CLOCK.json" "$D/raw/gate_evidence_$L/G-CLOCK.json"; done

cp "$H/RESULT.md" "$D/RESULT.md"
cp "$H/raw/q4c_ALL4572_rawmain.json" "$H/raw/ctl_compare.txt" "$H/raw/cross_slow_vs_ts.json" "$H/raw/unverified_42_breakdown.json" "$H/raw/unverified_55_breakdown.json" "$H/raw/add_notfit_sha12.json" "$D/raw/"
cp "$H/code/cmp_all.py" "$D/code/"
mkdir -p "$D/raw/control_DO_NOT_USE_AS_REFERENCE"
cp "$H/raw/q4c_grid_vs_egolog_ADD203_slowregime_SHIFT200ms_CONTROL.json" "$D/raw/control_DO_NOT_USE_AS_REFERENCE/"
