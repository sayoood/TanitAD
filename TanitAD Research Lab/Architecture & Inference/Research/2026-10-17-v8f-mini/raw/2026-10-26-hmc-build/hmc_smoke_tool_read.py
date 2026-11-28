"""SMOKE read on the MM's tool (81aa3e36): C-1, C-3 on its causal tick dumps; and tool-drift checks vs the hlat/frozen smoke on the same 24 windows (NOT outcome-bearing)."""
import json, os, sys
TREE = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc/tool_81aa3e36"
os.environ["HMC_READ_ROOT"] = TREE; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/Admin/hmc")
import numpy as np
import hmc_lib as L, hmc_controls as C
SM = "C:/Users/Admin/hmc/smoke"
sfk, st, T1, D = C.bind_statekin(TREE)
zt = C.load_ticks(f"{SM}/tool/ticks/ticks_SMOKE_TOOL_MCH_s0_m10.npz"); zs = C.load_ticks(f"{SM}/tool/ticks/ticks_SMOKE_TOOL_STEEP_s0_m10.npz")
zh = C.load_ticks(f"{SM}/ticks/ticks_SMOKE_MCH_s0_m10.npz")                       # hlat/frozen tool, logged bucket
W = {"MCH": C.table_from_ckpt(f"{L.ART}/M2b_s0_hmc_MCH.pt"), "STEEP": C.table_from_ckpt(f"{L.ART}/M2b_s0_hmc_STEEP.pt"), "HACC": C.table_from_ckpt(f"{L.HACC_ART}/M2b_s0_hacc.pt")}
out = {"LABEL": "SMOKE on tool 81aa3e36, 24 windows, --limit causal --dump-ticks; plumbing only"}
out["tick_keys_equal_hlat"] = sorted(zt) == sorted(zh)
out["tick_keys_only_in_one"] = sorted(set(zt) ^ set(zh))
out["n_ticks"] = int(len(zt["P"]))
out["tick0_P_identical_to_hlat_tool"] = bool(np.array_equal(zt["P"][zt["tick"] == 0], zh["P"][zh["tick"] == 0]))
out["fl_loop_identical_to_hlat_tool"] = None
dt_ = L.load_run_dumps(f"{SM}/tool/run", "SMOKE_TOOL_MCH_s0_m10"); dh = L.load_run_dumps(f"{SM}/run", "SMOKE_MCH_s0_m10")
out["P10_floor_loop_max_abs_vs_hlat_tool_m"] = float(np.abs(dt_["fl"] - dh["fl"]).max())
out["MCH_cl_max_abs_vs_hlat_tool_logged_bucket_m"] = float(np.abs(dt_["cl"] - dh["cl"]).max())
out["tau_tick0_differs_because_bucket_differs"] = float(np.abs(zt["tau"][zt["tick"] == 0] - zh["tau"][zh["tick"] == 0]).max())
out["C-1_MCH"] = C.c1_reach(zt, W["MCH"]); out["C-1_REGRESSION_HACC_table"] = C.c1_reach(zt, W["HACC"]); out["C-1_STEEP_own"] = C.c1_reach(zs, W["STEEP"])
for k in ("C-1_MCH", "C-1_REGRESSION_HACC_table", "C-1_STEEP_own"):
    v = out[k]; print(f"SMOKE-TOOL {k:28s} max|d| {v['max_abs_d_m']:.3e} m = {v['max_d_over_ulp32_of_plan']:.2f} ulp32, literal-1e-6 {v['PASS']}")
out["C-3_MCH"] = C.c3_summary(C.ramp_term(zt, W["MCH"], st)); out["C-3_REGRESSION_HACC_table"] = C.c3_summary(C.ramp_term(zt, W["HACC"], st)); out["C-3_REGRESSION_STEEP_own"] = C.c3_summary(C.ramp_term(zs, W["STEEP"], st))
for k in ("C-3_MCH", "C-3_REGRESSION_HACC_table", "C-3_REGRESSION_STEEP_own"):
    v = out[k]; print(f"SMOKE-TOOL {k:28s} max|a0| {v['max_abs_a0_mps2']:.4g} frac nonzero {v['frac_ticks_nonzero_a0']:.2f} is_zero {v['is_zero']}")
hdr = json.load(open(f"{SM}/tool/run/t1v8f_SMOKE_TOOL_MCH_s0_m10.json"))["t1v8f"]
out["header_limit"] = hdr["limit"]; out["header_has_causal_bucket"] = "causal_bucket" in hdr
print({k: out[k] for k in ("tick_keys_equal_hlat", "tick_keys_only_in_one", "n_ticks", "tick0_P_identical_to_hlat_tool", "P10_floor_loop_max_abs_vs_hlat_tool_m", "MCH_cl_max_abs_vs_hlat_tool_logged_bucket_m", "header_limit", "header_has_causal_bucket")})
json.dump(out, open(f"{SM}/tool/SMOKE_TOOL_CONTROLS.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
