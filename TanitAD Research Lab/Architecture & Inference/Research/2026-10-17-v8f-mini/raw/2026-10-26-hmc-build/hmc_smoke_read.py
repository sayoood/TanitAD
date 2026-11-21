"""SMOKE read (NOT outcome-bearing; every printed line is labelled SMOKE): C-1 and C-3 on the 24-window tick dumps of MC-H s0, with the STEEP and HACC regression arms.
python hmc_smoke_read.py  ->  smoke/SMOKE_CONTROLS.json"""
import json, os, sys, time
os.environ["HMC_READ_ROOT"] = "C:/Users/Admin/hlat/frozen"                  # the smoke tool's tree (taniteval + tanitad from ONE tree)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Users/Admin/hmc")
import numpy as np
import hmc_lib as L
import hmc_controls as C
SM = "C:/Users/Admin/hmc/smoke"
t0 = time.time()
sfk, st, T1, D = C.bind_statekin("C:/Users/Admin/hlat/frozen")
Z = {a: C.load_ticks(f"{SM}/ticks/ticks_SMOKE_{a}_s0_m10.npz") for a in ("MCH", "STEEP", "HACC")}
W = {"MCH": C.table_from_ckpt(f"{L.ART}/M2b_s0_hmc_MCH.pt"), "STEEP": C.table_from_ckpt(f"{L.ART}/M2b_s0_hmc_STEEP.pt"), "HACC": C.table_from_ckpt(f"{L.HACC_ART}/M2b_s0_hacc.pt")}
out = {"LABEL": "SMOKE: plumbing only, 24 windows, logged bucket, hlat/frozen tool. Not outcome-bearing.", "ticks": {a: int(len(z["P"])) for a, z in Z.items()}}
cmch = float(W["MCH"][0, 0])
assert np.allclose(W["MCH"][:12, 0], cmch) and cmch == np.float32(0.200220178988411)
# ---- C-1
c1 = {"MCH_own_table": C.c1_reach(Z["MCH"], W["MCH"]),
      "REGRESSION_HACC_table_on_MCH_ticks": C.c1_reach(Z["MCH"], W["HACC"]),
      "REGRESSION_HACC_ticks_vs_constant_c": C.c1_reach(Z["HACC"], W["MCH"]),
      "STEEP_own_table_beside": C.c1_reach(Z["STEEP"], W["STEEP"]),
      "REGRESSION_STEEP_ticks_vs_constant_c": C.c1_reach(Z["STEEP"], W["MCH"])}
out["C-1"] = c1
for k, v in c1.items():
    print(f"SMOKE C-1 {k:40s} max|d| {v['max_abs_d_m']:.3e} m  (bar 1e-6: {'PASS' if v['PASS'] else 'FAIL'})  max|d|/ulp32 {v['max_d_over_ulp32_of_plan']:.2f}  max|plan_along| {v['max_abs_plan_along_m']:.1f} m  "
          f"over-bar {v['frac_tick_knots_over_bar']:.3f}  f32 lerp replica max {v['float32_lerp_replica_max_abs_vs_dumped_plan_m']:.1e} m = {v['float32_lerp_replica_max_over_ulp32_of_plan']:.1f} ulp")
# ---- C-3
t1 = time.time()
r_mch = C.ramp_term(Z["MCH"], W["MCH"], st)
c3 = {"MCH_ticks_MCH_table": C.c3_summary(r_mch)}
print(f"SMOKE C-3 ramp term on MC-H ticks ({len(r_mch)} ticks, {time.time()-t1:.1f}s): ", c3["MCH_ticks_MCH_table"])
for nm, z, w in (("REGRESSION_MCH_ticks_HACC_table", Z["MCH"], W["HACC"]), ("REGRESSION_MCH_ticks_STEEP_table", Z["MCH"], W["STEEP"]),
                 ("REGRESSION_STEEP_ticks_STEEP_table", Z["STEEP"], W["STEEP"]), ("REGRESSION_HACC_ticks_HACC_table", Z["HACC"], W["HACC"])):
    c3[nm] = C.c3_summary(C.ramp_term(z, w, st))
    print(f"SMOKE C-3 {nm:38s} max|a0| {c3[nm]['max_abs_a0_mps2']:.4f} m/s2  median a0 {c3[nm]['median_a0_mps2']:+.4f}  frac nonzero {c3[nm]['frac_ticks_nonzero_a0']:.2f}  is_zero {c3[nm]['is_zero']}")
t2 = time.time()
c3["HACC_TRAIN_known_value_s0"] = C.c3_train_known(st, 0, W["HACC"])
print(f"SMOKE C-3 HACC TRAIN known value s0 ({time.time()-t2:.0f}s):", c3["HACC_TRAIN_known_value_s0"])
out["C-3"] = c3
# ---- floor (C-5 style, 24 windows): the P10 loop of the smoke equals the banked floor at the same grid windows
bf = L.load_npz("C:/Users/Admin/hmc/inputs/banked_causal_f64/dumps_HACC_s0_m10_causal_f64.npz")
idx = np.linspace(0, 2398, 24).round().astype(int)
d = L.load_run_dumps(f"{SM}/run", "SMOKE_MCH_s0_m10")
out["floor_vs_banked_P10_loop_m10"] = {"max_abs_m": float(np.abs(d["fl"] - bf["fl"][idx]).max()), "n_windows": int(len(idx)), "ws_equal": bool((d["ws"] == bf["ws"][idx]).all())}
print("SMOKE floor (P10 loop, m10) vs banked HACC dump's fl at the 24 grid windows:", out["floor_vs_banked_P10_loop_m10"])
out["wall_s"] = time.time() - t0
json.dump(out, open(f"{SM}/SMOKE_CONTROLS.json", "w"), indent=1)
