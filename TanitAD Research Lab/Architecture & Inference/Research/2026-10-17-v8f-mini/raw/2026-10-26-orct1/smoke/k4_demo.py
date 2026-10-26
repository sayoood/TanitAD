"""SMOKE (diagnostic, not a result): which T0 eval can K4 be evaluated against?   K4 = "tick 0 of the --limit logged rollout equals the arm's own T0 eval to 1e-3".

Real orc_read.RealBackend.k4 (the reader's own code), FULL SIZE, on ONE checkpoint (the banked M2b_s0 HACC checkpoint, sha256 71de3c61...; HACC only post-maps the plan, S and T are plain M2b's):
  logged rollout : C:/Users/Admin/t1hd_run (T1-HEADS banked, bit-identical to this chain's command: smoke/equiv_compare.py)
  A. T0 eval --ages fresh, produced by THIS chain's optional stage (ORCT_T0_FRESH=1, smoke/out_t0f/t0fresh)
  B. T0 eval --ages cycle  = what box 2's `evalorc` produces (the standard eval; here the banked eval_M2b_s0 of box 2's M2b_s0 run)
python k4_demo.py > k4_demo.out
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
TREE = "E:/Projects/TanitAD-artifacts/oc/tree"
PK = TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
sys.path.insert(0, PK + "/2026-10-26-orcread")
import orc_read as O  # noqa: E402

W = "E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36"
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh"
RUN = "C:/Users/Admin/t1hd_run"
T = O.load_t1hd(PK + "/2026-10-26-t1hdread/t1hd_read.py", W)
ns = T.parse(["--mode", "real", "--tool-root", W, "--eval-cache", SC + "/cache", "--tracks", SC + "/inputs/native_tracks.pt",
              "--eval-grid", f"{W}/{PK.split('/tree/')[1]}/eval_grid_windows.json", "--eval-join", f"{W}/{PK.split('/tree/')[1]}/join_eval141.npz",
              "--banked-dir", "C:/Users/Admin/t1hdread/banked", "--out", HERE + "/_k4_unused.json"])
work = "E:/Projects/TanitAD-artifacts/orct1/k4_work"
os.makedirs(work, exist_ok=True)
be = O.RealBackend(T, ns, 300, work)
tag = "HACC_s0_m10_logged_heads"
lg = {"ticks": f"{RUN}/ticks/ticks_{tag}.npz", "dump_dir": f"{RUN}/dump_{tag}", "tag": tag}
be.analyze_t1_light(lg, "logged_s0", 0)
cases = {
    "A_fresh_T0_from_this_chain_stage": {"json": HERE + "/out_t0f/t0fresh/eval_SMOKE_M2b-orc_s0-fresh.json", "npz": HERE + "/out_t0f/t0fresh/eval_SMOKE_M2b-orc_s0-fresh.npz"},
    "B_cycle_T0_what_box2_evalorc_produces": {"json": PK + "/full/eval/eval_M2b_s0.json", "npz": PK + "/full/eval/eval_M2b_s0.npz"},
}
out = {}
for name, t0e in cases.items():
    ok, det = be.k4(lg, None, t0e, 0)
    out[name] = {"PASS": ok, "abs_diff_vs_own_T0": det["abs_diff_vs_own_T0"], "tolerance": det["tolerance"], "per_window": {k: det["per_window"][k] for k in ("max_abs_s_ckpt_now_xy_vs_T0_ckpt_m", "lat_argmax_equal", "lon_argmax_equal", "n")}}
print(json.dumps(out, indent=1, default=O.jenc))
