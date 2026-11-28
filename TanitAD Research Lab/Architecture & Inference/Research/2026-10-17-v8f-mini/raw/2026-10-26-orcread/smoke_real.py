"""WIRING smoke of orc_read.RealBackend on PRE-EXISTING T1-HEADS data (M2b + HACC post-map, seeds 0 / 1; NOT O-RECOVER data, no O-RECOVER run exists).
Purpose: prove the glue to t1hd_read is reachable from this reader's caller path (the 'built, tested, unreachable from its caller' class):
  * RealBackend.analyze_t1 on a real decisive ticks file: K1 / K2 / K3 / K5 / K6 / K7(banked) / GRID must read what T1HD_READ.json banked;
  * RealBackend.k4 on the real logged-bucket run vs the banked T0 eval (staged under the name k4_tick0_is_T0 expects);
  * f1_pair_row on (run, itself) -> exactly 0 [0, 0]; on (seed 0, seed 1) -> a real network-vs-network delta (informative of the F1 floor, decides nothing).
python smoke_real.py > out/smoke_real.out
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import orc_read as O  # noqa: E402

TREE = "E:/Projects/TanitAD-artifacts/orcread_build/tree"
PK = TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
W = "E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36"
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh"
RUN, BANK = "C:/Users/Admin/t1hd_run", "C:/Users/Admin/t1hdread/banked"
NB = 300
t_start = time.time()
T = O.load_t1hd(PK + "/2026-10-26-t1hdread/t1hd_read.py", W)
K = O.load_t1k(PK + "/2026-10-26-t1kin-hacc/t1k_read.py", W)
ns = T.parse(["--mode", "real", "--tool-root", W, "--eval-cache", SC + "/cache", "--tracks", SC + "/inputs/native_tracks.pt",
              "--eval-grid", f"{W}/{PK.split('/tree/')[1]}/eval_grid_windows.json", "--eval-join", f"{W}/{PK.split('/tree/')[1]}/join_eval141.npz",
              "--banked-dir", BANK, "--out", HERE + "/out/_unused.json"])
work = "E:/Projects/TanitAD-artifacts/orcread_build/smoke_work"
os.makedirs(work, exist_ok=True)
be = O.RealBackend(T, ns, NB, work)
out = {"LABEL": "WIRING SMOKE on pre-existing T1-HEADS data (M2b + HACC); NOT O-RECOVER data; B = %d; decides nothing" % NB}


def ent(sd, bucket, k7=None):
    tag = f"HACC_s{sd}_m10_{bucket}_heads"
    e = {"ticks": f"{RUN}/ticks/ticks_{tag}.npz", "dump_dir": f"{RUN}/dump_{tag}", "tag": tag}
    if k7:
        e["k7"] = k7
    return e


packs = {}
for sd in (0, 1):
    p = be.analyze_t1(ent(sd, "causal", {"mode": "banked", "banked_dir": BANK}), f"HACC_s{sd}_causal", sd)
    packs[sd] = p
    out[f"controls_causal_s{sd}"] = {k: v[0] for k, v in p.controls.items()}
    out[f"meta_causal_s{sd}"] = {k: (v if isinstance(v, (str, int)) else str(v)) for k, v in p.meta.items() if k != "clip_names"}
    out[f"tick0_f1_causal_s{sd}"] = p.tick0_f1
    print(f"s{sd} causal controls", out[f"controls_causal_s{sd}"], f"{time.time() - t_start:.0f}s", flush=True)
# the x1 (no banked reference) K7 mode, the one a plain fine-tuned arm will use
p_x1 = be.analyze_t1(ent(0, "causal", {"mode": "x1"}), "HACC_s0_causal_x1", 0)
out["K7_mode_x1_on_causal_s0"] = {"PASS": p_x1.controls["K7"][0], "detail": p_x1.controls["K7"][1]}
# K4 on the real logged-bucket run vs the banked T0 eval of the same network
lg = ent(0, "logged")
lp = be.analyze_t1_light(lg, "HACC_s0_logged", 0)
out["controls_logged_s0"] = {k: v[0] for k, v in lp.controls.items()}
t0e = {"json": f"{BANK}/eval_M2b_s0-sfresh-fresh.json", "npz": f"{BANK}/eval_M2b_s0-sfresh-fresh.npz"}
ok, det = be.k4(lg, packs[0], t0e, 0)
out["K4_logged_source_s0"] = {"PASS": ok, "abs_diff": det["abs_diff_vs_own_T0"], "source": det["source"]}
ok2, det2 = be.k4(None, packs[0], t0e, 0)                         # the decisive CAUSAL run's own tick 0 vs the T0 eval: the PREREG's wording, taken literally
out["K4_causal_tick0_s0_(PREREG_wording_literally)"] = {"PASS": ok2, "abs_diff": det2["abs_diff_vs_own_T0"], "source": det2["source"]}
# the pairing step on real cells
for hd in ("lat", "lon"):
    ident, _ = O.f1_pair_row(T, packs[0], packs[0], hd, NB)
    cross, _ = O.f1_pair_row(T, packs[0], packs[1], hd, NB)
    out[f"f1_{hd}_run_vs_itself"] = {k: ident[k] for k in ("delta", "lo", "hi", "separated", "n_cells_a", "n_cells_b")}
    out[f"f1_{hd}_seed0_minus_seed1_(two different networks)"] = {k: cross[k] for k in ("delta", "lo", "hi", "separated", "n_cells_a", "n_cells_b", "n_cells_matched_pairs", "f1_a", "f1_b")}
# the CLI path with the REAL backend (argparse -> t1hd_read.parse -> Ctx on the real join / cache -> manifest): a synthetic manifest whose ticks files do not exist must be a
# clean REFUSED (rc 2), reached THROUGH RealBackend's constructor
import pathlib, shutil, contextlib, io  # noqa: E402
sys.path.insert(0, HERE)
import test_orc_read as TO  # noqa: E402
root = pathlib.Path("E:/Projects/TanitAD-artifacts/orcread_build/smoke_work/cli")
shutil.rmtree(root, ignore_errors=True)
fx = TO.Fx(root, TO.World(**TO.STRONG))
err = io.StringIO()
with contextlib.redirect_stderr(err):
    rc = O.main(["--manifest", fx.save(), "--out", str(root / "ORC_READ.json"), "--tool-root", W, "--t1hd-read", PK + "/2026-10-26-t1hdread/t1hd_read.py",
                 "--t1k-read", PK + "/2026-10-26-t1kin-hacc/t1k_read.py", "--eval-cache", SC + "/cache", "--tracks", SC + "/inputs/native_tracks.pt",
                 "--eval-grid", f"{W}/{PK.split('/tree/')[1]}/eval_grid_windows.json", "--eval-join", f"{W}/{PK.split('/tree/')[1]}/join_eval141.npz", "--banked-dir", BANK,
                 "--m2b-ckpt", f"0={root}/m2b/M2b_s0/ckpt_last.pt", "--m2b-ckpt", f"1={root}/m2b/M2b_s1/ckpt_last.pt", "--work-dir", str(root / "work")])
out["cli_with_real_backend_on_a_manifest_without_ticks_files"] = {"rc": rc, "wrote_output": (root / "ORC_READ.json").exists(), "stderr_tail": err.getvalue().strip().splitlines()[-3:]}
out["wall_s"] = time.time() - t_start
print(json.dumps(out, indent=1, default=O.jenc))
