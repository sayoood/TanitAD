#!/usr/bin/env python3
"""YAX CPU tests: gates + deliberate-regression arms (each must be RED), build/identity gates, G8-mini. Writes test_ax.out lines `PASS|FAIL <name> ...`; last line ZZAXTEST_DONE n_fail=<k>."""
import json, os, subprocess, sys
from pathlib import Path
import torch
H = Path(__file__).resolve().parent
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
C, M = str(H.parent / "YF2" / "cache_u400_4k"), str(H.parent / "YF2" / "manifest_yaw_4k.json")
U = H / "unit"
U.mkdir(exist_ok=True)
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8", PYTHONIOENCODING="utf-8")
SMALL = ["--steps", "3", "--lr", "5e-5", "--max-frames", "400", "--batch", "16", "--accum", "2", "--gate-frames", "64", "--val-every", "2"]
fails, out = [], open(H / "test_ax.out", "w", encoding="utf-8")
def rec(ok, name, note=""):
    out.write(f"{'PASS' if ok else 'FAIL'} {name} {note}\n"); out.flush()
    if not ok: fails.append(name)
def run(args):
    r = subprocess.run([PY] + [str(x) for x in args], env=ENV, capture_output=True, text=True, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr
def ft(name, pol, extra=(), seed=0, small=SMALL):
    return run([H / "ax_ft.py", "--cache", C, "--manifest", M, "--ax-policy", pol, "--seed", seed, "--out", U / name] + list(small) + list(extra))
for pol in ("zero", "real"):
    rc, t = ft(f"{pol}_ok", pol); rec(rc == 0 and "ZZAXFT_OK" in t, f"train_{pol}_gates_green", f"rc={rc}")
for name, pol, reg, gate in (("red_fed_in_zero", "zero", "ax_fed_in_zero", "GA_FEED"), ("red_not_fed", "real", "ax_not_fed", "GB_STEP0"), ("red_keep_col", "zero", "keep_ax_col", "GA_COL")):
    rc, t = ft(name, pol, ["--regress", reg, "--gates-only"]); g = json.load(open(U / name / "gates.json"))
    rec(rc == 3 and "ZZAXFT_GATE_RED" in t and g[gate] is False, f"RED_arm_{reg}", f"rc={rc} {gate}={g[gate]}")
# G8-mini: two runs of the same arm+seed
tiny = ["--steps", "5", "--lr", "5e-5", "--max-frames", "400", "--batch", "16", "--accum", "2", "--gate-frames", "64", "--val-every", "5"]
ft("g8a", "zero", small=tiny); ft("g8b", "zero", small=tiny)
a, b = torch.load(U / "g8a/final.pt", weights_only=False), torch.load(U / "g8b/final.pt", weights_only=False)
rec(all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]), "G8_mini_determinism", f"{len(a['state'])} tensors")
rec(bool((a["state"]["ego_enc.0.weight"][:, 2] == 0).all()) and len(a["changed_keys"]) > 0, "AXOFF_col2_zero_and_params_moved", f"changed={len(a['changed_keys'])}")
# build + identity
for pol in ("zero", "real"):
    ck = U / f"{pol}.pt"
    rc, t = run([H / "ax_ckpt_tools.py", "build", "--final", U / f"{pol}_ok/final.pt", "--out", ck]); rec(rc == 0 and "ZZAXCKPT_OK" in t, f"build_{pol}", f"rc={rc}")
    rc, t = run([H / "ax_ckpt_tools.py", "identity", "--ckpt", ck, "--final", U / f"{pol}_ok/final.pt", "--out", U / f"id_{pol}.json", "--frames", "128"]); rec(rc == 0 and "ZZAXID_OK" in t, f"identity_{pol}", f"rc={rc}")
# RED: tampered final.pt (a scorer tensor sneaks in) -> K1 red
f = torch.load(U / "zero_ok/final.pt", weights_only=False)
base = torch.load("E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt", map_location="cpu", weights_only=False)
sk = next(k for k in base["model"] if not k.startswith(("ego_enc.", "queries", "dec.", "traj_head.")) and base["model"][k].is_floating_point())
f["state"][sk] = base["model"][sk] + 1.0
torch.save(f, U / "tampered_final.pt")
rc, t = run([H / "ax_ckpt_tools.py", "build", "--final", U / "tampered_final.pt", "--out", U / "tampered.pt"]); rec(rc == 3 and "ZZAXCKPT_GATE_RED" in t and not (U / "tampered.pt").exists(), "RED_arm_build_tampered_state", f"rc={rc} key={sk}")
# RED: a zero-policy checkpoint that kept u400's column 2 -> GA_GAIN red
sd = torch.load(U / "zero.pt", map_location="cpu", weights_only=False)
sd["model"]["ego_enc.0.weight"] = sd["model"]["ego_enc.0.weight"].clone(); sd["model"]["ego_enc.0.weight"][:, 2] = base["model"]["ego_enc.0.weight"][:, 2]
torch.save(sd, U / "zero_keepcol.pt")
fz = torch.load(U / "zero_ok/final.pt", weights_only=False); fz["state"]["ego_enc.0.weight"] = sd["model"]["ego_enc.0.weight"].clone(); torch.save(fz, U / "zero_keepcol_final.pt")
rc, t = run([H / "ax_ckpt_tools.py", "identity", "--ckpt", U / "zero_keepcol.pt", "--final", U / "zero_keepcol_final.pt", "--out", U / "id_keepcol.json", "--frames", "128"])
g = json.load(open(U / "id_keepcol.json")); rec(rc == 3 and g["GA_GAIN"] is False and "ZZAXID_RED" in t, "RED_arm_identity_kept_u400_column2", f"rc={rc} gain_zero_exact={g.get('gain_zero_exact')}")
for p in ("zero.pt", "real.pt", "zero_keepcol.pt"):
    try: os.remove(U / p)
    except OSError: pass
out.write(f"ZZAXTEST_DONE n_fail={len(fails)} {fails}\n"); out.close()
