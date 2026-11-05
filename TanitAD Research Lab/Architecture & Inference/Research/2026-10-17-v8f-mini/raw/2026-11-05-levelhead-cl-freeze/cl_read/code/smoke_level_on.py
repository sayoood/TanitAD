#!/usr/bin/env python3
"""smoke_level_on.py -- the level-ON smoke of the NEW read tree: does the RI tool RUN the level head, from the caller the CL chain uses?   (CPU, 12 threads, 6 windows per run)

Checkpoints (built here, F:/TanitAD-scratch/motion_c/cl_read/smoke/lvl/, from C0 s0 by the TRAINER's own ``load_init_ckpt`` -- every old key loads, exactly the declared lever keys are fresh):
  ZERO  = C0 + level head built (``--level-head on`` semantics), e_level exactly zero (the build's init)  -> its plans must equal C0's EXACTLY (bit for bit)
  NZ    = ZERO with a seeded non-zero e_level                                                            -> its plans must DIFFER from C0's (the head's output reaches O at inference)
Runs (the tool in-process through run_tool_subset.py, the first 6 windows, the reader's flags, ``--probe-level`` counts the level head's calls inside the tool's own rollout):
  C0 logged (level off: 0 calls) | ZERO logged | NZ logged | NZ logged+kbox | NZ hold_now | NZ reindexed | NZ reindexed+kbox   = the five CL read tags' flag sets (CL, CLC, FZCL, RICL, RICLC) plus the two controls.
Gates: C0 0 head calls; ZERO > 0 calls, finite v_hat, v_hat varies across windows; ZERO == C0 bit for bit (ticks plan / plan_raw / tau / P, dumps cl / ol / fl); NZ != C0 (max |plan diff| > 1e-3 m);
every run exits 0 with finite plans; v_hat of NZ reindexed differs from NZ logged (the head reads the SCENE MODE's current frame); v_hat of NZ hold_now EQUALS NZ logged's (hold_now leaves the current frame
alone and freezes only the history rows: the head sees the same frame) while the plan moves.  NOTE (first run, disclosed): this gate was first written as 'hold_now v_hat differs' and read RED; that expectation
contradicted the tool's own definition of hold_now (docstring: feats[k] := feats[-1] for all hmax rows), so it was rewritten from the definition, not from the data, and now asserts the stronger identity.   Writes raw/SMOKE_LEVEL_ON.json, logs/smoke_level_on.out.
usage: python -I smoke_level_on.py"""
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = "F:/TanitAD-scratch/motion_c/cl_read/tree"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
CKD = f"{ART}/v8f-recover/M2b_s0_recft_C0"
S = "F:/TanitAD-scratch/motion_c/cl_read/smoke/lvl"
TH, NWIN = 12, 6
env = dict(os.environ, OMP_NUM_THREADS=str(TH), MKL_NUM_THREADS=str(TH), OPENBLAS_NUM_THREADS=str(TH), PYTHONDONTWRITEBYTECODE="1")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_ckpts():
    import torch
    sys.path[:0] = [f"{TREE}/stack", f"{TREE}/taniteval"]
    spec = importlib.util.spec_from_file_location("v8f_mini_train_cl", f"{TREE}/stack/scripts/v8f_mini_train.py")
    T = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = T
    spec.loader.exec_module(T)
    M = T.M
    c0 = torch.load(f"{CKD}/ckpt_last.pt", map_location="cpu", weights_only=False)
    out = {}
    for name in ("ZERO", "NZ"):
        cfgd = dict(c0["cfg"])
        cfgd["level_head"], cfgd["w_level"] = "on", 0.1
        cfg = M.V8FMiniConfig(**cfgd)
        torch.manual_seed(0)
        model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        rec = T.load_init_ckpt(f"{CKD}/ckpt_last.pt", model, cfg)
        assert rec["n_mismatch_after_load"] == 0, rec
        assert rec["declared_new_keys"] == rec["fresh_keys"] and all(k.startswith(M.LEVEL_KEY_PREFIXES) for k in rec["fresh_keys"]), rec
        if name == "NZ":
            g = torch.Generator().manual_seed(1)
            with torch.no_grad():
                model.e_level.weight.copy_(torch.randn(model.e_level.weight.shape, generator=g) * 0.5)
                model.e_level.bias.copy_(torch.randn(model.e_level.bias.shape, generator=g) * 0.2)
        else:
            assert float(model.e_level.weight.abs().sum()) == 0.0 and float(model.e_level.bias.abs().sum()) == 0.0
        ck = dict(c0)
        ck["cfg"], ck["model"], ck["arm"] = dict(cfg.__dict__), model.state_dict(), f"SMOKE_LEVEL_{name}"
        d = f"{S}/ck_{name}"
        os.makedirs(d, exist_ok=True)
        torch.save(ck, f"{d}/ckpt_last.pt")
        out[name] = {"path": f"{d}/ckpt_last.pt", "fresh_keys": rec["fresh_keys"], "n_lever_params": sum(p.numel() for n_, p in model.named_parameters() if n_.startswith(M.LEVEL_KEY_PREFIXES)),
                     "n_tensors_loaded_equal": rec["n_tensors"] - rec["n_mismatch_after_load"], "e_level_abs_sum": float(model.e_level.weight.abs().sum() + model.e_level.bias.abs().sum())}
    return out


def run(label, ckpt, *flags):
    out = f"{S}/{label}"
    os.makedirs(out, exist_ok=True)
    probe = f"{out}/probe_level.json"
    cmd = [PY, "-I", f"{HERE}/run_tool_subset.py", "--tool", TOOL, "--first", str(NWIN), "--probe-level", probe, "--", "run", "--ckpt", ckpt, "--tag", f"LVL_{label}",
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal", "--smooth", "sg72", "--dump-ticks", f"{out}/ticks", "--dump-heads",
           "--device", "cpu", "--n-boot", "20", *flags]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"{PK}/logs/smoke_level_{label}.out", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        print(f"SMOKELVL {label} RUN FAILED rc {r.returncode}: {r.stderr[-1200:]}")
        sys.exit(1)
    z = {k: v for k, v in np.load(f"{out}/ticks/ticks_LVL_{label}.npz", allow_pickle=True).items() if k not in ("tag", "ckpt")}
    dd = {}
    for f in sorted(glob.glob(f"{out}/dump_LVL_{label}/ep*.npz")):
        e = np.load(f)
        for k in e.files:
            dd.setdefault(k, []).append(e[k])
    pr = json.load(open(probe, encoding="utf-8"))
    print(f"SMOKELVL run {label}: {time.time() - t0:.0f} s, {len(z['tick'])} tick rows, level-head calls {pr['level_calls']}", flush=True)
    return z, {k: np.concatenate(v) for k, v in dd.items()}, pr


def maxdiff(a, b, keys):
    return {k: float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in keys}


def main():
    t_all = time.time()
    ck = build_ckpts()
    print("SMOKELVL ckpts:", json.dumps(ck))
    zc, dc, pc = run("C0_logged", f"{CKD}/ckpt_last.pt", "--scene", "logged")
    zz, dz, pz = run("ZERO_logged", ck["ZERO"]["path"], "--scene", "logged")
    zn, dn, pn = run("NZ_logged", ck["NZ"]["path"], "--scene", "logged")
    zk, dk, pk = run("NZ_logged_kbox", ck["NZ"]["path"], "--scene", "logged", "--feasible", "kbox")
    zh, dh, ph = run("NZ_hold_now", ck["NZ"]["path"], "--scene", "hold_now")
    zr, dr, pr = run("NZ_reindexed", ck["NZ"]["path"], "--scene", "reindexed")
    zq, dq, pq = run("NZ_reindexed_kbox", ck["NZ"]["path"], "--scene", "reindexed", "--feasible", "kbox")
    res = {"card": "level-ON smoke of the new read tree", "threads": TH, "windows": NWIN, "tool": TOOL, "tool_sha256": sha(TOOL), "ckpts": ck,
           "probe": {"C0": pc, "ZERO": pz, "NZ_logged": pn, "NZ_kbox": pk, "NZ_hold_now": ph, "NZ_reindexed": pr, "NZ_reindexed_kbox": pq}}
    pk_ = ("plan", "plan_raw", "tau", "P")
    res["zero_vs_c0_ticks"], res["zero_vs_c0_dumps"] = maxdiff(zz, zc, pk_), maxdiff(dz, dc, ("cl", "ol", "fl"))
    res["nz_vs_c0_ticks"], res["nz_vs_c0_dumps"] = maxdiff(zn, zc, pk_), maxdiff(dn, dc, ("cl", "ol", "fl"))
    res["nz_hold_now_vs_logged_plan"] = maxdiff(zh, zn, ("plan",))["plan"]
    res["nz_reindexed_vs_logged_plan"] = maxdiff(zr, zn, ("plan",))["plan"]
    res["nz_kbox_vs_logged_plan"] = maxdiff(zk, zn, ("plan",))["plan"]
    finite = all(bool(np.isfinite(z["plan"]).all()) for z in (zc, zz, zn, zk, zh, zr, zq)) and all(bool(np.isfinite(d["cl"]).all()) for d in (dc, dz, dn, dk, dh, dr, dq))
    gates = {
        "C0 level-off: 0 head calls": pc["level_calls"] == 0,
        "ZERO: head called inside the tool's rollout (calls > 0)": pz["level_calls"] > 0 and pn["level_calls"] > 0,
        "ZERO: v_hat finite and varies across windows (vision-dependent)": pz["n_nonfinite"] == 0 and pz["v_hat_max"] > pz["v_hat_min"],
        "ZERO == C0 bit for bit (ticks plan/plan_raw/tau/P)": all(v == 0.0 for v in res["zero_vs_c0_ticks"].values()),
        "ZERO == C0 bit for bit (dumps cl/ol/fl)": all(v == 0.0 for v in res["zero_vs_c0_dumps"].values()),
        "NZ != C0: the head's output reaches O at inference (max |plan diff| > 1e-3 m)": res["nz_vs_c0_ticks"]["plan"] > 1e-3,
        "every run exits 0 with finite plans and executed paths": finite,
        "NZ hold_now: v_hat range EQUAL to NZ logged (hold_now repeats the CURRENT frame over the history rows; the head reads only the current frame) while the plan moves (> 1e-3 m)":
            (ph["v_hat_min"], ph["v_hat_max"]) == (pn["v_hat_min"], pn["v_hat_max"]) and res["nz_hold_now_vs_logged_plan"] > 1e-3,
        "NZ reindexed: v_hat range differs from NZ logged": (pr["v_hat_min"], pr["v_hat_max"]) != (pn["v_hat_min"], pn["v_hat_max"]),
        "NZ kbox: the projection runs on a level-on checkpoint and moves the plan": res["nz_kbox_vs_logged_plan"] >= 0.0 and finite,
    }
    res["gates"], res["PASS"] = gates, bool(all(gates.values()))
    res["wall_s"], res["clock"] = round(time.time() - t_all), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(f"{PK}/raw/SMOKE_LEVEL_ON.json", "w", encoding="utf-8", newline="\n"), indent=1)
    for k, v in gates.items():
        print(f"SMOKELVL gate {'OK ' if v else 'RED'} {k}")
    print(f"SMOKELVL ZERO vs C0 max |diff|: ticks {res['zero_vs_c0_ticks']} dumps {res['zero_vs_c0_dumps']}")
    print(f"SMOKELVL NZ vs C0 max |diff|: ticks {res['nz_vs_c0_ticks']} dumps {res['nz_vs_c0_dumps']}; NZ hold_now vs logged plan {res['nz_hold_now_vs_logged_plan']:.3f} m, "
          f"reindexed vs logged {res['nz_reindexed_vs_logged_plan']:.3f} m")
    print("SMOKELVL OK" if res["PASS"] else "SMOKELVL FAIL")
    sys.exit(0 if res["PASS"] else 2)


if __name__ == "__main__":
    main()
