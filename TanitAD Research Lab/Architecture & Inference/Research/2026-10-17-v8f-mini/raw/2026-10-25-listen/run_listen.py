"""PREREG_V8F_LISTEN -- the real-data driver. CPU only (CUDA_VISIBLE_DEVICES=-1), torch capped at 8 threads, zero training.

  python run_listen.py collect --seed s0 --donor-seed 0 --out runs/s0_d0.npz        one full grid (all cells, Q-NAV, the clean pass)
  python run_listen.py collect --seed s0 --identity --out runs/s0_id.npz             control 1 (donor = self)
  python run_listen.py collect --seed s0 --regress nav_const --cells S/nav,T/nav,O/nav,O/ego --out runs/s0_navc.npz     control 3
  python run_listen.py all                                                           the whole pipeline, skipping outputs that exist
  python run_listen.py analyze                                                       controls + grid + outcome -> READ.json, LISTEN_RESULT.md

The eval context is built exactly as ``v8f_mini_eval.cmd_run`` builds it (same join, grid, kinematics, tracks, cache, dataset arguments).
All inputs are read from SFRESH's directory READ-ONLY; everything written lives under this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import listen as LS  # noqa: E402  (puts the exported tip first on sys.path)
import listen_read as LR  # noqa: E402
import v8f_mini_eval as EV  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import v8f_consistency as C  # noqa: E402

torch.set_num_threads(int(os.environ.get("LISTEN_THREADS", "8")))                           # resume agent: run with LISTEN_THREADS=6

SF = Path(os.environ.get("LISTEN_SFRESH", HERE.parent / "sfresh"))
RAWD = SF / "code" / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
PATHS = {
    "grid": RAWD / "eval_grid_windows.json", "kin": RAWD / "eval141_kin.npz", "join": RAWD / "join_eval141.npz",
    "tracks": SF / "inputs" / "native_tracks.pt", "cache": SF / "cache",
    "ckpt": {"s0": SF / "inputs" / "M2b_s0" / "ckpt_last.pt", "s1": SF / "inputs" / "M2b_s1" / "ckpt_last.pt"},
    "banked": {s: RAWD / "full" / "eval" / f"eval_M2b_{s}" for s in ("s0", "s1")},                 # + .npz / .json
    "cpu_rerun": {s: SF / "eval" / f"eval_M2b_{s}-sfresh-cycle" for s in ("s0", "s1")},
    "banked_tac": {s: HERE / "banked" / f"eval_M2b_{s}-tacshuf" for s in ("s0", "s1")},
    "banked_tacpred": {s: HERE / "banked" / f"eval_M2b_{s}-tacpred" for s in ("s0", "s1")},
}
RUNS = HERE / "runs"
TIP = "3fdb1b61"


def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(n):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ context (cmd_run's, verbatim)
def load_eval(paths=PATHS, kin_mode="yaw"):
    grid = json.load(open(paths["grid"], encoding="utf-8"))
    ej = D.load_join(str(paths["join"]))
    X = D.eval_context(grid, str(paths["kin"]), ej, kin_mode=kin_mode)
    wins = D.eval_grid_windows(str(paths["grid"]), ej)
    assert len(wins) == len(X["eid"]) == 2399 or os.environ.get("LISTEN_SYNTH"), len(wins)
    return grid, ej, X, wins


def load_model(paths, seed_tag):
    ck = torch.load(paths["ckpt"][seed_tag], map_location="cpu", weights_only=False)      # our own checkpoint (sha256 = the banked one)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck


def load_ctx(paths, seed_tag):
    model, cfg, ck = load_model(paths, seed_tag)
    grid, ej, X, wins = load_eval(paths, cfg.kin_mode)
    tracks = D.load_native_tracks(str(paths["tracks"])) if cfg.kin_mode == "yaw" else None
    store = D.ClipStore(str(paths["cache"]), "eval", lru=int(os.environ.get("LISTEN_LRU", "8")), grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    ds = D.V8FMiniWindows(store, ej, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                          window_start=cfg.window_start)
    return {"model": model, "cfg": cfg, "X": X, "ds": ds, "wins": wins}


# ------------------------------------------------------------------------------------------------ collect
def parse_cells(s):
    if not s:
        return None
    return [tuple(c.split("/")) for c in s.split(",")]


def cmd_collect(a):
    t0 = time.time()
    ctx = load_ctx(PATHS, a.seed)
    model, X, wins = ctx["model"], ctx["X"], ctx["wins"]
    if a.regress:
        model = LS.regress_copy(model, a.regress)
    N = len(wins)
    perm = np.arange(N) if a.identity else LS.donor_perm(N, X["eid"], a.donor_seed)
    assert (perm == np.arange(N)).all() == bool(a.identity)
    if not a.identity:
        assert (np.asarray(X["eid"])[perm] != np.asarray(X["eid"])).all(), "a donor from the same episode"
    cells = parse_cells(a.cells)
    last = [0.0]

    def prog(b0, n):
        if time.time() - last[0] > 20:
            print(f"  [{a.out}] {b0}/{n} windows, {time.time() - t0:.0f}s", flush=True)
            last[0] = time.time()

    with torch.no_grad():
        res = LS.collect(model, ctx["ds"], wins, X["gt"], perm, batch=a.batch, cells=cells, progress=prog)
    meta = {"prereg": "PREREG_V8F_LISTEN", "tip": TIP, "seed": a.seed, "donor_seed": None if a.identity else a.donor_seed, "identity": bool(a.identity),
            "regress": a.regress, "cells": a.cells, "ages": "cycle", "n_windows": N, "ckpt": str(PATHS["ckpt"][a.seed]),
            "ckpt_sha256": sha256(PATHS["ckpt"][a.seed]), "threads": torch.get_num_threads(), "device": "cpu", "seconds": round(time.time() - t0, 1),
            "perm_sha256": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    LR.save_run(res, a.out, meta)
    print(f"wrote {a.out} ({meta['seconds']} s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ controls
def _maxabs(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return float(np.nanmax(d)) if d.size else 0.0


def control_identity(res):
    """Control 1: donor = self => every cell's per-window arrays equal the clean arrays EXACTLY (so every delta is exactly 0.0)."""
    n_cells, bad, worst = 0, [], 0.0
    ref_of = {"S": "clean/S", "T": "clean/T", "O": "clean/O"}
    for nm, r in res.items():
        if nm.startswith("clean/"):
            continue
        ref = res[ref_of[nm.split("/")[0]] if nm[0] in "STO" and "/" in nm else "clean/O"]
        n_cells += 1
        for k, v in r.items():
            worst = max(worst, _maxabs(v, ref[k]) if np.issubdtype(np.asarray(v).dtype, np.number) else 0.0)
            if not np.array_equal(v, ref[k]):
                bad.append(f"{nm}:{k}")
    return {"n_cells": n_cells, "max_abs_diff_vs_clean": worst, "cells_not_identical": bad, "pass": (not bad) and worst == 0.0}


def control_regress(res, kind):
    """Control 3: the regressed copy reads the probed cell EXACTLY 0.0 (arrays equal ITS OWN clean arrays) while an unprobed cell is alive."""
    ref_of = {"S": "clean/S", "T": "clean/T", "O": "clean/O"}
    probed = {"nav_const": ("S/nav", "T/nav", "O/nav"), "tscene_off": ("T/scene",)}[kind]
    out = {"kind": kind, "probed": {}, "alive": {}}
    for nm, r in res.items():
        if nm.startswith("clean/"):
            continue
        ref = res[ref_of[nm[0]]]
        diff = max(_maxabs(r[k], ref[k]) for k in r)
        (out["probed"] if nm in probed else out["alive"])[nm] = diff
    out["pass"] = all(v == 0.0 for v in out["probed"].values()) and len(out["probed"]) == len(probed) and all(v > 0.0 for v in out["alive"].values())
    return out


def control_repro(seed, res, X):
    """Control 2: the clean pass reproduces the banked own-code eval (GPU, box 2) and SFRESH's CPU re-run of the same checkpoint."""
    f = res["clean/full"]
    out = {}
    for label, base in (("banked_box2", PATHS["banked"][seed]), ("cpu_rerun_sfresh", PATHS["cpu_rerun"][seed])):
        z = np.load(str(base) + ".npz")
        j = json.load(open(str(base) + ".json", encoding="utf-8"))["raw"]
        mine_pe = C.checkpoint_errors(res["clean/S"]["ck_now"], X["rc"], X["rc_ok"])["pos_err_m"]
        d = {"max_abs_diff": {"traj30": _maxabs(f["traj30"], z["pred"]), "ckpt_tick_frame": _maxabs(f["ckpt"], z["ckpt"]),
                              "goal_prob": _maxabs(f["goal_prob"], z["goal_prob"]), "lat_logits": _maxabs(f["lat_logits"], z["lat_logits"]),
                              "lon_logits": _maxabs(f["lon_logits"], z["lon_logits"]), "cons_pred": _maxabs(f["cons_pred"], z["cons_pred"]),
                              "ck_pos_err_m": _maxabs(mine_pe, z["ck_pos_err"]), "ade6": _maxabs(res["clean/O"]["ade6"], z["ade6"])}}
        d["argmax_lat_flips"] = int((f["lat_logits"].argmax(-1) != z["lat_logits"].argmax(-1)).sum())
        d["argmax_lon_flips"] = int((f["lon_logits"].argmax(-1) != z["lon_logits"].argmax(-1)).sum())
        m = {"S_pos_err_50m": (float(np.nanmean(mine_pe[X["rc_ok"][:, 1], 1])), float(j["strategic"]["pos_err_m@50"]["mean"])),
             "ADE6": (float(res["clean/O"]["ade6"].mean()), float(j["traj"]["ade6_m"]["mean"])),
             "T_lat_macro_F1": (float(LR.level("T-lat", LR.vec("T-lat", res["clean/T"], X), X)), float(j["tactical"]["lat"]["macro_f1"])),
             "T_lon_macro_F1": (float(LR.level("T-lon", LR.vec("T-lon", res["clean/T"], X), X)), float(j["tactical"]["lon"]["macro_f1"]))}
        d["metrics_mine_vs_banked"] = {k: {"mine": a, "banked": b, "abs_diff": abs(a - b)} for k, (a, b) in m.items()}
        d["max_metric_diff"] = max(v["abs_diff"] for v in d["metrics_mine_vs_banked"].values())
        d["pass_1e-3"] = bool(d["max_metric_diff"] <= 1e-3)
        out[label] = d
    out["pass"] = all(v["pass_1e-3"] for v in out.values())
    return out


def control_banked_shuffle(seed, res_d1, X):
    """Control 5 (descriptive) + implementation check: O x T code at donor seed 1 IS ``--tac shuffle`` (donor seed 1, same ages)."""
    z = np.load(str(PATHS["banked_tac"][seed]) + ".npz")
    zp = np.load(str(PATHS["banked_tacpred"][seed]) + ".npz")
    mine = res_d1["O/code"]["ade6"]
    return {"seed": seed, "max_abs_diff_per_window_ade6_vs_banked_tacshuf": _maxabs(mine, z["ade6"]),
            "mine_O_x_code_ade6_mean": float(mine.mean()), "banked_tacshuf_ade6_mean": float(z["ade6"].mean()),
            "banked_tacpred_ade6_mean": float(zp["ade6"].mean()), "banked_delta_m": float(z["ade6"].mean() - zp["ade6"].mean()),
            "mine_delta_m": float(mine.mean() - res_d1["clean/O"]["ade6"].mean())}


# ------------------------------------------------------------------------------------------------ analyze
RUN_FILES = {"s0": "s0_d0.npz", "s1": "s1_d0.npz", "s0_d1": "s0_d1.npz", "s0_id": "s0_id.npz", "s1_id": "s1_id.npz",
             "s0_navc": "s0_navc.npz", "s0_tsoff": "s0_tsoff.npz", "s1_navc": "s1_navc.npz", "s1_tsoff": "s1_tsoff.npz"}


def cmd_analyze(a):
    t0 = time.time()
    _, _, X, _ = load_eval(PATHS, "yaw")
    runs, metas = {}, {}
    for k, fn in RUN_FILES.items():
        p = RUNS / fn
        if p.exists():
            runs[k], metas[k] = LR.load_run(p)
    need = ("s0", "s1", "s0_d1")
    missing = [k for k in need if k not in runs]
    if missing:
        print("missing runs:", missing)
        return 2
    ctrl = {"identity": {s: control_identity(runs[f"{s}_id"]) for s in ("s0", "s1") if f"{s}_id" in runs},
            "reproduction": {s: control_repro(s, runs[s], X) for s in ("s0", "s1")},
            "structural_zero": {}, "banked_shuffle_consistency": control_banked_shuffle("s0", runs["s0_d1"], X)}
    for s in ("s0", "s1"):
        for kind, key in (("nav_const", f"{s}_navc"), ("tscene_off", f"{s}_tsoff")):
            if key in runs:
                ctrl["structural_zero"][f"{s}/{kind}"] = control_regress(runs[key], kind)
    ctrl["banked_shuffle_consistency_descriptive"] = {
        s: {"mine_O_x_code_delta_m_donor0": float(runs[s]["O/code"]["ade6"].mean() - runs[s]["clean/O"]["ade6"].mean())} for s in ("s0", "s1")}
    ctrl["all_pass"] = bool(all(c["pass"] for c in ctrl["identity"].values()) and len(ctrl["identity"]) == 2
                            and all(c["pass"] for c in ctrl["reproduction"].values())
                            and all(c["pass"] for c in ctrl["structural_zero"].values()) and len(ctrl["structural_zero"]) == 4
                            and ctrl["banked_shuffle_consistency"]["max_abs_diff_per_window_ade6_vs_banked_tacshuf"] < 1e-2)
    recs = {k: LR.run_record(runs[k], X, LR.B_BOOT) for k in need}
    an = LR.analyse(recs)
    read = {"prereg": "PREREG_V8F_LISTEN", "tip": TIP, "executed": time.strftime("%Y-%m-%d"), "tier": "T1-class open loop, DIAGNOSTIC (donor inputs are interventions, not driving)",
            "estimator": "paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/taniteval/ci.py); answers 'another draw of EPISODES', not 'another training run' (two seeds side by side) and not 'another inference run' (deterministic)",
            "evidence_class": "MEASURED (ours); artifacts in this directory (runs/*.npz)", "controls": ctrl, "analysis": an,
            "grid": recs, "run_meta": metas, "readings_fixed_before_data": LR.__doc__.split("READINGS OF THE PREREG TEXT")[1].strip(),
            "seconds_analysis": round(time.time() - t0, 1)}
    json.dump(read, open(HERE / "READ.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("outcome:", an["outcome"]["headline"], "| Q-NAV:", an["qnav"]["headline"], "| controls all pass:", ctrl["all_pass"])
    return 0


def cmd_all(a):
    RUNS.mkdir(exist_ok=True)
    full = lambda seed, d: dict(seed=seed, donor_seed=d)                                           # noqa: E731
    plan = [("s0_d0.npz", full("s0", 0)), ("s1_d0.npz", full("s1", 0)), ("s0_d1.npz", full("s0", 1)),
            ("s0_id.npz", dict(seed="s0", identity=True)), ("s1_id.npz", dict(seed="s1", identity=True))]
    for s in ("s0", "s1"):
        plan.append((f"{s}_navc.npz", dict(seed=s, regress="nav_const", cells="S/nav,T/nav,O/nav,O/ego,T/ego")))
        plan.append((f"{s}_tsoff.npz", dict(seed=s, regress="tscene_off", cells="T/scene,T/ego,O/scene")))
    for fn, kw in plan:
        out = RUNS / fn
        if out.exists():
            print("have", fn, flush=True)
            continue
        cmd_collect(argparse.Namespace(**{"donor_seed": 0, "identity": False, "regress": None, "cells": None, "batch": a.batch, **kw, "out": str(out)}))
        if fn == "s0_d0.npz":                                              # fail fast: a mismatch above 1e-3 stops the read
            res, _ = LR.load_run(out)
            _, _, X, _ = load_eval(PATHS, "yaw")
            c = control_repro("s0", res, X)
            json.dump(c, open(HERE / "repro_s0.json", "w"), indent=1)
            print("REPRODUCTION s0:", c["pass"], {k: v["max_metric_diff"] for k, v in c.items() if isinstance(v, dict)}, flush=True)
            if not c["pass"]:
                print("STOP: the clean pass does not reproduce the banked eval within 1e-3; the grid is not reported")
                return 3
    return cmd_analyze(a)


def _peak_report():
    """Print and log this process's peak private commit and peak working set at exit (Windows GetProcessMemoryInfo)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps = ctypes.windll.psapi
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            line = (f"PEAKMEM argv={' '.join(sys.argv[1:])[:120]} peak_private_commit_MB={pmc.PeakPagefileUsage / 2**20:.0f} "
                    f"peak_working_set_MB={pmc.PeakWorkingSetSize / 2**20:.0f}")
            print(line, flush=True)
            with open(HERE / "peakmem.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as e:                                                                         # noqa: BLE001
        print("PEAKMEM unavailable:", e, flush=True)


import atexit  # noqa: E402
atexit.register(_peak_report)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--seed", required=True, choices=("s0", "s1")); p.add_argument("--donor-seed", type=int, default=0)
    p.add_argument("--identity", action="store_true"); p.add_argument("--regress", default=None, choices=("nav_const", "tscene_off"))
    p.add_argument("--cells", default=None); p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("analyze"); p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("all"); p.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(argv)
    return {"collect": cmd_collect, "analyze": cmd_analyze, "all": cmd_all}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
