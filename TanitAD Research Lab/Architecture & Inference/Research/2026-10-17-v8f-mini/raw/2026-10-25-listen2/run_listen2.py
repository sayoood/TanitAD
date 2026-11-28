"""PREREG_V8F_LISTEN2 -- the real-data driver. CPU only (CUDA_VISIBLE_DEVICES=-1), torch threads 6, zero training, ONE process at a time.

  python run_listen2.py means   --seed s0                          the training-set means of every input (HBLEND's 9,582 windows) + their control
  python run_listen2.py collect --seed s0 --mode donor --donor-seed 0 --out runs/s0_d0.npz        E1 + E3 (+ the reference cells), the clean pass
  python run_listen2.py collect --seed s0 --mode donor --identity --out runs/s0_id.npz            control 1 (donor = self)
  python run_listen2.py collect --seed s0 --mode mean --out runs/s0_mean.npz                      E2 (mean replacement)
  python run_listen2.py collect --seed s0 --mode own  --out runs/s0_own.npz                       control 3 (mean path, the window's own value)
  python run_listen2.py analyze [--controls-only]                                                  controls (+ the three tables) -> READ.json

Inputs are read from SFRESH's / HBLEND's / LISTEN's directories READ-ONLY; everything written lives under this directory.
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
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LISTEN_THREADS", "6")
os.environ.setdefault("LISTEN_LRU", "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import listen as LS  # noqa: E402
import listen2 as L2  # noqa: E402
import listen_read as LR  # noqa: E402
import read2 as R2  # noqa: E402
import run_listen as RL  # noqa: E402  (LISTEN's driver, a copy: PATHS, load_ctx, load_eval, sha256, and the peak-memory hook -> peakmem.log here)
from tanitad.data import v8f_mini_data as D  # noqa: E402

SCR = HERE.parent
HB = SCR / "hblend"
LISTEN_DIR = SCR / "listen"
RUNS = HERE / "runs"
PREREG_COMMIT = "067c8b74"
HB_SHA = {"s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", "s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
N_EVAL = 2399


# ------------------------------------------------------------------------------------------------ the training-set means
def cmd_means(a):
    t0 = time.time()
    seed = a.seed
    model, cfg, ck = RL.load_model(RL.PATHS, seed)
    sha = RL.sha256(RL.PATHS["ckpt"][seed])
    assert sha == HB_SHA[seed], f"the checkpoint is not HBLEND's ({sha})"
    fc = json.load(open(HB / "fit_clips.json", encoding="utf-8"))
    J = D.load_join(str(HB / "box2" / "join_train.npz"))
    tracks = D.load_native_tracks(str(RL.PATHS["tracks"]))
    store = D.ClipStore(str(HB / "cache"), "train", lru=int(os.environ.get("LISTEN_LRU", "8")), grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.train_windows(store, J, fc["clips"], hist=cfg.hist, start=cfg.window_start)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    N = len(wins)
    z = np.load(HB / f"fit_windows_M2b_{seed}.npz")
    ctrl = {"n_windows": N, "n_fit_windows_hblend": int(len(z["ci"])), "fit_clips_sha256": fc["sha256_newline_joined"], "ckpt_sha256": sha,
            "native_tracks_sha256": RL.sha256(RL.PATHS["tracks"])}
    assert N == len(z["ci"]) == 9582, N
    acc = L2.MeanAccumulator(model)
    got = {k: [] for k in ("traj30", "ci", "t", "a_t", "a_s")}
    with torch.no_grad():
        for b0 in range(0, N, a.batch):
            idxs = list(range(b0, min(b0 + a.batch, N)))
            P = L2.collate_own(model, ds, wins, idxs)
            acc.add(P)
            got["traj30"].append(P.out["traj30"].float().numpy())
            for k in ("ci", "t", "a_t", "a_s"):
                got[k].append(P.b[k].numpy())
            if (b0 // a.batch) % 20 == 0:
                print(f"  [means {seed}] {b0 + len(idxs)}/{N} windows, {time.time() - t0:.0f}s", flush=True)
    got = {k: np.concatenate(v) for k, v in got.items()}
    ctrl["window_list_equal"] = {k: bool(np.array_equal(got[k], z[k])) for k in ("ci", "t", "a_t", "a_s")}
    ctrl["max_abs_traj30_vs_hblend_fit_windows_m"] = float(np.abs(got["traj30"].astype(np.float64) - z["traj30"].astype(np.float64)).max())
    ctrl["pass"] = bool(all(ctrl["window_list_equal"].values()) and ctrl["max_abs_traj30_vs_hblend_fit_windows_m"] <= 1e-3)
    mn = acc.finish({"seed": seed, "n_windows": N, "ckpt_sha256": sha, "prereg": "PREREG_V8F_LISTEN2", "mean_ages_ticks": [float(x) for x in acc.s[("age",)] / N],
                     "control": ctrl})
    L2.save_means(mn, RUNS / f"means_{seed}.npz")
    ctrl["pad_any_windows_per_layer"] = mn.meta["pad_any_windows"]
    ctrl["mean_ages_ticks"] = mn.meta["mean_ages_ticks"]
    ctrl["seconds"] = round(time.time() - t0, 1)
    json.dump(ctrl, open(RUNS / f"means_{seed}_control.json", "w", encoding="utf-8"), indent=1)
    print("MEANS", seed, "control pass:", ctrl["pass"], ctrl, flush=True)
    return 0 if ctrl["pass"] else 4


# ------------------------------------------------------------------------------------------------ collect
def cmd_collect(a):
    t0 = time.time()
    ctx = RL.load_ctx(RL.PATHS, a.seed)
    model, X, wins = ctx["model"], ctx["X"], ctx["wins"]
    N = len(wins)
    assert N == N_EVAL, N
    means = None
    if a.mode == "mean":
        means = L2.load_means(RUNS / f"means_{a.seed}.npz")
        assert means.n == 9582, means.n
    if a.mode == "donor":
        perm = np.arange(N) if a.identity else LS.donor_perm(N, X["eid"], a.donor_seed)
        assert (perm == np.arange(N)).all() == bool(a.identity)
        if not a.identity:
            assert (np.asarray(X["eid"])[perm] != np.asarray(X["eid"])).all(), "a donor from the same episode"
    else:
        perm = None
    last = [0.0]

    def prog(b0, n):
        if time.time() - last[0] > 20:
            print(f"  [{Path(a.out).name}] {b0}/{n} windows, {time.time() - t0:.0f}s", flush=True)
            last[0] = time.time()

    res, n_asserted = L2.collect2(model, ctx["ds"], wins, X["gt"], perm, a.mode, means, batch=a.batch, progress=prog)
    meta = {"prereg": "PREREG_V8F_LISTEN2", "prereg_commit": PREREG_COMMIT, "mode": a.mode, "seed": a.seed, "identity": bool(a.identity),
            "donor_seed": a.donor_seed if a.mode == "donor" and not a.identity else None, "n_windows": N, "n_windows_asserted_E1_invariance": int(n_asserted),
            "ckpt": str(RL.PATHS["ckpt"][a.seed]), "ckpt_sha256": RL.sha256(RL.PATHS["ckpt"][a.seed]), "threads": torch.get_num_threads(), "device": "cpu",
            "ages": "cycle", "seconds": round(time.time() - t0, 1),
            "perm_sha256": None if perm is None else hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest(),
            "means_sha256": None if means is None else RL.sha256(RUNS / f"means_{a.seed}.npz")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    LR.save_run(res, a.out, meta)
    print(f"wrote {a.out} ({meta['seconds']} s), E1 invariance asserted on {n_asserted} windows", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ controls
def _maxabs(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return float(np.nanmax(d)) if d.size else 0.0


#: the clean row that holds each stored key
_CLEAN_OF_KEY = {"ck_now": "clean/S", "lat_pred": "clean/T", "lon_pred": "clean/T", "ade6": "clean/O", "ade2": "clean/O"}


def control_identical_to_clean(res):
    """Controls 1 and 3: every non-clean row of the run is BIT-IDENTICAL to the clean pass (so every delta is exactly 0.0)."""
    bad, worst, n = [], 0.0, 0
    for nm, r in res.items():
        if nm.startswith("clean/"):
            continue
        n += 1
        for k, v in r.items():
            ref = res[_CLEAN_OF_KEY[k]][k]
            worst = max(worst, _maxabs(v, ref))
            if not np.array_equal(v, ref):
                bad.append(f"{nm}:{k}")
    return {"n_rows": n, "max_abs_diff_vs_clean": worst, "rows_not_identical": bad, "pass": (not bad) and worst == 0.0 and n > 0}


def control_delta_zero(res, X):
    """Controls 1 / 3 on the READ side: every delta the reading would compute from this run is exactly 0.0 and not separated."""
    worst, sep = 0.0, []
    if L2.E1_NAME in res:
        b = R2.e1_block(res, X, 200)
        cells = list(b["rows"].values()) + list(b["ref"].values())
        e3 = R2.e3_block(res, X, 200)
        cells += [e3["E3"], e3["joint_vs_clean"]] + list(e3["single_vs_clean"].values())
    else:
        rec = LR.run_record(res, X, 200)
        cells = [c for ent in list(rec["rows"].values()) + list(rec["beside"].values()) for c in ent["cells"].values() if c is not None]
    for c in cells:
        worst = max(worst, abs(c["delta"]))
        if c["separated"]:
            sep.append(c)
    return {"n_cells": len(cells), "max_abs_delta": worst, "n_separated": len(sep), "pass": worst == 0.0 and not sep}


def control_repro(seed, res, read_listen):
    """Control 2: the clean pass reproduces LISTEN's clean levels (stop threshold 1e-4) and LISTEN's clean per-window arrays."""
    _, _, X, _ = RL.load_eval(RL.PATHS, "yaw")
    levels = {}
    for row in ("S", "T-lat", "T-lon", "O"):
        L = LR.LAYER_OF[row]
        mine = LR.level(row, LR.vec(row, res[f"clean/{L}"], X), X)
        theirs = float(read_listen["grid"][seed]["rows"][row]["clean_level"])
        levels[row] = {"mine": mine, "listen": theirs, "abs_diff": abs(mine - theirs)}
    lres, _ = LR.load_run(LISTEN_DIR / "runs" / f"{seed}_d0.npz")
    arr = {}
    for nm in ("clean/S", "clean/T", "clean/O", "clean/full"):
        for k, v in res[nm].items():
            arr[f"{nm}:{k}"] = _maxabs(v, lres[nm][k])
    out = {"levels": levels, "max_level_diff": max(v["abs_diff"] for v in levels.values()), "max_perwindow_diff_vs_listen_clean": max(arr.values()),
           "perwindow": arr}
    out["pass_1e-4"] = bool(out["max_level_diff"] <= 1e-4)
    # LISTEN's own cells, recomputed here (DIAGNOSTIC, not a stop condition): the reference cells and Q-NAV's nav@S arm
    refs = {}
    if L2.E1_NAME in res:
        for (L, x) in L2.REF_CELLS:
            nm = LS.cell_name(L, x)
            if nm in res and nm in lres:
                refs[nm] = max(_maxabs(v, lres[nm][k]) for k, v in res[nm].items())
        refs["e1/nav@S vs LISTEN nav@S (ade6)"] = _maxabs(res[L2.E1_NAME]["ade6"], lres["nav@S"]["ade6"])
        refs["e1/nav@S S-row vs own S/nav cell (bit-identity expected)"] = _maxabs(res[L2.E1_NAME]["ck_now"], res["S/nav"]["ck_now"])
    out["listen_cells_recomputed_maxabs_DIAGNOSTIC"] = refs
    return out


def analyze_controls(runs, metas, read_listen):
    c = {"identity": {}, "own_value": {}, "reproduction": {}, "means": {}, "e1_invariance": {}}
    _, _, X, _ = RL.load_eval(RL.PATHS, "yaw")
    for s in ("s0", "s1"):
        if f"{s}_id" in runs:
            c["identity"][s] = {**control_identical_to_clean(runs[f"{s}_id"]), "delta_zero": control_delta_zero(runs[f"{s}_id"], X)}
        if f"{s}_own" in runs:
            c["own_value"][s] = {**control_identical_to_clean(runs[f"{s}_own"]), "delta_zero": control_delta_zero(runs[f"{s}_own"], X)}
        if s in runs:
            c["reproduction"][s] = control_repro(s, runs[s], read_listen)
        p = RUNS / f"means_{s}_control.json"
        if p.exists():
            c["means"][s] = json.load(open(p, encoding="utf-8"))
    for k, m in metas.items():
        if m.get("mode") == "donor":
            c["e1_invariance"][k] = {"windows_asserted": m["n_windows_asserted_E1_invariance"], "windows": m["n_windows"],
                                     "pass": m["n_windows_asserted_E1_invariance"] == m["n_windows"] == N_EVAL}
    # a pass requires every control that exists to pass; the full reading additionally requires them all to exist
    c["required_present"] = {"identity": sorted(c["identity"]), "own_value": sorted(c["own_value"]), "reproduction": sorted(c["reproduction"]),
                             "means": sorted(c["means"]), "e1_invariance": sorted(c["e1_invariance"])}
    ok = all(v["pass"] and v["delta_zero"]["pass"] for v in c["identity"].values())
    ok &= all(v["pass"] and v["delta_zero"]["pass"] for v in c["own_value"].values())
    ok &= all(v["pass_1e-4"] for v in c["reproduction"].values())
    ok &= all(v["pass"] for v in c["means"].values())
    ok &= all(v["pass"] for v in c["e1_invariance"].values())
    c["all_present_pass"] = bool(ok)
    c["complete"] = bool(len(c["identity"]) == 2 and len(c["own_value"]) == 2 and len(c["reproduction"]) == 2 and len(c["means"]) == 2 and len(c["e1_invariance"]) >= 5)
    return c


RUN_FILES = {"s0": "s0_d0.npz", "s1": "s1_d0.npz", "s0_d1": "s0_d1.npz", "s0_id": "s0_id.npz", "s1_id": "s1_id.npz", "s0_mean": "s0_mean.npz",
             "s1_mean": "s1_mean.npz", "s0_own": "s0_own.npz", "s1_own": "s1_own.npz"}


def _jdefault(o):
    return o.item() if hasattr(o, "item") else str(o)


def cmd_analyze(a):
    t0 = time.time()
    _, _, X, _ = RL.load_eval(RL.PATHS, "yaw")
    read_listen = json.load(open(LISTEN_DIR / "READ.json", encoding="utf-8"))
    runs, metas = {}, {}
    for k, fn in RUN_FILES.items():
        p = RUNS / fn
        if p.exists():
            runs[k], metas[k] = LR.load_run(p)
    ctrl = analyze_controls(runs, metas, read_listen)
    read = {"prereg": "PREREG_V8F_LISTEN2", "prereg_commit": PREREG_COMMIT, "executed": time.strftime("%Y-%m-%d"),
            "tier": "T1-class open loop, DIAGNOSTIC (donor / mean inputs are interventions, not driving)",
            "estimator": "paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py); answers 'another draw of EPISODES', not 'another training "
                         "run' (two seeds side by side) and not 'another inference run' (deterministic)",
            "evidence_class": "MEASURED (ours); artifacts in this directory (runs/*.npz)", "controls": ctrl,
            "readings_fixed_before_data": L2.__doc__.split("READINGS OF THE PREREG TEXT")[1].strip(), "run_meta": metas}
    out = HERE / ("READ_controls.json" if a.controls_only else "READ.json")
    if a.controls_only or not (ctrl["all_present_pass"] and ctrl["complete"]):
        json.dump(read, open(out, "w", encoding="utf-8"), indent=1, default=_jdefault)
        print("controls: complete", ctrl["complete"], "| all present pass", ctrl["all_present_pass"], "->", out.name)
        return 0 if ctrl["all_present_pass"] else 5
    need = ("s0", "s1", "s0_d1", "s0_mean", "s1_mean")
    miss = [k for k in need if k not in runs]
    if miss:
        print("missing runs:", miss)
        return 2
    B = LR.B_BOOT
    # ---- E1 / E3
    blk = {k: R2.e1_block(runs[k], X, B) for k in ("s0", "s1", "s0_d1")}
    e3 = {k: R2.e3_block(runs[k], X, B) for k in ("s0", "s1", "s0_d1")}
    floor = R2.donor_floor_e1(blk["s0"], blk["s0_d1"])
    flags = {s: R2.e1_seed_flags(blk[s], floor) for s in ("s0", "s1")}
    e1 = {"blocks": blk, "donor_seed_floor": floor, "flags": flags, "outcome": R2.e1_outcome(flags), "meaning_committed": R2.E1_MEANING}
    e1["outcome"]["meaning"] = R2.E1_MEANING[e1["outcome"]["headline"]]
    e3out = {"blocks": e3, "outcome": R2.e3_outcome({s: e3[s] for s in ("s0", "s1")})}
    # ---- E2
    garg = R2.grid_argmax(read_listen)
    mrec = {s: LR.run_record(runs[f"{s}_mean"], X, B) for s in ("s0", "s1")}
    for r in mrec.values():
        LR.add_n(r)
    e2 = {"grid_argmax": garg, "records": mrec, "verdict": R2.e2_verdict(mrec, garg)}
    read.update({"E1": e1, "E2": e2, "E3": e3out, "seconds_analysis": round(time.time() - t0, 1)})
    json.dump(read, open(out, "w", encoding="utf-8"), indent=1, default=_jdefault)
    print("E1:", e1["outcome"]["headline"], "| E2:", e2["verdict"]["headline"], "| E3:", e3out["outcome"]["headline"], "| controls all pass:", ctrl["all_present_pass"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("means")
    p.add_argument("--seed", required=True, choices=("s0", "s1")); p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("collect")
    p.add_argument("--seed", required=True, choices=("s0", "s1")); p.add_argument("--mode", required=True, choices=("donor", "mean", "own"))
    p.add_argument("--donor-seed", type=int, default=0); p.add_argument("--identity", action="store_true")
    p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("analyze"); p.add_argument("--controls-only", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        return cmd_analyze(a)
    return {"means": cmd_means, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
