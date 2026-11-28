#!/usr/bin/env python3
"""PREREG_SOCARVE.md stage 1: the banked R0 / C0 scorer states (A and B) read through the A2 harness, PER-SET arrays saved, and the section-4 harness control.
CPU only, zero training.  Imports the A2 harness UNCHANGED from the dev-box archive of the safety-head package (file sha256s are in the prereg).
    PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe so_eval_states.py
Writes out/persets_<run>_u<u>.npz (one per state, banked incrementally) and harness_control.json."""
import json, os, sys, time
import numpy as np, torch, psutil

HERE = os.path.dirname(os.path.abspath(__file__))
SHD = os.environ.get("SOCARVE_SHD", "C:/Users/Admin/AppData/Local/Temp/safehead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-safety-head")
BANKED = os.path.join(HERE, "src", "sh_realstates.json")        # = PK/raw/2026-10-16-safety-head/realstates.json @ ac07b1a (sha256 84b2f1df72840e17...)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SHD)
import sh_common as SH            # noqa: E402
import sh_summarize as S          # noqa: E402
PC = SH.PC

try:
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception as e:            # noqa: BLE001
    print("could not lower priority:", e, flush=True)

SH.VIEW_FN["base"] = lambda sxB, sxA, sx0: sx0      # the frozen base scorer's own selection (exposure-free covariates, PREREG s.8 S-EXCH)
VIEWS = ["own", "A_own", "base"]
STATES = [("r0", 250), ("r0", 500), ("r0", 1000), ("c0", 1000)]
MIN_FREE_GB = 8.0
TOL_MEAN, TOL_CI = 1e-4, 1e-3
t0 = time.time()


def free_gb():
    return psutil.virtual_memory().available / 1e9


print(f"start free={free_gb():.1f} GB", flush=True)
SF, a, holder, cfg, tr, ev = SH.setup(int(os.environ.get("SH_THREADS", "4")))
print(f"setup done {time.time()-t0:.0f}s free={free_gb():.1f} GB rss={psutil.Process().memory_info().rss/1e9:.1f} GB", flush=True)
banked = json.load(open(BANKED, encoding="utf-8"))["states"]


def state_theta(ss, which):
    for m in SF.SCORER:
        getattr(tr.armB, m).load_state_dict(ss[which][m], strict=True)
    return [p.detach().clone() for p in tr.armB.parameters()]


def keep(px):
    return {m: {"pooled": px[m]["pooled"], "ci": px[m]["pooled_ci"], "train": px[m].get("train"), "val": px[m].get("val")} for m in px}


def compare(mine, ref):
    """-> (max_mean_diff, max_ci_diff, rows, missing): every key of the section-4 control."""
    rows, mm, mc, missing = [], 0.0, 0.0, []
    for k, r in ref.items():
        if k not in mine:
            missing.append(k)
            continue
        for fld in ("pooled", "train", "val"):
            if r.get(fld) is None:
                continue
            d = abs(mine[k][fld] - r[fld])
            mm = max(mm, d)
            rows.append({"key": k, "field": fld, "mine": mine[k][fld], "banked": r[fld], "absdiff": d})
        for j in (0, 1):
            d = abs(mine[k]["ci"][j] - r["ci"][j])
            mc = max(mc, d)
            rows.append({"key": k, "field": f"ci{j}", "mine": mine[k]["ci"][j], "banked": r["ci"][j], "absdiff": d})
    return mm, mc, rows, missing


control = {"tool": "so_eval_states/1", "tol_mean": TOL_MEAN, "tol_ci": TOL_CI, "states": {}}
for run, u in STATES:
    if free_gb() < MIN_FREE_GB:
        print(f"ABORT: only {free_gb():.1f} GB available (< {MIN_FREE_GB}) before {run} u{u}", flush=True)
        control["aborted_before"] = f"{run}_u{u}"
        break
    ss = torch.load(os.path.join(PC.AC.DL, f"rls1-{run}", "scorer_states", f"scorer_states_u{u}.pt"), map_location="cpu", weights_only=False)
    thA, thB = state_theta(ss, "A"), state_theta(ss, "B")
    E = {pn: SH.eval_views(e, tr, thB, thA, VIEWS) for pn, e in ev.items()}
    full = keep(S.prox({p: E[p]["own"] for p in S.POOLS}, {p: E[p]["A_own"] for p in S.POOLS}))
    arrs = {}
    for pn in S.POOLS:
        e = ev[pn]
        arrs[f"{pn}__logs"] = np.asarray(e.logs).astype(str)
        arrs[f"{pn}__keys"] = np.asarray(["|".join(str(x) for x in k) for k in e.pool.keys]).astype(str)
        arrs[f"{pn}__tg6mean"] = e.tg6.mean(1).numpy().astype(np.float64)
        arrs[f"{pn}__tg6max"] = e.tg6.max(1).values.numpy().astype(np.float64)
        for v in VIEWS:
            for k, x in E[pn][v].items():
                arrs[f"{pn}__{v}__{k}"] = x
    np.savez_compressed(os.path.join(OUT, f"persets_{run}_u{u}.npz"), **arrs)
    rec = {"FULL": full}
    if run == "r0":
        mm, mc, rows, missing = compare(full, banked[f"r0_u{u}"]["FULL"])
        rec.update({"max_mean_absdiff": mm, "max_ci_absdiff": mc, "n_compared": len(rows), "missing_keys": missing,
                    "pass": bool(mm <= TOL_MEAN and mc <= TOL_CI and not missing), "rows": rows})
        print(f"{run} u{u}: control max|d_mean|={mm:.2e} max|d_ci|={mc:.2e} n={len(rows)} missing={missing} -> {'PASS' if rec['pass'] else 'FAIL'}"
              f" | P2 TTC pooled {full['P2_TTC']['pooled']:+.4f} train {full['P2_TTC']['train']:+.4f} val {full['P2_TTC']['val']:+.4f}"
              f" | P1 TTC pooled {full['P1_TTC']['pooled']:+.4f} | {time.time()-t0:.0f}s free={free_gb():.1f}GB", flush=True)
    else:
        floor = abs(full["P2_TTC"]["pooled"])
        rec.update({"c0_floor_abs_P2_TTC_pooled": floor, "pass": bool(floor < 0.5)})
        print(f"{run} u{u}: C0 floor |P2 TTC pooled| = {floor:.6f} -> {'PASS' if rec['pass'] else 'FAIL'} | {time.time()-t0:.0f}s", flush=True)
    control["states"][f"{run}_u{u}"] = rec
    json.dump(control, open(os.path.join(HERE, "harness_control.json"), "w"), indent=1)
    del ss, E

need = [f"r0_u{u}" for u in (250, 500, 1000)] + ["c0_u1000"]
control["all_states_present"] = all(k in control["states"] for k in need)
control["verdict"] = "VALID" if (control["all_states_present"] and all(control["states"][k]["pass"] for k in need)) else "INVALID"
json.dump(control, open(os.path.join(HERE, "harness_control.json"), "w"), indent=1)
print("ZZSOCARVE_STAGE1", control["verdict"], f"{time.time()-t0:.0f}s", flush=True)
