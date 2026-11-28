"""along_track_localise.py -- WHERE in refcv8's planner is the along-track excess over the causal prior produced?
(refcv9 L1 design annex, item 1.  CPU only, banked dumps only, no model call, no GPU.)

Objects compared on the SAME windows (S2 grid 0.5/1.0/1.5/2.0 s = model slots 0..3; ceiling-compliant windows; eval139 EMA trees and
the 139-clip TRAIN tree at EMA-30k):
    os    the emitted plan                      (panel `os`, == decisions `plan_full_nav_true[:, :4]`, asserted bitwise)
    bank  the SELECTED candidate's UNREFINED bank path  (decisions `sel_bank_nav_true`; rows exist only where sel_idx < 117)
    orc   the a_star candidate, refined         (panel `oracle_sel`; a_star = GT-nearest BANK anchor -- the object the trainer's
                                                 loss_traj supervises, refc_v3_train.py:5545-5560)
    P     the causal prior ha0_ext_pose         (RT.references -> kinematic_prior.prior_path; the model's own residual_prior_path)
    H     the human future                      (panel `g`)
Arc length s_X(t_k) = cumulative chord length of the polyline through (0,0) and X's points (RT.polyline_geometry, the shipped
stage's geometry).  d_X = s_X - s_P (the arm's along-track residual over the prior), h = s_H - s_P (the human's).

Statistics, pooled over windows x instants, per-episode sums so the episode-cluster bootstrap is exact:
    g*_X = <d_X, h> / <d_X, d_X>    the GAINTRAJ flat re-timing gain (rule-6 control: os must reproduce GAINTRAJ.json)
    A_X  = <d_X, h> / <h, h>        regression of the arm's residual on the human's.  LINEAR in d_X, hence EXACTLY ADDITIVE:
                                     A_os = A_bank + (A_os - A_bank)   [bank/selection + refinement]
                                     A_os = A_orc  + (A_os - A_orc)    [trained candidate + selection]
    rho_X = sqrt(<d_X,d_X>/<h,h>)  the residual's SIZE relative to the human's (2.0 = "runs 2x as far from the prior")
    r_X  = <d_X,h>/sqrt(<d,d><h,h>) the residual's ALIGNMENT with the human's;  g* = r / rho,  A = r * rho
So a small g* can come from an over-sized residual (rho >> 1) or from a mis-aligned one (r << 1); the two need different fixes.

Known-value controls (must read their values before anything is reported):
    K1 d := h          -> g* = A = rho = r = 1 exactly
    K2 d := 2h         -> g* = 0.5, A = 2, rho = 2, r = 1
    K3 d := h + e, e orthogonal to h with |e| = |h|  -> g* = 0.5, A = 1, rho = sqrt(2), r = 1/sqrt(2)
    KM mutation: prior and os swapped in g* -> must NOT reproduce GAINTRAJ (RED)
    R6 os g* on EMA-30k compliant S2 reproduces GAINTRAJ.json (eval) and GAINTRAJ_TRAIN.json (train) to 1e-9

Tier: T1 self-action OPEN LOOP on logged frames (os); orc is a T0 ceiling (uses GT to pick a_star).  Estimator: episode-cluster
bootstrap B 2000 (paired within a tree; UNPAIRED train vs eval); answers 'another draw of EPISODES' only -- not another training run
(H-ESTIM-SEED-1), not another inference run (seeds 0/1 reported separately).

Run:
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<battery>/code' \
        C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 along_track_localise.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

BATT = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery")
sys.path.insert(0, str(BATT / "code"))
import a0k0_headroom_retime_r8 as RT  # noqa: E402  (imported, never edited)

OUT = Path(__file__).resolve().parent
EVAL_TREES = ("step30000ema_unc", "step40000ema_unc", "step5000ema_unc")
PRIMARY_TREE = "step30000ema_unc"
SEEDS = (0, 1)
N_BANK = 117
N_BOOT = 2000
BOOT_SEED = 0
MASK_SHA = "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129"   # GAINTRAJ's registered eval mask
GT_JSON = BATT / "raw" / "gain_traj" / "GAINTRAJ.json"
GT_TRAIN_JSON = BATT / "raw" / "gain_traj" / "GAINTRAJ_TRAIN.json"
ARMS = ("os", "bank", "orc")


def arc(x) -> np.ndarray:
    return RT.polyline_geometry(np.asarray(x, np.float64))["L"][:, 1:]


def ep_sums(d: np.ndarray, h: np.ndarray, eid: np.ndarray):
    """per-episode <d,h>, <d,d>, <h,h> over windows x instants."""
    uniq, inv = np.unique(eid, return_inverse=True)
    dh = np.bincount(inv, (d * h).sum(1), len(uniq))
    dd = np.bincount(inv, (d * d).sum(1), len(uniq))
    hh = np.bincount(inv, (h * h).sum(1), len(uniq))
    return uniq, dh, dd, hh


def stats_from(dh, dd, hh) -> dict:
    DH, DD, HH = float(dh.sum()), float(dd.sum()), float(hh.sum())
    return {"g_star": DH / DD, "A": DH / HH, "rho": (DD / HH) ** 0.5, "r": DH / (DD * HH) ** 0.5}


def counts(n_ep: int, seed: int = BOOT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_ep, size=(N_BOOT, n_ep))
    C = np.zeros((N_BOOT, n_ep))
    for b in range(N_BOOT):
        C[b] = np.bincount(idx[b], minlength=n_ep)
    return C


def boot(C, dh, dd, hh) -> dict:
    DH, DD, HH = C @ dh, C @ dd, C @ hh
    return {"g_star": DH / DD, "A": DH / HH, "rho": np.sqrt(DD / HH), "r": DH / np.sqrt(DD * HH)}


def ci(v) -> list:
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def cell(per_arm: dict, C) -> dict:
    """per_arm: {arm: (dh, dd, hh)} on ONE window set (same episodes, same order).  Point + CI of every stat and of every arm difference."""
    out, B = {}, {}
    for a, (dh, dd, hh) in per_arm.items():
        p = stats_from(dh, dd, hh)
        B[a] = boot(C, dh, dd, hh)
        out[a] = {k: {"est": p[k], "ci": ci(B[a][k])} for k in p}
    names = list(per_arm)
    for i, a in enumerate(names):
        for b2 in names[i + 1:]:
            pa, pb = stats_from(*per_arm[a]), stats_from(*per_arm[b2])
            for k in ("A", "g_star", "rho", "r"):
                dv = B[a][k] - B[b2][k]
                lo, hi = ci(dv)
                out[f"{a}-{b2}"] = out.get(f"{a}-{b2}", {})
                out[f"{a}-{b2}"][k] = {"est": pa[k] - pb[k], "ci": [lo, hi], "separated": bool(lo > 0 or hi < 0)}
    return out


def per_instant(d: dict, h: np.ndarray) -> dict:
    return {a: {"A": [float((x[:, k] * h[:, k]).sum() / (h[:, k] ** 2).sum()) for k in range(h.shape[1])],
                "rho": [float(np.sqrt((x[:, k] ** 2).sum() / (h[:, k] ** 2).sum())) for k in range(h.shape[1])],
                "mean_signed_m": [float(x[:, k].mean()) for k in range(h.shape[1])]}
            for a, x in d.items()} | {"human": {"mean_signed_m": [float(h[:, k].mean()) for k in range(h.shape[1])],
                                                 "rms_m": [float(np.sqrt((h[:, k] ** 2).mean())) for k in range(h.shape[1])]}}


def load_extras(NR, pdir: str, ws_panel: np.ndarray) -> dict:
    """decisions rows in the panel's own file order (NR.panel_layout, as RT.load_decisions); sel_bank expanded with NaN off-bank."""
    _m, files, _e = NR.panel_layout(pdir)
    sb_all, sel_all, pf_all, ws_all, orc_all, os_all = [], [], [], [], [], []
    for f in files:
        with np.load(f) as pz:                      # the panel file itself: oracle_sel + os (row-order control)
            orc_all.append(pz["oracle_sel"])
            os_all.append(pz["os"])
        with np.load(os.path.join(pdir, "decisions", os.path.basename(f))) as d:
            n = d["ws"].shape[0]
            sel = d["sel_idx_nav_true"].astype(np.int64)
            inb = sel < N_BANK
            sb = d["sel_bank_nav_true"]
            full = np.full((n, 8, 2), np.nan, np.float32)
            if sb.shape[0] == int(inb.sum()):
                full[inb] = sb
            elif sb.shape[0] == n:            # a tree that stored every row: keep only in-bank rows (an allocator extra has no bank row)
                full[inb] = sb[inb]
            else:
                raise SystemExit(f"[l1probe] {f}: sel_bank rows {sb.shape[0]} fit neither in-bank {int(inb.sum())} nor all {n}")
            sb_all.append(full)
            sel_all.append(sel)
            pf_all.append(d["plan_full_nav_true"])
            ws_all.append(d["ws"].astype(np.int64))
    X = {"sel_bank": np.concatenate(sb_all), "sel_idx": np.concatenate(sel_all), "plan_full": np.concatenate(pf_all), "ws": np.concatenate(ws_all),
         "oracle_sel": np.concatenate(orc_all), "os_panel": np.concatenate(os_all)}
    if not np.array_equal(X["ws"], np.asarray(ws_panel, np.int64)):
        raise SystemExit(f"[l1probe] {pdir}: decisions ws != panel ws")
    return X


def gather(tools, ctx, S, tree: str) -> dict:
    P = ctx["panels"]["S2"]
    A = P["A"]
    X = S["X"]
    ext = load_extras(tools["NR"], P["dir"], S["ws"])
    ctl = {"plan_full_slots0_3_eq_os_maxabs_m": float(np.abs(ext["plan_full"][:, :4].astype(np.float64) - X["O"].astype(np.float64)).max()),
           "panel_file_os_eq_loader_os_maxabs_m": float(np.abs(ext["os_panel"].astype(np.float64) - X["O"].astype(np.float64)).max())}
    if ctl["plan_full_slots0_3_eq_os_maxabs_m"] != 0.0 or ctl["panel_file_os_eq_loader_os_maxabs_m"] != 0.0:
        raise SystemExit(f"[l1probe] {tree}: row order / plan identity control failed {ctl}")
    comp = np.asarray(S["mask"], bool)
    return {"O": np.asarray(X["O"]), "HP": np.asarray(X["HP"]), "G": np.asarray(X["G"]), "ORC": np.asarray(ext["oracle_sel"]),
            "BANK": ext["sel_bank"][:, :4], "inb": ext["sel_idx"] < N_BANK, "comp": comp, "eid": np.asarray([str(e) for e in S["eid"]]),
            "sel_idx": ext["sel_idx"], "ctl": ctl}


def analyse(D: dict) -> dict:
    sp, sh = arc(D["HP"]), arc(D["G"])
    h = sh - sp
    dX = {"os": arc(D["O"]) - sp, "orc": arc(D["ORC"]) - sp}
    bank_ok = D["inb"] & np.isfinite(D["BANK"]).all(axis=(1, 2))
    bank = np.full_like(dX["os"], np.nan)
    bank[bank_ok] = arc(D["BANK"][bank_ok]) - sp[bank_ok]
    dX["bank"] = bank
    res = {}
    for set_name, sel in (("compliant", D["comp"]), ("compliant_inbank", D["comp"] & bank_ok)):
        arms = ARMS if set_name == "compliant_inbank" else ("os", "orc")
        eid = D["eid"][sel]
        per = {}
        for a in arms:
            uniq, dh, dd, hh = ep_sums(dX[a][sel], h[sel], eid)
            per[a] = (dh, dd, hh)
        C = counts(len(uniq))
        c = cell(per, C)
        c["n_windows"], c["n_episodes"] = int(sel.sum()), int(len(uniq))
        c["per_instant"] = per_instant({a: dX[a][sel] for a in arms}, h[sel])
        res[set_name] = c
        res[set_name]["_ep"] = {a: [x.tolist() for x in per[a]] for a in arms}
    res["inbank_frac_of_compliant"] = float((D["comp"] & bank_ok).sum() / D["comp"].sum())
    res["sel_is_a_star_frac"] = None
    return res


def known_value_controls() -> dict:
    rng = np.random.default_rng(7)
    h = rng.normal(size=(500, 4))
    eid = np.repeat(np.arange(50), 10)
    e = rng.normal(size=h.shape)
    e -= (e * h).sum() / (h * h).sum() * h          # orthogonal to h (pooled inner product)
    e *= np.sqrt((h * h).sum() / (e * e).sum())
    out = {}
    for name, d, want in (("K1", h, {"g_star": 1, "A": 1, "rho": 1, "r": 1}),
                          ("K2", 2 * h, {"g_star": 0.5, "A": 2, "rho": 2, "r": 1}),
                          ("K3", h + e, {"g_star": 0.5, "A": 1, "rho": 2 ** 0.5, "r": 2 ** -0.5})):
        _u, dh, dd, hh = ep_sums(d, h, eid)
        got = stats_from(dh, dd, hh)
        out[name] = {"got": got, "want": want, "PASS": all(abs(got[k] - want[k]) < 1e-9 for k in want)}
    out["PASS"] = all(v["PASS"] for v in out.values())
    return out


def main() -> int:
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("CPU only: set CUDA_VISIBLE_DEVICES=-1")
    kv = known_value_controls()
    log(f"known-value controls PASS={kv['PASS']}")
    if not kv["PASS"]:
        raise SystemExit("known-value control failed")
    import compliant_reread_r8 as CR
    tools = RT.tools_load()
    gt_banked = json.loads(GT_JSON.read_text(encoding="utf-8"))
    gt_train = json.loads(GT_TRAIN_JSON.read_text(encoding="utf-8"))
    out = {"tool": "along_track_localise.py", "generated": time.strftime("%FT%T"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "tier": "T1 self-action open loop (os, bank); orc = T0 ceiling (a_star uses GT)",
           "estimator": "episode-cluster bootstrap B 2000 over episodes, models fixed; paired within a tree; train vs eval UNPAIRED",
           "caveats": "single training seed; S2 grid only (oracle_sel is dumped on S2 only); compliant windows; bank rows exist only where "
                      "the selected candidate is in-bank (sel_idx < 117), so the bank decomposition is on that subset",
           "known_value_controls": kv, "trees": {}, "controls": {}}
    first = {}
    for tree in EVAL_TREES:
        out["trees"][tree] = {}
        need = [Path(f"{CR.ARCHIVE}/{tree}/dump_s{s_}/refcv6_extras.npz") for s_ in SEEDS] +                [Path(f"{CR.ARCHIVE}/{tree}/panel_s{s_}/manifest.json") for s_ in SEEDS]
        if not all(p_.exists() for p_ in need):
            out["trees"][tree] = {"SKIPPED": "tree incomplete on disk (battery not finished?): " + ", ".join(str(p_) for p_ in need if not p_.exists())}
            log(f"skip {tree}: incomplete")
            continue
        for seed in SEEDS:
            log(f"eval {tree} s{seed}")
            ctx = CR.prepare_seed(tree, seed, tools, tools["rc"])
            if ctx["sha"] != MASK_SHA:
                raise SystemExit(f"{tree} s{seed}: mask sha {ctx['sha']} != {MASK_SHA}")
            fs = first.setdefault(tree, {})
            S = RT.surface_inputs(tools, ctx, "S2", fs, tree)
            if not (S["dec"]["chk"]["PASS"] and S["V"]["PASS"]):
                raise SystemExit(f"{tree} s{seed}: decisions / reference validation failed")
            D = gather(tools, ctx, S, tree)
            r = analyse(D)
            r["ctl"] = D["ctl"]
            if tree == PRIMARY_TREE:
                g_mine = r["compliant"]["os"]["g_star"]["est"]
                g_bank = gt_banked["cells"][tree]["S2"][str(seed)]["g_star"]
                sp, so, sh = arc(D["HP"][D["comp"]]), arc(D["O"][D["comp"]]), arc(D["G"][D["comp"]])
                g_swapped = float(((sp - so) * (sh - so)).sum() / ((sp - so) ** 2).sum())
                out["controls"][f"R6_eval_s{seed}"] = {"mine": g_mine, "banked": g_bank, "PASS": abs(g_mine - g_bank) < 1e-9}
                out["controls"][f"KM_swap_s{seed}"] = {"swapped_g": g_swapped, "banked": g_bank, "RED": abs(g_swapped - g_bank) > 1e-3}
            out["trees"][tree][str(seed)] = r
            del ctx
    # ---- the 139 TRAIN clips at EMA-30k (the A6 sample; 0 overlap with eval139) ----
    import retime_trainfit_r8 as TF
    _CP, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    out["trees"]["train139_ema30k"] = {}
    ftr = {}
    for seed in SEEDS:
        log(f"train s{seed}")
        ctx = CR2.prepare_seed(TF.TRAIN_TREE, seed, tools, tools["rc"])
        S = TF.train_surface_inputs(tools, ctx, "S2", ftr, TF.TRAIN_TREE)
        if not (S["dec"]["chk"]["PASS"] and S["V"]["PASS"]):
            raise SystemExit(f"train s{seed}: decisions / validation failed")
        D = gather(tools, ctx, S, TF.TRAIN_TREE)
        r = analyse(D)
        r["ctl"] = D["ctl"]
        g_mine = r["compliant"]["os"]["g_star"]["est"]
        g_bank = gt_train["cells"][f"S2|compliant|s{seed}"]["g_star"]
        out["controls"][f"R6_train_s{seed}"] = {"mine": g_mine, "banked": g_bank, "PASS": abs(g_mine - g_bank) < 1e-9}
        out["trees"]["train139_ema30k"][str(seed)] = r
        del ctx
    # ---- train vs eval (EMA-30k), UNPAIRED: is the eval-only excess in bank, refinement or selection? ----
    out["train_minus_eval_ema30k"] = {}
    for seed in SEEDS:
        ev = out["trees"][PRIMARY_TREE][str(seed)]["compliant_inbank"]["_ep"]
        tr = out["trees"]["train139_ema30k"][str(seed)]["compliant_inbank"]["_ep"]
        Ce = counts(len(ev["os"][0]), BOOT_SEED)
        Ct = counts(len(tr["os"][0]), BOOT_SEED + 1)
        cc = {}
        for a in ARMS:
            be = boot(Ce, *[np.asarray(x) for x in ev[a]])
            bt = boot(Ct, *[np.asarray(x) for x in tr[a]])
            pe, pt = stats_from(*[np.asarray(x) for x in ev[a]]), stats_from(*[np.asarray(x) for x in tr[a]])
            cc[a] = {k: {"est": pt[k] - pe[k], "ci": ci(bt[k] - be[k]), "train": pt[k], "eval": pe[k]} for k in ("A", "g_star", "rho", "r")}
            for k in cc[a]:
                lo, hi = cc[a][k]["ci"]
                cc[a][k]["separated"] = bool(lo > 0 or hi < 0)
        out["train_minus_eval_ema30k"][str(seed)] = cc
    out["controls"]["ALL_PASS"] = bool(all(v.get("PASS", v.get("RED", False)) for k, v in out["controls"].items() if isinstance(v, dict)))
    (OUT / "ALONG_TRACK_LOCALISE.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"WROTE ALONG_TRACK_LOCALISE.json controls ALL_PASS={out['controls']['ALL_PASS']}")
    print("ZZL1PROBEDONEZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
