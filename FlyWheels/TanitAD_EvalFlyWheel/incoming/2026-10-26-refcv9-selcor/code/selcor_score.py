"""selcor_score.py -- PREREG_SELCOR.md: fit the selection correction on FRESH rows, score SELCOR-G on eval139 (s0 primary, s1 replicate).  CPU only.

    CUDA_VISIBLE_DEVICES=-1 R8_LAUNCH_TREE=<tree> PYTHONPATH='<tree>/stack;<tree>/taniteval;<battery>/code' python -X utf8 selcor_score.py --out <dir>
    ... --controls-only     # K-FRESH-ROLL / K-JOIN / K-PRIOR / K-ELIG / K-ADE / K-ID only: no fit, no bar

Operationalisations fixed before any number (DEVIATIONS.md): "ORACLE FD far below BASE" (section 6) = the ORACLE's FD cell vs BASE separated below;
the DER arm reuses the configuration CV chose for SELCOR-G (no second CV); v0 on both sides is the trace's own `v0_in` (the model's input).
TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in ("C:/Users/Admin/refcv8_pf/refcv9_l2/selfsharp/ssgate", "C:/Users/Admin/refcv8_pf/refcv9_l2/code", "C:/Users/Admin/refcv8_pf/refcv9_l1/code",
           "C:/Users/Admin/refcv8_pf/refcv9_l1c/code"):
    sys.path.insert(0, _p)
import l1c_fit_score as L1C  # noqa: E402  (load_fresh; sets F.DRY = False)
import ssgate_score as G  # noqa: E402  (helpers + constants; its main() is not run)
import l2b_score as B  # noqa: E402  (load_locus, eligible)

F = L1C.F
C, S = G.C, G.S
AL, PH = F.AL, F.PH
N_FAN, N_FRESH = 149, 10456
K_NONE = {0: (0.46948, 0.13955), 1: (0.47105, 0.13739)}     # BASE-os|stage (MD, FD), literal (PREREG_SELCOR section 4)
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
KS = (4, 8, 16)
LOCUS = "F:/refcv9_l2_locus/locus_s{}.npz"
L1_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
NEXT = {"PASS+REPLICATED": "SELCOR-G is a refcv8 deliverable candidate (score correction, no retraining): a SHIP-R8 amendment needs a PI go first; then "
                           "compose with the SELFSHARP progress lever as ONE sealed arm.",
        "PASS+NOT_REPLICATED": "the seed-0 gain is inside the inference-seed floor: report both words; no SHIP-R8 request.",
        "FAIL_P1_HEADROOM": "the correction cannot see the brake/cruise distinction from these features: next = a per-candidate vision feature, pre-registered first.",
        "FAIL_P1_NOHEADROOM": "this composition has no FD headroom: move to the lon decoder (refcv9 design input).",
        "FAIL_P2": "FD is traded for MD: read the FD/MD frontier over lambda on FIT rows; next = an MD-constrained fit.",
        "FAIL_G": "name the rows; next = a cross-fitted margin gate (SSGUARD's GUARD-CF recipe, FIT-side tau).",
        "INVALID": "a control failed: fix the instrument, re-score (0 GPU)."}


def guard(P_path, P_prog):
    return PH.at_progress(P_path, AL.arc(P_prog))


def cand_prog(fan, P0):
    """[n,N,8,2] candidates -> the native path P0 [n,8,2] re-sampled at each candidate's per-instant arc length."""
    n, N = fan.shape[:2]
    Q = guard(np.repeat(P0, N, axis=0).astype(np.float32), fan.reshape(-1, 8, 2))
    return np.asarray(Q, np.float32).reshape(n, N, 8, 2)


def staged_ade(Q, prior8, g6, chunk=300):
    """Q [n,N,8,2] -> staged S6 ADE [n,N] against g6 [n,6,2] (the battery's mean over the six instants, float32 as PR.components)."""
    n, N = Q.shape[:2]
    out = np.empty((n, N), np.float64)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        st = np.asarray(C.apply_stage(Q[a:b].reshape(-1, 8, 2), np.repeat(prior8[a:b], N, axis=0)), np.float32).reshape(b - a, N, 6, 2)
        out[a:b] = np.linalg.norm(st - g6[a:b, None].astype(np.float32), axis=-1).mean(-1)
    return out


def feats(fan, sel, lon6, kind, v0):
    n = len(sel)
    P0 = fan[np.arange(n), sel]
    a0 = AL.arc(P0)[:, -1]
    ac = AL.arc(fan.reshape(-1, 8, 2)).reshape(n, -1, 8)[..., -1]
    return {"lon": lon6.astype(np.int64), "kind": kind.astype(np.float64), "v0n": np.asarray(v0, np.float64) / 10.0,
            "d": (ac - a0[:, None]) / np.maximum(a0, 1.0)[:, None]}


# ------------------------------------------------------------------------------------------------------------- the correction (torch, float64)
def _tensors(X, sv3, elig, zp):
    import torch
    t = lambda a, dt=torch.float64: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)  # noqa: E731
    return {"sv3": t(sv3), "lon": t(X["lon"], torch.long), "kind": t(X["kind"]), "v0n": t(X["v0n"]), "d": t(X["d"]), "elig": t(elig, torch.bool),
            "zp": (t(zp) if zp is not None else None)}


def _logits(T, p, variant):
    import torch
    lg = p["a"] * T["sv3"] + p["b"][T["lon"]]
    if variant != "PRIOR":
        lg = lg + T["v0n"][:, None] * p["bv"][T["lon"]] + p["wd"] * T["d"] + p["wdv"] * T["d"] * T["v0n"][:, None] + p["wk"] * T["kind"]
    if variant == "Z":
        lg = lg + torch.gather(T["zp"] @ p["Wz"].T, 1, T["lon"])
    return lg.masked_fill(~T["elig"], float("-inf"))


def fit(T, y, variant, lam, k=0):
    import torch
    torch.manual_seed(0)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, requires_grad=True)  # noqa: E731
    p = {"a": torch.ones(1, dtype=torch.float64, requires_grad=True), "b": z(6), "bv": z(6), "wd": z(1), "wdv": z(1), "wk": z(1)}
    if variant == "Z":
        p["Wz"] = z(6, k)
    use = ["a", "b"] if variant == "PRIOR" else list(p)
    opt = torch.optim.LBFGS([p[n] for n in use], max_iter=300, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    yt = torch.as_tensor(y, dtype=torch.long)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(_logits(T, p, variant), yt) + lam * sum((p[n] ** 2).sum() for n in use if n != "a")
        loss.backward()
        return loss
    opt.step(closure)
    return {n: v.detach().clone() for n, v in p.items()}


def pick(T, p, variant):
    import torch
    with torch.no_grad():
        return torch.argmax(_logits(T, p, variant), 1).numpy()


def zpca_fit(Z, k):
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    U, s_, Vt = np.linalg.svd((Z - mu) / sd, full_matrices=False)
    V = Vt[:k].T
    sc = ((Z - mu) / sd @ V).std(0) + 1e-8
    return {"mu": mu, "sd": sd, "V": V, "sc": sc}


def zpca_apply(st, Z):
    return ((Z - st["mu"]) / st["sd"] @ st["V"]) / st["sc"]


def folds_of(eid, nf=5):
    ue = np.unique(eid)
    perm = np.random.default_rng(0).permutation(len(ue))
    f = {e: int(r % nf) for e, r in zip(ue[perm], range(len(ue)))}
    return np.array([f[e] for e in eid])


def sub_T(T, m):
    return {k: (v[m] if v is not None else None) for k, v in T.items()}


def derange_labels(y, eid, rng):
    n = len(y)
    perm = rng.permutation(n)
    for _ in range(50):
        bad = np.where(eid[perm] == eid)[0]
        if len(bad) == 0:
            break
        sw = rng.permutation(n)[:len(bad)]
        perm[bad], perm[sw] = perm[sw], perm[bad]
    return y[perm], int((eid[perm] == eid).sum())



def kprior_ulp(pr, hp, slots) -> bool:
    """PREREG_SELCOR AMENDMENT 1: |a - b| <= 8 * ulp32(max(|a|, |b|)) elementwise, and the same rule on the S6 arc lengths."""
    hp = np.asarray(hp)
    if hp.shape[1:] != (6, 2):
        return False
    a = pr[:, list(slots)].astype(np.float32)
    b = hp.astype(np.float32)
    tol = 8.0 * np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(np.float32)).astype(np.float64)
    ok_pts = bool((np.abs(a.astype(np.float64) - b.astype(np.float64)) <= tol).all())
    def arc(P):
        P0 = np.concatenate([np.zeros_like(P[:, :1]), P], 1).astype(np.float32)
        return np.linalg.norm(np.diff(P0, axis=1), axis=-1).astype(np.float32).cumsum(1, dtype=np.float32)
    aa, ab = arc(a), arc(b)
    tola = 8.0 * np.spacing(np.maximum(np.abs(aa), np.abs(ab))).astype(np.float64)
    return bool(ok_pts and (np.abs(aa.astype(np.float64) - ab.astype(np.float64)) <= tola).all())

# ------------------------------------------------------------------------------------------------------------- loading
def load_fit_rows(ctl, fresh_trace):
    import compliant_reread_r8 as CR  # noqa: F401  (the battery's own loaders, as l1c_fit_score.main)
    import retime_trainfit_r8 as TF
    tools = F.RT.tools_load()
    TF.TRAIN_ROOT, TF.TRAIN_TREE = "F:/l1c/tree", "step48852ema_fresh"
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    fr = L1C.load_fresh(tools, CR2, TF, "S6", {})
    trf = B.load_locus(fresh_trace)
    n = trf["n"]
    mt = trf["meta"]
    pc = mt["present_counts"]
    eq_sel = np.array_equal(trf["fan"][np.arange(n), trf["sel"]], trf["plan8"])
    zb = F.load_bank("F:/l1c/bank/ztac_fresh.npz")
    ks = [k for k in trf["idx"] if k in zb["idx"]]
    d = np.array([np.abs(trf["plan8"][trf["idx"][k]].astype(np.float64) - zb["traj"][zb["idx"][k]]).max() for k in ks])
    ctl["K-FRESH-ROLL"] = {"n": n, "final": mt.get("final"), "present_counts": pc, "fan_n_seen": mt["counters"]["fan_n_seen"], "batch_gt1": mt["counters"]["batch_gt1"],
                           "emitted_equals_fan_at_sel": bool(eq_sel), "n_vs_l1c_bank": len(ks), "n_bit_equal_l1c_bank": int((d == 0).sum()),
                           "max_abs_vs_l1c_bank": float(d.max()) if len(d) else None}
    ctl["K-FRESH-ROLL"]["pass"] = bool(n == N_FRESH and mt.get("final") is True and all(pc.get(k, 0) == n for k in ("fan", "sel_score_v3", "prior_path", "kind", "lon6"))
                                       and mt["counters"]["fan_n_seen"] == [N_FAN] and mt["counters"]["batch_gt1"] == 0 and eq_sel and len(ks) == n and (d == 0).all())
    keys = list(zip(fr["clip"].tolist(), fr["ws"].tolist()))
    j = np.array([trf["idx"].get(k, -1) for k in keys])
    ctl["K-JOIN"] = {"n_fit_rows": len(keys), "n_unjoined": int((j < 0).sum()), "n_distinct": int(len(set(j.tolist()))), "pass": bool((j >= 0).all() and len(set(j.tolist())) == len(j))}
    jz = np.array([zb["idx"].get(k, -1) for k in keys])
    j = np.where(j < 0, 0, j)
    pr = trf["prior_path"][j]
    dp = np.abs(pr[:, C.S6_SLOTS].astype(np.float64) - np.asarray(fr["HP"], np.float64)).max() if np.asarray(fr["HP"]).shape[1:] == (6, 2) else float("inf")
    ctl["K-PRIOR"] = {"max_abs_m": float(dp), "HP_shape": list(np.asarray(fr["HP"]).shape), "pass": kprior_ulp(pr, fr["HP"], C.S6_SLOTS)}
    # PREREG_SELCOR AMENDMENT 1: 8 float32 ULP elementwise (+ the same on S6 arc lengths); the one-slot-shifted mutation must FAIL
    ctl["K-PRIOR"]["mutation_shift_minus1_pass"] = kprior_ulp(pr, fr["HP"], [s - 1 for s in C.S6_SLOTS])
    ctl["K-PRIOR"]["mutation_red"] = not ctl["K-PRIOR"]["mutation_shift_minus1_pass"]
    _d = np.abs(pr[:, C.S6_SLOTS].astype(np.float64) - np.asarray(fr["HP"], np.float64))
    ctl["K-PRIOR"].update({"per_slot_max_abs_m": [float(x) for x in _d.max(axis=(0, 2))], "share_rows_bit_equal": float((_d.max(axis=(1, 2)) == 0).mean()),
                           "share_rows_le_1e-5": float((_d.max(axis=(1, 2)) <= 1e-5).mean()), "rule": "|a-b| <= 8*ulp32(max(|a|,|b|)) elementwise and on S6 arcs (AMENDMENT 1)"})
    ctl["K-PRIOR"]["pass"] = bool(ctl["K-PRIOR"]["pass"] and ctl["K-PRIOR"]["mutation_red"])
    return fr, trf, j, (zb["z"][jz] if (jz >= 0).all() else None)


def win(tr, j, v0=None):
    elig = B.eligible(tr["sel_score_v3"][j], tr["reach_keep"][j], tr["ceil_keep"][j], tr["emit_keep"][j])
    sv3 = tr["sel_score_v3"][j].astype(np.float64)
    sel = tr["sel"][j]
    mx = np.where(elig, sv3, -np.inf).max(1)
    X = feats(tr["fan"][j], sel, tr["lon6"][j], tr["kind"][j], tr["v0_in"][j] if v0 is None else v0)
    return {"fan": tr["fan"][j], "sv3": sv3, "elig": elig, "sel": sel, "prior": tr["prior_path"][j], "X": X,
            "elig_ok": bool((sv3[np.arange(len(sel)), sel] == mx).all())}


# ------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--fresh-trace", default="F:/refcv9_selcor/locus_fresh_s0.npz")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = G.sha(ROOT / "PREREG_SELCOR.md")
    if pre != (ROOT / "PREREG_SELCOR.sha256").read_text().split()[0] and not a.controls_only:
        raise SystemExit("[selcor] PREREG_SELCOR.md does not match its seal: refusing")
    am1 = G.sha(ROOT / "PREREG_SELCOR_AMENDMENT1.md")
    if am1 != (ROOT / "PREREG_SELCOR_AMENDMENT1.sha256").read_text().split()[0] and not a.controls_only:
        raise SystemExit("[selcor] PREREG_SELCOR_AMENDMENT1.md does not match its seal: refusing")
    st = C.stack()
    PR = st["PR"]
    rows = S.metric_rows(PR)
    keys = [k for (_f, k, _l, _u, _lw, _s) in PR.METRICS]
    ctl = {}
    # ---------------------------------------------------------------- FIT side
    fr, trf, jf, zfit = load_fit_rows(ctl, a.fresh_trace)
    Wf = win(trf, jf)
    ctl["K-ELIG"] = {"fit": Wf["elig_ok"]}
    eid_f = np.asarray(fr["eid"])
    # ---------------------------------------------------------------- SCORE side (both seeds)
    ev = {}
    for sd in (0, 1):
        E = C.load_eval_pf(sd)
        b1 = S.load_l1_base(str(L1_BANK), sd)
        p_base = S.join_plan8(b1, E)
        tr = B.load_locus(LOCUS.format(sd))
        ek = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
        je = np.array([tr["idx"].get(k, -1) for k in ek])
        zb = F.load_bank(L1_BANK / f"ztac_eval_s{sd}.npz")
        jz = np.array([zb["idx"].get(k, -1) for k in ek])
        We = win(tr, je)
        nat_eq = bool(np.array_equal(We["fan"][np.arange(len(je)), We["sel"]], p_base))
        cells = lambda P: S.arm_cells(E, P)["stage"]  # noqa: E731
        c_base = cells(p_base)
        r = lambda c: (round(S.rate(c, "missed_decel"), 5), round(S.rate(c, "false_decel"), 5))  # noqa: E731
        T0 = _tensors(We["X"], We["sv3"], We["elig"], None)
        p0 = {"a": __import__("torch").ones(1, dtype=__import__("torch").float64), "b": __import__("torch").zeros(6, dtype=__import__("torch").float64)}
        pk0 = pick(T0, p0, "PRIOR")
        P_id = guard(p_base, We["fan"][np.arange(len(je)), pk0]).astype(p_base.dtype)
        ctl[f"K-ID_s{sd}"] = {"picks_equal_sel": bool(np.array_equal(pk0, We["sel"])), "got": r(cells(P_id)), "want": K_NONE[sd]}
        ctl[f"K-ID_s{sd}"]["pass"] = bool(ctl[f"K-ID_s{sd}"]["picks_equal_sel"] and ctl[f"K-ID_s{sd}"]["got"] == K_NONE[sd])
        ctl[f"K-EVALJOIN_s{sd}"] = {"n": len(je), "unjoined_locus": int((je < 0).sum()), "unjoined_ztac": int((jz < 0).sum()), "native_equals_L1_BASEos": nat_eq,
                                    "pass": bool((je >= 0).all() and (jz >= 0).all() and nat_eq)}
        ctl["K-ELIG"][f"eval_s{sd}"] = We["elig_ok"]
        if sd == 0:
            comp = np.asarray(E["compliant"], bool)
            mine = staged_ade(p_base[:, None], E["prior8"], E["g"])[:, 0]
            dd = np.abs(mine - c_base["ade_m"])[comp]
            ctl["K-ADE"] = {"n": int(comp.sum()), "max_abs_m": float(dd.max()), "pass": bool(dd.max() <= 1e-5)}
        ev[sd] = {"E": E, "p_base": p_base, "W": We, "z": zb["z"][np.where(jz < 0, 0, jz)], "c_base": c_base, "cells": cells, "r": r, "eid": np.asarray(E["eid"])}
    ctl["K-ELIG"]["pass"] = bool(all(v for k, v in ctl["K-ELIG"].items() if k != "pass"))
    res = {"tool": "selcor_score.py", "prereg_sha256": pre, "script_sha256": G.sha(__file__), "estimator": G.ESTIMATOR,
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)", "controls": ctl,
           "families_not_computed": {"LONGITUDINAL distance-keeping (headway/TTC)": "no lead-agent track in the banked refcv8 dumps; n = 0",
                                     "STRATEGIC": "refcv8 runs --no-strategic; n = 0"},
           "fit_population": {"rows": int(len(jf)), "episodes": int(len(np.unique(eid_f)))}}
    ctl_fail = sorted(k for k, v in ctl.items() if isinstance(v, dict) and v.get("pass") is False)
    print(json.dumps({k: v.get("pass") for k, v in ctl.items()}), flush=True)
    if a.controls_only:
        C.jdump(res, out / "SELCOR_CONTROLS.json")
        return 0 if not ctl_fail else 6
    # ---------------------------------------------------------------- labels on FIT rows (GT used ONLY here)
    g6 = np.asarray(fr["G"], np.float64)
    print(f"[selcor] fit rows {len(jf)}; staging {len(jf) * N_FAN} candidates", flush=True)
    Qf = cand_prog(Wf["fan"], Wf["fan"][np.arange(len(jf)), Wf["sel"]])
    adef = staged_ade(Qf, Wf["prior"], g6)
    adef_m = np.where(Wf["elig"], adef, np.inf)
    y = adef_m.argmin(1)
    res["fit_population"].update({"label_ade_native_mean": float(adef[np.arange(len(jf)), Wf["sel"]].mean()), "label_ade_oracle_mean": float(adef_m.min(1).mean()),
                                  "label_is_native_share": float((y == Wf["sel"]).mean()),
                                  "label_lon6_shares": np.bincount(Wf["X"]["lon"][np.arange(len(jf)), y], minlength=6).tolist()})
    # ---------------------------------------------------------------- CV on FIT rows only
    fold = folds_of(eid_f)
    configs = [("S", 0, lam) for lam in LAMS] + [("Z", k, lam) for k in KS for lam in LAMS]
    cv = []
    for variant, k, lam in configs:
        tot, cnt = 0.0, 0
        for f in range(5):
            trm, tem = fold != f, fold == f
            zp_tr = zp_te = None
            if variant == "Z":
                stz = zpca_fit(zfit[trm], k)
                zp_tr, zp_te = zpca_apply(stz, zfit[trm]), zpca_apply(stz, zfit[tem])
            Ttr = _tensors({kk: v[trm] for kk, v in Wf["X"].items()}, Wf["sv3"][trm], Wf["elig"][trm], zp_tr)
            Tte = _tensors({kk: v[tem] for kk, v in Wf["X"].items()}, Wf["sv3"][tem], Wf["elig"][tem], zp_te)
            p = fit(Ttr, y[trm], variant, lam, k)
            pk = pick(Tte, p, variant)
            tot += float(adef[np.where(tem)[0], pk].sum())
            cnt += int(tem.sum())
        cv.append({"variant": variant, "k": k, "lam": lam, "cv_ade": tot / cnt})
        print(f"[selcor] CV {variant} k={k} lam={lam:g} ade={tot / cnt:.4f}", flush=True)
    cv_native = float(adef[np.arange(len(jf)), Wf["sel"]].mean())
    best = min(cv, key=lambda c: (round(c["cv_ade"], 9), 0 if c["variant"] == "S" else 1, c["k"], -c["lam"]))
    res["cv"] = {"grid": cv, "chosen": best, "native_ade_on_fit_rows": cv_native}
    # ---------------------------------------------------------------- final fits (all FIT rows)
    stz = zpca_fit(zfit, best["k"]) if best["variant"] == "Z" else None
    Tf = _tensors(Wf["X"], Wf["sv3"], Wf["elig"], zpca_apply(stz, zfit) if stz else None)
    arms_p = {"SELCOR-G": fit(Tf, y, best["variant"], best["lam"], best["k"]),
              "PRIOR-ONLY": fit(Tf, y, "PRIOR", best["lam"])}
    y_der, n_same = derange_labels(y, eid_f, np.random.default_rng(0))
    arms_p["DER"] = fit(Tf, y_der, best["variant"], best["lam"], best["k"])
    res["fitted"] = {k: {n: v.numpy().round(6).tolist() for n, v in p.items()} for k, p in arms_p.items()}
    res["der_same_episode_labels"] = n_same
    var_of = {"SELCOR-G": best["variant"], "PRIOR-ONLY": "PRIOR", "DER": best["variant"]}
    # ---------------------------------------------------------------- score
    for sd in (0, 1):
        V = ev[sd]
        We, p_base, c_base, cells, r, eid = V["W"], V["p_base"], V["c_base"], V["cells"], V["r"], V["eid"]
        Te = _tensors(We["X"], We["sv3"], We["elig"], zpca_apply(stz, V["z"]) if stz else None)
        n = len(We["sel"])
        rs = {"BASE": {"MD": r(c_base)[0], "FD": r(c_base)[1]}}
        for nm, p in arms_p.items():
            pk = pick(Te, p, var_of[nm])
            P = guard(p_base, We["fan"][np.arange(n), pk]).astype(p_base.dtype)
            cA = cells(P)
            cmp, worse = G.g_worse(PR, eid, cA, c_base, rows, keys)
            rs[nm] = {"MD": r(cA)[0], "FD": r(cA)[1], "picks_changed_share": float((pk != We["sel"]).mean()),
                      "P1": bool(cmp["false_decel"]["hi"] < 0), "P2": bool(not (cmp["missed_decel"]["lo"] > 0)), "G_rows_separated_worse": worse,
                      "G_strict": not worse, "comparison_vs_BASE": cmp}
        Qe = cand_prog(We["fan"], p_base)
        ade_e = np.where(We["elig"], staged_ade(Qe, V["E"]["prior8"], V["E"]["g"]), np.inf)
        po = ade_e.argmin(1)
        cO = cells(guard(p_base, We["fan"][np.arange(n), po]).astype(p_base.dtype))
        cmpO, _w = G.g_worse(PR, eid, cO, c_base, rows, keys)
        rs["ORACLE"] = {"MD": r(cO)[0], "FD": r(cO)[1], "FD_cell": cmpO["false_decel"], "MD_cell": cmpO["missed_decel"], "ade_cell": cmpO.get("ade_m")}
        res[f"s{sd}"] = rs
        print(f"[selcor] s{sd} " + json.dumps({k: (v["MD"], v["FD"]) for k, v in rs.items()}), flush=True)
    # ---------------------------------------------------------------- verdict
    s0, s1 = res["s0"]["SELCOR-G"], res["s1"]["SELCOR-G"]
    passed = s0["P1"] and s0["P2"] and s0["G_strict"]
    repl = s1["P1"] and s1["G_strict"]
    if ctl_fail:
        word, nl = f"INVALID (control failed: {ctl_fail})", NEXT["INVALID"]
    elif passed:
        word = "PASS (s0) / " + ("REPLICATED (s1)" if repl else "NOT REPLICATED (s1)")
        nl = NEXT["PASS+REPLICATED" if repl else "PASS+NOT_REPLICATED"]
    else:
        failing = [k for k in ("P1", "P2", "G_strict") if not s0[k]]
        word = "FAIL (s0: " + ", ".join(failing) + ")"
        if not s0["P1"]:
            nl = NEXT["FAIL_P1_HEADROOM" if res["s0"]["ORACLE"]["FD_cell"]["hi"] < 0 else "FAIL_P1_NOHEADROOM"]
        else:
            nl = NEXT["FAIL_P2"] if not s0["P2"] else NEXT["FAIL_G"]
    res.update(verdict=word, next_lever=nl, wall_s=round(time.time() - t0, 1))
    C.jdump(res, out / "SELCOR_RESULT.json")
    fmt = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c and "delta" in c else "-"  # noqa: E731
    Lx = [word, "", f"# SELCOR RESULT (mechanical; PREREG_SELCOR.md sha256 {pre[:12]})", "", f"tier: {res['tier']}", f"estimator: {G.ESTIMATOR}", "",
          f"FIT: {res['fit_population']['rows']} fresh S6 compliant rows, {res['fit_population']['episodes']} episodes; CV chose {json.dumps(best)} "
          f"(native CV ADE {cv_native:.4f})", ""]
    for sd in (0, 1):
        rs = res[f"s{sd}"]
        Lx += [f"## seed {sd}" + (" (primary)" if sd == 0 else " (replicate)"), "", "| arm | MD | FD | picks changed | P1 | P2 | G-strict |", "|---|---|---|---|---|---|---|",
               f"| BASE-os | {rs['BASE']['MD']:.4f} | {rs['BASE']['FD']:.4f} | - | - | - | - |"]
        for nm in ("SELCOR-G", "PRIOR-ONLY", "DER"):
            x = rs[nm]
            Lx.append(f"| {nm} | {x['MD']:.4f} | {x['FD']:.4f} | {x['picks_changed_share']:.3f} | {x['P1']} | {x['P2']} | {x['G_strict']} ({', '.join(x['G_rows_separated_worse']) or 'none'}) |")
        Lx += [f"| ORACLE (GT, ceiling) | {rs['ORACLE']['MD']:.4f} | {rs['ORACLE']['FD']:.4f} | - | - | - | - |", "",
               "| S6 row (SELCOR-G vs BASE) | delta [95% CI] | worse |", "|---|---|---|"]
        Lx += [f"| {k} | {fmt(c)} | {c.get('worse')} |" for k, c in rs["SELCOR-G"]["comparison_vs_BASE"].items()] + [""]
    Lx += [f"controls: {json.dumps({k: v.get('pass') for k, v in ctl.items()})}", "", f"NEXT: {nl}", "", "ZZSELCORENDZZ"]
    (out / "VERDICT.md").write_text("\n".join(Lx), encoding="utf-8")
    print(word)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
