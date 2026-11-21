"""P3 -- does `progress` bind on the COLLISION objective?

0 GPU, on the banked fan. The brief's prior finding is about the FEASIBILITY flags
(rho(progress, contact) = +0.3286 etc). The question here is narrower and is the one
that decides the arm: WITHIN a window, does each reward term rank COLLIDING candidates
higher or lower than non-colliding ones?

rho > 0  =>  the term prefers colliders  =>  it fights a collision objective  =>  BINDS
rho < 0  =>  the term already disprefers colliders             =>  does not bind

Computed BOTH on the unprojected fan (the banked c_* terms) and on the projected fan
(P2's substrate), because the answer may differ once the feasibility confound is gone.

CONTROLS:
  K1 CONSTANT     : a constant term must read rho = 0 EXACTLY (no-information value).
  K2 ORACLE       : -contact itself must read rho = -1 EXACTLY (the perfect ranker).
  K3 POPULATION   : only windows with BOTH a collider and a non-collider can carry a
                    rank correlation at all; n is printed and windows outside it are
                    excluded rather than silently scored as 0.
"""
import os, sys, json, argparse, importlib.util

_REPO = os.environ.get("TANITAD_REPO") or r"C:\Users\Admin\tanitad-wt"
for _p in (os.path.join(_REPO, "stack"), os.path.join(_REPO, "taniteval"),
           os.path.join(_REPO, "taniteval", "tools")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="raw/fan_bank_base_240w.npz")
ap.add_argument("--out", default="raw/p3_progress_binds.json")
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FS = _load_by_path("fan_safety_for_p3", os.path.join(_REPO, "taniteval", "tools", "fan_safety.py"))
from tanitad.refs import feasible_decode as FD
from tanitad.rl import rewards as RW, DEFAULT_WEIGHTS

z = np.load(args.bank, allow_pickle=True)
fan2 = torch.from_numpy(z["fan2"]).double()
v0 = torch.from_numpy(np.asarray(z["v0"])).double()
lead5 = torch.from_numpy(z["lead5"]).double()
has_lead = torch.from_numpy(z["has_lead"]).bool()
eid = np.asarray(z["eid"])
W, K = fan2.shape[0], fan2.shape[1]
print("OBJECT %s  fan2%s  DEFAULT_WEIGHTS=%s" % (args.bank, tuple(fan2.shape), DEFAULT_WEIGHTS))


def contact_of(paths):
    v0b = v0[:, None].expand(W, K)
    l5 = lead5[:, None, :, :].expand(W, K, 5, 2)
    sc = FS.score_paths(paths, v0b, l5)
    return (sc["contact"] & has_lead[:, None])


def terms_of(paths):
    """Every reward component on the SAME object, via the same COMPONENTS table the
    trainer uses. ctx mirrors fan_safety.score_paths."""
    ctx = {"dt": FS.DT_S, "lead_len_m": FS.LEAD_LEN_DEFAULT_M,
           "lead_path": lead5[:, None, :, :].expand(W, K, 5, 2), "v0": v0[:, None].expand(W, K)}
    out = {}
    for name in ("progress", "collision", "headway", "feasibility", "comfort"):
        out[name] = RW.COMPONENTS[name](paths, ctx).double()
    out["_composed_default"] = sum(DEFAULT_WEIGHTS[n] * out[n] for n in DEFAULT_WEIGHTS)
    out["_collision_only"] = out["collision"]
    out["_default_minus_progress"] = sum(
        DEFAULT_WEIGHTS[n] * out[n] for n in DEFAULT_WEIGHTS if n != "progress")
    return out


def rank_corr(score, flag, pop_w):
    """TIE-SAFE AUC: P(score of a COLLIDER > score of a NON-COLLIDER), ties counted 0.5.

    ⛔ The previous implementation used `argsort(argsort(x))` ordinal ranks. The flag is
    binary and ~97 % ties, and `np.argsort` is STABLE, so a CONSTANT score received the
    ranks 0..K-1 in index order and correlated with wherever the colliders happened to
    sit -- the K1 constant control read +0.6464 instead of 0, and the K2 oracle read
    +0.0810 instead of -1. Both controls caught it. AUC handles ties exactly:
      0.5 = no information (K1) ; 0.0 = perfectly disprefers colliders (K2) ;
      >0.5 = the term ranks COLLIDERS HIGHER, i.e. it BINDS against a collision objective.
    Returned per window; windows without both classes are nan (K3).
    """
    aucs = np.full(W, np.nan)
    for w in range(W):
        if not pop_w[w]:
            continue
        f = flag[w].numpy().astype(bool)
        n1, n0 = int(f.sum()), int((~f).sum())
        if n1 == 0 or n0 == 0:
            continue
        s = score[w].numpy().astype(float)
        if not np.isfinite(s).all():
            finite = s[np.isfinite(s)]
            s = np.where(np.isfinite(s), s, (finite.min() - 1.0) if finite.size else 0.0)
        pos, neg = s[f][:, None], s[~f][None, :]
        aucs[w] = float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (n1 * n0))
    return aucs


def boot_mean(vals, ep, n_boot, seed):
    ok = np.isfinite(vals)
    v, e = vals[ok], ep[ok]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), False, 0
    rng = np.random.default_rng(seed)
    ue = np.unique(e)
    idx = [np.nonzero(e == x)[0] for x in ue]
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(ue), len(ue))
        out[i] = v[np.concatenate([idx[j] for j in pick])].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0), int(v.size)


res = {}
for label, paths in (("unprojected", fan2),
                     ("projected", FD.project_feasible(fan2, v0[:, None].expand(W, K), enabled=True))):
    contact = contact_of(paths)
    T = terms_of(paths)
    n_mixed = int(((contact.sum(1) > 0) & (contact.sum(1) < K)).sum())
    print("\n=== %s ===  contact rate %.6f   windows with BOTH classes (K3 population) = %d"
          % (label, float(contact.double().mean()), n_mixed))
    T["_K1_constant"] = torch.ones_like(T["progress"])
    T["_K2_oracle"] = -contact.double()
    r = {}; controls = {}
    for name in ("progress", "collision", "headway", "feasibility", "comfort",
                 "_composed_default", "_default_minus_progress", "_K1_constant", "_K2_oracle"):
        rh = rank_corr(T[name], contact, np.ones(W, bool))
        m, lo, hi, sep, n = boot_mean(rh, eid, args.n_boot, args.seed)
        r[name] = {"rho": m, "lo": lo, "hi": hi, "sep": sep, "n_windows": n}
        tag = ""
        if name == "_K1_constant":
            ok = abs(m - 0.5) < 1e-12; controls["K1_constant"] = {"auc": m, "pass": bool(ok)}
            tag = "  <- K1 must be 0.5000 EXACTLY: %s" % ("PASS" if ok else "FAIL")
        if name == "_K2_oracle":
            ok = abs(m) < 1e-12; controls["K2_oracle"] = {"auc": m, "pass": bool(ok)}
            tag = "  <- K2 must be 0.0000 EXACTLY: %s" % ("PASS" if ok else "FAIL")
        binds = "BINDS" if (m > 0.5 and sep) else ("ok" if (m < 0.5 and sep) else "ns")
        print("  AUC(%-24s vs contact) = %0.4f  [%0.4f,%0.4f] sep=%-5s n=%-3d %-5s%s"
              % (name, m, lo, hi, sep, n, binds, tag))
    res[label] = {"contact_rate": float(contact.double().mean()),
                  "n_mixed_windows": n_mixed, "auc": r, "controls": controls,
                  "controls_pass": bool(all(v["pass"] for v in controls.values()))}
    print("  CONTROLS: %s" % ("PASS" if all(v["pass"] for v in controls.values()) else "FAIL"))

out = {"_tool": "p3_progress_binds.py",
       "_tier": "T0 ranking readout on the EMITTED fan (never a driving claim)",
       "_evidence_class": "MEASURED (ours)",
       "bank": args.bank, "W": W, "K": K,
       "default_weights": dict(DEFAULT_WEIGHTS),
       "estimator": "episode-cluster bootstrap over per-window tie-safe AUC",
       "n_boot": args.n_boot, "seed": args.seed,
       "reading": "AUC = P(score of a collider > score of a non-collider), ties 0.5. "
                  "0.5 = no information; >0.5 means the term ranks COLLIDERS HIGHER, "
                  "i.e. it fights a collision objective and BINDS; <0.5 means it "
                  "already disprefers colliders.",
       "results": res}
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print("\nWROTE %s" % args.out)
