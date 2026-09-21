"""Which of the QUIET fields are correctly quiet, and which are genuinely failing?

`3e3dac9` cleared `v_rel_y` (alignment 0.088): even target-side geometry cannot predict it
linearly, so there was nothing to miss. It explicitly left `yaw_rate_rel` (0.025), `w` (0.246) and
`occluded` (0.284) UNEXAMINED and said they may be either kind.

⭐ THIS CONVERTS THE SPECTRUM INTO A DEFECT LIST, which is what specifying a head change needs.
Same upper-bound logic: predict each field's TARGET from the OTHER target-side features — which
the model never sees at inference and which bound what an image could supply.

  probe FAILS  -> correctly quiet; the head cannot be blamed
  probe PASSES -> signal exists that the head is not using ⇒ a genuine defect

⭐ `w` IS THE SHARP CASE: `cls` predicts size strongly (cars have typical widths), so if the probe
passes there while the head is quiet, that is a real failure rather than absent signal.

⛔ Same discipline as `vy_learnable.py`: episode-DISJOINT fit/score, λ on the FIT split only via an
inner episode split, a CONSTANT control that must read the no-information value, n and d printed,
and the function class (LINEAR ridge) stated.
⛔ No model pass, no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics as st
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
N_WIN = 300
ALIGN = {"yaw_rate_rel": 0.02459, "w": 0.2462, "l": 1.38458, "v_rel_y": 0.08807}


def probe(X, Y, EPI, label, align):
    eps = sorted(set(EPI.tolist()))
    half = set(eps[: len(eps) // 2])
    fit = np.array([e in half for e in EPI])
    sc = ~fit
    if fit.sum() < 30 or sc.sum() < 30:
        return {"field": label, "note": "split too small", "n": int(X.shape[0])}
    fe = sorted(set(EPI[fit].tolist()))
    inner = set(fe[: len(fe) // 2])
    f2 = np.array([e in inner for e in EPI]) & fit
    v2 = fit & ~f2
    best, bm = None, float("inf")
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]:
        A = X[f2].T @ X[f2] + lam * np.eye(X.shape[1])
        b = np.linalg.solve(A, X[f2].T @ Y[f2])
        m = float(np.abs(X[v2] @ b - Y[v2]).mean())
        if m < bm:
            best, bm = lam, m
    beta = np.linalg.solve(X[fit].T @ X[fit] + best * np.eye(X.shape[1]), X[fit].T @ Y[fit])
    mae_p = float(np.abs(X[sc] @ beta - Y[sc]).mean())
    const = float(np.median(Y[fit]))
    mae_c = float(np.abs(Y[sc] - const).mean())
    by = collections.defaultdict(list)
    for e, a, b_ in zip(EPI[sc], np.abs(X[sc] @ beta - Y[sc]), np.abs(Y[sc] - const)):
        by[int(e)].append(b_ - a)
    rng = random.Random(20260921)
    keys = list(by)
    ms = sorted(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by[k])
                for _ in range(4000)) if len(keys) > 1 else None
    ci = [round(ms[100], 4), round(ms[3900], 4)] if ms else None
    beats = bool(ci and ci[0] > 0)
    return {"field": label, "alignment": align, "n_pairs": int(X.shape[0]), "d": int(X.shape[1]),
            "n_episodes": len(eps), "lambda": best,
            "MAE_probe": round(mae_p, 4), "MAE_constant_control": round(mae_c, 4),
            "gain_over_constant": round(mae_c - mae_p, 4), "CI95": ci,
            "probe_beats_constant": beats,
            "reading": ("⛔ SIGNAL EXISTS that the head is not using ⇒ GENUINE DEFECT" if beats
                        else "correctly quiet — nothing here to miss")}


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]

    it0 = corp.ds[wis[0]]
    print("target shapes:", {k: tuple(v.shape) for k, v in it0.items()
                             if hasattr(v, "shape") and "agent" in k})
    n_rate = int(it0["agent_rates"].shape[-1])
    print(f"agent_rates width = {n_rate}")

    rows = []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        box = item["agent_box"].float().numpy()
        val = item["agent_valid"].numpy().astype(bool)
        rt = item["agent_rates"].float().numpy()
        rm = item["agent_rates_mask"].numpy().astype(bool)
        cls = item["agent_cls"].numpy()
        for j in range(box.shape[0]):
            if not val[j]:
                continue
            ok = bool(rm[j][0]) if rm.ndim > 1 else bool(rm[j])
            cx, cy, l, w_ = (float(box[j, 0]), float(box[j, 1]),
                             float(box[j, 2]), float(box[j, 3]))
            rows.append({"e": int(e_i), "cx": cx, "cy": cy, "rng": float(np.hypot(cx, cy)),
                         "l": l, "w": w_, "cls": float(cls[j]),
                         "vx": float(rt[j, 0]) if ok else None,
                         "vy": float(rt[j, 1]) if ok else None,
                         "yr": float(rt[j, 2]) if (ok and n_rate > 2) else None})
    print(f"collected {len(rows)} valid targets")

    out = []
    # w: predicted from geometry + LENGTH + CLASS (class predicts size strongly)
    sel = rows
    X = np.array([[r["cx"], r["cy"], r["rng"], r["l"], r["cls"], 1.0] for r in sel])
    out.append(probe(X, np.array([r["w"] for r in sel]),
                     np.array([r["e"] for r in sel]), "w", ALIGN["w"]))
    # l: same, swapping the size partner
    X = np.array([[r["cx"], r["cy"], r["rng"], r["w"], r["cls"], 1.0] for r in sel])
    out.append(probe(X, np.array([r["l"] for r in sel]),
                     np.array([r["e"] for r in sel]), "l", ALIGN["l"]))
    # yaw_rate_rel: from geometry + both velocities + class
    s2 = [r for r in sel if r["yr"] is not None]
    if s2:
        X = np.array([[r["cx"], r["cy"], r["rng"], r["vx"], r["vy"], r["cls"], 1.0] for r in s2])
        out.append(probe(X, np.array([r["yr"] for r in s2]),
                         np.array([r["e"] for r in s2]), "yaw_rate_rel",
                         ALIGN["yaw_rate_rel"]))
    else:
        out.append({"field": "yaw_rate_rel", "note": "no rate channel 2 available"})

    res = {"_what": "which QUIET fields are correctly quiet and which are genuinely failing?",
           "_evidence_class": "MEASURED (ours), CPU, no model pass",
           "_function_class": "LINEAR (ridge); a negative is about linear predictability only",
           "_logic": "target-side features bound what an image could supply ⇒ probe fails means "
                     "the head cannot be blamed",
           "fields": out}
    print(json.dumps(res, indent=1))
    for r in out:
        if "reading" in r:
            print(f"  {r['field']:<14} align {r['alignment']:>8.5f}  {r['reading']}")
    pathlib.Path("quiet_fields.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
