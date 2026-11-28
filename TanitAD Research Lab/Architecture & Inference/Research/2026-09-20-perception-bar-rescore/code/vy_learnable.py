"""Is `v_rel_y` PREDICTABLE AT ALL, or did the head fail? An UPPER-BOUND probe.

`4a7166a` measured `v_rel_y` at skill **0.0001** against a zero floor — the no-information value,
essentially exactly — and explicitly declined to call the head broken: lateral relative velocity
may simply carry little signal at this geometry (its floor is 2.19 against `v_rel_x`'s 7.51).

⭐ THE ASYMMETRY THAT MAKES THIS CHEAP AND DECISIVE. Probe `v_rel_y` from TARGET-SIDE features the
model never sees at inference — the target's own `cx, cy, range, v_rel_x, cls`. These are strictly
MORE informative than an image about the target's own kinematics. ⇒

  probe FAILS  -> there is nothing here to find; the HEAD CANNOT BE BLAMED
  probe PASSES -> signal exists in the geometry and the head missed it (does NOT prove an image
                  would carry it — an upper bound is not a target)

⛔ PROGRAMME PROBE DISCIPLINE (2026-08-22, which caught four estimator bugs):
  * fit and score on EPISODE-DISJOINT splits — never tune on the scored half;
  * a CONSTANT-ONLY control that must read the no-information value;
  * lambda chosen on the FIT split only, never on the scored one;
  * n and d printed, because n << d is underpowered BY CONSTRUCTION, not a negative.
⚠️ And the function class is stated: this is a LINEAR (ridge) probe. A negative is a negative
about LINEAR predictability from these features, not about learnability in general.
⛔ No model pass, no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
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


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    wis = [w for w in SP.trainer_windows(corp.ds, 1500)
           if corp.eligibility(w) is None][:N_WIN]

    X, Y, EPI = [], [], []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        box = item["agent_box"].float().numpy()
        val = item["agent_valid"].numpy().astype(bool)
        rates = item["agent_rates"].float().numpy()
        rm = item["agent_rates_mask"].numpy().astype(bool)
        cls = item["agent_cls"].numpy()
        for j in range(box.shape[0]):
            ok = bool(rm[j][0]) if rm.ndim > 1 else bool(rm[j])
            if not (val[j] and ok):
                continue
            cx, cy = float(box[j, 0]), float(box[j, 1])
            X.append([cx, cy, float(np.hypot(cx, cy)), float(rates[j, 0]),
                      float(cls[j]), 1.0])
            Y.append(float(rates[j, 1]))
            EPI.append(int(e_i))
    X, Y, EPI = np.asarray(X), np.asarray(Y), np.asarray(EPI)
    assert X.shape[0] > 50, f"ZZABORT too few rows: {X.shape[0]}"

    eps = sorted(set(EPI.tolist()))
    half = set(eps[: len(eps) // 2])
    fit = np.array([e in half for e in EPI])
    sc = ~fit
    print(f"n = {X.shape[0]} pairs, d = {X.shape[1]} features, "
          f"{len(eps)} episodes -> fit {fit.sum()} / scored {sc.sum()} (EPISODE-DISJOINT)")
    assert fit.sum() > 20 and sc.sum() > 20, "ZZABORT a split is too small"

    # lambda on the FIT split only, by an inner episode-disjoint split
    fe = sorted(set(EPI[fit].tolist()))
    inner = set(fe[: len(fe) // 2])
    f2 = np.array([e in inner for e in EPI]) & fit
    v2 = fit & ~f2
    best, best_mae = None, float("inf")
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]:
        A = X[f2].T @ X[f2] + lam * np.eye(X.shape[1])
        b = np.linalg.solve(A, X[f2].T @ Y[f2])
        m = float(np.abs(X[v2] @ b - Y[v2]).mean())
        if m < best_mae:
            best, best_mae = lam, m
    A = X[fit].T @ X[fit] + best * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X[fit].T @ Y[fit])

    mae_probe = float(np.abs(X[sc] @ beta - Y[sc]).mean())
    mae_zero = float(np.abs(Y[sc]).mean())                       # the head's own floor
    const = float(np.median(Y[fit]))                             # best constant, fit-side only
    mae_const = float(np.abs(Y[sc] - const).mean())

    by = collections.defaultdict(list)
    for e, a, b_ in zip(EPI[sc], np.abs(X[sc] @ beta - Y[sc]), np.abs(Y[sc])):
        by[int(e)].append(b_ - a)
    import random
    rng = random.Random(20260921)
    keys = list(by)
    ms = sorted(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by[k])
                for _ in range(4000)) if len(keys) > 1 else None
    ci = [round(ms[100], 4), round(ms[3900], 4)] if ms else None

    res = {"_what": "is v_rel_y linearly predictable from TARGET-side geometry? an UPPER BOUND",
           "_evidence_class": "MEASURED (ours), CPU, no model pass",
           "_function_class": "LINEAR (ridge). A negative is about LINEAR predictability only.",
           "_features": ["cx", "cy", "range", "v_rel_x", "cls", "bias"],
           "n_pairs": int(X.shape[0]), "d": int(X.shape[1]), "n_episodes": len(eps),
           "lambda_chosen_on_FIT_only": best,
           "MAE_probe": round(mae_probe, 4),
           "MAE_zero_floor": round(mae_zero, 4),
           "MAE_best_constant_control": round(mae_const, 4),
           "gain_over_zero": round(mae_zero - mae_probe, 4),
           "CI95_gain_episode_clustered": ci,
           "probe_beats_zero": bool(ci and ci[0] > 0)}
    res["_VERDICT"] = (
        "⛔ EVEN TARGET-SIDE GEOMETRY CANNOT PREDICT v_rel_y LINEARLY ⇒ there is nothing here to "
        "find and the HEAD CANNOT BE BLAMED for missing it"
        if not res["probe_beats_zero"] else
        "⭐ signal EXISTS in the target geometry ⇒ the head missed something real (an upper bound: "
        "this does NOT prove an image would carry it)")
    print(json.dumps(res, indent=1))
    print("\n" + res["_VERDICT"])
    pathlib.Path("vy_learnable.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
