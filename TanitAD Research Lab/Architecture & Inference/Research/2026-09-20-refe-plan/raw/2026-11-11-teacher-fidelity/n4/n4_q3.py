"""N4-DAC Q3: run every repair on every banked form-R label whose tracked DAC = 0 (6 arms), record certified repairs and what they change."""
import json
import sys
import time
import numpy as np
sys.path.insert(0, ".")
import n4_geom as G
import n4_common as N
import n4_repair as RP

Rl, S = N.load_rolls(), N.load_scores()
common = sorted(set.intersection(*[set(v) for v in S.values()]))
inst = [(r, t) for r in N.RUNS for t in common if S[r][t]["arms"]["R"]["tracked"]["dac"] == 0]
print("failing form-R instances:", len(inst), "tokens", len({t for _, t in inst}))
by = {}
for r, t in inst:
    by.setdefault(S["RJ5"][t]["log"], set()).add(t)
by = {k: sorted(v) for k, v in by.items()}
out, ctl = [], {"base_reproduced": 0, "n": 0}
t0 = time.time()
for lg, tk, cx in G.contexts(by):
    for run in N.RUNS:
        if (run, tk) not in inst:
            continue
        t20 = np.asarray(Rl[run][tk]["traj20"], float)
        lab = cx.labels(t20)["R"]
        bres, _ = cx.score([lab])
        base = bres[0]
        ctl["n"] += 1
        bb = S[run][tk]["arms"]["R"]["tracked"]
        ctl["base_reproduced"] += int(abs(base["pdms"] - bb["pdms"]) < 1e-6 and base["dac"] == bb["dac"])
        rec = {"run": run, "token": tk, "log": lg, "base": base, "methods": {}}
        for m, fn in RP.METHODS.items():
            ts = time.time()
            try:
                r = fn(cx, lab, t20, base)
            except Exception as e:                                                    # noqa: BLE001
                rec["methods"][m] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                continue
            if r is None:
                rec["methods"][m] = {"repaired": False, "sec": round(time.time() - ts, 2)}
            else:
                rec["methods"][m] = {"repaired": True, "metrics": r["m"], "param": r["param"], "n_scored": r["n_scored"],
                                     "label_shift_xy_max_m": round(float(np.linalg.norm(r["plan"][:, :2] - lab[:, :2], axis=1).max()), 4),
                                     "label_shift_vs_teacher_xy_max_m": round(float(np.linalg.norm(r["plan"][:, :2] - t20[:, :2], axis=1).max()), 4),
                                     "plan20": np.asarray(r["plan"]).tolist(), "sec": round(time.time() - ts, 2)}
        out.append(rec)
        print(run, tk, {m: v.get("repaired", v.get("error")) for m, v in rec["methods"].items()}, round(time.time() - t0, 1), flush=True)
json.dump({"control": ctl, "instances": out}, open("Q3_repairs.json", "w", encoding="utf-8"))
print(ctl)
