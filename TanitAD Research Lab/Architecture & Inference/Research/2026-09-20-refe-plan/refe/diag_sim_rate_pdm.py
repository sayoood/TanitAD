"""Is the teacher a BETTER driver at nuPlan's 10 Hz or at the DB's 20 Hz? Score both with our own PDM calculators.

`diag_sim_rate.py` showed the rate moves the targets a lot (ADE 0.58 m, FDE 1.61 m on 12 frames), so
the rate is a FIDELITY question before it is a speed question. This scores the teacher's OWN
trajectory (the "teacher" row of the FULL candidate set -- `--candidates 1` is refused frame by frame
by the scorer's discrimination rule, ndiff < 3, MEASURED 73/73 aborted) from each rate's bank with the SAME scorer
context (REFE_SIM_HZ unset for the scorer -> identical ScenarioData), so only the injected
trajectory differs. Reports per-component means and paired wins/losses.

  python diag_sim_rate_pdm.py --a <hz20 bank dir> --b <hz10 bank dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = {  # name -> (target key, higher_is_better)
    "collision": ("collision.NuPlanCollision.info", False),
    "dac_violation": ("dac.violation", False),
    "ddc_violation": ("ddc.violation", False),
    "ttc": ("ttc.NuPlanTTC.reward", True),
    "comfort": ("comfort.Comfort.reward", True),
    "ep": ("progress.ep", True),
    "advance_m": ("progress.advance_m", True),
    "centerline": ("center_line.CenterLine.reward", True),
    "curb_clearance": ("center_line.CurbClearance.reward", True),
    "overspeed": ("center_line.Overspeed.reward", True),
    "offroad_info": ("off_road.OffRoad.info", False),
}


def pdm_like(t):
    """NC x DAC x DDC x (5 TTC + 2 C + 5 EP) / 12 from the banked components (a PDM-style summary)."""
    nc = 1.0 - float(t.get("collision.NuPlanCollision.info", 0.0) > 0)
    dac = 1.0 - float(t.get("dac.violation", 0.0) > 0)
    ddc = 1.0 - float(t.get("ddc.violation", 0.0) > 0)
    return nc * dac * ddc * (5 * t.get("ttc.NuPlanTTC.reward", 0.0) + 2 * t.get("comfort.Comfort.reward", 0.0)
                             + 5 * t.get("progress.ep", 0.0)) / 12.0


def score(bank_dir, out_dir):
    f = os.path.join(bank_dir, "targets_rank0.jsonl")
    logs = sorted({json.loads(l)["log_name"] for l in open(f, encoding="utf-8") if l.strip()})
    lf = os.path.join(out_dir, "logs.txt")
    os.makedirs(out_dir, exist_ok=True)
    open(lf, "w", encoding="utf-8").write("\n".join(logs) + "\n")
    env = {k: v for k, v in os.environ.items() if k != "REFE_SIM_HZ"}
    env["PYTHONIOENCODING"] = "utf-8"
    with open(os.path.join(out_dir, "scorer.log"), "w", encoding="utf-8") as log:
        rc = subprocess.call([sys.executable, "build_scorer_targets.py", "--source", "navtrain",
                              "--perframe-bank", bank_dir, "--perframe-file", f, "--out", out_dir,
                              "--rank", "0", "--frame-stride", "1",
                              "--logs-file", lf], cwd=HERE, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
    rows = {}
    p = os.path.join(out_dir, "scorer_targets.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("candidate") == "teacher":
                    rows[(r["log_name"], r["token"], int(r["step"]))] = r["targets"]
    return rc, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="bank dir A (reference, e.g. 20 Hz)")
    ap.add_argument("--b", required=True, help="bank dir B (e.g. 10 Hz)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:                  # both arms at once: same machine load
        fa = ex.submit(score, a.a, os.path.join(a.out, "A"))
        fb = ex.submit(score, a.b, os.path.join(a.out, "B"))
        (rc_a, A), (rc_b, B) = fa.result(), fb.result()
    common = sorted(set(A) & set(B))
    comp = {}
    for name, (key, hib) in KEYS.items():
        va = np.array([A[k].get(key, np.nan) for k in common], dtype=float)
        vb = np.array([B[k].get(key, np.nan) for k in common], dtype=float)
        better_b = int(np.sum((vb > va) if hib else (vb < va)))
        worse_b = int(np.sum((vb < va) if hib else (vb > va)))
        comp[name] = {"A_mean": float(np.nanmean(va)) if len(va) else None,
                      "B_mean": float(np.nanmean(vb)) if len(vb) else None,
                      "B_better": better_b, "B_worse": worse_b}
    pa = np.array([pdm_like(A[k]) for k in common]); pb = np.array([pdm_like(B[k]) for k in common])
    out = {"rc": [rc_a, rc_b], "frames": len(common), "components": comp,
           "pdm_like": {"A_mean": float(pa.mean()) if len(pa) else None,
                        "B_mean": float(pb.mean()) if len(pb) else None,
                        "B_better": int(np.sum(pb > pa + 1e-9)), "B_worse": int(np.sum(pb < pa - 1e-9))}}
    json.dump(out, open(os.path.join(a.out, "diag_sim_rate_pdm.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    ok = rc_a == 0 and rc_b == 0 and len(common) > 0
    print("ZZSIMRATE_PDM_" + ("MEASUREDZZ" if ok else "FAILZZ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
