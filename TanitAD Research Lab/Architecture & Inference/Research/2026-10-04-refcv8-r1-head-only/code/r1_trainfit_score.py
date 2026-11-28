"""SPEC_R1 sec. 7 TRAIN-FIT diagnostic (secondary, never a bar): each arm scored on 1,000 of its OWN training windows
(`r1_arms.py --phase eval --eval-split train` -> arms/<arm>/trainfit_s<seed>.pt). Reads: an arm that cannot reach the
gate on windows it was trained on says the frozen features or the head capacity do not separate the turn; an arm that
fits TRAIN but not EVAL says generalisation. Same metrics and estimator as r1_score.py (episode-cluster bootstrap over
the train clips). Writes arms/trainfit_score.json. sha12 only."""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_lib as L1  # noqa: E402
from r1_arms import load_split  # noqa: E402
from r1_score import boot_ratio, ci, paired, rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/nvidia/refcv8_r1/cache")
    ap.add_argument("--arms", default="/home/nvidia/refcv8_r1/arms")
    a = ap.parse_args()
    import torch
    D, _ = load_split(a.cache, "train")
    pos = {int(i): j for j, i in enumerate(D["b_r1_i"])}
    out = {}
    base = None
    files = sorted(glob.glob(str(Path(a.arms) / "*" / "trainfit_s0.pt")))
    runs = {}
    for p in files:
        r = torch.load(p, map_location="cpu", weights_only=False)
        runs[r["arm"]] = r
    if "H0" not in runs:
        raise SystemExit("[trainfit] H0 has no trainfit_s0.pt")
    ix = np.array([pos[int(i)] for i in runs["H0"]["b_r1_i"]])
    gt = D["gt"].numpy()[ix].astype(np.float64)
    gv = D["gt_valid"].numpy()[ix]
    cls = np.array(D["_cls"])[ix]
    ep = np.array(D["win_sha12"])[ix]
    draws = np.random.default_rng(0).integers(0, len(np.unique(ep)), (2000, len(np.unique(ep))))
    turn = np.isin(cls, ["turnL", "turnR"])
    straight = cls == "straight"
    th_g = L1.terminal_heading(gt)
    side_g = np.sign(th_g)

    def met(r):
        tr = r["traj"].numpy().astype(np.float64)
        th = L1.terminal_heading(tr)
        return {"dir_ok": (L1.dir_class(th) == side_g).astype(float),
                "h15": (np.abs(L1.wrap(th - th_g)) <= math.radians(15.0)).astype(float),
                "ade": L1.ade(tr, gt, gv)}
    for arm, r in sorted(runs.items()):
        if [int(i) for i in r["b_r1_i"]] != [int(i) for i in runs["H0"]["b_r1_i"]]:
            raise SystemExit(f"[trainfit] {arm}: window set differs from H0's")
        m = met(r)
        if arm == "H0":
            base = m
        out[arm] = {"turn_dir_correct": rate(m["dir_ok"], turn, ep, draws),
                    "turn_heading15": rate(m["h15"], turn, ep, draws),
                    "dADE_straight_vs_H0": paired(m["ade"], met(runs["H0"])["ade"], straight & np.isfinite(m["ade"]),
                                                  ep, draws),
                    "n_turn": int(turn.sum()), "n_straight": int(straight.sum())}
    res = {"n_windows": int(len(ix)), "n_turn": int(turn.sum()), "n_straight": int(straight.sum()),
           "n_clips": int(len(np.unique(ep))), "arms": out,
           "note": "TRAIN-FIT diagnostic (SPEC_R1 sec. 7): windows the arms were TRAINED on; never a bar"}
    (Path(a.arms) / "trainfit_score.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: {"dir": round(v["turn_dir_correct"]["pt"], 3), "h15": round(v["turn_heading15"]["pt"], 3),
                          "dADE_str": round(v["dADE_straight_vs_H0"]["pt"], 3)} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
