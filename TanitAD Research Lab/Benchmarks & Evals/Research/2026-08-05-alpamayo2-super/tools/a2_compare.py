"""Alpamayo 2 Super vs TanitAD flagship — same clips, 2 s horizon, and THE test.

⭐ THE QUESTION THIS EXISTS TO ANSWER, and it is not "who wins".
Our flagship over-predicts speed: `speed_bias +0.484 m/s`, ahead of the human at
2 s on **71.95 %** of windows. If a 34 B surround model trained on ~115,000 h
ALSO runs ahead of the human, the bias lives in the DATA or the LABEL CONVENTION
and we would be optimising against ground truth. If it does not, the defect is
ours. Either answer redirects the programme's largest open work item — and,
unlike an ADE ranking, neither answer is decided in advance by parameter count.

⛔ FIVE THINGS THAT ARE NOT LIKE-FOR-LIKE, carried with every number:
  1. 34.3 B vs < 0.3 B parameters (~115x)
  2. 6 cameras x 4 frames at 1920x1080 vs ONE 256x256 front crop (~190x pixels)
  3. Alpamayo native horizon 6.4 s / 64 wp; ours 2.0 s / 20 wp -> TRUNCATED to 20
  4. Alpamayo is NF4-QUANTISED here and that is NOT an NVIDIA-validated config
  5. ⛔ CONTAMINATION UNRESOLVED: these clips are PhysicalAI-AV, which Alpamayo
     lists as a TRAINING dataset. They may be inside its training split.

⚠️ ALIGNMENT: Alpamayo ran at clip t0 = 5.1 s; our window origins sit on an 0.8 s
stride, so |dt| <= 0.4 s. Each model is scored against ITS OWN ground truth at
ITS OWN t0, and the per-clip offset is reported. The over-speed RATE is robust to
a sub-0.4 s offset; a per-clip ADE difference is not, and is labelled accordingly.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def geom(p, dt=0.1):
    """[K,2] ego-frame path -> per-step speed and along-track, origin prepended."""
    p = np.asarray(p, dtype=np.float64)
    q = np.concatenate([np.zeros((1, 2)), p], 0)
    d = q[1:] - q[:-1]
    return {"speed": np.linalg.norm(d, axis=-1) / dt, "along": q[1:, 0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpamayo-jsonl", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--flagship-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--alpamayo-gt", required=True,
                    help="reconstructed GT from a2_gt_from_ego.py, VALIDATED "
                         "against NVIDIA's own printed min_ade_m")
    a = ap.parse_args()

    alp = {r["sample_index"]: r for r in
           (json.loads(l) for l in open(a.alpamayo_jsonl) if l.strip())
           if "min_ade_m" in r}
    fs = {r["i"]: r for r in json.load(open(a.flagship_json)) if "ade_2s_m" in r}
    gtj = json.load(open(a.alpamayo_gt))
    agt = {int(k): np.asarray(v) for k, v in gtj["gt_xy_by_index"].items()}
    print("[gt] " + gtj["_validated"])
    both = sorted(set(alp) & set(fs) & set(agt))
    print(f"alpamayo {len(alp)} · flagship {len(fs)} · paired {len(both)}")

    rows = []
    for i in both:
        npz = os.path.join(a.traj_dir, f"traj_{i:04d}.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        pred = z["pred_xyz"]
        # squeeze leading sample dims: pred [1,1,1,64,3] -> [64,3].
        # GT comes from the VALIDATED egomotion reconstruction, not the npz --
        # the captured `data` is the model's INPUT and carries no future.
        P = np.asarray(pred).reshape(-1, pred.shape[-2], pred.shape[-1])[0]
        G = agt[i]
        H = min(a.horizon, P.shape[0], G.shape[0])
        Pa, Ga = P[:H, :2], G[:H, :2]                   # Alpamayo, truncated
        Pf = np.asarray(fs[i]["pred"])[:H]
        Gf = np.asarray(fs[i]["gt"])[:H]

        gA, gGA = geom(Pa), geom(Ga)
        gF, gGF = geom(Pf), geom(Gf)
        rows.append({
            "i": i, "clip_id": alp[i]["clip_id"], "cot": alp[i].get("cot"),
            "align_err_s": fs[i]["align_err_s"], "v0_mps": fs[i]["v0_mps"],
            "alp_ade_2s": float(np.linalg.norm(Pa - Ga, axis=-1).mean()),
            "fs_ade_2s": float(np.linalg.norm(Pf - Gf, axis=-1).mean()),
            "alp_ade_full_6p4s": alp[i]["min_ade_m"],
            "alp_speed_bias": float((gA["speed"] - gGA["speed"]).mean()),
            "fs_speed_bias": float((gF["speed"] - gGF["speed"]).mean()),
            "alp_along_final_bias": float(gA["along"][-1] - gGA["along"][-1]),
            "fs_along_final_bias": float(gF["along"][-1] - gGF["along"][-1]),
        })

    if not rows:
        raise SystemExit("no paired rows")
    m = lambda k: float(np.mean([r[k] for r in rows]))            # noqa: E731
    frac = lambda k: float(np.mean([r[k] > 0 for r in rows]))     # noqa: E731
    summary = {
        "n_paired": len(rows), "horizon_steps": a.horizon, "horizon_s": a.horizon * 0.1,
        "alpamayo": {
            "ade_2s_m": round(m("alp_ade_2s"), 4),
            "ade_native_6p4s_m": round(m("alp_ade_full_6p4s"), 4),
            "speed_bias_mps": round(m("alp_speed_bias"), 4),
            "frac_faster_than_human": round(frac("alp_speed_bias"), 4),
            "along_final_bias_m": round(m("alp_along_final_bias"), 4),
            "frac_ahead_at_2s": round(frac("alp_along_final_bias"), 4),
        },
        "flagship": {
            "ade_2s_m": round(m("fs_ade_2s"), 4),
            "speed_bias_mps": round(m("fs_speed_bias"), 4),
            "frac_faster_than_human": round(frac("fs_speed_bias"), 4),
            "along_final_bias_m": round(m("fs_along_final_bias"), 4),
            "frac_ahead_at_2s": round(frac("fs_along_final_bias"), 4),
        },
        "_not_like_for_like": (
            "34.3B vs <0.3B (~115x params); 6 cameras x 4 frames at 1920x1080 vs "
            "ONE 256x256 front crop (~190x pixels); Alpamayo truncated from its "
            "native 6.4s/64wp to 2.0s/20wp; Alpamayo is NF4-QUANTISED and that is "
            "NOT an NVIDIA-validated configuration."),
        "_contamination": (
            "these clips are PhysicalAI-AV, which Alpamayo lists as a TRAINING "
            "dataset. Overlap with its training split is UNRESOLVED, so any "
            "advantage may be contamination."),
        "_alignment": (
            "Alpamayo ran at clip t0 5.1s; our window origins sit on an 0.8s "
            "stride so |dt| <= 0.4s. Each model is scored against ITS OWN GT at "
            "ITS OWN t0. The over-speed RATE is robust to that; a per-clip ADE "
            "difference is not."),
        "_estimator": "unweighted mean over paired clips; n is small (see n_paired)",
        "rows": rows,
    }
    json.dump(summary, open(a.out, "w"), indent=1)
    A, F = summary["alpamayo"], summary["flagship"]
    print(f"\n{'':22s} {'ALPAMAYO 2 SUPER':>18s} {'TANITAD FLAGSHIP':>18s}")
    for lab, k in (("ADE @2s (m)", "ade_2s_m"),
                   ("speed bias (m/s)", "speed_bias_mps"),
                   ("frac faster than human", "frac_faster_than_human"),
                   ("along bias @2s (m)", "along_final_bias_m"),
                   ("frac ahead at 2s", "frac_ahead_at_2s")):
        print(f"{lab:22s} {A[k]:>18} {F[k]:>18}")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
