"""actdiv_pairs.py — paired clip-bootstrap factors from raw/actdiv.json (R1 post-processing).

The actdiv windows are defined by the corpus alone (24 clips x first 60 frames, W=6, stride n//5),
so every arm sees the IDENTICAL 144 windows and the per-window action spreads pair across arms.
Factors reported with a paired clip bootstrap (resample the 24 clips, keep both arms' windows):

  * cross-arm ACTION-spread factor  (k60clip05p30k / k8clip05p30k, k60 / postrain30k, k8 / postrain30k)
    per variant and per speed scale — the quantity SMAS-1 attributed as "0.83x" at v/30;
  * within-arm FULL-TUPLE / BANKED factor (roll [steer,accel,v] vs roll [steer,accel] with v held)
    and V-ONLY / BANKED — how much of the conditioning response lives in the speed channel.
Scene-spread factors are point values from the same JSON (no per-window form exists for a
between-window std), so they carry the per-arm ratio CI only.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent / "raw"


def boot_factor(num, den, cid, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(cid)
    idx = {u: np.where(cid == u)[0] for u in uniq}
    point = float(num.mean() / den.mean())
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        bs.append(float(num[sel].mean() / max(den[sel].mean(), 1e-12)))
    return point, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    d = json.load(open(RAW / "actdiv.json", encoding="utf-8"))
    arms = d["arms"]
    out = {"_source": "raw/actdiv.json", "_evidence_class": d["_evidence_class"],
           "eval_tier": "T0-DIAGNOSTIC", "n_boot": 2000, "cross_arm_action_factor": {},
           "within_arm_channel_factor": {}, "scene_factor_point": {}}
    variants = ("banked_roll_sa_hold_v", "roll_full_tuple", "roll_v_only")
    pairs = (("k60clip05p30k", "k8clip05p30k"), ("k60clip05p30k", "postrain30k"),
             ("k8clip05p30k", "postrain30k"), ("postrain30k_freeze", "postrain30k"))
    for scale in ("10", "30"):
        for a1, a0 in pairs:
            if a1 not in arms or a0 not in arms:
                continue
            h1, h0 = arms[a1]["by_scale"][scale]["h1"], arms[a0]["by_scale"][scale]["h1"]
            c1, c0 = np.asarray(h1["per_window_clip_id"]), np.asarray(h0["per_window_clip_id"])
            assert (c1 == c0).all(), "windows differ across arms — pairing invalid"
            key = f"{a1}_over_{a0}"
            out["cross_arm_action_factor"].setdefault(f"v_over_{scale}", {})[key] = {}
            for v in variants:
                num = np.asarray(h1["per_window_action_spread_h1"][v])
                den = np.asarray(h0["per_window_action_spread_h1"][v])
                pt, ci = boot_factor(num, den, c1)
                out["cross_arm_action_factor"][f"v_over_{scale}"][key][v] = {
                    "factor": round(pt, 4), "ci95_paired_clip_boot": [round(ci[0], 4), round(ci[1], 4)],
                    "separated_from_1": bool(ci[0] > 1.0 or ci[1] < 1.0)}
            out["scene_factor_point"].setdefault(f"v_over_{scale}", {})[key] = round(
                h1["scene_spread"] / h0["scene_spread"], 4)
        for arm, av in arms.items():
            h = av["by_scale"][scale]["h1"]
            cid = np.asarray(h["per_window_clip_id"])
            base = np.asarray(h["per_window_action_spread_h1"]["banked_roll_sa_hold_v"])
            res = {}
            for v in ("roll_full_tuple", "roll_v_only"):
                pt, ci = boot_factor(np.asarray(h["per_window_action_spread_h1"][v]), base, cid)
                res[f"{v}_over_banked"] = {"factor": round(pt, 4),
                                           "ci95_paired_clip_boot": [round(ci[0], 4), round(ci[1], 4)]}
            out["within_arm_channel_factor"].setdefault(f"v_over_{scale}", {})[arm] = res
    (RAW / "actdiv_pairs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for sc, dd in out["cross_arm_action_factor"].items():
        for key, vv in dd.items():
            print(f"[{sc}] {key}: " + "  ".join(f"{v}={r['factor']:.3f} CI[{r['ci95_paired_clip_boot'][0]:.3f},{r['ci95_paired_clip_boot'][1]:.3f}]"
                                               for v, r in vv.items()) + f"  scene x{out['scene_factor_point'][sc][key]:.3f}")
    for sc, dd in out["within_arm_channel_factor"].items():
        for arm, vv in dd.items():
            print(f"[{sc}] {arm}: " + "  ".join(f"{k}={r['factor']:.2f} CI[{r['ci95_paired_clip_boot'][0]:.2f},{r['ci95_paired_clip_boot'][1]:.2f}]" for k, r in vv.items()))
    print(f"-> {RAW / 'actdiv_pairs.json'}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
