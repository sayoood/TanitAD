"""smoke_acc.py -- card PREREG_V8F_PROOF_ADD153 section 6: the smoke job on box 2 (names nothing). Each stage-1 arm: 200 steps on 100 fit, 20 held-out and 10 eval clips,
with the reference head read beside. Writes SMOKE.json with the seconds per step and per read clip and the projection of the whole card; "stop" is true when the
projection exceeds 11 h (the sequence then stops and the PI is asked). Prints QUEUE-COMPLETE at the end in every case.

  python smoke_acc.py --out D:/tanitad/leadacc/smoke --ref-ckpt D:/tanitad/leadspeed/all/head_30000.pt --ref-sha 83faaa6d355562f7
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leadacc as A  # noqa: E402
import train_acc_box2 as TA  # noqa: E402

STAGE1 = ("K25D", "K25DM", "K10D", "K40D")
LOAD_FULL_S = 443.0                     # row 217: 3,984 clips loaded cold (the warm load took 111 s)
N_FIT, N_HO, N_EV, STEPS, LIMIT_H = 3984, 443, 141, 30000, 11.0


def projection(per_arm):
    """per_arm: {arm: {s_per_step, read_ho_s_per_clip, read_ev_s_per_clip, load_s_per_clip}} -> hours per arm and the card's total (stage 2 = twice the slowest arm)."""
    hrs = {}
    for arm, m in per_arm.items():
        load = max(LOAD_FULL_S, m["load_s_per_clip"] * N_FIT)
        hrs[arm] = (load + m["s_per_step"] * STEPS + m["read_ho_s_per_clip"] * N_HO + m["read_ev_s_per_clip"] * N_EV + 60.0) / 3600.0
    total = sum(hrs.values()) + 2 * max(hrs.values())
    return {"hours_per_arm": {k: round(v, 2) for k, v in hrs.items()}, "stage2_hours": round(2 * max(hrs.values()), 2), "total_hours": round(total, 2),
            "limit_hours": LIMIT_H, "stop": bool(total > LIMIT_H)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-ckpt", required=True)
    ap.add_argument("--ref-sha", required=True)
    ap.add_argument("--extra", default="", help="tests only: extra trainer arguments, one string")
    ap.add_argument("--arms", nargs="*", default=list(STAGE1))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"card": "PREREG_V8F_PROOF_ADD153", "job": "smoke (names nothing)", "arms": {}}
    try:
        per = {}
        for arm in a.arms:
            t0 = time.time()
            r = TA.main(["--arm", arm, "--out", os.path.join(a.out, arm), "--steps", "200", "--ckpt-every", "200", "--n-fit", "100", "--n-ho", "20", "--n-eval", "10",
                         "--ref-ckpt", a.ref_ckpt, "--ref-sha", a.ref_sha, "--no-marker"] + a.extra.split())
            res["arms"][arm] = {"error": r.get("error"), "controls": {k: v.get("PASS") for k, v in r.get("controls", {}).items()}, "timing": r.get("timing"),
                                "s_per_step": r.get("train", {}).get("s_per_step"), "wall_s": round(time.time() - t0, 1), "guard": r.get("guard")}
            if r.get("error"):
                raise RuntimeError("%s: %s" % (arm, r["error"]))
            tm = r["timing"]
            per[arm] = {"s_per_step": r["train"]["s_per_step"], "load_s_per_clip": tm["load_fit_s"] / 100.0,
                        "read_ho_s_per_clip": tm["read_held_out_s"] / max(1, r["split"]["held_out"]), "read_ev_s_per_clip": tm["read_eval_s"] / max(1, r["split"]["eval"])}
        res["per_arm"] = per
        res["projection"] = projection(per)
    except Exception as ex:                                                                     # noqa: BLE001
        res["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:500])
        res["traceback"] = traceback.format_exc()[-2000:]
        res["projection"] = {"stop": True}
    json.dump(res, open(os.path.join(a.out, "SMOKE.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1, default=float), flush=True)
    print("QUEUE-COMPLETE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    return res


if __name__ == "__main__":
    main()
    sys.exit(0)
