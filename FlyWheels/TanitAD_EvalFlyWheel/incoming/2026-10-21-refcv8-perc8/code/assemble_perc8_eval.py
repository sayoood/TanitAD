"""assemble_perc8_eval.py -- merge every banked read into ONE PERC8_EVAL.json (the artifact RESULT.md quotes).  Missing optional parts are
recorded as {"status": "NOT RUN", "reason": ...}, never silently dropped.   python assemble_perc8_eval.py --root <perc8 dir> --out <json>"""
import argparse, hashlib, json, os, sys, time


def rd(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"status": "NOT AVAILABLE", "path": p, "reason": f"{type(e).__name__}: {e}"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.root
    out = {"artifact": "PERC8_EVAL.json", "assembled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "evidence_class": "MEASURED (ours; every number below traces to a file under raw/)",
           "tier": "T0 -- the perception outputs of the model's forward on the S2 eval windows (the box and map heads read the frames; no future frame, no "
                   "closed loop).  A perception read, NOT a driving-performance claim.",
           "estimator": "episode-cluster bootstrap over the 139 eval episodes (taniteval.ci draw stream, seed 0); paired = the same draws for both arms",
           "ci_answers": "the CI answers another draw of EPISODES only.  ONE training seed per arm: it says nothing about run-to-run TRAINING variance "
                         "(H-ESTIM-SEED-1) and the perception forward is deterministic, so there is no inference-seed variance to measure here.",
           "arms": {"refcv8": "refcv8-r101-s0 EMA weights at step 48,852 (ckpt_ema_48852.pt, md5 b4364029b68f29a33eeb4aebbbbd002b), 1 training seed",
                    "refcv7": "refcv7-r101-s0 at step 50,400 (banked D:/refcv7_eval_kit/perc/step50400_a6text/, ckpt_50400.pt md5 b418d0fc4a92a6848c246a6a7c50207b), 1 training seed"}}
    for name, rel in (("perc_pass_record_A_nosync", "raw/A_nosync/perc.json"), ("train_diag_record", "raw/T_nosync/train_diag.json"),
                      ("refit_gates", "raw/T_nosync/refcv8_det_presence_gates_train.json"), ("refit_nms", "raw/T_nosync/refcv8_det_nms_train.json"),
                      ("box_read_baseline", "raw/box_read_declared.json"), ("box_read_final", "raw/box_read_final.json"),
                      ("map_read", "raw/map_read.json"), ("gt_as_prediction_control", "raw/gt_control.json"), ("rescore_refcv7_control", "raw/rescore_refcv7_control.json"),
                      ("perc_pass_record_W_dec_sub", "raw/W_dec_sub/perc.json"),
                      ("wise_control", "raw/wise_control.json"), ("clock_sensitivity_box", "raw/clock_sub_box_read.json"),
                      ("clock_sensitivity_map", "raw/clock_sub_map_read.json")):
        p = os.path.join(R, rel)
        out[name] = rd(p) if os.path.exists(p) else {"status": "NOT RUN", "path": rel}
    out["code_md5"] = {f: md5(os.path.join(R, "code", f)) for f in sorted(os.listdir(os.path.join(R, "code"))) if f.endswith((".py", ".sh", ".json"))}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("wrote", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    sys.exit(main())
