"""G0-A1 (SPEC.md AMENDMENT A1): the deliberate regressions M2-M4 on the SAME 128 windows / 8 batches,
inference seed 0, then the G0-A1 verdict against the banked 8-seed reproduction.

    python g0_mutations.py --g0-json raw/g0_step1000.json --out raw/g0_step1000_A1.json

M2  max-speed input withheld   (batch["v_max_valid"] := 0)
M3  ego-history window zeroed  (batch["pose_hist"]   := 0)
M4  tactical-goal pos_weight / class mask absent (model attributes := None), the literal state
    `refcv3_arm.load_model` leaves the model in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402
import reproduce_inrun_eval as G  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402

MUTATIONS = ("m2_vmax_withheld", "m3_ego_hist_zeroed", "m4_no_goal_posweight_mask")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro", default="3,3,3,3,4")
    a = ap.parse_args()
    g0 = json.load(open(a.g0_json, encoding="utf-8"))
    tr = L.trainer()
    device = "cuda"
    config = L.load_config(g0["config"])
    model, cfg, args, mrec = L.build_model(config, g0["ckpt"], device)
    if L.md5_file(g0["ckpt"]) != g0["ckpt_md5"]:
        raise SystemExit("[A1] checkpoint md5 differs from the G0 artifact's")
    e_ds, _eps, _drec = L.build_eval_dataset(
        model, cfg, args, config, with_perception_targets=True,
        dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), int(args.batch))
    if hashlib.sha256(json.dumps(perm).encode()).hexdigest() != g0["perm_sha256"]:
        raise SystemExit("[A1] the 128-window subset differs from the G0 run's")
    law_idx = int(tr.LAW_AHEAD) - 1
    G.patch_frames_to_device(tr)
    mb = MicroBatchForward(model, [int(x) for x in a.micro.split(",")]).install()
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "g0_mutations.py", "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
           "g0_json": a.g0_json, "seed": a.seed, "mutations": {}}

    def batches_with(transform):
        def gen():
            for eb in G.iter_batches(e_ds, perm, int(args.batch), int(args.eval_batches), law_idx):
                yield transform(eb)
        return gen

    for mname in MUTATIONS:
        t0 = time.time()
        saved = (model._tac_goal_pos_weight, model._tac_goal_class_mask)
        if mname == "m2_vmax_withheld":
            def tf(eb):
                eb["v_max_valid"] = torch.zeros_like(eb["v_max_valid"])
                return eb
        elif mname == "m3_ego_hist_zeroed":
            def tf(eb):
                eb["pose_hist"] = torch.zeros_like(eb["pose_hist"])
                return eb
        else:
            def tf(eb):
                return eb
            model._tac_goal_pos_weight = None
            model._tac_goal_class_mask = None
        try:
            row, pb = G.run_eval(tr, model, batches_with(tf), device, mode, abl, a.seed)
        finally:
            model._tac_goal_pos_weight, model._tac_goal_class_mask = saved
        rec["mutations"][mname] = {"row": row, "wall_s": round(time.time() - t0, 1)}
        print(f"[A1] {mname}: eval_loss {row.get('eval_loss')} tacv6_goal_bce "
              f"{row.get('eval_tacv6_goal_bce')} traj {row.get('eval_traj')} "
              f"({time.time() - t0:.0f}s)", flush=True)
    mb.remove()
    # ---- the G0-A1 verdict -------------------------------------------------------------- #
    terms = g0["verdict"]["terms"]
    det = {}
    for mname, m in rec["mutations"].items():
        out = []
        for k, t in terms.items():
            c = t.get("cls", t["class"])
            if c in ("COUNT", "EXCLUDED") or k not in m["row"]:
                continue
            y = float(m["row"][k])
            x = float(t["inrun"])
            if c == "SMOOTH":
                bad = (abs(y - x) > 1e-3) if abs(x) < 0.1 else (abs(y - x) / abs(x) > 0.01)
            elif c == "STOCHASTIC":
                bad = not (t["pi_lo"] <= y <= t["pi_hi"])
            else:                                   # MATCHED
                bad = abs(y - x) / max(abs(x), 1e-12) > 0.15
            if bad:
                out.append({"term": k, "class": c, "inrun": x, "mutated": y,
                            "rel": abs(y - x) / max(abs(x), 1e-12)})
        det[mname] = {"detected": bool(out), "n_terms_out": len(out),
                      "terms_out": sorted(out, key=lambda r: -r["rel"])}
    non_m1 = [r for r in g0["verdict"]["reasons"] if not r.startswith("M1")]
    detected = [m for m, d in det.items() if d["detected"]]
    rec["detection"] = det
    rec["G0_A1"] = ("PASS" if (not non_m1 and detected) else
                    "VOID" if (not non_m1 and not detected) else "FAIL")
    rec["G0_A1_reasons"] = (non_m1 + ([] if detected else ["no mutation detected"]) +
                            [f"BLIND SPOT: {m} not detected" for m, d in det.items()
                             if not d["detected"]])
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"G0_A1": rec["G0_A1"], "reasons": rec["G0_A1_reasons"],
                      "detected": {m: d["n_terms_out"] for m, d in det.items()}}, indent=1))


if __name__ == "__main__":
    main()
