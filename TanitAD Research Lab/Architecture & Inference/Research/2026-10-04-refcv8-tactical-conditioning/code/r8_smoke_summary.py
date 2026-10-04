"""Read the refcv8 full-size smoke's OWN artifacts (metrics.jsonl, config.json, ckpt.pt) -> smoke_summary.json.

Checks (each a named PASS / FAIL, never folded):
* S1  the run reached its last step (metrics row at step == steps) and wrote config.json with init_from.applied True;
* S2  every refcv8 loss key present on every logged training row and finite (r8_cons, r8_alloc_l1, r8_sat,
      r8_listwise, r8_subscore), and the in-run eval row exists;
* S3  the in-run gradient share was read (gs_trunk_* rows, linearity <= 1e-4) -- its tac_v6 / r8 / traj / agent shares
      are REPORTED (the first full-scale reading for SPEC_WPB_LADDER L2's premise);
* S4  every refcv8 seam parameter MOVED from its init (zero-init weights are no longer all zero / the source's
      parameters changed): the G-LIVE proxy that each new module took gradient;
* S5  s/step from the logged elapsed time (rows after the first, excluding the eval step), against the 10.5 s ceiling
      (B-COST; the refcv7 reference is 9.9 s/step MEASURED).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", required=True)
    a = ap.parse_args()
    import torch
    run = Path(a.run)
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    train = [r for r in rows if "loss" in r and "step" in r and not any(k.startswith("eval_") for k in r)]
    evals = [r for r in rows if any(k.startswith("eval_") for k in r)]
    cfg = json.loads((run / "config.json").read_text())
    argv = list(cfg.get("argv") or [])
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else None
    if steps is None:
        raise SystemExit("[r8-smoke] config.json carries no --steps in its argv -- cannot judge S1")
    res = {"run": str(run), "n_train_rows": len(train), "n_eval_rows": len(evals), "checks": {}}
    last = max((int(r["step"]) for r in train), default=-1)
    init = cfg.get("init_from") or {}
    res["checks"]["S1_reached_end_and_warm_started"] = bool(last >= steps and init.get("applied") is True)
    res["init_from"] = init
    keys = ("r8_cons", "r8_alloc_l1", "r8_sat", "r8_listwise", "r8_subscore")
    miss = [(int(r["step"]), k) for r in train for k in keys
            if k not in r or r[k] is None or not math.isfinite(float(r[k]))]
    res["checks"]["S2_r8_terms_live_and_finite"] = bool(train and not miss and evals)
    res["r8_missing_or_nonfinite"] = miss[:10]
    gs = [{k: v for k, v in r.items() if k.startswith("gs_")} for r in train if any(k.startswith("gs_") for k in r)]
    res["grad_share"] = gs
    res["checks"]["S3_grad_share_read"] = bool(gs) and all(
        float(g.get("gs_trunk_lin_rel_err", 1.0)) <= 1e-4 for g in gs)
    ck = torch.load(str(run / "ckpt.pt"), map_location="cpu", weights_only=False)["model"]
    src = torch.load(a.init, map_location="cpu", weights_only=False)
    src = src.get("model", src)
    r8 = {k: v for k, v in ck.items() if any(p.startswith("r8_") for p in k.split(".")) and torch.is_floating_point(v)}
    still = [k for k, v in r8.items() if float(v.abs().sum()) == 0.0 and k.endswith("weight")]
    res["n_r8_tensors"] = len(r8)
    res["r8_weights_still_all_zero"] = still[:20]
    moved_src = sum(1 for k, v in src.items() if k in ck and torch.is_floating_point(v)
                    and not torch.equal(v, ck[k]))
    res["n_source_tensors_moved"] = moved_src
    res["checks"]["S4_seams_took_gradient"] = bool(r8) and not still and moved_src > 0
    el = sorted((int(r["step"]), float(r["elapsed_s"]), any(k.startswith("gs_") for k in r)) for r in train
                if "elapsed_s" in r)
    eval_steps = {int(r["step"]) for r in evals if "step" in r}
    iv = [{"from": s1, "to": s2, "s_per_step": (e2 - e1) / (s2 - s1), "clean": (not g2) and s2 not in eval_steps}
          for (s1, e1, _g1), (s2, e2, g2) in zip(el, el[1:]) if s2 > s1]
    res["intervals"] = iv
    clean = [x["s_per_step"] for x in iv if x["clean"] and x["from"] > 0]
    res["s_per_step_clean"] = sorted(clean)[len(clean) // 2] if clean else None
    res["checks"]["S5_cost_under_ceiling_10p5"] = (res["s_per_step_clean"] is not None
                                                   and res["s_per_step_clean"] <= 10.5)
    res["PASS"] = all(res["checks"].values())
    (run / "smoke_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["checks"]), "PASS" if res["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
