"""P1f - WHERE DO THE LANE-CHANGE LOGITS RANK, AND BY WHAT MARGIN?

The brief's second candidate for the LC absence is "the class is present but
never argmax -- report the runner-up margin distribution".  The banked sidecar
carries only argmaxes, so this captures the raw `lat_head_tac` logits with a
forward hook on the REAL pipeline (`refcv3_arm.py` is invoked unmodified; only
`load_model` is wrapped, so the corpus, the gating and the conditioning are the
tool's own).
"""
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BUF = {"lat": [], "lon": []}


def main(argv):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "refcv3_arm",
        r"C:\Users\Admin\tanitad-selq-20260906\taniteval\tools\refcv3_arm.py")
    ra = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm"] = ra
    spec.loader.exec_module(ra)

    real = ra.load_model

    def wrapped(*a, **k):
        out = real(*a, **k)
        model = out[0]
        model.lat_head_tac.register_forward_hook(
            lambda m, i, o: BUF["lat"].append(o.detach().float().cpu()))
        model.lon_head_tac.register_forward_hook(
            lambda m, i, o: BUF["lon"].append(o.detach().float().cpu()))
        print("[p1f] forward hooks attached to lat_head_tac / lon_head_tac")
        return out

    ra.load_model = wrapped
    ra.main(argv)

    from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7 as LAT,
                                         TACTICAL_LON_ACTIONS_V7 as LON)
    res = {}
    for axis, names, key in (("LATERAL", LAT, "lat"),
                             ("LONGITUDINAL", LON, "lon")):
        if not BUF[key]:
            print(f"[p1f] no {axis} logits captured")
            continue
        L = torch.cat(BUF[key]).numpy()
        n = L.shape[0]
        top = L.max(1)
        order = (-L).argsort(1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(L.shape[1])[None, :], 1)
        print(f"\n=== {axis} logits, n = {n} forward rows, "
              f"{L.shape[1]} classes ===")
        print(f"  {'class':<18}{'mean logit':>12}{'argmax n':>10}"
              f"{'mean rank':>11}{'best rank':>11}{'mean margin':>13}")
        res[axis] = {}
        for i, nm in enumerate(names):
            marg = top - L[:, i]
            res[axis][nm] = {
                "mean_logit": float(L[:, i].mean()),
                "n_argmax": int((L.argmax(1) == i).sum()),
                "mean_rank_1_is_best": float(rank[:, i].mean() + 1),
                "best_rank_achieved": int(rank[:, i].min() + 1),
                "mean_margin_to_argmax": float(marg.mean()),
                "min_margin_to_argmax": float(marg.min()),
                "n": int(n)}
            r = res[axis][nm]
            m = ("  <== 0 POSITIVES IN THE CORPUS"
                 if nm in ("LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
                           "YIELD_MERGE") else "")
            print(f"  {nm:<18}{r['mean_logit']:>12.4f}{r['n_argmax']:>10}"
                  f"{r['mean_rank_1_is_best']:>11.2f}"
                  f"{r['best_rank_achieved']:>11}"
                  f"{r['mean_margin_to_argmax']:>13.4f}{m}")
        print(f"  [read-control] classes that are argmax at least once: "
              f"{sum(1 for i in range(L.shape[1]) if (L.argmax(1)==i).any())}"
              f" / {L.shape[1]}")
    with open("out_p1f_logit_rank.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("\nwrote out_p1f_logit_rank.json")


if __name__ == "__main__":
    main(sys.argv[1:])
