"""validate_fix.py -- run the REAL wrapper_probe_r7.probe on CPU (P3 condition only, the clause's own condition) on the
same 4 rows, once UNFIXED and once through wrapper_probe_r8fix.probe. Banks raw/validate_fix.json.
`torch.cuda.synchronize` is a no-op here (CPU only); nothing else in the probe is touched."""
from __future__ import annotations

import json
import time

import r8_wrapper_diag as D  # noqa: E402

torch, G, W, L = D.torch, D.G, D.W, D.L
import wrapper_probe_r8fix as FIX  # noqa: E402

OUT = D.RAW / "validate_fix.json"


def main():
    t0 = time.time()
    torch.cuda.synchronize = lambda *a, **k: None
    W.CONDITIONS = {"P3_fp32_det": W.CONDITIONS["P3_fp32_det"]}
    tr = L.trainer()
    config = L.load_config(D.CONFIG)
    model, cfg, args, mrec = L.build_model(config, D.CKPT, "cpu")
    G.set_flags(model, G.training_flags(model)["flags"])
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    eb4 = D.load_or_build_batch(tr, model, cfg, args, config)
    G.patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    rec = {"tool": "validate_fix.py", "condition": "P3_fp32_det only, CPU", "batch_meta": D.REC.get("batch_meta")}
    for name, fn in (("unfixed", W.probe), ("fixed", FIX.probe)):
        t = time.time()
        r = fn(tr, model, eb4, "cpu", mode, abl, trunk, sub_batch_fn=G.sub_batch)
        an = r["conditions"]["P3_fp32_det"].get("analysis", {})
        rec[name] = {"clause": r.get("clause"), "reasons": r.get("reasons"), "max_wrapper_rel": an.get("max_wrapper_rel"),
                     "max_floor_rel": an.get("max_floor_rel"), "terms_over_bar": an.get("terms_over_bar"),
                     "W1_detected_terms": an.get("W1_detected_terms"), "n_W2_detected": len(an.get("W2_detected_terms") or []),
                     "keys": {k: an.get("terms", {}).get(k) for k in ("cascade", "r8_subscore", "traj", "r8_alloc_l1",
                                                                    "sel_v3", "loss")},
                     "wall_s": round(time.time() - t, 1)}
        print(f"[fix] {name}: clause {r.get('clause')} max wrapper {an.get('max_wrapper_rel')} max floor "
              f"{an.get('max_floor_rel')} over {an.get('terms_over_bar')}", flush=True)
        json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print("[fix] DONE", flush=True)


if __name__ == "__main__":
    main()
