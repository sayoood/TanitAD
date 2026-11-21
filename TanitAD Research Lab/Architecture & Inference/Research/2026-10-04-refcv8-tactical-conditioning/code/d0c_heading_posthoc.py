"""D0c -- POST-HOC (NOT pre-registered; written after D0's numbers existed): can a terminal-HEADING constraint of
accuracy sigma_psi (deg) lift heading-within-15 deg on GT-turn windows to the R8-4 (iv) bar 0.70 by re-selecting
refcv7's banked fan? Same instrument as d0_dose_response.py (imported), TRAIN-fitted lambda, EVAL s0 + s1.
Writes raw/d0c_heading_posthoc.json. A design input, never a bar result."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d0_dose_response as d0  # noqa: E402
rm = d0.rm
LAMS = (0.5, 1, 2, 4, 8, 16, 32)
SIGS = (0.0, 5.0, 10.0, 15.0, 20.0)

def head_term(d, th_hat):
    return np.abs(rm.wrap(d["th_c"] - th_hat[:, None]))

def th_hat(d, sig, r, tag):
    rng = np.random.default_rng(30_000 * tag + 100 * r + int(sig * 10))
    ok = d["target"] != 9
    return np.where(ok, d["th_gt"] + math.radians(sig) * rng.standard_normal(d["W"]), 0.0), ok

def main():
    Z = {k: d0.load(k) for k in ("train_s0", "eval_s0g", "eval_s1")}
    D = {k: d0.derive(v) for k, v in Z.items()}
    zt, ze, zs1 = Z["train_s0"], Z["eval_s0g"], Z["eval_s1"]
    dt, de, ds1 = D["train_s0"], D["eval_s0g"], D["eval_s1"]
    draws = rm.make_draws(de["ep"], d0.B, d0.SEED)
    v0e, v0s1 = d0.per_window(ze["sel_idx"], ze, de), d0.per_window(zs1["sel_idx"], zs1, ds1)
    out = {"_what": "POST-HOC heading-constraint dose response (not pre-registered)"}
    for with_lon in (False, True):
        for sig in SIGS:
            def score(z, d, r, tag, lam, mu):
                th, ok = th_hat(d, sig, r, tag)
                s = z["s_e9"] - lam * np.where(ok[:, None], head_term(d, th), 0.0)
                if with_lon:
                    s = s - mu * d0.lon_term(d["P_c"], d0.phat_noisy(d, 0.10, r, tag), d["pg_ok"])
                return d0.masked_argmax(s, d["reach"], d["reach"])
            grid = [(l, m) for l in LAMS for m in ((8.0,) if with_lon else (0.0,))]
            fit = {g: float(np.mean([d0.mean_ade(score(zt, dt, r, 3, *g), dt, dt["masks"]["all"])
                                     for r in range(d0.R)])) for g in grid}
            g = min(fit, key=lambda k: (fit[k], k))
            p0 = [d0.per_window(score(ze, de, r, 1, *g), ze, de) for r in range(d0.R)]
            p1 = [d0.per_window(score(zs1, ds1, r, 1, *g), zs1, ds1) for r in range(d0.R)]
            out[f"{'lon0.10+' if with_lon else ''}head_sig{sig:.0f}"] = {
                "lam_mu": g, "eval_s0": d0.summarise(p0, v0e, de, draws), "eval_s1": d0.summarise(p1, v0s1, ds1, draws)}
    (d0.PKG / "raw" / "d0c_heading_posthoc.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in out.items():
        if k.startswith("_"):
            continue
        s = v["eval_s0"]
        print(f"{k:22s} {v['lam_mu']} all {s['dADE_all']['d']:+.3f} turn {s['dADE_turn']['d']:+.3f} {s['dADE_turn']['ci']} "
              f"dir {s['turn_dir']} h15 {s['turn_head15']} | s1 h15 {v['eval_s1']['turn_head15']}")

if __name__ == "__main__":
    main()
