"""D-APPEAR P4 — run the promoted 0-GPU LATENT SCREEN on every latent we can reach.

The screen now lives in the repo as ``stack/tanitad/eval/latent_screen.py`` with contract tests
(``stack/tests/test_latent_screen.py``), so it can gate a launch instead of living in one run
directory. This script is its FIRST FLEET APPLICATION and it does two jobs:

  1  REPRODUCE the reference measurement (frozen v1 on comma2k19: jitter 51.0x, derivative
     corr +0.0891, derived accel R2 -0.3773) through the promoted module rather than through
     the original one-off script. A promoted instrument that does not reproduce its own
     reference number is not promoted, it is forked.
  2  WIDEN THE CALIBRATION. LATENT_BOTTLENECK.md Sec 6 states plainly that the thresholds come
     from ONE encoder plus an oracle. Screening the same encoder on PhysicalAI-AV, on rig A
     alone and on rig B alone turns 1 substrate into 4, plus a positive control.

⚠️ Same encoder throughout (``v1_speedjerk_ckpt.pt`` step 29999). This widens the CORPUS
   calibration, not the ENCODER calibration -- a second encoder is still missing and that is
   stated in the deliverable rather than papered over.

usage:
  OMP_NUM_THREADS=6 python run_p4_screen.py --out results_p4_screen.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

PAI = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_pai_substrate.pt")
PAI_A = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_rigA.pt")
PAI_B = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_rigB.pt")
COMMA_LAT = Path(r"C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def splits(eps):
    """The programme's standard nesting: episode-disjoint held-out, inner fit/sel of TRAIN."""
    tr = [e for i, e in enumerate(eps) if i % 3 != 0]
    ho = [e for i, e in enumerate(eps) if i % 3 == 0]
    fit = [e for i, e in enumerate(tr) if i % 3 != 0]
    sel = [e for i, e in enumerate(tr) if i % 3 == 0]

    def z(el, k="Z"):
        return torch.cat([e[k] for e in el]).float()

    def q(el):
        return torch.cat([e["Q"] for e in el]).float()

    return (z(fit), q(fit), z(sel), q(sel), z(tr), q(tr), z(ho), q(ho))


def oracle_from(eps):
    """The positive control: Z IS the true speed window (one channel). Must PASS."""
    tr = [e for i, e in enumerate(eps) if i % 3 != 0]
    ho = [e for i, e in enumerate(eps) if i % 3 == 0]
    fit = [e for i, e in enumerate(tr) if i % 3 != 0]
    sel = [e for i, e in enumerate(tr) if i % 3 == 0]

    def q(el):
        return torch.cat([e["Q"] for e in el]).float()

    return (q(fit)[:, :, None], q(fit), q(sel)[:, :, None], q(sel),
            q(tr)[:, :, None], q(tr), q(ho)[:, :, None], q(ho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_p4_screen.json")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from tanitad.eval import latent_screen as LS

    cases = {}
    lat = torch.load(COMMA_LAT, map_location="cpu", weights_only=False)
    cases["flagship_v1_step29999_comma2k19"] = lat["episodes"]
    for tag, p in (("physicalai_av_mixed", PAI), ("physicalai_av_rigA", PAI_A),
                   ("physicalai_av_rigB", PAI_B)):
        if p.exists():
            cases[f"flagship_v1_step29999_{tag}"] = torch.load(
                p, map_location="cpu", weights_only=False)["episodes"]

    res = {"meta": {
        "instrument": "stack/tanitad/eval/latent_screen.py (promoted from "
                      "…/2026-08-03-latent-bottleneck LATENT_BOTTLENECK.md §6)",
        "encoder": "v1_speedjerk_ckpt.pt step 29999 for EVERY case -- this widens the "
                   "CORPUS calibration, not the ENCODER calibration",
        "gates": LS.SCREEN_GATES, "sigma_estimator": LS.SIGMA_ESTIMATOR,
        "reference_record_before_this_run": LS.REFERENCE_LATENTS},
        "screens": {}}

    for name, eps in cases.items():
        log(f"screening {name} ({len(eps)} episodes)")
        r = LS.screen_latent(*splits(eps), name=name, device=a.device)
        res["screens"][name] = r
        log(LS.format_screen(r))
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))

    # the ORACLE positive control on the PhysicalAI rows -- if THIS fails, the run is void
    ok_eps = cases.get("flagship_v1_step29999_physicalai_av_mixed") or next(iter(cases.values()))
    log("screening the ORACLE positive control")
    r = LS.screen_latent(*oracle_from(ok_eps), name="ORACLE_true_speed_window_physicalai",
                         device=a.device)
    res["screens"]["ORACLE_true_speed_window_physicalai"] = r
    log(LS.format_screen(r))
    res["meta"]["admissible"] = (r["verdict"] == "PASS")
    if r["verdict"] != "PASS":
        res["meta"]["VOID_reason"] = ("the oracle positive control did not pass the screen -- "
                                      "no screen verdict in this file is admissible")

    # the reproduction check against the reference measurement
    c = res["screens"].get("flagship_v1_step29999_comma2k19")
    if c:
        ref = LS.REFERENCE_LATENTS["flagship_v1_step29999_comma2k19"]
        res["reproduction_check"] = {
            "reference": ref,
            "measured_now": {
                "jitter_ratio": c["screens"]["jitter_ratio"]["value"],
                "derivative_corr": c["screens"]["derivative_corr"]["value"],
                "derivative_corr_per_position": c["screens"]["derivative_corr"][
                    "per_position_form"],
                "derived_accel_r2": c["screens"]["derived_accel_r2"]["value"],
                "derived_accel_r2_upper_bound": c["screens"]["derived_accel_r2"][
                    "upper_bound_best_stencil"],
                "cos_adjacent_100ms": c["screens"]["cos_adjacent_100ms"]["value"]},
            "note": "the reference numbers were produced by the one-off run_mechanism.py / "
                    "run_precision_ladder.py; small differences are expected where the "
                    "promoted module fixes a stencil instead of sweeping it, and any "
                    "difference is reported rather than tuned away."}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
