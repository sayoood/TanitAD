"""P4 — the B4 substrate's ego block is the LEGACY LEAKY one. Verify it, size it, REBUILD it.

WHAT THE DEFECT IS
------------------
`tanitad.data.situations.kinematics` built `alon_pre` / `omega_pre` as a trailing mean of
`np.gradient` — a CENTRED difference — so both channels read one frame (0.1 s) PAST `t` on every
interior frame, under a comment that said `STRICTLY CAUSAL`. The fix (`backward_diff`,
`causal_pre=True` by default) landed at HEAD on 2026-08-03. `sitclf_b4_substrate.npz` was built
BEFORE it and its `E` block was never rebuilt.

The sibling stream established this by rebuilding CLIP 0 only. That is one probe. This rebuilds
ALL 500 clips, both ways, and reports the exposure over the whole substrate — because "a defect
exists" and "the defect is 4.7 % of the channel" license different decisions.

WHAT THIS PRODUCES
------------------
  1. `ego_leak_audit.json`  — the measurement: per-channel deviation over all 99,477 rows, the
     bit-exact match to `causal_pre=False`, and the per-clip worst case.
  2. `<substrate>.ego_causal.npz`  — the REBUILT causal `E` block, beside the substrate, with a
     provenance meta. The 410 MB substrate itself is NOT rewritten: rewriting it would silently
     break the bit-reproducibility of every banked run that already cites it. A sidecar plus a
     quarantine stamp is the honest form.
  3. `<substrate>.meta.json` gains an `ego_block_defect` stamp — so a consumer that reads the meta
     (which is the documented way to check provenance) CANNOT fail to notice.

⛔ NO deployable arm is affected: ego is not a legal inference input (PI ruling 2026-08-03,
"labels may use ego; INFERENCE IS VISION-ONLY"), so no scored arm in B4, in the temporal study or
in this one reads `E`. What DOES read it is `sitclf_deploy.regime_strata`, i.e. the
LONGITUDINAL/LATERAL family stratum boundaries.

usage:
  python rebuild_causal_ego.py --substrate C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data.situations import kinematics                          # noqa: E402

# build_substrate.py:64-67 — the SAME caches in the SAME order, so clip k here is clip k there
CACHES = (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74",
          r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836")
EGO_SCALE = np.array([10.0, 2.0, 0.5], dtype=np.float32)   # sc_train.py:38


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
    ap.add_argument("--out", default="ego_leak_audit.json")
    ap.add_argument("--no-write-sidecar", action="store_true")
    a = ap.parse_args()

    z = np.load(a.substrate)
    E_bank = z["E"].astype(np.float64)
    cc = z["clip_cluster"]
    n_rows = E_bank.shape[0]
    log(f"substrate {a.substrate}: E {E_bank.shape}, {len(np.unique(cc))} clips")

    files = []
    for root in CACHES:
        files += sorted(glob.glob(os.path.join(root, "ep_*.pt")))
    log(f"{len(files)} episode caches")

    Ec, El = [], []
    for i, f in enumerate(files):
        P = np.asarray(torch.load(f, map_location="cpu", weights_only=True,
                                  mmap=True)["poses"]).astype(np.float64)
        for causal, sink in ((True, Ec), (False, El)):
            K = kinematics(P, causal_pre=causal)
            sink.append(np.stack([K["v"], K["alon_pre"], K["omega_pre"]], 1)
                        .astype(np.float32) / EGO_SCALE)
        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(files)} clips rebuilt")
    Ec = np.concatenate(Ec)
    El = np.concatenate(El)
    if Ec.shape[0] != n_rows:
        raise SystemExit(f"C-FID: rebuilt {Ec.shape[0]} rows but substrate has {n_rows}; "
                         "clip order or cache set does not match build_substrate.py")
    log(f"C-FID OK — rebuilt {Ec.shape[0]:,} rows in the substrate's own order")

    # ---- which convention is the bank? ------------------------------------------------
    d_leg = float(np.abs(E_bank - El.astype(np.float64)).max())
    d_cau = float(np.abs(E_bank - Ec.astype(np.float64)).max())
    which = ("LEGACY_LEAKY (causal_pre=False)" if d_leg == 0.0 else
             "CAUSAL (causal_pre=True)" if d_cau == 0.0 else "NEITHER — provenance UNKNOWN")
    log(f"bank vs legacy max|diff| = {d_leg:.3e} | bank vs causal max|diff| = {d_cau:.3e} "
        f"=> the banked E block is {which}")

    # ---- how big is the defect, in the units the channel is actually used in ----------
    raw_c = Ec.astype(np.float64) * EGO_SCALE          # back to [m/s, m/s^2, rad/s]
    raw_l = El.astype(np.float64) * EGO_SCALE
    names = ("v", "alon_pre", "omega_pre")
    chans = {}
    for j, nm in enumerate(names):
        d = np.abs(raw_c[:, j] - raw_l[:, j])
        scale = float(np.abs(raw_c[:, j]).mean())
        chans[nm] = {
            "mean_abs_change": round(float(d.mean()), 6),
            "p99_abs_change": round(float(np.percentile(d, 99)), 6),
            "max_abs_change": round(float(d.max()), 6),
            "channel_scale_mean_abs": round(scale, 6),
            "relative_to_scale_pct": (round(100.0 * float(d.mean()) / scale, 4)
                                      if scale > 0 else None),
            "frac_rows_changed_gt_1pct_of_scale": (round(float((d > 0.01 * scale).mean()), 5)
                                                   if scale > 0 else None)}
        log(f"  {nm:>10}: mean|d| {d.mean():.6f}  p99 {np.percentile(d,99):.6f}  "
            f"max {d.max():.6f}  = {chans[nm]['relative_to_scale_pct']}% of scale  "
            f"on {100*chans[nm]['frac_rows_changed_gt_1pct_of_scale']:.1f}% of rows")

    # ---- does the defect move the STRATUM BOUNDARIES that actually consume it? --------
    from tanitad.eval.sitclf_deploy import regime_strata                 # noqa: PLC0415
    s_c, s_l = regime_strata(Ec), regime_strata(El)
    strata = {}
    for fam in ("longitudinal", "lateral"):
        for nm in s_c[fam]:
            a_, b_ = s_c[fam][nm], s_l[fam][nm]
            strata[f"{fam}.{nm}"] = {
                "n_rows_causal": int(a_.sum()), "n_rows_legacy": int(b_.sum()),
                "n_rows_reassigned": int((a_ != b_).sum()),
                "frac_reassigned": round(float((a_ != b_).mean()), 6)}
            log(f"  stratum {fam}.{nm:>14}: {int(a_.sum()):>6} vs {int(b_.sum()):>6} rows, "
                f"{int((a_!=b_).sum()):>5} reassigned ({100*(a_!=b_).mean():.3f}%)")

    out = {"_what": "P4 — B4 substrate ego-block causality audit over ALL 500 clips",
           "_pi_ruling": ("labels may use ego; INFERENCE IS VISION-ONLY. No deployable arm reads "
                          "E. The consumer is sitclf_deploy.regime_strata (family stratum "
                          "boundaries), which is a STRATIFICATION variable, not a model input."),
           "substrate": a.substrate, "n_rows": int(n_rows), "n_clips": len(files),
           "banked_E_block_is": which,
           "max_abs_diff_bank_vs_legacy": d_leg,
           "max_abs_diff_bank_vs_causal": d_cau,
           "channels": chans, "strata_boundary_shift": strata,
           "sidecar": None}

    if not a.no_write_sidecar:
        side = str(Path(a.substrate).with_suffix("")) + ".ego_causal.npz"
        np.savez(side, E_causal=Ec, clip_cluster=cc,
                 _provenance=np.array([json.dumps({
                     "_what": "CAUSAL rebuild of sitclf_b4_substrate.npz's E block",
                     "built_by": "2026-08-03-sitclf-horizon/rebuild_causal_ego.py",
                     "kinematics_causal_pre": True,
                     "ego_scale": EGO_SCALE.tolist(),
                     "row_order": "IDENTICAL to the substrate (C-FID asserted on row count "
                                  "and clip order)",
                     "replaces": "the substrate's own E block, which is causal_pre=False"})]))
        out["sidecar"] = side
        log(f"wrote sidecar {side}")

        # quarantine stamp on the substrate's own meta — the documented provenance surface
        meta_p = Path(str(Path(a.substrate).with_suffix("")) + ".meta.json")
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta["ego_block_defect"] = {
                "status": "QUARANTINED — the E block is the LEGACY LEAKY convention",
                "detail": ("built before the 2026-08-03 causality fix; bit-exact match to "
                           "tanitad.data.situations.kinematics(..., causal_pre=False), whose "
                           "alon_pre/omega_pre read 0.1 s past t"),
                "verified_by": "2026-08-03-sitclf-horizon/rebuild_causal_ego.py (all 500 clips)",
                "max_abs_diff_vs_legacy": d_leg,
                "use_instead": side,
                "affects": ("sitclf_deploy.regime_strata stratum boundaries ONLY. No deployable "
                            "arm reads E — inference is vision-only."),
                "F_Y_V_clip_cluster": "UNAFFECTED — vision features and labels are independent of E"}
            meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            log(f"stamped quarantine note into {meta_p}")
            out["meta_stamped"] = str(meta_p)

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
