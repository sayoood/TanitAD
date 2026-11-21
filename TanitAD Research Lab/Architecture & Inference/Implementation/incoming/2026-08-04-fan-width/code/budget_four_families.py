"""FOUR METRIC FAMILIES across the decode-budget ladder — per family, never pooled.

Binding rule (Sayed, 2026-08-02): every eval reports LONGITUDINAL (target-speed
AND distance-keeping/TTC), LATERAL (heading, curvature, yaw-rate, cross-track),
TACTICAL and STRATEGIC — ADE alone is an INCOMPLETE result and an ADE horizon
sweep is ONE ROW OF FOUR.

⭐ Distance-keeping is computable here with NO re-inference: the row-aligned
``val40_lead_block.npz`` (Benchmarks & Eval / 2026-08-04-distance-keeping-arms)
reproduces the canonical 881 windows in the banks' own row order, and
``taniteval.lead_metrics`` scores an arbitrary predicted path against it.

⚠️ STRATIFIED BY SPEED, because 20.7 % of lead windows sit at 0-1 m/s where the
metric cannot discriminate at all, and the 15+ band is UNPOWERED (n = 2). A
pooled headway would average regimes that do not resemble each other.

ESTIMATOR: ``paired_episode_cluster_bootstrap``, unit = episode, n_boot = 2000.
⛔ ``overlapping_holdout_se`` is never called.

Pre-registration: ``PREREG_FAN_WIDTH.md`` §8, blob
``1bffa9db6a6047325dceff1ef787d67ab2fd5152``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

ACCEL_MAX, HORIZON_S = 2.5, 2.0
N_BOOT = 2000
LADDER = [8, 16, 23, 32, 46, 48, 64, 92, 96, 128, 192, 256]


def band_prefix_idx(keep, n):
    return torch.argsort((~keep).long(), dim=1, stable=True)[:, :n].contiguous()


def three_sided(p, *, signed_metric: bool = False):
    """better / worse / not separated — prereg §4.

    ⛔ ON A **SIGNED** COMPONENT THERE IS NO SUCH VERDICT. `speed_signed_err_mps`
    is a BIAS: a more negative delta means the arm under-predicts speed MORE, not
    that it is better. Labelling a bias shift "better" because the delta is
    negative would be a direction predicate reading the wrong direction — the
    same class as a registered trigger that fired literally while its controls
    beat the score. Signed rows therefore carry the delta and NO verdict.
    """
    if signed_metric:
        return ("n/a — SIGNED bias; the sign of the delta is a DIRECTION, not a "
                "quality verdict. Read it beside the |abs| row.")
    if not p["separated"]:
        return "not separated"
    return "better" if p["delta"] < 0 else "worse"


def per_window_components(pred, gt, dt, four_families):
    """[n] per-window LONGITUDINAL / LATERAL components for PAIRED intervals.

    Built from ``four_families._seq_geometry`` — the SAME geometry the aggregate
    block uses, at the SAME derived ``dt`` — so the paired delta and the headline
    scalar can never drift apart. ⚠️ ``dt`` must come from ``infer_dt``: on the
    sparse 4-waypoint view it is 0.5 s, and a hard-coded 0.1 inflates speed 5x
    and accel 25x (R-2026-08-03-c).
    """
    P = four_families._seq_geometry(pred, dt)
    G = four_families._seq_geometry(gt, dt)
    both = P["valid"] & G["valid"]
    bp = P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    def rm(x, m):
        m = m.to(x.dtype)
        return (x.abs() * m).sum(1) / m.sum(1).clamp_min(1e-9)

    return {
        "LONGITUDINAL/speed_abs_err_mps": (P["speed"] - G["speed"]).abs().mean(1),
        "LONGITUDINAL/speed_signed_err_mps": (P["speed"] - G["speed"]).mean(1),
        "LONGITUDINAL/along_abs_err_m": (P["along"] - G["along"]).abs().mean(1),
        "LONGITUDINAL/along_signed_err_m": (P["along"] - G["along"]).mean(1),
        "LATERAL/cross_abs_err_m": (P["cross"] - G["cross"]).abs().mean(1),
        "LATERAL/heading_abs_err_deg": rm(dh, both) * 180.0 / math.pi,
        "LATERAL/curvature_abs_err_1pm": rm(P["curvature"] - G["curvature"], bp),
        "LATERAL/yaw_rate_abs_err_degps": rm(P["yaw_rate"] - G["yaw_rate"], bp)
                                          * 180.0 / math.pi,
    }


def run(bank, anchors_path, lead_npz, out_dir, arm, repo):
    t0 = time.time()
    for p in (repo / "taniteval", repo / "stack"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from taniteval import ci, four_families, lead_metrics          # noqa: PLC0415
    from tanitad.refs import refc_select as sl                     # noqa: PLC0415

    d = torch.load(bank, map_location="cpu", weights_only=False)
    fan, logits, gt = d["fan"], d["logits"], d["gt"]
    B, NMAX = fan.shape[:2]
    eid = list(d["eid"])
    v0 = d["v0"].to(fan.dtype)
    V = torch.load(anchors_path, map_location="cpu", weights_only=False)
    anc = (V[arm] if arm in V else V[max(V, key=lambda k: V[k].shape[0])]
           )[:NMAX].to(fan.dtype)
    keep = sl.reachability_mask(anc[None].expand(B, -1, -1, -1).contiguous(),
                                v0, accel_max=ACCEL_MAX, horizon_s=HORIZON_S)
    sel_full = logits.argmax(1)
    dt, dt_prov = four_families.infer_dt({"wp_steps": list(d["wp_steps"]),
                                          "dt_s": 0.1})
    bidx = torch.arange(B)
    ref_pred = fan[bidx, sel_full]
    ref_cmp = per_window_components(ref_pred, gt, dt, four_families)

    # ---- the LEAD block — row-aligned, no re-inference --------------------- #
    z = np.load(lead_npz, allow_pickle=True)
    lead_ok = z["leads"].shape[0] == B
    align = {
        "npz": str(lead_npz), "npz_rows": int(z["leads"].shape[0]),
        "bank_rows": B,
        "eid_agrees_as_a_partition": bool(
            len({(a, b) for a, b in zip(z["eid"].tolist(), eid)}) == len(set(eid))),
        "ts_rel_s": z["ts_rel_s"].tolist(),
        "wp_steps": list(d["wp_steps"]),
        "n_LEAD": int((z["state"] == "LEAD").sum()),
        "n_NO_LEAD": int((z["state"] == "NO_LEAD").sum()),
        "n_NO_LABEL": int((z["state"] == "NO_LABEL").sum()),
        "status": "OK" if lead_ok else "SKIPPED — row counts differ",
    }

    def dk_of(pred: torch.Tensor) -> dict:
        return lead_metrics.distance_keeping(
            pred.numpy().astype(np.float64), z["leads"], z["lead_lens"],
            z["speeds"], dt=dt)

    dk_ref = dk_of(ref_pred) if lead_ok else None

    res = {
        "what": "four metric families across the decode-budget ladder",
        "arm": arm, "bank": str(bank), "n_windows": B, "n_anchors": NMAX,
        "n_episodes": len(set(eid)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dt_s": dt, "dt_provenance": dt_prov,
        "estimator": {"paired": "paired_episode_cluster_bootstrap",
                      "unit": "episode", "n_boot": N_BOOT,
                      "⛔": "overlapping_holdout_se is NEVER called"},
        "lead_block_alignment": align,
        "reference_arm": "FULL fan (argmax over all N_max candidates)",
        "families_at_full_fan": {
            "LONGITUDINAL_and_LATERAL": four_families.all_families(
                {"pred": ref_pred, "gt": gt, "wp_steps": list(d["wp_steps"]),
                 "dt_s": 0.1}),
            "LONGITUDINAL/distance_keeping": dk_ref,
        },
        "TACTICAL": {
            "measured_half": ("goal/anchor SELECTION — rank_acc, sel_gap, "
                              "frac_sel_2x_worse — reported in "
                              "fan_width_<arm>.json and reachable_budget.json. "
                              "This is the half fan width acts on."),
            "unavailable_half": ("manoeuvre DECISION (selected vs executed, the "
                                 "5-way confusion) needs decoded manoeuvre "
                                 "logits, which refc_rerank.dump does not "
                                 "store. n = 0. A WORK ITEM, not a pass."),
        },
        "STRATEGIC": {
            "status": "UNAVAILABLE", "n": 0,
            "reason": ("no route/goal label in a fan bank, and the decode ran "
                       "with nav_mode='follow_constant' so the route input was "
                       "never exercised. n = 0 of "
                       f"{B} windows. A WORK ITEM, not a pass."),
        },
        "rungs": [],
    }

    for n in [x for x in LADDER if x <= NMAX] + [NMAX]:
        ix = band_prefix_idx(keep, n)
        j = logits.gather(1, ix).argmax(1)
        sel_n = ix.gather(1, j[:, None]).squeeze(1)
        pred = fan[bidx, sel_n]
        cmp = per_window_components(pred, gt, dt, four_families)
        row = {"n_decodes": n, "compute_saving_x": round(NMAX / n, 3),
               "selection_identical_to_full_fan_frac":
                   round(float((sel_n == sel_full).double().mean()), 6),
               "LONGITUDINAL": {}, "LATERAL": {}}
        for k, v in cmp.items():
            fam, name = k.split("/")
            p = ci.paired_episode_cluster_bootstrap(
                v.numpy().astype(np.float64),
                ref_cmp[k].numpy().astype(np.float64), eid, n_boot=N_BOOT)
            row[fam][name] = {
                "value": round(float(v.mean()), 6),
                "paired_vs_full_fan": p,
                "verdict": three_sided(p, signed_metric="signed" in name)}
        # ---- distance-keeping, and stratified by speed --------------------- #
        if lead_ok:
            dk = dk_of(pred)
            row["LONGITUDINAL"]["distance_keeping"] = {
                "status": dk["status"], "n_with_lead": dk["n"],
                "n_windows": dk["n_windows"],
                "mean_headway_min_m": dk.get("mean_headway_min_m"),
                "mean_time_gap_min_s": dk.get("mean_time_gap_min_s"),
                "n_time_gap": dk.get("n_time_gap"),
                "mean_min_ttc_s": dk.get("mean_min_ttc_s"),
                "n_closing": dk.get("n_closing"),
                "censoring_note": dk.get("censoring_note"),
            }
            if dk["status"] == "OK":
                pr = lead_metrics.paired_distance_keeping(
                    dk, dk_ref, eid, names=(f"budget_{n}", "full_fan"),
                    n_boot=N_BOOT)
                row["LONGITUDINAL"]["distance_keeping"]["paired_vs_full_fan"] = pr
                row["LONGITUDINAL"]["distance_keeping"]["by_speed_band"] = \
                    lead_metrics.distance_keeping_by_speed(
                        dk, z["speeds"], eid, states=z["state"], n_boot=N_BOOT)
        else:
            row["LONGITUDINAL"]["distance_keeping"] = {
                "status": "UNAVAILABLE", "n": 0,
                "reason": align["status"]}
        res["rungs"].append(row)
        print(f"  N={n:>4} sel_same={row['selection_identical_to_full_fan_frac']:.4f}"
              f" speed_mae={row['LONGITUDINAL']['speed_abs_err_mps']['value']:.4f}"
              f" cross_mae={row['LATERAL']['cross_abs_err_m']['value']:.4f}",
              flush=True)

    def _c(o):
        if isinstance(o, dict):
            return {k: _c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_c(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return (o.tolist() if o.size < 32 else f"<ndarray {o.shape}>")
        if isinstance(o, torch.Tensor):
            return (o.tolist() if o.numel() < 32 else f"<tensor {tuple(o.shape)}>")
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"budget_four_families_{arm}.json"
    dest.write_text(json.dumps(_c(res), indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[families] {arm} -> {dest} ({res['wall_s']}s)", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--lead-npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.bank, a.anchors, a.lead_npz, a.out, a.arm, Path(a.repo))
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
