"""r7base stage B (CPU only): LEAD-VIOL (X7) on the refcv7-50,400 S6 (6 s) and S2 (2 s) panels, both inference seeds.

`refcv3_arm._distance_keeping` patches `four_families.longitudinal.distance_keeping` with a block that has NO
`lead_viol` (MEASURED: lead_metrics.lead_viol is called only from four_families._distance_keeping, which the
refcv3 analysis path replaces), so LEAD-VIOL is computed here from the SAME join (refcv3_arm.lead_block_common_grid +
refav1_arm.join_lead_block, copied call for call) and the SAME lead_metrics functions.

usage: python r7base_stageB.py <out_root> <seed> [<seed> ...]
"""
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

out_root = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
BAT = Path(__file__).resolve().parent / "battery"
sys.path.insert(0, str(BAT))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import tanitad  # noqa: E402
import r7_panel as P  # noqa: E402
import r7_roll as RR  # noqa: E402
from taniteval import lead_metrics as lm  # noqa: E402
from taniteval.ci import episode_cluster_bootstrap  # noqa: E402

print("tanitad from", tanitad.__file__, "| lead_metrics", lm.__file__, flush=True)
BANK = Path("D:/refcv7_eval_kit/battery/step50400_a6text")
LEAD12 = "C:/Users/Admin/qland/landed_hold_x7/artifacts/b1_eval_lead_block_k12_dt0p5.npz"
ra = RR.ra3()
N_BOOT = 2000


def join(panel_dir: str, man: dict):
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    fr = man["corpus"]["frames"]
    raw_off = int(fr["provider_to_raw_frame_offset"])
    dt, k = float(man["grid"]["dt_s"]), int(man["grid"]["k"])
    view, idx, meta, info = ra.lead_block_common_grid(LEAD12, dt, k)
    assert view is not None, info
    shim_dir = tempfile.mkdtemp(prefix="r7base_leadjoin_")
    shim = []
    for f in files:
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1)
        sf = os.path.join(shim_dir, os.path.basename(f))
        np.savez(sf, ws=ws + raw_off, v0=v0)
        shim.append(sf)
    lead = ra.ra.join_lead_block(shim, man, view, idx, k=info["k"], dt=info["dt_s"], frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    return lead, cov, info, raw_off


def boot_mean(x, ids):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"status": "NOT-APPLICABLE", "n": 0}
    r = episode_cluster_bootstrap(x, list(ids), reduce="mean", n_boot=N_BOOT, seed=0)
    return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator") if k in r}


def run_one(label: str, panel_dir: str, seed: int) -> dict:
    t0 = time.time()
    man, arrs, eid_f = P.load_panel(panel_dir)
    lead, cov, info, raw_off = join(panel_dir, man)
    cols = info["dump_cols"]
    dtv = info["dt_s"]
    state = np.asarray(lead["state"]).astype(str)
    lv = state == "LEAD"
    W = len(state)
    assert W == arrs["g"].shape[0] == len(eid_f), (W, arrs["g"].shape, len(eid_f))
    eid = np.asarray(eid_f)
    G = arrs["g"][:, cols].astype(np.float64)
    human_dk = lm.distance_keeping(G, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
    arms = [a for a in ("os", "ha0", "ha0_ext", "ha", "stop") if a in arrs]
    res = {"label": label, "seed": seed, "panel_dir": panel_dir, "n_windows": int(W),
           "n_episodes": int(len(set(eid.tolist()))),
           "grid": {"instants_s": info["instants_s"], "dt_s": dtv, "k": info["k"],
                    "block_grid_s": info["block_grid_s"], "dropped_dump_instants_s": info["dropped_dump_instants_s"]},
           "raw_offset": raw_off, "lead_coverage": cov["counts"], "n_lead_windows": int(lv.sum()),
           "lead_block": LEAD12, "arms": {}, "paired": {}}
    flags = {}
    prog_h = G[:, -1, 0]
    dks = {}
    for arm in arms:
        plan = arrs[arm][:, cols].astype(np.float64)
        dk = lm.distance_keeping(plan, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
        dks[arm] = dk
        lvres = lm.lead_viol(dk, human_dk, eid, lead_valid=lv, n_boot=N_BOOT, seed=0)
        flags[arm] = lvres
        v = lvres["_per_window"]["viol"]
        htg = np.asarray(human_dk["_per_window"]["time_gap_min_s"] if "_per_window" in human_dk
                         else human_dk["time_gap_min_s"], dtype=np.float64)
        at_risk = lv & np.isfinite(htg) & (htg >= 1.0)
        prog = plan[:, -1, 0]
        ratio = prog / np.where(np.abs(prog_h) > 1e-6, prog_h, np.nan)
        keep_prog = at_risk & np.isfinite(ratio) & (ratio >= 0.9)
        r = {k_: v_ for k_, v_ in lvres.items() if k_ != "_per_window"}
        r["share_of_at_risk_ci"] = boot_mean(v[at_risk], eid[at_risk])
        r["progress_control"] = {
            "what": "plan along-track x at the LAST grid instant (%.1f s), vs the human's, over the lead windows" % info["instants_s"][-1],
            "plan_mean_progress_m": float(np.mean(prog[lv])) if lv.any() else None,
            "human_mean_progress_m": float(np.mean(prog_h[lv])) if lv.any() else None,
            "mean_progress_ratio_plan_over_human": (float(np.nanmean(ratio[lv])) if lv.any() else None),
            "n_at_risk": int(at_risk.sum()),
            "n_at_risk_with_progress_ratio_ge_0p9": int(keep_prog.sum()),
            "n_viol_with_progress_ratio_ge_0p9": int(np.nansum(v[keep_prog])),
            "share_of_at_risk_progress_matched": (float(np.nansum(v[keep_prog]) / keep_prog.sum()) if keep_prog.sum() else None),
            "_is": "a plan that brakes (STOP) reads LEAD-VIOL 0; the progress-matched share counts only at-risk windows where the plan covered >= 90 % of the human's distance",
        }
        res["arms"][arm] = r
    for b in ("stop", "ha0", "ha0_ext"):
        if "os" in flags and b in flags:
            res["paired"]["os_minus_" + b] = lm.paired_lead_viol(flags["os"], flags[b], eid, names=("os", b),
                                                                 n_boot=N_BOOT, seed=0)
    # distance keeping level for os (n, closing) beside the share
    dkos = dks["os"]
    res["os_distance_keeping_level"] = {k_: dkos.get(k_) for k_ in
        ("status", "n", "n_windows", "mean_headway_min_m", "mean_time_gap_min_s", "n_time_gap", "mean_min_ttc_s", "n_closing")}
    res["human_distance_keeping_level"] = {k_: human_dk.get(k_) for k_ in
        ("status", "n", "mean_headway_min_m", "mean_time_gap_min_s", "n_time_gap", "mean_min_ttc_s", "n_closing")}
    res["wall_s"] = round(time.time() - t0, 1)
    return res


for s in seeds:
    out = {}
    for label, pdir in (("S6", str(out_root / f"s6_s{s}")), ("S2", str(BANK / f"panel_s{s}"))):
        out[label] = run_one(label, pdir, s)
        a = out[label]["arms"]["os"]
        print(label, "seed", s, "n_lead", out[label]["n_lead_windows"], "os share", a.get("share"),
              "at-risk", a.get("share_of_at_risk"), "n_risk", a.get("n_at_risk"), flush=True)
    json.dump(out, open(out_root / f"lead_viol_s{s}.json", "w", encoding="utf-8"), indent=1, default=str)
print("ZZSTAGEB_DONE")
