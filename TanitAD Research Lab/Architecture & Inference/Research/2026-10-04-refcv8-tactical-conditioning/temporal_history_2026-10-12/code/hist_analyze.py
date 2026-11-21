"""CPU analysis of the temporal-history experiment (PREREG.md). usage:
python hist_analyze.py <out_root> <real_s0> <real_s1> <hold> <st> <nt>   (dump dirs)
Builds the S2 panel and the S6 dump with arms os (REAL), os_rep (inference seed 1), os_hold, os_st, os_nt; runs the battery
analysis (four families per arm), the paired cross-arm deltas vs REAL, LEAD-VIOL (S6, S2; paired), and the PREREG s5 verdict."""
import glob, json, os, sys, tempfile, time
from pathlib import Path
out_root = Path(sys.argv[1]); real, rep, hold, st, nt = sys.argv[2:7]
BAT = Path(os.environ.get("BATT_CODE", "E:/hold_temporal/work/battery")); sys.path.insert(0, str(BAT))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import tanitad  # noqa: E402
import r7_panel as P  # noqa: E402
import r7_roll as RR  # noqa: E402
from taniteval import lead_metrics as lm  # noqa: E402
print("tanitad from", tanitad.__file__, flush=True)
LEAD12 = "C:/Users/Admin/qland/landed_hold_x7/artifacts/b1_eval_lead_block_k12_dt0p5.npz"
assert os.path.exists(LEAD12)
P.LEAD_BLOCK = LEAD12
KIT_LABELS = "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"
ra = RR.ra3()
N_BOOT = int(os.environ.get("N_BOOT", "2000"))
ARMS = {"os_rep": rep, "os_hold": hold, "os_st": st, "os_nt": nt}
P.TIERS.update({k: "T1" for k in ARMS})
out_root.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def J(o, n):
    json.dump(o, open(out_root / n, "w", encoding="utf-8"), indent=1, default=str)


# ---- panels ----------------------------------------------------------------------------------------------------
s2 = str(out_root / "panel_s2")
s6 = str(out_root / "s6")
if not os.path.exists(s2 + "/manifest.json"):
    J(P.build_panel_dump(real, s2, None, baselines=ARMS), "panel_s2_build.json")
if not os.path.exists(s6 + "/manifest.json"):
    J(P.build_s6_dump(real, s6, baselines=ARMS), "s6_build.json")
print("[A] panels", round(time.time() - t0), flush=True)
res = {}
for lab, pdir in (("S2", s2), ("S6", s6)):
    an = P.analyze(pdir, str(out_root / f"analysis_{lab.lower()}.json"), KIT_LABELS, n_boot=N_BOOT)
    pairs = [("os", x, f"{x}_minus_os") for x in ARMS]
    cp = P.cross_paired(pdir, pairs, n_boot=N_BOOT)
    J(cp, f"cross_paired_{lab.lower()}.json")
    res[lab] = {"an": an, "cp": cp}
    print("[A]", lab, "analysed", round(time.time() - t0), flush=True)


# ---- LEAD-VIOL (S6 and S2), paired vs REAL ----------------------------------------------------------------------
def join(panel_dir, man):
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    raw_off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
    dt, k = float(man["grid"]["dt_s"]), int(man["grid"]["k"])
    view, idx, meta, info = ra.lead_block_common_grid(LEAD12, dt, k)
    assert view is not None, info
    sd = tempfile.mkdtemp(prefix="hist_leadjoin_")
    shim = []
    for f in files:
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1)
        sf = os.path.join(sd, os.path.basename(f))
        np.savez(sf, ws=ws + raw_off, v0=v0)
        shim.append(sf)
    lead = ra.ra.join_lead_block(shim, man, view, idx, k=info["k"], dt=info["dt_s"], frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    return lead, cov, info


def lead_one(pdir):
    man, A, eid = P.load_panel(pdir)
    lead, cov, info = join(pdir, man)
    cols, dtv = info["dump_cols"], info["dt_s"]
    lv = np.asarray(lead["state"]).astype(str) == "LEAD"
    G = A["g"][:, cols].astype(np.float64)
    hdk = lm.distance_keeping(G, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
    flags = {}
    out = {"n_windows": int(len(eid)), "n_lead_windows": int(lv.sum()), "instants_s": info["instants_s"],
           "arms": {}, "paired_vs_os": {}}
    for a in ["os", "os_rep", "os_hold", "os_st", "os_nt", "stop", "ha0_ext"]:
        if a not in A:
            continue
        plan = A[a][:, cols].astype(np.float64)
        dk = lm.distance_keeping(plan, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
        lvr = lm.lead_viol(dk, hdk, eid, lead_valid=lv, n_boot=N_BOOT, seed=0)
        flags[a] = lvr
        prog, ph = plan[:, -1, 0], G[:, -1, 0]
        ratio = prog / np.where(np.abs(ph) > 1e-6, ph, np.nan)
        out["arms"][a] = {**{k: v for k, v in lvr.items() if k != "_per_window"},
                          "progress_ratio_mean": float(np.nanmean(ratio[lv])) if lv.any() else None,
                          "distance_keeping": {k: dk.get(k) for k in ("n", "mean_headway_min_m", "mean_time_gap_min_s",
                                                                      "n_time_gap", "mean_min_ttc_s", "n_closing")}}
    for x in ("os_rep", "os_hold", "os_st", "os_nt"):
        if x in flags:
            out["paired_vs_os"][x] = lm.paired_lead_viol(flags[x], flags["os"], eid, names=(x, "os"),
                                                         n_boot=N_BOOT, seed=0)
    return out


res["lead"] = {lab: lead_one(p) for lab, p in (("S6", s6), ("S2", s2))}
J(res["lead"], "lead_viol.json")
print("[A] lead-viol", round(time.time() - t0), flush=True)

# ---- PREREG s5 verdict (primaries P1-P5, S6) --------------------------------------------------------------------
cp6, ld6 = res["S6"]["cp"], res["lead"]["S6"]
means = cp6["means"]
PRIM = [("P1 ADE_m (S6)", "ADE", "ade_m", "err"), ("P2 along-track MAE (S6)", "longitudinal", "LON_along_mae_m", "err"),
        ("P3 cross-track MAE (S6)", "lateral", "LAT_cross_mae_m", "err"),
        ("P4 TAC lon-decision acc (S6)", "tactical", "TAC_traj_lon_correct", "acc")]


def classify(delta, lo, hi, base, kind, floor):
    d = 0.01 if kind in ("acc", "rate") else 0.02 * abs(base)
    if lo >= -d and hi <= d:
        return "EQUIVALENT", d
    err_like = kind in ("err", "rate")
    excl_harm = (lo > 0) if err_like else (hi < 0)
    excl_good = (hi < 0) if err_like else (lo > 0)
    if excl_harm and abs(delta) > max(d, 3 * floor):
        return "DEGRADED", d
    if excl_good and abs(delta) > max(d, 3 * floor):
        return "IMPROVED", d
    return "INDETERMINATE", d


ver = {}
for x in ARMS:
    rows = []
    for nm, fam, key, kind in PRIM:
        c = cp6["pairs"][f"{x}_minus_os"]["families"][fam][key]
        fl = cp6["pairs"]["os_rep_minus_os"]["families"][fam][key]["delta"]
        base = means["os"].get(key, float("nan")) if kind == "err" else 0.0
        v, d = classify(c["delta"], c["lo"], c["hi"], base, kind, abs(fl))
        rows.append({"metric": nm, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "base_REAL": base, "margin": d,
                     "seed_floor_abs": abs(fl), "class": v, "n": c["n_windows"], "n_ep": c["n_episodes"]})
    pv = ld6["paired_vs_os"][x].get("at_risk", {})
    fl = ld6["paired_vs_os"]["os_rep"].get("at_risk", {}).get("delta", 0.0) or 0.0
    if "delta" in pv:
        v, d = classify(pv["delta"], pv["lo"], pv["hi"], 0.0, "rate", abs(fl))
        rows.append({"metric": "P5 LEAD-VIOL share_of_at_risk (S6)", "delta": pv["delta"], "lo": pv["lo"], "hi": pv["hi"],
                     "base_REAL": None, "margin": d, "seed_floor_abs": abs(fl), "class": v,
                     "n": pv.get("n_windows"), "n_ep": pv.get("n_episodes")})
    else:
        rows.append({"metric": "P5 LEAD-VIOL share_of_at_risk (S6)", "class": "UNAVAILABLE", "raw": pv})
    cls = [r["class"] for r in rows]
    nd = cls.count("DEGRADED")
    verdict = "EQUIVALENT" if all(c == "EQUIVALENT" for c in cls) else (
        "DEGRADED" if (rows[0]["class"] == "DEGRADED" or nd >= 2) else "MIXED")
    ver[x] = {"verdict": verdict, "rows": rows}
J(ver, "verdict_s6.json")
for x, v in ver.items():
    print(x, v["verdict"])
    for r in v["rows"]:
        if "delta" in r:
            print("   ", r["metric"], "d=%+.4f [%+.4f,%+.4f] margin %.4f floor %.4f -> %s" % (
                r["delta"], r["lo"], r["hi"], r["margin"], r["seed_floor_abs"], r["class"]))
        else:
            print("   ", r)
print("ZZHISTANALYZE_DONE", round(time.time() - t0))
