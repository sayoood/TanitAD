"""Pilot statistics for PILOT_REPORT.md (every number MEASURED on the pilot output; the script prints and writes F:/strat_out/pilot/PILOT_STATS.json).
Reads: pilot job logs (throughput), merged frame parquets (distributions, flips, agreement, illumination cross-check, sizes), the schema pin."""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_common as P
import extract_context as X
import merge_context as M

POUT = os.environ.get("PILOT_DIR", "F:/strat_out/pilot")
N_FULL = 4659
SIG = {"weather": X.WEATHER, "surface": X.SURFACE, "visibility": X.VISIB, "view_quality": X.VIEWQ, "illum_img": X.ILLUM}


def throughput(logdirs):
    A, B, jobs = [], [], []
    for d in logdirs:
        for lf in glob.glob(d + "/job_*.log"):
            load = done = el = None
            stage = os.path.basename(lf).split("_")[1]
            for l in open(lf, errors="ignore"):
                if not l.startswith("{"):
                    continue
                try:
                    j = json.loads(l)
                except ValueError:
                    continue
                if "model_load_s" in j:
                    load = j["model_load_s"]
                if "done_this_job" in j:
                    done, el = j["done_this_job"], j["elapsed_s"]
                if "siglip_s" in j:
                    A.append(j)
                if "qwen_s" in j:
                    B.append(j)
            if done is not None:
                jobs.append({"stage": stage, "log": lf, "model_load_s": load, "done": done, "elapsed_s": el})
    return A, B, jobs


def rate(jobs, stage):
    js = [j for j in jobs if j["stage"] == stage and j["done"] and j["model_load_s"] is not None]
    if not js:
        return None
    work = sum(j["elapsed_s"] - j["model_load_s"] for j in js)
    done = sum(j["done"] for j in js)
    loads = [j["model_load_s"] for j in js]
    return {"clips": done, "s_per_clip_wall": work / done, "model_load_s_mean": float(np.mean(loads)), "model_load_s_max": max(loads), "jobs": len(js)}


def project(r, per_job_min=30):
    if not r:
        return None
    clips_per_job = max(1, int((per_job_min * 60 - r["model_load_s_mean"]) / r["s_per_clip_wall"]))
    jobs = math.ceil(N_FULL / clips_per_job)
    h = (N_FULL * r["s_per_clip_wall"] + jobs * r["model_load_s_mean"]) / 3600
    return {"clips_per_30min_job": clips_per_job, "jobs": jobs, "hours": round(h, 1), "hours_no_load_overhead": round(N_FULL * r["s_per_clip_wall"] / 3600, 1)}


def flips(df, col):
    """share of consecutive 0.2 s frame pairs (same clip) whose label differs."""
    d = df.sort_values(["clip_id", "frame_idx"])
    same = d.clip_id.values[1:] == d.clip_id.values[:-1]
    ch = (d[col].values[1:] != d[col].values[:-1]) & same
    return float(ch.sum() / max(same.sum(), 1)), int(ch.sum()), int(same.sum())


def main():
    import pandas as pd
    st = {}
    A, B, jobs = throughput([POUT + "/logs"])
    smokeA, smokeB, smoke_jobs = throughput(["F:/strat_smoke/logs"])
    ra, rb = rate(jobs, "A"), rate(jobs, "B")
    st["throughput"] = {"A": ra, "B": rb, "n_A_clip_lines": len(A), "n_B_clip_lines": len(B),
                        "A_stage_means_s": {k: round(float(np.mean([a[k] for a in A])), 2) for k in ("decode_wait_s", "stats_s", "siglip_s")} if A else None,
                        "B_stage_means_s": {k: round(float(np.mean([b[k] for b in B])), 2) for k in ("decode_wait_s", "qwen_s")} if B else None,
                        "siglip_frames_per_s": round(101 / float(np.mean([a["siglip_s"] for a in A])), 1) if A else None,
                        "qwen_prompts_per_s": round(84 / float(np.mean([b["qwen_s"] for b in B])), 2) if B else None,
                        "jobs": jobs, "evidence": "MEASURED on the pilot job logs"}
    pa, pb = project(ra), project(rb)
    st["projection_full_4659"] = {"A": pa, "B": pb,
                                  "hours_without_qwen": pa["hours"] if pa else None,
                                  "hours_with_qwen_1hz": round(pa["hours"] + pb["hours"], 1) if pa and pb else None,
                                  "evidence": "ESTIMATED from the measured pilot rates, linear in clips; ignores lock waits and other sessions' GPU use"}
    fs = sorted(glob.glob(POUT + "/frames/frames_*.parquet"))
    if not fs:
        json.dump(st, open(POUT + "/PILOT_STATS.json", "w"), indent=1)
        print("no merged frames yet")
        return
    cols = None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    st["n_clips"], st["n_frames"] = int(df.clip_id.nunique()), int(len(df))
    st["schema_errors"] = M.check_frames_schema(POUT + "/frames")
    st["n_columns"] = len(df.columns)
    st["parquet_bytes_total"] = int(sum(os.path.getsize(f) for f in fs))
    st["parquet_bytes_per_clip"] = int(st["parquet_bytes_total"] / st["n_clips"])
    st["projected_full_parquet_GB"] = round(st["parquet_bytes_per_clip"] * N_FULL / 2 ** 30, 2)
    st["npz_bytes_per_clip"] = {s: int(np.mean([os.path.getsize(p) for p in glob.glob(f"{POUT}/ctx{s}/*.npz")])) for s in "AB" if glob.glob(f"{POUT}/ctx{s}/*.npz")}
    # coverage
    st["coverage"] = {"frame_present_share": float(df.frame_present.mean()) if "frame_present" in df else None,
                      "rows_per_clip_min_max": [int(df.groupby("clip_id").size().min()), int(df.groupby("clip_id").size().max())],
                      "vlm_frame_share": float(df.vlm_frame.mean()), "countries": int(df.country.nunique())}
    # distributions and unknown rate
    dist = {}
    for c, classes in SIG.items():
        vc = df[c].value_counts()
        dist[c] = {"frames": {k: int(v) for k, v in vc.items()}, "unknown_rate": float((df[c] == "unknown").mean()),
                   "mean_max_prob": float(np.stack(df[c + "_p"].to_numpy()).max(1).mean())}
        if c + "_src" in df:
            dist[c]["src_share"] = {k: round(float(v), 3) for k, v in df[c + "_src"].value_counts(normalize=True).items()}
    st["distribution"] = dist
    # flips
    fl = {}
    for c in SIG:
        r, n, tot = flips(df, c)
        fl[c] = {"all_frames_flip_rate": round(r, 4), "flips": n, "pairs": tot}
        v = df[df.vlm_frame]
        r2, n2, t2 = flips(v, c)
        fl[c]["qwen_frames_only_flip_rate_1s"] = round(r2, 4)
        fl[c]["qwen_pairs"] = t2
    for c, zc in (("weather", "z_weather_p"), ("surface", "z_surface_p"), ("view_quality", "z_view_quality_p")):
        df["_z" + c] = np.array(SIG[c])[np.stack(df[zc].to_numpy()).argmax(1)]
        r, n, tot = flips(df, "_z" + c)
        fl[c]["siglip_zeroshot_flip_rate_0.2s"] = round(r, 4)
    df["_zillum"] = df.illum_img
    st["flip_rate"] = {"estimator": "share of consecutive frame pairs (same clip) with a different label; weather should be low", **fl}
    # SigLIP vs Qwen at Qwen frames (argmax, no unknown threshold)
    v = df[df.vlm_frame]
    agree = {}
    for c, zc, qc in (("weather", "z_weather_p", "weather_p"), ("surface", "z_surface_p", "surface_p"), ("view_quality", "z_view_quality_p", "view_quality_p")):
        z, q = np.stack(v[zc].to_numpy()).argmax(1), np.stack(v[qc].to_numpy()).argmax(1)
        K = len(SIG[c])
        conf = np.zeros((K, K), int)
        for a_, b_ in zip(q, z):
            conf[a_, b_] += 1
        agree[c] = {"n": int(len(v)), "agree": int((z == q).sum()), "rate": round(float((z == q).mean()), 4), "classes": list(SIG[c]),
                    "confusion_rows_qwen_cols_siglip": conf.tolist()}
    agree["visibility"] = "no SigLIP prompt set for visibility: Qwen only (not comparable)"
    st["siglip_vs_qwen"] = agree
    # illumination image against hour and sun
    ill = {}
    ct = pd.crosstab(df.illum_img, df.sun_band_est)
    ill["image_x_sun_band_frames"] = {a_: {b_: int(ct.loc[a_, b_]) for b_ in ct.columns} for a_ in ct.index}
    ct2 = pd.crosstab(df.illum_img, df.period_of_day)
    ill["image_x_period_of_day_frames"] = {a_: {b_: int(ct2.loc[a_, b_]) for b_ in ct2.columns} for a_ in ct2.index}
    ill["flag_share_two_classes_away_from_sun_band"] = float(df.illum_sun_flag.mean())
    ill["agree_sun_share"] = float(df.illum_agree_sun.mean())
    rule = [P.illum_rule(*r) for r in df[["luma_mean", "luma_p95", "sky_luma", "headlight_frac", "lit_frac"]].itertuples(index=False)]
    df["_rule"] = rule
    ill["rule_vs_siglip_agree"] = round(float((df._rule == df.illum_img).mean()), 4)
    ill["rule_distribution"] = {k: int(x) for k, x in df._rule.value_counts().items()}
    try:
        from scipy.stats import spearmanr
        ill["spearman_sky_luma_vs_sun_el"] = round(float(spearmanr(df.sky_luma, df.sun_el_deg_est)[0]), 3)
    except Exception as e:
        ill["spearman_sky_luma_vs_sun_el"] = f"unavailable: {e!r}"
    night_clock = df[df.period_of_day == "night"]
    ill["clock_night_frames"] = int(len(night_clock))
    ill["clock_night_frames_image_day_or_glare_share"] = round(float(night_clock.illum_img.isin(["day", "glare"]).mean()), 4) if len(night_clock) else None
    ill["clock_night_frames_luma_mean_below_30_share"] = round(float((night_clock.luma_mean < 30).mean()), 4) if len(night_clock) else None
    st["illumination_crosscheck"] = {**ill, "evidence": "MEASURED; hour and sun band are weak cross-checks, never labels"}
    json.dump(st, open(POUT + "/PILOT_STATS.json", "w"), indent=1, default=float)
    print(json.dumps({k: st[k] for k in ("n_clips", "n_frames", "schema_errors", "parquet_bytes_per_clip")}))


if __name__ == "__main__":
    main()
