"""Merge geometry shards with context parts into the release frame parquet, fill non-VLM frames with a probe, write the clip table.

  merge_context.py --geom F:/strat_out/geom --out F:/strat_out [--splits train4512,eval]

Per clip: VLM (Qwen) answers at frames k = 0, 5, ..., 100 are the reference for weather / surface / visibility / view_quality. A logistic
probe on [SigLIP2 embedding, Tier 1 statistics] fitted to the VLM argmax labels fills the other frames (out-of-fold by clip for the
training clips, so a clip's own VLM label never produces its own probe value); the probe probabilities are median-smoothed over 1 s
and the VLM label wins at its own frames. Illumination is image-only (SigLIP2 zero-shot), uncalibrated until the PI audit.
`country`, `month`, `hour_of_day` are metadata copies; the sun band is an ESTIMATE kept as a cross-check flag; `period_of_day` is a
label built from the hour for training and is never an input.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C
import extract_context as X

LAT = {'Austria': 47.5, 'Belgium': 50.6, 'Bulgaria': 42.7, 'Croatia': 45.1, 'Czechia': 49.8, 'Denmark': 56.0, 'Estonia': 58.7, 'Finland': 64.0,
       'France': 46.6, 'Germany': 51.1, 'Greece': 39.0, 'Hungary': 47.2, 'Italy': 42.8, 'Latvia': 56.9, 'Lithuania': 55.3, 'Luxembourg': 49.8,
       'Netherlands': 52.2, 'Poland': 52.1, 'Portugal': 39.6, 'Romania': 45.9, 'Slovakia': 48.7, 'Slovenia': 46.1, 'Spain': 40.2, 'Sweden': 62.0,
       'United States': 39.8}      # approximate country centroid latitudes (ESTIMATED)
CUM = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
SUN_ORD = {"night": 0, "twilight": 1, "low_sun": 2, "day": 2}
IMG_ORD = {"dark": 0, "lit_night": 0, "twilight": 1, "day": 2, "glare": 2}


def sun_el_deg(country, month, hour):
    """ESTIMATED solar elevation: country-centroid latitude, day 15 of the month, hour read as local solar time (mid-hour)."""
    doy = CUM[month - 1] + 15
    dec = math.radians(23.44) * math.sin(2 * math.pi * (doy - 81) / 365.0)
    lat = math.radians(LAT[country])
    ha = math.radians(15.0 * (hour + 0.5 - 12.0))
    return math.degrees(math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)))


def sun_band(el):
    return "night" if el < -6 else "twilight" if el < 0 else "low_sun" if el < 15 else "day"


def period_of_day(h):
    return "night" if (h >= 21 or h < 5) else "morning" if h < 11 else "midday" if h < 15 else "afternoon" if h < 18 else "evening"


FIELDS = (("weather", "weather", X.WEATHER, "v_weather", "z_weather"), ("surface", "surface", X.SURFACE, "v_surface", "z_surface"),
          ("visibility", "visibility", X.VISIB, "v_visibility", None), ("view_quality", "viewq", X.VIEWQ, "v_viewq", "z_viewq"))


CTX_COLUMNS = ['luma_mean', 'luma_p05', 'luma_p95', 'sky_luma', 'sat_mean', 'headlight_frac', 'lit_frac', 'illum_img_p', 'illum_img', 'weather_p', 'weather',
               'weather_src', 'z_weather_p', 'surface_p', 'surface', 'surface_src', 'z_surface_p', 'visibility_p', 'visibility', 'visibility_src',
               'view_quality_p', 'view_quality', 'view_quality_src', 'z_view_quality_p', 'vlm_frame', 'sun_el_deg_est', 'sun_band_est', 'sun_conf',
               'illum_agree_sun', 'illum_sun_flag', 'period_of_day', 'model_sha_siglip', 'model_sha_qwen', 'prompt_sha']   # pinned from the 2026-11-08 smoke merge
ROWS_PER_CLIP = 101


def check_frames_schema(frames_dir):
    """Schema pin for the merged frame parquets: geometry columns (C.GEOM_COLUMNS) then CTX_COLUMNS, 101 rows per clip, probabilities finite and summing to 1."""
    import pandas as pd
    errs = []
    fs = sorted(glob.glob(frames_dir + "/frames_*.parquet"))
    if not fs:
        return ["no frame parquets"]
    for f in fs:
        d = pd.read_parquet(f)
        want = list(C.GEOM_COLUMNS) + CTX_COLUMNS
        if list(d.columns) != want:
            errs.append(f"{os.path.basename(f)}: columns differ, missing {[c for c in want if c not in d.columns]}, extra {[c for c in d.columns if c not in want]}")
            continue
        if not (d.groupby("clip_id").size().between(ROWS_PER_CLIP - 1, ROWS_PER_CLIP).all() and bool(d.frame_present.all())):      # one row per frame that exists: a video can end one frame short of the 0.2 s grid (3 of the 200 pilot clips)
            errs.append(f"{os.path.basename(f)}: rows per clip outside {ROWS_PER_CLIP - 1} to {ROWS_PER_CLIP}, or a row without a frame")
        for c in ("weather_p", "surface_p", "visibility_p", "view_quality_p", "illum_img_p"):
            a = np.stack(d[c].to_numpy())
            if not np.isfinite(a).all() or abs(a.sum(1) - 1).max() > 1e-2:
                errs.append(f"{os.path.basename(f)}: {c} not a finite distribution")
    return errs


def median_smooth(p, vlm_mask, w=5):
    out = p.copy()
    h = w // 2
    for i in range(len(p)):
        if not vlm_mask[i]:
            m = np.median(p[max(0, i - h): i + h + 1], axis=0)
            out[i] = m / max(float(m.sum()), 1e-9)      # a per-class median of distributions does not sum to 1
    return out


def sticky_posterior(p, switch):
    """Forward-backward over one clip with a sticky transition matrix: p [n,k] per-frame distributions used as emissions, `switch` the probability of leaving a
    state per 0.2 s step. Labels are built offline, so a two-sided estimate is admissible. Returns the posterior [n,k]."""
    n, k = p.shape
    e = np.clip(p.astype(np.float64), 1e-6, None)
    e /= e.sum(1, keepdims=True)
    T = (1 - switch) * np.eye(k) + (switch / (k - 1)) * (1 - np.eye(k))
    a, b = np.zeros((n, k)), np.ones((n, k))
    a[0] = e[0] / e[0].sum()
    for t in range(1, n):
        a[t] = e[t] * (a[t - 1] @ T)
        a[t] /= a[t].sum()
    for t in range(n - 2, -1, -1):
        b[t] = T @ (e[t + 1] * b[t + 1])
        b[t] /= b[t].sum()
    q = a * b
    return (q / q.sum(1, keepdims=True)).astype(np.float32)


def feats(A):
    st = np.stack([A[f"t1_{k}"] for k in ("luma_mean", "luma_p05", "luma_p95", "sky_luma", "sat_mean", "headlight_frac", "lit_frac")], 1)
    st = st / np.array([255, 255, 255, 255, 1, 1, 1], np.float32)
    return np.concatenate([A["emb"].astype(np.float32), st.astype(np.float32)], 1)


def load_parts(out, clip):
    pa_, pb_ = X.part_path(out, "A", clip), X.part_path(out, "B", clip)
    if not (os.path.exists(pa_) and os.path.exists(pb_)):
        return None
    return dict(np.load(pa_)), dict(np.load(pb_))


def fit_probes(data, train_clips, seed=0):
    """data: clip -> (A, B). Returns {field: (oof dict clip->[101,ncls], full model)} using VLM argmax labels at frames k%5==0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    res = {}
    for fname, _, classes, vk, _z in FIELDS:
        X_, y_, g_ = [], [], []
        for c in train_clips:
            if c in data:
                A, B = data[c]
                F = feats(A)[::5][:len(B[vk])]
                X_.append(F); y_.append(B[vk].argmax(1)); g_ += [c] * len(F)
        if not X_:
            continue
        Xa, ya, ga = np.concatenate(X_), np.concatenate(y_), np.array(g_)
        full = LogisticRegression(max_iter=300, C=0.5).fit(Xa, ya) if len(set(ya)) > 1 else None
        oof = {}
        ng = len(set(ga))
        if full is not None and ng >= 3:
            for tr, te in GroupKFold(min(5, ng)).split(Xa, ya, ga):
                if len(set(ya[tr])) < 2:
                    continue
                m = LogisticRegression(max_iter=300, C=0.5).fit(Xa[tr], ya[tr])
                for c in set(ga[te]):
                    A, _B = data[c]
                    oof[c] = _proba(m, feats(A), len(classes))
        res[fname] = (oof, full)
    return res


def _proba(m, F, ncls):
    p = np.zeros((len(F), ncls), np.float32)
    p[:, m.classes_] = m.predict_proba(F)
    return p


def clip_context(clip, A, B, probes, meta):
    n = len(A["emb"])
    k = np.arange(n)
    vlm = (k % 5 == 0)
    r = {}
    for nm in ("luma_mean", "luma_p05", "luma_p95", "sky_luma", "sat_mean", "headlight_frac", "lit_frac"):
        r[nm] = A[f"t1_{nm}"]
    zi = A["z_illum"]
    r["illum_img_p"] = zi
    r["illum_img"] = np.array(X.ILLUM)[zi.argmax(1)]
    for fname, zname, classes, vk, zk in FIELDS:
        vp = np.zeros((n, len(classes)), np.float32)
        vp[::5][:len(B[vk])] = B[vk][: len(vp[::5])]
        src = np.where(vlm, "vlm", "probe")
        if fname in probes and probes[fname][1] is not None:
            oof, full = probes[fname]
            pp = oof.get(clip)
            pp = pp if pp is not None else _proba(full, feats(A), len(classes))
        else:
            pp = A[zk] if zk else np.full((n, len(classes)), 1.0 / len(classes), np.float32)
            src = np.where(vlm, "vlm", "zeroshot")
        P = np.where(vlm[:, None], vp, pp)
        P = median_smooth(P, vlm)
        src = np.where(vlm, "vlm", np.where(src == "probe", "smoothed", "zeroshot"))
        r[fname + "_p"] = P.astype(np.float32)
        lab = np.array(list(classes))[P.argmax(1)]
        r[fname] = np.where(P.max(1) < 0.4, "unknown", lab)
        r[fname + "_src"] = src
        if zk:
            r["z_" + fname + "_p"] = A[zk]
    r["vlm_frame"] = vlm
    el = sun_el_deg(meta["country"], int(meta["month"]), int(meta["hour_of_day"]))
    sb = sun_band(el)
    r["sun_el_deg_est"] = np.full(n, el, np.float32)
    r["sun_band_est"] = np.full(n, sb)
    r["sun_conf"] = np.full(n, "low")
    d = np.array([abs(IMG_ORD[x] - SUN_ORD[sb]) for x in r["illum_img"]])
    r["illum_agree_sun"] = d <= 1
    r["illum_sun_flag"] = d >= 2
    r["period_of_day"] = np.full(n, period_of_day(int(meta["hour_of_day"])))
    return r


def to_arrow(rowdict, n):
    cols = {}
    for k, v in rowdict.items():
        v = np.asarray(v)
        if v.ndim == 2:
            flat = pa.array(v.astype(np.float32).ravel(), pa.float32())
            cols[k] = pa.FixedSizeListArray.from_arrays(flat, v.shape[1])
        elif v.dtype.kind in "US":
            cols[k] = pa.array(v.astype(str).tolist(), pa.string())
        elif v.dtype == bool:
            cols[k] = pa.array(v.tolist(), pa.bool_())
        else:
            cols[k] = pa.array(v.astype(np.float32).tolist(), pa.float32())
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="F:/strat_out/geom")
    ap.add_argument("--out", default="F:/strat_out")
    ap.add_argument("--splits", default="train4512,eval")
    ap.add_argument("--model-sha", default="F:/strat_out/MODEL_SHA.json")
    a = ap.parse_args()
    import pandas as pd
    tab = pd.read_parquet(os.environ.get("STRAT_OUT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels") + "/splits/clips_table.parquet")
    msha = json.load(open(a.model_sha)) if os.path.exists(a.model_sha) else {}
    shards = sorted(glob.glob(f"{a.geom}/geom_*.parquet"))
    clips = []
    for s in shards:
        clips += pq.read_table(s, columns=["clip_id"]).column(0).unique().to_pylist()
    data = {c: p for c in clips if (p := load_parts(a.out, c)) is not None}
    train_clips = tab[tab.split == "train4512"].clip_id.tolist()
    probes = fit_probes(data, train_clips)
    os.makedirs(a.out + "/frames", exist_ok=True)
    clip_rows = []
    for s in shards:
        g = pq.read_table(s)
        have = [c for c in g.column("clip_id").unique().to_pylist() if c in data]
        if not have:
            continue
        mask = pa.compute.is_in(g.column("clip_id"), pa.array(have))
        g = g.filter(mask)
        parts = []
        for c in have:
            sub = g.filter(pa.compute.equal(g.column("clip_id"), c))
            meta = {"country": sub.column("country")[0].as_py(), "month": sub.column("month")[0].as_py(),
                    "hour_of_day": sub.column("hour_of_day")[0].as_py()}
            A, B = data[c]
            if len(sub) != int(A["n_frames"]):                                # a clip whose video decodes fewer frames than the 0.2 s grid: one row per frame that exists (frame_present)
                sub = sub.filter(sub.column("frame_present"))
                assert len(sub) == int(A["n_frames"]), (c, len(sub), int(A["n_frames"]))
            r = clip_context(c, A, B, probes, meta)
            cols = to_arrow(r, len(sub))
            cols["model_sha_siglip"] = pa.array([msha.get("siglip2", "NA")] * len(sub))
            cols["model_sha_qwen"] = pa.array([msha.get("qwen25vl3b", "NA")] * len(sub))
            cols["prompt_sha"] = pa.array([X.PROMPT_SHA] * len(sub))
            parts.append(pa.table({**{n: sub.column(n) for n in sub.column_names}, **cols}))
            clip_rows.append({"clip_id": c, **{f"{f}_mode": pd.Series(r[f]).mode().iat[0] for f in ("weather", "surface", "visibility", "view_quality", "illum_img")},
                              "period_of_day": r["period_of_day"][0], "sun_band_est": r["sun_band_est"][0],
                              "illum_sun_flag_frac": float(np.mean(r["illum_sun_flag"]))})
        t = pa.concat_tables(parts)
        pq.write_table(t, f"{a.out}/frames/{os.path.basename(s).replace('geom_', 'frames_')}", compression="zstd")
    cl = pd.DataFrame(clip_rows).merge(tab, on="clip_id", how="left")
    cl["label_version"] = C.LABEL_VERSION
    os.makedirs(a.out + "/clips", exist_ok=True)
    cl.to_parquet(a.out + "/clips/clips.parquet", index=False)
    errs = check_frames_schema(a.out + "/frames")
    print(json.dumps({"clips_merged": len(clip_rows), "probe_fields": list(probes), "schema_errors": errs}))
    if errs:
        sys.exit(3)


if __name__ == "__main__":
    import pyarrow.compute  # noqa: F401
    main()
