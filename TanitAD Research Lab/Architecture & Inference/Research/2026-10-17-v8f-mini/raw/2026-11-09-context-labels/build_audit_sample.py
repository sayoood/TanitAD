"""PI audit sample: 600 frames, blind-then-reveal. Builds the frame LIST (parquet + json) from the merged frame labels, and optionally renders
640 px JPEGs of the 600 frames to a private folder (never into the repo). No page is built or published here.

  build_audit_sample.py --frames F:/strat_out/frames --out E:/.../strat_labels/audit [--render F:/strat_out/audit_frames]

Composition (AUDIT_DATA 4.1): 300 stratified random frames (25 countries x sun-band, >= 6 per country where the country has clips, at most 1 frame per
clip, at most 1 clip per chunk) + 300 targeted rare-class candidates (model score for rain, snow, fog, wet, snow_ice, reduced / poor visibility, glare,
wiper, tunnel; rare-class precision only, recall unmeasured). The model labels are stored in `hidden_*` columns and revealed only after the PI answers.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C

RARE = [("weather", "rain"), ("weather", "snow"), ("weather", "fog"), ("surface", "wet"), ("surface", "snow_ice"), ("visibility", "reduced"),
        ("visibility", "poor"), ("view_quality", "glare"), ("view_quality", "wiper"), ("view_quality", "tunnel"), ("illum_img", "glare"), ("illum_img", "dark")]
FIELDS_UI = {"weather": ["clear", "rain", "snow", "fog", "unsure"], "surface": ["dry", "wet", "snow_ice", "unsure"], "visibility": ["good", "reduced", "poor", "unsure"],
             "illumination": ["dark", "lit_night", "twilight", "day", "glare", "unsure"], "view_quality": ["clean", "wiper", "glare", "occluded", "tunnel", "unsure"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="F:/strat_out/frames")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)) + "/audit")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--n-rare", type=int, default=300)
    ap.add_argument("--render", default="")
    a = ap.parse_args()
    cols = ["clip_id", "frame_idx", "video_frame", "country", "chunk", "split", "sun_band_est", "weather", "surface", "visibility", "view_quality", "illum_img",
            "weather_p", "surface_p", "visibility_p", "view_quality_p", "illum_img_p", "vlm_frame", "weather_src"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(a.frames + "/frames_*.parquet"))], ignore_index=True)
    df = df[df.split == "train4512"] if (df.split == "train4512").any() else df
    df["h"] = [C.stable_hash(f"{c}:{k}", "audit-v1") for c, k in zip(df.clip_id, df.frame_idx)]
    rng = np.random.default_rng(0)
    used_clips, used_chunks, rows = set(), set(), []

    def take(sub, n, stratum):
        got = 0
        for _, r in sub.sort_values("h").iterrows():
            if got >= n:
                break
            if r.clip_id in used_clips or r.chunk in used_chunks:
                continue
            used_clips.add(r.clip_id); used_chunks.add(r.chunk)
            rows.append((r, stratum)); got += 1
        return got

    # random half: country x sun band, round robin until n_random
    countries = sorted(df.country.unique())
    bands = ["night", "twilight", "low_sun", "day"]
    per_cell = 0
    cells = [(c, b) for c in countries for b in bands]
    for rnd in range(10):
        for c, b in cells:
            if len([1 for _, s in rows if s.startswith("random")]) >= a.n_random:
                break
            take(df[(df.country == c) & (df.sun_band_est == b)], 1, f"random:{c}:{b}")
    # rare half: top score per class, balanced across classes
    per_class = int(np.ceil(a.n_rare / len(RARE)))
    classes = {"weather": ["clear", "rain", "snow", "fog"], "surface": ["dry", "wet", "snow_ice"], "visibility": ["good", "reduced", "poor"],
               "view_quality": ["clean", "wiper", "glare", "occluded", "tunnel"], "illum_img": ["dark", "lit_night", "twilight", "day", "glare"]}
    for fld, cls in RARE:
        P = np.stack(df[fld + "_p"].to_numpy())
        sc = pd.Series(P[:, classes[fld].index(cls)], index=df.index)
        sub = df.assign(score=sc).sort_values("score", ascending=False)
        sub = sub[sub.score > 0.3]
        take(sub.assign(h=-sub.score), per_class, f"rare:{fld}={cls}")
    rare_n = len([1 for _, s in rows if s.startswith("rare")])
    out = []
    for i, (r, s) in enumerate(rows):
        out.append({"audit_id": i, "clip_id": r.clip_id, "frame_idx": int(r.frame_idx), "video_frame": int(r.video_frame), "stratum": s, "country": r.country,
                    "sun_band_est": r.sun_band_est, **{f"hidden_{k}": r[k] for k in ("weather", "surface", "visibility", "view_quality", "illum_img")},
                    "hidden_vlm_frame": bool(r.vlm_frame)})
    t = pd.DataFrame(out)
    t = t.sample(frac=1.0, random_state=1).reset_index(drop=True)       # shuffled order shown to the PI
    t["order"] = np.arange(len(t))
    os.makedirs(a.out, exist_ok=True)
    t.to_parquet(a.out + "/AUDIT_SAMPLE.parquet", index=False)
    json.dump({"fields": FIELDS_UI, "n": len(t), "n_random": int(len(t) - rare_n), "n_rare": rare_n,
               "session_size": 100, "ui": "blind-then-reveal: image only; one tap per field; skip; reveal shows the model label and p; result table = per-field "
               "accuracy with Wilson 95% interval and confusion matrix; rare classes report precision only",
               "frames": t[["order", "audit_id", "clip_id", "frame_idx", "video_frame", "stratum"]].to_dict("records")},
              open(a.out + "/AUDIT_SAMPLE_FRAMES.json", "w"))
    print(json.dumps({"n": len(t), "random": int(len(t) - rare_n), "rare": rare_n, "strata_top": t.stratum.str.split(":").str[0].value_counts().to_dict()}))
    if a.render:
        os.makedirs(a.render, exist_ok=True)
        for r in t.itertuples():
            dst = f"{a.render}/{r.order:04d}.jpg"
            if os.path.exists(dst):
                continue
            subprocess.run(["ffmpeg", "-v", "error", "-i", f"{C.CAM_DIR}/{r.clip_id}.mp4", "-vf", f"select=eq(n\\,{r.video_frame}),scale=640:-2", "-frames:v", "1",
                            "-q:v", "3", dst], stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
