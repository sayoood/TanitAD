"""Build the PILOT validation page: ONE self-contained local HTML file, 200 frames (100 random, 100 rare by predicted probability), JPEGs ~480 px wide
embedded, no external request. The model labels sit in an encoded block that the page decodes only after the PI pressed reveal. LOCAL ONLY: nothing is
uploaded, no Artifact, no repo copy; the file lives under F:/strat_out/pilot/.

  build_pilot_audit.py [--frames F:/strat_out/pilot/frames] [--out F:/strat_out/pilot/PILOT_AUDIT.html]
Only frames with a Qwen answer (vlm_frame, k = 0, 5, ..., 100) are eligible, so the PI can be compared against both models; at most 1 frame per clip.
"""
import argparse
import base64
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_common as P
import strat_common as C

WEATHER = ("clear", "rain", "snow", "fog")
SURFACE = ("dry", "wet", "snow_ice")
VISIB = ("good", "reduced", "poor")
# (signal, class index, quota among the 100 rare frames); sums to 100
RARE = [("weather", 1, 12), ("weather", 2, 12), ("weather", 3, 12), ("surface", 1, 12), ("surface", 2, 10), ("visibility", 1, 12), ("visibility", 2, 8),
        ("illum", 0, 12), ("illum", 4, 10)]
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def arg(a):
    return int(np.argmax(a))


def make_row(r):
    """hidden record of one frame; r is a pandas row of the merged frame parquet."""
    wq, sq, vq = np.asarray(r["weather_p"]), np.asarray(r["surface_p"]), np.asarray(r["visibility_p"])
    ws, ss = np.asarray(r["z_weather_p"]), np.asarray(r["z_surface_p"])
    ip = np.asarray(r["illum_img_p"])
    rule = P.illum_rule(r["luma_mean"], r["luma_p95"], r["sky_luma"], r["headlight_frac"], r["lit_frac"])
    return {"clip": r["clip_id"], "k": int(r["frame_idx"]), "country": r["country"], "hour": int(r["hour_of_day"]), "sun": r["sun_band_est"],
            "src": {"weather": {"siglip": WEATHER[arg(ws)], "qwen": WEATHER[arg(wq)]}, "surface": {"siglip": SURFACE[arg(ss)], "qwen": SURFACE[arg(sq)]},
                    "visibility": {"qwen": VISIB[arg(vq)]},
                    "illumination": {"siglip": P.ILLUM[arg(ip)], "rule": rule},
                    "tod": {"rule": P.ILLUM_TO_TOD[rule], "sun": P.SUN_TO_TOD[r["sun_band_est"]]}},
            "p": {"weather": wq.tolist(), "surface": sq.tolist(), "visibility": vq.tolist(), "z_weather": ws.tolist(), "z_surface": ss.tolist(), "illum": ip.tolist()}}


def rare_score(h, sig, idx):
    """predicted probability of the rare class: mean of the models that score it."""
    p = h["p"]
    if sig == "weather":
        return (p["weather"][idx] + p["z_weather"][idx]) / 2
    if sig == "surface":
        return (p["surface"][idx] + p["z_surface"][idx]) / 2
    if sig == "visibility":
        return p["visibility"][idx]
    return p["illum"][idx]


def select(df, seed="pilot-audit-v1", n_random=100, n_rare=100):
    """df: merged frames restricted to vlm frames. Returns (items list of dict with hidden + stratum). Deterministic."""
    df = df[df.vlm_frame].copy()
    df["h"] = [C.stable_hash(f"{c}:{k}", seed) for c, k in zip(df.clip_id, df.frame_idx)]
    df = df.sort_values("h")
    used, items = set(), []
    # random: round-robin over country x sun band, one frame per clip
    cells = {k: list(g.index) for k, g in df.groupby(["country", "sun_band_est"])}
    cur = {k: 0 for k in cells}
    while len([x for x in items if x["stratum"] == "random"]) < n_random:
        moved = False
        for key in sorted(cells):
            idxs = cells[key]
            while cur[key] < len(idxs) and df.loc[idxs[cur[key]], "clip_id"] in used:
                cur[key] += 1
            if cur[key] < len(idxs) and len([x for x in items if x["stratum"] == "random"]) < n_random:
                r = df.loc[idxs[cur[key]]]
                used.add(r["clip_id"]); items.append({**make_row(r), "stratum": "random"}); moved = True
        if not moved:
            break
    # rare: top predicted probability per class, one frame per clip, quotas; shortfall filled from the best remaining rare score
    pool = [make_row(r) for _, r in df.iterrows() if r["clip_id"] not in used]
    nrare = lambda: len([x for x in items if x["stratum"] != "random"])
    for sig, idx, q in RARE:
        q = max(1, round(q * n_rare / 100))
        got = 0
        for h in sorted(pool, key=lambda h: -rare_score(h, sig, idx)):
            if got >= q or nrare() >= n_rare:
                break
            if h["clip"] in used:
                continue
            used.add(h["clip"]); items.append({**h, "stratum": f"rare_{sig}_{idx}"}); got += 1
    if nrare() < n_rare:
        allscore = lambda h: max(rare_score(h, s, i) for s, i, _ in RARE)
        for h in sorted(pool, key=lambda h: -allscore(h)):
            if nrare() >= n_rare:
                break
            if h["clip"] not in used:
                used.add(h["clip"]); items.append({**h, "stratum": "rare_fill"})
    rng = np.random.default_rng(0)
    order = rng.permutation(len(items))
    return [items[i] for i in order]


def jpeg(clip_id, video_frame, width=480):
    cmd = [FFMPEG, "-v", "error", "-i", f"{C.CAM_DIR}/{clip_id}.mp4", "-vf", f"select=eq(n\\,{video_frame}),scale={width}:-2:flags=area", "-fps_mode", "passthrough",
           "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "6", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg failed for {clip_id} frame {video_frame}: {r.stderr[-200:]!r}")
    return base64.b64encode(r.stdout).decode()


def build_html(items, imgs, out):
    tpl = open(HERE + "/pilot_audit_template.html", encoding="utf-8").read()
    hidden = base64.b64encode(json.dumps(items, separators=(",", ":")).encode()).decode()
    html = tpl.replace("__HIDDEN__", hidden).replace("__IMGS__", json.dumps(imgs))
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html
    open(out, "w", encoding="utf-8").write(html)
    return len(html)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="F:/strat_out/pilot/frames")
    ap.add_argument("--out", default="F:/strat_out/pilot/PILOT_AUDIT.html")
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-rare", type=int, default=100)
    a = ap.parse_args()
    cols = ["clip_id", "frame_idx", "video_frame", "country", "hour_of_day", "sun_band_est", "vlm_frame", "weather_p", "surface_p", "visibility_p", "z_weather_p",
            "z_surface_p", "illum_img_p", "luma_mean", "luma_p95", "sky_luma", "headlight_frac", "lit_frac"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(a.frames + "/frames_*.parquet"))], ignore_index=True)
    items = select(df, n_random=a.n_random, n_rare=a.n_rare)
    vf = {(r.clip_id, int(r.frame_idx)): int(r.video_frame) for r in df[["clip_id", "frame_idx", "video_frame"]].itertuples()}
    imgs = [jpeg(h["clip"], vf[(h["clip"], h["k"])]) for h in items]
    n = build_html(items, imgs, a.out)
    json.dump([{"order": i, "clip_id": h["clip"], "frame_idx": h["k"], "stratum": h["stratum"], "country": h["country"]} for i, h in enumerate(items)],
              open(os.path.dirname(a.out) + "/PILOT_AUDIT_FRAMES.json", "w"))
    print(json.dumps({"frames": len(items), "random": sum(h["stratum"] == "random" for h in items), "html_bytes": n}))
    assert n < 40 * 2 ** 20, "page over 40 MB"


if __name__ == "__main__":
    main()
