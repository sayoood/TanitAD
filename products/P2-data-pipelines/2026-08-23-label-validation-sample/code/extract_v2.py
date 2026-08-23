"""Validation sample v2 — adds TACTICAL labels, ALPAMAYO outputs, and the FULL
strategic horizon the PI defined (key frame + 8 s .. 30 s).

Three changes over v1, each from PI feedback 2026-08-23:
 1. FUTURE OUT TO THE STRATEGIC BAND, not +-2 s. Frames at t-2, t0, +2, +4, +8
    and clip-end, so the band a strategic label actually describes is visible.
 2. TACTICAL labels (`refc_tactical.window_factored_labels` v1 AND the
    curvature-gated v2) — the factored LAT x LON pair, not just strategic.
 3. ALPAMAYO-Super per-clip outputs (longitudinal / lateral / lane / CoT) joined
    in, so the VLM's own read sits beside ours.

⛔ AND THE FINDING THAT CAME OUT OF (1), stated here because it bounds every
strategic number this programme will ever quote off this corpus:
    clip length 19.9 s, anchor 7.8 s => 12.0 s of future exists.
    The PI's strategic band is [t0+8, t0+30] = [15.8, 37.8] s of clip time.
    The clip ENDS at 19.9 s. => 4.1 s of a 22 s band is observable = 18.6 %,
    and 17.9 s of it lies past the end of the recording.
A strategic label on this corpus is therefore an extrapolation for 81 % of its
own definition. That is a CORPUS limitation, not a labeller defect.
"""
from __future__ import annotations

import base64
import collections
import glob
import io
import json
from pathlib import Path

import numpy as np
import torch

STACK = Path("C:/Users/Admin/tanitad-wt/stack")
import sys
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))
from tanitad.refs import refc_tactical as tac                    # noqa: E402

HUB = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/.claude/worktrees/"
           "interesting-tharp-463cf3/TanitAD Research Hub/Data Engineering/"
           "Implementation/incoming")
ALPA = HUB / "2026-08-16-tactical-labels/raw/a1_alpamayo_taxonomy_per_clip.jsonl"
LBL = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_v3")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
EPCACHE = glob.glob("C:/Users/Admin/tanitad-data/physicalai/_epcache/*/ep_*.pt")

HZ, N_STACK, T0_RAW = 10.0, 3, 80
KEY = T0_RAW - (N_STACK - 1)
STRAT_LO_S, STRAT_HI_S = 8.0, 30.0          # PI definition, relative to key
FRAME_OFFSETS_S = [-2.0, 0.0, 2.0, 4.0, 8.0, 12.0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def png_b64(chw):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(chw.permute(1, 2, 0).numpy()).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ci = json.loads((LBL / "clip_index.json").read_text(encoding="utf-8"))
    clips = ci["clips"]
    leg = collections.defaultdict(list)
    for u, v in clips.items():
        leg[v["episode_id_legacy"]].append(u)
    amb_label = {k for k, v in leg.items() if len(v) > 1}

    alpa = {}
    for line in ALPA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            alpa[r["clip_id"]] = r

    rows = {}
    for sp in ("aug120", "w120val"):
        for line in (LBL / f"s2_labels_{sp}.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["clip_id"]] = r

    ep_by_leg = collections.defaultdict(list)
    cache = {}
    for p in sorted(EPCACHE):
        d = torch.load(p, map_location="cpu", weights_only=False)
        ep_by_leg[int(d["episode_id"])].append(p)
        cache[p] = d
    amb_cache = {k for k, v in ep_by_leg.items() if len(v) > 1}

    out = []
    for p in sorted(EPCACHE):
        d = cache[p]
        eid = int(d["episode_id"])
        if eid in amb_label or eid in amb_cache or eid not in leg:
            continue
        uid = leg[eid][0]
        if uid not in rows:
            continue
        poses = np.asarray(d["poses"], dtype=np.float64)
        T = poses.shape[0]
        k = KEY
        if k + 20 >= T:
            continue
        r = rows[uid]
        x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]

        c, s = np.cos(-yaw[k]), np.sin(-yaw[k])
        dx, dy = x - x[k], y - y[k]
        ex, ey = c * dx - s * dy, s * dx + c * dy

        # ---- TACTICAL labels (the factored LAT x LON pair) -----------------
        pl = torch.tensor(poses[k:k + 1], dtype=torch.float32)
        fp = torch.tensor(poses[k + 1:k + 1 + tac.LABEL_HORIZON][None],
                          dtype=torch.float32)
        lat1, lon1 = tac.window_factored_labels(pl, fp)
        lat2, lon2 = tac.window_factored_labels_v2(pl, fp)
        tactical = {
            "horizon_s": tac.LABEL_HORIZON / HZ,
            "v1": {"lat": tac.LAT_CLASSES[int(lat1)],
                   "lon": tac.LON_CLASSES[int(lon1)]},
            "v2": {"lat": tac.LAT_CLASSES[int(lat2)],
                   "lon": tac.LON_CLASSES[int(lon2)]},
            "v1_v2_agree": (int(lat1) == int(lat2) and int(lon1) == int(lon2)),
        }

        # ---- strategic-band coverage ---------------------------------------
        avail_s = (T - 1 - k) / HZ
        band = STRAT_HI_S - STRAT_LO_S
        obs = float(np.clip(avail_s - STRAT_LO_S, 0.0, band))
        horizon = {
            "clip_len_s": round((T - 1) / HZ, 1),
            "anchor_s": round(k / HZ, 1),
            "future_available_s": round(avail_s, 1),
            "strategic_band_s": [STRAT_LO_S, STRAT_HI_S],
            "band_observable_s": round(obs, 1),
            "band_observable_frac": round(obs / band, 3),
            "band_missing_s": round(band - obs, 1),
        }

        yaw_from_key = np.degrees(wrap(yaw[k:] - yaw[k]))
        i_peak = int(np.argmax(np.abs(yaw_from_key)))
        big = np.where(np.abs(yaw_from_key) >= 25.0)[0]
        onset = float(big[0] / HZ) if big.size else None
        # what happens INSIDE the observable part of the strategic band
        lo_i = int(STRAT_LO_S * HZ)
        band_yaw = (yaw_from_key[lo_i:] if lo_i < len(yaw_from_key)
                    else np.array([]))
        geom = {
            "net_yaw_4s_deg": round(float(np.degrees(
                wrap(yaw[k + 40] - yaw[k]))) if k + 40 < T else float("nan"), 1),
            "peak_yaw_after_key_deg": round(float(yaw_from_key[i_peak]), 1),
            "t_yaw_onset_25deg_s": round(onset, 1) if onset is not None else None,
            "yaw_in_strategic_band_deg": (
                round(float(band_yaw[np.argmax(np.abs(band_yaw))]), 1)
                if band_yaw.size else None),
            "v_at_key_mps": round(float(v[k]), 2),
            "v_min_future_mps": round(float(v[k:].min()), 2),
            "v_end_mps": round(float(v[-1]), 2),
            "dv_key_to_end_mps": round(float(v[-1] - v[k]), 2),
            "comes_to_a_stop": bool(v[k:].min() < 0.5),
        }

        frames, fmeta = {}, []
        for off in FRAME_OFFSETS_S:
            idx = int(round(k + off * HZ))
            idx = max(0, min(T - 1, idx))
            key = f"t{off:+.0f}".replace("+0", "0")
            frames[key] = png_b64(d["frames_u8"][idx][6:9])
            fmeta.append({"key": key, "offset_s": off,
                          "clipped": idx != int(round(k + off * HZ)),
                          "in_strategic_band": off >= STRAT_LO_S})

        a = alpa.get(uid)
        out.append({
            "clip_id": uid, "label_split": clips[uid]["label_split"],
            "t0_s": r["t0_s"],
            "g_str": {"token": r["g_str"]["token"],
                      "confidence": r["g_str"].get("confidence"),
                      "sources": r["g_str"].get("sources")},
            "a_str": {"token": (None if r["a_str"].get("abstain")
                                else r["a_str"].get("token")),
                      "abstain": bool(r["a_str"].get("abstain")),
                      "confidence": r["a_str"].get("confidence")},
            "tactical": tactical,
            "alpamayo": ({"longitudinal": a["longitudinal"],
                          "lateral": a["lateral"], "lane": a["lane"],
                          "cot": a["cot"]} if a else None),
            "horizon": horizon, "geometry": geom,
            "frame_meta": fmeta, "frames": frames,
            "traj_ego": {
                "past_x": [round(float(q), 2) for q in ex[max(0, k - 60):k + 1]],
                "past_y": [round(float(q), 2) for q in ey[max(0, k - 60):k + 1]],
                "fut_x": [round(float(q), 2) for q in ex[k:]],
                "fut_y": [round(float(q), 2) for q in ey[k:]],
                "band_start_i": int(STRAT_LO_S * HZ),
            },
        })

    (OUT / "sample_v2.json").write_text(json.dumps(out), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "frames"} for r in out]
    (OUT / "sample_v2_slim.json").write_text(json.dumps(slim, indent=1),
                                             encoding="utf-8")
    n = len(out)
    na = sum(1 for r in out if r["alpamayo"])
    ag = sum(1 for r in out if r["tactical"]["v1_v2_agree"])
    print(f"[v2] {n} clips | alpamayo joined {na}/{n} | "
          f"tactical v1==v2 on {ag}/{n}")
    print(f"[v2] strategic band observable: "
          f"{out[0]['horizon']['band_observable_frac']:.1%} "
          f"({out[0]['horizon']['band_observable_s']}s of 22s)")
    lat = collections.Counter(r["tactical"]["v2"]["lat"] for r in out)
    lon = collections.Counter(r["tactical"]["v2"]["lon"] for r in out)
    print(f"[v2] tactical v2 LAT {dict(lat)}")
    print(f"[v2] tactical v2 LON {dict(lon)}")


if __name__ == "__main__":
    main()
