"""LABEL VALIDATION SAMPLE — key frame + past + future + labels + INDEPENDENT geometry.

Purpose (PI, 2026-08-23): see whether the extracted strategic labels are RIGHT,
on a small sample, before scaling the label build to the full corpus.

The judgement must not be an impression. For every sampled clip this computes
GEOMETRIC EVIDENCE from the ego poses alone — net heading change, speed
profile, stop detection, lateral excursion — and the verdict compares the label
against THAT, not against how the picture feels.

⚠️ NON-PARITY / PARTIAL-JOIN CAVEATS, stated up front because they bound every
conclusion drawn from this sample:
  * frames+poses come from the LOCAL cache `physicalai-train-14231cd29c74`
    (400 eps), NOT the parity corpus `physicalai-train-e438721ae894`/f09e44db.
  * the join runs through `episode_id_legacy`, which the clip index itself warns
    COLLIDES (69/2400 train + 7/600 val clips share an id). Ambiguous keys are
    REFUSED, never resolved by picking one.
  * the sample is whatever happens to be BOTH labelled and locally cached — it
    is a convenience sample, not a random one, so class frequencies here say
    nothing about the corpus.
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

EPCACHE = glob.glob("C:/Users/Admin/tanitad-data/physicalai/_epcache/*/ep_*.pt")
LBL = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_v3")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
OUT.mkdir(parents=True, exist_ok=True)

HZ = 10.0
N_STACK = 3
T0_RAW_IDX = 80          # t0_s = 8.0 s on the RAW clip timeline @10 Hz
KEY = T0_RAW_IDX - (N_STACK - 1)      # provider index the epcache is on
PAST = KEY - 20          # -2.0 s
FUT = KEY + 20           # +2.0 s
HORIZON = 40             # 4 s of future used for the geometry verdict


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def png_b64(chw_u8, scale=1):
    from PIL import Image
    arr = chw_u8.permute(1, 2, 0).numpy()          # HWC RGB
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ci = json.loads((LBL / "clip_index.json").read_text(encoding="utf-8"))
    clips = ci["clips"]
    leg = collections.defaultdict(list)
    for u, v in clips.items():
        leg[v["episode_id_legacy"]].append(u)
    ambiguous = {k for k, v in leg.items() if len(v) > 1}

    rows = {}
    for sp in ("aug120", "w120val"):
        for line in (LBL / f"s2_labels_{sp}.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["clip_id"]] = r

    # ⛔ THE COLLISION HAS TWO SIDES AND I ONLY CHECKED ONE THE FIRST TIME.
    # `ambiguous` above catches a legacy id claimed by >1 LABELLED CLIP. But a
    # legacy id can equally be claimed by >1 EPISODE IN THE CACHE — and then a
    # single uuid silently receives two different trajectories. MEASURED: clip
    # 4879e5f3 came out TWICE, once with net_yaw_4s +70.3 deg and once with
    # +1.4 deg, i.e. two different scenes wearing one label. Both sides are
    # refused now; a duplicate is never resolved by taking the first.
    ep_by_leg = collections.defaultdict(list)
    for p in sorted(EPCACHE):
        d = torch.load(p, map_location="cpu", weights_only=False)
        ep_by_leg[int(d["episode_id"])].append(p)
    cache_ambiguous = {k for k, v in ep_by_leg.items() if len(v) > 1}
    refused = {"label_side": sorted(ambiguous),
               "cache_side": sorted(cache_ambiguous)}
    print(f"[val] refused legacy ids — label-side {len(ambiguous)}, "
          f"cache-side {len(cache_ambiguous)}")

    out = []
    for p in sorted(EPCACHE):
        d = torch.load(p, map_location="cpu", weights_only=False)
        eid = int(d["episode_id"])
        if eid in ambiguous or eid in cache_ambiguous or eid not in leg:
            continue
        uid = leg[eid][0]
        if uid not in rows:
            continue
        poses = np.asarray(d["poses"], dtype=np.float64)
        T = poses.shape[0]
        if not (0 <= PAST and FUT < T and KEY + HORIZON < T):
            continue
        r = rows[uid]

        x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        k = KEY
        # ---- ego-frame trajectory at the key index (x fwd, y left) ---------
        c, s = np.cos(-yaw[k]), np.sin(-yaw[k])
        dx, dy = x - x[k], y - y[k]
        ex, ey = c * dx - s * dy, s * dx + c * dy

        # ---- INDEPENDENT geometric evidence --------------------------------
        net_yaw_4s = float(np.degrees(wrap(yaw[k + HORIZON] - yaw[k])))
        net_yaw_end = float(np.degrees(wrap(yaw[-1] - yaw[k])))
        v_key = float(v[k])
        fut_v = v[k:]
        v_min = float(fut_v.min())
        v_end = float(v[-1])
        stops = bool(v_min < 0.5)
        # signed lateral excursion of the future path, in the key ego frame
        lat_4s = float(ey[k + HORIZON])
        path_len_4s = float(np.sum(np.hypot(np.diff(x[k:k + HORIZON + 1]),
                                            np.diff(y[k:k + HORIZON + 1]))))

        # ⚠️ A STRATEGIC label must be judged on the STRATEGIC horizon.
        # HIERARCHY_VOCABULARY.md §3 puts g_str at 8-30 s; judging it on the
        # 4 s tactical window would score the label against a question it was
        # never asked, and would manufacture "mismatches" out of manoeuvres
        # that simply have not started yet. Both horizons are reported, and
        # the verdict below uses the strategic one.
        yaw_from_key = np.degrees(wrap(yaw[k:] - yaw[k]))
        i_peak = int(np.argmax(np.abs(yaw_from_key)))
        peak_yaw = float(yaw_from_key[i_peak])
        # when does the manoeuvre actually happen, relative to the anchor?
        big = np.where(np.abs(yaw_from_key) >= 25.0)[0]
        t_onset = float(big[0] / HZ) if big.size else None

        geom = {
            "net_yaw_4s_deg": round(net_yaw_4s, 1),
            "net_yaw_to_end_deg": round(net_yaw_end, 1),
            "peak_yaw_after_key_deg": round(peak_yaw, 1),
            "t_yaw_onset_25deg_s": (round(t_onset, 1)
                                    if t_onset is not None else None),
            "horizon_available_s": round((T - 1 - k) / HZ, 1),
            "v_at_key_mps": round(v_key, 2),
            "v_min_future_mps": round(v_min, 2),
            "v_end_mps": round(v_end, 2),
            "dv_key_to_end_mps": round(v_end - v_key, 2),
            "comes_to_a_stop": stops,
            "lat_excursion_4s_m": round(lat_4s, 2),
            "path_len_4s_m": round(path_len_4s, 1),
        }

        g_tok = r["g_str"]["token"]
        a_blk = r["a_str"]
        a_tok = None if a_blk.get("abstain") else a_blk.get("token")

        out.append({
            "clip_id": uid,
            "episode_file": Path(p).name,
            "episode_id_legacy": eid,
            "label_split": clips[uid]["label_split"],
            "t0_s": r["t0_s"],
            "key_provider_idx": k,
            "g_str": {"token": g_tok,
                      "args": r["g_str"].get("args"),
                      "arg_mask": r["g_str"].get("arg_mask"),
                      "confidence": r["g_str"].get("confidence"),
                      "sources": r["g_str"].get("sources"),
                      "corroboration": r["g_str"].get("corroboration")},
            "a_str": {"token": a_tok,
                      "abstain": bool(a_blk.get("abstain")),
                      "reason": a_blk.get("reason"),
                      "superseded_token_v2": a_blk.get("superseded_token_v2"),
                      "args": a_blk.get("args"),
                      "confidence": a_blk.get("confidence"),
                      "corroboration": a_blk.get("corroboration")},
            "geometry": geom,
            "traj_ego": {
                "past_x": [round(float(q), 2) for q in ex[max(0, k - 60):k + 1]],
                "past_y": [round(float(q), 2) for q in ey[max(0, k - 60):k + 1]],
                "fut_x": [round(float(q), 2) for q in ex[k:k + HORIZON + 21]],
                "fut_y": [round(float(q), 2) for q in ey[k:k + HORIZON + 21]],
                "fut_v": [round(float(q), 2) for q in v[k:k + HORIZON + 21]],
            },
            "frames": {
                "past_t_minus_2s": png_b64(d["frames_u8"][PAST][6:9]),
                "key_t0": png_b64(d["frames_u8"][KEY][6:9]),
                "future_t_plus_2s": png_b64(d["frames_u8"][FUT][6:9]),
            },
        })

    (OUT / "sample.json").write_text(json.dumps(out), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "frames"} for r in out]
    (OUT / "sample_slim.json").write_text(json.dumps(slim, indent=1),
                                          encoding="utf-8")
    print(f"[val] {len(out)} clips extracted -> {OUT/'sample.json'}")
    ids = [r["clip_id"] for r in out]
    assert len(ids) == len(set(ids)), \
        f"DUPLICATE clip_id in the sample: {collections.Counter(ids).most_common(3)}"
    print(f"[val] uniqueness check PASSED — {len(ids)} distinct clip_ids")
    (OUT / "refused_legacy_ids.json").write_text(
        json.dumps(refused, indent=1), encoding="utf-8")

    for r in out:
        g = r["geometry"]
        print(f"  {r['clip_id'][:8]} {r['label_split']:8s} "
              f"{r['g_str']['token']:17s} / "
              f"{(r['a_str']['token'] or 'ABSTAIN'):15s} | "
              f"yaw4s {g['net_yaw_4s_deg']:+7.1f}  "
              f"peak {g['peak_yaw_after_key_deg']:+7.1f}  "
              f"onset {str(g['t_yaw_onset_25deg_s']):5s}s  "
              f"horiz {g['horizon_available_s']:4.1f}s  "
              f"v {g['v_at_key_mps']:5.2f}->{g['v_end_mps']:5.2f} "
              f"min {g['v_min_future_mps']:5.2f} "
              f"stop={str(g['comes_to_a_stop']):5s}")


if __name__ == "__main__":
    main()
