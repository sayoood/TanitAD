"""Situation classifier — STEP 1 (dev box, CPU): build the THREE SITUATION LABELS.

Runs the frozen detectors of `sc_situations.py` over every cached episode's `poses`, produces the
per-frame anticipation targets, and writes:

  * `sc_labels.npz`   — the DE-IDENTIFIED bundle that goes to the pod (integer clip index only)
  * `universe.json`   — per-chunk / per-side counts, per situation, BEFORE anything is trained
  * `join_proof.json` — the episode -> clip join, PROVEN by replaying the cache's own recipe
  * `round_sweep.json`— the DEV roundabout constant sweep behind PRE_REGISTRATION Sec 2.2
  * `_LOCAL_ONLY_k2clip.json` — the UUID map, dev box only, NEVER staged (gated corpus)

usage:  python sc_build_labels.py <poses_dir> <pod2_r0_selection.parquet> <out_dir>
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_situations as S  # noqa: E402

SITS = ("lane_change", "roundabout", "intersection")
LHT = {"United Kingdom", "Ireland", "Malta", "Cyprus"}   # left-hand traffic


def rebuild_join(meta, sel_path):
    """Reproduce the cache's own build order and PROVE it against the stored episode ids."""
    sel = pd.read_parquet(sel_path)
    sel["chunk"] = sel["chunk"].astype(str).str.zfill(4)
    clips = sorted(sel["clip_id"].astype(str))
    g = torch.Generator().manual_seed(0)                  # cfg.train.seed == 0
    perm = torch.randperm(len(clips), generator=g).tolist()
    val_i = set(perm[:max(1, int(len(clips) * 0.2))])
    tr = [c for i, c in enumerate(clips) if i not in val_i]
    va = [c for i, c in enumerate(clips) if i in val_i]
    c2chunk = dict(zip(sel["clip_id"].astype(str), sel["chunk"]))
    c2country = dict(zip(sel["clip_id"].astype(str), sel["country"].astype(str)))
    ok = {"train": 0, "val": 0}
    for m in meta:
        i = int(m["file"].split("_")[1].split(".")[0])
        c = (tr if m["cache"] == "train" else va)[i]
        m["clip_id"] = c
        m["chunk"] = c2chunk[c]
        m["country"] = c2country[c]
        ok[m["cache"]] += int(m["episode_id"] == int.from_bytes(c.encode()[:4].ljust(4, b"\0"), "big"))
    n = {k: sum(1 for m in meta if m["cache"] == k) for k in ("train", "val")}
    proof = {
        "recipe": "sorted(clip_id) -> split_clips(val_frac=0.2, seed=0); ep_NNNNN.pt index == list index",
        "episode_id_rule": "int.from_bytes(clip_id.encode()[:4], 'big')  (physicalai.py:496)",
        "n_selection": len(clips), "n_episodes": len(meta),
        "train_files": n["train"], "val_files": n["val"],
        "train_ids_reproduced": ok["train"], "val_ids_reproduced": ok["val"],
        "join_proven": bool(ok["train"] == n["train"] and ok["val"] == n["val"]),
        "left_hand_traffic_clips_in_selection": int(sel["country"].astype(str).isin(LHT).sum()),
        "countries": sorted(sel["country"].astype(str).unique()),
    }
    if not proof["join_proven"]:
        raise SystemExit(f"JOIN REFUSED: {ok} vs {n}")
    return proof


def round_sweep(meta, Z, train_ch):
    """The DEV sweep behind PRE_REGISTRATION Sec 2.2 — published so the choice is auditable."""
    rows = []
    nb = int(round(S.RB_BRACKET_S * S.HZ))
    for m in meta:
        if m["chunk"] not in train_ch:
            continue
        K = S.kinematics(Z[f"p{m['i']}"])
        psi, v, kap = K["psi"], K["v"], K["kappa"]
        for a, b, s in S.curvature_runs(K):
            seg = kap[a:b + 1]
            mm = float(np.abs(seg).mean())
            pre = -s * np.degrees(psi[a] - psi[max(0, a - nb)])
            post = -s * np.degrees(psi[min(len(psi) - 1, b + nb)] - psi[b])
            rows.append((s, (b - a + 1) / S.HZ, abs(np.degrees(psi[b] - psi[a])),
                         seg.std() / mm if mm > 0 else 9.9, float(v[a:b + 1].mean()),
                         max(pre, post)))
    R = np.array(rows, dtype=float) if rows else np.zeros((0, 6))
    out = []
    for dur, dpsi, cv, br in itertools.product((3.0, 3.5, 4.0, 4.5), (80, 90, 100, 110),
                                               (0.4, 0.5, 0.6), (0, 3, 5, 8, 10)):
        m_ = ((R[:, 1] >= dur) & (R[:, 2] >= dpsi) & (R[:, 3] <= cv)
              & (R[:, 4] >= 2) & (R[:, 4] <= 14) & (R[:, 5] >= br))
        if not m_.any():
            continue
        out.append(dict(dur_min_s=dur, dpsi_min_deg=dpsi, cv_max=cv, bracket_deg=br,
                        n_events=int(m_.sum()), ccw_frac=round(float((R[m_, 0] > 0).mean()), 4)))
    sel = [o for o in out if o["ccw_frac"] >= 0.90]
    sel.sort(key=lambda o: (-o["n_events"], -o["ccw_frac"]))
    return {"rule": "maximise n_events subject to DEV ccw_frac >= 0.90 "
                    "(the corpus has ZERO left-hand-traffic clips)",
            "n_curvature_runs_dev": int(len(R)), "selected": sel[0] if sel else None,
            "grid": out}


def main():
    poses_dir, sel_path, out = sys.argv[1:4]
    # ⚠️ The study universe is restricted to the caches that actually carry decoded FRAMES on the
    # assigned host. pod3 holds `physicalai-train-e438721ae894` (2,376 episodes) but NOT the val
    # cache, so the classifier universe is the train-cache side. Declared, not silently applied.
    only_cache = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True)
    Z = np.load(os.path.join(poses_dir, "poses.npz"))
    meta = json.load(open(os.path.join(poses_dir, "poses_meta.json")))
    proof = rebuild_join(meta, sel_path)
    if only_cache:
        proof["universe_restricted_to_cache"] = only_cache
        proof["restriction_reason"] = ("pod3 (assigned host) holds only the train parity cache; "
                                       "the val cache has no decoded frames there")
        meta = [m for m in meta if m["cache"] == only_cache]
    json.dump(proof, open(os.path.join(out, "join_proof.json"), "w"), indent=2)
    print(f"[join] PROVEN {proof['train_ids_reproduced']}/{proof['train_files']} train, "
          f"{proof['val_ids_reproduced']}/{proof['val_files']} val; "
          f"{proof['left_hand_traffic_clips_in_selection']} left-hand-traffic clips")

    CH = sorted({m["chunk"] for m in meta})
    TRAIN_CH = {c for j, c in enumerate(CH) if j % 3 == 0}
    for m in meta:
        m["side"] = "TRAIN" if m["chunk"] in TRAIN_CH else "HELDOUT"

    sw = round_sweep(meta, Z, TRAIN_CH)
    json.dump(sw, open(os.path.join(out, "round_sweep.json"), "w"), indent=2)
    print(f"[round] DEV sweep selected {sw['selected']}")

    packs, rec, k2clip = {}, [], {}
    for k, m in enumerate(meta):
        P = Z[f"p{m['i']}"]
        K = S.kinematics(P)
        T = K["T"]
        ev = {"lane_change": S.detect_lane_change(K),
              "roundabout": S.detect_roundabout(K, bracket=True)}
        inter, turns, _x = S.detect_intersection(K, cross=None)
        ev["intersection"] = inter
        rb_core = S.detect_roundabout(K, bracket=False)

        row = dict(k=k, chunk=m["chunk"], side=m["side"], cache=m["cache"], T=int(T),
                   country=m["country"], encoder_seen=bool(m["cache"] == "train"),
                   v_mean=round(float(K["v"].mean()), 3))
        for s in SITS:
            y, valid = S.anticipation_target(T, ev[s])
            packs[f"c{k}_y_{s}"] = y
            packs[f"c{k}_valid_{s}"] = valid
            packs[f"c{k}_ongoing_{s}"] = _mask(T, ev[s])
            row[f"n_ev_{s}"] = len(ev[s])
            row[f"n_pos_{s}"] = int((y & valid).sum())
            row[f"n_valid_{s}"] = int(valid.sum())
        packs[f"c{k}_onset_lane_change"] = S.event_onsets(ev["lane_change"])
        packs[f"c{k}_onset_roundabout"] = S.event_onsets(ev["roundabout"])
        packs[f"c{k}_onset_intersection"] = S.event_onsets(ev["intersection"])
        packs[f"c{k}_turn_ab"] = np.array(turns, dtype=np.int32).reshape(-1, 2)
        packs[f"c{k}_curve_ab"] = np.array(S.detect_curves(K), dtype=np.int32).reshape(-1, 2)
        packs[f"c{k}_rbcore_ab"] = np.array(rb_core, dtype=np.int32).reshape(-1, 2)
        packs[f"c{k}_rb_ab"] = np.array(ev["roundabout"], dtype=np.int32).reshape(-1, 2)
        packs[f"c{k}_lc_ab"] = np.array(ev["lane_change"], dtype=np.int32).reshape(-1, 2)
        # ego channels the head may receive (STRICTLY CAUSAL) + the privileged summary (C-POS only)
        packs[f"c{k}_ego"] = np.stack([K["v"], K["alon_pre"], K["omega_pre"]], 1).astype(np.float32)
        packs[f"c{k}_priv"] = _privileged(K).astype(np.float32)
        packs[f"c{k}_kappa"] = K["kappa"].astype(np.float32)
        row["rb_ccw"] = int(np.sign(K["kappa"][ev["roundabout"][0][0]:ev["roundabout"][0][1] + 1]
                                    .mean())) if ev["roundabout"] else 0
        row["rbcore_ccw"] = int(np.sign(K["kappa"][rb_core[0][0]:rb_core[0][1] + 1].mean())) if rb_core else 0
        row["n_ev_rbcore"] = len(rb_core)
        row["n_ev_turn"] = len(turns)
        rec.append(row)
        k2clip[str(k)] = m["clip_id"]

    D = pd.DataFrame(rec)
    np.savez_compressed(os.path.join(out, "sc_labels.npz"), **packs)
    D.to_parquet(os.path.join(out, "sc_index.parquet"))
    json.dump(k2clip, open(os.path.join(out, "_LOCAL_ONLY_k2clip.json"), "w"))
    # pod-side meta: pod3 has NO pandas, so the pod reads plain json (never the parquet)
    json.dump([{"k": int(r["k"]), "file": meta[int(r["k"])]["file"], "T": int(r["T"]),
                "side": r["side"], "chunk": r["chunk"], "cache": r["cache"]}
               for _, r in D.iterrows()],
              open(os.path.join(out, "sc_meta.json"), "w"))

    uni = {"n_clips": len(D), "lead_s": S.LEAD_S, "min_useful_lead_s": S.MIN_USEFUL_LEAD_S,
           "split_rule": "sorted(chunks)[j] -> TRAIN iff j mod 3 == 0",
           "n_chunks": len(CH), "n_chunks_train": len(TRAIN_CH),
           "per_side": {}, "per_situation": {}}
    for side in ("TRAIN", "HELDOUT"):
        d = D[D.side == side]
        uni["per_side"][side] = {"clips": int(len(d)), "chunks": int(d.chunk.nunique()),
                                 "frames": int(d["T"].sum()),
                                 "encoder_seen": int(d.encoder_seen.sum())}
    for s in SITS:
        e = {}
        for side in ("TRAIN", "HELDOUT"):
            d = D[D.side == side]
            e[side] = {"events": int(d[f"n_ev_{s}"].sum()),
                       "event_clips": int((d[f"n_ev_{s}"] > 0).sum()),
                       "pos_frames": int(d[f"n_pos_{s}"].sum()),
                       "valid_frames": int(d[f"n_valid_{s}"].sum()),
                       "base_rate": round(float(d[f"n_pos_{s}"].sum() / max(d[f"n_valid_{s}"].sum(), 1)), 6),
                       "pos_clips": int((d[f"n_pos_{s}"] > 0).sum())}
        uni["per_situation"][s] = e
    hs = D[D.side == "HELDOUT"]
    uni["roundabout_ccw_purity"] = {
        "TRAIN": _pur(D[(D.side == "TRAIN") & (D.n_ev_roundabout > 0)], "rb_ccw"),
        "HELDOUT_OUT_OF_SAMPLE": _pur(hs[hs.n_ev_roundabout > 0], "rb_ccw"),
        "core_TRAIN": _pur(D[(D.side == "TRAIN") & (D.n_ev_rbcore > 0)], "rbcore_ccw"),
        "core_HELDOUT": _pur(hs[hs.n_ev_rbcore > 0], "rbcore_ccw"),
        "note": "corpus has 0 left-hand-traffic clips -> a true roundabout label must be ~100% ccw",
    }
    uni["turn_ccw_balance_HELDOUT"] = round(float(
        (hs["n_ev_turn"] > 0).sum() and 0.0) or 0.0, 4)   # filled by the validator (needs per-event)
    uni["C_POW_min_clusters"] = 40
    uni["C_POW_verdict"] = {s: ("OK" if uni["per_situation"][s]["HELDOUT"]["pos_clips"] >= 40
                                else "UNDERPOWERED") for s in SITS}
    per_chunk = D.groupby(["chunk", "side"]).agg(
        clips=("k", "size"), **{f"pos_{s}": (f"n_pos_{s}", "sum") for s in SITS}).reset_index()
    uni["per_chunk"] = per_chunk.to_dict("records")
    json.dump(uni, open(os.path.join(out, "universe.json"), "w"), indent=2)
    print(json.dumps({kk: vv for kk, vv in uni.items() if kk != "per_chunk"}, indent=2))


def _mask(T, events):
    m = np.zeros(T, bool)
    for a, b in events:
        m[a:b + 1] = True
    return m


def _pur(d, col):
    if not len(d):
        return {"n": 0, "ccw_frac": None}
    return {"n": int(len(d)), "ccw_frac": round(float((d[col] > 0).mean()), 4)}


def _privileged(K):
    """⭐ C-POS ONLY. The FUTURE 3 s summary the labels are literally built from — net heading
    change, net lateral offset, |curvature| integral, speed change. A probe on this MUST separate;
    if it cannot, the instrument is not sensitive enough at this n and no null is readable."""
    T = K["T"]
    L = int(round(S.LEAD_S * S.HZ))
    psi, x, y, v, kap = K["psi"], K["x"], K["y"], K["v"], K["kappa"]
    j = np.minimum(np.arange(T) + L, T - 1)
    dpsi = psi[j] - psi
    c, s = np.cos(psi), np.sin(psi)
    lat = -s * (x[j] - x) + c * (y[j] - y)
    lon = c * (x[j] - x) + s * (y[j] - y)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(kap))])
    kint = cum[j] - cum[np.arange(T)]
    kmax = np.array([np.abs(kap[i:jj + 1]).max() if jj > i else 0.0 for i, jj in enumerate(j)])
    return np.stack([dpsi, lat, lon, kint, kmax, v[j] - v], 1)


if __name__ == "__main__":
    main()
