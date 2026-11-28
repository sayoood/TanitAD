"""H2 classifier — STEP 1 (dev box, CPU): join the L2 label table to the pod2 episode cache.

Produces the study universe and a COMPACT, DE-IDENTIFIED label bundle for pod2.

Three things happen here and each one is published as an artifact:

1. **The episode -> clip join is PROVEN, not assumed.** `physicalai.build_episode` stores
   ``episode_id = int.from_bytes(clip_id.encode()[:4])``. Re-running the cache's own recipe
   (``discover_r0_clips`` -> ``sorted`` -> ``split_clips(val_frac=0.2, seed=0)``) must reproduce
   every stored id, or the join is refused.  -> `artifacts/join_proof.json`
2. **The universe is counted before anything is trained** (clips / frames / positives per chunk
   and per side of the pre-registered split).  -> `artifacts/universe.json`
3. **A de-identified bundle** goes to the pod: clip UUIDs are replaced by an integer index.
   PhysicalAI-AV is gated-confidential; no UUID may leave the dev box in a derived artifact.

usage:  python h2c_prep.py <l2tab_dir> <ep_ids.json> <r0_selection.parquet> <out_dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
L2S = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-26-h2-label-v2", "scripts"))
sys.path.insert(0, L2S)
from l2_label import (CONFIRM_CHUNKS, DEV_CHUNKS, response_r2,  # noqa: E402
                      trigger_l2_percam)

TAU = 0.5
# Columns carried to the pod. The `_res` suffix is NOT cosmetic: `trigger_l2(..., resolvable=True)`
# — the gated definition, amendment A2 — reads `areq_off_{L,R}_res` / `areq_seen_res`, i.e. the
# population with conflicts that braking can actually resolve. Using the un-suffixed columns
# reproduces the S3 sensitivity (1.89x), not the verdict (2.41x).
COLS = ["gi", "areq_off_L_res", "areq_off_R_res", "areq_seen_res",
        "areq_off_Lr_res", "areq_off_Rr_res",
        "ego_v", "alon_pre", "alon_fut_min", "ego_dv4",
        "n_agents", "junction", "lane_change"]


def eid_of(clip_id: str) -> int:
    return int.from_bytes(clip_id.encode()[:4].ljust(4, b"\0"), "big")


def rebuild_join(ep_ids_path: str, sel_path: str):
    """Reproduce the cache's own build order and PROVE it against the stored episode ids."""
    E = json.load(open(ep_ids_path))
    sel = pd.read_parquet(sel_path)
    clips = sorted(sel["clip_id"].astype(str))
    g = torch.Generator().manual_seed(0)                      # cfg.train.seed == 0
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * 0.2))
    val_i = set(perm[:n_val])
    tr = [c for i, c in enumerate(clips) if i not in val_i]
    va = [c for i, c in enumerate(clips) if i in val_i]

    rows, ok_tr, ok_va = [], 0, 0
    for (f, e, T, _s), c in zip(E["val"]["rows"], va):
        ok_va += int(e == eid_of(c))
        rows.append(("val", f, c, int(T)))
    for f, e, T, _s in E["train"]["rows"]:
        i = int(f.split("_")[1].split(".")[0])
        ok_tr += int(e == eid_of(tr[i]))
        rows.append(("train", f, tr[i], int(T)))
    proof = {
        "recipe": "discover_r0_clips -> sorted(clip_id) -> split_clips(val_frac=0.2, seed=0)",
        "episode_id_rule": "int.from_bytes(clip_id.encode()[:4], 'big')  (physicalai.py:496)",
        "n_selection": len(clips), "n_train_files": len(E["train"]["rows"]),
        "n_val_files": len(E["val"]["rows"]),
        "train_ids_reproduced": ok_tr, "val_ids_reproduced": ok_va,
        "skip_indices": [int(s.split("_")[1]) for s in E["train"]["skips"]],
        "join_proven": bool(ok_tr == len(E["train"]["rows"]) and ok_va == len(E["val"]["rows"])),
    }
    if not proof["join_proven"]:
        raise SystemExit(f"JOIN REFUSED: {ok_tr}/{len(E['train']['rows'])} train, "
                         f"{ok_va}/{len(E['val']['rows'])} val ids reproduced")
    return pd.DataFrame(rows, columns=["cache", "file", "clip_id", "T"]), proof


def main():
    tab, ep_ids, sel_path, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    M, proof = rebuild_join(ep_ids, sel_path)
    json.dump(proof, open(os.path.join(out, "join_proof.json"), "w"), indent=2)
    print(f"[join] PROVEN {proof['train_ids_reproduced']}/{proof['n_train_files']} train, "
          f"{proof['val_ids_reproduced']}/{proof['n_val_files']} val")

    # NB: never use `row.T` on a Series here — that is the transpose attribute, not the column.
    have = {r["clip_id"]: (r["cache"], r["file"], int(r["T"])) for _, r in M.iterrows()}
    per_chunk, packs, meta, k2clip = [], [], [], {}
    for p in sorted(glob.glob(os.path.join(tab, "l2_*.parquet"))):
        ch = os.path.basename(p)[3:7]
        side = "TRAIN" if ch in DEV_CHUNKS else ("HELDOUT" if ch in CONFIRM_CHUNKS else None)
        if side is None:
            raise SystemExit(f"chunk {ch} is in neither DEV nor CONFIRM")
        D = pd.read_parquet(p)
        D = D[D.clip_id.isin(have)]
        if not len(D):
            continue
        c_frames = c_pos = c_posL = c_posR = c_lab = c_clips = c_posclips = 0
        for cid, d in D.groupby("clip_id", sort=True):
            d = d.sort_values("gi")
            cache, fname, T = have[cid]
            # IMPORTED from the gated label module — never re-implemented here, so the
            # classifier provably trains on the predicate that was GO-gated.
            tL, tR = trigger_l2_percam(d, TAU)          # resolvable=True, scope="crop"
            r2 = response_r2(d)                          # the amended (A1) behavioural response
            k = len(meta)
            k2clip[str(k)] = cid
            meta.append({"k": k, "chunk": ch, "side": side, "cache": cache, "file": fname,
                         "T_episode": int(T), "n_rows": int(len(d)),
                         "encoder_seen": bool(cache == "train")})
            packs.append({f"c{k}_{c}": d[c].to_numpy() for c in COLS})
            c_clips += 1
            c_frames += len(d)
            c_pos += int((tL | tR).sum())
            c_posL += int(tL.sum())
            c_posR += int(tR.sum())
            c_lab += int(((tL | tR) & r2).sum())
            c_posclips += int((tL | tR).any())
        per_chunk.append(dict(chunk=ch, side=side, clips=c_clips, frames=c_frames, pos=c_pos,
                              posL=c_posL, posR=c_posR, label_pos=c_lab, pos_clips=c_posclips))
    P = pd.DataFrame(per_chunk)
    uni = {
        "tau_star": TAU, "scope": "crop (outside the 51.4 deg encoder crop)",
        "note": "clip UUIDs never leave the dev box; clips are indexed by k",
        "per_chunk": P.to_dict("records"),
        "per_side": {s: {c: int(v) for c, v in P[P.side == s].sum(numeric_only=True).items()}
                     for s in ("TRAIN", "HELDOUT")},
        "n_clips": len(meta),
        "encoder_seen_breakdown": {
            s: {"seen": sum(1 for m in meta if m["side"] == s and m["encoder_seen"]),
                "unseen": sum(1 for m in meta if m["side"] == s and not m["encoder_seen"])}
            for s in ("TRAIN", "HELDOUT")},
    }
    json.dump(uni, open(os.path.join(out, "universe.json"), "w"), indent=2)
    print(P.to_string())
    print(json.dumps(uni["per_side"], indent=2))
    print(json.dumps(uni["encoder_seen_breakdown"], indent=2))

    flat = {}
    for pk in packs:
        flat.update(pk)
    np.savez_compressed(os.path.join(out, "h2c_labels.npz"), **flat)
    json.dump(meta, open(os.path.join(out, "h2c_meta.json"), "w"))
    # the UUID map stays on the dev box ONLY (gated corpus) and is never uploaded
    json.dump(k2clip, open(os.path.join(out, "_LOCAL_ONLY_k2clip.json"), "w"))
    print(f"[prep] {len(meta)} clips -> {out}")


if __name__ == "__main__":
    main()
