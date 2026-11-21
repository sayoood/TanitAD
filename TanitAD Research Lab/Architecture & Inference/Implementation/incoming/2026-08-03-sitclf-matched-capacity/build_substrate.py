"""B4 — build a LOCAL sitclf substrate: v1-frozen vision features + the canonical situation labels.

WHY THIS EXISTS (and why it is not the banked substrate)
-------------------------------------------------------
BACKLOG **B4** asks for a matched-capacity vision head. That needs the image FEATURES.
The banked bundle
``…/2026-07-26-situation-classifier/artifacts/heldout_frames.npz`` stores **scores only** —
per-frame outputs of 10 arms, the 3-d ego block, the labels and the clip ids. The 2048-d
``SpatialGridReadout`` features those arms consumed were written to ``pod3:/workspace/sitclf/feats``
and pod3 is gone. Probed THREE ways at HEAD, all negative:

  1. ``find`` for ``clip_0*.npy`` over the repo and ``C:/Users/Admin`` -> nothing;
  2. the banked ``heldout_frames.npz`` / ``scores.npz`` bank ``E`` (ego) and the score columns but
     never ``F``; ``_pod_backup/`` holds pod2 *model* checkpoints only;
  3. ``tanitad-thor`` (the one idle host) has no ``sitclf`` dir and no PhysicalAI *train* cache.

=> the features are UNRECOVERABLE, so B4 is answered on a substrate rebuilt here.

WHAT IS IDENTICAL TO THE BANKED RUN
  * the ENCODER: v1 = ``flagship4b-speedjerk-30k``, step 29999, encoder 87,022,848 +
    readout 98,432 params, STRICT-loaded, frozen, fp32 weights, ``uint8 -> /255`` preprocessing —
    the same ``WorldModel.encode`` call ``sc_features.py:88-91`` makes.
  * the LABELS: ``tanitad.data.situations`` — its detector constants are BYTE-IDENTICAL to the
    banked ``sc_situations.py`` (verified by diff), ``lead_s = 3.0``, ``cross=None`` so
    ``intersection`` is the turn half alone, exactly as the emitter passes it.
  * the EGO block: ``[v, alon_pre, omega_pre] / EGO_SCALE`` as ``sc_train.py:93``.

WHAT IS NOT
  * the EPISODES. The parity cache ``physicalai-train-e438721ae894`` (2376 clips) is not on this
    box; the local caches are ``physicalai-train-14231cd29c74`` (400) and
    ``physicalai-val-bb543bdf7836`` (100). **Absolute APs here are therefore NOT comparable to the
    banked table.** Every claim in this stream is a WITHIN-substrate paired contrast between arms
    scored on IDENTICAL rows, which is what a capacity comparison needs.
  * this does not touch, re-select or re-hash any training corpus: it is an offline probe of a
    frozen encoder's latents, not a model arm.

usage:
  python build_substrate.py --out substrate.npz [--limit 0] [--device cuda]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data.situations import (  # noqa: E402
    HZ, anticipation_target, detect_intersection, detect_lane_change,
    detect_roundabout, kinematics)

SITS = ("lane_change", "roundabout", "intersection")
EGO_SCALE = np.array([10.0, 2.0, 0.5], dtype=np.float32)   # sc_train.py:38
LEAD_S = 3.0                                               # universe.json: lead_s = 3.0
V1_CKPT = r"C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt"
CACHES = (
    (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74", "train"),
    (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836", "val"),
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_trunk(device):
    """STRICT-load v1's frozen encoder + readout. A renamed key is a hard failure."""
    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel

    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    w = WorldModel(flagship4b_config())
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    ro = {k[len("readout."):]: v for k, v in sd.items() if k.startswith("readout.")}
    w.encoder.load_state_dict(enc, strict=True)
    w.readout.load_state_dict(ro, strict=True)
    w = w.to(device).eval()
    for p in w.parameters():
        p.requires_grad_(False)
    meta = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)),
            "encoder_params": int(sum(v.numel() for v in enc.values())),
            "readout_params": int(sum(v.numel() for v in ro.values())),
            "state_dim": int(w.state_dim), "strict_load": True, "frozen": True}
    log(f"trunk STRICT-loaded: {json.dumps(meta)}")
    return w, meta


def labels_for(P):
    """The canonical anticipation targets for one clip -> (y [T,3] bool, valid [T,3] bool)."""
    K = kinematics(np.asarray(P, dtype=np.float64))
    T = int(K["T"])
    ev = {
        "lane_change": detect_lane_change(K),
        "roundabout": detect_roundabout(K, bracket=True),
        # cross=None -> the TURN half alone, exactly as emit_situation_labels.py passes it
        "intersection": detect_intersection(K, cross=None)[0],
    }
    y = np.zeros((T, 3), bool)
    v = np.zeros((T, 3), bool)
    for i, s in enumerate(SITS):
        y[:, i], v[:, i] = anticipation_target(T, ev[s], lead_s=LEAD_S)
    E = np.stack([K["v"], K["alon_pre"], K["omega_pre"]], 1).astype(np.float32) / EGO_SCALE
    n_ev = {s: len(ev[s]) for s in SITS}
    return y, v, E, n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="substrate.npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    files = []
    for root, tag in CACHES:
        fs = sorted(glob.glob(os.path.join(root, "ep_*.pt")))
        files += [(f, tag) for f in fs]
        log(f"{tag}: {len(fs)} episodes in {root}")
    if a.limit:
        files = files[:a.limit]

    world, tmeta = load_trunk(device)
    F, Y, V, E, CC, TAG, TT = [], [], [], [], [], [], []
    n_ev_tot = {s: 0 for s in SITS}
    t0 = time.time()
    for k, (f, tag) in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        fr = d["frames_u8"]
        P = d["poses"]
        P = P.numpy() if torch.is_tensor(P) else np.asarray(P)
        if int(fr.shape[0]) != int(P.shape[0]):
            raise SystemExit(f"C-FID: {f} frames {fr.shape[0]} != poses {P.shape[0]}")
        y, v, e, n_ev = labels_for(P)
        feats = np.empty((fr.shape[0], world.state_dim), np.float16)
        with torch.no_grad():
            for b in range(0, fr.shape[0], a.batch):
                x = fr[b:b + a.batch].to(device, non_blocking=True).float().div_(255.0)
                feats[b:b + a.batch] = world.encode(x).to(torch.float16).cpu().numpy()
        F.append(feats); Y.append(y); V.append(v); E.append(e)
        CC.append(np.full(len(y), k, np.int32))
        TAG.append(np.full(len(y), 0 if tag == "train" else 1, np.int8))
        TT.append(np.arange(len(y), dtype=np.int32))
        for s in SITS:
            n_ev_tot[s] += n_ev[s]
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            log(f"  {k+1}/{len(files)} clips  ({el:.0f}s, {el/(k+1):.2f}s/clip)")

    F = np.concatenate(F); Y = np.concatenate(Y); V = np.concatenate(V)
    E = np.concatenate(E); CC = np.concatenate(CC); TAG = np.concatenate(TAG)
    TT = np.concatenate(TT)
    log(f"substrate: {F.shape[0]:,} frames x {F.shape[1]} feat, "
        f"{len(np.unique(CC))} clip clusters")
    for i, s in enumerate(SITS):
        m = V[:, i]
        log(f"  {s}: events {n_ev_tot[s]}, scorable {int(m.sum()):,}, "
            f"pos {int(Y[m, i].sum()):,}, base rate {float(Y[m, i].mean()):.5f}")

    np.savez(a.out, F=F, Y=Y.astype(np.uint8), V=V.astype(np.uint8), E=E,
             clip_cluster=CC, cache_tag=TAG, t=TT, situations=np.array(SITS))
    meta = {"trunk": tmeta, "caches": [c[0] for c in CACHES], "n_clips": len(files),
            "n_frames": int(F.shape[0]), "state_dim": int(F.shape[1]),
            "lead_s": LEAD_S, "hz": HZ, "ego_scale": EGO_SCALE.tolist(),
            "labels": "tanitad.data.situations (constants byte-identical to sc_situations.py)",
            "intersection_cross": None,
            "n_events": n_ev_tot,
            "per_situation": {s: {"scorable": int(V[:, i].sum()),
                                  "pos": int(Y[V[:, i], i].sum()),
                                  "base_rate": round(float(Y[V[:, i], i].mean()), 6)}
                              for i, s in enumerate(SITS)},
            "wallclock_s": round(time.time() - t0, 1)}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {a.out} + {Path(a.out).with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
