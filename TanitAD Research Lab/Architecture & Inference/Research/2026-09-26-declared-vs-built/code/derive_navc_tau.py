"""E1 (informative until the PI decides): the nav-compliance tolerance ``tau_rad`` for
``--graft-nav-compliance``, derived on the FULL refcv6/refcv7 TRAIN split, recorded with the
sha256 of every input.

THE RULE (stated in the model code, not invented here): ``refc_selector_targets.compliance_target``
scores a candidate by the heading of its LAST path segment (0.0 when that segment is < 0.05 m) and
``taniteval.nav_compliance.derive_tolerance(pos, neg)`` picks tau by Youden's J of |heading| on
windows whose nav command has a SIDE (left / right: ``pos``) against FOLLOW windows (``neg``) --
GT + label only, never a model output. ``straight`` is neither (no side, not follow).

THE INPUTS -- the trainer's own objects, on the split the model TRAINS on:
* POSES: the train split's v2 MANIFEST (`clip_id`, `poses`, `n_stack` per clip), pulled read-only
  from Thor for the A16 audit (md5 3c9f8bc8..., the same file the clip-clock sidecar was built
  from). No frame is decoded; each clip becomes a pose-only episode (frames are a [T, 3*n_stack,
  1, 1] zero stub so `V3Dataset` builds its window index exactly as the trainer does).
* LABELS: `s2_labels_v8_train.jsonl.gz` (md5 b45377a1... == the run's `config.json.v7_labels.md5`),
  joined by `stable_episode_id` and turned into nav by `V3Dataset.enable_nav_from_v7` -- the SAME
  nav the trainer feeds (`--nav-from-v7`).
* GT: `refb_labels.waypoint_targets` on `V3_HORIZONS` from each window's NOW row -- the SAME
  target the trajectory loss uses; windows whose future does not cover the full horizon are skipped.

    python derive_navc_tau.py <tree> <manifest.pt> <labels.jsonl.gz> <out.json> [--stride 1]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

tree = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(tree / "stack"), str(tree / "stack" / "scripts"), str(tree / "taniteval")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tanitad  # noqa: E402

assert str(Path(tanitad.__file__).resolve()).lower().startswith(str(tree).lower()), tanitad.__file__
import refb_labels  # noqa: E402
import refc_v3_train as T  # noqa: E402
from tanitad.data import v7_labels as v7l  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from taniteval.nav_compliance import derive_tolerance  # noqa: E402

WINDOW = 8          # refcv6-r101-s0 core.window (config.json ego_history.steps)
STALL_M = 0.05      # compliance_target's stall threshold


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def terminal_heading(xy: np.ndarray) -> float:
    d = xy[-1] - xy[-2]
    return 0.0 if float(np.hypot(d[0], d[1])) < STALL_M else float(np.arctan2(d[1], d[0]))


def main() -> None:
    man_p, lab_p, out = sys.argv[2], sys.argv[3], sys.argv[4]
    if Path(out).exists():
        # a second runner (Thor) may have banked it first: never overwrite, keep both
        out = str(Path(out).with_suffix("")) + ".devbox.json"
    stride = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv else 1
    t0 = time.time()
    man = torch.load(man_p, map_location="cpu", weights_only=False)
    eps = []
    for i, cid in enumerate(man["clip_id"]):
        poses = torch.as_tensor(man["poses"][i]).to(torch.float32)
        n_ch = 3 * int(man["n_stack"][i])
        eps.append(SimpleNamespace(
            poses=poses, frames=torch.zeros(poses.shape[0], n_ch, 1, 1, dtype=torch.uint8),
            actions=torch.zeros(poses.shape[0], 2), episode_id=stable_episode_id(str(cid))))
    chans = sorted({int(e.frames.shape[1]) for e in eps})
    assert len(chans) == 1, f"mixed stack widths {chans}"
    ds = T.V3Dataset(eps, window=WINDOW, max_horizon=20, channels=chans[0])
    labels, lman = v7l.load_v7_labels(lab_p, allow_oracle_nav=True)
    ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    ds.v7_dt = 0.1
    nav_stats = ds.enable_nav_from_v7(lman)
    nav_by_sid = ds._nav_by_sid
    NAV = T.refb.NAV_COMMANDS
    left, right, follow = NAV.index("left"), NAV.index("right"), NAV.index("follow")
    hz = tuple(v3.V3_HORIZONS)
    pos, neg = [], []
    n_scored = n_short = n_nonav = 0
    for i, (e_i, t) in enumerate(ds.index):
        if i % stride:
            continue
        ep = ds.episodes[e_i]
        nav = nav_by_sid.get(int(ep.episode_id))
        if nav is None:
            n_nonav += 1
            continue
        f = t + WINDOW - 1
        T_ = int(ep.poses.shape[0])
        if f + max(hz) > T_ - 1:
            n_short += 1
            continue
        fut = ep.poses[torch.arange(f + 1, f + 1 + max(hz))][None]
        wp = refb_labels.waypoint_targets(ep.poses[f][None], fut, hz)[0].numpy()
        th = terminal_heading(wp)
        n_scored += 1
        if nav in (left, right):
            pos.append(th)
        elif nav == follow:
            neg.append(th)
    res = derive_tolerance(pos, neg)
    res.update({
        "what": "nav-compliance tau for --graft-nav-compliance (E1; informative until the PI decides)",
        "evidence_class": "MEASURED (dev box, CPU; GT poses + labels only, no model)",
        "split": "refcv6/refcv7 TRAIN (4,369 clips; config.json v2_cache "
                 "/home/nvidia/data/refcv6-b1-416x1024-train)",
        "inputs": {"manifest": {"path": man_p, "sha256": sha256(man_p),
                                "n_clips": len(eps)},
                   "labels": {"path": lab_p, "sha256": sha256(lab_p),
                              "md5": getattr(lman, "md5", None)}},
        "code": {"tree": str(tree), "script_sha256": sha256(__file__)},
        "window": WINDOW, "stride": stride, "horizons": list(hz),
        "windows_total": len(ds.index), "windows_scored": n_scored,
        "windows_skipped_short_future": n_short, "windows_skipped_no_nav": n_nonav,
        "n_pos_left_or_right": len(pos), "n_neg_follow": len(neg),
        "nav_from_v7_stats": nav_stats,
        "heading_rule": "compliance_target: last-segment heading of the GT waypoint path, 0.05 m stall",
        "method": {"function": "taniteval.nav_compliance.derive_tolerance",
                   "args": {"pos": "|terminal heading| of GT paths on LEFT or RIGHT windows",
                            "neg": "|terminal heading| of GT paths on FOLLOW windows",
                            "grid": "None -> np.linspace(0, max(|pos| U |neg|), 401)[1:]",
                            "unit": "rad"},
                   "selection": "argmax over the grid of TPR(pos >= tau) - FPR(neg >= tau)"},
        "seconds": round(time.time() - t0, 1)})
    Path(out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("nav_from_v7_stats",)},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
