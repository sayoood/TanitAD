"""The stratum-D sweep pack — the only place a FALSE-PASS can live, plus the physical candidates.

Routes 1 and 2 found no real departure (see `raw/positive_search.json`), so this pack cannot be
positive-enriched. What it can do is look where a miss would hide: the 408 windows where NO rule
fires, which the first pack sampled ten times.

⛔ DRAWN BY CLIP FIRST — one window per clip — because the last pack counted WINDOWS as if they
were independent observations and its most-flattered candidate rested on three clips. The cluster
count is printed beside every n here.

Mixed in, with blind ids drawn from the same shuffle so a labeller cannot tell them apart: the
route-1 vertical-shock events and the route-2 widest-margin window. If any of those is a real
departure, it should be labelled over-boundary without the labeller knowing why it was included.

⛔ The sheet design is unchanged except for the one fix the Master Mind required: panel A now
states that the overlay is NOT depth-tested.

    python sweep_pack.py --raw <raw dir> --out <media dir> --key <path OUTSIDE the repo>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from adjudication_pack import P1, P2, RUBRIC, V0, sheet
from dac_anatomy import DEFAULTS, _import_harness
from dac_project import controls

SEED = 20260921
# ⚠️ plain ASCII on purpose: matplotlib's DejaVu Sans has no glyph for the warning sign,
# and the first render put a tofu box in the one sentence the sheet MUST carry.
DEPTH_NOTE = " -- NOT DEPTH-TESTED: a ground point BEHIND a raised object is drawn OVER it"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--extrinsics",
                    default="D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json")
    a = ap.parse_args()
    raw, out, key_path = Path(a.raw), Path(a.out), Path(a.key)
    if str(key_path.resolve()).startswith(str(Path("D:/Projects/TanitAD").resolve())):
        raise SystemExit(f"⛔ the key would land inside the repo: {key_path}")
    out.mkdir(parents=True, exist_ok=True)
    S1, P, SMG = _import_harness()
    import refc_v3_train as V3
    from tanitad.data.rig_projection import RigCamera
    frame = V3._agent_cam_frames()[(416, 1024)]
    _, table = V3._read_rig_extrinsics(a.extrinsics)

    rows = {(r["sha12"], r["t0"]): r for r in
            (json.loads(l) for l in (raw / "dac_windows.jsonl").read_text(encoding="utf-8").splitlines())}
    search = json.loads((raw / "positive_search.json").read_text(encoding="utf-8"))
    src: dict = {}
    # (a) stratum D, BY CLIP FIRST: one window per clip
    pool_d = [k for k, r in rows.items() if r["variants"][V0] == 1.0
              and r["variants"][P1] == 1.0 and r["variants"][P2] == 1.0]
    by_clip = defaultdict(list)
    for k in pool_d:
        by_clip[k[0]].append(k)
    rng = random.Random(SEED)
    clips = sorted(by_clip)
    rng.shuffle(clips)
    for c in clips:
        src[rng.choice(sorted(by_clip[c]))] = "D sweep (no rule fires)"
    # (b) the route-1 shock events: ONE window per distinct event (per clip), the loudest
    hits = search["route_1_ego_dynamics"]["hits"]
    best: dict = {}
    for h in hits:
        if h["az_hf_max"] > best.get(h["sha12"], (0.0,))[0]:
            best[h["sha12"]] = (h["az_hf_max"], (h["sha12"], h["t0"]))
    for _, k in best.values():
        src.setdefault(k, "route 1: vertical-shock event")
    # (c) the route-2 widest margin
    kd = [json.loads(l) for l in (raw / "kerb_depth_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    widest = max(kd, key=lambda s: s["distance_to_mapped_drivable_m"])
    src.setdefault((widest["sha12"], widest["t0"]), "route 2: widest margin past the mapped edge")

    ids = [f"S{i:02d}" for i in range(1, len(src) + 1)]
    rng.shuffle(ids)
    blind = dict(zip(sorted(src), ids))

    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    todo = {}
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        k = (S1.sha12(str(corp.clip_ids[e_i])), int(t + corp.W - 1))
        if k in blind:
            todo[wi] = (k, e_i)
    if len(todo) != len(blind):
        raise SystemExit(f"⛔ {len(todo)} of {len(blind)} drawn windows found — pack would be short")

    ctrl, made = None, []
    for wi in sorted(todo, key=lambda w: todo[w][1]):          # episode order: the LRU is 2
        k, e_i = todo[wi]
        cid = str(corp.clip_ids[e_i])
        it = corp.light_item(wi)
        corners = P._ego_boxes(it["human"][None], P.PROXY)[0].numpy()
        cam = RigCamera.from_extrinsics(table[cid], frame)
        if ctrl is None:
            ctrl = controls(cam, frame, corners[0])
            if not ctrl["all_pass"]:
                raise SystemExit(f"⛔ a projection control missed; nothing rendered: {ctrl}")
        pts = torch.tensor(np.concatenate([corners, np.zeros(corners.shape[:-1] + (1,))], -1),
                           dtype=torch.float64).reshape(-1, 3)
        col, row, _ = cam.project(pts)
        zc = cam.to_cam(pts)[..., 2].numpy()
        sh = corners.shape[:-1]
        col, row, front = col.numpy().reshape(sh), row.numpy().reshape(sh), (zc > 0).reshape(sh)
        fr = corp.eps[e_i].frames
        im = [fr[min(it["t0"] + d, fr.shape[0] - 1)][-3:].permute(1, 2, 0).numpy().astype(np.uint8)
              for d in (0, 20, 40)]
        sheet(im[0], im[1], im[2], col, row, front, corners, blind[k],
              out / f"{blind[k]}.jpg", panel_a_note=DEPTH_NOTE)
        made.append(blind[k])

    key = {"_what": "⛔ KEY to the stratum-D sweep pack — do NOT land before the labels",
           "seed": SEED,
           "rows": [{"blind_id": blind[k], "sha12": k[0], "t0": k[1], "source": src[k],
                     "V0_fires": rows[k]["variants"][V0] == 0.0,
                     "P1_fires": rows[k]["variants"][P1] == 0.0,
                     "P2_fires": rows[k]["variants"][P2] == 0.0} for k in sorted(src)]}
    key_path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8", newline="\n")
    pack = {"_what": "stratum-D sweep pack: where a FALSE-PASS would hide",
            "_evidence_class": "MEASURED (ours; CPU, read-only)",
            "seed": SEED, "sheets": len(made),
            "draw_rule": "BY CLIP FIRST — shuffle clips, one window per clip",
            "stratum_D": {"windows": len(pool_d), "clips": len(by_clip),
                          "drawn": len(clips), "cluster_count_equals_n": True},
            "mixed_in": {"route 1 vertical-shock events": len(best),
                         "route 2 widest margin": 1,
                         "note": "blind ids come from the same shuffle, so a labeller cannot "
                                 "tell these from the sweep windows"},
            "sheet_design": "unchanged, except panel A now states the overlay is NOT depth-tested",
            "depth_note": DEPTH_NOTE.strip(" —"),
            "projection_controls": ctrl, "rubric": RUBRIC,
            "key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
            "key_location": "OUTSIDE the repo, held until the labels are landed"}
    (raw / "sweep_pack.json").write_text(json.dumps(pack, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    tmpl = ["# H-DAC-DEF-1 stratum-D sweep. One label per sheet, from the images alone.",
            "# No map is shown. ⛔ The overlay is NOT depth-tested: a ground point behind a raised",
            "# object is drawn over it, which biases a reader TOWARD over-boundary.",
            *[f"# {k}: {v}" for k, v in RUBRIC.items()],
            "blind_id,label,note"]
    tmpl += [f"{i},," for i in sorted(made)]
    (out.parent / "LABELS_SWEEP_TEMPLATE.csv").write_text("\n".join(tmpl) + "\n",
                                                          encoding="utf-8", newline="\n")
    print(json.dumps({k: pack[k] for k in ("sheets", "stratum_D", "mixed_in", "key_sha256")},
                     indent=1))
    print("clips with >1 D window:", dict(Counter(len(v) for v in by_clip.values())))
    print("projection controls all pass:", ctrl["all_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
