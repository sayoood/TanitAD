"""Q4d -- q4b's window count, redone on each clip's TRUE clock from q4c (the egomotion-log
inversion): t_true(row) = grid_start + (row + n_stack - 1) * dt, both MEASURED per clip.

q4b assumed the camera grid starts at the label timeline's zero; q4c MEASURED that it starts
+0.113 s later (median, 4,357 train clips), so q4b's 0.25 s under-states the offset. This file
re-counts with the measured per-clip clock and the trainer's own label functions.
Clips q4c could not fit (too little motion) use the split median -- counted and reported.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tanitad.data import v7_labels as v7l  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

SCRATCH = C.SCRATCH
SPLITS = {
    "train": (SCRATCH / "refcv6-b1-416x1024-train___v2manifest.pt",
              Path("C:/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_train.jsonl.gz"),
              "b45377a1f25263b5c0f3d318c126b1ac", C.RAW / "q4c_grid_vs_egolog_ALLTRAIN.json"),
    "eval139": (C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
                C.KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz",
                "eefc38d1453bd1c73802d44d45affced", C.RAW / "q4c_grid_vs_egolog.json"),
}
W, MAX_H = 8, 20


def run(split, man_path, lab_path, md5_want, q4c_path):
    q = json.loads(Path(q4c_path).read_text(encoding="utf-8"))["splits"][split]
    clock = {r["clip"]: (r["grid_start_on_label_timeline_s"], r["dt_s"]) for r in q["per_clip"]
             if r["fit_resid_max_ms"] < 5.0}
    med = (q["grid_start_on_label_timeline_s"]["median"], q["dt_s"]["median"])
    labels, lman = v7l.load_v7_labels(str(lab_path), allow_oracle_nav=True)
    assert lman.md5 == md5_want, (lman.md5, md5_want)
    by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    m = torch.load(str(man_path), map_location="cpu", weights_only=False)
    tot = dict(windows=0, adm_trainer=0, adm_true=0, trainer_only=0, true_only=0,
               clips_median_clock=0, clips=0, adm_fixA=0, fixA_only=0, true_only_vs_fixA=0)
    off_anchor = []
    for i in range(len(m["poses"])):
        T = int(m["poses"][i].shape[0])
        ns = int(m["n_stack"][i])
        lab = by_sid.get(stable_episode_id(m["clip_id"][i]))
        n_win = T - W - MAX_H
        if lab is None or n_win <= 0:
            continue
        tot["clips"] += 1
        g0, dt = clock.get(C.sha12(m["clip_id"][i]), (None, None))
        if g0 is None:
            g0, dt = med
            tot["clips_median_clock"] += 1
        off_anchor.append(g0 + (80 + ns - 1) * dt - 8.0)
        for t in range(n_win):
            r = t + W - 1
            a_tr = v7l.tactical_class_ids(lab, r * 0.1)[0] != v7l.IGNORE_ID
            a_true = v7l.tactical_class_ids(lab, g0 + (r + ns - 1) * dt)[0] != v7l.IGNORE_ID
            tot["windows"] += 1
            tot["adm_trainer"] += int(a_tr)
            tot["adm_true"] += int(a_true)
            tot["trainer_only"] += int(a_tr and not a_true)
            tot["true_only"] += int(a_true and not a_tr)
            # fix part A (the patch in A16_fix_f3_and_label_clock.diff): + (n_stack - 1) rows, 0.1 s
            a_fa = v7l.tactical_class_ids(lab, (r + ns - 1) * 0.1)[0] != v7l.IGNORE_ID
            tot["adm_fixA"] += int(a_fa)
            tot["fixA_only"] += int(a_fa and not a_true)
            tot["true_only_vs_fixA"] += int(a_true and not a_fa)
    o = np.asarray(off_anchor)
    tot["offset_at_trainer_anchor_row_s"] = {"median": float(np.median(o)),
                                             "p05": float(np.quantile(o, .05)),
                                             "p95": float(np.quantile(o, .95))}
    tot["frac_trainer_admitted_outside_true_band"] = tot["trainer_only"] / max(tot["adm_trainer"], 1)
    tot["frac_true_band_ignored_by_trainer"] = tot["true_only"] / max(tot["adm_true"], 1)
    tot["fixA_frac_admitted_outside_true_band"] = tot["fixA_only"] / max(tot["adm_fixA"], 1)
    tot["fixA_frac_true_band_ignored"] = tot["true_only_vs_fixA"] / max(tot["adm_true"], 1)
    return tot


if __name__ == "__main__":
    C.ram_guard("q4d_label_offset_true_clock (light job; the brief's 8 GB floor applies to every job)")
    out = {"what": "Q4d: tactical-label admission on each clip's MEASURED clock (q4c)",
           "evidence_class": "MEASURED (ours)", "splits": {}}
    for s, args in SPLITS.items():
        out["splits"][s] = run(s, *args)
        print(s, out["splits"][s], flush=True)
    C.write_json("q4d_label_offset_true_clock.json", out)
