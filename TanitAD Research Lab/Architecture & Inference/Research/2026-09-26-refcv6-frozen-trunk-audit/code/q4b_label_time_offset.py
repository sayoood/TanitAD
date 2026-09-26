"""Q4b -- how many refcv6 TRAINING windows the tactical-label time offset moves, counted with the
programme's OWN label functions (not a re-implementation).

The defect (see q4_timebase_identity.py): refcv6's V3Dataset reads a v7/v8 label at
    t_now = (t + w - 1) * v7_dt,   v7_dt = 0.1          (refc_v3_train.py:3009-3010, :3029-3030)
where `t + w - 1` is a PROVIDER row (v2_dataset.py:36, poses[n_stack-1:]). The label's t0/bands
live on the RAW clip timeline (egomotion_source.py:56-57, true 10 Hz from the recording start),
and the programme's own S2 join converts provider -> raw with +(n_stack - 1)
(s2_labels.py:736-740). The raw row step is the builder's span/(n_target-1), MEASURED ~0.10067 s.

    true t_now = (t + w - 1 + n_stack - 1) * dt_clip

This script enumerates every training window exactly as the trainer does
(refb_train.build_window_index: range(T - window - max_horizon), window 8, max_horizon 20),
joins each clip to its label record by stable_episode_id (the trainer's join), and asks the
REAL `v7_labels.tactical_class_ids` / `tactical_goal_targets` whether each window is supervised
under the trainer's time and under the true time.

CONTROLS (literals):
  K1  with offset 0 and dt 0.1 the "true" mapping must reproduce the trainer's admission EXACTLY
      (0 differing windows) -- the comparison is live, not reading two copies of one thing;
  K2  per-clip admitted-window count under the trainer's mapping is the literal 41 (rows 60..100)
      for every clip whose NOW-row range [7, T-22] contains [60, 100].
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402
from tanitad.data import v7_labels as v7l  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

SCRATCH = C.SCRATCH
SPLITS = {
    "train": (SCRATCH / "refcv6-b1-416x1024-train___v2manifest.pt",
              Path("C:/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_train.jsonl.gz"),
              "b45377a1f25263b5c0f3d318c126b1ac"),
    "eval139": (C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
                C.KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz",
                "eefc38d1453bd1c73802d44d45affced"),
}
W, MAX_H = 8, 20          # cfg.core.window (config: ego_history.steps 8) ; refc_v3_train kw max_horizon=20
V_MIN = 2.0


def clip_dt(p: np.ndarray) -> float:
    d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    vb = 0.5 * (p[1:, 3] + p[:-1, 3])
    m = (p[1:, 3] > V_MIN) & (p[:-1, 3] > V_MIN)
    return float(d[m].sum() / vb[m].sum()) if m.sum() >= 10 else float("nan")


def run(split: str, man_path: Path, lab_path: Path, md5_want: str, dt_fallback: float) -> dict:
    labels, lman = v7l.load_v7_labels(str(lab_path), allow_oracle_nav=True)
    if lman.md5 != md5_want:
        raise SystemExit(f"[q4b] {split} label md5 {lman.md5} != run stamp {md5_want}")
    by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    m = torch.load(str(man_path), map_location="cpu", weights_only=False)
    ns_all = sorted(set(int(x) for x in m["n_stack"]))
    tot = dict(windows=0, clips=0, clips_with_record=0, adm_trainer=0, adm_true=0,
               trainer_only=0, true_only=0, control_K1_diff=0, K2_violations=0,
               K2_checked=0, goal_cells_supervised_trainer=0, goal_cells_supervised_true=0,
               goal_cells_flipped=0, dt_fallback_clips=0)
    offs = []
    for i in range(len(m["poses"])):
        p = m["poses"][i].double().numpy()
        T = int(p.shape[0])
        ns = int(m["n_stack"][i])
        lab = by_sid.get(stable_episode_id(m["clip_id"][i]))
        tot["clips"] += 1
        n_win = T - W - MAX_H
        tot["windows"] += max(n_win, 0)
        if lab is None or n_win <= 0:
            continue
        tot["clips_with_record"] += 1
        dt = clip_dt(p)
        if math.isnan(dt):
            dt = dt_fallback
            tot["dt_fallback_clips"] += 1
        adm_tr_clip = 0
        for t in range(n_win):
            r = t + W - 1
            t_tr = r * 0.1
            t_true = (r + ns - 1) * dt
            a_tr = v7l.tactical_class_ids(lab, t_tr)[0] != v7l.IGNORE_ID
            a_true = v7l.tactical_class_ids(lab, t_true)[0] != v7l.IGNORE_ID
            a_k1 = v7l.tactical_class_ids(lab, (r + 0) * 0.1)[0] != v7l.IGNORE_ID
            tot["control_K1_diff"] += int(a_k1 != a_tr)
            tot["adm_trainer"] += int(a_tr)
            tot["adm_true"] += int(a_true)
            tot["trainer_only"] += int(a_tr and not a_true)
            tot["true_only"] += int(a_true and not a_tr)
            adm_tr_clip += int(a_tr)
            if a_tr != a_true:
                _y1, w1 = v7l.tactical_goal_targets(lab, t_tr, negatives="measured")
                _y2, w2 = v7l.tactical_goal_targets(lab, t_true, negatives="measured")
                tot["goal_cells_flipped"] += int(sum(1 for a, b in zip(w1, w2) if a != b))
            if a_tr:
                _y, w1 = v7l.tactical_goal_targets(lab, t_tr, negatives="measured")
                tot["goal_cells_supervised_trainer"] += int(sum(1 for x in w1 if x > 0))
            if a_true:
                _y, w2 = v7l.tactical_goal_targets(lab, t_true, negatives="measured")
                tot["goal_cells_supervised_true"] += int(sum(1 for x in w2 if x > 0))
        last_now_row = (n_win - 1) + W - 1          # NOW rows are [W-1, T-MAX_H-2]
        if last_now_row >= 100:                     # the clip's NOW rows cover [60, 100]
            tot["K2_checked"] += 1
            tot["K2_violations"] += int(adm_tr_clip != 41)
        # true time of the row the trainer calls the anchor (row 80 = 8.0 s at 0.1 s/row)
        offs.append((80 + ns - 1) * dt - 8.0)
    tot["n_stack_values"] = ns_all
    tot["frac_trainer_admitted_outside_true_band"] = tot["trainer_only"] / max(tot["adm_trainer"], 1)
    tot["frac_true_band_ignored_by_trainer"] = tot["true_only"] / max(tot["adm_true"], 1)
    o = np.asarray(offs)
    tot["offset_at_anchor_row_s"] = {"median": float(np.median(o)), "min": float(o.min()),
                                     "max": float(o.max())}
    tot["label_md5"] = lman.md5
    return tot


if __name__ == "__main__":
    C.ram_guard("q4b_label_time_offset (light job; the brief's 8 GB floor applies to every job)")
    out = {"what": "Q4b: training/eval windows whose tactical-label admission moves under the "
                   "correct provider->raw time mapping",
           "evidence_class": "MEASURED (ours; label functions are the trainer's own, "
                             "tanitad.data.v7_labels)",
           "window": W, "max_horizon": MAX_H, "splits": {}}
    for split, (mp, lp, md5) in SPLITS.items():
        out["splits"][split] = run(split, mp, lp, md5, dt_fallback=0.100667)
        print(split, out["splits"][split], flush=True)
    C.write_json("q4b_label_time_offset.json", out)
