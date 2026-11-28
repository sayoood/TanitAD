"""Q7b -- the 2-D agent join's PER-WINDOW coverage on the eval split: which window NOWs have no
`obstacle.offline` record, and does that correlate with position-in-clip, speed or clip?

The trainer supervises the detector only on windows whose NOW frame has a join line
(`_agent_item`, refc_v3_train.py:2764-2800: NO_LABEL vs LABELLED-CLEAR). The run's stamp says
4.67 % of eval windows (1,109 / 23,772) and 3.64 % of train windows are unlabelled. This streams
the SAME join file the run read (md5 0c31a3a6..., train + eval), keeps only (clip, provider
frame_idx) keys for the 139 eval clips, and tests the unlabelled windows against the labelled ones.
Light: regex over each line's prefix, no box parsing.
"""
from __future__ import annotations

import json
import lzma
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402

JOIN = C.KIT / "data/joins/b1_train_plus_eval_agents.jsonl.xz"
RX = re.compile(r'"clip_id":\s*"([^"]+)".*?"frame_idx":\s*(-?\d+)')


def main():
    C.ram_guard("q7b_agent_join_coverage (light job)")
    m = torch.load(str(C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt"),
                   map_location="cpu", weights_only=False)
    clips = list(m["clip_id"])
    want = set(clips)
    have: dict[str, set] = {c: set() for c in clips}
    n_lines = n_eval = 0
    with lzma.open(str(JOIN), "rt", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            mm = RX.search(line[:400])
            if mm and mm.group(1) in want:
                have[mm.group(1)].add(int(mm.group(2)))
                n_eval += 1
    W, MAXH = 8, 20
    rows = []
    for i, c in enumerate(clips):
        p = m["poses"][i].numpy()
        T = int(p.shape[0])
        for t in range(T - W - MAXH):
            r = t + W - 1
            rows.append((C.sha12(c), r, float(p[r, 3]), r in have[c]))
    lab = np.array([x[3] for x in rows])
    pos = np.array([x[1] for x in rows], dtype=np.float64)
    spd = np.array([x[2] for x in rows], dtype=np.float64)
    unl = ~lab
    per_clip = Counter(x[0] for x in rows if not x[3])
    n_clips_with_unl = len(per_clip)
    top = per_clip.most_common(5)
    # position: first / last 10 NOW rows vs middle
    edge = (pos < 17) | (pos > (np.max(pos) - 10))
    rng = np.random.default_rng(0)

    def perm_mean_diff(x, mask, n=10000):
        obs = abs(x[mask].mean() - x[~mask].mean())
        k = int(mask.sum())
        ge = 0
        for _ in range(n // 500):
            idx = np.argsort(rng.random((500, x.size)), axis=1)[:, :k]
            d = np.abs(x[idx].mean(1) - (x.sum() - x[idx].sum(1)) / (x.size - k))
            ge += int((d >= obs - 1e-12).sum())
        return round(float(obs), 4), round((ge + 1) / (n + 1), 5)
    out = {"what": "agent-join per-window coverage on the eval split",
           "evidence_class": "MEASURED (the run's own join file, streamed)",
           "join_lines_total": n_lines, "join_lines_eval": n_eval,
           "config_eval_join_lines": 26394,
           "n_windows": int(lab.size), "n_unlabelled": int(unl.sum()),
           "config_n_unlabelled": 23772 - 22663,
           "frac_unlabelled": float(unl.mean()),
           "n_clips_with_any_unlabelled_window": n_clips_with_unl,
           "top_clips_unlabelled_windows": [[k, v] for k, v in top],
           "share_of_unlabelled_in_top1_clip": (top[0][1] / max(int(unl.sum()), 1)) if top else None,
           "position_mean_unlabelled_vs_labelled": [float(pos[unl].mean()), float(pos[lab].mean())],
           "position_perm_test_(absdiff,p)": perm_mean_diff(pos, unl),
           "speed_mean_unlabelled_vs_labelled_ms": [float(spd[unl].mean()), float(spd[lab].mean())],
           "speed_perm_test_(absdiff,p)": perm_mean_diff(spd, unl),
           "frac_unlabelled_at_clip_edges_vs_middle": [float(unl[edge].mean()), float(unl[~edge].mean())]}
    print(json.dumps(out, indent=1))
    C.write_json("q7b_agent_join_coverage.json", out)


if __name__ == "__main__":
    main()
