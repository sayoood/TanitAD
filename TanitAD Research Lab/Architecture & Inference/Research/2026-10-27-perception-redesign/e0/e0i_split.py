"""e0i_split.py -- E0i (REFCV9_DESIGN.md s4.1, PI decision D6 of 2026-11-02): the eval139 DEV / HOLD split.

Rule (frozen, from the design): order the 139 eval episodes by sha256(clip_id) hex ASCENDING (the convention of
`l1c_select.py`); rank 0, 2, 4, ... = DEV, rank 1, 3, 5, ... = HOLD.  Selection (E1 / E2 / D1 reads) is on DEV only;
the sealed bars (design s5.1) are on HOLD.

    python -X utf8 e0i_split.py --windows-s2 <s2_windows.json> --windows-s6 <s6_windows.json> \
        --labels-npz <v9_labels_eval139.npz> --out DEV_HOLD_SPLIT.json

Inputs are the kit's window grids: a dict {clip_id: [window ids]}.  Raw clip ids are read IN PROCESS ONLY; the output carries
sha12 (= sha256(clip_id)[:12]) and counts.  The output file's own sha256 goes in the sidecar `DEV_HOLD_SPLIT.sha256`
(a file cannot contain its own hash); the JSON carries `lists_digest` = sha256 over the canonical rank-ordered lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RULE = "order by sha256(clip_id) hex ascending; rank even = DEV, rank odd = HOLD"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def split_ids(ids):
    """ids -> (dev, hold), both lists of FULL sha256 hex strings in rank order.  Pure function of the id strings."""
    order = sorted(sha(c) for c in ids)
    if len(set(order)) != len(order):
        raise ValueError("duplicate clip ids")
    return order[0::2], order[1::2]


def build(w2: dict, w6: dict) -> dict:
    if set(w2) != set(w6):
        raise ValueError("S2 and S6 window grids cover different clip sets")
    dev, hold = split_ids(w2.keys())
    by_hash = {sha(c): c for c in w2}
    pre = [h[:12] for h in dev + hold]
    if len(set(pre)) != len(pre):
        raise ValueError("sha12 prefixes are not unique; use more digits")

    def half(hs):
        eps = [by_hash[h] for h in hs]
        return {"n_episodes": len(eps),
                "n_windows_s2": int(sum(len(w2[c]) for c in eps)),
                "n_windows_s6": int(sum(len(w6[c]) for c in eps)),
                "sha12": [h[:12] for h in hs],
                "windows_per_episode": {h[:12]: [len(w2[by_hash[h]]), len(w6[by_hash[h]])] for h in hs}}
    out = {"tool": "e0i_split.py", "rule": RULE, "n_clips": len(w2), "DEV": half(dev), "HOLD": half(hold),
           "windows_per_episode_columns": ["S2 (all battery windows)", "S6 (the kit's S6 window grid, before the ceiling-compliance filter)"]}
    out["lists_digest"] = hashlib.sha256(("DEV:" + ",".join(out["DEV"]["sha12"]) + "|HOLD:" +
                                          ",".join(out["HOLD"]["sha12"])).encode()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-s2", required=True)
    ap.add_argument("--windows-s6", required=True)
    ap.add_argument("--labels-npz", default="", help="optional independent clip-set check (v9 labels, key clip__sha12)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw2, raw6 = Path(a.windows_s2).read_bytes(), Path(a.windows_s6).read_bytes()
    w2, w6 = json.loads(raw2), json.loads(raw6)
    out = build(w2, w6)
    out["inputs"] = {"s2_windows_json_sha256": hashlib.sha256(raw2).hexdigest(), "s6_windows_json_sha256": hashlib.sha256(raw6).hexdigest()}
    # controls that must hold on the real data (printed counts only, never ids)
    d, h = out["DEV"], out["HOLD"]
    assert d["n_episodes"] + h["n_episodes"] == len(w2) == 139, "episode partition"
    assert not (set(d["sha12"]) & set(h["sha12"])), "DEV and HOLD overlap"
    assert d["n_windows_s2"] + h["n_windows_s2"] == sum(len(v) for v in w2.values()), "S2 window partition"
    assert d["n_windows_s6"] + h["n_windows_s6"] == sum(len(v) for v in w6.values()), "S6 window partition"
    ctl = {"episode_partition_139": True, "disjoint": True, "s2_windows_sum": True, "s6_windows_sum": True}
    if a.labels_npz:
        import numpy as np
        lab = set(np.load(a.labels_npz, allow_pickle=True)["clip__sha12"].tolist())
        ok = lab == set(d["sha12"]) | set(h["sha12"])
        assert ok, "clip set differs from the v9 label release"
        ctl["clip_set_equals_v9_labels_release"] = True
    out["controls_passed_at_build"] = ctl
    p = Path(a.out)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    dig = hashlib.sha256(p.read_bytes()).hexdigest()
    p.with_suffix(".sha256").write_text(f"{dig}  {p.name}\n", encoding="utf-8", newline="\n")
    print(f"DEV  {d['n_episodes']} episodes, {d['n_windows_s2']} S2 windows, {d['n_windows_s6']} S6 windows")
    print(f"HOLD {h['n_episodes']} episodes, {h['n_windows_s2']} S2 windows, {h['n_windows_s6']} S6 windows")
    print(f"file sha256 {dig}")
    print(f"lists_digest {out['lists_digest']}")


if __name__ == "__main__":
    main()
