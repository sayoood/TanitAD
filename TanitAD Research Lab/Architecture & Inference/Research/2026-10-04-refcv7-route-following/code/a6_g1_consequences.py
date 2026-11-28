"""A6 (STOPPED at G1): WINDOW COUNTS ONLY -- what a faithful announced() would change on A5's EVAL grid.

No arm is scored here and no ΔADE is computed: SPEC_ADDENDUM_A6 forbids scoring before G1 passes, and G1 did not. This
script only counts windows, so the Master Mind can see what the G1 failure means for the arms before amending A6.

Reads raw/a6_announced_table.json (written by announced_a6.py --stage table, which imports the builder) and A5's inputs.
Plain process: NO tanitad import.   python a6_g1_consequences.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_route as A  # noqa: E402
import nav_tl_a5 as N  # noqa: E402

RAW = HERE.parent / "raw"


def nav_ann_state(entries, flags, t_rel, H=N.H_DEFAULT):
    """A6: A5's nav_tl rule restricted to ANNOUNCED entries; a non-announced entry is skipped (the next announced one is
    considered)."""
    return N.nav_tl_state([e for e, f in zip(entries, flags) if f], t_rel, H)


def main():
    tab = json.loads((RAW / "a6_announced_table.json").read_text(encoding="utf-8"))
    recs = N.load_records("D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
    # per-record cause (needs reason strings + turn_suppression, which load_records drops): re-read minimal fields
    cause, contested_t = {}, {}
    with gzip.open("D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            s = N.sha12(r["clip_id"])
            e0 = r["nav_30s"]["entries"][0]["token"]
            nc = r["nav_command"]
            sup = bool((r.get("turn_suppression") or {}).get("applied"))
            if sup:
                cause[s] = "contested (turn_suppression applied)"
                contested_t[s] = (float(r["turn_suppression"]["t_start_s"]), r["turn_suppression"]["side"])
            elif "curve" in str(nc.get("reason")) and e0 != "NAV_FOLLOW_ROAD":
                cause[s] = "curve-first (nav_command saw a curve at seq[0]; entries[0] is a later real turn)"
            elif nc["token"] != "NAV_FOLLOW_ROAD" and e0 == "NAV_FOLLOW_ROAD":
                cause[s] = "turn beyond the 30 s nav_30s cap"
            else:
                cause[s] = "agree"
    sidecar = N.read_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
    z = N.load_pack("D:/refcv7_route_bin/2026-10-04", RAW, "eval_s0g")
    d = A.derive(z)
    t_rel, _ = N.window_clock(z, recs, sidecar)
    ws = np.asarray(z["win_sha12"]).astype(str)
    side_tl = np.zeros(len(ws), int)
    side_ann = np.zeros(len(ws), int)
    side_ann_alt = np.zeros(len(ws), int)       # alternative: suppress ONLY the contested turn (matching t_start), not the whole clip
    clip = np.array([recs[s]["token_side"] for s in ws])
    for i, s in enumerate(ws):
        ents = recs[s]["entries"]
        side_tl[i] = N.nav_tl_state(ents, float(t_rel[i]))[0]
        side_ann[i] = nav_ann_state(ents, tab["eval"][s], float(t_rel[i]))[0]
        if s in contested_t:
            fl = [not (e["token"] in N.TOKEN_SIDE and e["t_start_s"] == contested_t[s][0]) for e in ents]
        else:
            fl = tab["eval"][s]
        side_ann_alt[i] = nav_ann_state(ents, fl, float(t_rel[i]))[0]
    M = {"GT-turn": A.cls_mask(d, "turn"), "GT-straight": A.cls_mask(d, "straight"), "gentle": A.cls_mask(d, "gentle"),
         "unclassified": d["cls"] == "unclassified", "every window": np.ones(len(ws), bool)}
    out = {"label": "WINDOW COUNTS ONLY; no arm scored (A6 G1 failed)",
           "eval_records_by_cause": {c: int(sum(1 for s in set(ws) if cause[s] == c)) for c in sorted(set(cause[s] for s in set(ws)))},
           "A5_EVAL_grid_windows": {}}
    for nm, m in M.items():
        extra = m & (clip == 0) & (side_tl != 0)          # A5's "clip token FOLLOW but nav_tl active" windows
        out["A5_EVAL_grid_windows"][nm] = {
            "n": int(m.sum()), "nav_tl_active": int((m & (side_tl != 0)).sum()),
            "nav_ann_active_(builder mirror, clip-level suppression)": int((m & (side_ann != 0)).sum()),
            "nav_ann_alt_active_(only the contested turn suppressed)": int((m & (side_ann_alt != 0)).sum()),
            "windows_where_nav_ann_differs_from_nav_tl": int((m & (side_ann != side_tl)).sum()),
            "windows_where_alt_differs_from_nav_tl": int((m & (side_ann_alt != side_tl)).sum()),
            "clip_FOLLOW_and_nav_tl_active": int(extra.sum()),
            "...of_which_by_cause": {c: int((extra & np.array([cause[s] == c for s in ws])).sum())
                                     for c in sorted(set(cause.values())) if (extra & np.array([cause[s] == c for s in ws])).any()},
            "...of_which_still_active_under_nav_ann": int((extra & (side_ann != 0)).sum())}
    Path(RAW / "a6_g1_consequences.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
