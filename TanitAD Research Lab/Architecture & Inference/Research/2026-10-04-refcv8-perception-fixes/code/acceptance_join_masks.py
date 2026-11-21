"""WP-C fixes 3 + 4, REAL-DATA acceptance of the join masks (CPU, < 2 GB; ids are sha12 only).

Streams the 3-D agent join ONCE and keeps the records of (a) the 18 TRAIN clips of the ego-footprint list and (b) the 40 clips
with the most track-jump events. Reads that excerpt through ``JoinFileReader`` WITHOUT and WITH ``defect_masks`` and reports:

  fix 3: boxes in the footprint before / after (must be 693 -> 0 over the 18 clips), rows removed, the ego-track leftovers
         (boxes carrying a footprint track id but sitting OUTSIDE the footprint -- the cheap measure of what the rule leaves);
  fix 4: rate rows masked, and the largest |v_rel| among OBSERVED rate rows before / after (the defect is the tail).

Run:  python acceptance_join_masks.py --join <join.jsonl.xz> --masks <refcv8_join_label_defects.json> --out <json>
"""
import argparse
import collections
import json
import lzma
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from tanitad.data import join_label_hygiene as H  # noqa: E402
from train_p8_occupancy import JoinFileReader, episode_uid_of_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-jump-clips", type=int, default=40)
    a = ap.parse_args()
    M = H.JoinDefectMasks.load(a.masks)
    ego_clips = set(M.ego_frames)
    by_n = sorted(M.track_events, key=lambda k: -len(M.track_events[k]))
    jump_clips = set(by_n[: a.n_jump_clips])
    keep = ego_clips | jump_clips
    tmp = Path(tempfile.mkdtemp()) / "excerpt.jsonl"
    n_keep = 0
    clip_of = {}
    with lzma.open(a.join, "rt", encoding="utf-8") as fh, open(tmp, "w", encoding="utf-8") as out:
        for line in fh:
            # cheap prefilter is not possible (the id is inside the JSON) -> parse the clip id only
            i = line.index('"clip_id"')
            cid = line[i:].split('"')[3]
            ck = H.clip_key(cid)
            if ck in keep:
                out.write(line)
                n_keep += 1
                clip_of[ck] = cid
    print("excerpt records", n_keep, "clips", len(clip_of), flush=True)
    res = {"n_records": n_keep, "n_clips": len(clip_of), "ego_clips": len(ego_clips), "jump_clips": len(jump_clips)}
    base = JoinFileReader(tmp, with_rates=True, with_track_ids=True)
    masked = JoinFileReader(tmp, with_rates=True, with_track_ids=True, defect_masks=M)

    def footprint_boxes(rd):
        n = 0
        for (cid, fi), ag in rd._by_clip.items():
            n += int(H.is_ego_footprint(ag[:, 0], ag[:, 1]).sum()) if len(ag) else 0
        return n

    def ego_tracks_outside(rd):
        """boxes whose track id ever sat in the footprint of its clip, found OUTSIDE the footprint (rule leftovers)."""
        tr = collections.defaultdict(set)
        for (cid, fi), ag in base._by_clip.items():
            if H.clip_key(cid) in ego_clips and len(ag):
                m = H.is_ego_footprint(ag[:, 0], ag[:, 1])
                for t in base._tid_by_clip[(cid, fi)][m]:
                    tr[cid].add(t)
        n = 0
        for (cid, fi), ag in rd._by_clip.items():
            if cid in tr and len(ag):
                m = ~H.is_ego_footprint(ag[:, 0], ag[:, 1])
                n += int(sum(t in tr[cid] for t in rd._tid_by_clip[(cid, fi)][m]))
        return n

    def vmax(rd, clips):
        mx, n_obs, big = 0.0, 0, 0
        for (cid, fi), (rt, rm) in rd._rates_by_clip.items():
            if H.clip_key(cid) in clips and rm.any():
                v = np.hypot(rt[rm, 0], rt[rm, 1])
                n_obs += int(rm.sum())
                big += int((v > 100.0).sum())
                mx = max(mx, float(v.max()))
        return {"observed_rate_rows": n_obs, "max_abs_v_rel_m_s": round(mx, 1), "rows_over_100_m_s": big}

    res["fix3"] = {"footprint_boxes_before": footprint_boxes(base), "footprint_boxes_after": footprint_boxes(masked),
                   "rows_removed": masked.defect_masks.n_ego_rows_removed,
                   "boxes_with_a_footprint_track_id_outside_the_footprint_before": ego_tracks_outside(base),
                   "after": ego_tracks_outside(masked)}
    res["fix4"] = {"jump_clips": vmax(base, jump_clips), "jump_clips_masked": vmax(masked, jump_clips),
                   "rate_rows_masked": masked.defect_masks.n_rate_rows_masked,
                   "rate_records_hit": masked.defect_masks.n_rate_records_hit,
                   "events_in_selected_clips": int(sum(len(M.track_events[k]) for k in jump_clips))}
    # alignment control: every aligned array has the same length per record
    bad = 0
    for key, ag in masked._by_clip.items():
        n = len(ag)
        if len(masked._cls_by_clip.get(key, [])) != n or len(masked._tid_by_clip.get(key, [])) != n \
                or masked._rates_by_clip[key][0].shape[0] != n or masked._rates_by_clip[key][1].shape[0] != n:
            bad += 1
    res["alignment_violations"] = bad
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
