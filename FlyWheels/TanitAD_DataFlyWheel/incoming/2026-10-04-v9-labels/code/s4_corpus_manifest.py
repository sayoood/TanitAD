"""WP-A Stage 4 item 6 — the refcv8 corpus manifest: train clips to DROP and agent-box frames to MASK (from D3).

DROP (train only; eval139 is unchanged so refcv8 stays comparable to refcv7):
  * the train RECORDING PARTNERS of eval clips (D3 C1): tier A = image-confirmed (score < 0.01), tier B = the looser
    tier (< 0.1; it includes f0a6ab615da8, whose recording ends 0.1 s before an eval clip starts, image NCC 0.998);
    DEFAULT = tier B (all 6 train partners);
  * one clip of each duplicate-video pair (D3: two pairs; rule fixed here: keep the lexicographically SMALLER sha12).
MASK: every agent / box3d box whose centre lies in the ego footprint, D3's own rule (rig frame, rear-axle origin:
x in [-1, 4] m and |y| < 1 m) — listed per (sha12, raw frame k, track_id) from the join itself.
Ids are sha12 only.

usage: python s4_corpus_manifest.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import lzma
import sys

import torch

D3 = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D3_raw_plausibility/raw"
TRAIN_MAN = "D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt"
EVAL_MAN = "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"
JOIN = "D:/refcv6_eval_kit/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
DUP_PAIRS = (("26f668b218f6", "a8f0a8b2476a"), ("1aca93cd9155", "0d6f01ac5244"))   # D3 RESULT D-1


def sha12(c):
    return hashlib.sha256(str(c).encode("utf-8")).hexdigest()[:12]


def partners(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for p in d["eval_train_leak_pairs"]:
        tr = p["window_clip"] if p["window_split"] == "train" else p["log_clip"]
        ev = p["log_clip"] if p["window_split"] == "train" else p["window_clip"]
        out.append({"train_sha12": tr, "eval_sha12": ev, "score": p["score"], "tau_s": p["tau_s"],
                    "gap_between_clips_s": p["gap_between_clips_s"]})
    return d["components"]["thr_exact"], out


def main(out_p):
    tr_ids = [str(c) for c in torch.load(TRAIN_MAN, map_location="cpu", weights_only=False)["clip_id"]]
    ev_ids = [str(c) for c in torch.load(EVAL_MAN, map_location="cpu", weights_only=False)["clip_id"]]
    tr_sha, ev_sha = {sha12(c): c for c in tr_ids}, {sha12(c): c for c in ev_ids}
    thrA, A = partners(f"{D3}/c1c_verify_result_thr0.01.json")
    thrB, Bp = partners(f"{D3}/c1c_verify_result.json")
    tierA = sorted({p["train_sha12"] for p in A})
    tierB = sorted({p["train_sha12"] for p in Bp})
    dups = [{"keep": min(a, b), "drop": max(a, b)} for a, b in DUP_PAIRS]
    for s in tierB + [d["drop"] for d in dups] + [d["keep"] for d in dups]:
        assert s in tr_sha, f"{s} is not a train clip"
    for p in Bp:
        assert p["eval_sha12"] in ev_sha, f"{p['eval_sha12']} is not an eval139 clip"
    drop_default = sorted(set(tierB) | {d["drop"] for d in dups})
    # ---- the ego-footprint mask, from the join itself (D3's rule) --------------------------------------------- #
    mask, n_lines = [], 0
    with lzma.open(JOIN, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            i = line.find('"clip_id":')
            q0 = line.index('"', i + 10)
            cid = line[q0 + 1:line.index('"', q0 + 1)]
            if '"cx"' not in line:
                continue
            r = json.loads(line)
            for a in r["agents"]:
                if -1.0 <= float(a["cx"]) <= 4.0 and abs(float(a["cy"])) < 1.0:
                    mask.append({"sha12": sha12(cid), "split": "train" if cid in set_tr else ("eval139" if cid in set_ev else "other"),
                                 "k": int(r["frame"]), "track_id": str(a.get("track_id")), "cls": a.get("cls"),
                                 "cx": round(float(a["cx"]), 3), "cy": round(float(a["cy"]), 3)})
    clips = sorted({m["sha12"] for m in mask})
    out = {"schema": "tanitad.refcv8_corpus_manifest/1", "source": "refcv8 data audit D3 (C1 recording overlap, C3b agents)",
           "train_clips_in": len(tr_ids), "eval139_clips_in": len(ev_ids),
           "decision": "PI decision 8 — Master Mind default: KEEP all 4,369 train clips and apply the MASK only; "
                       "the DROP LIST is shipped for the alternative",
           "drop_list": drop_default, "n_drop_list": len(drop_default),
           "train_clips_if_dropped": len(tr_ids) - len(drop_default),
           "drop_reasons": {"recording_partner_of_eval_tierB_score_lt_%s" % thrB: tierB,
                            "recording_partner_of_eval_tierA_image_confirmed_score_lt_%s" % thrA: tierA,
                            "duplicate_video_pair_drop_larger_sha12": dups},
           "drop_list_tierA_only": sorted(set(tierA) | {d["drop"] for d in dups}),
           "partner_pairs_tierB": Bp,
           "eval139": "UNCHANGED (comparability with refcv7)",
           "mask_rule": "agent / box3d box centre in the ego footprint: x in [-1, 4] m and |y| < 1 m (rig frame, rear-axle origin) — D3 c3b",
           "mask_boxes": mask, "n_mask_boxes": len(mask), "n_mask_frames": len({(m['sha12'], m['k']) for m in mask}),
           "n_mask_clips": len(clips), "mask_clips": clips, "join_lines_read": n_lines,
           "v9_labels_note": "build_v9_labels.py already refuses an ego-footprint box as a FOLLOW lead (wider box -1.5..4.5 m, |y| < 1.2 m)"}
    json.dump(out, open(out_p, "w", encoding="utf-8"), indent=1)
    print({k: v for k, v in out.items() if k not in ("mask_boxes", "partner_pairs_tierB")})


if __name__ == "__main__":
    set_tr = set(str(c) for c in torch.load(TRAIN_MAN, map_location="cpu", weights_only=False)["clip_id"])
    set_ev = set(str(c) for c in torch.load(EVAL_MAN, map_location="cpu", weights_only=False)["clip_id"])
    main(sys.argv[1])
