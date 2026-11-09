"""Session key (chunk merged with continuation matches), clip table, CTX-HO definition. CPU only, reads local files.

Continuation matches: clip B's first 20 s speed profile (0.2 s grid, std >= 1 m/s) against every 20 s window of clip A's egomotion
that starts at >= 20 s, within a chunk, RMSE < 0.05 m/s (the audit rule, AUDIT_DATA 1.5). Tested only for clips with a local
egomotion parquet (the 4,800 alpamayo clips); cross-chunk continuation is not tested (stated in the report).
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C

OUT = os.environ.get("STRAT_OUT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels")
TRAIN_JSON = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/raw/TRAIN_CLIPS_4512.json"
EVAL_GZ = "E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v72/s2_labels_v7.2_eval.jsonl.gz"


def speed_5hz(clip_id):
    poses, _ = C.load_track(clip_id)
    return poses[::2, 3]                      # 10 Hz track -> 0.2 s grid


def continuation_edges(by_chunk_local):
    edges = []
    for ch, clips in by_chunk_local.items():
        if len(clips) < 2:
            continue
        sp = {c: speed_5hz(c) for c in clips}
        for b in clips:
            sb = sp[b][:100]
            if len(sb) < 100 or sb.std() < 1.0:
                continue
            for a in clips:
                if a == b or len(sp[a]) < 200:
                    continue
                w = np.lib.stride_tricks.sliding_window_view(sp[a][100:], 100)      # windows starting at >= 20 s
                rmse = np.sqrt(((w - sb[None]) ** 2).mean(1))
                if rmse.min() < 0.05:
                    edges.append((a, b))
    return edges


def main():
    ci = pd.read_parquet(C.CLIP_INDEX)
    ci = ci.reset_index() if "clip_id" not in ci.columns else ci
    meta = pd.read_parquet(C.META)
    meta = (meta.reset_index() if "clip_id" not in meta.columns else meta).set_index("clip_id")
    train = json.load(open(TRAIN_JSON))["train_clips_4512"]
    ev = [json.loads(l)["clip_id"] for l in gzip.open(EVAL_GZ, "rt")]
    ci = ci[ci.clip_is_valid].copy()
    by_chunk = {int(ch): g.clip_id.tolist() for ch, g in ci.groupby("chunk")}
    local = {os.path.basename(p)[:-8] for p in glob.glob(C.EGO_DIR + "/*.parquet")}
    by_chunk_local = {ch: [c for c in cl if c in local] for ch, cl in by_chunk.items()}
    by_chunk_local = {ch: v for ch, v in by_chunk_local.items() if len(v) > 1}
    edges = continuation_edges(by_chunk_local)
    skey, chunk_of = C.session_groups(by_chunk, edges)
    train_groups = {skey[c] for c in train if c in skey}
    train_chunks = {chunk_of[c] for c in train if c in chunk_of}
    cands = [{"clip_id": c, "country": meta.at[c, "country"], "chunk": chunk_of[c], "session_key": skey[c]}
             for c in ci.clip_id if c in meta.index]
    ho = C.pick_ctx_ho(cands, train_groups, per_country=40, max_per_chunk=2)
    split_of = {c: "train4512" for c in train}
    split_of.update({c: "eval" for c in ev if c not in split_of})
    split_of.update({c: "ctx_ho" for c in ho})
    rows = []
    for c, sp in split_of.items():
        rows.append({"clip_id": c, "split": sp, "chunk": int(chunk_of.get(c, -1)), "session_key": skey.get(c, "NA"),
                     "country": meta.at[c, "country"] if c in meta.index else "NA",
                     "month": int(meta.at[c, "month"]) if c in meta.index else -1,
                     "hour_of_day": int(meta.at[c, "hour_of_day"]) if c in meta.index else -1,
                     "video_local": os.path.exists(f"{C.CAM_DIR}/{c}.mp4"), "ego_local": c in local})
    tab = pd.DataFrame(rows)
    os.makedirs(OUT + "/splits", exist_ok=True)
    tab.to_parquet(OUT + "/splits/clips_table.parquet", index=False)
    json.dump({"ctx_ho": ho}, open(OUT + "/splits/CTX_HO_clips.json", "w"))
    ho_t = tab[tab.split == "ctx_ho"]
    rep = {"label_version": C.LABEL_VERSION, "n_train": len(train), "n_eval": len(ev), "n_ctx_ho": len(ho),
           "ctx_ho_per_country": ho_t.country.value_counts().to_dict(),
           "ctx_ho_video_local": int(ho_t.video_local.sum()),
           "n_continuation_edges_local": len(edges), "n_sessions_total": len(set(skey.values())),
           "n_train_chunks": len(train_chunks), "n_train_sessions": len(train_groups),
           "eval_in_train_session": int(tab[(tab.split == "eval")].session_key.isin(train_groups).sum()),
           "ctx_ho_in_train_session": int(ho_t.session_key.isin(train_groups).sum()),
           "ctx_ho_chunk_overlap_train": int(ho_t.chunk.isin(train_chunks).sum()),
           "train_eval_overlap_clips": len(set(train) & set(ev)),
           "note": "continuation matches tested within chunk for clips with a local egomotion parquet only; cross-chunk not tested",
           "evidence_class": "MEASURED (this script)"}
    json.dump(rep, open(OUT + "/splits/SPLITS_REPORT.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
