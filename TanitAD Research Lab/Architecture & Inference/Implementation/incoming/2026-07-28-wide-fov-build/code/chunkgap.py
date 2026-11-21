import json, sys
from pathlib import Path
import pandas as pd
R = Path(sys.argv[1])
sel = pd.read_parquet(R / "r0" / "r0_selection.parquet")
sel["clip_id"] = sel["clip_id"].astype(str); sel["chunk"] = sel["chunk"].astype(int)
cam = R / "r0" / "camera_front_wide"
have_mp4 = {p.name.split(".")[0] for p in cam.rglob("*.mp4")}
have_ts  = {p.name.split(".")[0] for p in cam.rglob("*.timestamps.parquet")}
sel["has_mp4"] = sel["clip_id"].isin(have_mp4)
sel["has_ts"]  = sel["clip_id"].isin(have_ts)
by = sel.groupby("chunk").agg(n=("clip_id","size"), have=("has_mp4","sum")).reset_index()
missing_chunks = by[by.have < by.n]
ego = R / "labels" / "egomotion"
have_ego = {int(p.name.split("chunk_")[1][:4]) for p in ego.glob("egomotion.chunk_*.zip")}
out = {
 "root": str(R),
 "selection_rows": int(len(sel)),
 "distinct_chunks_in_selection": int(sel.chunk.nunique()),
 "clips_with_mp4": int(sel.has_mp4.sum()),
 "clips_with_timestamps": int(sel.has_ts.sum()),
 "clips_missing_mp4": int((~sel.has_mp4).sum()),
 "clips_missing_timestamps": int((~sel.has_ts).sum()),
 "chunks_fully_present": int((by.have == by.n).sum()),
 "chunks_needing_download": int(len(missing_chunks)),
 "clips_per_chunk_mean": round(float(by.n.mean()), 2),
 "egomotion_zips_present": len(have_ego),
 "egomotion_zips_needed": int(sel.chunk.nunique()),
 "egomotion_chunks_missing": sorted(set(sel.chunk.unique().tolist()) - have_ego),
 "missing_chunk_ids_head": missing_chunks.chunk.tolist()[:20],
}
print("CHUNKGAP_JSON " + json.dumps(out))
