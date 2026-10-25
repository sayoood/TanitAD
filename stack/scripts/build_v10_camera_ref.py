"""v10 `camera_ref`: where each v9 clip's seven cameras live UPSTREAM (nvidia/PhysicalAI-Autonomous-Vehicles), by reference.

The programme corpus ships only `camera_front_wide_120fov`. Elastic sensing (paper sec:elastic) needs the other six cameras; v10 does not
copy them, it records for every v9 clip the upstream chunk, the chunk path of each camera and whether the camera is present, so a pod can
pull exactly the chunks it needs (range-read the `{clip_id}.{camera}.mp4` member, as `tools/pull_camera.py` does for the front camera).

Inputs (downloaded with the HF token read in place, never printed): upstream `clip_index.parquet` (clip_id -> chunk),
`metadata/feature_presence.parquet` (clip_id x 36 features, bool), `features.csv` (chunk path templates); corpus
`index/clip_index_v7.2_{train,eval}.json` (UUIDs) and the v9 label npz (`clip__sha12`) for the split and row order.
Optional `--sizes`: list the upstream camera directories once to report the bytes of the chunks a full pull needs.

    python stack/scripts/build_v10_camera_ref.py --probe /tmp/claude-0/probe --v9 /tmp/claude-0/hfdata --out DataEng/v10/camera_ref [--sizes]
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, urllib.request
import numpy as np
import pandas as pd

CAMERAS = ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov", "camera_rear_left_70fov",
           "camera_rear_right_70fov", "camera_rear_tele_30fov", "camera_front_tele_30fov"]
UPSTREAM = "nvidia/PhysicalAI-Autonomous-Vehicles"
KEYS = "/root/.claude/uploads/47a1cfa2-478d-5914-8f27-fb9c6388aa94/34144268-Keys.txt"


def sha12(u: str) -> str: return hashlib.sha256(u.encode()).hexdigest()[:12]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def hf_tree(path: str) -> list[dict]:
    """Paginated HF tree listing; the token is read in place and only placed in a request header."""
    tok = re.search(r"hf_[A-Za-z0-9]+", open(KEYS).read()).group(0)
    url, out = f"https://huggingface.co/api/datasets/{UPSTREAM}/tree/main/{path}", []
    while url:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + tok}), timeout=120)
        out += json.loads(r.read())
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = m.group(1) if m else None
    return out


def build(probe: str, v9: str, out: str, sizes: bool = False) -> dict:
    ci = pd.read_parquet(os.path.join(probe, "clip_index.parquet"))
    fp = pd.read_parquet(os.path.join(probe, "feature_presence.parquet"))
    tmpl = pd.read_csv(os.path.join(probe, "features.csv")).set_index("feature")["chunk_path"].to_dict()
    uu = {}
    for f in ("clip_index_v7.2_train.json", "clip_index_v7.2_eval.json"):
        for u in json.load(open(os.path.join(probe, f)))["clips"]: uu[sha12(u)] = u
    rows = []
    for split, fn in (("train", "v9_labels_train.npz"), ("eval139", "v9_labels_eval139.npz")):
        z = np.load(os.path.join(v9, fn), allow_pickle=True)
        for s in (str(x) for x in z["clip__sha12"]):
            u = uu.get(s)
            r = {"split": split, "sha12": s, "clip_id": u, "chunk": int(ci.loc[u, "chunk"]) if u in ci.index else -1}
            for cam in CAMERAS:
                present = bool(fp.loc[u, cam]) if u in fp.index else False
                r[f"{cam}__present"] = present
                r[f"{cam}__path"] = tmpl[cam].format(chunk_id=r["chunk"]) if present and r["chunk"] >= 0 else None
            rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    pq = os.path.join(out, "v10_camera_ref.parquet"); df.to_parquet(pq, index=False)
    man = {"schema": "tanitad.v10_camera_ref/1", "upstream": UPSTREAM, "member_name": "{clip_id}.{camera}.mp4",
           "n_clips": {k: int(v) for k, v in df.split.value_counts().items()}, "unmapped_clips": int((df.chunk < 0).sum()),
           "n_chunks": int(df.chunk[df.chunk >= 0].nunique()),
           "camera_present": {c: int(df[f"{c}__present"].sum()) for c in CAMERAS},
           "all_seven_present": int(df[[f"{c}__present" for c in CAMERAS]].all(axis=1).sum()),
           "inputs_sha256": {f: sha256_file(os.path.join(probe, f)) for f in
                             ("clip_index.parquet", "feature_presence.parquet", "features.csv")},
           "output_sha256": sha256_file(pq)}
    if sizes:
        need = set(df.chunk[df.chunk >= 0]); ours = df.chunk[df.chunk >= 0].value_counts().to_dict()
        man["pull_bytes"] = {}
        for cam in CAMERAS[1:]:
            listing = hf_tree(f"camera/{cam}")
            by = {int(re.search(r"chunk_(\d+)", e["path"]).group(1)): e.get("size", 0) for e in listing if "chunk_" in e["path"]}
            man["pull_bytes"][cam] = {"chunks": len(need & set(by)), "whole_chunk_bytes": int(sum(by.get(c, 0) for c in need)),
                                      "member_bytes_estimated": int(sum(by.get(c, 0) * ours[c] / 100 for c in need))}
        man["whole_chunk_bytes_six_cameras"] = int(sum(v["whole_chunk_bytes"] for v in man["pull_bytes"].values()))
        man["member_bytes_six_cameras_ESTIMATED"] = int(sum(v["member_bytes_estimated"] for v in man["pull_bytes"].values()))
        man["member_bytes_note"] = "ESTIMATED: chunk size x (our clips in the chunk / 100), assuming 100 equal-size clips per chunk; range-read members, never whole chunks"
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True); ap.add_argument("--v9", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build(a.probe, a.v9, a.out, a.sizes), indent=1))
