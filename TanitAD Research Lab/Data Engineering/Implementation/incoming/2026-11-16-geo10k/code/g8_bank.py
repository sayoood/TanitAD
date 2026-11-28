"""GEO10K step 8: build the per-clip manifest + file table, upload the label files to the private HF dataset under labels/
(HF_HUB_DISABLE_XET=1, token read in place from the pod's token file, never printed), verify each upload by sha256."""
import glob, gzip, hashlib, json, os, shutil, sys, tempfile
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

OUT = "/root/geo10k/out"
REPO = "Sayood/tanitad-datascale-10k"
sel = [l.strip() for l in open("/root/datascale/selection/clips_10k.txt") if l.strip()]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---- per-clip manifest
z = np.load(f"{OUT}/v9_labels_new.npz", allow_pickle=False)
sid = {int(s): i for i, s in enumerate(z["clip__sid"])}
lead = np.load(f"{OUT}/lead_new10k.npz", allow_pickle=True)
lead_n = {}
for s in lead["sha12"]:
    lead_n[str(s)] = lead_n.get(str(s), 0) + 1
speed = {json.loads(l)["clip_id"]: json.loads(l) for l in open(f"{OUT}/speed_sidecar_new10k.jsonl")}
import hashlib as _h
rows = []
for c in sel:
    s = int.from_bytes(_h.blake2b(c.encode(), digest_size=8).digest(), "big") >> 1
    i = sid[s]
    s12 = _h.sha256(c.encode()).hexdigest()[:12]
    rows.append({"clip": c, "sid": s, "v9_row0": int(z["clip__row0"][i]), "v9_n_rows": int(z["clip__n_rows"][i]), "clock_src": int(z["clip__clock_src"][i]),
                 "lead_rows": lead_n.get(s12, 0), "speed_valid": int(speed[c]["valid"]), "speed_limit_kmh": speed[c]["limit_kmh"]})
with gzip.open(f"{OUT}/geo10k_clip_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

FILES = {
    "labels/v9_labels_datascale10k.npz": f"{OUT}/v9_labels_new.npz",
    "labels/v9_labels_datascale10k.manifest.json": f"{OUT}/v9_labels_new.manifest.json",
    "labels/lead_datascale10k.npz": f"{OUT}/lead_new10k.npz",
    "labels/speed_sidecar_datascale10k.jsonl": f"{OUT}/speed_sidecar_new10k.jsonl",
    "labels/native_tracks_datascale10k.pt": f"{OUT}/native_tracks_new10k.pt",
    "labels/clip_clock_datascale10k.jsonl": f"{OUT}/clock_new10k.jsonl",
    "labels/v2manifest_poses_datascale10k.pt": f"{OUT}/new10k_v2manifest.pt",
    "labels/context/clips_datascale10k.parquet": f"{OUT}/clips_datascale10k.parquet",
    "labels/geo10k_clip_manifest.jsonl.gz": f"{OUT}/geo10k_clip_manifest.jsonl.gz",
}
for p in sorted(glob.glob(f"{OUT}/controls/*")) + sorted(glob.glob(f"{OUT}/docs/*")):
    FILES[("labels/controls/" if "/controls/" in p else "labels/") + os.path.basename(p)] = p
table = {k: {"sha256": sha256(v), "bytes": os.path.getsize(v), "pod_path": v} for k, v in FILES.items()}
json.dump(table, open(f"{OUT}/geo10k_files.json", "w"), indent=1)
FILES["labels/geo10k_files.json"] = f"{OUT}/geo10k_files.json"

tok = open(glob.glob("/root/.hf_*")[0]).read().strip()
api = HfApi(token=tok)
bad = []
for repo_path, local in FILES.items():
    want = sha256(local)
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=repo_path, repo_id=REPO, repo_type="dataset", commit_message=f"GEO10K labels: {os.path.basename(repo_path)}")
            break
        except Exception as e:                                      # noqa: BLE001
            print("upload retry", repo_path, type(e).__name__, str(e)[:100], flush=True)
    info = api.get_paths_info(REPO, [repo_path], repo_type="dataset")[0]
    got = getattr(getattr(info, "lfs", None), "sha256", None)
    if got is None:                                                   # small file stored as a plain blob: read it back and hash it
        d = tempfile.mkdtemp(prefix="geo10k_dl_", dir="/root/geo10k")
        p = hf_hub_download(REPO, repo_path, repo_type="dataset", token=tok, local_dir=d)
        got = sha256(p)
        shutil.rmtree(d, ignore_errors=True)
    ok = got == want
    print(f"[bank] {'OK ' if ok else 'BAD'} {repo_path} {os.path.getsize(local)} B sha256 {want[:12]}", flush=True)
    if not ok:
        bad.append(repo_path)
print("ZZ GEO10K DONE ZZ" if not bad else f"ZZ GEO10K FAIL ZZ bank mismatch {bad}")
