"""Process the Alpamayo-augmented clips at 120-DEGREE geometry, incrementally.

PI 2026-08-14: "pull this, look how much of it was already processed by vlm and
sam3, continue the work but use the index of the 120 degree cameras, push
incrementally the new dataset to hf".

WHAT THIS IS. The Alpamayo augmentation is LABELS (4729 clips x 5 tasks:
trajectory / meta_action / auto_labeling / vqa / grounding_via_vqa), not video.
The video for a clip lives in the w120 corpus on HF as
`<clip_id>.v2ep.pt` -- 256x640 CYLINDRICAL, the camera_front_wide_120fov view.
That is the 120-degree index, and it is what the v6 trunk trains on.

⛔ NOT the 256px epcache. That cache is PINHOLE 256x256 of the same physical
camera; a VLM run on it sees a narrower field and its recall is not comparable.
An earlier attempt used it and is superseded by this.

MEASURED COVERAGE (2026-08-14): 4729 augmented clips; 257 have w120 video; 56
are already VLM+SAM3 done; 201 runnable now; 4472 (94.6%) have no w120 cache at
all and would need building from the PhysicalAI source under a NEW parity key.

DISK DISCIPLINE. Shards are pulled per batch and DELETED after the batch is
bridged, so peak disk is one batch (~35 MB/clip), not 85 GB.

PUSH DISCIPLINE. After every batch the v2 + sam3 JSON go to HF, so an
interrupted pod costs one batch, never the run.
"""
import json, os, shutil, subprocess, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

SRC = "Sayood/tanitad-physicalai-w120-256x640cyl"
DST = os.environ.get("DST_REPO", "Sayood/tanitad-ph0-aug120")
TRAIN = "physicalai-train-e438721ae894-w120-256x640cyl"
VAL = "physicalai-val-0c5f7dac3b11-w120-256x640cyl"
ROOT = "/workspace/aug120"
STACK = "/workspace/TanitAD_head/stack"
PY = "/workspace/a2venv/bin/python"
BATCH = int(os.environ.get("BATCH", "40"))

api = HfApi()
os.makedirs(ROOT, exist_ok=True)

rec = set(pd.read_parquet("/workspace/aug_pack/records.parquet")["clip_id"].unique())
info = api.dataset_info(SRC, files_metadata=True)
loc = {}
for f in info.siblings:
    if f.rfilename.endswith(".v2ep.pt"):
        cid = f.rfilename.split("/")[-1][: -len(".v2ep.pt")]
        loc[cid] = f.rfilename
done = set(json.load(open("/workspace/ph0_prod4/clips.json")))
todo = sorted((rec & set(loc)) - done)
print(f"AUG120 rec={len(rec)} with_w120={len(rec & set(loc))} done={len(rec & done)} "
      f"todo={len(todo)}", flush=True)

api.create_repo(DST, repo_type="dataset", private=True, exist_ok=True)


def run(cmd, log):
    with open(log, "ab") as fh:
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=STACK, env={**os.environ,
                                               "PYTHONPATH": STACK,
                                               "OMP_NUM_THREADS": "6"})


for b0 in range(0, len(todo), BATCH):
    batch = todo[b0:b0 + BATCH]
    tag = f"batch_{b0:05d}"
    cdir = f"{ROOT}/{tag}/corpus"
    os.makedirs(cdir, exist_ok=True)
    # --- pull only this batch's shards + the corpus sidecars ---------------
    seg = loc[batch[0]].split("/")[0]
    for side in ("_geometry.json", "_v2manifest.pt"):
        try:
            shutil.copyfile(hf_hub_download(SRC, f"{seg}/{side}",
                                            repo_type="dataset"),
                            f"{cdir}/{side}")
        except Exception as e:                                  # noqa: BLE001
            print(f"SIDE_MISS {side} {type(e).__name__}", flush=True)
    for cid in batch:
        shutil.copyfile(hf_hub_download(SRC, loc[cid], repo_type="dataset"),
                        f"{cdir}/{cid}.v2ep.pt")
    print(f"{tag} PULLED {len(batch)}", flush=True)

    out = f"{ROOT}/{tag}"
    rc = run([PY, "scripts/v2_to_pilot.py", "--corpus", cdir,
              "--records", "/workspace/aug_pack/records.parquet",
              "--out", out, "--n", str(len(batch))], f"{out}/bridge.log")
    print(f"{tag} BRIDGE_RC={rc}", flush=True)
    shutil.rmtree(cdir, ignore_errors=True)          # bound the disk

    rc = run([PY, "-u", "scripts/ph0_v2.py", "--clips", f"{out}/clips.json",
              "--video-root", f"{out}/videos", "--ego-root", f"{out}/ego",
              "--arm", "Qwen/Qwen3.5-9B", "--n", str(len(batch)), "--resume",
              "--out", f"{out}/v2"], f"{out}/v2.log")
    print(f"{tag} VLM_RC={rc}", flush=True)

    # ⛔ --n is REQUIRED here: ph0_sam3.py defaults to n=4 (ph0_sam3.py:411),
    # and this invocation originally omitted it — every aug120 batch got SAM3
    # on only its first 4 clips while SAM3_RC=0 read as full coverage
    # (MEASURED 2026-08-15: 86/201 distinct clips covered; the fusion marks
    # the other 115 as named partials). Same class as the df/quota trap: a
    # stage that succeeds on the wrong scope looks like an answer.
    rc = run([PY, "-u", "scripts/ph0_sam3.py", "--v2-json",
              f"{out}/v2/ph0_v2.json", "--video-root", f"{out}/videos",
              "--n", str(len(batch)),
              "--out", f"{out}/sam3"], f"{out}/sam3.log")
    print(f"{tag} SAM3_RC={rc}", flush=True)

    for sub in ("v2", "sam3"):
        p = f"{out}/{sub}"
        if os.path.isdir(p):
            api.upload_folder(folder_path=p, path_in_repo=f"{tag}/{sub}",
                              repo_id=DST, repo_type="dataset")
    print(f"{tag} PUSHED", flush=True)
print("AUG120_DONE")
