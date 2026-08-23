"""Post-run check: is encode-once a REFACTOR, and how much faster is it?

Re-runs ONE clip that the 115-clip backfill already banked (produced by the
per-concept-encode path) with the encode-once path, on the same T4, and
compares the two records FIELD BY FIELD. Anything but an exact match on the
detection payload means the optimisation changed the science and must be
reverted.
"""
import copy
import json
import os
import sys
import tarfile
import time

os.chdir("/content")
os.makedirs("/content/repo", exist_ok=True)
with tarfile.open("/content/repo_closure.tgz") as tf:
    tf.extractall("/content/repo")
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import importlib                                                 # noqa: E402
for m in ("ph0_sam3", "s2_lab_lib"):
    if m in sys.modules:
        importlib.reload(sys.modules[m])
import ph0_sam3                                                  # noqa: E402
import s2_lab_lib as L                                           # noqa: E402
import ph0_pilot                                                 # noqa: E402
import torch                                                     # noqa: E402

src = open("/content/repo/stack/scripts/ph0_sam3.py", "rb").read()
assert b"def detect_many" in src, "encode-once code did NOT arrive"
print("[chk] detect_many present; run_clip_frames uses it:",
      b"detect_many(processor, img, concepts" in src)

CID = json.load(open("/content/repo/colab/fixtures/"
                     "sam3_backfill_expected.json"))["clips"][0]
old = json.load(open(L.hf_download(L.DS_LABELS,
                                   f"{L.BACKFILL_PREFIX}{CID}.json",
                                   force=True)))
print(f"[chk] banked (per-concept encode): {CID[:8]} "
      f"n_frames={old.get('n_frames_run')} det={old.get('n_det_total')} "
      f"wall_s={old.get('wall_s')}")

import shutil                                                    # noqa: E402
mp4 = "/content/eqchk.mp4"
shutil.copyfile(L.hf_download(
    L.DS_LABELS, f"bridged_w120train_2400/videos/{CID}.mp4"), mp4)
frames = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0)[0]
v2 = L.load_v2_records(L.hf_api(), {CID})[CID]

if "PROC" not in globals():
    PROC, _meta = L.load_sam3()
    assert _meta["dtype_fix"]["applied"], "C77 dtype fix did NOT install"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
new = L.sam3_leg(PROC, frames, v2, frame_stride=8)               # noqa: F821
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"[chk] re-run (encode-once): n_frames={new['n_frames_run']} "
      f"det={new['n_det_total']} wall_s={new['wall_s']} peak={peak:.2f} GB")

a = copy.deepcopy(old)
b = copy.deepcopy(new)
for d in (a, b):
    for k in ("wall_s", "_n_explicit"):
        d.pop(k, None)
    (d.get("liveness") or {}).pop("wall_s", None)
same = a == b
print("[chk] RECORDS IDENTICAL:", same)
if not same:
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            print("   DIFF key:", k)
            if k == "frames":
                for fk in sorted(set(a[k]) | set(b[k])):
                    if a[k].get(fk) != b[k].get(fk):
                        print("     frame", fk,
                              "n_det", a[k].get(fk, {}).get("n_det"),
                              "->", b[k].get(fk, {}).get("n_det"))

out = {"clip_id": CID,
       "old_path": "detect() per concept (image encoded once per CONCEPT)",
       "new_path": "detect_many() (image encoded once per FRAME)",
       "n_frames_run": new["n_frames_run"],
       "n_concepts": len(ph0_sam3.AGENT_CONCEPTS),
       "encodes_old": new["n_frames_run"] * len(ph0_sam3.AGENT_CONCEPTS)
       + len(ph0_sam3.LIVENESS_CONCEPTS),
       "encodes_new": new["n_frames_run"] + 1,
       "wall_s_old": old.get("wall_s"), "wall_s_new": new.get("wall_s"),
       "speedup": (round(old["wall_s"] / new["wall_s"], 2)
                   if new.get("wall_s") else None),
       "peak_gb_new": round(peak, 2),
       "records_identical_excluding_wall_s": same,
       "n_det_total_old": old.get("n_det_total"),
       "n_det_total_new": new.get("n_det_total"),
       "per_concept_old": old.get("per_concept_hits"),
       "per_concept_new": new.get("per_concept_hits"),
       "liveness_old": old.get("liveness"), "liveness_new": new.get("liveness"),
       "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "evidence_class": "MEASURED"}
json.dump(out, open("/content/encode_once_equivalence.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("EQCHK_DONE")
