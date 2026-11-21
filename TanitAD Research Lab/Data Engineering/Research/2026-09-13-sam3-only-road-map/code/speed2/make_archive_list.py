"""Explicit archive list for the Thor cleanup (PI 2026-09-14) from disk_inventory.tsv. ARCHIVE = moved to the dev box
(D:/thor_archive/2026-09-14/, md5-verified) and only then removed from Thor. Rules:
  outside sam3map: whole level-1 / level-2 directories untouched for >= 21 days that no cron job, unit, live process or recent
  script uses (listed by name below, each checked), never the training caches in data/, never venvs/.cache/TanitAD.
  sam3map: every directory >= 50 MB that is NOT on the keep list (inputs, map r's lineage, the reference / production speed arms,
  anything the running validation chain reads or writes).
Output: archive_list.tsv (path, bytes, reason) + keep_list.txt"""
import re
from pathlib import Path

GB = 1024 ** 3
rows = [l.rstrip("\n").split("\t") for l in open("disk_inventory.tsv", encoding="utf-8")]
inv = {p: (int(b), int(n), int(a), int(r)) for p, b, n, a, r in rows}
H = "/home/nvidia"
OUTSIDE = {  # path: reason (all checked: newest file >= 21 d, not referenced by cron / units / live processes)
    f"{H}/epcache_prefix": "episode-cache prefix of the parity corpus, 41 d untouched; the canonical cyl cache stays in data/",
    f"{H}/backup/v1arch-v2bal": "v1arch-v2bal backup, 42 d (the small refav1-ship-20260902 code snapshot beside it stays)",
    f"{H}/ckpt_snaps": "v6F fp16 snapshots, 23-28 d",
    f"{H}/valdata": "val raw + small cyl copy, 41 d (thor_phase.sh, 22 d old, reads it); canonical val cache stays in data/",
    f"{H}/nurec_scenes": "NuRec sample scenes, 42 d",
    f"{H}/nurec_work": "NuRec work dir, 42 d",
    f"{H}/alpasim": "AlpaSim local files, 42 d",
    f"{H}/rq_out": "42 d",
    f"{H}/trt": "TensorRT engines, 43 d (regenerable from ONNX)",
    f"{H}/trt_c2": "TensorRT engines, 42 d", f"{H}/trt_d1": "TensorRT engines, 42 d", f"{H}/trt_b1": "TensorRT engines, 42 d",
    f"{H}/trt_deploy": "TensorRT engines, 42 d", f"{H}/trt_b1b": "TensorRT engines, 42 d", f"{H}/trt_c2b": "TensorRT engines, 42 d",
    f"{H}/trt_c5": "TensorRT engines, 42 d",
    f"{H}/experiments/v6F-SW-30k": "finished v6F run, 23 d",
    f"{H}/models/flagship-v1-speedjerk": "checkpoint, 42 d (also on HF)", f"{H}/models/v5f": "checkpoint, 42 d",
    f"{H}/models/flagship-v4.2b": "checkpoint, 42 d", f"{H}/models/rollout-recovery": "checkpoint, 43 d (pull_weights.sh, 43 d old, downloads it)",
    f"{H}/models/refc-xl": "checkpoint, 43 d", f"{H}/models/refc-base": "checkpoint, 43 d", f"{H}/models/refc-base-e1b-clsft": "checkpoint, 43 d",
}
C8S = ["73495082f98b", "4fbd97b6a4b7"]
KEEP_SAM3 = {"native7", "front", "front_native", "data", "eval", "spots", "long_fast", "long_frames"}
for c8 in C8S:
    KEEP_SAM3 |= {f"{c8}_{t}" for t in ("v6raw", "v6sraw", "v6s", "v61s", "v6sfraw", "v6sf", "v61sf")}
    KEEP_SAM3 |= {f"render5_{c8}_{t}" for t in ("v65ma", "v65mf", "v65ra", "v65rf", "v65d0a", "v65d0f")}
    for arm in ("spd0", "spdF4a", "spdF5a"):
        KEEP_SAM3 |= {f"{c8}_{arm}raw", f"{c8}_{arm}", f"{c8}_{arm}c", f"render5_{c8}_{arm}m", f"render5_{c8}_{arm}r"}
VALID = re.compile(r"(spdS[A45]|native7|front_native|long_)")
arch, keep = [], []
for p, (b, n, a, r) in inv.items():
    if p in OUTSIDE:
        assert a >= 21 and r == 0, (p, a, r)
        arch.append((p, b, OUTSIDE[p]))
for p, (b, n, a, r) in sorted(inv.items()):
    if not p.startswith(f"{H}/sam3map/") or p.count("/") != 4:
        continue
    name = p.split("/")[-1]
    if not Path(name).suffix == "" or b < 50 * 1024 ** 2:
        continue
    if name in KEEP_SAM3 or VALID.search(name):
        keep.append(p); continue
    arch.append((p, b, "superseded SAM3-map iteration / intermediate (results banked in the repo; regenerable from native7 + code)"))
with open("archive_list.tsv", "w", encoding="utf-8", newline="\n") as f:
    for p, b, why in sorted(arch, key=lambda x: -x[1]):
        f.write(f"{p}\t{b}\t{why}\n")
Path("keep_list.txt").write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
out_b = sum(b for p, b, _ in arch if not p.startswith(f"{H}/sam3map/")); in_b = sum(b for p, b, _ in arch if p.startswith(f"{H}/sam3map/"))
print(f"archive: {len(arch)} directories, {(out_b + in_b) / GB:.1f} GB (outside sam3map {out_b / GB:.1f} GB, sam3map {in_b / GB:.1f} GB); sam3map kept (>= 50 MB): {len(keep)}")
