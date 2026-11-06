"""Builds OVERLAY 2 of the stage-2 package (AGENT J): the reference frames stage 2 forgot to ship, plus the fail-closed driver. Dev box, CPU. Nothing is committed, staged or overwritten.

    python -I make_overlay2.py build --overlay1 stage2/t2v8f_stage2_overlay.tar --refs stage2/ref_frames/REF_FRAMES_SOURCES.json --out-dir stage2/overlay2 [--tag 2]

Overlay 2 is DERIVED from overlay 1 (whose 24 GB of scene hashes were verified at its build): MANIFEST2 v2 = MANIFEST2 v1 + a `ref_frames` entry per scene (size, sha256, the arrays and
their shapes, a content sha256 over the frame bytes) + the new driver's size and sha256. The tar holds ONLY what changes on box 2, extracted with `tar -xf ... -C D:\\t2v8f`:
    pkg/MANIFEST2.json                 (replaces v1)
    pkg/code/t2v8f_stage2.py           (replaces v1: the fail-closed preflight)
    scenes2/<id>/ref_frames.npz        (41 new files; the kit scene 7c72937c keeps the stage-1 npz it already has, which is only hashed)
The mp4 files are NOT shipped. REF_FRAMES_SOURCES.json is written by `extract_ref_frames.py`: {scene: {path, ...}} for all 42 scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
T2 = os.path.dirname(HERE)
import make_stage2 as MK  # noqa: E402  (sha256_file, alloc_mib, MTIME)

REF_IDX = (0, 60, 120)


def npz_facts(path, idxs=REF_IDX):
    """The arrays the falsifier reads, checked: uint8 [H, W, 3], one shape per scene. -> {frames, shapes, content_sha256}."""
    import numpy as np
    h, shapes = hashlib.sha256(), {}
    with np.load(path) as z:
        for i in idxs:
            assert f"f{i}" in z.files, f"{path}: frame f{i} is missing (has {sorted(z.files)})"
            a = z[f"f{i}"]
            assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3, f"{path}: f{i} is {a.dtype}{a.shape}, not uint8 HxWx3"
            shapes[f"f{i}"] = list(a.shape)
            h.update(f"f{i}|{a.dtype}|{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
    assert len({tuple(v) for v in shapes.values()}) == 1, f"{path}: frames differ in size {shapes}"
    return {"frames": list(idxs), "shapes": shapes, "content_sha256": h.hexdigest()}


def build(overlay1, refs_path, out_dir, tag="2", stage1_manifest=None):
    t0 = time.time()
    s1_sha = open(overlay1 + ".sha256").read().split()[0]
    assert MK.sha256_file(overlay1) == s1_sha, "overlay 1 does not match its .sha256"
    with tarfile.open(overlay1) as tf:
        m1_raw = tf.extractfile("pkg/MANIFEST2.json").read()
        v1_files = {n: tf.extractfile(n).read() for n in tf.getnames() if n != "pkg/MANIFEST2.json"}
    m = json.loads(m1_raw)
    assert "overlay_version" not in m, "overlay 1 is expected (no overlay_version key)"
    for e in m["files"]:                                                          # the v1 file list is checked against the v1 tar before anything is derived from it
        got = hashlib.sha256(v1_files[f"pkg/{e['path']}"]).hexdigest()
        assert got == e["sha256"] and len(v1_files[f"pkg/{e['path']}"]) == e["size"], f"overlay 1: {e['path']} does not match its own MANIFEST2"
    drv = open(f"{HERE}/t2v8f_stage2.py", "rb").read()
    assert b"\r" not in drv, "t2v8f_stage2.py has CR characters (LF only)"
    assert b"def check_ref_frames" in drv and b"NO_REF_FRAMES_IN_MANIFEST2" in drv, "t2v8f_stage2.py is not the fail-closed driver"
    for e in m["files"]:
        if e["path"] == "code/t2v8f_stage2.py":
            e["size"], e["sha256"] = len(drv), hashlib.sha256(drv).hexdigest()
    refs = json.load(open(refs_path, encoding="utf-8"))
    pkg_s1 = None
    if stage1_manifest:                                                           # a MANIFEST.json file, or the stage-1 tar (its pkg/MANIFEST.json)
        if stage1_manifest.endswith(".tar"):
            with tarfile.open(stage1_manifest) as t1:
                man1 = json.load(t1.extractfile("pkg/MANIFEST.json"))
        else:
            man1 = json.load(open(stage1_manifest, encoding="utf-8"))
        pkg_s1 = {e["path"]: e["sha256"] for e in man1["files"]}
        assert m["stage1_manifest_sha256"] is not None
    ship, total = [], 0
    for sc in m["scenes"]:
        sid = sc["scene"]
        assert sid in refs, f"{sid[:8]}: no reference-frames source listed"
        p = refs[sid]["path"]
        facts = npz_facts(p)
        size, sha = os.path.getsize(p), MK.sha256_file(p)
        if sc["where"] == "kit" and pkg_s1 is not None:                           # the kit scene keeps the npz stage 1 installed: it must be THAT file
            want = pkg_s1[f"refs/scenes/{sid}/ref_frames.npz"]
            assert sha == want, f"{sid[:8]}: the kit npz source {sha} is not stage 1's npz {want}"
        sc["ref_frames"] = {"rel": "ref_frames.npz", "bytes": size, "sha256": sha, "shipped_by_overlay2": sc["where"] == "scenes2", **facts}
        if sc["where"] == "scenes2":
            ship.append((f"scenes2/{sid}/ref_frames.npz", p))
            total += size
    n_new = len(ship)
    assert n_new == m["footprint"]["scenes_copied"], f"{n_new} npz to ship, the plan copies {m['footprint']['scenes_copied']} scenes"
    m["overlay_version"] = 2
    m["supersedes_manifest2_sha256"] = hashlib.sha256(m1_raw).hexdigest()
    m["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    m["purpose"] = m["purpose"] + " | OVERLAY 2: + ref_frames.npz per scene (fail-closed preflight)"
    m["ref_frames_totals"] = {"n_scenes": len(m["scenes"]), "n_shipped": n_new, "shipped_bytes": total, "frames": list(REF_IDX)}
    man2 = json.dumps(m, indent=1).encode("utf-8")
    os.makedirs(out_dir, exist_ok=True)
    name = f"t2v8f_stage2_overlay{tag}.tar"
    tar_path = f"{out_dir}/{name}"
    assert not os.path.exists(tar_path), f"{tar_path} exists: move it aside first (nothing is overwritten)"
    with tarfile.open(tar_path + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        def add_bytes(nm, b):
            ti = tarfile.TarInfo(nm)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(b), MK.MTIME, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(b))
        add_bytes("pkg/MANIFEST2.json", man2)
        add_bytes("pkg/code/t2v8f_stage2.py", drv)
        for nm, p in sorted(ship):
            ti = tarfile.TarInfo(nm)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(p), MK.MTIME, 0o644, 0, 0, "", ""
            with open(p, "rb") as fh:
                tf.addfile(ti, fh)
    os.replace(tar_path + ".tmp", tar_path)
    tar_sha, tar_bytes = MK.sha256_file(tar_path), os.path.getsize(tar_path)
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{tar_sha} *{name}\n")
    npz_mib = sum(MK.alloc_mib(os.path.getsize(p)) for _, p in ship)
    res = {"tar": tar_path, "tar_bytes": tar_bytes, "tar_sha256": tar_sha, "n_entries": 2 + n_new, "n_ref_frames_shipped": n_new, "ref_frames_bytes": total,
           "manifest2_v2_sha256": hashlib.sha256(man2).hexdigest(), "manifest2_v1_sha256": m["supersedes_manifest2_sha256"], "driver_sha256": hashlib.sha256(drv).hexdigest(),
           "footprint_mib_box2": {"ref_frames_npz_allocated": npz_mib, "overlay2_tar_allocated": MK.alloc_mib(tar_bytes), "replaced_files_allocated_delta": 0,
                                  "note": "1 MiB clusters; MANIFEST2.json and t2v8f_stage2.py replace files of the same cluster count"},
           "seconds": round(time.time() - t0, 1)}
    with open(f"{out_dir}/BUILD_RESULT{tag}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("--overlay1", default=f"{T2}/stage2/t2v8f_stage2_overlay.tar")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out-dir", default=f"{T2}/stage2/overlay2")
    ap.add_argument("--tag", default="2")
    ap.add_argument("--stage1-manifest", default=None, help="stage 1's MANIFEST.json: the kit scene's npz must equal the file stage 1 installed")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.overlay1, a.refs, a.out_dir, a.tag, a.stage1_manifest), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
