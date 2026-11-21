"""Tar each per-episode dump dir of the DAG-HACC rollouts (stored, uncompressed) into t1/per_episode_dump_<tag>.tar, verify the member count and the per-member sha256 against the
packed manifest's per-episode digest, and ONLY THEN remove that directory (a directory this task created). exFAT allocates 1 MiB per small file, so 846 loose files would cost ~850 MB."""
import glob
import hashlib
import json
import os
import shutil
import sys
import tarfile

PK = "E:/Projects/TanitAD-artifacts/daghacc"
man = json.load(open(f"{PK}/packed/PACK_MANIFEST.json"))
sha_of = {}
for name, m in man.items():
    d = m["src_dir"]
    tag = os.path.basename(d)[len("dump_"):]
    if not os.path.isdir(d):
        print("skip (dir gone)", tag)
        continue
    files = sorted(glob.glob(d + "/ep*.npz"))
    assert len(files) == 141, (d, len(files))
    tp = f"{PK}/t1/per_episode_dump_{tag}.tar"
    with tarfile.open(tp, "w") as tf:
        for f in files:
            tf.add(f, arcname=f"dump_{tag}/{os.path.basename(f)}")
    h = hashlib.sha256()
    with tarfile.open(tp) as tf:
        ms = sorted((m_ for m_ in tf.getmembers() if m_.isfile()), key=lambda m_: m_.name)
        assert len(ms) == 141
        for m_ in ms:
            h.update(f"{os.path.basename(m_.name)} {hashlib.sha256(tf.extractfile(m_).read()).hexdigest()}\n".encode())
    assert h.hexdigest() == m["per_episode_files_digest_sha256"], ("tar digest differs from the packed manifest", tag)
    shutil.rmtree(d)
    print("tarred + removed", tag, os.path.getsize(tp))
print("ZZTAR-DONE")
