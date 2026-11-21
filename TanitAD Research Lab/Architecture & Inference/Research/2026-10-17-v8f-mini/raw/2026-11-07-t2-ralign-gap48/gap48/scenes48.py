"""GAP48: the scene list. The 62 NuRec scenes of stages 2 to 4 minus the scenes rule R-SCN excluded (AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb).

    python -I scenes48.py [--out SCENES48.json]

Inputs (read only, never edited):
  stage4/SCENE_SOURCES_2p3.json   one row per scene file: idx, scene, rel, bytes, sha256, where (kit | scenes2 | scenes3)
  stage4/EXCLUSIONS4.json         the R-SCN exclusions as of stage 4 (`excluded`: scene8 -> {scene, reason, ...})
  stage2/ref_frames/DOWNLOAD_RESULT.json, stage3/ref_frames/DOWNLOAD_RESULT3.json   the real mp4 of 61 scenes (path, bytes, sha256)
  t2_scenes/raw/RELEASE_LISTING.json   `lfs_sha256` of every release file (the kit scene's mp4 comes from the thor archive and is checked against it)

The count is DERIVED, never typed: `derive()` returns it and `main()` exits 2 when it is not EXPECTED (48), so a changed input stops the build instead of shipping a different list.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
MC = os.path.dirname(T2)
EXPECTED = 48
EXPECTED_ALL = 62
EXPECTED_EXCLUDED = 14
SOURCES = f"{T2}/stage4/SCENE_SOURCES_2p3.json"
EXCLUSIONS = f"{T2}/stage4/EXCLUSIONS4.json"
DL2 = f"{T2}/stage2/ref_frames/DOWNLOAD_RESULT.json"
DL3 = f"{T2}/stage3/ref_frames/DOWNLOAD_RESULT3.json"
LISTING = f"{MC}/t2_scenes/raw/RELEASE_LISTING.json"
KIT_SCENE = "7c72937c-c620-4776-9555-d57222c0081f"
KIT_MP4 = f"E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/{KIT_SCENE}/camera_front_wide_120fov.mp4"
MP4 = "camera_front_wide_120fov.mp4"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def derive(sources_path=SOURCES, exclusions_path=EXCLUSIONS):
    """-> dict(scenes=[{idx, scene, where, files:[{rel, bytes, sha256}]}] in idx order, n_all, n_excluded, n_kept, excluded=[full ids])."""
    rows = json.load(open(sources_path, encoding="utf-8"))
    ex = json.load(open(exclusions_path, encoding="utf-8"))
    excluded = sorted({v["scene"] for v in ex["excluded"].values()})
    by = {}
    for x in rows:
        s = by.setdefault(x["scene"], {"idx": x["idx"], "scene": x["scene"], "where": x["where"], "files": []})
        if s["idx"] != x["idx"] or s["where"] != x["where"]:
            raise ValueError(f"scene {x['scene']}: rows disagree on idx/where")
        s["files"].append({"rel": x["rel"], "bytes": x["bytes"], "sha256": x["sha256"]})
    unknown = [e for e in excluded if e not in by]
    if unknown:
        raise ValueError(f"excluded scenes absent from the sources: {unknown}")
    kept = sorted((s for k, s in by.items() if k not in set(excluded)), key=lambda s: s["idx"])
    return {"scenes": kept, "n_all": len(by), "n_excluded": len(excluded), "n_kept": len(kept), "excluded": excluded}


def mp4_sources():
    """scene -> {path, bytes, sha256, listing_sha256}: the stage-2 and stage-3 downloads plus the kit scene's file, every one against the release listing's `lfs_sha256`."""
    out = {}
    for p in (DL2, DL3):
        for sid, r in json.load(open(p, encoding="utf-8"))["scenes"].items():
            out[sid] = {"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"], "listing_sha256": r["listing_lfs_sha256"]}
    lst = json.load(open(LISTING, encoding="utf-8"))["scenes"][KIT_SCENE][MP4]
    out[KIT_SCENE] = {"path": KIT_MP4, "bytes": lst["size"], "sha256": None, "listing_sha256": lst["lfs_sha256"]}
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/SCENES48.json")
    a = ap.parse_args(argv)
    d = derive()
    print(json.dumps({k: d[k] for k in ("n_all", "n_excluded", "n_kept")}))
    if (d["n_all"], d["n_excluded"], d["n_kept"]) != (EXPECTED_ALL, EXPECTED_EXCLUDED, EXPECTED):
        print(f"ZZ GAP48-SCENE-COUNT-NOT-{EXPECTED} {d['n_all']} - {d['n_excluded']} = {d['n_kept']} ZZ")
        return 2
    mp4 = mp4_sources()
    rows = []
    for s in d["scenes"]:
        m = mp4.get(s["scene"])
        if m is None or not os.path.isfile(m["path"]):
            print(f"ZZ GAP48-MP4-ABSENT {s['scene']} {m and m['path']} ZZ")
            return 3
        got = sha256_file(m["path"])
        if got != m["listing_sha256"] or (m["sha256"] and got != m["sha256"]) or os.path.getsize(m["path"]) != m["bytes"]:
            print(f"ZZ GAP48-MP4-HASH {s['scene']} got={got} listing={m['listing_sha256']} ZZ")
            return 3
        rows.append({**s, "mp4": {"path": m["path"], "bytes": m["bytes"], "sha256": got}})
    res = {"rule": "R-SCN, AMENDMENT_T2_SCENE_REFUSAL.md (frozen 2167a7eb)", "n_all": d["n_all"], "n_excluded": d["n_excluded"], "n_kept": d["n_kept"], "excluded": d["excluded"],
           "by_root": {w: sum(1 for r in rows if r["where"] == w) for w in ("kit", "scenes2", "scenes3")},
           "inputs": {"sources_sha256": sha256_file(SOURCES), "exclusions_sha256": sha256_file(EXCLUSIONS)}, "scenes": rows}
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(rows)} scenes, {sum(r['mp4']['bytes'] for r in rows)} mp4 bytes, roots {res['by_root']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
