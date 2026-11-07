"""Builds the STAGE-5 package of arm C0C-RA (PREREG_T2_RALIGN.md section 5) for box 2. Dev box, CPU. Nothing is committed, staged, downloaded or overwritten; box 2 is not touched.

    python -I make_ralign_stage5.py build --map-dir <ralign_fit out-dir> --m-align M_ALIGN.json --g-idmap G_IDMAP.json                 [--tag T] [--out-dir ..] [--stage-dir ..]   (status GATED: the real run)
    python -I make_ralign_stage5.py build --map-dir <ralign_fit identity out-dir> --status IDMAP --tag ID                              (the box-2 G-IDMAP package: identity maps)
    python -I make_ralign_stage5.py build --map-dir <any maps> --status REHEARSAL --tag RH                                           (stub rehearsal only; the driver refuses it unless RALIGN_ALLOW_REHEARSAL=1)

ONE tar `t2v8f_stage5_overlay_<RUN>.tar` (RUN = C0C-RA[_TAG]); the chain `audit/stage5_chain.sh <RUN>` reads `stage5/<RUN>/` (the default --out-dir) exactly as for every stage-5 arm.
Members (pkg/, all NEW paths except the two banked stage-5 scripts, which ship BYTE-IDENTICAL to code/ so the package does not depend on what an earlier overlay left on box 2):
  code/t2v8f_stage5.py  code/t2v8f_stage5_cell.py            (banked, unchanged, sha256-asserted against code/)
  code/ralign_lib.py  code/ralign_policy.py  code/ralign_cell.py  code/ralign_driver.py          (this folder)
  ralign/MAP_A.pt  ralign/MAP_B.pt  ralign/MAPS.json  jobs/stage5_<RUN>.cmd  MANIFEST5.json
NO checkpoint ships: C0C-RA runs C0's own weights, which are already in the stage-1 package (ckpt/C0_s0.pt, ckpt/C0_s1.pt; their sha256 come from the stage-1 manifest and the driver's preflight
re-hashes them). The job runs `ralign_driver.py` instead of `t2v8f_stage5.py`.
GATED needs the evidence in hand and bound by sha256: M_ALIGN.json (verdict M-ALIGN-PASS, gate_valid, the same two maps as --map-dir) and G_IDMAP.json (verdict G-IDMAP-PASS, written for the
same ralign_policy.py / ralign_lib.py bytes that ship). Edit the policy after G-IDMAP and the build refuses until G-IDMAP is run again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
CODE = f"{T2}/code"
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import make_stage2 as MK  # noqa: E402
import make_stage5 as M5  # noqa: E402
import ralign_lib as L  # noqa: E402
import t2v8f_c0c as C0C  # noqa: E402
import t2v8f_stage4 as D4  # noqa: E402
import t2v8f_stage5 as D5  # noqa: E402

NAME = "C0C-RA"
KIND = "frozen_dinov3_ralign"
SHIP = ["ralign_lib.py", "ralign_policy.py", "ralign_cell.py", "ralign_driver.py"]       # from HERE -> pkg/code/
BANKED = ["t2v8f_stage5.py", "t2v8f_stage5_cell.py"]                                      # from CODE (byte-identical) -> pkg/code/
BANKED_SHA = {"t2v8f_stage5.py": "1fb1af10682f7ddf782dee23b6bc32dc2d048c34ad841857ebfd75636d9c0e01", "t2v8f_stage5_cell.py": "7fb89e5dda15e63f13955c25dad910c7f5751c477aeb46706c8316bd9ed303a0"}
STATUS = ("GATED", "IDMAP", "REHEARSAL")


def lf_bytes(path):
    b = open(path, "rb").read()
    assert b"\r" not in b, f"{path} has CR characters (LF only)"
    return b


def cmd_text(tag, subset_launch=None):
    """The banked stage-5 job with the wrapper driver in place of `t2v8f_stage5.py` (everything else, the pull directory and the log name included, is the banked recipe's).
    A subset package (the 2-scene G-IDMAP run) adds the driver's own `--expect-launch N --expect-guard 0`."""
    t = M5.cmd_text(NAME, tag)
    old = "pkg\\code\\t2v8f_stage5.py all"
    assert t.count(old) == 1, "the banked job text changed shape"
    t = t.replace(old, "pkg\\code\\ralign_driver.py all")
    if subset_launch:
        tail = "--manifest5 D:/t2v8f/pkg/MANIFEST5.json\n"
        assert t.count(tail) == 1, "the banked job text changed shape (manifest5 tail)"
        t = t.replace(tail, f"--manifest5 D:/t2v8f/pkg/MANIFEST5.json --expect-launch {int(subset_launch)} --expect-guard 0\n")
    return t


def check_evidence(status, maps_idx, m_align, g_idmap):
    """-> the manifest's `gate` block. Raises AssertionError named for a missing or mismatched piece of evidence."""
    if status != "GATED":
        return {"status": status, "m_align_verdict": None, "g_idmap_verdict": None}
    assert m_align and g_idmap, "RALIGN_NOT_GATED: status GATED needs --m-align and --g-idmap"
    ma, gi = json.load(open(m_align, encoding="utf-8")), json.load(open(g_idmap, encoding="utf-8"))
    assert ma.get("verdict") == "M-ALIGN-PASS" and ma.get("gate_valid") is True, f"RALIGN_NOT_GATED M_ALIGN verdict {ma.get('verdict')!r} gate_valid {ma.get('gate_valid')!r}"
    assert ma.get("card_sha256") == L.CARD_SHA256, "RALIGN_NOT_GATED M_ALIGN names another card"
    for f in L.FOLDS:
        assert ma["maps"][f]["tensor_sha256"] == maps_idx["maps"][f]["tensor_sha256"], f"RALIGN_GATE_MAP_MISMATCH map {f}: M_ALIGN was run on other maps"
    assert gi.get("verdict") == "G-IDMAP-PASS", f"RALIGN_NOT_GATED G_IDMAP verdict {gi.get('verdict')!r}"
    for fn in ("ralign_policy.py", "ralign_lib.py"):
        assert gi.get("code_sha256", {}).get(fn) == L.sha256_file(f"{HERE}/{fn}"), f"RALIGN_GIDMAP_STALE G_IDMAP was run for another {fn}: run it again"
    return {"status": status, "m_align_verdict": ma["verdict"], "m_align_sha256": L.sha256_file(m_align), "g_idmap_verdict": gi["verdict"], "g_idmap_sha256": L.sha256_file(g_idmap),
            "m_align_median_before": ma["main"]["median_before"], "m_align_median_after": ma["main"]["median_after"]}


def build(out_dir, stage, tag, map_dir, status, m_align=None, g_idmap=None, src_tar4=M5.TAR4, n_launch=D4.N_LAUNCH, n_guard=D4.N_GUARD, fixture_tag=None, stage1_dir=None, ship_dir=None, subset_launch=None):
    t_start = time.time()
    assert status in STATUS, status
    assert not tag or M5.RUN_RE.fullmatch(tag), f"bad tag {tag!r}"
    run = M5.run_id(NAME, tag)
    assert not os.path.exists(stage) or not os.listdir(stage), f"{stage} must be new and empty (nothing is overwritten)"
    idx = json.load(open(f"{map_dir}/MAPS.json", encoding="utf-8"))
    assert idx.get("card_sha256") == L.CARD_SHA256 and idx.get("status") in ("FITTED", "IDMAP"), f"RALIGN_MAPS_STATUS {idx.get('status')!r}"
    assert (idx["status"] == "IDMAP") == (status == "IDMAP"), f"RALIGN_MAPS_STATUS maps are {idx['status']!r}, requested status {status!r}"
    maps = {f: L.load_map(f"{map_dir}/{idx['maps'][f]['file']}", idx["maps"][f]["file_sha256"], idx["maps"][f]["tensor_sha256"], tuple(idx["shape"])) for f in L.FOLDS}
    assert all(m.is_identity for m in maps.values()) == (status == "IDMAP"), "RALIGN_IDMAP_NOT_IDENTITY: the identity status and the identity maps must coincide"
    gate = check_evidence(status, idx, m_align, g_idmap)
    m4, h4 = M5.read_member(src_tar4, "pkg/MANIFEST4.json")
    assert m4.get("stage") == 4 and len(m4["scenes"]) == n_launch + n_guard, (m4.get("stage"), len(m4["scenes"]))
    assert subset_launch is None or status == "IDMAP", "a subset package is the G-IDMAP run only (status IDMAP)"
    s1_manifest, _ = M5.read_member(f"{stage1_dir or T2 + '/stage1'}/t2v8f_stage1.tar", "pkg/MANIFEST.json")
    taken = {e["path"] for e in s1_manifest["files"]} | {e["path"] for e in m4["files"]} | set(m4["requires"])
    s1_ck = {e["path"]: e["sha256"] for e in s1_manifest["files"] if e["path"].startswith("ckpt/")}
    pkg = f"{stage}/pkg"
    for d in ("code", "jobs", "ralign"):
        os.makedirs(f"{pkg}/{d}", exist_ok=True)
    files = []                                                                                          # (tar member, source path)
    for fn in BANKED:
        b = lf_bytes(f"{CODE}/{fn}")
        if fn in BANKED_SHA:
            assert hashlib.sha256(b).hexdigest() == BANKED_SHA[fn], f"{fn} is not the banked file (code/ changed)"
        open(f"{pkg}/code/{fn}", "wb").write(b)
        files.append((f"code/{fn}", f"{pkg}/code/{fn}"))
    for fn in SHIP:
        open(f"{pkg}/code/{fn}", "wb").write(lf_bytes(f"{ship_dir or HERE}/{fn}"))
        files.append((f"code/{fn}", f"{pkg}/code/{fn}"))
    for fn in ("MAP_A.pt", "MAP_B.pt", "MAPS.json"):
        open(f"{pkg}/ralign/{fn}", "wb").write(open(f"{map_dir}/{fn}", "rb").read())
        files.append((f"ralign/{fn}", f"{pkg}/ralign/{fn}"))
    cmd = cmd_text(tag, subset_launch).encode()
    assert b"\r" not in cmd
    open(f"{pkg}/jobs/stage5_{run}.cmd", "wb").write(cmd)
    files.append((f"jobs/stage5_{run}.cmd", f"{pkg}/jobs/stage5_{run}.cmd"))
    arms = D5.arm_names(NAME)
    ck_file = {arms[0]: "C0_s0.pt", arms[1]: "C0_s1.pt"}
    ck_sha = {arms[0]: s1_ck["ckpt/C0_s0.pt"], arms[1]: s1_ck["ckpt/C0_s1.pt"]}
    ov = [{"path": rel, "size": os.path.getsize(src), "sha256": MK.sha256_file(src)} for rel, src in sorted(files)]
    assert not ({e["path"] for e in ov} & taken), sorted({e["path"] for e in ov} & taken)
    ri_b = open(C0C.RI_TOOL_DEV, "rb").read()
    assert hashlib.sha256(ri_b).hexdigest() == C0C.RI_TOOL_SHA and len(ri_b) == D4.RI_BYTES
    req = {p: {"size": e["size"], "sha256": e["sha256"]} for p, e in m4["requires"].items()}
    by4 = {e["path"]: e for e in m4["files"]}
    for p in D5.REQUIRED_STAGE4:
        req[p] = {"size": by4[p]["size"], "sha256": by4[p]["sha256"]}
    scenes = json.loads(json.dumps(m4["scenes"]))
    if subset_launch:                                                                                   # the first N launch scenes of stage 4's manifest, in its order
        scenes = [s for s in scenes if str(s.get("slot", "")).startswith("launch")][:int(subset_launch)]
        assert len(scenes) == int(subset_launch), (len(scenes), subset_launch)
    n_starts = sum(len(s["starts"]) for s in scenes)
    if fixture_tag is None and not subset_launch:
        assert n_starts == 121, n_starts
    digit = int(idx["fold_digit"])
    scene_fold = {s["scene"]: L.fold_of(s["scene"], digit) for s in scenes}
    ov_by = {e["path"]: e for e in ov}
    block = {"card": "PREREG_T2_RALIGN.md", "card_sha256": L.CARD_SHA256, "status": status, "fold_digit": digit, "tier": "T2-RA (never pooled with T2 or T1-kin numbers)",
             "maps": {f: {"file": f"ralign/MAP_{f}.pt", "file_sha256": ov_by[f"ralign/MAP_{f}.pt"]["sha256"], "tensor_sha256": idx["maps"][f]["tensor_sha256"], "n_pairs": idx["maps"][f]["n_pairs"],
                          "n_fold_scenes": len(idx["maps"][f]["fold_scenes"])} for f in L.FOLDS},
             "applied": "a scene in fold A reads MAP_B (fitted on fold B's pairs); a scene in fold B reads MAP_A", "scene_fold": scene_fold,
             "scene_fold_counts": {f: sum(1 for v in scene_fold.values() if v == f) for f in L.FOLDS}, "gate": gate,
             "code_sha256": {fn: ov_by[f"code/{fn}"]["sha256"] for fn in SHIP + BANKED}}
    P = D5.passes5(NAME)
    man = {"package": "t2v8f_stage5_overlay", "stage": 5, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "purpose": f"STAGE 5: the composite arm {NAME} (C0C with the cross-fitted render-to-real token map, status {status}, tag {tag or 'none'}) at T2 (deciding) then RR (beside), 62 scenes, 121 starts; overlay on the stage-1 package",
           "stage1_manifest_sha256": m4["stage1_manifest_sha256"], "stage1_tar_sha256": m4["stage1_tar_sha256"], "files": ov, "requires": req,
           "ri_tool": m4["ri_tool"], "scene_roots_box2": m4["scene_roots_box2"], "scenes": scenes,
           "arm5": {"name": NAME, "tag": tag, "dose": None, "run": run, "arms": list(arms),
                    "encoder": {"kind": KIND, "note": "t2v8f_run.build_arm's frozen DINOv3-L InLoopEncoder as stages 2 to 4, with ralign_policy.AlignPolicy mapping the scene tokens (arm5.ralign)"},
                    "ckpt_file": ck_file, "ckpt_sha256": ck_sha, "ckpt_source": {a: {"path": f"pkg/ckpt/{ck_file[a]}", "note": "C0's own checkpoint, already in the stage-1 package; not shipped"} for a in arms},
                    "composite": {"smooth": C0C.SMOOTH, "feasible": C0C.FEASIBLE, "same_as": "stage 4 C0C (t2v8f_c0c.CompositePolicy); only the scene tokens differ (the map)"}, "ralign": block},
           "arms": {"deciding": list(arms), "beside": list(arms)}, "t2_conds": ["T2", "RR"], "cell_priority": [list(x) for x in P["deciding"]], "beside_order": [list(x) for x in P["beside"]],
           "source_manifests": {"stage4": {"member": "pkg/MANIFEST4.json", "sha256": h4, "tar": os.path.basename(src_tar4)}}, "totals": m4["totals"], "start_classes": m4["start_classes"]}
    if subset_launch:
        man["subset"] = {"first_launch_scenes": int(subset_launch), "of": len(m4["scenes"]), "scenes": [s["scene"] for s in scenes], "why": "the 2-scene G-IDMAP run; totals/start_classes above are stage 4's"}
    est_tar = sum(e["size"] for e in ov) + (1 << 20)
    man["footprint"] = M5.footprint5(scenes, est_tar, [e["size"] for e in ov] + [1 << 20])
    with open(f"{pkg}/MANIFEST5.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    os.makedirs(out_dir, exist_ok=True)
    name_tar = f"t2v8f_stage5_overlay_{run}{fixture_tag or ''}.tar"
    tar_path = f"{out_dir}/{name_tar}"
    assert not os.path.exists(tar_path), f"{tar_path} exists: move it aside first (nothing is overwritten)"
    members = sorted([("MANIFEST5.json", f"{pkg}/MANIFEST5.json")] + files)
    with tarfile.open(tar_path + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for rel, src in members:
            ti = tarfile.TarInfo(f"pkg/{rel}")
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(src), MK.MTIME, 0o644, 0, 0, "", ""
            with open(src, "rb") as fh:
                tf.addfile(ti, fh)
    os.replace(tar_path + ".tmp", tar_path)
    tar_sha = MK.sha256_file(tar_path)
    with open(tar_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{tar_sha} *{name_tar}\n")
    assert os.path.getsize(tar_path) <= est_tar and MK.alloc_mib(os.path.getsize(tar_path)) <= MK.alloc_mib(est_tar), "the tar is larger than the footprint estimate in MANIFEST5"
    fp = man["footprint"]
    res = {"run": run, "arm": NAME, "tag": tag, "dose": None, "status": status, "tar": tar_path, "tar_bytes": os.path.getsize(tar_path), "tar_sha256": tar_sha, "n_files": len(members),
           "members": [f"pkg/{r}" for r, _ in members], "manifest5_sha256": MK.sha256_file(f"{pkg}/MANIFEST5.json"), "n_scenes": len(scenes), "n_starts": n_starts, "n_cells": fp["cells"],
           "footprint": fp, "driver_sha256": MK.sha256_file(f"{pkg}/code/ralign_driver.py"), "cell_sha256": MK.sha256_file(f"{pkg}/code/ralign_cell.py"), "cmd_sha256": MK.sha256_file(f"{pkg}/jobs/stage5_{run}.cmd"),
           "ckpt_sha256": ck_sha, "ri_tool_sha256": C0C.RI_TOOL_SHA, "stage1_manifest_sha256": m4["stage1_manifest_sha256"], "requires": req, "source_manifests": man["source_manifests"],
           "overlay_files": ov, "ralign": {k: v for k, v in block.items() if k != "scene_fold"}, "seconds": round(time.time() - t_start, 1)}
    with open(f"{out_dir}/BUILD_RESULT5_{run}{fixture_tag or ''}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--map-dir", required=True)
    b.add_argument("--status", choices=STATUS, default="GATED")
    b.add_argument("--m-align", default=None)
    b.add_argument("--g-idmap", default=None)
    b.add_argument("--tag", default="")
    b.add_argument("--out-dir", default=None)
    b.add_argument("--stage-dir", default=None)
    b.add_argument("--subset-launch", type=int, default=None, help="IDMAP only: keep the first N launch scenes (the G-IDMAP run is 2 scenes)")
    a = ap.parse_args(argv)
    run = M5.run_id(NAME, a.tag)
    out = (a.out_dir or f"{T2}/stage5/{run}").replace("\\", "/")
    stg = (a.stage_dir or f"E:/Projects/TanitAD-artifacts/scratch_ralign_t2v8f/build_{run}").replace("\\", "/")
    res = build(out, stg, a.tag, a.map_dir.replace("\\", "/"), a.status, a.m_align, a.g_idmap, subset_launch=a.subset_launch)
    print(json.dumps({k: res[k] for k in ("run", "status", "tar", "tar_sha256", "manifest5_sha256", "n_cells", "n_files")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
