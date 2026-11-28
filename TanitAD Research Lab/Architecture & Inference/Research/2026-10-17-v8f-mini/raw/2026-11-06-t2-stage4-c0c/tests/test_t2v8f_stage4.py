"""Stage 4 (AGENT N): the composite arm C0C over the 62 scenes. The manifest rules, the cell order, the P10 pairing, the real tar, and the NEW stage-4 driver rehearsed on the stub (3 + 3 scene fixture).

Every `check_*(mut=None)` is one arm; `mut="<name>"` (or `"not_applied"`) makes the SAME assertion fail so the arm can fail; `code/run_mutations_stage4.py` runs them (AGENT O).
  manifest   `check_manifest4` on the REAL MANIFEST4: no refusal; a stage-3 manifest, a moved `where`, a bad idx, a wrong RI pin, a late start, other arms are each named.   RED: the fault is not planted
  order      248 cells = 124 T2 deciding (C0C_s0, C0C_s1 per scene) then 124 RR beside, scene-major, unique.                                                                RED: beside first
  pairing    MANIFEST4's (scene idx, f0) keys equal stage 2's 81 + stage 3's 40 (the keys P10(T2) was run on): `t2v8f_score.pair_check` accepts.                              RED: one f0 shifted
  artifact   the real tar: 6 members, every sha256 = MANIFEST4 = the dev file, RI tool 76,704 B = pinned, LF only, no stage-1 / stage-2 / stage-3 path replaced, frozen files unchanged. RED: an extra member
  positive   the rehearsal on a 3 + 3 scene fixture: STAGE4.done, 24 cells, scenes read from two roots that are NOT the pull and never written.                                RED: a cell dropped
  changed_byte / scenes_absent / busy_files / ri_copy / requires / wrong_manifest / missing_ckpt   each REFUSED, named, in the preflight, before any gate or cell (the shared fixture package is only ever hard-link copied). RED: not_applied
  composite  fresh process per cell (kit scene, T2, tick 0): C0C_s0's plan == composite_plan(C0's raw, P10's prior) bit for bit and != C0's; RED: composite_off, wrong_prior (a wrapper changes the policy)
  real_path  the kit's REAL falsify_actors + read_ref_frame on REAL banked npz files (a stage-2 and a guard scene) that MANIFEST4 pins, no mp4.                                    RED: no_npz
  contract   the PINNED reader's g3 / g2d / g2e / g2a / g2_check, scenes_from and arrays_for_seed (19 rows per seed, DK rows empty: no lead agent on the stub) on the rehearsal's stage-2 / 3 / 4 pulls.    RED: a DONE removed, an f0 shifted, a stage 4 not COMPLETE, no manifest4 hash
  footprint  62-scene arithmetic (1,016 + 746 MiB), BUILD_RESULT4 = footprint4, pass_need_mib under bindings(), a stub C0C cell = CELL_MIB.                                        RED: off_by_one
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_stage4.py -q
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
T2 = os.path.dirname(HERE)
CODE = f"{T2}/code"
for _p in (CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import make_stage2 as MK  # noqa: E402
import make_stage3 as M3  # noqa: E402
import make_stage4 as M4  # noqa: E402
import t2v8f_c0c as C0C  # noqa: E402
import t2v8f_nocv2 as N  # noqa: E402
import t2v8f_score as S  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402
import t2v8f_stage4 as D4  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402
import test_t2v8f_stage2 as T  # noqa: E402
import test_t2v8f_stage2_refs as TR  # noqa: E402
import test_t2v8f_stage3 as T3  # noqa: E402

PY = T1.PY
S4 = f"{T2}/stage4"
TAR4 = f"{S4}/t2v8f_stage4_overlay.tar"
_C = {}
FROZEN = {"code/t2v8f_stage2.py": "33b65b19a563b8fe275de5e46e0f569db6487bd7fdf40b7eab34e021094f4898", "code/t2v8f_stage3.py": "0688bb4732d8ef54f19e318de08008a54ce9276b66c815331e797b04019f733c",
          "code/read_t2_dominance.py": "6345d4c704ea46806271761ad3b53e376228f942e605da140fff3057cb9d2f5c"}


def real_manifest():
    with tarfile.open(TAR4) as tf:
        return json.loads(tf.extractfile("pkg/MANIFEST4.json").read())


def stage_manifests():
    return M4.load_real()


# ============================================================================================================================ the manifest rules, order, pairing, artifact
def check_manifest(mut=None):
    """`check_manifest4` on the REAL MANIFEST4: no refusal. Each planted fault is named."""
    m = real_manifest()
    assert D4.check_manifest4(m) == [], D4.check_manifest4(m)
    cp = lambda: json.loads(json.dumps(m))      # noqa: E731
    faults = []
    a = cp(); a["stage"] = 3
    faults.append((a, "NOT_A_STAGE4_MANIFEST"))
    b = cp(); b["scenes"][50]["where"] = "kit"
    faults.append((b, f"BAD_WHERE {m['scenes'][50]['scene'][:8]}"))
    c = cp(); c["scenes"][5]["idx"] = 7
    faults.append((c, f"BAD_IDX {m['scenes'][5]['scene'][:8]}"))
    d = cp(); d["ri_tool"]["sha256"] = "0" * 64
    faults.append((d, "BAD_RI_TOOL_PIN"))
    e = cp(); e["scenes"][6]["starts"] = [e["scenes"][6]["n_ticks"] - 60]
    faults.append((e, f"BAD_STARTS {m['scenes'][6]['scene'][:8]}"))
    f = cp(); f["arms"]["deciding"] = ["C0_s0", "C0_s1"]
    faults.append((f, "BAD_ARMS"))
    g = cp(); g["scenes"] = g["scenes"][:61]
    faults.append((g, "BAD_SCENE_COUNT"))
    h = cp(); h["files"] = [x for x in h["files"] if x["path"] != D4.RI_REL]
    faults.append((h, "RI_TOOL_NOT_IN_OVERLAY"))
    i = cp(); i["requires"] = {}
    faults.append((i, "NO_REQUIRES"))
    for bad, want in faults:
        got = D4.check_manifest4(bad if mut != "not_applied" else m)
        assert any(x.startswith(want) for x in got), f"{want} was not named: {got}"
    return {"faults": len(faults)}


def check_order(mut=None):
    """248 cells over the real MANIFEST4: 124 deciding (T2 x {C0C_s0, C0C_s1}) then 124 beside (RR x the same), scene-major, every cell unique. RED: beside first."""
    m = real_manifest()
    order = D4.cell_order4(m["scenes"], ("beside", "deciding") if mut == "beside_first" else ("deciding", "beside"))
    assert len(order) == 248 and len({(c["idx"], c["tc"], c["arm"]) for c in order}) == 248
    assert [c["pass"] for c in order] == ["deciding"] * 124 + ["beside"] * 124, "the deciding cells must all run before any beside cell"
    assert {(c["tc"], c["arm"]) for c in order[:124]} == set(D4.DECIDING4) and {(c["tc"], c["arm"]) for c in order[124:]} == set(D4.BESIDE4)
    assert [c["idx"] for c in order[:124]] == [i for k in range(62) for i in [m["scenes"][k]["idx"]] * 2]
    return len(order)


def check_pairing(mut=None):
    """The (scene idx, f0) keys of MANIFEST4 equal those of MANIFEST2 (81) + MANIFEST3 (40): the starts P10(T2) ran on. `pair_check` accepts; RED: one f0 shifted by a tick."""
    src = stage_manifests()
    p10 = sorted((s["idx"], f0) for m in (src["m2"], src["m3"]) for s in m["scenes"] for f0 in s["starts"])
    m4 = real_manifest()
    c0c = sorted((s["idx"], f0) for s in m4["scenes"] for f0 in s["starts"])
    if mut == "shifted":
        c0c[17] = (c0c[17][0], c0c[17][1] + 1)
    assert len(p10) == 121 and sum(1 for k in p10 if k[0] < 42) == 81
    try:
        S.pair_check(c0c, p10)
    except ValueError as e:                                            # the guard raises ValueError; the arm's RED is an AssertionError
        raise AssertionError(f"pair_check refused: {e}"[:200]) from e
    return len(c0c)


def check_artifact(mut=None):
    """The REAL tar: six members, each sha256 = MANIFEST4's `files` = the dev file (the RI tool = its pin, 76,704 B), LF only, tar sha256 = its .sha256 = BUILD_RESULT4, no path replaced of stage 1 / 2 / 3,
    and the frozen files (stage-2 and stage-3 drivers, the pinned reader) are byte-identical. RED: an extra member."""
    names = ["pkg/MANIFEST4.json", "pkg/code/t2v8f_c0c.py", "pkg/code/t2v8f_stage4.py", "pkg/code/t2v8f_stage4_cell.py", "pkg/jobs/stage4.cmd", "pkg/refs/tool/v8f_t1_eval_ri.py"]
    with tarfile.open(TAR4) as tf:
        got = sorted(tf.getnames())
        blobs = {n: tf.extractfile(n).read() for n in got}
    if mut == "extra":
        got = got + ["pkg/code/extra.py"]
    assert got == sorted(names), got
    m = json.loads(blobs["pkg/MANIFEST4.json"])
    assert sorted(f"pkg/{e['path']}" for e in m["files"]) == sorted(names[1:])
    for e in m["files"]:
        b = blobs[f"pkg/{e['path']}"]
        assert hashlib.sha256(b).hexdigest() == e["sha256"] and len(b) == e["size"], e["path"]
        if e["path"].startswith("code/"):
            assert b == open(f"{CODE}/{os.path.basename(e['path'])}", "rb").read(), f"{e['path']} is not the dev file"
        if not e["path"].startswith("refs/"):
            assert b"\r" not in b, f"{e['path']} has CR"
    ri = blobs["pkg/refs/tool/v8f_t1_eval_ri.py"]
    assert len(ri) == 76704 and hashlib.sha256(ri).hexdigest() == C0C.RI_TOOL_SHA
    br = json.load(open(f"{S4}/BUILD_RESULT4.json", encoding="utf-8"))
    assert br["tar_sha256"] == MK.sha256_file(TAR4) == open(TAR4 + ".sha256").read().split()[0] and br["manifest4_sha256"] == hashlib.sha256(blobs["pkg/MANIFEST4.json"]).hexdigest()
    s1 =json.loads(tarfile.open(T1.TAR).extractfile("pkg/MANIFEST.json").read())
    taken = {e["path"] for e in s1["files"]} | {e["path"] for mm in stage_manifests().values() if isinstance(mm, dict) for e in mm.get("files", [])}
    assert not (taken & {e["path"] for e in m["files"]}), sorted(taken & {e["path"] for e in m["files"]})
    for rel, want in FROZEN.items():
        assert MK.sha256_file(f"{T2}/{rel}") == want, f"{rel} changed"
    return {"tar_bytes": os.path.getsize(TAR4)}


# ============================================================================================================================ the rehearsal fixture (3 + 3 scenes, layered on the stage-2 and stage-3 fixtures)
def fixture4():
    if "fx4" in _C:
        return _C["fx4"]
    f3 = T3.fixture3()
    fx = f3["fx"]
    pk = fx["pkg"]
    j = lambda n: json.load(open(f"{pk}/{n}", encoding="utf-8"))      # noqa: E731
    src = {"m2": j("MANIFEST2.json"), "h2": MK.sha256_file(f"{pk}/MANIFEST2.json"), "m3": j("MANIFEST3.json"), "h3": MK.sha256_file(f"{pk}/MANIFEST3.json"), "tar2": "fixture_overlay2.tar", "tar3": "fixture_overlay3.tar"}
    root = f"{fx['root']}/s4"
    res = M4.build(src, f"{root}/out4", f"{root}/stage4", n_launch=3, n_guard=3, fixture_tag="_fx4", requires=M3.stage2_requires())
    with tarfile.open(res["tar"]) as tf:
        tf.extractall(fx["x"], filter="data")                         # pkg/MANIFEST4.json, the three code files, the job, the RI tool: all NEW, beside stages 2 and 3
    _C["fx4"] = {"f3": f3, "fx": fx, "root": root, "build": res}
    return _C["fx4"]


def run_s4(f4, pull, s2, s3, extra=(), manifest=None, busy2=None, busy3=None, timeout=2400, pkg=None):
    fx = f4["fx"]
    pkg = pkg or fx["pkg"]
    cmd = [PY, "-I", "-u", f"{pkg}/code/t2v8f_stage4.py", "all", "--pkg", pkg, "--kit", fx["kit"], "--hf", fx["hf"], "--pull", pull, "--scenes2", s2, "--scenes3", s3,
           "--manifest4", manifest or f"{pkg}/MANIFEST4.json", "--expect-launch", "3", "--expect-guard", "3", "--busy-file2", busy2 or f"{f4['root']}/NO_STAGE2_RUNNING.txt",
           "--busy-file3", busy3 or f"{f4['root']}/NO_STAGE3_RUNNING.txt", "--stub", "--inproc", *extra]
    p = subprocess.run(cmd, env=T1.clean_env(), capture_output=True, text=True, timeout=timeout)
    os.makedirs(os.path.dirname(pull), exist_ok=True)
    with open(f"{pull}.stdout.txt", "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode


def res4(pull):
    return json.load(open(f"{pull}/STAGE4_RESULT.json", encoding="utf-8"))


def positive():
    if "pos" not in _C:
        f4 = fixture4()
        s2, s3 = T.new_scenes2(f4["fx"], "s4pos"), T3.new_scenes3(f4["f3"], "s4pos")
        pull = f"{f4['root']}/pull4_pos_{time.strftime('%H%M%S')}"
        before = (T3.tree_hash(s2), T3.tree_hash(s3))
        t0 = time.time()
        rc = run_s4(f4, pull, s2, s3)
        _C["pos"] = (pull, s2, s3, rc, round(time.time() - t0, 1), before)
    return _C["pos"]


def check_positive(mut=None):
    """All three jobs PASS on the stub: STAGE4.done (never a STAGE2/3 marker), 24 cells DONE (12 deciding T2 then 12 beside RR, `executor=base`), records `deciding candidate` with scene idx 0..2 and 42..44,
    both scene roots byte-identical after the run, `manifest4_sha256` and the package file counts recorded, every record's plans carry the composite."""
    pull, s2, s3, rc, secs, before = positive()
    r = res4(pull)
    assert rc == 0, f"driver rc {rc}: {open(pull + '.stdout.txt', encoding='utf-8').read()[-1800:]}"
    assert os.path.exists(f"{pull}/STAGE4.done") and not os.path.exists(f"{pull}/STAGE4.fail") and not os.path.exists(f"{pull}/STAGE4.running")
    assert not glob.glob(f"{pull}/STAGE2*") and not glob.glob(f"{pull}/STAGE3*"), "another stage's marker in the stage-4 pull"
    assert r["gates"] == {"preflight": "PASS", "gate": "PASS", "rollouts": "PASS"} and r["overall"] == "COMPLETE" and r["stage"] == "stage4", r["gates"]
    info = r["steps"]["preflight"]["info"]
    assert info["pkg_files_ok"] == info["pkg_files"] and info["overlay_files_ok"] == info["overlay_files"] == 5 and info["ref_frames_ok"] == info["ref_frames"] == 6, info
    assert r["manifest4_sha256"] == MK.sha256_file(f"{_C['fx4']['fx']['pkg']}/MANIFEST4.json"), "STAGE4_RESULT.json does not record the sha256 of the MANIFEST4.json it ran"
    assert "manifest2_sha256" not in r and "manifest3_sha256" not in r
    done = sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))
    assert len(done) == 24 == r["steps"]["rollouts"]["n_cells_planned"] + (1 if mut == "cell_dropped" else 0), len(done)
    assert {os.path.basename(os.path.dirname(f)) for f in done} == {"C0C_s0", "C0C_s1"} and {os.path.basename(os.path.dirname(os.path.dirname(f))) for f in done} == {"T2", "RR"}
    for f in done:
        assert open(f, encoding="utf-8").read().strip().startswith("rc=0 records=") and open(f, encoding="utf-8").read().strip().endswith("executor=base")
    recs = [json.loads(x) for f in glob.glob(f"{pull}/rollouts/**/records.jsonl", recursive=True) for x in open(f, encoding="utf-8")]
    assert {x["purpose"] for x in recs} == {"deciding candidate"} and {int(x["scene"]) for x in recs} == {0, 1, 2, 42, 43, 44}, sorted({int(x["scene"]) for x in recs})
    assert (T3.tree_hash(s2), T3.tree_hash(s3)) == before, "the driver changed a file under a scenes root"
    g = json.load(open(f"{pull}/GATE_SCENES.json", encoding="utf-8"))
    assert all(x["finite"] and x["max_abs_diff_m"] <= 1e-6 for x in g["rows"]) and not g["failures"] and len(g["rows"]) == len({(int(x["scene"]), int(x["f0"])) for x in recs})
    return {"seconds": secs, "cells": len(done), "records": len(recs), "record_keys": sorted(recs[0])[:30]}


# ============================================================================================================================ the refusals: each REFUSED, named, in the preflight, before any gate or cell
_SEQ = itertools.count(1)


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)                             # NTFS: no new bytes
    except OSError:                                   # exFAT (no hard links): a real copy, slower and larger, same result
        shutil.copy2(src, dst)


def pkg_copy(f4, tag):
    """A HARD-LINKED copy of the fixture package (486 files, no new bytes). The shared fixture package is never mutated: a test replaces ONE name in the copy (`replace_in_copy`)."""
    dst = f"{f4['root']}/pkgcopy_{tag}_{next(_SEQ)}_{time.strftime('%H%M%S')}/pkg"
    shutil.copytree(f4["fx"]["pkg"], dst, copy_function=_link_or_copy)
    return dst


def replace_in_copy(path, data):
    os.remove(path)                                   # drops the copy's own link only; the fixture's file keeps its inode and its bytes
    with open(path, "wb") as fh:
        fh.write(data)


def refused4(tag, mutate=None, extra=(), manifest=None, busy2=None, busy3=None, scenes2=None, scenes3=None, pkg=None):
    """A fresh scenes2 / scenes3 copy, `mutate(s2, s3)` applied, the whole stub driver run. -> (pull, rc, STAGE4_RESULT)."""
    f4 = fixture4()
    t = f"{tag}{next(_SEQ)}"
    s2, s3 = T.new_scenes2(f4["fx"], t), T3.new_scenes3(f4["f3"], t)
    pull = f"{f4['root']}/pull4_{t}_{time.strftime('%H%M%S')}"
    if mutate:
        mutate(s2, s3)
    rc = run_s4(f4, pull, scenes2 or s2, scenes3 or s3, extra, manifest, busy2, busy3, pkg=pkg)
    return pull, rc, res4(pull)


def only_refusal4(pull, rc, r, want):
    """STAGE4.fail, exactly the named refusals (prefix match, in order) out of the preflight and nothing after it: no gate, no cell. -> the refusal texts."""
    assert rc == 1 and os.path.exists(f"{pull}/STAGE4.fail") and not os.path.exists(f"{pull}/STAGE4.done"), (rc, sorted(os.listdir(pull)))
    assert list(r["steps"]) == ["preflight"] and r["overall"] == "INFRASTRUCTURE_FAILURE", list(r["steps"])
    got = r["steps"]["preflight"]["failures"]
    assert len(got) == len(want) and all(g.startswith(w) for g, w in zip(got, want)), f"failures {got} != {want}"
    assert not os.path.exists(f"{pull}/rollouts") and not os.path.exists(f"{pull}/GATE_SCENES.json"), "a cell or the gate ran after the refusal"
    return got


def check_changed_byte(mut=None):
    """One byte of a stage-2 scene's volume (scenes2 root) AND one of a stage-3 scene's volume (scenes3 root) flipped, same size: both `HASH_MISMATCH scenes2/<id>/volume.nurec` are named, in scene order, and
    nothing else, no cell. RED: no byte flipped."""
    f4 = fixture4()
    one, g = f4["fx"]["one"], f4["f3"]["ids"][2]

    def mutate(s2, s3):
        if mut == "not_applied":
            return
        for root, sid in ((s2, one), (s3, g)):
            p = f"{root}/{sid}/volume.nurec"
            b = bytearray(open(p, "rb").read())
            b[3] ^= 0x01
            open(p, "wb").write(bytes(b))
    pull, rc, r = refused4("flip", mutate)
    only_refusal4(pull, rc, r, [f"HASH_MISMATCH scenes2/{one}/volume.nurec", f"HASH_MISMATCH scenes2/{g}/volume.nurec"])
    return {"rc": rc}


def check_scenes_absent(mut=None):
    """`--scenes3` pointing at a folder that does not exist: `SCENES3_ABSENT` and only that (stage 3 has not run, or its scenes were removed; stage 4 ships no scene), refused BEFORE the stage-2 preflight, so nothing
    was hashed. The same for `--scenes2`: `SCENES2_ABSENT`. RED: both roots present."""
    f4 = fixture4()
    gone = f"{f4['root']}/NO_SUCH_ROOT_{time.strftime('%H%M%S')}"
    assert not os.path.exists(gone)
    pull, rc, r = refused4("no3", scenes3=None if mut == "not_applied" else gone)
    only_refusal4(pull, rc, r, ["SCENES3_ABSENT"])
    assert r["steps"]["preflight"]["info"]["refused_before"].startswith("the stage-2 preflight"), r["steps"]["preflight"]["info"]
    pull, rc, r = refused4("no2", scenes2=None if mut == "not_applied" else gone)
    only_refusal4(pull, rc, r, ["SCENES2_ABSENT"])
    return {"rc": rc}


def check_busy_files(mut=None):
    """A stage-2 heartbeat, a stage-3 heartbeat, or both present: REFUSED by name (box 2 runs ONE job at a time), nothing hashed, no cell. RED: no heartbeat file written."""
    f4 = fixture4()
    stamp = time.strftime("%H%M%S")
    b2, b3 = f"{f4['root']}/STAGE2_busy_{stamp}.running", f"{f4['root']}/STAGE3_busy_{stamp}.running"
    if mut != "not_applied":
        open(b2, "w").write("pid=1 start=x phase=rollouts\n")
        open(b3, "w").write("pid=2 start=x phase=rollouts\n")
    pull, rc, r = refused4("busy2", busy2=b2)
    only_refusal4(pull, rc, r, ["STAGE2_STILL_RUNNING"])
    pull, rc, r = refused4("busy3", busy3=b3)
    only_refusal4(pull, rc, r, ["STAGE3_STILL_RUNNING"])
    pull, rc, r = refused4("busy23", busy2=b2, busy3=b3)
    only_refusal4(pull, rc, r, ["STAGE2_STILL_RUNNING", "STAGE3_STILL_RUNNING"])
    return {"rc": rc}


def check_ri_copy(mut=None):
    """The RI tool copy is pinned: one byte flipped in a hard-linked COPY of the package (the fixture's file keeps its sha256): `HASH_MISMATCH_RI_TOOL` and only that; the copy removed: `MISSING_RI_TOOL`; nothing
    hashed, no cell. The overlay's own per-file hash line (`HASH_MISMATCH refs/tool/...`, in the stage-2 preflight over MANIFEST4's `files`) is the second wall, and `check_manifest` pins that MANIFEST4 lists the copy
    (`RI_TOOL_NOT_IN_OVERLAY`). RED: no byte flipped, the copy not removed."""
    f4 = fixture4()
    pkg = pkg_copy(f4, "ri")
    p = f"{pkg}/{D4.RI_REL}"
    b = bytearray(open(p, "rb").read())
    assert len(b) == 76704 and hashlib.sha256(bytes(b)).hexdigest() == C0C.RI_TOOL_SHA, "the fixture's RI copy is not the pinned one"
    if mut != "not_applied":
        b[1000] ^= 0x01
        replace_in_copy(p, bytes(b))
    pull, rc, r = refused4("ri", pkg=pkg)
    only_refusal4(pull, rc, r, ["HASH_MISMATCH_RI_TOOL"])
    assert MK.sha256_file(f"{f4['fx']['pkg']}/{D4.RI_REL}") == C0C.RI_TOOL_SHA, "the test touched the shared fixture package"
    pkg2 = pkg_copy(f4, "rigone")
    if mut != "not_applied":
        os.remove(f"{pkg2}/{D4.RI_REL}")
    pull, rc, r = refused4("rigone", pkg=pkg2)
    only_refusal4(pull, rc, r, ["MISSING_RI_TOOL"])
    return {"rc": rc}


def check_requires(mut=None):
    """MANIFEST4 pins the stage-2 files the driver imports or spawns. A box-2 copy of the stage-2 driver that is another version: `HASH_MISMATCH_STAGE2_FILE code/t2v8f_stage2.py`; a pinned file absent from the package:
    `MISSING_STAGE2_FILE code/t2v8f_stage2_cell.py`. RED: the pin is not planted, the file is not removed."""
    f4 = fixture4()
    m = json.load(open(f"{f4['fx']['pkg']}/MANIFEST4.json", encoding="utf-8"))
    assert sorted(m["requires"]) == ["code/t2v8f_exec_hold.py", "code/t2v8f_stage2.py", "code/t2v8f_stage2_cell.py"], sorted(m["requires"])
    if mut != "not_applied":
        m["requires"]["code/t2v8f_stage2.py"]["sha256"] = "0" * 64
    p = f"{f4['root']}/MANIFEST4_req_{time.strftime('%H%M%S')}.json"
    json.dump(m, open(p, "w", encoding="utf-8", newline="\n"), indent=1)
    pull, rc, r = refused4("req", manifest=p)
    only_refusal4(pull, rc, r, ["HASH_MISMATCH_STAGE2_FILE code/t2v8f_stage2.py"])
    pkg = pkg_copy(f4, "reqgone")
    if mut != "not_applied":
        os.remove(f"{pkg}/code/t2v8f_stage2_cell.py")
    pull, rc, r = refused4("reqgone", pkg=pkg)
    only_refusal4(pull, rc, r, ["MISSING_STAGE2_FILE code/t2v8f_stage2_cell.py"])
    return {"rc": rc}


def check_wrong_manifest(mut=None):
    """The stage-3 or the stage-2 manifest handed to the stage-4 driver is REFUSED by name (`NOT_A_STAGE4_MANIFEST`): the wrong package cannot run as stage 4. RED: the stage-4 manifest is handed in."""
    f4 = fixture4()
    pk = f4["fx"]["pkg"]
    for n in ("MANIFEST3.json", "MANIFEST2.json"):
        pull, rc, r = refused4("wm", manifest=f"{pk}/MANIFEST4.json" if mut == "not_applied" else f"{pk}/{n}")
        only_refusal4(pull, rc, r, ["NOT_A_STAGE4_MANIFEST"])
    return {"rc": rc}


def check_missing_ckpt(mut=None):
    """C0C_s1 runs C0's checkpoint: `ckpt/C0_s1.pt` absent from a hard-linked copy of the package is `MISSING_CKPT ckpt/C0_s1.pt` and only that. RED: the file is not removed."""
    f4 = fixture4()
    pkg = pkg_copy(f4, "ck")
    if mut != "not_applied":
        os.remove(f"{pkg}/ckpt/C0_s1.pt")
    pull, rc, r = refused4("ck", pkg=pkg)
    only_refusal4(pull, rc, r, ["MISSING_CKPT ckpt/C0_s1.pt"])
    assert os.path.isfile(f"{f4['fx']['pkg']}/ckpt/C0_s1.pt"), "the test touched the shared fixture package"
    return {"rc": rc}


# ============================================================================================================================ the composite arm through the shipping cell entry (fresh process per cell)
WRAP = '''"""Test wrapper (scratch): the stage-4 cell entry with CompositePolicy changed BEFORE install, for the deliberate-regression arms. argv: MODE CODE_DIR <cell arguments>."""
import sys
mode, code = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
import numpy as np
import t2v8f_c0c as C0C
import t2v8f_run as R
orig = C0C.CompositePolicy
if mode == "composite_off":
    C0C.CompositePolicy = lambda inner: orig(inner, smooth="off", feasible="off")
elif mode == "wrong_prior":
    C0C.CompositePolicy = lambda inner: orig(inner, prior_hook=lambda p: np.zeros_like(p))
else:
    raise SystemExit("unknown mode " + mode)
C0C.install(R)
import t2v8f_stage2_cell as C
C.run_cell(sys.argv[3:])
'''


def cell_proc(f4, scenes2, tag, script, arm, pre=()):
    """ONE fresh process for the kit scene 7c72937c, T2, `arm` (what the real driver runs: a child per cell), with T2V8F_RI_TOOL on the package's pinned RI copy. -> its records."""
    fx = f4["fx"]
    f, sc = T.kit_cell(fx, scenes2)
    out = f"{f4['root']}/{tag}_{time.strftime('%H%M%S')}_{next(_SEQ)}"
    os.makedirs(out)
    env = T.cell_env(fx)
    env["T2V8F_RI_TOOL"] = f"{fx['pkg']}/{D4.RI_REL}"
    p = subprocess.run([PY, "-I", "-u", script, *pre] + D.cell_argv(f, sc, "T2", arm, out), env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-1500:]
    return [json.loads(x) for x in open(f"{out}/records.jsonl", encoding="utf-8")]


def base_cells(f4, s2):
    if "base_cells" not in _C:
        cd = f"{f4['fx']['pkg']}/code"
        _C["base_cells"] = (cell_proc(f4, s2, "c0", f"{cd}/t2v8f_stage2_cell.py", "C0_s0"), cell_proc(f4, s2, "p10", f"{cd}/t2v8f_stage2_cell.py", "P10"))
    return _C["base_cells"]


def check_composite(mut=None):
    """The shipping path, a FRESH process per cell: kit scene 7c72937c, T2, tick 0 (its history is logged, so every arm sees the same frames). C0_s0 records the raw plan; P10 the prior of the same forward (the arms share
    C0_s0.pt: C0's own `p10_prior` equals P10's plan bit for bit). The C0C_s0 cell, run through `t2v8f_stage4_cell.py`, must record EXACTLY `composite_plan(raw, prior)` (float32 in, widened float64 out), a plan that
    differs from C0's, the C0C `extra` (composite smooth / feasible, the loaded RI sha256, plan_raw, plan_s, p10_prior), and the same record keys as C0.
    RED (the same assertions, a wrapper that changes the policy before the cell): `composite_off` (smooth and feasible off: the plan IS C0's raw), `wrong_prior` (the prior replaced by zeros)."""
    f4 = fixture4()
    fx = f4["fx"]
    pull, s2, s3, rc, _, _ = positive()
    assert rc == 0
    c0, p10 = base_cells(f4, s2)
    cd = f"{fx['pkg']}/code"
    if mut in ("composite_off", "wrong_prior"):
        wrap = f"{f4['root']}/wrap_c0c.py"
        open(wrap, "w", encoding="utf-8", newline="\n").write(WRAP)
        cc = cell_proc(f4, s2, f"c0c_{mut}", wrap, "C0C_s0", pre=(mut, cd))
    else:
        cc = cell_proc(f4, s2, "c0c", f"{cd}/t2v8f_stage4_cell.py", "C0C_s0")
    assert len(c0) == len(p10) == len(cc) == 2
    assert sorted(cc[0]) == sorted(c0[0]), "the C0C record has other keys than the C0 record (the scorer reads both)"
    for a, b, c in zip(c0, p10, cc):
        assert a["f0"] == b["f0"] == c["f0"] and a["tcond"] == c["tcond"] == "T2" and c["arm"] == "C0C_s0"
        assert a["extra"][0]["p10_prior"] == b["plans"][0], "premise: C0_s0's own prior is P10's plan (the same forward and checkpoint)"
        raw, prior = np.asarray(a["plans"][0], np.float32), np.asarray(b["plans"][0], np.float32)
        want = C0C.composite_plan(raw, prior)[0].astype(np.float64).tolist()
        assert c["plans"][0] == want, "the C0C plan is not the composite of C0's raw plan and the P10 prior"
        assert c["plans"][0] != a["plans"][0], "the C0C plan equals C0's raw plan: the composite was not applied"
        e = c["extra"][0]
        assert e["arm"] == "C0C" and e["composite"]["smooth"] == "sg72" and e["composite"]["feasible"] == "kbox", e["composite"]
        assert e["composite"]["ri_tool_sha256"] == e["composite"]["ri_tool_loaded_sha256"] == C0C.RI_TOOL_SHA
        assert e["plan_raw"] == a["plans"][0] and e["p10_prior"] == b["plans"][0] and "plan_s" in e
    return {"records": len(cc)}


# ============================================================================================================================ the REAL read_ref_frame / falsify_actors on REAL banked reference-frame npz files
def falsify_banked(npz, subdir, sid, mut):
    """The kit's REAL `falsify_actors` (the nocv2 reader installed as on box 2) on a `<subdir>/<sid>/` folder holding a REAL banked `ref_frames.npz` and nothing else (no mp4); frames resized by the REAL
    `resize_area` as box 2 does. The banked frame scores grad_ncc 1.0 against itself. -> [grad_ncc]."""
    d = f"{TR.scratch(f's4real{next(_SEQ)}')}/{subdir}/{sid}"
    os.makedirs(d)
    if mut != "no_npz":
        shutil.copyfile(npz, f"{d}/ref_frames.npz")
    W, H = 960, 540
    with np.load(npz) as z:
        fr = {f"f{i}": N.resize_area(z[f"f{i}"], (W, H)) for i in D.REF_IDX}
    fake = TR.FakeRenderer(fr)
    fake.width, fake.height = W, H
    saved = {n: sys.modules.pop(n, None) for n in ("gsplat_renderer", "actor_map")}
    sys.path.insert(0, TR.KIT_HARNESS)
    try:
        import actor_map as A
        import gsplat_renderer as G
        assert os.path.dirname(os.path.abspath(G.__file__)).replace("\\", "/") == TR.KIT_HARNESS
        old = (G.grad_ncc, G.read_ref_frame)
        N.install()
        try:
            out = []
            for i in D.REF_IDX:
                try:
                    out.append(A.falsify_actors(fake, d, i))
                except Exception as e:                                                  # noqa: BLE001
                    raise AssertionError(f"falsify_actors(frame {i}) failed: {type(e).__name__}: {e}"[:200]) from e
        finally:
            G.grad_ncc, G.read_ref_frame = old
    finally:
        sys.path.remove(TR.KIT_HARNESS)
        for n in ("gsplat_renderer", "actor_map"):
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]
    assert not os.path.exists(f"{d}/camera_front_wide_120fov.mp4")
    assert [o["frame"] for o in out] == list(D.REF_IDX) and all(abs(o["grad_ncc_actors_on"] - 1.0) < 1e-6 for o in out), [o["grad_ncc_actors_on"] for o in out]
    return [round(o["grad_ncc_actors_on"], 6) for o in out]


def check_real_path(mut=None):
    """The REAL path stage 4's cells take on box 2 (no mp4 there): for the first REAL stage-2 scene and the first REAL stage-3 (guard) scene of MANIFEST4, the banked npz is exactly the file MANIFEST4 pins
    (bytes and sha256), and the kit's REAL `falsify_actors` + `read_ref_frame` read all three frames from it (grad_ncc 1.0 on itself). RED: the npz is not there."""
    m = real_manifest()
    out = {}
    for label, where, srcf in (("stage2", "scenes2", f"{T2}/stage2/ref_frames/REF_FRAMES_SOURCES.json"), ("stage3", "scenes3", f"{T2}/stage3/ref_frames/REF_FRAMES_SOURCES3.json")):
        sc = next(s for s in m["scenes"] if s["where"] == where)
        src = json.load(open(srcf, encoding="utf-8"))[sc["scene"]]
        assert src["sha256"] == sc["ref_frames"]["sha256"] and src["bytes"] == sc["ref_frames"]["bytes"], f"{label}: MANIFEST4 pins another npz than the banked one"
        assert MK.sha256_file(src["path"]) == src["sha256"], f"{label}: the banked npz changed on disk"
        out[label] = falsify_banked(src["path"], where, sc["scene"], mut)
    return out


# ============================================================================================================================ the contract test: the PINNED reader's gates on the rehearsal's own pulls
def contract_inputs():
    f4 = fixture4()
    f3, fx = f4["f3"], f4["fx"]
    p2, p3, p4 = T.positive()[0], T3.positive()[0], positive()[0]
    assert T.positive()[2] == 0 and T3.positive()[2] == 0 and positive()[3] == 0
    m = json.load(open(f"{fx['pkg']}/MANIFEST4.json", encoding="utf-8"))
    sids = {k: s["scene"] for k, s in enumerate(m["scenes"])}                     # the reader indexes by POSITION (0..n_l-1 launch, then guard): equal to idx for the real 0..61
    srcs = {s["idx"]: f"{(fx['src'] if s['where'] != 'scenes3' else f3['src'])}/{s['scene']}/extracted/sequence_tracks.json" for s in m["scenes"]}
    n_l = sum(1 for s in m["scenes"] if s["where"] != "scenes3")
    expect = {"scenes_launch": n_l, "scenes_guard": len(m["scenes"]) - n_l, "starts_launch": sum(len(s["starts"]) for s in m["scenes"][:n_l]), "starts_guard": sum(len(s["starts"]) for s in m["scenes"][n_l:])}
    ov = {2: f"{fx['build2']['tar']}::pkg/MANIFEST2.json", 3: f"{f3['build']['tar']}::pkg/MANIFEST3.json", 4: f"{f4['build']['tar']}::pkg/MANIFEST4.json"}
    pulls = {2: p2, 3: p3, 4: p4}
    return {"pulls": pulls, "roots": {s: f"{p}/rollouts" for s, p in pulls.items()}, "results": {2: f"{p2}/STAGE2_RESULT.json", 3: f"{p3}/STAGE3_RESULT.json", 4: f"{p4}/STAGE4_RESULT.json"},
            "sids": sids, "srcs": srcs, "expect": expect, "overlays": ov}


def check_contract(mut=None):
    """The pinned reader (`code/read_t2_dominance.py`, sha256 6345d4c7..., never edited) accepts the layout the stage-4 DRIVER wrote, fed the rehearsal's own pulls (stage 2 and 3 from their real drivers on the
    stub, stage 4 from this one): `g3_check` PASS and PAIRED at both seeds, `g2d_check` (ZERO == P10 on every window), `g2e_check` (actor_attach per scene), `g2a_check` (one package manifest in every stage, the
    overlay manifest = what each stage recorded, no stage-1 file replaced; the stage-4 result carries `manifest4_sha256` and `pkg_files_ok == pkg_files`), `g2_check`, then the reader's own row builder
    (`scenes_from` + `arrays_for_seed`: the 19 rows per seed from C0C and P10, windows identical; `read_blocks` itself needs a lead agent for the three DK rows and cannot run on the stub fixture, so the statistics stay with `test_t2v8f_dominance.py`). RED, planted in COPIES of the stage-4 pull: one DONE removed (`G3_CELL_NOT_DONE`), one f0
    shifted (`G3_PAIRING`), the stage-4 overall not COMPLETE (`G3_OVERALL_NOT_COMPLETE stage 4`), the result without its manifest hash (`G2A_OVERLAY_DIFFERS stage 4`)."""
    import read_t2_dominance as RT
    ci = contract_inputs()
    gate, recs = RT.g3_check(ci["roots"], ci["results"], ci["sids"], ci["expect"])
    assert gate["PASS"] and recs and gate["pairing"] == {"s0": "PAIRED", "s1": "PAIRED"}, gate
    assert gate["n_starts"] == {"P10": 11, "C0C_s0": 11, "C0C_s1": 11}, gate["n_starts"]
    g2d = RT.g2d_check({2: ci["roots"][2], 3: ci["roots"][3]})
    assert g2d["PASS"] and g2d["windows"] == {2: 5, 3: 6}, g2d
    g2e = RT.g2e_check([("stage2", ci["roots"][2], "P10"), ("stage3", ci["roots"][3], "P10"), ("stage4", ci["roots"][4], "C0C_s0"), ("stage4", ci["roots"][4], "C0C_s1")])
    assert g2e["PASS"] and len(g2e["per_scene"]) == 6, g2e
    g2a = RT.g2a_check(RT.DEFAULT_RESULT1, ci["results"], RT.DEFAULT_MANIFEST1, ci["overlays"])
    assert g2a["PASS"], g2a["why"]
    assert g2a["stages"][4]["pkg_files_ok"] == g2a["stages"][4]["pkg_files"] and g2a["stages"][4]["preflight"] == "PASS"
    g2c = RT.g2_check(ci["results"])
    assert g2c["PASS"], g2c
    scenes = RT.scenes_from(recs, ci["srcs"], ci["roots"])
    assert sorted(scenes) == [0, 1, 2, 42, 43, 44], sorted(scenes)
    arrays = {s: RT.arrays_for_seed(recs[f"C0C_s{s}"], recs["P10"], scenes, s) for s in RT.SEEDS}
    assert all(len(a["eid"]) == 11 for a in arrays.values())
    GD = RT.STATE["rr"].GD
    dk = {"DK-VIOL", "DK-HW", "DK-TG"}                                           # distance keeping needs a LEAD agent in frame: the stub fixture has none (MEASURED: finite 0 of 11 at both arms), so `read_blocks` ("no valid pair") cannot run on it
    for s, ar in arrays.items():
        assert len(GD.LABELS) == 19 and sum(1 for k in ar if k.startswith("a_")) == 19 == sum(1 for k in ar if k.startswith("b_")), "the 19 C0C rows and 19 P10 rows are not all built"
        for i, lab in enumerate(GD.LABELS):
            fa, fb = int(np.isfinite(ar[f"a_{i}"]).sum()), int(np.isfinite(ar[f"b_{i}"]).sum())
            assert fa == fb and (fa == 0 if lab in dk else fa >= 10), f"seed {s} row {lab}: finite C0C {fa} / P10 {fb} of 11 windows"
    # RED arms: planted in copies of the stage-4 pull
    cp = f"{ci['pulls'][4]}_contract_{next(_SEQ)}"
    shutil.copytree(ci["pulls"][4], cp)
    roots, results = {**ci["roots"], 4: f"{cp}/rollouts"}, {**ci["results"], 4: f"{cp}/STAGE4_RESULT.json"}
    victim = sorted(glob.glob(f"{cp}/rollouts/*/objects/T2/C0C_s1/DONE"))[0]
    if mut != "not_applied":
        os.remove(victim)
    bad, none = RT.g3_check(roots, results, ci["sids"], ci["expect"])
    assert not bad["PASS"] and none is None and any("G3_CELL_NOT_DONE" in w and "no DONE marker" in w for w in bad["why"]), bad["why"]
    shutil.copyfile(ci["pulls"][4] + "/rollouts/" + os.path.relpath(victim, f"{cp}/rollouts").replace("\\", "/"), victim)
    f0file = sorted(glob.glob(f"{cp}/rollouts/*/objects/T2/C0C_s0/records.jsonl"))[0]
    rows = [json.loads(x) for x in open(f0file, encoding="utf-8")]
    if mut != "not_applied":
        rows[0]["f0"] += 1
    open(f0file, "w", encoding="utf-8", newline="\n").write("".join(json.dumps(x) + "\n" for x in rows))
    bad, none = RT.g3_check(roots, results, ci["sids"], ci["expect"])
    assert not bad["PASS"] and none is None and any("G3_PAIRING seed 0" in w for w in bad["why"]), bad["why"]
    shutil.copyfile(ci["pulls"][4] + "/rollouts/" + os.path.relpath(f0file, f"{cp}/rollouts").replace("\\", "/"), f0file)
    r4 = json.load(open(f"{cp}/STAGE4_RESULT.json", encoding="utf-8"))
    r4["overall"] = "COMPLETE" if mut == "not_applied" else "INCOMPLETE"
    r4.pop("manifest4_sha256") if mut != "not_applied" else None
    json.dump(r4, open(f"{cp}/STAGE4_RESULT.json", "w", encoding="utf-8", newline="\n"), indent=1)
    bad, none = RT.g3_check(roots, results, ci["sids"], ci["expect"])
    assert any("G3_OVERALL_NOT_COMPLETE stage 4" in w for w in bad["why"]), bad["why"]
    bad = RT.g2a_check(RT.DEFAULT_RESULT1, results, RT.DEFAULT_MANIFEST1, ci["overlays"])
    assert not bad["PASS"] and any("G2A_OVERLAY_DIFFERS stage 4" in w for w in bad["why"]), bad["why"]
    return {"g3": gate["pairing"], "g2d_windows": g2d["windows"], "rows_per_seed": 19, "dk_rows_empty": sorted(dk)}


# ============================================================================================================================ the footprint
def check_footprint(mut=None):
    """The per-drive numbers for the 62 scenes: the deciding pass is 124 x 6 + 62 x 4 + 24 = 1,016 MiB and the beside pass 124 x 6 + 2 = 746 MiB (`make_stage4.footprint4`, the stage-2 per-cell figure MEASURED on stage 1's
    pull); the REAL MANIFEST4 and BUILD_RESULT4 record exactly those and the stage-4 `pass_need_mib` (under `bindings()`) reads them; the overlay is 7 MiB on D:, C: gets no new write; D: free must be >= 11.00 GiB
    (deciding only) / 11.73 GiB (both). A stub C0C cell, as the driver wrote it, costs `MK.CELL_MIB` on a 1 MiB-cluster volume. RED: off by one."""
    m = real_manifest()
    fp = m["footprint"]
    assert len(m["scenes"]) == 62
    assert fp["pass_mib"] == {"deciding": 124 * 6 + 62 * 4 + 24 + (1 if mut == "off_by_one" else 0), "beside": 124 * 6 + 2}, fp["pass_mib"]
    br = json.load(open(f"{S4}/BUILD_RESULT4.json", encoding="utf-8"))
    assert br["footprint"] == M4.footprint4(m["scenes"], os.path.getsize(TAR4), br["n_files"]), "BUILD_RESULT4's footprint is not what footprint4 computes for this tar"
    assert br["footprint"]["pass_mib"] == fp["pass_mib"] and br["footprint"]["box2_C_new_writes_mib"] == 0 and br["footprint"]["box2_D_overlay_mib"] == 7
    assert (br["footprint"]["box2_D_total_mib"], br["footprint"]["D_free_gib_needed_for_deciding_pass_only"], br["footprint"]["D_free_gib_needed_to_run_both_passes"]) == (1769, 11.0, 11.73), br["footprint"]
    ctx = types.SimpleNamespace(pull=f"{TR.scratch(f'fp4{next(_SEQ)}')}/pull", man2={"footprint": fp})
    ctx.live_scenes = lambda: [{"idx": s["idx"], "scene": s["scene"]} for s in m["scenes"]]
    with D4.bindings():
        assert abs(D.pass_need_mib(ctx, "deciding") - 1016) < 1e-9 and abs(D.pass_need_mib(ctx, "beside") - 746) < 1e-9, (D.pass_need_mib(ctx, "deciding"), D.pass_need_mib(ctx, "beside"))
    assert D.PASSES is not D4.PASSES4, "bindings() did not restore the stage-2 passes"
    pull, _, _, rc, _, _ = positive()
    assert rc == 0
    done = sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))
    costs = {T1.allocated_mib(os.path.dirname(f)) for f in done}
    assert len(done) == 24 and costs == {MK.CELL_MIB - (1 if mut == "off_by_one" else 0)}, costs
    return {"pass_mib": fp["pass_mib"], "cell_mib": sorted(costs)}


def test_manifest():
    check_manifest()


def test_order():
    check_order()


def test_pairing():
    check_pairing()


def test_artifact():
    check_artifact()


def test_positive():
    check_positive()


def test_changed_byte():
    check_changed_byte()


def test_scenes_absent():
    check_scenes_absent()


def test_busy_files():
    check_busy_files()


def test_ri_copy():
    check_ri_copy()


def test_requires():
    check_requires()


def test_wrong_manifest():
    check_wrong_manifest()


def test_missing_ckpt():
    check_missing_ckpt()


def test_composite():
    check_composite()


def test_real_path():
    check_real_path()


def test_contract():
    check_contract()


def test_footprint():
    check_footprint()
