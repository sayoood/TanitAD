"""Stage 5 (AGENT R): ANY composite arm over the 62 scenes, built for EGAIN. The manifest rules, the cell order, the identity with stage 4's C0C recipe, the two REAL tars, and the NEW stage-5 driver
rehearsed on the stub (3 + 3 scene fixture, layered on the stage-2/3/4 rehearsals), then read by the PINNED reader's gates with `arm_name='EGAIN', arm_stage=5`.

Every `check_*(mut=None)` is one arm; `mut="<name>"` (or `"not_applied"`) makes the SAME assertion fail so the arm can be shown RED.
  artifact   both REAL tars (EGAIN a05, a10): 6 members, every sha256 = MANIFEST5 = its source (the checkpoints = the wspedit files), LF-only text, no stage-1/2/3/4 path replaced, frozen files unchanged,
             `check_manifest5` clean; planted faults (arm name, ckpt map, scene idx, requires) each NAMED.                                                                 RED: an extra member
  order      248 cells; `cell_order5(scenes, 'C0C') == cell_order4(scenes)`.                                                                                               RED: beside first
  identity   (1) per-cell argv over the 62 real scenes: stage 5 with arm X == stage 4's C0C argv once X_s -> C0C_s (all 248); (2) a FRESH process per cell, C0's own checkpoint through
             `t2v8f_stage5_cell.py` as arm C0ID: plans, extra and record keys equal stage 4's C0C_s0 bit for bit.                                                         RED: wrong_ckpt
  positive   the rehearsal: STAGE5.done, 24 cells, the real edited checkpoints loaded, `manifest5_sha256` + package counts recorded, both scene roots untouched.      RED: a cell dropped
  refusals   MISSING_CKPT, HASH_MISMATCH_CKPT, STAGE4_STILL_RUNNING, NOT_A_STAGE5_MANIFEST, MISSING_REQUIRED_FILE: each named, in the preflight, before any gate or cell.   RED: not_applied
  contract   the pinned reader's g3 / g2e / g2a / g2_check / rscn_check (EXCLUSIONS 2-5 from the REAL builder) / scenes_from / arrays_for_seed (19 rows) on the rehearsal's stage-2/3/4/5 pulls with
             arm_name='EGAIN', arm_stage=5; RED: a DONE removed, the result without manifest5_sha256, the arm named differently.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_stage5.py -q
"""
from __future__ import annotations

import glob
import hashlib
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
import make_stage5 as M5  # noqa: E402
import scene_exclusions as SE  # noqa: E402
import t2v8f_c0c as C0C  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402
import t2v8f_stage4 as D4  # noqa: E402
import t2v8f_stage5 as D5  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402
import test_t2v8f_stage2 as T  # noqa: E402
import test_t2v8f_stage3 as T3  # noqa: E402
import test_t2v8f_stage4 as T4  # noqa: E402

PY = T1.PY
WSP = "E:/Projects/TanitAD-artifacts/v8f-wspedit"
DOSES = ("a05", "a10")
CK = {d: {s: f"{WSP}/M2b_s{s}_wspedit_{d}/ckpt_last.pt" for s in (0, 1)} for d in DOSES}
S5 = f"{T2}/stage5"
_C = {}
FROZEN = {**T4.FROZEN, "code/t2v8f_stage4.py": "588e764264cdbb4ec785184bae92c1c7d9a0d528abf9aeb000a312fb94744c87", "code/t2v8f_c0c.py": "1a78317951ae5086a8e30839ec3c827fe55ce0195b7eb8537a21e336268a62bc"}
RUN_NAMES = ["pkg/MANIFEST5.json", "pkg/ckpt/EGAIN_s0.pt", "pkg/ckpt/EGAIN_s1.pt", "pkg/code/t2v8f_stage5.py", "pkg/code/t2v8f_stage5_cell.py"]


def real_tar(dose):
    return f"{S5}/EGAIN_{dose}/t2v8f_stage5_overlay_EGAIN_{dose}.tar"


def real_manifest(dose="a10"):
    with tarfile.open(real_tar(dose)) as tf:
        return json.loads(tf.extractfile("pkg/MANIFEST5.json").read())


# ============================================================================================================================ the REAL tars, the manifest rules, the order
def check_artifact(mut=None):
    for dose in DOSES:
        run = f"EGAIN_{dose}"
        with tarfile.open(real_tar(dose)) as tf:
            got = sorted(tf.getnames())
            blobs = {n: (tf.extractfile(n).read() if not n.endswith(".pt") else None) for n in got}
        want = sorted(RUN_NAMES + [f"pkg/jobs/stage5_{run}.cmd"])
        if mut == "extra":
            got = got + ["pkg/code/extra.py"]
        assert got == want, got
        m = real_manifest(dose)
        assert D5.check_manifest5(m) == [], D5.check_manifest5(m)
        assert sorted(f"pkg/{e['path']}" for e in m["files"]) == sorted(n for n in want if n != "pkg/MANIFEST5.json")
        br = json.load(open(f"{S5}/{run}/BUILD_RESULT5_{run}.json", encoding="utf-8"))
        assert br["tar_sha256"] == MK.sha256_file(real_tar(dose)) == open(real_tar(dose) + ".sha256").read().split()[0]
        assert br["manifest5_sha256"] == hashlib.sha256(blobs["pkg/MANIFEST5.json"]).hexdigest() and br["ckpt_sha256"] == m["arm5"]["ckpt_sha256"]
        for e in m["files"]:
            if e["path"].startswith("ckpt/"):
                arm = e["path"][5:-3]
                src = m["arm5"]["ckpt_source"][arm]
                assert src["path"] == CK[dose][int(arm[-1])] and src["sha256"] == e["sha256"] == MK.sha256_file(src["path"]) and src["bytes"] == e["size"], e["path"]
                continue
            b = blobs[f"pkg/{e['path']}"]
            assert hashlib.sha256(b).hexdigest() == e["sha256"] and len(b) == e["size"] and b"\r" not in b, e["path"]
            if e["path"].startswith("code/"):
                assert b == open(f"{CODE}/{os.path.basename(e['path'])}", "rb").read(), f"{e['path']} is not the dev file"
        assert m["arm5"]["ckpt_sha256"]["EGAIN_s0"] != m["arm5"]["ckpt_sha256"]["EGAIN_s1"] and m["arm5"]["dose"] == dose
        s1 = json.loads(tarfile.open(T1.TAR).extractfile("pkg/MANIFEST.json").read())
        taken = {e["path"] for e in s1["files"]} | {e["path"] for e in T4.real_manifest()["files"]}
        assert not (taken & {e["path"] for e in m["files"]}), sorted(taken & {e["path"] for e in m["files"]})
        assert MK.sha256_file(f"{T2}/stage4/t2v8f_stage4_overlay.tar") == json.load(open(f"{T2}/stage4/BUILD_RESULT4.json"))["tar_sha256"], "the stage-4 tar moved"
    m = real_manifest()
    assert m["source_manifests"]["stage4"]["sha256"] == json.load(open(f"{T2}/stage4/BUILD_RESULT4.json"))["manifest4_sha256"]
    cp = lambda: json.loads(json.dumps(m))      # noqa: E731
    faults = []
    a = cp(); a["stage"] = 4
    faults.append((a, "NOT_A_STAGE5_MANIFEST"))
    b = cp(); b["arm5"]["name"] = "C0C"
    faults.append((b, "BAD_ARM_NAME"))
    c = cp(); c["arm5"]["name"] = "EGAIN_s0"
    faults.append((c, "BAD_ARM_NAME"))
    d = cp(); d["arm5"]["ckpt_file"]["EGAIN_s1"] = "../x.pt"
    faults.append((d, "BAD_CKPT_FILE"))
    e = cp(); e["arm5"]["ckpt_sha256"].pop("EGAIN_s0")
    faults.append((e, "BAD_CKPT_MAP"))
    f = cp(); f["scenes"][5]["idx"] = 7
    faults.append((f, f"BAD_IDX {m['scenes'][5]['scene'][:8]}"))
    g = cp(); g["requires"].pop("code/t2v8f_stage4.py")
    faults.append((g, "REQUIRES_INCOMPLETE code/t2v8f_stage4.py"))
    h = cp(); h["requires"][D4.RI_REL]["sha256"] = "0" * 64
    faults.append((h, "BAD_RI_TOOL_REQUIRES"))
    i = cp(); i["cell_priority"] = i["cell_priority"][::-1]
    faults.append((i, "BAD_ARMS"))
    j = cp(); j["arm5"]["encoder"]["kind"] = "dinov3_lora_merged"
    faults.append((j, "ENCODER_UNSUPPORTED"))
    k = cp(); k["arm5"].pop("encoder")
    faults.append((k, "ENCODER_UNSUPPORTED"))
    for bad, w in faults:
        got = D5.check_manifest5(bad if mut != "not_applied" else m)
        assert any(x.startswith(w) for x in got), f"{w} was not named: {got}"
    for rel, want in FROZEN.items():
        assert MK.sha256_file(f"{T2}/{rel}") == want, f"{rel} changed"
    return {"faults": len(faults)}


def check_order(mut=None):
    m = T4.real_manifest()
    assert D5.cell_order5(m["scenes"], "C0C") == D4.cell_order4(m["scenes"]), "stage 5's order is not stage 4's"
    order = D5.cell_order5(m["scenes"], "EGAIN", ("beside", "deciding") if mut == "beside_first" else ("deciding", "beside"))
    assert len(order) == 248 and len({(c["idx"], c["tc"], c["arm"]) for c in order}) == 248
    assert [c["pass"] for c in order] == ["deciding"] * 124 + ["beside"] * 124
    assert {(c["tc"], c["arm"]) for c in order[:124]} == {("T2", "EGAIN_s0"), ("T2", "EGAIN_s1")} and {(c["tc"], c["arm"]) for c in order[124:]} == {("RR", "EGAIN_s0"), ("RR", "EGAIN_s1")}
    return len(order)


# ============================================================================================================================ the rehearsal fixture: stage 5 built on the stage-4 fixture
def fixture5():
    if "fx5" in _C:
        return _C["fx5"]
    f4 = T4.fixture4()
    fx = f4["fx"]
    root = f"{f4['root']}/s5"
    res = M5.build(f"{root}/out5", f"{root}/stage5", "EGAIN", "a10", CK["a10"], "a10", src_tar4=f4["build"]["tar"], n_launch=3, n_guard=3, fixture_tag="_fx5")
    with tarfile.open(res["tar"]) as tf:
        tf.extractall(fx["x"], filter="data")                          # pkg/MANIFEST5.json, the two scripts, the job, the two checkpoints: all NEW, beside stages 2 to 4
    _C["fx5"] = {"f4": f4, "fx": fx, "root": root, "build": res}
    return _C["fx5"]


def run_s5(f5, pull, s2, s3, extra=(), manifest=None, busy4=None, pkg=None, timeout=2400):
    fx = f5["fx"]
    pkg = pkg or fx["pkg"]
    r = f5["root"]
    cmd = [PY, "-I", "-u", f"{pkg}/code/t2v8f_stage5.py", "all", "--pkg", pkg, "--kit", fx["kit"], "--hf", fx["hf"], "--pull", pull, "--scenes2", s2, "--scenes3", s3,
           "--manifest5", manifest or f"{pkg}/MANIFEST5.json", "--expect-launch", "3", "--expect-guard", "3", "--busy-file2", f"{r}/NO_STAGE2_RUNNING.txt", "--busy-file3", f"{r}/NO_STAGE3_RUNNING.txt",
           "--busy-file4", busy4 or f"{r}/NO_STAGE4_RUNNING.txt", "--stub", "--inproc", *extra]
    p = subprocess.run(cmd, env=T1.clean_env(), capture_output=True, text=True, timeout=timeout)
    os.makedirs(os.path.dirname(pull), exist_ok=True)
    with open(f"{pull}.stdout.txt", "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode


def res5(pull):
    return json.load(open(f"{pull}/STAGE5_RESULT.json", encoding="utf-8"))


def positive():
    if "pos" not in _C:
        f5 = fixture5()
        f4 = f5["f4"]
        s2, s3 = T.new_scenes2(f4["fx"], "s5pos"), T3.new_scenes3(f4["f3"], "s5pos")
        pull = f"{f5['root']}/pull5_pos_{time.strftime('%H%M%S')}"
        before = (T3.tree_hash(s2), T3.tree_hash(s3))
        t0 = time.time()
        rc = run_s5(f5, pull, s2, s3)
        _C["pos"] = (pull, s2, s3, rc, round(time.time() - t0, 1), before)
    return _C["pos"]


def check_positive(mut=None):
    pull, s2, s3, rc, secs, before = positive()
    r = res5(pull)
    assert rc == 0, f"driver rc {rc}: {open(pull + '.stdout.txt', encoding='utf-8').read()[-1800:]}"
    assert os.path.exists(f"{pull}/STAGE5.done") and not os.path.exists(f"{pull}/STAGE5.fail") and not os.path.exists(f"{pull}/STAGE5.running")
    assert not glob.glob(f"{pull}/STAGE2*") and not glob.glob(f"{pull}/STAGE3*") and not glob.glob(f"{pull}/STAGE4*"), "another stage's marker in the stage-5 pull"
    assert r["gates"] == {"preflight": "PASS", "gate": "PASS", "rollouts": "PASS"} and r["overall"] == "COMPLETE" and r["stage"] == "stage5", r["gates"]
    info = r["steps"]["preflight"]["info"]
    assert info["pkg_files_ok"] == info["pkg_files"] and info["overlay_files_ok"] == info["overlay_files"] == 5 and info["ref_frames_ok"] == info["ref_frames"] == 6, info
    f5 = fixture5()
    assert r["manifest5_sha256"] == MK.sha256_file(f"{f5['fx']['pkg']}/MANIFEST5.json") and r["manifest_sha256"] == json.load(open(f"{f5['fx']['pkg']}/MANIFEST4.json"))["stage1_manifest_sha256"]
    assert not any(k in r for k in ("manifest2_sha256", "manifest3_sha256", "manifest4_sha256")) and r["arm"] == "EGAIN"
    done = sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))
    assert len(done) == 24 == r["steps"]["rollouts"]["n_cells_planned"] + (1 if mut == "cell_dropped" else 0), len(done)
    assert {os.path.basename(os.path.dirname(f)) for f in done} == {"EGAIN_s0", "EGAIN_s1"} and {os.path.basename(os.path.dirname(os.path.dirname(f))) for f in done} == {"T2", "RR"}
    for f in done:
        assert open(f, encoding="utf-8").read().strip().startswith("rc=0 records=") and open(f, encoding="utf-8").read().strip().endswith("executor=base")
    recs = [json.loads(x) for f in glob.glob(f"{pull}/rollouts/**/records.jsonl", recursive=True) for x in open(f, encoding="utf-8")]
    assert {x["purpose"] for x in recs} == {"deciding candidate"} and {int(x["scene"]) for x in recs} == {0, 1, 2, 42, 43, 44}
    assert all(e["composite"]["ri_tool_sha256"] == C0C.RI_TOOL_SHA for x in recs for e in x["extra"]), "a record without the composite stamp"
    assert (T3.tree_hash(s2), T3.tree_hash(s3)) == before, "the driver changed a file under a scenes root"
    c0c = {(int(x["scene"]), int(x["f0"])): x for f in glob.glob(f"{T4.positive()[0]}/rollouts/**/T2/C0C_s0/records.jsonl", recursive=True) for x in map(json.loads, open(f, encoding="utf-8"))}
    eg = {(int(x["scene"]), int(x["f0"])): x for f in glob.glob(f"{pull}/rollouts/**/T2/EGAIN_s0/records.jsonl", recursive=True) for x in map(json.loads, open(f, encoding="utf-8"))}
    assert sorted(c0c) == sorted(eg) and any(c0c[k]["plans"] != eg[k]["plans"] for k in eg), "EGAIN's plans equal C0C's on every window: the edited weights were not loaded"
    return {"seconds": secs, "cells": len(done), "records": len(recs)}


# ============================================================================================================================ identity with stage 4's C0C recipe
def check_identity(mut=None):
    m = T4.real_manifest()
    ctx = types.SimpleNamespace(scene_dir2=lambda sc: f"/scenes/{sc['where']}/{sc['scene']}", encoder="dinov3", device="cuda", hf="D:/tanitad/hf", pkg="D:/t2v8f/pkg")
    pull = "D:/t2v8f/pullX"
    with D4.bindings():
        a4 = [D.cell_argv(ctx, next(s for s in m["scenes"] if s["scene"] == c["scene"]), c["tc"], c["arm"], D.cell_dir(pull, c["scene"], c["tc"], c["arm"])) for c in D4.cell_order4(m["scenes"])]
    with D5.bindings5("X"):
        o5 = D5.cell_order5(m["scenes"], "X")
        a5 = [D.cell_argv(ctx, next(s for s in m["scenes"] if s["scene"] == c["scene"]), c["tc"], c["arm"], D.cell_dir(pull, c["scene"], c["tc"], c["arm"])) for c in o5]
    n5 = [[x.replace("X_s", "C0C_s") for x in av] for av in a5]
    assert len(a4) == len(n5) == 248 and (n5 == a4) == (mut != "argv_changed"), "stage 5's per-cell argv differs from stage 4's C0C argv beyond the names"
    assert all(av.count("--arm") == 1 and av[av.index("--arm") + 1].startswith("X_s") for av in a5)
    f4 = fixture5()["f4"]                                                # builds stage 5 on the stage-4 fixture: the stage-5 scripts are extracted into the SAME package dir
    fx = f4["fx"]
    pull4, s2, s3, rc, _, _ = T4.positive()
    assert rc == 0
    f, sc = T.kit_cell(fx, s2)
    man = {"arm5": {"name": "C0ID", "encoder": {"kind": D5.ENCODER_FROZEN}, "ckpt_file": {"C0ID_s0": "C0_s0.pt" if mut != "wrong_ckpt" else "C0_s1.pt", "C0ID_s1": "C0_s1.pt"}}}
    mp = f"{f4['root']}/MANIFEST5_identity_{time.strftime('%H%M%S')}.json"
    json.dump(man, open(mp, "w", encoding="utf-8", newline="\n"))
    cd = f"{fx['pkg']}/code"

    def cell(script, arm, env_extra):
        out = f"{f4['root']}/id_{arm}_{time.strftime('%H%M%S')}_{next(T4._SEQ)}"
        os.makedirs(out)
        env = T.cell_env(fx)
        env["T2V8F_RI_TOOL"] = f"{fx['pkg']}/{D4.RI_REL}"
        env.update(env_extra)
        p = subprocess.run([PY, "-I", "-u", script] + D.cell_argv(f, sc, "T2", arm, out), env=env, capture_output=True, text=True, timeout=900)
        assert p.returncode == 0, p.stdout[-500:] + p.stderr[-1500:]
        return [json.loads(x) for x in open(f"{out}/records.jsonl", encoding="utf-8")]
    c4 = cell(f"{cd}/t2v8f_stage4_cell.py", "C0C_s0", {})
    c5 = cell(f"{cd}/t2v8f_stage5_cell.py", "C0ID_s0", {D5.ENV_MANIFEST: mp})
    assert len(c4) == len(c5) == 2 and sorted(c4[0]) == sorted(c5[0])
    for a, b in zip(c4, c5):
        assert a["f0"] == b["f0"] and a["plans"] == b["plans"], "C0's own checkpoint through stage 5 does not give stage 4's C0C plan"
        assert a["extra"] == b["extra"] and a["arm"] == "C0C_s0" and b["arm"] == "C0ID_s0"
    return {"cells": 248, "records": len(c5)}


# ============================================================================================================================ refusals: each named, in the preflight, before any gate or cell
def refused5(tag, mutate=None, pkg=None, manifest=None, busy4=None):
    f5 = fixture5()
    f4 = f5["f4"]
    t = f"{tag}{next(T4._SEQ)}"
    s2, s3 = T.new_scenes2(f4["fx"], t), T3.new_scenes3(f4["f3"], t)
    pull = f"{f5['root']}/pull5_{t}_{time.strftime('%H%M%S')}"
    if mutate:
        mutate(s2, s3)
    rc = run_s5(f5, pull, s2, s3, manifest=manifest, busy4=busy4, pkg=pkg)
    return pull, rc, res5(pull)


def only_refusal5(pull, rc, r, want):
    assert rc == 1 and os.path.exists(f"{pull}/STAGE5.fail") and not os.path.exists(f"{pull}/STAGE5.done"), (rc, sorted(os.listdir(pull)))
    assert list(r["steps"]) == ["preflight"] and r["overall"] == "INFRASTRUCTURE_FAILURE", list(r["steps"])
    got = r["steps"]["preflight"]["failures"]
    assert len(got) == len(want) and all(g.startswith(w) for g, w in zip(got, want)), f"failures {got} != {want}"
    assert not os.path.exists(f"{pull}/rollouts") and not os.path.exists(f"{pull}/GATE_SCENES.json")


def pkg_copy5(tag):
    return T4.pkg_copy(fixture5()["f4"], tag)


def check_refusals(mut=None):
    f5 = fixture5()
    na = mut == "not_applied"
    pkg = pkg_copy5("ck5")
    if not na:
        os.remove(f"{pkg}/ckpt/EGAIN_s1.pt")
    pull, rc, r = refused5("mck", pkg=pkg)
    only_refusal5(pull, rc, r, ["MISSING_CKPT ckpt/EGAIN_s1.pt"])
    pkg = pkg_copy5("hk5")
    p = f"{pkg}/ckpt/EGAIN_s0.pt"
    b = bytearray(open(p, "rb").read())
    if not na:
        b[5000] ^= 0x01
        T4.replace_in_copy(p, bytes(b))
    pull, rc, r = refused5("hck", pkg=pkg)
    only_refusal5(pull, rc, r, ["HASH_MISMATCH_CKPT ckpt/EGAIN_s0.pt"])
    assert MK.sha256_file(f"{f5['fx']['pkg']}/ckpt/EGAIN_s0.pt") == f5["build"]["ckpt_sha256"]["EGAIN_s0"], "the test touched the shared fixture package"
    busy = f"{f5['root']}/STAGE4_busy_{time.strftime('%H%M%S')}.running"
    if not na:
        open(busy, "w").write("pid=1 start=x phase=rollouts\n")
    pull, rc, r = refused5("b4", busy4=busy)
    only_refusal5(pull, rc, r, ["STAGE4_STILL_RUNNING"])
    pull, rc, r = refused5("wm", manifest=f"{f5['fx']['pkg']}/{'MANIFEST5' if na else 'MANIFEST4'}.json")
    only_refusal5(pull, rc, r, ["NOT_A_STAGE5_MANIFEST"])
    pkg = pkg_copy5("rq5")
    if not na:
        os.remove(f"{pkg}/code/t2v8f_stage2_cell.py")
    pull, rc, r = refused5("rq", pkg=pkg)
    only_refusal5(pull, rc, r, ["MISSING_REQUIRED_FILE code/t2v8f_stage2_cell.py"])
    return {"rc": rc}


# ============================================================================================================================ the contract with the PINNED reader, arm EGAIN at stage 5
def check_contract(mut=None):
    import read_t2_dominance as RT
    ci = T4.contract_inputs()
    f5 = fixture5()
    p5 = positive()[0]
    assert positive()[3] == 0
    roots, results = {**ci["roots"], 5: f"{p5}/rollouts"}, {**ci["results"], 5: f"{p5}/STAGE5_RESULT.json"}
    ov = {**ci["overlays"], 5: f"{f5['build']['tar']}::pkg/MANIFEST5.json"}
    kw = {"arm_name": "EGAIN", "arm_stage": 5}
    gate, recs = RT.g3_check(roots, results, ci["sids"], ci["expect"], **kw)
    assert gate["PASS"] and recs and gate["pairing"] == {"s0": "PAIRED", "s1": "PAIRED"}, gate
    assert gate["n_starts"] == {"P10": 11, "EGAIN_s0": 11, "EGAIN_s1": 11}, gate["n_starts"]
    g2e = RT.g2e_check([("stage2", roots[2], "P10"), ("stage3", roots[3], "P10"), ("stage5", roots[5], "EGAIN_s0"), ("stage5", roots[5], "EGAIN_s1")])
    assert g2e["PASS"] and len(g2e["per_scene"]) == 6, g2e
    g2a = RT.g2a_check(RT.DEFAULT_RESULT1, results, RT.DEFAULT_MANIFEST1, ov)
    assert g2a["PASS"] and g2a["stages"][5]["pkg_files_ok"] == g2a["stages"][5]["pkg_files"] and g2a["stages"][5]["preflight"] == "PASS", g2a["why"]
    assert RT.g2_check(results)["PASS"]
    exclusions = {}
    for st in (2, 3, 4, 5):                                                                    # the REAL builder, no failed cell anywhere: ATTRIBUTED, nothing excluded
        r = json.load(open(results[st], encoding="utf-8"))
        e = SE.build(r, os.path.dirname(roots[st]), {}, [])
        e["inputs"] = {"result_sha256": SE.sha256(results[st]), "sources_sha256": None}
        exclusions[st] = e
    rs = RT.rscn_check(roots, results, ci["sids"], exclusions, ci["expect"], **kw)
    assert rs["problems"] == [] and rs["union"] == {} and sorted(rs["per_stage"]) == ["2", "3", "4", "5"], rs["problems"][:3]
    assert rs["per_stage"]["5"]["cells_per_scene"] == {"deciding": 2, "beside": 2} and rs["per_stage"]["5"]["deciding_done"] == 12 == rs["per_stage"]["5"]["deciding_required"]
    g3r, recs_r = RT.g3_check(roots, results, ci["sids"], ci["expect"], exclusions=exclusions, **kw)
    assert g3r["PASS"] and sorted(recs_r) == ["EGAIN_s0", "EGAIN_s1", "P10"], g3r["why"]
    scenes = RT.scenes_from(recs, ci["srcs"], roots)
    arrays = {s: RT.arrays_for_seed(recs[f"EGAIN_s{s}"], recs["P10"], scenes, s, arm_name="EGAIN") for s in RT.SEEDS}
    GD = RT.STATE["rr"].GD
    for s, ar in arrays.items():
        assert sum(1 for k in ar if k.startswith("a_")) == 19 == sum(1 for k in ar if k.startswith("b_")) and len(ar["eid"]) == 11
    sp = RT.arm_spec(types.SimpleNamespace(arm="EGAIN", arm_stage=5, arm_root=roots[5], arm_result=results[5], arm_exclusions="e5", arm_overlay=ov[5], arm_gate_log="log"))
    assert sp["arm"] == "EGAIN" and sp["stage"] == 5
    # RED arms, planted in a COPY of the stage-5 pull
    cp = f"{p5}_contract_{next(T4._SEQ)}"
    shutil.copytree(p5, cp)
    roots2, results2 = {**roots, 5: f"{cp}/rollouts"}, {**results, 5: f"{cp}/STAGE5_RESULT.json"}
    victim = sorted(glob.glob(f"{cp}/rollouts/*/objects/T2/EGAIN_s1/DONE"))[0]
    os.remove(victim)
    bad, none = RT.g3_check(roots2, results2, ci["sids"], ci["expect"], **kw)
    assert not bad["PASS"] and none is None and any("G3_CELL_NOT_DONE" in w and "no DONE marker" in w for w in bad["why"]), bad["why"]
    bad, none = RT.g3_check(roots2, results2, ci["sids"], ci["expect"], arm_name="EGAIN2", arm_stage=5)
    assert not bad["PASS"] and any("G3_CELLS stage 5 EGAIN2_s0" in w for w in bad["why"]), bad["why"]
    r5 = json.load(open(f"{cp}/STAGE5_RESULT.json", encoding="utf-8"))
    if mut != "not_applied":
        r5.pop("manifest5_sha256")
    json.dump(r5, open(f"{cp}/STAGE5_RESULT.json", "w", encoding="utf-8", newline="\n"), indent=1)
    bad = RT.g2a_check(RT.DEFAULT_RESULT1, results2, RT.DEFAULT_MANIFEST1, ov)
    assert not bad["PASS"] and any("G2A_OVERLAY_DIFFERS stage 5" in w for w in bad["why"]), bad["why"]
    return {"g3": gate["pairing"], "rows_per_seed": 19, "rscn_stages": sorted(rs["per_stage"])}


def test_artifact():
    check_artifact()


def test_order():
    check_order()


def test_identity():
    check_identity()


def test_positive():
    check_positive()


def test_refusals():
    check_refusals()


def test_contract():
    check_contract()
