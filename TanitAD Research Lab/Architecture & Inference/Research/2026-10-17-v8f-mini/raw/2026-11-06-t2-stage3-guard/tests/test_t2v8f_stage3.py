"""Stage 3 (AGENT K): the 20 guard scenes of PREREG_T2_CRAWL. The draw, the reference frames, the NEW stage-3 driver (rehearsed on the stub AND through the REAL reader), the two-root disk rule, the pool.

Every `check_*(mut=None)` is one arm; `mut="<name>"` (or `"not_applied"`) makes the SAME assertion fail so the arm can fail; `code/run_mutations_stage3.py` runs them.
  draw           the guard starts are reproducible bit for bit (same process and a fresh one) and equal the kit and the hand derivation.   RED: default_rng(2)
  classes        every recorded v0 equals the scorer's `logged_v0`, the class is the 0.5 m/s rule.                                          RED: a 50 m/s threshold
  refs_real      20 mp4 = lfs_sha256, 20 npz = the extraction record, f0/f60/f120 uint8 3840x2160x3.                                        RED: a wrong listing hash
  real_guard     the REAL `read_ref_frame` + the kit's REAL `falsify_actors` on a REAL guard npz (resized as box 2 resizes), no mp4.         RED: no npz
  manifest       `check_manifest3`: a stage-3 manifest has no refusal; a stage-2 manifest, a kit scene, a bad idx, bad starts are each named. RED: the fault is not planted
  positive       the rehearsal on a 3-scene guard fixture: STAGE3.done, 42 cells, scenes read from a root that is NOT the pull and never written.   RED: a cell dropped
  missing_npz / changed_byte / stage2_manifest / busy_file / requires    each REFUSED, named, before any cell.                              RED: not_applied
  two_roots      scenes on one root, the pull on another: the floor is decided by the PULL drive (a disk_usage spy), not the scenes drive.     RED: a driver that probes the scenes drive
  resume         DONE cells are skipped on relaunch; one removed DONE is the only cell re-run.                                              RED: nothing removed
  sftp / footprint / artifact / pool    the put batch, the per-drive numbers, the real tar, the stage-2 + stage-3 pool.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_stage3.py -q
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
import unittest.mock as mock

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
import make_starts3 as MS3  # noqa: E402
import t2v8f_kit as K  # noqa: E402
import t2v8f_nocv2 as N  # noqa: E402
import t2v8f_stage2 as D  # noqa: E402
import t2v8f_stage2_score as SC2  # noqa: E402
import t2v8f_stage3 as D3  # noqa: E402
import t2v8f_stage3_score as SC3  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402
import test_t2v8f_stage2 as T  # noqa: E402
import test_t2v8f_stage2_refs as TR  # noqa: E402

PY = T1.PY
S3 = f"{T2}/stage3"
GS, PLAN3 = f"{S3}/GUARD_STARTS.json", f"{S3}/stage3_plan.json"
TAR3 = f"{S3}/t2v8f_stage3_overlay.tar"
RAW = f"{os.path.dirname(T2)}/t2_scenes/raw"
_C = {}


def table():
    return json.load(open(K.SCREEN_TABLE, encoding="utf-8"))["scenes"]


# ============================================================================================================================ item 1: the draw
def check_draw(mut=None):
    """ONE default_rng(1), twice per guard scene: reproducible in this process and in a fresh one (bit for bit: the sha256 over the starts), equal to GUARD_STARTS.json, to the kit's `all_starts`
    and to the hand derivation; two draws in [18, n-61] per scene. RED: default_rng(2) draws other starts."""
    seed = 2 if mut == "seed2" else 1
    tab = table()
    a, b = MS3.derive(tab, seed), MS3.derive(tab, seed)
    assert a == b, "two draws in one process differ"
    sha = MS3.starts_sha256([(d["scene"], d["starts"]) for d in a])
    code = ("import sys; sys.path.insert(0, %r); import json, make_starts3 as M; t = json.load(open(%r))['scenes']; "
            "print(M.starts_sha256([(d['scene'], d['starts']) for d in M.derive(t, %d)]))") % (CODE, K.SCREEN_TABLE, seed)
    fresh = subprocess.run([PY, "-I", "-c", code], capture_output=True, text=True, timeout=120, env=T1.clean_env()).stdout.strip()
    assert fresh == sha, f"a fresh process draws {fresh}, this one {sha}"
    gs = json.load(open(GS, encoding="utf-8"))
    assert sha == gs["starts_sha256"], f"the draw (seed {seed}) {sha[:12]} is not GUARD_STARTS.json's {gs['starts_sha256'][:12]}"
    kit = K.all_starts(tab)
    assert all(kit[d["scene"]] == d["starts"] for d in a), "the kit's all_starts differs from the hand derivation"
    assert len(a) == 20 and all(len(d["raw"]) == 2 and all(18 <= f <= d["n_ticks"] - 61 for f in d["raw"]) for d in a)
    assert [d["starts"] for d in a] == [s["starts"] for s in gs["scenes"]]
    return {"sha256": sha[:12], "n_starts": sum(len(d["starts"]) for d in a)}


def check_classes(mut=None):
    """Every GUARD_STARTS.json v0 equals the scorer's `logged_v0` on the scene's own rig (two implementations), and the class is v0 < 0.5 m/s (MM ruling)."""
    thr = 50.0 if mut == "threshold" else SC2.LAUNCH_V0_MPS
    plan = {p["scene"]: p["root"] for p in json.load(open(f"{RAW}/DOWNLOAD_PLAN.json"))["scenes"]}
    n = 0
    for s in json.load(open(GS, encoding="utf-8"))["scenes"]:
        gt, _ = D.load_rig(f"{plan[s['scene']]}/{s['scene']}/rig_trajectories.json")
        for f0, v0, cl in zip(s["starts"], s["start_v0_mps"], s["start_class"]):
            assert abs(SC2.logged_v0(gt, f0) - v0) < 5e-5, (s["slot"], f0)
            assert cl == ("launch" if v0 < thr else "moving"), (s["slot"], f0, v0, cl)
            n += 1
    assert n == 40
    return {"n": n}


# ============================================================================================================================ item 2: the references
def check_refs_real(mut=None):
    """The download and the npz, on disk: 20 mp4 = the listing's lfs_sha256 (and the download record), 20 npz = the extraction record, three uint8 3840x2160x3 frames each."""
    lst = json.load(open(f"{RAW}/RELEASE_LISTING.json", encoding="utf-8"))["scenes"]
    dl = json.load(open(f"{S3}/ref_frames/DOWNLOAD_RESULT3.json", encoding="utf-8"))
    ex = json.load(open(f"{S3}/ref_frames/REF_FRAMES_SOURCES3.json", encoding="utf-8"))
    assert dl["n_ok"] == 20 and dl["n_failed"] == 0 and len(ex) == 20 and dl["total_bytes"] == 850400161
    for sid, o in dl["scenes"].items():
        want = lst[sid]["camera_front_wide_120fov.mp4"]["lfs_sha256"]
        if mut == "bad_listing":
            want = "0" * 64
        assert MK.sha256_file(o["path"]) == want == o["sha256"], f"{sid[:8]}: the mp4 is not the listing's"
        e = ex[sid]
        assert MK.sha256_file(e["path"]) == e["sha256"] and os.path.getsize(e["path"]) == e["bytes"], f"{sid[:8]}: the npz is not the extraction record's"
        with np.load(e["path"]) as z:
            assert sorted(z.files) == ["f0", "f120", "f60"] and all(z[k].dtype == np.uint8 and z[k].shape == (2160, 3840, 3) for k in z.files), sid[:8]
    return {"npz": len(ex)}


def check_real_guard(mut=None):
    """The kit's REAL `falsify_actors`, with the nocv2 reader installed as on box 2, on a scenes3-style folder holding a REAL guard npz and nothing else: each frame is read from the npz and resized by the
    REAL `resize_area` to the renderer size, and the banked frame scores grad_ncc 1.0 against itself. No mp4 exists. RED: the npz is not there."""
    src = json.load(open(f"{S3}/ref_frames/REF_FRAMES_SOURCES3.json", encoding="utf-8"))
    sid = sorted(src)[0]
    d = f"{TR.scratch('guard')}/scenes3/{sid}"
    os.makedirs(d)
    if mut != "no_npz":
        shutil.copyfile(src[sid]["path"], f"{d}/ref_frames.npz")
    W, H = 960, 540
    with np.load(src[sid]["path"]) as z:
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
    return {"grad_ncc_on": [round(o["grad_ncc_actors_on"], 6) for o in out]}


# ============================================================================================================================ the manifest rules
def real_manifest():
    with tarfile.open(TAR3) as tf:
        return json.loads(tf.extractfile("pkg/MANIFEST3.json").read())


def check_manifest(mut=None):
    """`check_manifest3` on the REAL MANIFEST3: no refusal. Each planted fault is named: a stage-2 manifest, a kit scene, a skipped idx, a start inside the last 6 s, a non-guard slot."""
    m = real_manifest()
    assert D3.check_manifest3(m) == [], D3.check_manifest3(m)
    faults = []
    m2 = json.loads(json.dumps(m)); m2.pop("stage"); m2["package"] = "t2v8f_stage2_overlay"
    faults.append((m2, "NOT_A_STAGE3_MANIFEST"))
    m3 = json.loads(json.dumps(m)); m3["scenes"][4]["where"] = "kit"
    faults.append((m3, f"BAD_WHERE {m['scenes'][4]['scene'][:8]}"))
    m4 = json.loads(json.dumps(m)); m4["scenes"][5]["idx"] = 7
    faults.append((m4, f"BAD_IDX {m['scenes'][5]['scene'][:8]}"))
    m5 = json.loads(json.dumps(m)); m5["scenes"][6]["starts"] = [m5["scenes"][6]["n_ticks"] - 60]
    faults.append((m5, f"BAD_STARTS {m['scenes'][6]['scene'][:8]}"))
    m6 = json.loads(json.dumps(m)); m6["scenes"][7]["slot"] = "launch_1"
    faults.append((m6, f"NOT_A_GUARD_SCENE {m['scenes'][7]['scene'][:8]}"))
    for bad, want in faults:
        got = D3.check_manifest3(bad if mut != "not_applied" else m)
        assert any(g.startswith(want) for g in got), f"{want} was not named: {got}"
    return {"faults": len(faults)}


# ============================================================================================================================ the rehearsal fixture (3 guard scenes, layered on the stage-2 fixture's package)
def fixture3():
    if "fx3" in _C:
        return _C["fx3"]
    fx = T.fixture()
    plan = json.load(open(PLAN3, encoding="utf-8"))
    root = f"{fx['root']}/s3"
    src, scs, refs = f"{root}/src", [], {}
    for k, s in enumerate(plan["scenes"][:3]):
        sid = s["scene"]
        for f in s["files"]:
            if f["member"] != "volume.nurec":
                T.copy_small(f["location"], f"{src}/{sid}/{MK.REL[f['member']]}")
        open(f"{src}/{sid}/volume.nurec", "wb").write(b"PLACEHOLDER-0000\n"[:16])
        rng = np.random.default_rng(100 + k)
        np.savez_compressed(f"{src}/{sid}/ref_frames.npz", **{f"f{i}": (rng.random((36, 64, 3)) * 255).astype(np.uint8) for i in D.REF_IDX})
        files = [{"member": mem, "bytes": os.path.getsize(f"{src}/{sid}/{rel}"), "sha256": MK.sha256_file(f"{src}/{sid}/{rel}"), "location": f"{src}/{sid}/{rel}"} for mem, rel in MK.REL.items()]
        scs.append(dict(s, files=files))
        refs[sid] = {"path": f"{src}/{sid}/ref_frames.npz", "bytes": os.path.getsize(f"{src}/{sid}/ref_frames.npz"), "sha256": MK.sha256_file(f"{src}/{sid}/ref_frames.npz")}
    json.dump({"scenes": scs, "starts_sha256": plan["starts_sha256"]}, open(f"{root}/fx_plan3.json", "w"), indent=1)
    json.dump(refs, open(f"{root}/refs3.json", "w"), indent=1)
    res = M3.build(f"{root}/fx_plan3.json", f"{root}/refs3.json", f"{root}/out3", f"{root}/stage3", verify_src=True, plan_root=src, fixture_tag="_fx3", requires=M3.stage2_requires())
    with tarfile.open(res["tar"]) as tf:
        tf.extractall(fx["x"], filter="data")                                          # pkg/MANIFEST3.json, pkg/code/t2v8f_stage3.py, pkg/jobs/stage3.cmd, beside stage 2's files
    _C["fx3"] = {"fx": fx, "root": root, "src": src, "ids": [s["scene"] for s in scs], "build": res, "refs": refs}
    return _C["fx3"]


def new_scenes3(f3, tag):
    d = f"{f3['root']}/scenes3_{tag}_{time.strftime('%H%M%S')}"
    for sid in f3["ids"]:
        shutil.copytree(f"{f3['src']}/{sid}", f"{d}/{sid}")
    return d


def new_pull3(f3, tag):
    p = f"{f3['root']}/pull3_{tag}_{time.strftime('%H%M%S')}"
    assert not os.path.exists(p)
    return p


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(glob.glob(f"{root}/**/*", recursive=True)):
        if os.path.isfile(p):
            h.update(os.path.relpath(p, root).replace("\\", "/").encode())
            h.update(MK.sha256_file(p).encode())
    return h.hexdigest()


def run_s3(f3, pull, scenes3, extra=(), manifest=None, busy=None, timeout=1500):
    fx = f3["fx"]
    cmd = [PY, "-I", "-u", f"{fx['pkg']}/code/t2v8f_stage3.py", "all", "--pkg", fx["pkg"], "--kit", fx["kit"], "--hf", fx["hf"], "--pull", pull, "--scenes3", scenes3,
           "--manifest3", manifest or f"{fx['pkg']}/MANIFEST3.json", "--expect-scenes", "3", "--busy-file", busy or f"{f3['root']}/NO_STAGE2_RUNNING.txt", "--stub", "--inproc", *extra]
    p = subprocess.run(cmd, env=T1.clean_env(), capture_output=True, text=True, timeout=timeout)
    os.makedirs(os.path.dirname(pull), exist_ok=True)
    with open(f"{pull}.stdout.txt", "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode


def res3(pull):
    return json.load(open(f"{pull}/STAGE3_RESULT.json", encoding="utf-8"))


def positive():
    if "pos" not in _C:
        f3 = fixture3()
        s3, pull = new_scenes3(f3, "pos"), new_pull3(f3, "pos")
        before = tree_hash(s3)
        t0 = time.time()
        rc = run_s3(f3, pull, s3)
        _C["pos"] = (pull, s3, rc, round(time.time() - t0, 1), before)
    return _C["pos"]


def check_positive(mut=None):
    """All three jobs PASS on the stub: STAGE3.done (never a STAGE2.* marker), 42 cells DONE, 84 records `deciding candidate` with scene idx 42..44, every ref npz checked, the scenes root is NOT the pull
    root and is byte-identical after the run (the driver reads scenes, never writes them)."""
    pull, s3, rc, secs, before = positive()
    r = res3(pull)
    assert rc == 0, f"driver rc {rc}: {open(pull + '.stdout.txt', encoding='utf-8').read()[-1500:]}"
    assert os.path.exists(f"{pull}/STAGE3.done") and not os.path.exists(f"{pull}/STAGE3.fail") and not os.path.exists(f"{pull}/STAGE3.running")
    assert not glob.glob(f"{pull}/STAGE2*"), "a stage-2 marker or result in the stage-3 pull"
    assert r["gates"] == {"preflight": "PASS", "gate": "PASS", "rollouts": "PASS"} and r["overall"] == "COMPLETE" and r["stage"] == "stage3", r["gates"]
    info = r["steps"]["preflight"]["info"]
    assert info["ref_frames_ok"] == info["ref_frames"] == 3 and info["scene_files_ok"] == info["scene_files"] == 9 and info["overlay_files_ok"] == info["overlay_files"]
    assert info["scenes_root"] == s3.replace("\\", "/") and info["pull_root"] == pull.replace("\\", "/") and info["floor_checked_on"] == info["pull_root"] and not info["pull_root"].startswith(info["scenes_root"])
    done = sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))
    assert len(done) == 42 == r["steps"]["rollouts"]["n_cells_planned"] + (1 if mut == "cell_dropped" else 0), len(done)
    recs = [json.loads(x) for f in glob.glob(f"{pull}/rollouts/**/records.jsonl", recursive=True) for x in open(f, encoding="utf-8")]
    assert len(recs) == 84 and {x["purpose"] for x in recs} == {"deciding candidate"} and {int(x["scene"]) for x in recs} == {42, 43, 44}
    assert tree_hash(s3) == before, "the driver changed a file under the scenes root"
    g = json.load(open(f"{pull}/GATE_SCENES.json", encoding="utf-8"))
    assert len(g["rows"]) == 6 and all(x["finite"] and x["max_abs_diff_m"] <= 1e-6 for x in g["rows"]) and not g["failures"]
    return {"seconds": secs, "cells": len(done), "records": len(recs)}


def refused3(tag, mutate, extra=(), manifest=None, busy=None):
    f3 = fixture3()
    s3 = new_scenes3(f3, tag)
    pull = new_pull3(f3, tag)
    undo = mutate(f3, s3)
    try:
        rc = run_s3(f3, pull, s3, extra, manifest=manifest, busy=busy)
    finally:
        if undo:
            undo()
    return pull, rc, res3(pull)


def only_refusal(pull, rc, r, want, prefix=False):
    assert rc == 1 and os.path.exists(f"{pull}/STAGE3.fail") and not os.path.exists(f"{pull}/STAGE3.done"), (rc, os.listdir(pull))
    assert list(r["steps"]) == ["preflight"] and r["overall"] == "INFRASTRUCTURE_FAILURE", list(r["steps"])
    got = r["steps"]["preflight"]["failures"]
    assert (len(got) == len(want) and all(g.startswith(w) for g, w in zip(got, want))) if prefix else got == want, f"failures {got} != {want}"
    assert not os.path.exists(f"{pull}/rollouts") and not os.path.exists(f"{pull}/GATE_SCENES.json"), "a cell or the gate ran after the refusal"


def check_missing_npz(mut=None):
    """A scene has no `ref_frames.npz` (a decoy mp4 beside it): REFUSED, named, before the scene hash pass, no gate, no cell."""
    f3 = fixture3()
    miss = f3["ids"][1]

    def mutate(_f, s3):
        open(f"{s3}/{miss}/camera_front_wide_120fov.mp4", "wb").write(b"not an mp4\n")
        if mut != "not_applied":
            os.replace(f"{s3}/{miss}/ref_frames.npz", f"{s3}/{miss}/ref_frames.npz.aside_by_test")
    pull, rc, r = refused3("nonpz", mutate)
    only_refusal(pull, rc, r, [f"MISSING_REF_FRAMES scenes2/{miss}/ref_frames.npz"])
    assert "scene_hash_skipped" in r["steps"]["preflight"]["info"]
    return {"rc": rc}


def check_changed_byte(mut=None):
    """One byte of a scene's volume flipped (same size): `HASH_MISMATCH .../volume.nurec` and only that, nothing ran."""
    f3 = fixture3()
    one = f3["ids"][2]

    def mutate(_f, s3):
        if mut != "not_applied":
            p = f"{s3}/{one}/volume.nurec"
            b = bytearray(open(p, "rb").read())
            b[3] ^= 0x01
            open(p, "wb").write(bytes(b))
    pull, rc, r = refused3("flip", mutate)
    only_refusal(pull, rc, r, [f"HASH_MISMATCH scenes2/{one}/volume.nurec"])
    return {"rc": rc}


def check_stage2_manifest(mut=None):
    """The STAGE-2 manifest handed to the stage-3 driver is REFUSED by name (the wrong package cannot run as stage 3)."""
    f3 = fixture3()
    man = f"{f3['fx']['pkg']}/MANIFEST3.json" if mut == "not_applied" else f"{f3['fx']['pkg']}/MANIFEST2.json"
    pull, rc, r = refused3("m2", lambda *_: None, manifest=man)
    only_refusal(pull, rc, r, ["NOT_A_STAGE3_MANIFEST"], prefix=True)
    return {"rc": rc}


def check_busy_file(mut=None):
    """A stage-2 heartbeat file present: REFUSED (box 2 runs one job at a time)."""
    f3 = fixture3()
    busy = f"{f3['root']}/STAGE2_busy_{time.strftime('%H%M%S')}.running"
    if mut != "not_applied":
        open(busy, "w").write("pid=1 start=x phase=rollouts\n")
    pull, rc, r = refused3("busy", lambda *_: None, busy=busy)
    only_refusal(pull, rc, r, ["STAGE2_STILL_RUNNING"], prefix=True)
    return {"rc": rc}


def check_requires(mut=None):
    """MANIFEST3 pins the stage-2 files this driver imports; a box-2 copy of the stage-2 driver that is another version is REFUSED, named."""
    f3 = fixture3()
    m = json.load(open(f"{f3['fx']['pkg']}/MANIFEST3.json", encoding="utf-8"))
    if mut != "not_applied":
        m["requires"]["code/t2v8f_stage2.py"]["sha256"] = "0" * 64
    p = f"{f3['root']}/MANIFEST3_req_{time.strftime('%H%M%S')}.json"
    json.dump(m, open(p, "w", encoding="utf-8", newline="\n"), indent=1)
    pull, rc, r = refused3("req", lambda *_: None, manifest=p)
    only_refusal(pull, rc, r, ["HASH_MISMATCH_STAGE2_FILE code/t2v8f_stage2.py"], prefix=True)
    return {"rc": rc}


# ============================================================================================================================ two roots: the floor is the PULL drive's
def two_root_run(free_pull_gib, free_scenes_gib, mutated_driver=False):
    f3 = fixture3()
    s3, pull = new_scenes3(f3, "two"), new_pull3(f3, "two")
    fx = f3["fx"]
    real, g = shutil.disk_usage, 2**30
    probes = []

    def spy(path):
        p = str(path).replace("\\", "/")
        probes.append(p)
        r = real(path)
        free = free_pull_gib if p.startswith(pull.replace("\\", "/")) else free_scenes_gib if p.startswith(s3.replace("\\", "/")) else None
        return r if free is None else shutil._ntuple_diskusage(int(1000 * g), int(1000 * g - free * g), int(free * g))
    saved = D.preflight2
    if mutated_driver:                                                                   # an implementation that probes the SCENES drive for the floor
        src = open(D.__file__, encoding="utf-8").read()
        new = src.replace("pull_free = shutil.disk_usage(ctx.pull).free / 2**30", "pull_free = shutil.disk_usage(ctx.scenes2).free / 2**30")
        assert new != src, "the mutation did not apply: the line is not in the stage-2 driver"
        mod = types.ModuleType("t2v8f_stage2_mut")
        mod.__file__ = D.__file__
        exec(compile(new, D.__file__, "exec"), mod.__dict__)                             # noqa: S102
        D.preflight2 = mod.preflight2
    try:
        with mock.patch.dict(os.environ), mock.patch("shutil.disk_usage", spy):
            args = D3.build_parser().parse_args(["all", "--pkg", fx["pkg"], "--kit", fx["kit"], "--hf", fx["hf"], "--pull", pull, "--scenes3", s3, "--manifest3", f"{fx['pkg']}/MANIFEST3.json",
                                                 "--expect-scenes", "3", "--busy-file", f"{f3['root']}/NO_STAGE2_RUNNING.txt", "--stub"])
            ctx = D3.Ctx3(args)
            out = D3.preflight3(ctx)
    finally:
        D.preflight2 = saved
    return out, probes, s3, pull


def check_two_roots(mut=None):
    """The scenes root and the pull root are two roots. (A) the PULL reports 9 GiB free (floor 10) while the scenes drive reports 500: REFUSED `DISK_WILL_RUN_LOW`, before any scene hash.
    (B) the pull reports 500 while the scenes drive reports 1: PASS, every scene file read under the scenes root, floor decided on the pull. RED: a driver that probes the scenes drive passes (A)."""
    a, probes, s3, pull = two_root_run(9.0, 500.0, mutated_driver=(mut == "wrong_drive"))
    assert a["verdict"] == "REFUSED" and any(f.startswith("DISK_WILL_RUN_LOW") for f in a["failures"]), f"(A) the pull drive is below the floor: {a['verdict']} {a['failures'][:2]}"
    assert "scene_hash_skipped" in a["info"] and any(p.startswith(pull.replace("\\", "/")) for p in probes), "(A) the pull drive was never probed"
    b, _, s3b, pullb = two_root_run(500.0, 1.0)
    assert b["verdict"] == "PASS", f"(B) a nearly full scenes drive must not refuse: {b['failures'][:3]}"
    assert b["info"]["scene_files_ok"] == b["info"]["scene_files"] == 9 and b["info"]["scenes_root"] == s3b.replace("\\", "/") and b["info"]["floor_checked_on"] == pullb.replace("\\", "/")
    return {"A": [f[:60] for f in a["failures"]], "B": b["verdict"]}


# ============================================================================================================================ resume, batch, footprint, artifact, pool
def check_resume(mut=None):
    """A relaunch on a finished pull: every DONE cell is skipped, one DONE removed is the one cell re-run (DONE again). RED: nothing removed."""
    f3 = fixture3()
    pull0, s3, rc, _, _ = positive()
    assert rc == 0
    pull = f"{pull0}_resume_{time.strftime('%H%M%S')}"
    shutil.copytree(pull0, pull)
    if mut != "none_removed":
        victim = sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))[0]
        os.remove(victim)
    rc2 = run_s3(f3, pull, s3)
    assert rc2 == 0, open(pull + ".stdout.txt", encoding="utf-8").read()[-800:]
    st = res3(pull)["steps"]["rollouts"]["state"]
    done, skipped = sum(v["done"] for v in st.values()), sum(v["skipped_done"] for v in st.values())
    assert (done, skipped) == (1, 41), f"re-run {done}, skipped {skipped}"
    return {"rerun": done, "skipped": skipped}


def check_sftp(mut=None):
    """The put batch of the fixture: `-mkdir /C:/t2v8f` and `/C:/t2v8f/scenes3`, a `put` per file (volume, rig, tracks, npz) into `/C:/t2v8f/scenes3/<id>/...`; with the box's sizes a complete file is
    skipped, a short one `reput`, a missing one `put`. RED: a batch built for the stage-2 root."""
    f3 = fixture3()
    src = json.load(open(f"{f3['root']}/out3/SCENE_SOURCES3_fx3.json", encoding="utf-8"))
    root = "/D:/t2v8f/scenes2" if mut == "stage2_root" else M3.BOX2_SCENES_SFTP
    p = f"{f3['root']}/batch_{time.strftime('%H%M%S')}.sftp"
    M3.write_put_batch3(src, p, root=root)
    lines = open(p, encoding="utf-8").read().splitlines()
    assert lines[0] == "-mkdir /C:/t2v8f" and lines[1] == "-mkdir /C:/t2v8f/scenes3", lines[:2]
    puts = [ln for ln in lines if ln.startswith("put ")]
    assert len(puts) == len(src) == 12 and all(ln.split(" ", 2)[2].startswith("/C:/t2v8f/scenes3/") for ln in puts)
    assert sum(1 for ln in puts if ln.endswith("/ref_frames.npz")) == 3 and sum(1 for ln in puts if ln.endswith("/volume.nurec")) == 3
    rs = {}
    for k, s in enumerate(src):
        rs[f"{M3.BOX2_SCENES_SFTP}/{s['scene']}/{s['rel']}".lower()] = s["bytes"] if k % 3 == 0 else (s["bytes"] - 1 if k % 3 == 1 else None)
    rs = {k: v for k, v in rs.items() if v is not None}
    info = M3.write_put_batch3(src, p, remote_sizes=rs)
    assert info == {"batch": p, "put": 4, "reput": 4, "skipped_complete": 4}, info
    return info


def check_footprint(mut=None):
    """The per-drive numbers: NTFS 4 KiB rounding, and for 20 scenes the pull's deciding pass is 20 x 10 x 6 + 20 x 4 + 24 = 1,304 MiB, the beside pass 20 x 4 x 6 + 2 = 482 MiB; `D.pass_need_mib` reads them."""
    assert M3.ntfs_alloc(1) == 4096 and M3.ntfs_alloc(4096) == 4096 and M3.ntfs_alloc(4097) == 8192
    sc = [{"files": [{"bytes": 10_000_000}], "ref_frames": {"bytes": 9_000_000}} for _ in range(20)]
    fp = M3.footprint3(sc, 100_000)
    assert fp["pass_mib"] == {"deciding": 1304 + (1 if mut == "off_by_one" else 0), "beside": 482}, fp["pass_mib"]
    assert fp["box2_C_scenes_mib"] == round(20 * (M3.ntfs_alloc(10_000_000) + M3.ntfs_alloc(9_000_000)) / 2**20, 1)
    ctx = types.SimpleNamespace(pull=f"{TR.scratch('fp')}/pull", man2={"footprint": fp})
    ctx.live_scenes = lambda: [{"idx": 42 + k, "scene": f"s{k:07d}-0000-0000-0000-000000000000"} for k in range(20)]
    assert abs(D.pass_need_mib(ctx, "deciding") - 1304) < 1e-9
    return fp["pass_mib"]


def check_artifact(mut=None):
    """The shipped tar, opened: sha256 = its .sha256; exactly pkg/MANIFEST3.json, pkg/code/t2v8f_stage3.py, pkg/jobs/stage3.cmd (no npz, no mp4, no stage-2 file); the driver equals `code/t2v8f_stage3.py`;
    MANIFEST3 lists 20 guard scenes with an npz entry each, 40 starts = GUARD_STARTS.json, `requires` = overlay 2's driver sha256 33b65b19...; the files list matches the members."""
    assert os.path.isfile(TAR3), f"{TAR3} does not exist: build it with make_stage3.py"
    assert MK.sha256_file(TAR3) == open(TAR3 + ".sha256").read().split()[0]
    with tarfile.open(TAR3) as tf:
        names = tf.getnames()
        m = json.loads(tf.extractfile("pkg/MANIFEST3.json").read())
        drv = tf.extractfile("pkg/code/t2v8f_stage3.py").read()
        cmd = tf.extractfile("pkg/jobs/stage3.cmd").read()
    want = ["pkg/MANIFEST3.json", "pkg/code/t2v8f_stage3.py", "pkg/jobs/stage3.cmd"] + (["pkg/code/t2v8f_stage2.py"] if mut == "extra" else [])
    assert sorted(names) == sorted(want[:3]) and len(names) == len(want), names
    assert drv == open(f"{CODE}/t2v8f_stage3.py", "rb").read() and cmd == open(f"{T2}/stage3_src/jobs/stage3.cmd", "rb").read() and b"\r" not in drv + cmd
    assert m["stage"] == 3 and len(m["scenes"]) == 20 and m["totals"]["n_starts"] == 40 and m["starts_sha256"] == json.load(open(GS, encoding="utf-8"))["starts_sha256"]
    assert all(s["ref_frames"]["sha256"] and s["ref_frames"]["frames"] == [0, 60, 120] and s["where"] == "scenes3" and s["slot"].startswith("guard") for s in m["scenes"])
    assert m["requires"]["code/t2v8f_stage2.py"]["sha256"] == "33b65b19a563b8fe275de5e46e0f569db6487bd7fdf40b7eab34e021094f4898", m["requires"]
    assert MK.sha256_file(f"{CODE}/t2v8f_stage2.py") == m["requires"]["code/t2v8f_stage2.py"]["sha256"], "the dev stage-2 driver changed"
    for e in m["files"]:
        assert hashlib.sha256({"code/t2v8f_stage3.py": drv, "jobs/stage3.cmd": cmd}[e["path"]]).hexdigest() == e["sha256"]
    src = json.load(open(f"{S3}/SCENE_SOURCES3.json", encoding="utf-8"))
    assert len(src) == 80 and sum(s["bytes"] for s in src) == m["totals"]["scene_bytes"] + m["totals"]["ref_frames_bytes"]
    return {"scenes": 20}


def fake_cell(root, scene8, idx, name="C0_s0", cond="T2"):
    d = f"{root}/rollouts/{scene8}/objects/{cond}/{name}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/records.jsonl", "w").write("{}\n")
    open(f"{d}/scene_{idx}.npz", "wb").write(b"x")


def check_pool(mut=None):
    """The pool copies both roots' per-cell files and REFUSES a clash: the same scene idx in both roots, the same cell path, or an idx outside its stage's range. RED: a stage-3 scene with a stage-2 idx."""
    base = TR.scratch("pool")
    r2, r3, dest = f"{base}/r2", f"{base}/r3", f"{base}/pool"
    fake_cell(r2, "aaaaaaaa", 3)
    fake_cell(r3, "bbbbbbbb", 3 if mut == "idx_reused" else 42)
    try:
        info = SC3.pool(r2, r3, dest)
    except ValueError as e:
        assert "POOL_CLASH scene idx" in str(e), str(e)
        raise AssertionError(f"pool refused: {e}") from e
    assert info["n_cells"] == 2 and info["idx2"] == [3] and info["idx3"] == [42] and len(glob.glob(f"{dest}/*/rollouts/*/objects/*/*/scene_*.npz")) == 2
    for bad_r3, why in ((("aaaaaaaa", 42), "POOL_CLASH the same cell"), (("cccccccc", 7), "POOL_RANGE")):
        rb = f"{base}/rb_{bad_r3[0]}"
        fake_cell(rb, *bad_r3)
        try:
            SC3.pool(r2, rb, f"{base}/pool_{bad_r3[0]}")
            raise AssertionError(f"the pool accepted: {why}")
        except ValueError as e:
            assert why in str(e), str(e)
    return info


def check_score_end_to_end(mut=None):
    """`t2v8f_stage3_score.py` as a process: stage-1's MEASURED pull (idx 0/1, one moving + one launch start) as the 'stage-2' root and the rehearsal's records (idx 42..44) as the 'stage-3' root: both
    products are written, both start classes appear in the pooled families, the v0 cross-check holds."""
    f3 = fixture3()
    pull, _, rc, _, _ = positive()
    assert rc == 0
    out = f"{TR.scratch('score3')}/score"
    src3 = f"{f3['root']}/out3/SCENE_SOURCES3_fx3.json"
    p = subprocess.run([PY, "-I", f"{CODE}/t2v8f_stage3_score.py", "--root2", f"{T2}/stage1/pull_box2/pull/plumbing", "--root3", f"{pull}/rollouts", "--sources3", src3, "--out-dir", out],
                       env=T1.clean_env(), capture_output=True, text=True, timeout=1500)
    assert p.returncode == 0, p.stdout[-600:] + p.stderr[-1500:]
    fam = json.load(open(f"{out}/FAMILIES_ALL62.json", encoding="utf-8"))
    m1 = json.load(open(f"{out}/M1_LAUNCH42.json", encoding="utf-8"))
    assert set(m1["classes"]) == {"launch", "moving"} and len(m1["starts"]) == 2, m1["starts"]
    assert len(fam["starts"]) == 2 + 6 and set(fam["classes"]) >= {"moving"} and fam["checks"]["ok"] and m1["checks"]["ok"]
    assert {v["scene_idx"] for v in fam["starts"].values()} >= {42, 43, 44}
    return {"starts": len(fam["starts"])}


# ============================================================================================================================ pytest
def test_draw():
    check_draw()


def test_classes():
    check_classes()


def test_refs_real():
    check_refs_real()


def test_real_guard():
    check_real_guard()


def test_manifest():
    check_manifest()


def test_positive():
    check_positive()


def test_missing_npz():
    check_missing_npz()


def test_changed_byte():
    check_changed_byte()


def test_stage2_manifest():
    check_stage2_manifest()


def test_busy_file():
    check_busy_file()


def test_requires():
    check_requires()


def test_two_roots():
    check_two_roots()


def test_resume():
    check_resume()


def test_sftp():
    check_sftp()


def test_footprint():
    check_footprint()


def test_artifact():
    check_artifact()


def test_pool():
    check_pool()


def test_score_end_to_end():
    check_score_end_to_end()
