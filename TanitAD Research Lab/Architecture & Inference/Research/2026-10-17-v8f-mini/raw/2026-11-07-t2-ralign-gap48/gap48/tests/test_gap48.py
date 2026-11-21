"""GAP48 tests (CPU, stub renderer, small fixtures). Run from t2v8f/gap48:

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest tests/test_gap48.py -q -p no:cacheprovider

Scratch goes to C: (E: is nearly full): GAP48_SCRATCH, default C:/Users/Admin/t2a_scratch/gap48_rehearsal. The stage-1 package already extracted there for the stage-1 rehearsal is only READ.
Arms (a RED arm is a deliberate fault; the control arm beside it must stay green):
  scene list      derived list == 62 minus the R-SCN exclusions, count 48; fixture inputs that change the count make `scenes48.main` refuse
  ticks           every 10th tick, every tick below 20
  onepass         the one-pass clip read equals `t2v8f_nocv2.read_ref_frame` bit for bit (synthetic clip with a size mismatch, and one real clip frame)
  rehearsal       the shipped job, built by the real builder, extracted from the real tar, run with the stub renderer on 2 scenes: markers, result, tokens, resume
  reader          the reader recomputes the cosines, labels a stub run
  MUTANT          the job with its render replaced by the real frame: cosine 1.0 on every tick, test_hook recorded, the reader flags IDENTITY_DETECTED and pools nothing; the control run does not
  refusals        a changed pinned-code hash, a damaged clip, a missing scene file each end in preflight.fail + GAP48.fail with a named reason and no scene is measured
  chain           syntax; a present busy marker is untouched and nothing is called; a present box-2 marker moves only OUR marker aside; no delete verbs
  prose           no banned word in the package's text files
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
G = os.path.dirname(HERE)
T2 = os.path.dirname(G)
sys.path.insert(0, G)
sys.path.insert(0, f"{T2}/code")
import make_gap48 as MK  # noqa: E402
import read_gap48 as RD  # noqa: E402
import scenes48 as S  # noqa: E402
import t2v8f_gap48 as J  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCRATCH = os.environ.get("GAP48_SCRATCH", "C:/Users/Admin/t2a_scratch/gap48_rehearsal").replace("\\", "/")
S1_ROOT = "C:/Users/Admin/t2a_scratch/stage1_rehearsal/1785337f93d3"
PKG, KIT = f"{S1_ROOT}/x0/pkg", f"{S1_ROOT}/shadow_kit"
BANNED = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn", "closed loop")
GIT_BASH = "C:/Program Files/Git/bin/bash.exe"
_C = {}


def bash_exe():
    return GIT_BASH if os.path.isfile(GIT_BASH) else shutil.which("bash")


def clean_env():
    e = {k: v for k, v in os.environ.items() if not k.startswith(("T2A_", "T2V8F_", "PYTHONPATH", "TANITEVAL_"))}
    e.update(CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    return e


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return J.sha256_file(p)


_N = [0]


def work():
    _N[0] += 1
    d = f"{SCRATCH}/run_{time.strftime('%H%M%S')}_{os.getpid()}_{_N[0]}"
    os.makedirs(d, exist_ok=False)
    return d


# ------------------------------------------------------------------------------------------------ fixtures: a tiny clip, a 2-scene fixture, the real builder + tar
def make_clip(path, n=600, w=96, h=54):
    import av
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with av.open(path, "w") as c:
        try:
            st = c.add_stream("libx264", rate=30)
        except Exception:                                                           # noqa: BLE001
            st = c.add_stream("mpeg4", rate=30)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        yy, xx = np.mgrid[0:h, 0:w]
        for i in range(n):
            img = np.stack([(xx * 2 + i) % 256, (yy * 4 + 3 * i) % 256, np.full((h, w), (i * 5) % 256)], -1).astype(np.uint8)
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            for p in st.encode(fr):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def fixture_scenes(root):
    """Two fixture scenes: placeholder scene files (sizes + sha256 recorded), one tiny clip each. -> (scenes48-style json path, scene roots)."""
    rows = []
    roots = {"scenes2": f"{root}/scenes2", "scenes3": f"{root}/scenes3"}
    ids = [("aaaaaaaa-1111-4111-8111-000000000001", "scenes2", 0), ("bbbbbbbb-2222-4222-8222-000000000002", "scenes3", 1)]
    for sid, where, idx in ids:
        files = []
        for rel in ("volume.nurec", "rig_trajectories.json", "extracted/sequence_tracks.json"):
            p = f"{roots[where]}/{sid}/{rel}"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            b = (f"PLACEHOLDER {sid} {rel}\n").encode()
            open(p, "wb").write(b)
            files.append({"rel": rel, "bytes": len(b), "sha256": sha256_bytes(b)})
        open(f"{roots[where]}/{sid}/ref_frames.npz", "wb").write(b"PLACEHOLDER-NPZ\n")
        clip = f"{root}/clips/{sid}.mp4"
        make_clip(clip)
        rows.append({"idx": idx, "scene": sid, "where": where, "files": files, "mp4": {"path": clip, "bytes": os.path.getsize(clip), "sha256": sha256_file(clip)}})
    doc = {"rule": "fixture", "n_all": 2, "n_excluded": 0, "n_kept": 2, "excluded": [], "by_root": {"kit": 0, "scenes2": 1, "scenes3": 1}, "inputs": {"sources_sha256": "fixture", "exclusions_sha256": "fixture"}, "scenes": rows}
    p = f"{root}/SCENES_FIXTURE.json"
    json.dump(doc, open(p, "w"), indent=1)
    return p, roots


def package():
    """The REAL builder on the fixture, then the REAL tar extracted (clips included in the fixture tar). -> dict(gap, roots, build, scenes)."""
    if "pkg" in _C:
        return _C["pkg"]
    assert os.path.isdir(PKG) and os.path.isdir(KIT), "the stage-1 rehearsal extraction is missing (see tests/test_t2v8f_stage1.py in t2v8f/)"
    root = work()
    sc, roots = fixture_scenes(f"{root}/fx")
    bd = f"{root}/build"
    res = MK.build(sc, bd, fixture=True, mp4_in_tar=True)
    ex = f"{root}/box"
    os.makedirs(ex)
    import tarfile
    with tarfile.open(res["tar"]) as tf:
        tf.extractall(ex, filter="data")
    _C["pkg"] = {"root": root, "gap": f"{ex}/gap48", "roots": roots, "build": bd, "res": res, "scenes": json.load(open(sc))["scenes"]}
    return _C["pkg"]


def run_job(gap, roots, pull, extra=(), hook=None, manifest=None, timeout=900):
    argv = ["all", "--pkg", PKG, "--gap", gap, "--kit", KIT, "--hf", f"{os.path.dirname(KIT)}/hf", "--scenes2", roots["scenes2"], "--scenes3", roots["scenes3"], "--pull", pull, "--stub", "--min-free-gb", "0.1", *extra]
    if manifest:
        argv += ["--manifest", manifest]
    code = f"{gap}/code/t2v8f_gap48.py"
    if hook is None:
        cmd = [PY, "-I", "-u", code, *argv]
    else:
        boot = f"import sys; sys.path.insert(0, r'{gap}/code'); import t2v8f_gap48 as G; G._TEST_RENDER_HOOK = {hook}; sys.exit(G.main({argv!r}))"
        cmd = [PY, "-I", "-u", "-c", boot]
    t0 = time.time()
    p = subprocess.run(cmd, env=clean_env(), capture_output=True, text=True, timeout=timeout)
    open(f"{pull}.stdout.txt", "w", encoding="utf-8").write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode, round(time.time() - t0, 1)


def res_of(pull):
    return json.load(open(f"{pull}/GAP48_RESULT.json", encoding="utf-8"))


def positive():
    if "pos" in _C:
        return _C["pos"]
    k = package()
    pull = f"{k['root']}/pull_pos"
    rc, secs = run_job(k["gap"], k["roots"], pull)
    _C["pos"] = {"pull": pull, "rc": rc, "secs": secs}
    return _C["pos"]


def mutant():
    if "mut" in _C:
        return _C["mut"]
    k = package()
    pull = f"{k['root']}/pull_mut"
    rc, secs = run_job(k["gap"], k["roots"], pull, hook="lambda img, ref: ref")
    _C["mut"] = {"pull": pull, "rc": rc, "secs": secs}
    return _C["mut"]


# ================================================================================================ scene list
def test_scene_list_is_62_minus_rscn_exclusions():
    d = S.derive()
    rows = json.load(open(S.SOURCES, encoding="utf-8"))
    all_ids = {r["scene"] for r in rows}
    ex = {v["scene"] for v in json.load(open(S.EXCLUSIONS, encoding="utf-8"))["excluded"].values()}
    kept = {s["scene"] for s in d["scenes"]}
    assert len(all_ids) == 62 and len(ex) == 14 and ex <= all_ids
    assert kept == all_ids - ex and len(kept) == 48 == d["n_kept"]
    assert not (kept & ex)
    assert {s["where"] for s in d["scenes"]} <= {"kit", "scenes2", "scenes3"}
    assert all({"volume.nurec", "rig_trajectories.json", "extracted/sequence_tracks.json"} <= {f["rel"] for f in s["files"]} for s in d["scenes"])
    built = json.load(open(f"{G}/SCENES48.json", encoding="utf-8"))
    assert [s["scene"] for s in built["scenes"]] == [s["scene"] for s in d["scenes"]] and built["n_kept"] == 48


def test_scene_list_refuses_a_changed_input_instead_of_shipping_another_count():
    k = package()
    rows = json.load(open(S.SOURCES, encoding="utf-8"))
    ex = json.load(open(S.EXCLUSIONS, encoding="utf-8"))
    ex["excluded"].pop(next(iter(ex["excluded"])))                                   # one exclusion dropped: 49 scenes
    p_ex = f"{k['root']}/ex_minus1.json"
    json.dump(ex, open(p_ex, "w"))
    d = S.derive(S.SOURCES, p_ex)
    assert d["n_kept"] == 49
    ex2 = json.load(open(S.EXCLUSIONS, encoding="utf-8"))
    ex2["excluded"]["zzzzzzzz"] = {"scene": "zzzzzzzz-0000-0000-0000-000000000000"}
    p_ex2 = f"{k['root']}/ex_unknown.json"
    json.dump(ex2, open(p_ex2, "w"))
    with pytest.raises(ValueError):
        S.derive(S.SOURCES, p_ex2)
    assert len(rows) == 206


def test_ticks_every_10th_or_every_tick_below_20():
    assert J.tick_list(199) == list(range(0, 199, 10)) and len(J.tick_list(199)) == 20
    assert J.tick_list(200) == list(range(0, 200, 10))
    assert J.tick_list(19) == list(range(19)) and J.tick_list(20) == [0, 10] and J.tick_list(1) == [0]


# ================================================================================================ the one-pass read is read_ref_frame's read
def test_onepass_equals_read_ref_frame_bit_for_bit():
    import t2v8f_nocv2 as N
    root = work()
    clip = f"{root}/clip.mp4"
    make_clip(clip)
    frames = [0, 3, 30, 31, 300, 597]
    for size in ((1920, 1080), (96, 54)):                                           # a resize and no resize
        one = J.ref_frames_onepass(clip, frames, size)
        for fr in frames:
            assert np.array_equal(one[fr], N.read_ref_frame(clip, fr, size)), (size, fr)
    with pytest.raises(RuntimeError, match="MP4_TOO_SHORT"):
        J.ref_frames_onepass(clip, [10, 9999], (96, 54))
    real = S.KIT_MP4
    if os.path.isfile(real):                                                        # one real 3840x2160 clip, the cheapest nonzero frame the job uses
        one = J.ref_frames_onepass(real, [0, 30], (1920, 1080))
        for fr in (0, 30):
            assert np.array_equal(one[fr], N.read_ref_frame(real, fr, (1920, 1080)))


# ================================================================================================ the rehearsal: markers, result, tokens, resume
def test_rehearsal_writes_markers_result_and_tokens():
    p = positive()
    pull = p["pull"]
    assert p["rc"] == 0, open(f"{pull}.stdout.txt", encoding="utf-8").read()[-3000:]
    names = set(os.listdir(pull))
    assert {"preflight.done", "GAP48.done", "GAP48_RESULT.json", "tokens"} <= names
    assert "GAP48.running" not in names and "GAP48.fail" not in names and "preflight.fail" not in names
    k = package()
    r = res_of(pull)
    assert r["stage"] == "gap48" and r["stub"] is True and r["test_hook"] is False and r["condition"] == "objects" and r["n_done"] == 2 and r["n_failed"] == 0 and r["overall"] == "ALL_SCENES_BANKED"
    assert r["read_equiv_check"]["bit_equal"] is True
    import torch
    for sc in k["scenes"]:
        s8 = sc["scene"][:8]
        assert f"scene_{s8}.done" in names and f"scene_{s8}.fail" not in names
        sr = r["scenes"][sc["scene"]]
        assert sr["status"] == "DONE" and sr["n_pairs"] == 20 and sr["ticks"] == list(range(0, 199, 10)) and sr["cam_frames"] == [3 * t for t in sr["ticks"]] and sr["condition"] == "objects"
        assert sr["actors"]["verdict"] == "ACCEPTED" and sr["frame_size_wh"] == [1920, 1080]
        t = torch.load(f"{pull}/{sr['tokens_file']}", map_location="cpu", weights_only=True)
        assert t["render"].dtype == torch.float16 and tuple(t["render"].shape) == (20, 40, 1024) and tuple(t["real"].shape) == (20, 40, 1024)
        assert sha256_file(f"{pull}/{sr['tokens_file']}") == sr["tokens_sha256"]
        assert max(abs(RD.cos(t["render"][i].numpy(), t["real"][i].numpy()) - sr["cosine"][i]) for i in range(20)) < 1e-9
        assert all(c < 0.999 for c in sr["cosine"])                                 # a stub render is not its real frame
    assert (r["scene_list_sha256"] or "") == json.load(open(f"{k['gap']}/MANIFEST_GAP48.json"))["scene_list_sha256"]


def test_rehearsal_resume_skips_banked_scenes():
    p = positive()
    k = package()
    pull = p["pull"]
    before = {f: os.path.getmtime(f"{pull}/tokens/{f}") for f in os.listdir(f"{pull}/tokens")}
    time.sleep(1.1)
    rc, _ = run_job(k["gap"], k["roots"], pull)
    assert rc == 0
    after = {f: os.path.getmtime(f"{pull}/tokens/{f}") for f in os.listdir(f"{pull}/tokens")}
    assert before == after, "a banked scene was measured again"
    assert os.path.isfile(f"{pull}/GAP48.done") and res_of(pull)["n_done"] == 2


def test_reader_recomputes_and_labels_a_stub_run():
    p = positive()
    k = package()
    man = f"{k['gap']}/MANIFEST_GAP48.json"
    o = RD.read(p["pull"], man)
    assert o["status"] == "STUB_OR_HOOK_NOT_A_MEASUREMENT" and any(x.startswith("STUB_RESULT") for x in o["problems"])
    o = RD.read(p["pull"], man, allow_stub=True)
    assert o["n_scenes_read"] == 2 and o["identity_scenes"] == [] and o["n_scenes_used"] == 2 and o["status"] == "STUB_OR_HOOK_NOT_A_MEASUREMENT"
    assert o["render_vs_real"]["cosine_pairs"]["n"] == 40 and o["render_vs_mean"]["cos_render_to_mean"]["n"] == 40
    assert o["mean_sha256"] == RD.MEAN_SHA and "yardstick" in o
    md = RD.render_md(o)
    assert "RESULT_GAP48" in md and "Estimator and limits" in md
    import re
    assert not re.search(BANNED_RE, md, re.I)


# ================================================================================================ the mutant: the render is replaced by the real frame
def test_mutant_render_equals_real_gives_cosine_1_and_the_reader_flags_identity():
    k = package()
    m = mutant()
    assert m["rc"] == 0, open(f"{m['pull']}.stdout.txt", encoding="utf-8").read()[-3000:]
    r = res_of(m["pull"])
    assert r["test_hook"] is True
    for sid, sr in r["scenes"].items():
        assert all(abs(c - 1.0) < 1e-12 for c in sr["cosine"]) and all(x == 0.0 for x in sr["rel_l2"]), (sid, sr["cosine"][:3])          # float64 rounding of a/(|a||a|): 1.0 within 1 ulp-scale
    o = RD.read(m["pull"], f"{k['gap']}/MANIFEST_GAP48.json", allow_stub=True)
    assert o["status"] == "IDENTITY_DETECTED"
    assert sorted(o["identity_scenes"]) == sorted(s["scene"][:8] for s in k["scenes"]) and o["n_scenes_used"] == 0
    assert "render_vs_real" not in o and any(x.startswith("TEST_HOOK_RESULT") for x in o["problems"])
    md = RD.render_md(o)
    assert "IDENTITY" in md and "excluded from every pooled figure" in md
    # the control beside it: the unmutated run is not flagged
    c = RD.read(positive()["pull"], f"{k['gap']}/MANIFEST_GAP48.json", allow_stub=True)
    assert c["status"] != "IDENTITY_DETECTED" and c["identity_scenes"] == []


def test_reader_flags_partial_identity_per_scene():
    k = package()
    import torch
    pull = f"{k['root']}/pull_partial"
    shutil.copytree(positive()["pull"], pull)
    sid = k["scenes"][0]["scene"]
    f = f"{pull}/tokens/{sid[:8]}.pt"
    t = torch.load(f, map_location="cpu", weights_only=True)
    t["render"][:5] = t["real"][:5]                                                 # five ticks identical, fifteen not
    torch.save(t, f)
    rj = res_of(pull)
    rj["scenes"][sid]["tokens_sha256"] = sha256_file(f)
    rj["scenes"][sid]["cosine"] = [RD.cos(t["render"][i].numpy(), t["real"][i].numpy()) for i in range(20)]
    rj["scenes"][sid]["rel_l2"] = [RD.rel_l2(t["render"][i].numpy(), t["real"][i].numpy()) for i in range(20)]
    json.dump(rj, open(f"{pull}/GAP48_RESULT.json", "w"))
    o = RD.read(pull, f"{k['gap']}/MANIFEST_GAP48.json", allow_stub=True)
    assert o["identity_scenes"] == [] and any(x.startswith("PARTIAL_IDENTITY") for x in o["flags"]) and o["per_scene"][sid]["cosine"]["n"] == 20


def test_reader_names_a_tampered_token_file():
    k = package()
    pull = f"{k['root']}/pull_tamper"
    shutil.copytree(positive()["pull"], pull)
    sid = k["scenes"][1]["scene"]
    with open(f"{pull}/tokens/{sid[:8]}.pt", "ab") as fh:
        fh.write(b"x")
    o = RD.read(pull, f"{k['gap']}/MANIFEST_GAP48.json", allow_stub=True)
    assert any(x.startswith("TOKENS_SHA_MISMATCH") for x in o["problems"]) and o["n_scenes_used"] == 1


# ================================================================================================ refusals: named, before any scene
def refuse(tag, mutate):
    k = package()
    gap = f"{k['root']}/gap_{tag}"
    shutil.copytree(k["gap"], gap)
    mutate(gap)
    pull = f"{k['root']}/pull_{tag}"
    rc, _ = run_job(gap, k["roots"], pull)
    return rc, pull


def test_refusal_changed_pinned_code_hash():
    def mut(gap):
        m = json.load(open(f"{gap}/MANIFEST_GAP48.json"))
        m["pinned_code"][0]["sha256"] = "0" * 64
        json.dump(m, open(f"{gap}/MANIFEST_GAP48.json", "w"), indent=1)
    rc, pull = refuse("pin", mut)
    names = set(os.listdir(pull))
    assert rc == 1 and "preflight.fail" in names and "GAP48.fail" in names and "GAP48.done" not in names and not [n for n in names if n.startswith("scene_")]
    assert any(x.startswith("HASH_MISMATCH pkg/code/") for x in res_of(pull)["steps"]["preflight"]["failures"])


def test_refusal_damaged_clip_and_missing_scene_file():
    def mut(gap):
        k = package()
        sid = k["scenes"][0]["scene"]
        with open(f"{gap}/mp4/{sid}/camera_front_wide_120fov.mp4", "ab") as fh:
            fh.write(b"x")
    rc, pull = refuse("clip", mut)
    f = res_of(pull)["steps"]["preflight"]["failures"]
    assert rc == 1 and any(x.startswith("SIZE_MISMATCH mp4/") for x in f) and "GAP48.fail" in os.listdir(pull) and not [n for n in os.listdir(pull) if n.startswith("scene_")]

    def mut2(gap):
        m = json.load(open(f"{gap}/MANIFEST_GAP48.json"))
        m["scenes"][1]["files"][1]["rel"] = "rig_trajectories_absent.json"
        json.dump(m, open(f"{gap}/MANIFEST_GAP48.json", "w"), indent=1)
    rc, pull = refuse("scenefile", mut2)
    f = res_of(pull)["steps"]["preflight"]["failures"]
    assert rc == 1 and any(x.startswith("MISSING ") and "rig_trajectories_absent.json" in x for x in f)


def test_refusal_wrong_scene_count_for_a_real_manifest():
    def mut(gap):
        m = json.load(open(f"{gap}/MANIFEST_GAP48.json"))
        m["fixture"] = False                                                         # a non-fixture manifest with 2 scenes is not the 48
        json.dump(m, open(f"{gap}/MANIFEST_GAP48.json", "w"), indent=1)
    rc, pull = refuse("count", mut)
    assert rc == 1 and any(x.startswith("SCENE_COUNT 2 != 48") for x in res_of(pull)["steps"]["preflight"]["failures"])


# ================================================================================================ the real package
def test_real_builder_asserts_48_and_records_the_clips(tmp_path_factory=None):
    k = package()
    bad = json.load(open(f"{G}/SCENES48.json", encoding="utf-8"))
    bad["scenes"] = bad["scenes"][:47]
    p = f"{k['root']}/scenes47.json"
    json.dump(bad, open(p, "w"))
    with pytest.raises(AssertionError, match="is not 62 - 14 = 48"):
        MK.build(p, f"{k['root']}/build47")
    man = json.load(open(f"{k['build']}/MANIFEST_GAP48.json"))
    assert man["fixture"] is True and man["n_scenes"] == 2
    real = f"{G}/build/MANIFEST_GAP48.json"
    if os.path.isfile(real):
        m = json.load(open(real))
        assert m["n_scenes"] == 48 and m["fixture"] is False and m["n_all"] == 62 and m["n_excluded"] == 14
        assert len([e for e in m["own_files"] if e["path"].startswith("mp4/")]) == 48
        assert {x["file"] for x in m["pinned_code"]} == set(J.PINNED_CODE)
        for x in m["pinned_code"]:
            assert sha256_file(f"{T2}/code/{x['file']}") == x["sha256"] == sha256_file(f"{PKG}/code/{x['file']}")
        b = json.load(open(f"{G}/build/BUILD_RESULT_GAP48.json"))
        assert sha256_file(f"{G}/build/t2v8f_gap48.tar") == b["tar_sha256"]
        assert open(f"{G}/build/t2v8f_gap48.tar.sha256").read().split()[0] == b["tar_sha256"]
        assert sha256_file(f"{G}/t2v8f_gap48.py") == b["job_sha256"], "the shipped job differs from gap48/t2v8f_gap48.py: rebuild"
        lines = open(f"{G}/build/put_mp4.sftp").read().split("\n")
        assert len([x for x in lines if x.startswith("put ")]) == 48


def test_job_cmd_is_lf_only_and_has_no_labels():
    c = MK.JOB_CMD
    cmds = [ln for ln in c.split("\n") if ln and not ln.startswith("rem ")]
    assert "\r" not in c and not any(ln.startswith(":") or ln.lower().startswith("goto ") for ln in cmds)
    assert "t2v8f_gap48.py all" in c and "--pull D:/t2v8f/pull_gap48" in c and "pull_gap48" in c


# ================================================================================================ the chain
def run_chain(env_extra, timeout=120):
    e = dict(os.environ, **env_extra)
    return subprocess.run([bash_exe(), f"{G}/gap48_chain.sh"], env=e, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def stand_in_build(root):
    """A build dir with just enough for the chain's first checks (the chain reads BUILD_RESULT and the files' existence before it claims anything)."""
    d = f"{root}/chain_build"
    os.makedirs(d, exist_ok=True)
    for f, c in (("t2v8f_gap48.tar", b"x"), ("put_mp4.sftp", b""), ("expect_files.txt", b""), ("MANIFEST_GAP48.json", b"{}")):
        open(f"{d}/{f}", "wb").write(c)
    json.dump({"tar_sha256": "AB", "n_scenes": 48, "clip_bytes": 1, "footprint": {"d_free_gib_needed": 5.0}}, open(f"{d}/BUILD_RESULT_GAP48.json", "w"))
    return d


def test_chain_syntax_and_no_delete_verbs():
    p = subprocess.run([bash_exe(), "-n", f"{G}/gap48_chain.sh"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    txt = open(f"{G}/gap48_chain.sh", encoding="utf-8").read()
    assert b"\r" not in txt.encode()
    code = "\n".join(l for l in txt.split("\n") if not l.lstrip().startswith("#"))
    for verb in ("rm ", "rm -", "Remove-Item", "del ", "rmdir", "unlink"):
        assert verb not in code, verb
    assert "set -C" in code and "GAP48.done" in code and "GAP48.fail" in code and "BOX2_BUSY.txt" in code and "mv \"$B\"" in code


def test_chain_leaves_a_present_busy_marker_untouched_and_calls_nothing():
    root = work()
    marker = f"{root}/BOX2_BUSY.txt"
    open(marker, "w").write("owner=T2V8F stage5 BPC\nstart=x\n")
    mt0, sha0 = os.path.getmtime(marker), sha256_file(marker)
    log = f"{root}/ssh_calls.txt"
    stub = f"{root}/ssh_stub.sh"
    open(stub, "w", newline="\n").write(f'#!/bin/bash\necho "$@" >> "{log}"\necho B2EXISTS\n')
    p = run_chain({"GAP48_BUILD": stand_in_build(root), "GAP48_BUSY_MARKER": marker, "GAP48_ASIDE_DIR": f"{root}/aside", "GAP48_SSH": stub})
    assert p.returncode == 3 and "GAP48-BUSY" in p.stdout
    assert os.path.getmtime(marker) == mt0 and sha256_file(marker) == sha0
    assert not os.path.exists(log) and not os.path.exists(f"{root}/aside")


def test_chain_box2_marker_present_moves_only_our_marker_aside():
    root = work()
    marker = f"{root}/BOX2_BUSY.txt"                                               # absent: the chain may claim the dev-box marker
    log = f"{root}/ssh_calls.txt"
    stub = f"{root}/ssh_stub.sh"
    open(stub, "w", newline="\n").write(f'#!/bin/bash\necho "$@" >> "{log}"\necho B2EXISTS\n')
    p = run_chain({"GAP48_BUILD": stand_in_build(root), "GAP48_BUSY_MARKER": marker, "GAP48_ASIDE_DIR": f"{root}/aside", "GAP48_SSH": stub})
    assert p.returncode == 3 and "exists on box 2" in p.stdout
    assert not os.path.exists(marker), "our own dev-box marker must be moved aside"
    moved = os.listdir(f"{root}/aside")
    assert len(moved) == 1 and "box2busy" in moved[0]
    calls = open(log).read().strip().split("\n")
    assert len(calls) == 1 and "BOX2_BUSY.txt" in calls[0] and "Move-Item" not in calls[0], calls     # one claim attempt on box 2, no move of THEIR marker


def test_chain_mutant_without_noclobber_overwrites_the_marker_so_the_untouched_check_can_fail():
    root = work()
    marker = f"{root}/BOX2_BUSY.txt"
    open(marker, "w", newline="\n").write("owner=T2V8F stage5 BPC\nstart=x\n")
    sha0 = sha256_file(marker)
    src = open(f"{G}/gap48_chain.sh", encoding="utf-8").read()
    assert "( set -C; printf" in src
    mut = f"{root}/chain_mutant.sh"
    open(mut, "w", newline="\n").write(src.replace("( set -C; printf", "( printf"))
    log = f"{root}/ssh_calls.txt"
    stub = f"{root}/ssh_stub.sh"
    open(stub, "w", newline="\n").write(f'#!/bin/bash\necho "$@" >> "{log}"\necho B2EXISTS\n')
    e = dict(os.environ, GAP48_BUILD=stand_in_build(root), GAP48_BUSY_MARKER=marker, GAP48_ASIDE_DIR=f"{root}/aside", GAP48_SSH=stub)
    subprocess.run([bash_exe(), mut], env=e, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    changed = (not os.path.exists(marker)) or sha256_file(marker) != sha0
    assert changed, "the untouched-marker check could not tell a clobbering chain from the real one"


# ================================================================================================ prose
BANNED_RE = r"\b(void|superseded|retracted|formerly|withdrawn|correction)\b|no longer|closed loop"


def test_no_banned_word_in_the_packages_text():
    import re
    pat = re.compile(BANNED_RE, re.I)
    names = ("t2v8f_gap48.py", "make_gap48.py", "read_gap48.py", "scenes48.py", "gap48_chain.sh", "gap48_chain.cmd", "README.md", "HANDOFF_GAP48.md")
    for f in [f"{G}/{n}" for n in names if os.path.isfile(f"{G}/{n}")]:
        hit = pat.findall(open(f, encoding="utf-8").read())
        assert not hit, (f, hit)
    # the pattern itself must be able to fire (and must not fire on 'avoid')
    assert pat.search("this is void") and pat.search("no longer") and pat.search("a closed loop") and not pat.search("avoid it")
