"""The C0C-RA stage-5 package and its CPU rehearsal (stub renderer): the builder on the REAL stage-4 tar and on the stage-4 rehearsal fixture, the wrapper driver's named refusals, the banked driver
refusing the kind, and the stub rehearsal of the whole driver (preflight, gate, 24 cells) with a non-identity map and with the identity map (G-IDMAP against the fixture's stage-4 C0C cells).
Fixtures are the banked tests' own (`tests/test_t2v8f_stage4.fixture4`, imported read-only) built into this agent's scratch on E:. Synthetic maps only; no GAP48 file is opened.
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest ralign/tests/test_ralign_stage5.py -q --basetemp F:/TanitAD-scratch/ralign_pytest
Every refusal / RED arm is asserted next to its positive arm.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tarfile
import time

SCR = "E:/Projects/TanitAD-artifacts/scratch_ralign_t2v8f"
os.environ["T2V8F_S2_SCRATCH"] = f"{SCR}/fixtures"
os.environ["T2V8F_S1_SCRATCH"] = f"{SCR}/fixtures_s1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RA = os.path.dirname(HERE)
T2 = os.path.dirname(RA)
for _p in (RA, f"{T2}/code", f"{T2}/tests", HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pytest  # noqa: E402
import torch  # noqa: E402

import make_ralign_stage5 as MR  # noqa: E402
import ralign_common as K  # noqa: E402
import ralign_driver as RD  # noqa: E402
import ralign_fit as F  # noqa: E402
import ralign_idmap as ID  # noqa: E402
import ralign_lib as L  # noqa: E402
import t2v8f_stage5 as D5  # noqa: E402
import test_t2v8f_stage1 as T1  # noqa: E402
import test_t2v8f_stage2 as T  # noqa: E402
import test_t2v8f_stage3 as T3  # noqa: E402
import test_t2v8f_stage4 as T4  # noqa: E402

PY = T1.PY
_C = {}
_SEQ = itertools.count(1)
TAR4 = f"{T2}/stage4/t2v8f_stage4_overlay.tar"


# ------------------------------------------------------------------------------------------------------------------------------ synthetic maps (rehearsal only)
def write_synth_maps(d, a, b, status="FITTED"):
    os.makedirs(d, exist_ok=True)
    maps = {}
    for f in L.FOLDS:
        shas = L.save_map(f"{d}/MAP_{f}.pt", a, b, {"fold": f, "fold_scenes": [], "n_pairs": 0, "fold_digit": 0, "card_sha256": L.CARD_SHA256})
        maps[f] = {"file": f"MAP_{f}.pt", **shas, "fold_scenes": [], "n_pairs": 0}
    L.jdump({"card_sha256": L.CARD_SHA256, "status": status, "fold_digit": 0, "shape": list(L.SHAPE), "maps": maps, "applied_map_for_scene_in_fold": {"A": "B", "B": "A"}}, f"{d}/MAPS.json")
    return d


def maps_dir(kind, root):
    d = f"{root}/ra_maps_{kind}"
    if os.path.exists(f"{d}/MAPS.json"):
        return d
    if kind == "identity":
        F.write_identity(d, L.SHAPE)
    elif kind == "random":
        g = torch.Generator().manual_seed(5)
        write_synth_maps(d, 1.0 + 0.3 * torch.randn(L.SHAPE, generator=g), 0.2 * torch.randn(L.SHAPE, generator=g))
    elif kind == "perturbed":                                                  # a perturbed identity: a = 1 + 1e-3 everywhere
        write_synth_maps(d, torch.full(L.SHAPE, 1.001), torch.full(L.SHAPE, -0.0))
    else:
        raise KeyError(kind)
    return d


def fake_evidence(root, d, code_ok=True):
    """Hand-made M_ALIGN / G_IDMAP json for the BUILDER's binding checks only (the test is the author; they are not evidence of anything)."""
    idx = json.load(open(f"{d}/MAPS.json"))
    ma = {"verdict": "M-ALIGN-PASS", "gate_valid": True, "card_sha256": L.CARD_SHA256, "maps": {f: {"tensor_sha256": idx["maps"][f]["tensor_sha256"]} for f in "AB"}, "main": {"median_before": 0.7, "median_after": 0.93}}
    gi = {"verdict": "G-IDMAP-PASS", "code_sha256": {fn: (L.sha256_file(f"{RA}/{fn}") if code_ok else "0" * 64) for fn in ("ralign_policy.py", "ralign_lib.py")}}
    p1, p2 = f"{root}/fake_m_align_{next(_SEQ)}.json", f"{root}/fake_g_idmap_{next(_SEQ)}.json"
    L.jdump(ma, p1)
    L.jdump(gi, p2)
    return p1, p2


@contextlib.contextmanager
def patched():
    """The wrapper driver's rebinding of the banked driver's names, for one block (the banked module is shared by the whole test process)."""
    saved = (D5.CELL5, D5.check_arm5, D5.preflight5, D5.rollouts5)
    RD.patch()
    try:
        yield
    finally:
        D5.CELL5, D5.check_arm5, D5.preflight5, D5.rollouts5 = saved


def tar_manifest(tar):
    with tarfile.open(tar) as tf:
        return json.loads(tf.extractfile("pkg/MANIFEST5.json").read())


def build_real(root, kind, status, tag, subset=None, evidence=None):
    d = maps_dir(kind, root)
    out = f"{root}/out_{tag}_{next(_SEQ)}"
    return MR.build(out, f"{root}/stage_{tag}_{next(_SEQ)}", tag, d, status, *(evidence or (None, None)), src_tar4=TAR4, subset_launch=subset)


@pytest.fixture(scope="module")
def real_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("ra_real")).replace("\\", "/")


# ============================================================================================================================ the builder on the REAL stage-4 tar
def check_real_builds(root):
    idr = build_real(root, "identity", "IDMAP", "ID", subset=2)                         # the box-2 G-IDMAP package: 2 launch scenes
    m = tar_manifest(idr["tar"])
    assert idr["members"] == sorted(["pkg/MANIFEST5.json", "pkg/code/t2v8f_stage5.py", "pkg/code/t2v8f_stage5_cell.py", "pkg/code/ralign_lib.py", "pkg/code/ralign_policy.py", "pkg/code/ralign_cell.py",
                                     "pkg/code/ralign_driver.py", "pkg/ralign/MAP_A.pt", "pkg/ralign/MAP_B.pt", "pkg/ralign/MAPS.json", f"pkg/jobs/stage5_C0C-RA_ID.cmd"])
    assert len(m["scenes"]) == 2 and m["subset"]["first_launch_scenes"] == 2 and m["arm5"]["name"] == "C0C-RA" and m["arm5"]["run"] == "C0C-RA_ID" and m["arm5"]["ralign"]["status"] == "IDMAP"
    with patched():
        assert D5.check_manifest5(m, 2, 0) == [], D5.check_manifest5(m, 2, 0)
    banked = D5.check_manifest5(m, 2, 0)                                                # the BANKED driver (unpatched) refuses the kind by name
    assert any(x.startswith("ENCODER_UNSUPPORTED 'frozen_dinov3_ralign'") for x in banked), banked
    with tarfile.open(idr["tar"]) as tf:
        cmd = tf.extractfile("pkg/jobs/stage5_C0C-RA_ID.cmd").read().decode()
        for fn in ("t2v8f_stage5.py", "t2v8f_stage5_cell.py"):
            assert tf.extractfile(f"pkg/code/{fn}").read() == open(f"{T2}/code/{fn}", "rb").read(), f"{fn} is not byte-identical to code/"
        for fn in MR.SHIP:
            b = tf.extractfile(f"pkg/code/{fn}").read()
            assert b == open(f"{RA}/{fn}", "rb").read() and b"\r" not in b
    assert "pkg\\code\\ralign_driver.py all" in cmd and "t2v8f_stage5.py all" not in cmd and "--expect-launch 2 --expect-guard 0" in cmd and "pull5_C0C-RA_ID" in cmd and "\r" not in cmd
    assert idr["footprint"]["D_free_gib_needed_to_run_both_passes"] < 10.6 and not any(p.startswith("pkg/ckpt/") for p in idr["members"]), "no checkpoint ships: C0C-RA runs C0's own"
    assert {m["arm5"]["ckpt_file"]["C0C-RA_s0"], m["arm5"]["ckpt_file"]["C0C-RA_s1"]} == {"C0_s0.pt", "C0_s1.pt"}
    for k in ("tar_sha256", "footprint", "overlay_files", "manifest5_sha256", "requires"):               # what audit/stage5_chain.sh reads from BUILD_RESULT5_<RUN>.json
        assert k in idr, k
    for rel in ("code/t2v8f_c0c.py", "code/t2v8f_stage4.py", "refs/tool/v8f_t1_eval_ri.py", "code/t2v8f_stage2.py"):
        assert rel in idr["requires"], rel
    rh = build_real(root, "random", "REHEARSAL", "RH")                                  # the full 62 scenes
    mr = tar_manifest(rh["tar"])
    assert len(mr["scenes"]) == 62 and rh["n_starts"] == 121 and rh["n_cells"]["total"] == 248
    os.environ.pop(RD.ALLOW_REHEARSAL_ENV, None)
    with patched():
        assert any(x.startswith("RALIGN_REHEARSAL_NOT_ALLOWED") for x in D5.check_manifest5(mr)), "a rehearsal package must not pass without the explicit switch"
        os.environ[RD.ALLOW_REHEARSAL_ENV] = "1"
        try:
            assert D5.check_manifest5(mr) == [], D5.check_manifest5(mr)
        finally:
            os.environ.pop(RD.ALLOW_REHEARSAL_ENV, None)
    sf = mr["arm5"]["ralign"]["scene_fold"]
    assert len(sf) == 62 and all(sf[s["scene"]] == L.fold_of(s["scene"]) for s in mr["scenes"]) and set(sf.values()) == {"A", "B"}
    # the GATED binding
    d = maps_dir("random", root)
    for ev, why in (((None, None), "RALIGN_NOT_GATED"),):
        with pytest.raises(AssertionError, match=why):
            build_real(root, "random", "GATED", "GA", evidence=ev)
    ma, gi = fake_evidence(root, d)
    gated = build_real(root, "random", "GATED", "GA", evidence=(ma, gi))
    mg = tar_manifest(gated["tar"])
    assert mg["arm5"]["ralign"]["status"] == "GATED" and mg["arm5"]["ralign"]["gate"]["m_align_verdict"] == "M-ALIGN-PASS"
    with patched():
        assert D5.check_manifest5(mg) == [], D5.check_manifest5(mg)
    ma2, _ = fake_evidence(root, maps_dir("perturbed", root))
    with pytest.raises(AssertionError, match="RALIGN_GATE_MAP_MISMATCH"):
        build_real(root, "random", "GATED", "GA", evidence=(ma2, gi))                  # M_ALIGN was run on other maps
    _, gi_stale = fake_evidence(root, d, code_ok=False)
    with pytest.raises(AssertionError, match="RALIGN_GIDMAP_STALE"):
        build_real(root, "random", "GATED", "GA", evidence=(ma, gi_stale))           # G-IDMAP was run for other policy bytes
    with pytest.raises(AssertionError, match="RALIGN_MAPS_STATUS"):
        build_real(root, "random", "IDMAP", "X")                                      # IDMAP status with non-identity maps
    with pytest.raises(AssertionError, match="RALIGN_MAPS_STATUS"):
        build_real(root, "identity", "REHEARSAL", "X")                                # a non-identity status over the identity maps
    return {"members": len(idr["members"]), "cells": rh["n_cells"]["total"]}


def test_builder_on_the_real_stage4_tar(real_root):
    check_real_builds(real_root)


def test_stale_gidmap_is_refused_and_a_bypass_would_have_built(real_root, monkeypatch):
    d = maps_dir("random", real_root)
    ma, gi_stale = fake_evidence(real_root, d, code_ok=False)
    with pytest.raises(AssertionError, match="RALIGN_GIDMAP_STALE"):
        build_real(real_root, "random", "GATED", "GS", evidence=(ma, gi_stale))
    monkeypatch.setattr(MR, "check_evidence", lambda status, idx, a, b: {"status": status, "m_align_verdict": "M-ALIGN-PASS", "g_idmap_verdict": "G-IDMAP-PASS"})       # MUTANT: the binding removed
    assert build_real(real_root, "random", "GATED", "GS", evidence=(ma, gi_stale))["status"] == "GATED"


def planted_faults(m, mut=None):
    cp = lambda: json.loads(json.dumps(m))         # noqa: E731
    faults = []
    a = cp(); a["arm5"]["ralign"]["card_sha256"] = "0" * 64
    faults.append((a, "RALIGN_CARD_SHA"))
    b = cp(); b["arm5"]["ckpt_file"]["C0C-RA_s1"] = "C0_s0.pt"
    faults.append((b, "RALIGN_NOT_C0C_WEIGHTS"))
    c = cp(); k = next(iter(c["arm5"]["ralign"]["scene_fold"])); c["arm5"]["ralign"]["scene_fold"][k] = L.other_fold(c["arm5"]["ralign"]["scene_fold"][k])
    faults.append((c, "RALIGN_SCENE_FOLD"))
    d = cp(); d["arm5"]["ralign"]["maps"]["A"]["file_sha256"] = "1" * 64
    faults.append((d, "RALIGN_MAP_NOT_IN_OVERLAY"))
    e = cp(); e["arm5"]["ralign"]["status"] = "GATED"; e["arm5"]["ralign"]["gate"] = {"m_align_verdict": "M-ALIGN-PASS"}
    faults.append((e, "RALIGN_NOT_GATED"))
    f = cp(); f["arm5"]["ralign"]["fold_digit"] = 2
    faults.append((f, "RALIGN_FOLD_DIGIT"))
    g = cp(); g["arm5"]["encoder"]["kind"] = "frozen_dinov3"
    faults.append((g, "ENCODER_UNSUPPORTED"))
    h = cp(); h["arm5"].pop("ralign")
    faults.append((h, "RALIGN_BLOCK_MISSING"))
    with patched():
        os.environ[RD.ALLOW_REHEARSAL_ENV] = "1"
        try:
            for bad, want in faults:
                got = D5.check_manifest5(bad if mut != "not_applied" else m)
                assert any(x.startswith(want) for x in got), f"{want} was not named: {got}"
        finally:
            os.environ.pop(RD.ALLOW_REHEARSAL_ENV, None)
    return len(faults)


def test_wrapper_driver_names_every_planted_fault(real_root):
    m = tar_manifest(build_real(real_root, "random", "REHEARSAL", "RF")["tar"])
    assert planted_faults(m) == 8
    with pytest.raises(AssertionError, match="was not named"):
        planted_faults(m, "not_applied")


# ============================================================================================================================ the CPU rehearsal on the stage-4 fixture (stub renderer)
def f4():
    return T4.fixture4()


def ref4():
    """Stage 4's C0C on the same fixture (the reference of G-IDMAP). The disk floor is lowered to 2 GiB: E: holds this fixture and has ~7 GiB free (the driver's default floor is 10)."""
    if "ref4" not in _C:
        f = f4()
        s2, s3 = T.new_scenes2(f["fx"], "ra4ref"), T3.new_scenes3(f["f3"], "ra4ref")
        pull = f"{f['root']}/pull4_ra_ref_{time.strftime('%H%M%S')}"
        rc = T4.run_s4(f, pull, s2, s3, extra=("--min-free-gb", "2"))
        assert rc == 0, open(pull + ".stdout.txt", encoding="utf-8").read()[-1500:]
        _C["ref4"] = pull
    return _C["ref4"]


def build_fixture_variant(kind, status, tag):
    key = ("var", kind, status, tag)
    if key not in _C:
        fx = f4()
        res = MR.build(f"{fx['root']}/ra_out_{tag}", f"{fx['root']}/ra_stage_{tag}", tag, maps_dir(kind, fx["root"]), status, src_tar4=fx["build"]["tar"], n_launch=3, n_guard=3, fixture_tag=f"_fx{tag}")
        _C[key] = res
    return _C[key]


def extract_variant(res):
    """Extracts the variant's overlay into the fixture package (new paths + the same-named MANIFEST5.json and ralign/ files of the previous variant). Tests run one after another."""
    with tarfile.open(res["tar"]) as tf:
        tf.extractall(f4()["fx"]["x"], filter="data")


def run_ra(pull, s2, s3, passes="deciding,beside", allow_rehearsal=False, manifest=None, timeout=2400):
    fx = f4()["fx"]
    r = f4()["root"]
    pkg = fx["pkg"]
    cmd = [PY, "-I", "-u", f"{pkg}/code/ralign_driver.py", "all", "--pkg", pkg, "--kit", fx["kit"], "--hf", fx["hf"], "--pull", pull, "--scenes2", s2, "--scenes3", s3, "--manifest5", manifest or f"{pkg}/MANIFEST5.json",
           "--expect-launch", "3", "--expect-guard", "3", "--busy-file2", f"{r}/NO_STAGE2_RUNNING.txt", "--busy-file3", f"{r}/NO_STAGE3_RUNNING.txt", "--busy-file4", f"{r}/NO_STAGE4_RUNNING.txt",
           "--stub", "--inproc", "--passes", passes, "--min-free-gb", "2"]
    env = T1.clean_env()
    if allow_rehearsal:
        env[RD.ALLOW_REHEARSAL_ENV] = "1"
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    os.makedirs(os.path.dirname(pull), exist_ok=True)
    with open(f"{pull}.stdout.txt", "w", encoding="utf-8") as fh:
        fh.write(p.stdout + "\n---stderr---\n" + p.stderr)
    return p.returncode


def rehearse(kind, status, tag, passes="deciding,beside"):
    key = ("run", kind, status, tag, passes)
    if key not in _C:
        fx = f4()
        extract_variant(build_fixture_variant(kind, status, tag))
        s2, s3 = T.new_scenes2(fx["fx"], f"ra{tag}"), T3.new_scenes3(fx["f3"], f"ra{tag}")
        pull = f"{fx['root']}/pull5_ra_{tag}_{time.strftime('%H%M%S')}"
        t0 = time.time()
        rc = run_ra(pull, s2, s3, passes, allow_rehearsal=(status == "REHEARSAL"))
        _C[key] = (pull, rc, round(time.time() - t0, 1))
    return _C[key]


def recs_of(pull, tc, arm):
    return {(int(x["scene"]), int(x["f0"])): x for f in glob.glob(f"{pull}/rollouts/**/{tc}/{arm}/records.jsonl", recursive=True) for x in map(json.loads, open(f, encoding="utf-8"))}


def test_rehearsal_with_a_non_identity_map_runs_the_whole_driver_and_the_map_is_in_the_records():
    pull, rc, secs = rehearse("random", "REHEARSAL", "RH")
    r = json.load(open(f"{pull}/STAGE5_RESULT.json", encoding="utf-8"))
    assert rc == 0, open(pull + ".stdout.txt", encoding="utf-8").read()[-1800:]
    assert os.path.exists(f"{pull}/STAGE5.done") and r["gates"] == {"preflight": "PASS", "gate": "PASS", "rollouts": "PASS"} and r["overall"] == "COMPLETE" and r["arm"] == "C0C-RA", r["gates"]
    assert r["ralign"]["status"] == "REHEARSAL" and set(r["ralign"]["maps"]) == {"A", "B"} and not any(v["identity"] for v in r["ralign"]["maps"].values())
    done = sorted(glob.glob(f"{pull}/rollouts/*/objects/*/*/DONE"))
    assert len(done) == 24 and {os.path.basename(os.path.dirname(f)) for f in done} == {"C0C-RA_s0", "C0C-RA_s1"} and {os.path.basename(os.path.dirname(os.path.dirname(f))) for f in done} == {"T2", "RR"}
    c4 = ref4()
    for tc in ("T2", "RR"):
        ra, base = recs_of(pull, tc, "C0C-RA_s0"), recs_of(c4, tc, "C0C_s0")
        assert sorted(ra) == sorted(base) and len(ra) >= 6
        assert any(ra[k]["plans"] != base[k]["plans"] for k in ra), f"{tc}: the map did not change a single plan"
        for k, x in ra.items():
            st = x["extra"]
            assert all(e["ralign"]["map_fold"] == L.other_fold(e["ralign"]["scene_fold"]) and e["ralign"]["n_token_sets_mapped"] > 0 for e in st), "a plan was made without the map"
            assert all(e["composite"]["ri_tool_sha256"] for e in st), "the composite stamp is missing"
            assert {e["ralign"]["scene_fold"] for e in st} == {L.fold_of(x["scene_id"])}


def test_rehearsal_with_the_identity_map_is_stage4s_c0c_bit_for_bit_and_a_perturbed_identity_fails():
    c4 = ref4()
    pull, rc, _ = rehearse("identity", "IDMAP", "ID", "deciding")
    assert rc == 0, open(pull + ".stdout.txt", encoding="utf-8").read()[-1800:]
    man = f"{f4()['fx']['pkg']}/MANIFEST5.json"
    extract_variant(build_fixture_variant("identity", "IDMAP", "ID"))
    res = ID.compare(f"{pull}/rollouts", f"{c4}/rollouts", man, min_scenes=2)
    assert res["verdict"] == "G-IDMAP-PASS" and res["n_scenes"] == 6 and res["n_cells"] == 12 and res["n_records_equal"] > 0 and res["problems"] == [], res["problems"][:3]
    assert res["code_sha256"]["ralign_policy.py"] == L.sha256_file(f"{RA}/ralign_policy.py") and res["maps_status"] == "IDMAP"
    # RED: a perturbed identity (a = 1 + 1e-3), built as a rehearsal package, must FAIL the same comparison
    pull2, rc2, _ = rehearse("perturbed", "REHEARSAL", "PI", "deciding")
    assert rc2 == 0
    extract_variant(build_fixture_variant("identity", "IDMAP", "ID"))
    bad = ID.compare(f"{pull2}/rollouts", f"{c4}/rollouts", man, min_scenes=2)
    assert bad["verdict"] == "G-IDMAP-FAIL" and any(p.startswith("RECORDS_DIFFER") for p in bad["problems"]), bad["problems"][:3]
    # RED: stage 4's own cells compared with themselves are not evidence (the manifest must be the identity package's, and the map must have run)
    notid = ID.compare(f"{pull2}/rollouts", f"{c4}/rollouts", f"{f4()['fx']['pkg']}/MANIFEST5.json", min_scenes=2)
    assert notid["verdict"] == "G-IDMAP-FAIL"


def test_the_idmap_cli_writes_a_verdict_file(tmp_path):
    c4 = ref4()
    pull, rc, _ = rehearse("identity", "IDMAP", "ID", "deciding")
    extract_variant(build_fixture_variant("identity", "IDMAP", "ID"))
    out = str(tmp_path / "G_IDMAP.json")
    assert ID.main(["compare", "--ra-root", f"{pull}/rollouts", "--stage4-root", f"{c4}/rollouts", "--manifest", f"{f4()['fx']['pkg']}/MANIFEST5.json", "--out", out]) == 0
    assert json.load(open(out))["verdict"] == "G-IDMAP-PASS" and ID.main(["compare", "--ra-root", f"{pull}/rollouts", "--stage4-root", f"{c4}/rollouts", "--manifest", f"{f4()['fx']['pkg']}/MANIFEST5.json", "--out", out]) == 4


def test_preflight_refuses_a_rehearsal_package_without_the_switch_and_a_tampered_map():
    fx = f4()
    res = build_fixture_variant("random", "REHEARSAL", "RH")
    extract_variant(res)
    s2, s3 = T.new_scenes2(fx["fx"], "rano"), T3.new_scenes3(fx["f3"], "rano")
    pull = f"{fx['root']}/pull5_ra_noswitch_{time.strftime('%H%M%S')}"
    rc = run_ra(pull, s2, s3, "deciding", allow_rehearsal=False)
    r = json.load(open(f"{pull}/STAGE5_RESULT.json", encoding="utf-8"))
    assert rc == 1 and os.path.exists(f"{pull}/STAGE5.fail") and list(r["steps"]) == ["preflight"] and not os.path.exists(f"{pull}/rollouts")
    assert any(x.startswith("RALIGN_REHEARSAL_NOT_ALLOWED") for x in r["steps"]["preflight"]["failures"]), r["steps"]["preflight"]["failures"]
    p = f"{fx['fx']['pkg']}/ralign/MAP_B.pt"
    b = bytearray(open(p, "rb").read())
    b[len(b) // 2] ^= 1
    try:
        os.remove(p)
        open(p, "wb").write(bytes(b))
        pull2 = f"{fx['root']}/pull5_ra_tamper_{time.strftime('%H%M%S')}"
        rc2 = run_ra(pull2, s2, s3, "deciding", allow_rehearsal=True)
        r2 = json.load(open(f"{pull2}/STAGE5_RESULT.json", encoding="utf-8"))
        assert rc2 == 1 and any(x.startswith("RALIGN_MAPS_UNREADABLE RALIGN_MAP_HASH_MISMATCH") for x in r2["steps"]["preflight"]["failures"]), r2["steps"]["preflight"]["failures"]
    finally:
        extract_variant(res)                                               # put the map back
    assert L.sha256_file(p) == res["overlay_files"][[e["path"] for e in res["overlay_files"]].index("ralign/MAP_B.pt")]["sha256"]
