"""Tests of G2 parts (a), (b), (d), (e) of `code/read_t2_dominance.py` (the card's 17:42 reading), each with its RED arm. CPU only.
Fixtures: the REAL stage-1 pull (plumbing data, never a stage-2/3/4 rollout) for (a), (d), (e); the real overlay tars for (a); synthetic stage results; the real pytest logs for (b). No stage-2/3/4 row is read."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import pytest  # noqa: E402

import read_t2_dominance as RD  # noqa: E402

PLUMB = f"{RD.T2V8F}/stage1/pull_box2/pull/plumbing"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def stage_results(tmp_path, over=None):
    over = over or {}
    """Synthetic STAGE{2,3,4}_RESULT.json on the REAL stage-1 manifest sha256 and the real overlay manifests of stages 2 and 3 (stage 4: a synthetic overlay written to tmp). -> (results, overlays)."""
    r1 = json.load(open(RD.DEFAULT_RESULT1, encoding="utf-8"))
    sha1 = r1["manifest_sha256"]
    ov4 = {"package": "t2v8f_stage4_overlay", "stage1_manifest_sha256": sha1, "files": [{"path": "code/t2v8f_c0c.py", "size": 1, "sha256": "0" * 64}]}
    ov4.update(over.get("ov4", {}))
    p4 = tmp_path / "MANIFEST4.json"
    p4.write_text(json.dumps(ov4))
    overlays = {2: RD.DEFAULT_OVERLAYS[2], 3: RD.DEFAULT_OVERLAYS[3], 4: str(p4)}
    res = {}
    for st in (2, 3, 4):
        d = {"manifest_sha256": sha1, "steps": {"preflight": {"verdict": "PASS", "info": {"pkg_files_ok": 471, "pkg_files": 471}}},
             RD.MANIFEST_KEY[st]: _sha(RD.load_blob(overlays[st]))}
        d.update(over.get(st, {}))
        p = tmp_path / f"STAGE{st}_RESULT.json"
        p.write_text(json.dumps(d))
        res[st] = str(p)
    return res, overlays


def test_g2a_positive_on_the_real_stage1_package(tmp_path):
    res, ov = stage_results(tmp_path)
    out = RD.g2a_check(RD.DEFAULT_RESULT1, res, RD.DEFAULT_MANIFEST1, ov)
    assert out["PASS"], out["why"]
    assert "code/t2v8f_run.py" in out["rollout_code"] and len(out["rollout_code"]) >= 5            # the rollout code the three stages ran is listed with its sha256


def test_g2a_red_arms(tmp_path):
    res, ov = stage_results(tmp_path)
    bad = tmp_path / "S1.json"
    d = json.load(open(RD.DEFAULT_RESULT1, encoding="utf-8"))
    d["overall"] = "GATE_FAILED"
    bad.write_text(json.dumps(d))
    assert any("G2A_STAGE1_NOT_PASS" in w for w in RD.g2a_check(str(bad), res, RD.DEFAULT_MANIFEST1, ov)["why"])
    (tmp_path / "x").mkdir()
    r2, o2 = stage_results(tmp_path / "x", {3: {"manifest_sha256": "f" * 64}})
    assert any("G2A_PACKAGE_DIFFERS stage 3" in w for w in RD.g2a_check(RD.DEFAULT_RESULT1, r2, RD.DEFAULT_MANIFEST1, o2)["why"])           # a stage that ran another package
    (tmp_path / "y").mkdir()
    r3, o3 = stage_results(tmp_path / "y", {4: {"steps": {"preflight": {"verdict": "REFUSED", "info": {"pkg_files_ok": 470, "pkg_files": 471}, "failures": ["HASH_MISMATCH code/t2v8f_run.py"]}}}})
    w = RD.g2a_check(RD.DEFAULT_RESULT1, r3, RD.DEFAULT_MANIFEST1, o3)["why"]
    assert any("G2A_PKG_FILES stage 4" in x and "HASH_MISMATCH code/t2v8f_run.py" in x for x in w), w                                      # names the file box 2's preflight named
    (tmp_path / "z").mkdir()
    r4, o4 = stage_results(tmp_path / "z", {"ov4": {"files": [{"path": "code/t2v8f_run.py", "size": 1, "sha256": "0" * 64}]}})
    w = RD.g2a_check(RD.DEFAULT_RESULT1, r4, RD.DEFAULT_MANIFEST1, o4)["why"]
    assert any("G2A_CODE_REPLACED stage 4" in x and "code/t2v8f_run.py" in x for x in w), w                                                 # an overlay replacing a rollout-code file is named
    (tmp_path / "w").mkdir()
    r5, o5 = stage_results(tmp_path / "w")
    o5[4] = str(tmp_path / "nope.json")
    assert any("G2A_OVERLAY_MANIFEST_MISSING stage 4" in x for x in RD.g2a_check(RD.DEFAULT_RESULT1, r5, RD.DEFAULT_MANIFEST1, o5)["why"])


def test_g2b_positive_and_red_arms(tmp_path, monkeypatch):
    logs = {2: RD.DEFAULT_GATE_LOGS[2], 3: RD.DEFAULT_GATE_LOGS[3]}
    out = RD.g2b_check(logs=logs)
    assert out["PASS"] and all(out["tests"].values()) and out["logs"][3]["n_passed"] == 91, out
    red = tmp_path / "red.log"
    red.write_text("1 failed, 90 passed in 5s\n")
    assert any("G2B_LOG_NOT_GREEN stage 3" in w for w in RD.g2b_check(logs={3: str(red)})["why"])
    assert any("G2B_LOG_MISSING stage 4" in w for w in RD.g2b_check(logs={4: str(tmp_path / "nope.log")})["why"])
    old = tmp_path / "old.log"
    old.write_text("91 passed in 5s\n")
    os.utime(old, (1.0e9, 1.0e9))                                                                    # a green log older than the gate test files: a stale citation
    assert any("G2B_LOG_STALE stage 3" in w for w in RD.g2b_check(logs={3: str(old)})["why"])
    monkeypatch.setattr(RD, "GATE_TESTS", {"T4": (("tests/test_t2v8f_e5.py", "test_no_such_gate"),)})
    assert any("G2B_TEST_MISSING T4" in w for w in RD.g2b_check(logs={3: logs[3]})["why"])


def copy_cells(dst, arms=("ZERO", "P10"), files=("records.jsonl", "DONE", "actor_attach.json")):
    for sc in sorted(os.listdir(PLUMB)):
        for arm in arms:
            s, d = f"{PLUMB}/{sc}/objects/T2/{arm}", f"{dst}/{sc}/objects/T2/{arm}"
            os.makedirs(d, exist_ok=True)
            for f in files:
                if os.path.isfile(f"{s}/{f}"):
                    shutil.copyfile(f"{s}/{f}", f"{d}/{f}")
    return str(dst)


def test_g2d_zero_equals_p10_on_the_real_stage1_cells_and_red_arms(tmp_path):
    root = copy_cells(tmp_path / "ok")
    ok = RD.g2d_check({2: root})
    assert ok["PASS"] and ok["windows"][2] >= 2, ok
    bad = copy_cells(tmp_path / "bad")
    f = f"{bad}/{sorted(os.listdir(bad))[0]}/objects/T2/ZERO/records.jsonl"
    r = json.loads(open(f).readline())
    r["plans"][0][0][0] += 1e-9                                                                      # one plan number moved by 1e-9 m: not exactly equal
    open(f, "w").write(json.dumps(r) + "\n")
    w = RD.g2d_check({2: bad})["why"]
    assert any("G2D_ZERO_NE_P10" in x and "plans" in x for x in w), w
    miss = copy_cells(tmp_path / "miss")
    shutil.rmtree(f"{miss}/{sorted(os.listdir(miss))[0]}/objects/T2/ZERO")
    assert any("G2D_CELL_MISSING" in x for x in RD.g2d_check({2: miss})["why"])
    assert any("G2D_NO_WINDOWS" in x for x in RD.g2d_check({2: str(tmp_path / "empty")})["why"])


def test_g2e_actor_attach_per_scene_and_red_arms(tmp_path):
    root = copy_cells(tmp_path / "ok")
    ok = RD.g2e_check([("stage1", root, "P10")])
    assert ok["PASS"] and len(ok["per_scene"]) == 2 and all(set(v.values()) == {"ACCEPTED"} for v in ok["per_scene"].values()), ok
    sc = sorted(os.listdir(root))[0]
    os.remove(f"{root}/{sc}/objects/T2/P10/actor_attach.json")
    w = RD.g2e_check([("stage1", root, "P10")])["why"]
    assert any("G2E_ACTOR_ATTACH_MISSING" in x and sc in x for x in w), w
    rej = copy_cells(tmp_path / "rej")
    f = f"{rej}/{sorted(os.listdir(rej))[0]}/objects/T2/P10/actor_attach.json"
    d = json.load(open(f))
    d["verdict"] = "REJECTED"
    open(f, "w").write(json.dumps(d))
    out = RD.g2e_check([("stage1", rej, "P10")])
    assert out["PASS"] and len(out["non_accepted"]) == 1                                             # tabulated, reported, the MM's to rule on
    d.pop("verdict")
    open(f, "w").write(json.dumps(d))
    assert any("G2E_NO_VERDICT" in x for x in RD.g2e_check([("stage1", rej, "P10")])["why"])


def test_g2_full_fails_when_any_part_fails(tmp_path):
    a = type("A", (), {"root2": str(tmp_path / "r2"), "root3": str(tmp_path / "r3"), "root4": str(tmp_path / "r4"), "result1": RD.DEFAULT_RESULT1, "manifest1": RD.DEFAULT_MANIFEST1})()
    res = {s: str(tmp_path / f"S{s}.json") for s in (2, 3, 4)}
    out = RD.g2_full(a, res)
    assert not out["PASS"] and set(out["parts"]) == set("abcde") and any(w.startswith("(d) G2D_NO_WINDOWS") for w in out["why"]), out["why"][:4]
