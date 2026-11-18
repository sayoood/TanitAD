"""Tests of readchain_tools.py: computed evidence (C2/C3/C5), sha refusal, BAND-MET rule, manifest assembly. Fixture = a real FT-format run of the 4c396aff trainer (make_ft_ckpt.py)."""
import json, os, subprocess, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FX = os.path.join(HERE, "fixture", "FT")
sys.path.insert(0, HERE)


def run(*args):
    return subprocess.run([PY, os.path.join(HERE, "readchain_tools.py"), *args], capture_output=True, encoding="utf-8", env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), timeout=600)


def bank(tmp_path, name, tamper=False):
    bd = str(tmp_path / name)
    args = [PY, os.path.join(HERE, "mk_fakebank.py"), bd, os.path.join(FX, "ckpt_last.pt"), FX] + (["--tamper"] if tamper else [])
    subprocess.run(args, check=True, encoding="utf-8", capture_output=True)
    return bd


def test_evidence_computes_c2_c3_from_the_banked_files_and_refuses_a_non_r35_start(tmp_path):
    bd = bank(tmp_path, "optb_FTRSP_s0")
    r = run("evidence", "--tag", "FTRSP_s0", "--bank", bd, "--out", str(tmp_path / "ev.json"))
    ev = json.load(open(tmp_path / "ev.json"))
    assert ev["c2_hold"] is True and ev["c3_hold"] is True and ev["c2_c3_hold"] is True, ev
    assert ev["c2"]["evidence"]["rsp_w"]["n"] >= 1
    assert r.returncode == 4 and ev["c5_hold"] is False            # the synthetic fixture is not R35's descendant: the C5 / identity check REFUSES (it is not a pass-through)


def test_evidence_c2_goes_red_when_the_log_is_altered(tmp_path):
    bd = bank(tmp_path, "optb_FTRSP_s0")
    rows = [json.loads(l) for l in open(os.path.join(bd, "train_log.jsonl"))]
    for r_ in rows:
        r_["wm/rsp/L1"] = r_["wm/rsp/L1"] * 1.5
    with open(os.path.join(bd, "train_log.jsonl"), "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rows))
    # keep the manifest honest so only C2 is under test: rewrite RUN.done / manifest shas
    import hashlib
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    rd = json.load(open(os.path.join(bd, "RUN.done")))
    rd["evidence"]["train_log_sha256"] = sha(os.path.join(bd, "train_log.jsonl"))
    json.dump(rd, open(os.path.join(bd, "RUN.done"), "w"))
    bm = json.load(open(os.path.join(bd, "BANK_MANIFEST.json")))
    bm["files"]["train_log.jsonl"]["sha256"] = sha(os.path.join(bd, "train_log.jsonl"))
    bm["files"]["RUN.done"]["sha256"] = sha(os.path.join(bd, "RUN.done"))
    json.dump(bm, open(os.path.join(bd, "BANK_MANIFEST.json"), "w"))
    run("evidence", "--tag", "FTRSP_s0", "--bank", bd, "--out", str(tmp_path / "ev.json"))
    ev = json.load(open(tmp_path / "ev.json"))
    assert ev["c2_hold"] is False and ev["c2_c3_hold"] is False, ev


def test_one_flipped_byte_is_refused_with_rc3(tmp_path):
    bd = bank(tmp_path, "optb_FTRSP_s0", tamper=True)
    r = run("evidence", "--tag", "FTRSP_s0", "--bank", bd, "--out", str(tmp_path / "ev.json"))
    assert r.returncode == 3 and "sha256" in r.stdout


def test_bandmet_rule(tmp_path):
    ref = json.load(open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_read/results/arm_R35_s0.json"))["passes"]
    arm = json.loads(json.dumps(ref))
    p = tmp_path / "ia.json"
    json.dump({"passes": arm}, open(p, "w"))
    run("bandmet", "--ia", str(p), "--chassis", "0", "--tag", "T", "--out", str(tmp_path / "bm.json"))
    assert json.load(open(tmp_path / "bm.json"))["band_met_lost"] is False        # control: the arm reads exactly R35's passes
    arm["2"]["beats"] = False
    json.dump({"passes": arm}, open(p, "w"))
    run("bandmet", "--ia", str(p), "--chassis", "0", "--tag", "T", "--out", str(tmp_path / "bm.json"))
    j = json.load(open(tmp_path / "bm.json"))
    assert j["band_met_lost"] is True and j["lost_levels"] == ["2"]
    arm["2"]["beats"] = True; arm["1"]["worse"] = True
    json.dump({"passes": arm}, open(p, "w"))
    run("bandmet", "--ia", str(p), "--chassis", "0", "--tag", "T", "--out", str(tmp_path / "bm.json"))
    assert json.load(open(tmp_path / "bm.json"))["lost_levels"] == ["1"]


def test_manifest_assembles_only_what_exists_and_aggregates_controls(tmp_path):
    root = tmp_path / "o"
    for sub in ("plans/FTRSP_s0", "evidence", "bandmet"):
        os.makedirs(root / sub)
    for s in (0, 1):
        open(root / "plans" / "FTRSP_s0" / f"plans_hwm_oracle_s{s}.npz", "wb").write(b"x")
    open(root / "plans" / "FTRSP_s0" / "DONE", "w").write("1")
    json.dump({"c2_hold": True, "c3_hold": True, "c2_c3_hold": True, "c5_hold": True}, open(root / "evidence" / "FTRSP_s0.json", "w"))
    json.dump({"band_met_lost": False, "lost_levels": []}, open(root / "bandmet" / "FTRSP_s0.json", "w"))
    r = run("manifest", "--root", str(root), "--out", str(tmp_path / "m.json"))
    m = json.load(open(tmp_path / "m.json"))
    assert list(m["arms"]) == ["FTRSP_s0"] and m["controls"] == {"c2_c3_hold": True, "band_met_lost": {"0": False, "1": None}}, (m, r.stderr[-300:])
    json.dump({"c2_hold": True, "c3_hold": False, "c2_c3_hold": False, "c5_hold": True}, open(root / "evidence" / "FTRSP_s0.json", "w"))
    run("manifest", "--root", str(root), "--out", str(tmp_path / "m.json"))
    assert json.load(open(tmp_path / "m.json"))["controls"]["c2_c3_hold"] is False


REAL_FTRSP = "C:/Users/Admin/t2v8f_pulls/optb_FTRSP_s0"


@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL_FTRSP, "BANK.done")), reason="the real FTRSP_s0 bank is not pulled here")
def test_c5_default_is_the_checkers_129_and_the_old_141_is_refused_on_the_real_bank(tmp_path):
    """MM ruling 2026-11-18 (MM_RULINGS_POSTCHECK.md): C5 counts the 133 wm.* PARAMETERS minus 4 wm.proj = 129. The default must pass the real
    FTRSP_s0 bank; the pre-ruling constant 141 must refuse it (rc 4) and name 141, so the check is not a pass-through."""
    r = run("evidence", "--tag", "FTRSP_s0", "--bank", REAL_FTRSP, "--out", str(tmp_path / "ev.json"))
    ev = json.load(open(tmp_path / "ev.json"))
    assert r.returncode == 0 and ev["c5_hold"] is True and ev["c5"]["adam"]["n_with_state"] == 129, (ev.get("problems"), r.stderr[-300:])
    r = run("evidence", "--tag", "FTRSP_s0", "--bank", REAL_FTRSP, "--out", str(tmp_path / "ev141.json"), "--expect-adam", "141")
    ev = json.load(open(tmp_path / "ev141.json"))
    assert r.returncode == 4 and ev["c5_hold"] is False and any("wanted 141" in p for p in ev["problems"]), ev.get("problems")
