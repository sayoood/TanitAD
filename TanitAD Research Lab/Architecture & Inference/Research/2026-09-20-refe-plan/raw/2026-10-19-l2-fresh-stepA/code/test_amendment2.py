"""Tests for Amendment 2 (the next-arm mapping of O3 only).  CPU only (`CUDA_VISIBLE_DEVICES=-1`); reads no carve-B label or score.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_amendment2.py

GREEN: the sealed file passes `amendment2_check.check_amendment`; the calculation's identity passed; the branch word equals the ruling's rule applied to the JSON; the read's seal guard on the PARENT is untouched.
RED arms (each must refuse, exit status 5): a one-byte-edited copy under the original sha; a draft-marker copy; the branch word flipped (re-hashed, so only the consistency rule can catch it); one projected half-width
edited; the ruling altered by one word; a calculation JSON with one number changed; a parent file that changed.  `sealed_before_read` is exercised on a synthetic run dir (older / newer / absent artifacts).
"""
import datetime
import json
import os
import shutil
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amendment2_check as AC  # noqa: E402

W = AC.W
AMEND = f"{W}/PREREG_L2_FRESH_AMENDMENT2.md"
PARENT = f"{W}/PREREG_L2_FRESH.md"
RULING = f"{W}/MM_RULING_L2_FRESH_D.md"
CJ, CC = f"{W}/raw/design_calc_ball.json", f"{W}/code/design_calc_ball.py"
SHA = AC.sha_lf(AMEND)


def ok(**kw):
    a = dict(amendment=AMEND, sha_given=SHA, parent=PARENT, ruling=RULING, calc_json=CJ, calc_code=CC)
    a.update(kw)
    return AC.check_amendment(**a)


def mutate(tmp_path, old, new, name="m.md"):
    txt = open(AMEND, encoding="utf-8").read()
    assert old in txt, old
    p = tmp_path / name
    p.write_text(txt.replace(old, new, 1), encoding="utf-8", newline="\n")
    return str(p), AC.sha_lf(str(p))


def test_sealed_file_accepted_and_sha_file_matches():
    info = ok()
    assert info["branch"] == "UNSEEN-STRATUM DATA JOB"
    assert open(f"{W}/PREREG_L2_FRESH_AMENDMENT2.sha256").read().split()[0] == SHA


def test_calculation_identity_and_branch_rule():
    calc = json.load(open(CJ))
    assert calc["identity"]["pass"] is True
    for s, arm in calc["identity"]["arms"].items():
        assert arm["bit_exact_vs_banked"] and arm["token_level_ok"], s
    assert calc["identity"]["red_arm_rolled_tokens"]["differs_from_banked"]
    hw = [calc["arms"][s]["projection"]["proportional_form"]["HW"] for s in ("H_s0", "H_s1")]
    assert AC.branch_from_json(calc) == ("B-ALL" if max(hw) <= 0.025 else "UNSEEN-STRATUM DATA JOB") == calc["branch"]["selected"]
    assert calc["inputs"]["carve_B_labels_or_scores_read"] is False


def test_red_one_byte_edit_under_original_sha(tmp_path):
    p, _ = mutate(tmp_path, "Status: SEALED", "Status: SEALEd")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(amendment=p)                                                # original sha given, edited bytes
    assert e.value.code == 5 and "sha256" in str(e.value)


def test_red_draft_marker(tmp_path):
    p, s = mutate(tmp_path, "Status: SEALED", "Status: DRAFT")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(amendment=p, sha_given=s)
    assert e.value.code == 5 and "draft" in str(e.value)


def test_red_branch_word_flipped_and_rehashed(tmp_path):
    p, s = mutate(tmp_path, "**Selected branch: UNSEEN-STRATUM DATA JOB.**", "**Selected branch: B-ALL.**")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(amendment=p, sha_given=s)
    assert e.value.code == 5 and "branch" in str(e.value)


def test_red_projected_half_width_edited(tmp_path):
    calc = json.load(open(CJ))
    hw = f"{calc['arms']['H_s0']['projection']['proportional_form']['HW']:.4f}"
    txt = open(AMEND, encoding="utf-8").read()
    assert txt.count(hw) >= 1
    p = tmp_path / "hw.md"
    p.write_text(txt.replace(hw, "0.0249"), encoding="utf-8", newline="\n")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(amendment=str(p), sha_given=AC.sha_lf(str(p)))
    assert e.value.code == 5 and "half-width" in str(e.value)


def test_red_ruling_altered_by_one_word(tmp_path):
    txt = open(RULING, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
    assert "WITHDRAWN" in txt
    r = tmp_path / "ruling.md"
    r.write_text(txt.replace("WITHDRAWN", "KEPT", 1), encoding="utf-8", newline="\n")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(ruling=str(r))
    assert e.value.code == 5 and "verbatim" in str(e.value)


def test_red_calculation_json_changed(tmp_path):
    calc = json.load(open(CJ))
    calc["arms"]["H_s0"]["projection"]["proportional_form"]["HW"] = 0.02
    p = tmp_path / "calc.json"
    p.write_text(json.dumps(calc), encoding="utf-8")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(calc_json=str(p))
    assert e.value.code == 5 and "calculation result" in str(e.value)


def test_red_parent_changed(tmp_path):
    p = tmp_path / "parent.md"
    shutil.copyfile(PARENT, p)
    with open(p, "ab") as f:
        f.write(b"\nedited\n")
    with pytest.raises(AC.AmendmentRefused) as e:
        ok(parent=str(p))
    assert e.value.code == 5 and "parent" in str(e.value)


def test_sealed_before_read_synthetic(tmp_path):
    seal = datetime.datetime.now().astimezone()
    assert AC.sealed_before_read(str(tmp_path), seal) == {"ok": True, "artifacts_present": [], "older_than_seal": []}      # vacuous while nothing exists
    old = tmp_path / "labels.npz"
    old.write_bytes(b"x")
    t = (seal - datetime.timedelta(hours=1)).timestamp()
    os.utime(old, (t, t))
    r = AC.sealed_before_read(str(tmp_path), seal)
    assert r["ok"] is False and r["older_than_seal"][0]["file"] == "labels.npz"                                                   # RED: a label older than the seal
    t2 = (seal + datetime.timedelta(hours=1)).timestamp()
    os.utime(old, (t2, t2))
    assert AC.sealed_before_read(str(tmp_path), seal)["ok"] is True


def test_real_b_run_dir_has_no_label_or_score_older_than_the_seal():
    txt = open(f"{W}/SEALED_AT_AMENDMENT2.txt", encoding="utf-8").read()
    r = AC.sealed_before_read("D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint", AC.parse_sealed_at(txt))
    assert r["ok"], r                                                                                                             # anything the chain writes later is newer than the seal


def test_the_read_guard_is_untouched():
    """Amendment 2 touches no pinned file: the parent and the five code pins are exactly what the seal names, and the read's own guard still accepts them."""
    import analyse_fresh as AF
    parent_txt = open(PARENT, encoding="utf-8").read()
    assert AF.sha_lf(PARENT) == "5006ed844979578f4c42599cd79b09c15bd07862d9c9bda94b21481d5e8d4e54"
    pins = AF.current_code_pins()
    for n, h in pins.items():
        assert any(n in ln and h in ln for ln in parent_txt.splitlines()), n
    sample = json.load(open(f"{W}/raw/fresh_sample_token_disjoint.json"))
    seal = AF.check_seal(PARENT, AF.sha_lf(PARENT), AF.disk_pins(sample, AF.HEADS_DIR, AF.LOGOOF_DIR), pins)
    assert seal["prereg_sha256"] == AF.sha_lf(PARENT)
    # the read has no next-arm text of its own: the O3 mapping lives only in the sealed documents
    for fn in ("analyse_fresh.py", "score_fresh.py"):
        src = open(f"{HERE}/{fn}", encoding="utf-8").read()
        assert "READ 2" not in src and "next arm" not in src.lower(), fn
