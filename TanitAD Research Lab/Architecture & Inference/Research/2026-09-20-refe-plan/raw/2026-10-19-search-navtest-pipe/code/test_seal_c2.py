"""Tests of `seal_c2_amendment.py` (MM_RULING_C2AMEND.md DQ9): the tool that fills the `@@SEAL@@` pins of the FINAL DRAFT and writes the sealed amendment, its sidecar and `SEALED_AT.txt`.  Planted files only: no real banked
navtest file, checkpoint, head or harness is read, and the tool is never run for real here.

Every guard has a RED arm that must REFUSE with its NAMED reason; the most important one is DQ9's own: a pin left as an unfilled placeholder must refuse.  mutation_check.py deletes each `@GUARD` line in turn and this suite must then fail.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
import seal_c2_amendment as S  # noqa: E402
import test_navpipe as T  # noqa: E402

RULING = os.environ.get("C2_RULING") or os.path.join(NC.RAW_DIR, "2026-10-19-search-p2-confirm", "c2", "MM_RULING_C2AMEND.md")       # mutation_check.py's scratch copy names it by C2_RULING
SEALED_AT_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")


def sha_text(t):
    return hashlib.sha256(t.replace("\r\n", "\n").encode("utf-8")).hexdigest()


class Plant:
    """the planted files a seal run reads, and the FINAL DRAFT text that pins them"""

    def __init__(self, tmp):
        tmp = str(tmp)
        os.makedirs(tmp, exist_ok=True)
        self.tmp = tmp
        w = lambda name, data: (open(os.path.join(tmp, name), "wb").write(data), os.path.join(tmp, name))[1]       # noqa: E731
        self.banked_seam = w("banked_seam.bin", b"planted seam bytes")
        self.banked_csv = w("banked.csv", b"token,pdms\r\nt0,0.5\r\n")
        self.token_file = w("tokens.json", b'{"tokens": []}')
        self.ckpt = w("ckpt.pt", b"planted ckpt")
        self.head = w("head.pt", b"planted head")
        os.makedirs(os.path.join(tmp, "harness"), exist_ok=True)
        self.harness = w("harness/refe_harness.py", b"print('harness')\n")
        self.seam_module = w("harness/refe_navtest_seam.py", b"# seam module\n")
        self.c1 = T.C1(os.path.join(tmp, "c1"))
        os.makedirs(os.path.join(tmp, "ev"), exist_ok=True)
        self.ev = T.plant_evidence(os.path.join(tmp, "ev"), self.c1)
        self.dose_chain, self.fixture_record = self.ev.stub_path, self.ev.fixture_path
        self.ruling = RULING
        self.src = os.path.join(tmp, "AMENDMENT_C2_DRAFT.md")
        self.pkg = os.path.join(tmp, "sealed")
        self.scratch = os.path.join(tmp, "scratch")
        os.makedirs(self.scratch, exist_ok=True)
        self.set_draft(self.draft())

    # ---- the draft
    def static_pins(self):
        return {"token_file_sha256": NC.sha256_file(self.token_file), "chunk_plan_sha256": "b" * 64, "ckpt_sha256": NC.sha256_file(self.ckpt), "config_c8_sha256": NC.CONFIG_C8_SHA256,
                "config_o4_sha256": NC.CONFIG_O4_SHA256, "head_s0_sha256": NC.sha256_file(self.head), "c1_prereg_sha256": T.PLANT_PREREG, "licence_ruling_sha256": NC.LICENCE_RULING_SHA256,
                "c2amend_ruling_sha256": NC.C2AMEND_RULING_SHA256, "inference_cost_status": "UNMEASURED", "bar_dpdms_points": "0.10", "bar_safety_points": "-0.25",
                "harness_script_sha256": NC.sha_lf(self.harness), "seam_module_sha256": NC.sha_lf(self.seam_module)}

    def sealed_keys(self):
        return ["banked_seam_sha256", "banked_csv_sha256", "dose_chain_sha256", "fixture_record_sha256"] + ["code:" + f for f in S.CODE_PINS]

    def draft(self, static=None, sealed=None, title="# AMENDMENT C2 (FINAL DRAFT)", status="Status: **DRAFT** (planted).", ruled=True, before_pins="", after_pins="Tail text, terms.\n"):
        pins = dict(self.static_pins() if static is None else static)
        for k in (self.sealed_keys() if sealed is None else sealed):
            pins[k] = S.PLACEHOLDER
        body = "\n".join(f"{k} {v}" for k, v in pins.items())
        s14 = ("## 14. Open decisions\n\n" + S.RULED_HEAD + "…`).\n\n") if ruled else "## 14. Open decisions\n\nDQ1..DQ11 open.\n\n"
        return f"{title}\n\n{status}\n\nTerms text, byte for byte.\n\n{s14}{before_pins}<!-- PINS-BEGIN -->\n```\n{body}\n```\n<!-- PINS-END -->\n{after_pins}"

    def set_draft(self, text):
        open(self.src, "w", encoding="utf-8", newline="").write(text)
        return text

    # ---- the call
    def args(self, **over):
        a = types.SimpleNamespace(src=self.src, ruling=self.ruling, pkg=self.pkg, fixture_record=self.fixture_record, code_dir=HERE, dose_chain=self.dose_chain, banked_seam=self.banked_seam,
                                  banked_csv=self.banked_csv, token_file=self.token_file, ckpt=self.ckpt, head=self.head, harness=self.harness, check_only=False)
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def argv(self, *extra, **over):
        a = self.args(**over)
        v = ["--src", a.src, "--ruling", a.ruling, "--pkg", a.pkg, "--fixture-record", a.fixture_record, "--code-dir", a.code_dir, "--dose-chain", a.dose_chain, "--banked-seam", a.banked_seam,
             "--banked-csv", a.banked_csv, "--token-file", a.token_file, "--ckpt", a.ckpt, "--head", a.head, "--harness", a.harness]
        return v + list(extra)


@pytest.fixture
def P(tmp_path, monkeypatch):
    p = Plant(tmp_path)
    monkeypatch.setenv("C2SEAL_TMP", p.scratch)
    return p


def refused(fn, *args, **kw):
    with pytest.raises(S.SealRefused) as e:
        fn(*args, **kw)
    return str(e.value)


def nothing_written(P):
    return not os.path.exists(P.pkg) or os.listdir(P.pkg) == []


# ====================================================================================================== the happy path
def test_the_ruling_the_tool_checks_is_the_one_in_the_repo():
    assert os.path.isfile(RULING), RULING
    assert hashlib.sha256(S.lf(open(RULING, "rb").read())).hexdigest() == NC.C2AMEND_RULING_SHA256
    assert S.RULED_HEAD == "RULED (`MM_RULING_C2AMEND.md` sha256 `fe4d9ce6"


def test_seal_fills_every_placeholder_and_the_chains_own_check_accepts_the_text(P):
    r = S.seal(P.args())
    assert "@@" not in r["text"] and r["sha"] == sha_text(r["text"])
    pins = r["pins"]
    assert pins["banked_seam_sha256"] == NC.sha256_file(P.banked_seam) and pins["banked_csv_sha256"] == NC.sha256_file(P.banked_csv)
    assert pins["dose_chain_sha256"] == NC.sha_lf(P.dose_chain) and pins["fixture_record_sha256"] == NC.sha256_file(P.fixture_record)
    for f in NC.OPERATIVE + (S.SELF,):
        assert pins["code:" + f] == NC.sha_lf(os.path.join(HERE, f)), f
    assert re.search(r"^Status: \*\*SEALED\*\*", r["text"], re.M) and "Status: **DRAFT" not in r["text"] and "(FINAL DRAFT)" not in r["text"].split("\n", 1)[0] and "(SEALED)" in r["text"].split("\n", 1)[0]
    assert S.terms_identical(P.draft(), r["text"].replace("Status: **SEALED**", "Status: **SEALED**"))


def test_main_writes_the_three_files_and_nothing_else_and_the_sidecar_is_the_texts_sha(P, capsys):
    assert S.main(P.argv()) == 0
    out = capsys.readouterr().out
    assert out.startswith("ZZSEALED ") and os.path.join(P.pkg, S.SEALED_NAME).replace("\\", "/") in out.replace("\\", "/")
    assert sorted(os.listdir(P.pkg)) == sorted([S.SEALED_NAME, S.SIDECAR_NAME, S.STAMP_NAME])
    text = open(os.path.join(P.pkg, S.SEALED_NAME), "rb").read()
    assert b"\r" not in text and S.lf(text) == text
    sha = hashlib.sha256(text).hexdigest()
    assert open(os.path.join(P.pkg, S.SIDECAR_NAME)).read().split() == [sha]
    stamp = open(os.path.join(P.pkg, S.STAMP_NAME)).read().split("\n")
    assert SEALED_AT_RE.match(stamp[0]) and stamp[1] == sha
    assert os.listdir(P.scratch) == []                                                       # the scratch directory the own-check used is gone
    # and the CHAIN accepts exactly these files: the three-way sha binding, the pins, the code pins
    got, pins = NC.check_amendment_seal(os.path.join(P.pkg, S.SEALED_NAME), sha, HERE, (), NC.OPERATIVE, NC.REQUIRED_PINS, True)
    assert got == sha and pins["dose_chain_sha256"] == NC.sha_lf(P.dose_chain)


def test_the_sealed_amendment_unlocks_the_whole_c2_seal_check_end_to_end(P, capsys):
    assert S.main(P.argv()) == 0
    sealed = os.path.join(P.pkg, S.SEALED_NAME)
    sha = open(os.path.join(P.pkg, S.SIDECAR_NAME)).read().split()[0]
    r = NC.check_c2_seal(sealed, sha, P.c1.path, P.c1.sha, config=NC.CONFIG_C8, dose_result=P.ev.dose_path, dose_result_sha256=P.ev.dose_sha, fixture_record=P.fixture_record, dose_chain=P.dose_chain)
    assert r["selection"]["config"] == "C8" and r["fixture"]["verdict"] == "IDENTICAL" and r["pins"]["fixture_record_sha256"] == NC.sha256_file(P.fixture_record)


def test_check_only_verifies_and_writes_nothing(P, capsys):
    assert S.main(P.argv("--check-only")) == 0
    assert capsys.readouterr().out.startswith("ZZSEAL_CHECK_OK ") and nothing_written(P) and os.listdir(P.scratch) == []


def test_a_crlf_draft_seals_to_the_same_text_as_the_lf_one(P):
    lf_text = P.draft()
    P.set_draft(lf_text)
    a = S.seal(P.args())
    P.set_draft(lf_text.replace("\n", "\r\n"))
    b = S.seal(P.args())
    assert a["text"].replace("Status: **SEALED** ", "S ") != "" and re.sub(r"Status: \*\*SEALED\*\* \S+", "S", a["text"]) == re.sub(r"Status: \*\*SEALED\*\* \S+", "S", b["text"])
    assert "\r" not in b["text"]


def test_a_prefilled_pin_that_equals_its_file_passes_in_either_case(P):
    d = P.draft(sealed=[k for k in P.sealed_keys() if k != "banked_csv_sha256"])
    for form in (NC.sha256_file(P.banked_csv), NC.sha256_file(P.banked_csv).upper()):
        P.set_draft(d.replace("```\n", "```\nbanked_csv_sha256 " + form + "\n", 1))
        assert S.seal(P.args())["pins"]["banked_csv_sha256"].lower() == NC.sha256_file(P.banked_csv)


# ====================================================================================================== the RED arms: each refuses, by its own reason
def test_DQ9_red_arm_a_pin_left_as_an_unfilled_placeholder_refuses(P):
    """DQ9: `@@SEAL@@` on a pin this tool has no file to fill (here a pin the chain needs, `chunk_plan_sha256`) is a REFUSAL naming the pin"""
    d = P.draft().replace(f"chunk_plan_sha256 {'b' * 64}", f"chunk_plan_sha256 {S.PLACEHOLDER}")
    P.set_draft(d)
    m = refused(S.seal, P.args())
    assert "chunk_plan_sha256" in m and "placeholder" in m


def test_red_arm_every_static_pin_as_a_placeholder_refuses_and_every_sealed_pin_as_one_is_filled(P):
    for k in P.static_pins():
        P.set_draft(P.draft(static={**P.static_pins(), k: S.PLACEHOLDER}))
        assert k in refused(S.seal, P.args()), k
    P.set_draft(P.draft())
    assert "@@" not in S.seal(P.args())["text"]


def test_red_arm_an_unknown_placeholder_form_refuses(P):
    P.set_draft(P.draft(static={**P.static_pins(), "chunk_plan_sha256": "@@TO-BE-PINNED@@"}))
    m = refused(S.seal, P.args())
    assert "chunk_plan_sha256" in m and "does not know" in m


def test_red_arm_a_placeholder_outside_the_pins_block_refuses(P):
    for where in ("before", "after"):
        P.set_draft(P.draft(before_pins="Term @@TBD@@ here.\n\n" if where == "before" else "", after_pins="Tail with @@TBD@@.\n" if where == "after" else "Tail.\n"))
        assert "outside the PINS block" in refused(S.seal, P.args())


def test_red_arm_a_prefilled_pin_that_differs_from_its_file_refuses(P):
    P.set_draft(P.draft(sealed=[k for k in P.sealed_keys() if k != "dose_chain_sha256"]).replace("```\n", "```\ndose_chain_sha256 " + "0" * 64 + "\n", 1))
    m = refused(S.seal, P.args())
    assert "dose_chain_sha256" in m and "hashes to" in m
    P.set_draft(P.draft())
    open(P.dose_chain, "a").write("# edited\n")                                                        # the file moved on after the draft was made: its placeholder still fills, but a prefilled pin would now refuse
    assert S.seal(P.args())["pins"]["dose_chain_sha256"] == NC.sha_lf(P.dose_chain)


def test_red_arm_a_draft_without_a_line_for_a_pin_the_tool_computes_refuses(P):
    for k in ("code:" + S.SELF, "banked_seam_sha256", "code:navpipe_core.py"):
        P.set_draft(P.draft(sealed=[x for x in P.sealed_keys() if x != k]))
        m = refused(S.seal, P.args())
        assert "no line for" in m and k in m, (k, m)


def test_red_arm_the_ruling_must_be_the_pinned_file(P, tmp_path):
    other = tmp_path / "other_ruling.md"
    other.write_bytes(open(RULING, "rb").read() + b"\nedited\n")
    m = refused(S.seal, P.args(ruling=str(other)))
    assert "MM_RULING_C2AMEND.md" in m and "hashes to" in m
    crlf = tmp_path / "crlf_ruling.md"
    crlf.write_bytes(open(RULING, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert S.seal(P.args(ruling=str(crlf)))["sha"]                                                    # CRLF checkout of the right file is the right file


def test_red_arm_a_draft_whose_section_14_is_not_marked_ruled_refuses(P):
    P.set_draft(P.draft(ruled=False))
    assert "RULED" in refused(S.seal, P.args())


def test_red_arm_the_source_must_be_a_final_draft_with_a_draft_status_line(P):
    P.set_draft(P.draft(status="Status: **SEALED** already."))
    assert "Status: **DRAFT" in refused(S.seal, P.args())
    P.set_draft(P.draft(title="# AMENDMENT C2 (DRAFT)"))
    assert "FINAL DRAFT" in refused(S.seal, P.args())
    P.set_draft(P.draft(title="# AMENDMENT C2 (SEALED)"))
    assert "FINAL DRAFT" in refused(S.seal, P.args())


def test_red_arm_a_source_without_a_pins_block_refuses(P):
    P.set_draft("# AMENDMENT C2 (FINAL DRAFT)\n\nStatus: **DRAFT**\n\n" + S.RULED_HEAD + "…`)\n")
    assert "no PINS block" in refused(S.seal, P.args())


def test_red_arm_the_terms_must_be_byte_identical_to_the_draft(P, monkeypatch):
    real = S.build_sealed

    def tampered(src, fills, when=None):
        t, p = real(src, fills, when)
        return t.replace("Terms text, byte for byte.", "Terms text, softened."), p

    monkeypatch.setattr(S, "build_sealed", tampered)
    assert "terms differ" in refused(S.seal, P.args())


def test_terms_identical_is_blind_to_exactly_the_status_title_word_and_pins_block():
    a = "# T (FINAL DRAFT)\n\nStatus: **DRAFT** x\n\nbody\n\n<!-- PINS-BEGIN -->\nk @@SEAL@@\n<!-- PINS-END -->\ntail\n"
    b = "# T (SEALED)\n\nStatus: **SEALED** 2026 y\n\nbody\n\n<!-- PINS-BEGIN -->\nk " + "1" * 64 + "\n<!-- PINS-END -->\ntail\n"
    assert S.terms_identical(a, b)
    assert not S.terms_identical(a, b.replace("body", "bodyy"))
    assert not S.terms_identical(a, b.replace("tail", "tail2"))
    assert not S.terms_identical(a, b.replace("# T", "# U"))
    assert not S.terms_identical(a, b.replace("\n\nbody", "\n\nExtra.\n\nbody"))


def test_red_arm_a_placeholder_that_survives_the_fill_refuses(P, monkeypatch):
    real = S.fill_pins

    def leaky(text, fills):
        t, p = real(text, fills)
        return t.replace("Terms text, byte for byte.", "Terms text, byte for byte. @@LEAK@@"), p

    monkeypatch.setattr(S, "fill_pins", leaky)
    assert "@@" in refused(S.seal, P.args())


@pytest.mark.parametrize("which", ["token_file", "ckpt", "head", "harness", "seam_module"])
def test_red_arm_a_pin_that_names_a_file_must_equal_that_file(P, which):
    path = getattr(P, which)
    open(path, "ab").write(b"changed after the draft was written")
    if which == "seam_module":
        key = "seam_module_sha256"
    else:
        key = {"token_file": "token_file_sha256", "ckpt": "ckpt_sha256", "head": "head_s0_sha256", "harness": "harness_script_sha256"}[which]
    m = refused(S.seal, P.args())
    assert "do not equal their files" in m and key in m


def test_red_arm_a_missing_file_a_pin_names_refuses_rather_than_skips(P):
    os.remove(P.harness)
    with pytest.raises((S.SealRefused, OSError)):
        S.seal(P.args())
    P2 = P.args(ckpt=os.path.join(P.tmp, "no_such_ckpt.pt"))
    assert "ckpt_sha256" in refused(S.seal, P2)


def test_red_arm_the_fixture_record_is_refused_by_the_chains_own_check(P):
    rec = json.load(open(P.fixture_record))
    rec["verdict"] = "DIFFERS"
    json.dump(rec, open(P.fixture_record, "w"), sort_keys=True)
    m = refused(S.seal, P.args())
    assert "FIXTURE_NOT_IDENTICAL" in m
    rec["verdict"] = "IDENTICAL"
    rec["code_sha256"]["navpipe_chain.py"] = "0" * 64                                                 # a fixture that ran another chain than the one about to be sealed
    json.dump(rec, open(P.fixture_record, "w"), sort_keys=True)
    m = refused(S.seal, P.args())
    assert "FIXTURE_CODE_NOT_THE_SEALED_CODE" in m and "navpipe_chain.py" in m


def test_red_arm_a_code_dir_that_is_not_the_code_the_fixture_ran_refuses(P, tmp_path):
    import shutil
    other = tmp_path / "code_copy"
    other.mkdir()
    for f in S.CODE_PINS:
        shutil.copyfile(os.path.join(HERE, f), other / f)
    with open(other / "navpipe_core.py", "a") as fh:
        fh.write("\n# another revision\n")
    assert "FIXTURE_CODE_NOT_THE_SEALED_CODE" in refused(S.seal, P.args(code_dir=str(other)))


def test_main_exits_1_with_the_refusal_and_leaves_no_file_behind(P, capsys):
    P.set_draft(P.draft(ruled=False))
    assert S.main(P.argv()) == 1
    out = capsys.readouterr().out
    assert out.startswith("ZZSEAL_REFUSED ") and "RULED" in out and nothing_written(P) and os.listdir(P.scratch) == []


def test_red_arm_the_chains_own_seal_check_refuses_what_the_tool_alone_would_write(P, capsys):
    """every check in the tool passes, but the chain's own G-SEAL does not (here: the inference-cost status is neither MEASURED nor UNMEASURED): nothing is written"""
    P.set_draft(P.draft(static={**P.static_pins(), "inference_cost_status": "ASSUMED"}))
    assert S.main(P.argv()) == 1
    out = capsys.readouterr().out
    assert "the chain's own seal check refuses" in out and "SEAL_INFERENCE_COST_UNSTATED" in out and nothing_written(P) and os.listdir(P.scratch) == []
    P.set_draft(P.draft(static={k: v for k, v in P.static_pins().items() if k != "bar_safety_points"}))
    assert S.main(P.argv()) == 1
    out = capsys.readouterr().out
    assert "SEAL_PIN_MISSING" in out and "bar_safety_points" in out and nothing_written(P)


def test_red_arm_the_wrong_ruling_pin_is_refused_by_the_chains_own_check(P, capsys):
    P.set_draft(P.draft(static={**P.static_pins(), "c2amend_ruling_sha256": "0" * 64}))
    assert S.main(P.argv()) == 1
    assert "SEAL_AMEND_RULING_PIN" in capsys.readouterr().out and nothing_written(P)


def test_red_arm_it_never_overwrites_a_sealed_file(P, capsys):
    assert S.main(P.argv()) == 0
    capsys.readouterr()
    before = {n: open(os.path.join(P.pkg, n), "rb").read() for n in os.listdir(P.pkg)}
    assert S.main(P.argv()) == 1
    assert "already holds sealed files" in capsys.readouterr().out
    assert {n: open(os.path.join(P.pkg, n), "rb").read() for n in os.listdir(P.pkg)} == before
    for keep in (S.SEALED_NAME, S.SIDECAR_NAME, S.STAMP_NAME):
        for n in os.listdir(P.pkg):
            os.remove(os.path.join(P.pkg, n))
        open(os.path.join(P.pkg, keep), "w").write("x")
        assert S.main(P.argv()) == 1 and open(os.path.join(P.pkg, keep)).read() == "x"
        capsys.readouterr()


def test_check_only_also_refuses_when_files_exist_or_inputs_are_wrong(P, capsys):
    assert S.main(P.argv()) == 0
    capsys.readouterr()
    assert S.main(P.argv("--check-only")) == 1
    P2 = Plant(P.tmp + "_b")
    P2.set_draft(P2.draft(ruled=False))
    assert S.main(P2.argv("--check-only")) == 1


def test_the_draft_is_never_modified_and_the_tool_reads_no_label(P):
    before = open(P.src, "rb").read()
    S.main(P.argv())
    assert open(P.src, "rb").read() == before
    src = open(os.path.join(HERE, "seal_c2_amendment.py"), encoding="utf-8").read()
    for bad in ("import numpy", "np.load", "import subprocess", "import torch", "import paramiko", "csv.", "json.load"):
        assert bad not in src, bad                                                                    # bytes are hashed; no score, label or search output is parsed


def test_the_tools_own_pin_and_the_operative_list_cover_every_file_the_chain_loads():
    assert S.SELF == "seal_c2_amendment.py" and S.CODE_PINS[-1] == S.SELF and set(NC.OPERATIVE) <= set(S.CODE_PINS) and len(set(S.CODE_PINS)) == len(S.CODE_PINS)
    for f in S.CODE_PINS:
        assert os.path.isfile(os.path.join(HERE, f)), f


# ====================================================================================================== the REAL final draft is seal-ready (everything the tool needs from it, short of the external files it hashes)
DRAFT = os.environ.get("C2_DRAFT") or os.path.join(NC.RAW_DIR, "2026-10-19-search-p2-confirm", "c2", "AMENDMENT_C2_DRAFT.md")
SCRATCH = bool(os.environ.get("NPMUT_SCRATCH"))        # mutation_check.py's scratch copy has no sibling packages at their repo-relative place


@pytest.mark.skipif(SCRATCH or not os.path.isfile(DRAFT), reason="the draft is a sibling package; not in the mutation scratch copy")
def test_the_real_final_draft_is_seal_ready_and_the_chains_own_check_accepts_it_once_filled(tmp_path):
    text = open(DRAFT, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    assert text.split("\n", 1)[0].startswith("# AMENDMENT C2 (FINAL DRAFT)") and re.search(r"^Status: \*\*DRAFT", text, re.M)
    S.check_ruling(RULING, text)                                                              # the ruling file is MM_RULING_C2AMEND.md and section 14 is marked RULED with its sha256
    head, blk, tail = S.split_pins(text)
    assert "@@" not in head and "@@" not in tail                                                # no placeholder in the prose
    pins = NC.parse_pins(text)
    assert all(k in pins for k in NC.REQUIRED_PINS) and pins["mode"] == "navtest"
    assert sorted(k for k, v in pins.items() if v == S.PLACEHOLDER) == sorted(["banked_seam_sha256", "banked_csv_sha256", "dose_chain_sha256", "fixture_record_sha256"] + ["code:" + f for f in S.CODE_PINS])
    assert (pins["config_c8_sha256"], pins["config_o4_sha256"], pins["licence_ruling_sha256"], pins["c2amend_ruling_sha256"]) == (NC.CONFIG_C8_SHA256, NC.CONFIG_O4_SHA256, NC.LICENCE_RULING_SHA256, NC.C2AMEND_RULING_SHA256)
    assert pins["inference_cost_status"] == "UNMEASURED" and (pins["bar_dpdms_points"], pins["bar_safety_points"]) == ("0.10", "-0.25")
    for k, v in pins.items():
        assert v == S.PLACEHOLDER or re.fullmatch(r"[0-9a-f]{64}|UNMEASURED|navtest|0\.10|-0\.25", v), (k, v)
    fills = {k: "a" * 64 for k in ("banked_seam_sha256", "banked_csv_sha256", "dose_chain_sha256", "fixture_record_sha256")}
    fills.update({"code:" + f: NC.sha_lf(os.path.join(HERE, f)) for f in S.CODE_PINS})
    sealed, spins = S.build_sealed(text, fills, when="2026-10-20T00:00+00:00")
    assert "@@" not in sealed and S.terms_identical(text, sealed)
    am = tmp_path / "AMENDMENT_C2.md"
    am.write_text(sealed, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(sealed.encode("utf-8")).hexdigest()
    (tmp_path / "AMENDMENT_C2.sha256").write_text(sha + "\n")
    got, p2 = NC.check_amendment_seal(str(am), sha, HERE, (), NC.OPERATIVE, NC.REQUIRED_PINS, True)
    assert got == sha and p2["c2amend_ruling_sha256"] == NC.C2AMEND_RULING_SHA256


@pytest.mark.skipif(SCRATCH or not os.path.isfile(DRAFT), reason="the draft is a sibling package; not in the mutation scratch copy")
def test_every_ruling_term_is_in_the_final_draft():
    """the ruling's own words, one by one (a term of the ruling the draft lost is a seal of the wrong terms)"""
    text = open(DRAFT, encoding="utf-8").read()
    for must in ("exactly one C2 run", "H-SEARCH-P2-1", "selection-tier gain on navtest; deployability UNMEASURED", "p50 / p95 / p99", "pick-only forward", "IN THE HEADLINE LINE", "RE-SCORED", "1e-9",
                 "fixture RECORD", "IDENTICAL", "`dose_chain.py`", "`c2_config`", "`inert.frac_moved`", "dev-box GPU", "about 2.9x", "stepB_config_O4.json", "1ebd7196", "bar numbers", "DQ1", "DQ11"):
        assert must.lower() in text.lower(), must
    ruled = [ln for ln in text.splitlines() if ln.startswith("## 14.")]
    assert len(ruled) == 1 and "RULED (`MM_RULING_C2AMEND.md` sha256 `fe4d9ce6" in ruled[0]
    assert "(FINAL DRAFT)" in text.split("\n", 1)[0] and "DECISIONS PUT TO THE MASTER MIND" not in text
