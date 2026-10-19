#!/usr/bin/env python3
"""Seal SEARCH-P2-CONFIRM.  The Master Mind runs this AFTER ruling DQ1..DQ8 (the ruling is folded in VERBATIM); nothing here chooses a bar.

    python seal_spc.py --src <edited draft> [--ruling <MM ruling .md>] --pkg <package dir> [--check-only]

It (1) refuses a source that still has an unfilled placeholder or a code pin marked NOT YET WRITTEN, (2) recomputes the PINS block from the files on disk (new code, raw inputs, the Step B code and L1's four files),
and REFUSES if any Step B pin copied from `PREREG_L2_STEPB.md` no longer equals the file (the arm must be IDENTICAL to Step B's), (3) writes `PREREG_SEARCH_P2_CONFIRM.md` (LF, `Status: **SEALED**`),
`PREREG_SEARCH_P2_CONFIRM.sha256` and `SEALED_AT.txt`, (4) runs the chain's own seal guard on the result (`spc_chain.Chain.check_seal`) and fails the seal if the guard does not accept it.
No GPU, no lock.  Re-sealing after any pinned file changes is a NEW seal with a new date; never edit a sealed text.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spc_chain as CH  # noqa: E402
import spc_launch as SL  # noqa: E402

PACKAGE_CODE = ("spc_launch.py", "spc_chain.py", "spc_gpu_stage.py", "spc_cid.py", "spc_analyse.py", "b_rest.py", "make_cid_sample.py", "design_calc.py", "cost_spc.py", "power_table.py", "test_spc.py", "seal_spc.py", "mutation_check.py")
RAW_PINS = ("b_rest.json", "cid_sample.json", "design_calc.json", "cost.json")
L1_FILES = ("stepA.py", "tts_cem.py", "real_scorer.py", "label_perturbations.py", "make_stepA_bundle.py")


class SealRefused(RuntimeError):
    pass


def step_b_pins(path=os.path.join("C:/Users/Admin/par18/l2stepb", "PREREG_L2_STEPB.md")):
    return CH.parse_pins(open(path, encoding="utf-8").read())


def compute_pins(pkg, code_dir=HERE, stepb_code=CH.STEPB_CODE, l1_code=CH.L1_CODE, stepb_prereg=None, strict_stepb=True):
    sb = step_b_pins(stepb_prereg) if stepb_prereg else step_b_pins()
    pins = {}
    for k in ("ckpt_sha256", "config_sha256", "head_s0_sha256", "head_s1_sha256", "config_o4_sha256", "ruling_sha256", "draft_sha256"):
        pins[k] = sb[k]
    pins["b_rest_tokens_sha256"] = SL.B_REST_SHA256
    pins["carve_b_tokens_sha256"] = sb["carve_sha256"]
    pins["step_a_tokens_sha256"] = "211e26406f5b4eb083ae38b1c73839826d560042cd75ccf9ff6656262e7c0ffd"
    pins["cid_tokens_sha256"] = SL.CID_SHA256
    pins["bridge_bundle_script_sha256_lf"] = CH.sha_lf(os.path.join(l1_code, "make_stepA_bundle.py"))
    bad = []
    for k, v in sb.items():                                            # the arm is IDENTICAL to Step B's: its pinned files must still be what Step B pinned
        if k.startswith("code:") or k.startswith("l1:"):
            f = k.split(":", 1)[1]
            p = os.path.join(stepb_code if k.startswith("code:") else l1_code, f)
            if not os.path.isfile(p) or CH.sha_lf(p) != v:
                bad.append(k)
            else:
                pins[k] = v
    if bad and strict_stepb:
        raise SealRefused(f"Step B pins no longer equal their files (the arm would not be identical): {bad}")
    for f in CH.STEPB_OPERATIVE:                                       # imported by this chain but not in Step B's pin set (amend1_concordance, stepB_chain)
        pins.setdefault(f"code:{f}", CH.sha_lf(os.path.join(stepb_code, f)))
    for f in PACKAGE_CODE:
        p = os.path.join(code_dir, f)
        if not os.path.isfile(p):
            raise SealRefused(f"package code file missing: {f}")
        pins[f"code:{f}"] = CH.sha_lf(p)
    for f in RAW_PINS:
        pins[f"raw:{f}"] = CH.sha_lf(os.path.join(pkg, "raw", f))
    return pins


def render_pins(pins):
    return "\n".join(f"{k} {v}" for k, v in pins.items())


def build_sealed(src_text: str, pins: dict, ruling_text: str | None = None, when: str | None = None) -> str:
    if "@@" in src_text:
        raise SealRefused("the source still has an unfilled @@ placeholder")
    if "<NOT YET WRITTEN>" in src_text:
        raise SealRefused("the source still lists a code pin as NOT YET WRITTEN")
    t = src_text.replace("\r\n", "\n")
    if not re.search(r"^Status: \*\*DRAFT", t, re.M):
        raise SealRefused("the source has no 'Status: **DRAFT' line to turn into the seal")
    when = when or datetime.datetime.now().astimezone().isoformat(timespec="minutes")
    t = re.sub(r"^Status: \*\*DRAFT.*$", f"Status: **SEALED** {when} (the sha256 of this text is in `PREREG_SEARCH_P2_CONFIRM.sha256`; `code/spc_chain.py` REFUSES to run without both and without every pin below). No B-REST token had been bundled, searched or labelled at the seal.", t, count=1, flags=re.M)
    t = t.replace("(DRAFT, NOT SEALED)", "(SEALED)", 1)
    if ruling_text:
        rt = ruling_text.replace("\r\n", "\n").strip("\n")
        head, sep, tail = t.partition("\n## 0. Why")
        t = head + "\n## Revision 1 (MM ruling, pre-seal)\nThe Master Mind's ruling on DQ1..DQ8, folded in VERBATIM (sha256 of the ruling file: " + hashlib.sha256((rt + "\n").encode()).hexdigest() + \
            "). **Where the body and the ruling differ, THE RULING PREVAILS.**\n\n````markdown\n" + rt + "\n````\n" + sep + tail
    t = re.sub(r"<!-- PINS-BEGIN -->.*<!-- PINS-END -->", lambda m: "<!-- PINS-BEGIN -->\n```\n" + render_pins(pins) + "\n```\n<!-- PINS-END -->", t, flags=re.S)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ruling", default=None)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--check-only", action="store_true", help="build and verify in memory, write nothing")
    a = ap.parse_args(argv)
    pkg = os.path.abspath(a.pkg)
    try:
        pins = compute_pins(pkg)
        text = build_sealed(open(a.src, encoding="utf-8").read(), pins, open(a.ruling, encoding="utf-8").read() if a.ruling else None)
    except SealRefused as e:
        print("ZZSEAL_REFUSED", e)
        return 1
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if a.check_only:
        print("ZZSEAL_CHECK_OK", sha)
        return 0
    open(os.path.join(pkg, CH.SEALED_NAME), "w", encoding="utf-8", newline="\n").write(text)
    open(os.path.join(pkg, "PREREG_SEARCH_P2_CONFIRM.sha256"), "w", encoding="utf-8", newline="\n").write(sha + "\n")
    open(os.path.join(pkg, "SEALED_AT.txt"), "w", encoding="utf-8", newline="\n").write(datetime.datetime.now().astimezone().isoformat(timespec="seconds") + "\n" + sha + "\n")
    import io
    buf = io.StringIO()
    ch = CH.Chain(CH.Cfg(pkg), out=buf)
    try:
        ch.check_seal()
    except CH.StopChain as e:
        print("ZZSEAL_GUARD_REFUSED_ITS_OWN_SEAL", e.reason, e.detail)
        return 1
    print("ZZSEALED", sha, os.path.join(pkg, CH.SEALED_NAME))
    return 0


if __name__ == "__main__":
    sys.exit(main())
