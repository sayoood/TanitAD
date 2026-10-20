#!/usr/bin/env python3
"""Mutation check of the NEW lines in dose_chain.py: each mutation changes ONE line in a scratch copy of this directory, and `test_dose.py` MUST then fail.  A line whose change leaves the suite green is untested.
The sealed C1 files are imported from the spc package and are NOT mutated here (their own mutation check is spc/code/mutation_check.py).
    python mutation_dose.py --out ../logs/mutations_dose.json [--shard 0/2]     (the unmutated copy must pass first: the control)
Exit 0 only if the control passes and every mutation of the shard is caught."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = os.environ.get("MUT_PY", sys.executable)
F = "dose_chain.py"
F2 = "seal_dose.py"                                  # an entry's optional 4th element names the file it mutates (default F)
MUT = [
    ("O4 pin: file == C1-seal pin == constant", 'if not (want == got == O4_SHA256):', 'if False:'),
    ("the pins swap: config_sha256 := the O4 sha", 'return {**pins, "config_c8_sha256": pins["config_sha256"], "config_sha256": want}   # @SWAP', 'return {**pins, "config_c8_sha256": pins["config_sha256"]}   # @SWAP'),
    ("check_seal verifies C1's seal against the C8 file", 'c.config = CONFIG_C8\n', 'c.config = o4_path\n'),
    ("make_cfg default config is the O4 file", 'dose_pkg=None, config=CONFIG_O4, **kw):', 'dose_pkg=None, config=CONFIG_C8, **kw):'),
    ("O4 levers constant (M 256)", '"m_pop": 256, "n_elite": 32, "widen": 2.0}', '"m_pop": 64, "n_elite": 32, "widen": 2.0}'),
    ("O4 sha constant", 'O4_SHA256 = "1ebd7196', 'O4_SHA256 = "0ebd7196'),
    ("the bundle is C1's, not the dose work dir's", 'return os.path.join(self.cfg.c1_work, "B_REST", "bundle")', 'return os.path.join(self.cfg.work, "B_REST", "bundle")'),
    ("gpu_stage refuses", 'self.stop(3, "DOSE_NEVER_BUILDS_A_BUNDLE",', '(3, "DOSE_NEVER_BUILDS_A_BUNDLE",'),
    ("cid_gate refuses", 'self.stop(3, "DOSE_HAS_NO_CID",', '(3, "DOSE_HAS_NO_CID",'),
    ("dose work dir != C1 work dir", 'if os.path.abspath(c.work) == os.path.abspath(c.c1_work):', 'if False:'),
    ("C1 stages must all be done", 'if miss:                                                                       # @C1STAGES', 'if False:   # @C1STAGES'),
    ("C1's bundle on disk is the bundle C1 searched", 'if CH.sha256_file(os.path.join(self.bundle_dir("C1"), "bundle.npz")) != self.c1_bundle_sha:', 'if False:'),
    ("G-SAMEBUNDLE in the dose merged file", 'if getattr(self, "c1_bundle_sha", None) is None or meta.get("bundle_sha256") != self.c1_bundle_sha:', 'if False:'),
    ("amendment text vs its sha256", 'if hashlib.sha256(text).hexdigest() != want:', 'if False:'),
    ("amendment Status must be SEALED", 'if not m or not m.group(0).startswith("Status: **SEALED"):', 'if False:'),
    ("amendment pin mismatch stops", 'if bad:                                                                        # @AMENDPINS', 'if False:  # @AMENDPINS'),
    ("amendment pins C1's prereg sha", 'if ap.get("c1_prereg_sha256") != c1_sha:', 'if False:'),
    ("amendment pins the O4 and C8 shas", 'if ap.get("config_o4_sha256") != O4_SHA256 or ap.get("config_c8_sha256") != pins["config_c8_sha256"]:', 'if False:'),
    ("amendment must pin dose_chain.py", 'if "dose_chain.py" not in codes:', 'if False:'),
    ("amendment code pins equal the files", 'if not os.path.isfile(q) or CH.sha_lf(q) != v:', 'if False:'),
    ("G-ORDER of the dose analysis", 'if miss:                                                                       # @DOSEORDER', 'if False:  # @DOSEORDER'),
    ("contrast sign: O4 minus C8", 'd = AB.POINTS * (pd - pc)', 'd = AB.POINTS * (pc - pd)'),
    ("contrast refuses misaligned tokens", 'if tok != [str(t) for t in so_c1["token"]] or tok != [str(t) for t in lab["token"]] or tok != [str(t) for t in lab_c1["token"]]:', 'if False:'),
    ("INERT special case in the next arm", 'if c1_outcome == "O4_INERT":', 'if False:'),
    ("the dose NEXT arm replaces C1's", 'res["next_arm"] = next_arm(res["outcome"], c1_outcome)', 'pass'),
    ("stamps name the second read", 'SECOND READ OF THE SAME TOKENS', 'A FRESH DRAW OF TOKENS'),
    ("marker prefix ZZDOSE", 'print("ZZDOSE_" + marker', 'print("ZZSPC_" + marker'),
    ("dose merged file keeps C1's rules (no head, ckpt, O4 sha, data clause)", 'ok, why = super().merged_valid(path, which, n, pins)', 'ok, why = True, ""'),
    ("G-REACH: no stamp, no search", 'if rc != 0 or not self.stamp_ok("C1", pins):', 'if False:'),
    ("bundle gate", 'if not ok:\n            self.stop(3, "G_BUNDLE_INVALID"', 'if False:\n            self.stop(3, "G_BUNDLE_INVALID"'),
    ("C1 RESULT_SPC outcome is one of C1's outcomes", 'if res.get("outcome") not in AC.OUTCOMES:', 'if False:'),
    ("C1's merged search + labels valid under C1's C8 pin", 'if not ok:\n            self.stop(5, "C1_NOT_DONE", f"C1\'s merged', 'if False:\n            self.stop(5, "C1_NOT_DONE", f"C1\'s merged'),
    # ---- MM_RULING_DOSE.md DQ5: G-SUBID and G-SELFREP (and the rows_identical comparison they share)
    ("the replicate's rc is checked", 'self.stop(3, f"G_{tag}_NOT_PASSED", f"the replicate search exited rc={rc}")', '(3, f"G_{tag}_NOT_PASSED", f"the replicate search exited rc={rc}")'),
    ("the replicate must write a shard", 'self.stop(3, f"G_{tag}_NOT_PASSED", "the replicate wrote no shard")', '(3, f"G_{tag}_NOT_PASSED", "the replicate wrote no shard")'),
    ("the replicate's meta is checked", 'if not pins_ok:', 'if False:'),
    ("the bitwise comparison stops the chain", 'if not res["pass"]:', 'if False:'),
    ("a stale shard is removed before a replicate", 'shutil.rmtree(out_dir, ignore_errors=True)', 'pass'),
    ("rows_identical: key sets", 'if sorted(set(ref) - skip) != sorted(set(new) - skip):', 'if False:'),
    ("rows_identical: the bytes", 'if not same:', 'if False:'),
    ("rows_identical: non-vacuity", 'if not non_vacuous:', 'if False:'),
    ("rows_identical: the search seeds", 'for k in ROW_KEYS + ("seeds",):', 'for k in ROW_KEYS:'),
    ("rows_identical: bundle_index is compared", 'ROW_KEYS = CID.SEARCH_KEYS + ("token", "bundle_index")', 'ROW_KEYS = CID.SEARCH_KEYS + ("token",)'),
    ("rows_identical: timing values are not part of the identity", 'ROW_KEYS = CID.SEARCH_KEYS + ("token", "bundle_index")', 'ROW_KEYS = CID.SEARCH_KEYS + ("token", "bundle_index", "t_search_s")'),
    ("G-SUBID runs the C8 config", 'c.work, c.config = c.c1_work, CONFIG_C8', 'c.work, c.config = c.c1_work, c.config'),
    ("G-SUBID restores the dose work dir and config", 'finally:\n            c.work, c.config = old', 'finally:\n            pass'),
    ("G-SUBID needs C1's C8 stamp", 'if not self.stamp_ok("C1", p8):', 'if False:'),
    ("G-SELFREP replicates 8 tokens", 'SELFREP_N, SUBID_N = 8, 16', 'SELFREP_N, SUBID_N = 4, 16'),
    ("G-SUBID replicates 16 tokens", 'SELFREP_N, SUBID_N = 8, 16', 'SELFREP_N, SUBID_N = 8, 8'),
    ("G-SUBID is in the run", '            self.subid_gate(pins, c1)\n', ''),
    ("G-SELFREP is in the run", '            self.selfrep_gate(pins, merged)\n', ''),
    ("G-ORDER lists SUBID_PASS", '"BUNDLE_OK", "SUBID_PASS", "REACH_PASS"', '"BUNDLE_OK", "REACH_PASS"'),
    ("G-ORDER lists SELFREP_PASS", '"SEARCH_DONE", "SELFREP_PASS", "LABEL_DONE")', '"SEARCH_DONE", "LABEL_DONE")'),
    # ---- MM_RULING_DOSE.md DQ4: the mechanical C2 config rule
    ("c2 row 1: C1 licensed, dose not -> C8", 'cfg, row = "C8", 1', 'cfg, row = "O4", 1'),
    ("c2 row 3: the contrast decides", 'cfg, row = ("O4" if lo_pos else "C8"), 3', 'cfg, row = "O4", 3'),
    ("c2 row 2: O2 is NOT enough on the dose-only path", 'elif dose_strict:', 'elif dose_any:'),
    ("c2 row 4: no C2", 'cfg, row = None, 4', 'cfg, row = "C8", 4'),
    ("c2: the contrast bound is strict (lower CI > 0)", 'lo_pos = bool(float(contrast_lo) > 0.0)', 'lo_pos = bool(float(contrast_lo) >= 0.0)'),
    ("licence: strict means O1 only", 'ok = ("O1_CONFIRMED",) if strict else ("O1_CONFIRMED", "O2_DIRECTIONAL")', 'ok = ("O1_CONFIRMED", "O2_DIRECTIONAL")'),
    ("licence: moved share >= 5 %", 'LICENCE_SAFETY_BAR, LICENCE_MOVED_BAR = AC.BAR_SAFETY, 0.05', 'LICENCE_SAFETY_BAR, LICENCE_MOVED_BAR = AC.BAR_SAFETY, 0.0'),
    ("licence: safety bar -0.25", 'LICENCE_SAFETY_BAR, LICENCE_MOVED_BAR = AC.BAR_SAFETY, 0.05', 'LICENCE_SAFETY_BAR, LICENCE_MOVED_BAR = -1.0, 0.05'),
    ("licence: a missing term is not a licence", 'except (KeyError, TypeError, ValueError):\n        return False', 'except (KeyError, TypeError, ValueError):\n        return True'),
    ("the lower CI (not the upper) enters the C2 config", 'res["contrast_O4_minus_C8"]["pooled"]["lo"])', 'res["contrast_O4_minus_C8"]["pooled"]["hi"])'),
    ("C2CONFIG is printed", 'self.emit("C2CONFIG",', 'self.emit("C2CONFIGX",'),
    ("C1's licence terms are required from RESULT_SPC.json", 'if not isinstance(res.get("safety_points"), dict)', 'if False and not isinstance(res.get("safety_points"), dict)'),
    ("the ruling is pinned in the amendment", 'if not os.path.isfile(rp) or ap.get("ruling_sha256") != CH.sha_lf(rp):', 'if False:'),
    # ---- REVISION 1 (MM_RULING_C2FINAL.md point 1): "moved share" = inert.frac_moved.  The first block mutates dose_chain.py, the second (4th element F2) seal_dose.py
    ("R1 licence reads inert.frac_moved, not moved_share", 'return float(read["inert"]["frac_moved"])', 'return float(read["moved_share"])'),
    ("R1 licence: an absent frac_moved reads NaN, never a number", 'return float("nan")', 'return 1.0'),
    ("R1 licence: the moved term is applied", 'moved = frac_moved_of(read) >= LICENCE_MOVED_BAR', 'moved = True'),
    ("R1 licence: the moved bar is inclusive (>= 5 %)", 'moved = frac_moved_of(read) >= LICENCE_MOVED_BAR', 'moved = frac_moved_of(read) > LICENCE_MOVED_BAR'),
    ("R1 licence: moved_share is never a fallback", 'moved = frac_moved_of(read) >= LICENCE_MOVED_BAR',
     'moved = (frac_moved_of(read) if frac_moved_of(read) == frac_moved_of(read) else float(read.get("moved_share", 0.0))) >= LICENCE_MOVED_BAR'),
    ("R1 dose read: the licence gets the dose read's OWN inert block", '"safety_points": res["safety_points"], "inert": res["inert"]}', '"safety_points": res["safety_points"], "moved_share": res["moved_share"]}'),
    ("R1 C1 read: the licence gets C1's inert block", '"safety_points": res["safety_points"], "inert": res["inert"], "moved_share": res.get("moved_share"), "result_sha256"',
     '"safety_points": res["safety_points"], "moved_share": res.get("moved_share"), "result_sha256"'),
    ("R1 C1 read: moved_share is still reported (descriptive)", '"inert": res["inert"], "moved_share": res.get("moved_share"), "result_sha256"', '"inert": res["inert"], "result_sha256"'),
    ("R1 C1 read: an absent inert.frac_moved is refused by name", 'if isinstance(fm, bool) or not isinstance(fm, (int, float)):', 'if False:'),
    ("R1 C1 read: a bool is not a frac_moved", 'if isinstance(fm, bool) or not isinstance(fm, (int, float)):', 'if not isinstance(fm, (int, float)):'),
    ("R1 C1 read: an absent inert block is an absent frac_moved", 'if isinstance(res.get("inert"), dict) else None', 'if isinstance(res.get("inert"), dict) else 0.5'),
    ("R1 c2_config records c1 / dose frac_moved the right way round", '"c1": frac_moved_of(c1_read), "dose": frac_moved_of(dose_read)}}', '"c1": frac_moved_of(dose_read), "dose": frac_moved_of(c1_read)}}'),
    ("R1 c2_config names the quantity", '"quantity": "inert.frac_moved"', '"quantity": "moved_share"'),
    ("R1 c2_config records the bar applied", '"bar": LICENCE_MOVED_BAR', '"bar": 0.5'),
    ("R1 amendment: R1 and the original (with their .sha256) are all required", 'if not all(os.path.isfile(q) for q in (p, sp, p0, sp0)):', 'if not all(os.path.isfile(q) for q in (p, sp)):'),
    ("R1 amendment: the original equals its .sha256", 'if hashlib.sha256(orig).hexdigest() != open(sp0, encoding="utf-8").read().split()[0].strip().lower():', 'if False:'),
    ("R1 amendment: ruling_c2final_sha256 is pinned and equals the file", 'if not os.path.isfile(rp2) or ap.get("ruling_c2final_sha256") != CH.sha_lf(rp2):', 'if False:'),
    ("R1 amendment: amendment_original_sha256 is pinned", 'if ap.get("amendment_original_sha256") != hashlib.sha256(orig).hexdigest():', 'if False:'),
    ("R1 amendment: exactly one REVISION 1 note block", 'if t.count(REV1_BEGIN) != 1 or t.count(REV1_END) != 1:', 'if False:'),
    ("R1 amendment: the terms text equals the original's", 'if terms_text(t) != terms_text(orig.decode("utf-8")):', 'if False:'),
    ("terms_text normalises CRLF", r't = text.replace("\r\n", "\n")', 't = text'),
    ("terms_text removes the Status line", r't = re.sub(r"^Status: .*\n", "", t, count=1, flags=re.M)', 'pass'),
    ("terms_text removes the note block", r't = re.sub(re.escape(REV1_BEGIN) + r".*?" + re.escape(REV1_END) + r"\n?", "", t, flags=re.S)', 'pass'),
    ("terms_text removes the pins block", r'return re.sub(r"<!-- PINS-BEGIN -->.*?<!-- PINS-END -->\n?", "", t, flags=re.S)', 'return t'),
    ("seal R1: never overwrites", 'raise SealRefused("a sealed REVISION 1 already exists here: never overwritten")', 'pass', F2),
    ("seal R1: the original equals its .sha256", 'if orig_sha != open(j(DC.AMEND_SHA), encoding="utf-8").read().split()[0].strip().lower():', 'if False:', F2),
    ("seal R1: the original is SEALED", 'if not m or not m.group(0).startswith("Status: **SEALED"):', 'if False:', F2),
    ("seal R1: MM_RULING_C2FINAL.md is the pinned ruling", 'if CH.sha_lf(j(DC.RULING_C2FINAL_NAME)) != DC.RULING_C2FINAL_SHA256:', 'if False:', F2),
    ("seal R1: MM_RULING_DOSE.md is the ruling the original pinned", 'if CH.parse_pins(orig_text).get("ruling_sha256") != CH.sha_lf(j(DC.RULING_NAME)):', 'if False:', F2),
    ("seal R1: a note may not carry the note-begin delimiter", 'if DC.REV1_BEGIN in note_text or DC.REV1_END in note_text or "<!-- PINS-" in note_text:',
     'if DC.REV1_END in note_text or "<!-- PINS-" in note_text:', F2),
    ("seal R1: a note may not carry the note-end delimiter", 'if DC.REV1_BEGIN in note_text or DC.REV1_END in note_text or "<!-- PINS-" in note_text:',
     'if DC.REV1_BEGIN in note_text or "<!-- PINS-" in note_text:', F2),
    ("seal R1: a note may not carry a pins delimiter", 'if DC.REV1_BEGIN in note_text or DC.REV1_END in note_text or "<!-- PINS-" in note_text:',
     'if DC.REV1_BEGIN in note_text or DC.REV1_END in note_text:', F2),
    ("seal R1: the note names the ruling by its sha256", 'if DC.RULING_C2FINAL_SHA256[:8] not in note_text:', 'if False:', F2),
    ("seal R1: the terms-equality guard", 'if DC.terms_text(sealed) != DC.terms_text(orig_text):', 'if False:', F2),
    ("seal R1: the dose ruling stays pinned", 'pins = build_pins(c1_pkg, code_dir, j(DC.RULING_NAME), extra=', 'pins = build_pins(c1_pkg, code_dir, None, extra=', F2),
    ("seal R1: the original's sha is pinned", 'f"amendment_original_sha256 {orig_sha}"', 'f"amendment_original_sha256 {orig_sha[::-1]}"', F2),
    ("seal R1: the C2FINAL ruling is pinned", 'f"ruling_c2final_sha256 {CH.sha_lf(j(DC.RULING_C2FINAL_NAME))}"', 'f"ruling_c2final_sha256 {DC.RULING_C2FINAL_SHA256[::-1]}"', F2),
    ("seal R1: the status says REVISION 1", '(REVISION 1 of AMENDMENT_DOSE.md [sha256', '(rev. of AMENDMENT_DOSE.md [sha256', F2),
    ("build_r1_text: the note sits directly under the Status line", r't = t[:m.start()] + status + "\n" + block + t[m.end():]', r't = t[:m.start()] + block + status + "\n" + t[m.end():]', F2),
    ("build_r1_text: an original without a Status line is refused", 'if not m:\n        raise SealRefused("the original has no Status line")', 'if False:\n        raise SealRefused("the original has no Status line")', F2),
    ("build_r1_text: exactly one pins block", 'if t.count("<!-- PINS-BEGIN -->") != 1 or t.count("<!-- PINS-END -->") != 1:', 'if False:', F2),
    ("build_r1_text: an original that already has a note block is refused", 'if DC.REV1_BEGIN in t or DC.REV1_END in t:', 'if False:', F2),
    ("CLI: without --revision1 the draft and the ruling are required", 'if not a.src or not a.ruling:', 'if False:', F2),
    ("CLI: prints ZZSEAL_R1_OK", 'print("ZZSEAL_R1_OK", sha, flush=True)', 'print("ZZSEAL_OK", sha, flush=True)', F2),
    ("CLI: the default note is <pkg>/REVISION1_NOTE.md", 'a.note or os.path.join(a.pkg, "REVISION1_NOTE.md")', 'a.note or os.path.join(a.pkg, "NOTE.md")', F2),
]


PKG_FILES = ("AMENDMENT_DOSE.md", "AMENDMENT_DOSE.sha256", "AMENDMENT_DOSE_DRAFT.md", "MM_RULING_DOSE.md", "MM_RULING_C2FINAL.md", "REVISION1_NOTE.md")      # the package files the real-file tests read at `dirname(code dir)`
CODE_FILES = ("dose_chain.py", "test_dose.py", "seal_dose.py", "mutation_dose.py", "cost_dose.py")


def run_pytest(d, tag, k_expr=None):
    """pytest in <d>/code (the scratch mirrors the package layout, so the real-file tests RUN instead of skipping)"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8")
    cmd = [PY, "-m", "pytest", "test_dose.py", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(d, 'pt_' + tag)}"] + (["-k", k_expr] if k_expr else [])
    r = subprocess.run(cmd, cwd=os.path.join(d, "code"), env=env, capture_output=True, text=True)
    shutil.rmtree(os.path.join(d, "pt_" + tag), ignore_errors=True)                       # C: is nearly full: keep nothing between mutations
    return r.returncode, (r.stdout + r.stderr)[-400:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--scratch", default=os.path.join(PKG, "tmp", "mut"))
    ap.add_argument("--only", default=None, help="comma-separated mutation indices to run (the control is skipped unless --control)")
    ap.add_argument("--control", action="store_true", help="run the unmutated control first even with --only")
    ap.add_argument("--k", default=None, help="a pytest -k expression: a mutation caught by this SUBSET is caught by the suite; a survivor must be re-run with the full suite")
    a = ap.parse_args(argv)
    i, n = (int(x) for x in a.shard.split("/"))
    only = None if a.only is None else {int(x) for x in a.only.split(",")}
    d = os.path.abspath(f"{a.scratch}_{i}")                                          # absolute: pytest runs with cwd = <d>/code
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "code"))
    for f in CODE_FILES:
        shutil.copy(os.path.join(HERE, f), os.path.join(d, "code", f))
    for f in PKG_FILES:
        if os.path.isfile(os.path.join(PKG, f)):
            shutil.copy(os.path.join(PKG, f), os.path.join(d, f))
    srcs = {fn: open(os.path.join(d, "code", fn), encoding="utf-8", newline="").read() for fn in (F, F2)}
    res, ok_all = [], True
    if i == 0 and (only is None or a.control):
        rc, tail = run_pytest(d, "control")
        res.append({"id": "CONTROL (unmutated)", "rc": rc, "caught": None, "control_passes": rc == 0})
        ok_all &= rc == 0
        print("CONTROL", "PASS" if rc == 0 else "FAIL", flush=True)
    for k, entry in enumerate(MUT):
        desc, old, new = entry[:3]
        fn = entry[3] if len(entry) > 3 else F
        if k % n != i or (only is not None and k not in only):
            continue
        cnt = srcs[fn].count(old)
        if cnt != 1:
            res.append({"id": k, "desc": desc, "file": fn, "caught": False, "error": f"pattern occurs {cnt} times (must be exactly 1)"})
            ok_all = False
            print(k, desc, "BAD PATTERN", cnt, flush=True)
            continue
        open(os.path.join(d, "code", fn), "w", encoding="utf-8", newline="").write(srcs[fn].replace(old, new))
        rc, tail = run_pytest(d, f"m{k}", a.k)
        caught = rc == 1                                                                  # pytest rc 1 = tests FAILED; rc 2 (collection / import error), 3, 4, 5 are an environment fault, never a catch
        ok_all &= caught
        res.append({"id": k, "desc": desc, "file": fn, "caught": caught, "rc": rc, "subset": a.k, "tail": tail[-160:]})
        print(k, "RED (caught)" if caught else "GREEN (NOT CAUGHT)", desc, flush=True)
        open(os.path.join(d, "code", fn), "w", encoding="utf-8", newline="").write(srcs[fn])
    json.dump({"shard": a.shard, "n_mutations_total": len(MUT), "results": res, "all_caught": ok_all}, open(a.out, "w"), indent=1)
    shutil.rmtree(d, ignore_errors=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
