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
]


def run_pytest(d, tag):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8")
    r = subprocess.run([PY, "-m", "pytest", "test_dose.py", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={os.path.join(d, 'pt_' + tag)}"], cwd=d, env=env, capture_output=True, text=True)
    shutil.rmtree(os.path.join(d, "pt_" + tag), ignore_errors=True)                       # C: is nearly full: keep nothing between mutations
    return r.returncode, (r.stdout + r.stderr)[-400:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--scratch", default=os.path.join(PKG, "tmp", "mut"))
    a = ap.parse_args(argv)
    i, n = (int(x) for x in a.shard.split("/"))
    d = f"{a.scratch}_{i}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for f in ("dose_chain.py", "test_dose.py", "seal_dose.py", "mutation_dose.py", "cost_dose.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(d, f))
    src = open(os.path.join(d, F), encoding="utf-8", newline="").read()
    res, ok_all = [], True
    if i == 0:
        rc, tail = run_pytest(d, "control")
        res.append({"id": "CONTROL (unmutated)", "rc": rc, "caught": None, "control_passes": rc == 0})
        ok_all &= rc == 0
        print("CONTROL", "PASS" if rc == 0 else "FAIL", flush=True)
    for k, (desc, old, new) in enumerate(MUT):
        if k % n != i:
            continue
        cnt = src.count(old)
        if cnt != 1:
            res.append({"id": k, "desc": desc, "caught": False, "error": f"pattern occurs {cnt} times (must be exactly 1)"})
            ok_all = False
            print(k, desc, "BAD PATTERN", cnt, flush=True)
            continue
        open(os.path.join(d, F), "w", encoding="utf-8", newline="").write(src.replace(old, new))
        rc, tail = run_pytest(d, f"m{k}")
        caught = rc != 0
        ok_all &= caught
        res.append({"id": k, "desc": desc, "caught": caught, "rc": rc})
        print(k, "RED (caught)" if caught else "GREEN (NOT CAUGHT)", desc, flush=True)
        open(os.path.join(d, F), "w", encoding="utf-8", newline="").write(src)
    json.dump({"shard": a.shard, "n_mutations_total": len(MUT), "results": res, "all_caught": ok_all}, open(a.out, "w"), indent=1)
    shutil.rmtree(d, ignore_errors=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
