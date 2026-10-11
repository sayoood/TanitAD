#!/usr/bin/env python3
"""Test of sft4r_chain.sh's two gate helpers, extracted VERBATIM from the chain and run under bash on SFT-4's banked
smoke logs (raw/2026-10-04-sft4-smoke/: r2_smoke2.log = preflight PASS, r2_smoke2_g3mut.log = the G3 regression, RED).

Why it exists (PREREG_SFT4R.md Amendment 2): the chain used to GREP the preflight JSON, and json.dumps writes G3's "ok"
key LAST, so the G3 grep could never match and the chain would have refused every launch. Each helper is tested in
both directions -- the reading it must accept AND the one it must reject -- so a helper that always answers the same
way fails here.

Run: python refe/test_sft4r_chain.py   -> ZZTEST_PASS <n>     (needs bash on PATH; Git Bash on the dev box)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.join(HERE, "sft4r_chain.sh")
SMOKE = os.path.join(HERE, "..", "raw", "2026-10-04-sft4-smoke")
G3 = "G3_copies_equal_model_at_step0"
G9 = "G9_pdm_targets_in_bank"


def _helpers(text):
    m = re.search(r"^last_exit\(\).*?^PYEOF$", text, re.S | re.M)
    assert m, "the helpers are no longer in sft4r_chain.sh"
    return m.group(0) + "\n}\n"


def _sh(helpers_path, script):
    py = sys.executable.replace("\\", "/")
    r = subprocess.run([shutil.which("bash"), "-c", f'PY_C="{py}"; . "$0"; {script}', helpers_path],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main() -> int:
    n = 0
    text = open(CHAIN, encoding="utf-8").read()
    assert "\r\n" not in text, "sft4r_chain.sh must stay LF"; n += 1
    # the chain USES the parser for both regressions, and the order-dependent greps are gone
    assert f"gate_is $OUT/mutate_g3.log {G3} ok false" in text; n += 1
    assert f"gate_is $OUT/mutate.log {G9} ok false" in text; n += 1
    assert '{\\"ok\\": false' not in text and '{\\"train_pdm\\": 0' not in text; n += 1
    ok_log, mut_log = (os.path.join(SMOKE, f) for f in ("r2_smoke2.log", "r2_smoke2_g3mut.log"))
    # the defect, reproduced: the old grep MISSES the real G3 failure (its "ok" is the last key, not the first)
    old = open(mut_log, encoding="utf-8").read()
    assert '"%s": {"ok": false' % G3 not in old and '"ok": false' in old; n += 1
    with tempfile.TemporaryDirectory() as tmp:
        h = os.path.join(tmp, "helpers.sh")
        open(h, "w", encoding="utf-8", newline="\n").write(_helpers(text))
        q = lambda p: p.replace("\\", "/")  # noqa: E731
        gate = lambda log, g, k, v: _sh(h, f'gate_is "{q(log)}" {g} {k} {v}')[0]  # noqa: E731
        # gate_is: accept the true reading, reject the false one, on both logs
        assert gate(mut_log, G3, "ok", "false") == 0; n += 1
        assert gate(mut_log, G3, "ok", "true") != 0; n += 1
        assert gate(ok_log, G3, "ok", "false") != 0; n += 1
        assert gate(ok_log, G3, "ok", "true") == 0; n += 1
        assert gate(ok_log, G9, "ok", "false") != 0; n += 1
        assert gate(ok_log, G9, "train_pdm", "0") != 0; n += 1
        assert gate(ok_log, G9, "train_pdm", "1708") == 0; n += 1
        assert gate(ok_log, "G9_absent", "ok", "false") != 0; n += 1
        none, two = os.path.join(tmp, "none.log"), os.path.join(tmp, "two.log")
        open(none, "w", encoding="utf-8").write("ZZSFT_PREFLIGHT_FAIL\n")
        open(two, "w", encoding="utf-8").write(old + old)
        assert gate(none, G3, "ok", "false") != 0; n += 1   # no preflight row
        assert gate(two, G3, "ok", "false") != 0; n += 1    # two rows: a log written across attempts
        # last_exit: the LAST marker decides, in both directions
        def last(lines, marker="ZZPDM_TRAIN_EXIT"):
            p = os.path.join(tmp, "ex.log")
            open(p, "w", encoding="utf-8", newline="\n").write("".join(x + "\n" for x in lines))
            return _sh(h, f'last_exit {marker} "{q(p)}"')[1]
        assert last(["ZZPDM_TRAIN_EXIT 0", "ZZPDM_TRAIN_EXIT 1"]) == "1"; n += 1
        assert last(["ZZPDM_TRAIN_EXIT 1", "ZZPDM_TRAIN_EXIT 0"]) == "0"; n += 1
        assert last(["ZZPDM_TRAIN_EXIT_X 0", "ZZPDM_TRAIN_EXIT 2"]) == "2"; n += 1
        assert last(["ZZLANE_TRAIN_EXIT 0 9"], "ZZLANE_TRAIN_EXIT") == "0 9"; n += 1
        assert _sh(h, f'last_exit ZZPDM_TRAIN_EXIT "{q(tmp)}/missing.log"')[1] == ""; n += 1
    print(f"ZZTEST_PASS {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
