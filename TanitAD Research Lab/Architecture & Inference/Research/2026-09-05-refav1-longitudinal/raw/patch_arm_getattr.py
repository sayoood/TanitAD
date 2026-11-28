# -*- coding: utf-8 -*-
"""The arm tool is called with a hand-built argparse.Namespace by
`tests/test_refav1_arm.py` and `tests/test_refav1_kin_contract.py`, so a bare
`a.<flag>` raises AttributeError for any flag the test did not set. Every
other optional flag in this tool already uses `getattr(a, ..., default)` for
exactly that reason (`kamm_mu`, `seed_kappa_ladder`); these two must too."""
import io
import sys
import time

PATH = sys.argv[1]


def rd(p, tries=25):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE read")


def wr(p, t, tries=25):
    for _ in range(tries):
        try:
            with io.open(p, "w", encoding="utf-8", newline="") as f:
                f.write(t)
            if len(rd(p)) == len(t):
                return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE write")


t = rd(PATH)
if 'getattr(a, "a_sustain_mode", "none") == "a0"' in t:
    print("ALREADY PATCHED")
    raise SystemExit(0)
EOL = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")
n0 = len(t)

PAIRS = [
    ('                    a_sustain = _a0 if a.a_sustain_mode == "a0" else None\n'
     '                    jerk_seam = _a0 if a.jerk_seam == "a0" else None\n',
     '                    a_sustain = (_a0 if getattr(a, "a_sustain_mode",\n'
     '                                              "none") == "a0" else None)\n'
     '                    jerk_seam = (_a0 if getattr(a, "jerk_seam", "off")\n'
     '                                 == "a0" else None)\n',
     "per-window getattr"),
    ('                        if a.a_sustain_mode != "none":\n',
     '                        if getattr(a, "a_sustain_mode", "none") != "none":\n',
     "guard a_sustain getattr"),
    ('                        if a.jerk_seam != "off":\n',
     '                        if getattr(a, "jerk_seam", "off") != "off":\n',
     "guard jerk_seam getattr"),
    ('                                    % (got_s, a.a_sustain_mode))\n',
     '                                    % (got_s, getattr(a, "a_sustain_mode",\n'
     '                                                      "none")))\n',
     "guard msg a_sustain"),
    ('                                    % (got_j, a.jerk_seam))\n',
     '                                    % (got_j, getattr(a, "jerk_seam",\n'
     '                                                      "off")))\n',
     "guard msg jerk_seam"),
]
for old, new, label in PAIRS:
    c = t.count(old)
    if c != 1:
        raise SystemExit("ANCHOR %s occurs %d (need 1) -- ABORT" % (label, c))
    t = t.replace(old, new, 1)
    print("  patched %s" % label)

if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK %s  %d -> %d chars" % (PATH, n0, len(t)))
