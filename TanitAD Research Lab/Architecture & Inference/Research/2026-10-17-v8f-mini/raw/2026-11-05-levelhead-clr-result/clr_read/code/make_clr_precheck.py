#!/usr/bin/env python3
"""make_clr_precheck.py -- builds code/clr_precheck.py from cl_read/code/cl_precheck.py (sha256 pinned): role rename CL -> CLR plus the exact-once edits below (a replacement that does not match exactly once aborts).
CLR's argv base is RAPR-P01's argv (`v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json`), not C0's; the marker is clr_markers/RUN_CLR_{s}.done; the marker's evidence must carry recover_n > 0, level_head on, the frozen W;
RAPR-P01's own checkpoint (the attribution base) is checked against its SHA256SUMS.txt and recorded.   usage: python -I make_clr_precheck.py [--check]"""
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SRC = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read/code/cl_precheck.py"
SRC_SHA = "91827c90959a5aefd54475576ce7e60ff00de609ffd83af9ebc3db21b967787a"
DST = f"{HERE}/clr_precheck.py"
DOC = '''"""clr_precheck.py -- the pulled-file controls of the CLR read chain, in seconds, BEFORE any rollout (raw/CLR_PRECHECK.json; exit 0 = PASS, 8 = a control failed, 5 = a file is absent).

Per seed s in {0, 1}, run dir  <rdb>/M2b_s{s}_recft_CLR/  and marker  <rdb>/clr_markers/RUN_CLR_{s}.done :
  KP1  ckpt_last.pt, SHA256SUMS.txt, argv.json exist (REFUSED when the checkpoint is absent: no CLR checkpoint, no read)
  KP2  sha256(ckpt_last.pt) == the SHA256SUMS.txt line == the box-2 marker's evidence.ckpt_sha256 (a marker without one is refused)
  KP3  argv audit: the argv equals RAPR-P01's argv (<rdb>/M2b_s{s}_recft_RAPRP01/argv.json) plus EXACTLY `--level-head on --w-level W` (``--out`` ignored), W = the frozen W (cl_prereg/FROZEN_W.txt),
       the same W at both seeds, `--seed s`, the recovery lever (every --recover-* token) equal to RAPR-P01's
  KP4  the checkpoint's own cfg: level_head == 'on', w_level == W, step == the argv's --steps; the state dict carries every level-head key (lvl_head.*, e_level.*)
  KP5  REPORTED (not a gate): sum |e_level| of the trained checkpoint.  Exactly 0.0 means the lever never reached O in training (the read is then a read of a vacuous head).
  KP6  the box-2 marker's evidence: recover_n > 0 (the recovery really perturbed samples), level_head on, w_level == W
  KP7  the attribution base: RAPR-P01's own checkpoint M2b_s{s}_recft_RAPRP01/ckpt_last.pt present, sha256 == its SHA256SUMS.txt line (recorded as raprp01_ckpt_sha256), cfg level_head not 'on'
No GPU, torch is used to read the checkpoint dict only (CPU).
usage: python -I clr_precheck.py [--rdb E:/Projects/TanitAD-artifacts/v8f-recover] [--frozen-w <file>] [--raw <dir>] [--no-ckpt-load]"""
'''
RAPRP01_BLOCK = '''        bd = f"{a.rdb}/M2b_s{s}_recft_RAPRP01"
        if not (os.path.isfile(f"{bd}/ckpt_last.pt") and os.path.isfile(f"{bd}/SHA256SUMS.txt")):
            res["failures"].append(f"{k}: the attribution base {bd}/ckpt_last.pt or SHA256SUMS.txt is absent")
            absent = True
        else:
            gb = sha256_file(f"{bd}/ckpt_last.pt")
            sb = [l.split()[0] for l in open(f"{bd}/SHA256SUMS.txt", encoding="utf-8") if l.strip().split()[-1].lstrip("*").endswith("ckpt_last.pt")]
            r["raprp01_ckpt_sha256"], r["raprp01_sha256sums_line"] = gb, sb[0] if sb else None
            if not sb or gb != sb[0]:
                res["failures"].append(f"{k}: RAPRP01 ckpt sha256 {gb} != its SHA256SUMS.txt {sb[0] if sb else None}")
            if not a.no_ckpt_load:
                import torch
                bk = torch.load(f"{bd}/ckpt_last.pt", map_location="cpu", weights_only=False)
                r["raprp01_cfg_level_head"] = bk["cfg"].get("level_head")
                if bk["cfg"].get("level_head") == "on":
                    res["failures"].append(f"{k}: the attribution base RAPRP01 has level_head on (it must be the recovery recipe WITHOUT the level head)")
        res["seeds"][k] = r'''
EVIDENCE_BLOCK = '''                    res["failures"].append(f"{k}: ckpt sha256 {got} != box-2 marker {ev}")
                rn = evd.get("recover_n")
                r["marker_recover_n"] = rn
                if not (isinstance(rn, (int, float)) and not isinstance(rn, bool) and rn > 0):
                    res["failures"].append(f"{k}: marker evidence.recover_n {rn!r} is not > 0 (the recovery lever perturbed no sample)")
                if evd.get("level_head") != "on" or not _same_num(evd.get("w_level"), w):
                    res["failures"].append(f"{k}: marker evidence level_head {evd.get('level_head')!r} / w_level {evd.get('w_level')!r} is not on / {w!r}")'''
EDITS = [
    ("span", "doc", '"""cl_precheck.py -- the pulled-file controls', "import argparse", DOC),
    ("edit", "init_ckpt_key", '"init_ckpt_is_c0s": cl.get("--init-ckpt") == c0.get("--init-ckpt")}', '"init_ckpt_is_the_bases": cl.get("--init-ckpt") == c0.get("--init-ckpt")}'),
    ("edit", "recovery", '    c["diff"] = {"added": added,',
     '    rec = lambda d: {k: v for k, v in d.items() if k.startswith("--recover")}\n    p = str(rec(cl).get("--recover-p", "0"))\n    c["recovery_lever_is_RAPRP01s"] = bool(rec(cl) == rec(c0) and p.replace(".", "", 1).isdigit() and float(p) > 0)      # the lever is ON (p > 0) and equal to RAPR-P01 token for token\n    c["diff"] = {"added": added,'),
    ("all", "dir", 'f"{a.rdb}/M2b_s{s}_recft_C0/argv.json"', 'f"{a.rdb}/M2b_s{s}_recft_RAPRP01/argv.json"'),
    ("edit", "rundir", 'f"{a.rdb}/M2b_s{s}_recft_CL"', 'f"{a.rdb}/M2b_s{s}_recft_CLR"'),
    ("edit", "marker", 'mk = f"{a.rdb}/cl_markers/RUN_CL_{s}.done"', 'mk = f"{a.rdb}/clr_markers/RUN_CLR_{s}.done"'),
    ("edit", "marker_msg", 'RUN_CL_{s}.done carries no evidence.ckpt_sha256', 'RUN_CLR_{s}.done carries no evidence.ckpt_sha256'),
    ("edit", "evd", '                ev = (json.load(open(mk, encoding="utf-8")).get("evidence") or {}).get("ckpt_sha256")',
     '                evd = json.load(open(mk, encoding="utf-8")).get("evidence") or {}\n                ev = evd.get("ckpt_sha256")'),
    ("edit", "evidence_checks", '                    res["failures"].append(f"{k}: ckpt sha256 {got} != box-2 marker {ev}")', EVIDENCE_BLOCK),
    ("edit", "raprp01", '        res["seeds"][k] = r', RAPRP01_BLOCK),
    ("edit", "card", '"card": "CLR read chain pre-check"', '"card": "CLR read chain pre-check (PREREG_LEVELHEAD_A2.md)"'),
]


def build():
    raw = open(SRC, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "cl_precheck.py is not the pinned file"
    t = raw.decode("utf-8")
    t = re.sub(r"(?<![A-Za-z0-9_])CL(?![A-Za-z0-9])", "CLR", t)
    for e in EDITS:
        if e[0] == "edit":
            _, name, old, new = e
            if t.count(old) != 1:
                raise SystemExit(f"edit {name!r}: matches {t.count(old)} times (must be 1)")
            t = t.replace(old, new)
        elif e[0] == "all":
            _, name, old, new = e
            if t.count(old) < 1:
                raise SystemExit(f"all {name!r}: matches 0 times")
            t = t.replace(old, new)
        else:
            _, name, start, end, new = e
            if t.count(start) != 1 or t.count(end) != 1:
                raise SystemExit(f"span {name!r}: start/end match {t.count(start)}/{t.count(end)} times (must be 1/1)")
            i = t.index(start)
            t = t[:i] + new + t[t.index(end, i):]
    return t


if __name__ == "__main__":
    out = build()
    if "--check" in sys.argv:
        cur = open(DST, encoding="utf-8", newline="").read()
        print("clr_precheck.py == generated" if cur == out else "clr_precheck.py DIFFERS from generated")
        sys.exit(0 if cur == out else 1)
    open(DST, "w", encoding="utf-8", newline="\n").write(out)
    print("wrote", DST, len(out))
