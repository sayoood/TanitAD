#!/usr/bin/env python3
"""make_clr_readout.py -- builds code/clr_readout.py and code/clr_merge_readout.py from A1's instrument (motion_c/cl_readout/code/cl_readout.py and merge_readout.py, sha256 pinned).
A1's instrument is run UNCHANGED except the checkpoint path (M2b_s{s}_recft_CLR), the banked-ticks path (this package's CLR_s{s} rollout), the output directory / file names and the windows-dump path (so A1's own raw/ is never overwritten);
every replacement must match exactly once.   usage: python -I make_clr_readout.py [--check]"""
import hashlib
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
A1 = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_readout/code"
PINS = {"cl_readout.py": "7ab6fb050591b4b7ac5615c7ac82682ea28adc02264e4a4d90fc95b535e6c0ae", "merge_readout.py": "ed20ee832d9464cca75d8071b0ab719a1d24d7eac82c4f7ca3b8e30827312cf9"}   # A1's files as run for the CL read-out
READOUT = [
    ('CKPT = f"{ART}/v8f-recover/M2b_s{{s}}_recft_CL/ckpt_last.pt"', 'CKPT = f"{ART}/v8f-recover/M2b_s{{s}}_recft_CLR/ckpt_last.pt"'),
    ('BANKED_TICKS = "F:/TanitAD-scratch/motion_c/cl_read/out/CL_s{s}/ticks/ticks_CL_s{s}.npz"', 'BANKED_TICKS = "F:/TanitAD-scratch/motion_c/clr_read/out/CLR_s{s}/ticks/ticks_CLR_s{s}.npz"'),
    ('OUT_DIR = f"{ART}/dagland/motion_c/cl_readout/raw"', 'OUT_DIR = f"{ART}/dagland/motion_c/clr_read/readout"'),
    ('f"CL_READOUT_s{a.seed}.json"', 'f"CLR_READOUT_s{a.seed}.json"'),
    ('f"F:/TanitAD-scratch/motion_c/cl_readout/windows_s{a.seed}.npz"', 'f"F:/TanitAD-scratch/motion_c/clr_read/readout/windows_s{a.seed}.npz"'),
]
MERGE = [
    ('RAW = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_readout/raw"', 'RAW = "E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_read/readout"'),
    ('json.load(open(f"{RAW}/CL_READOUT_s{s}.json"))', 'json.load(open(f"{RAW}/CLR_READOUT_s{s}.json"))'),
    ('open(f"{RAW}/CL_READOUT.json", "w", newline="\\n")', 'open(f"{RAW}/CLR_READOUT.json", "w", newline="\\n")'),
]


def build(name, edits):
    raw = open(f"{A1}/{name}", "rb").read()
    t = raw.decode("utf-8")
    for old, new in edits:
        if t.count(old) != 1:
            raise SystemExit(f"{name}: {old!r} matches {t.count(old)} times (must be 1)")
        t = t.replace(old, new)
    got = hashlib.sha256(raw).hexdigest()
    if got != PINS[name]:
        raise SystemExit(f"{name} is {got}, not the pinned {PINS[name]}")
    return got, t


if __name__ == "__main__":
    ok = True
    for name, edits, dst in (("cl_readout.py", READOUT, "clr_readout.py"), ("merge_readout.py", MERGE, "clr_merge_readout.py")):
        sha, t = build(name, edits)
        print(name, "source sha256", sha)
        if "--check" in sys.argv:
            same = open(f"{HERE}/{dst}", encoding="utf-8", newline="").read() == t
            print(dst, "== generated" if same else "DIFFERS from generated")
            ok &= same
        else:
            open(f"{HERE}/{dst}", "w", encoding="utf-8", newline="\n").write(t)
            print("wrote", dst)
    sys.exit(0 if ok else 1)
