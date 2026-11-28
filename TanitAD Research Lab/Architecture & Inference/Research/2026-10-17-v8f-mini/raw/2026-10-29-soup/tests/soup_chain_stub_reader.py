#!/usr/bin/env python3
"""STUB of soup_read.py for the chain's self-test: takes the real reader's flags, records the call (argv, the two environment variables the chain must clear, the manifest's slots) into $SOUP_WORLD/reader_call.json,
writes --out unless the hook says REFUSED, and exits with the code in $SOUP_WORLD/stub_reader_rc.txt (default 0).  It reads no dump and decides nothing."""
import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    for f in ("--manifest", "--out", "--prereg", "--prereg-sha", "--li-reader", "--li-reader-sha", "--orc-dir", "--banked-li-read", "--soup-build", "--soup-build-sha", "--spec", "--work-dir"):
        ap.add_argument(f, required=True)
    a = ap.parse_args(argv)
    w = os.environ["SOUP_WORLD"]
    man = json.load(open(a.manifest, encoding="utf-8"))
    rc = int(open(os.path.join(w, "stub_reader_rc.txt")).read().strip()) if os.path.isfile(os.path.join(w, "stub_reader_rc.txt")) else 0
    rec = {"argv": sys.argv[1:], "args": vars(a), "TANITEVAL_STACK_OVERRIDE": os.environ.get("TANITEVAL_STACK_OVERRIDE"), "PYTHONPATH": os.environ.get("PYTHONPATH"), "omp": os.environ.get("OMP_NUM_THREADS"), "mkl": os.environ.get("MKL_NUM_THREADS"), "TMP": os.environ.get("TMP"),
           "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "slots": {r: sorted(v) for r, v in man["arms"].items()}, "prereg_exists": os.path.isfile(a.prereg), "rc": rc}
    with open(os.path.join(w, "reader_call.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    tam = os.path.join(w, "stub_reader_tamper.txt")
    if os.path.isfile(tam):                                                                                  # append a byte to each listed path (the PREREG, to prove the chain checks it after the reader)
        for ln in open(tam, encoding="utf-8").read().split("\n"):
            if ln.strip():
                with open(ln.strip(), "ab") as f:
                    f.write(b"\0")
    if rc in (0, 3):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"STUB_READER": True}, f)
        print('wrote ' + a.out + ' {"OUTCOME": "STUB"}')
    else:
        print("REFUSED: stub")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
