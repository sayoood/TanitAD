#!/usr/bin/env python3
"""STUB of the T1-kin rollout tool (v8f_t1_eval.py) and the T0 eval tool (v8f_mini_eval.py) for the chain's self-test (test_soup_chain.py).  It parses the REAL flag set (argparse, as the real tools do: the chain's dry run
parses every command with it), writes the files a real run leaves (tiny), and obeys hooks placed in the throw-away world by the test.  It computes nothing and loads no model.

Hooks (files in $SOUP_WORLD):  stub_fail_<tag>.txt  exit 3 after writing a partial output | stub_segv_<tag>.txt  exit 139 once (the file is consumed) | stub_stop_<tag>.txt  touch the path inside
stub_tamper_<tag>.txt  append a byte to each path listed (one per line) | stub_calls.log  every call appends its tag."""
import argparse
import json
import os
import sys

FLAGS = ("--ckpt", "--tag", "--tracks", "--eval-cache", "--eval-grid", "--eval-join", "--out-dir", "--replan-steps", "--nav", "--scene", "--schedule", "--limit", "--dump-ticks", "--device", "--v9-eval", "--lead-eval",
         "--speed-eval", "--kin-mode", "--p10", "--eval-kin", "--ages")
BOOLS = ("--dump-heads", "--project", "--families")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for f in FLAGS:
        p.add_argument(f)
    for f in BOOLS:
        p.add_argument(f, action="store_true")
    a = ap.parse_args(argv)
    run(a, os.path.basename(__file__))


def hook(name):
    w = os.environ.get("SOUP_WORLD", "")
    fp = os.path.join(w, name)
    return fp if w and os.path.isfile(fp) else None


def run(a, me):
    tag = a.tag
    with open(os.path.join(os.environ["SOUP_WORLD"], "stub_calls.log"), "a", encoding="utf-8") as f:
        f.write(tag + "\n")
    t1 = a.limit is not None
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    partial = hook(f"stub_fail_{tag}.txt") or hook(f"stub_segv_{tag}.txt")
    if t1:
        with open(f"{out}/t1v8f_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"ckpt": a.ckpt, "limit": a.limit, "tag": tag}, f)
        if not partial:
            os.makedirs(os.path.dirname(a.dump_ticks + "/x"), exist_ok=True)
            with open(f"{a.dump_ticks}/ticks_{tag}.npz", "wb") as f:
                f.write(b"ticks")
            os.makedirs(f"{out}/dump_{tag}", exist_ok=True)
            for i in range(141):
                with open(f"{out}/dump_{tag}/ep{i:03d}.npz", "wb") as f:
                    f.write(b"d")
    else:
        with open(f"{out}/eval_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"ckpt": a.ckpt, "tag": tag, "ages": a.ages}, f)
        if not partial:
            with open(f"{out}/eval_{tag}.npz", "wb") as f:
                f.write(b"npz")
            if a.project:
                with open(f"{out}/eval_{tag}_proj.npz", "wb") as f:
                    f.write(b"proj")
    h = hook(f"stub_tamper_{tag}.txt")
    if h:
        for ln in open(h, encoding="utf-8").read().split("\n"):
            if ln.strip():
                with open(ln.strip(), "ab") as f:
                    f.write(b"\0")
    h = hook(f"stub_stop_{tag}.txt")
    if h:
        open(open(h, encoding="utf-8").read().strip(), "w").close()
    if hook(f"stub_segv_{tag}.txt"):
        os.remove(hook(f"stub_segv_{tag}.txt"))
        sys.exit(139)
    if hook(f"stub_fail_{tag}.txt"):
        print(f"STUB: the step {tag} fails by hook", file=sys.stderr)
        sys.exit(3)
    print(f"STUB ok {tag}")


if __name__ == "__main__":
    main(sys.argv[1:])
