#!/usr/bin/env python3
"""Build the navtest quarantine lists for the A2 behaviour-label gate (EVAL ONLY data must never become a label or a training row).

Reads NAVSIM's navtest scene logs (one pkl per log, a list of frames) and writes, next to this script's parent (gate_data/):
  navtest_logs.txt        sorted log names (the pkl stems; = nuPlan DB stems)
  navtest_tokens.txt.gz   sorted unique frame tokens AND scene tokens (hex, as nuPlan lidar_pc.token / navsim frame['token'])
  navtest_gate.json       counts + sha256 of both files + the source dir  (the gate verifies these on load)
usage: python build_navtest_gate.py [navsim_logs_test_dir]
"""
import glob, gzip, hashlib, json, os, pickle, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gate_data")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    logs, toks = [], set()
    for fp in sorted(glob.glob(os.path.join(SRC, "*.pkl"))):
        stem = os.path.basename(fp)[:-4]
        logs.append(stem)
        with open(fp, "rb") as fh:
            frames = pickle.load(fh)
        for f in frames:
            toks.add(str(f["token"]).lower())
            if f.get("scene_token"):
                toks.add(str(f["scene_token"]).lower())
            assert f.get("log_name", stem) == stem, (stem, f.get("log_name"))
    os.makedirs(OUT, exist_ok=True)
    lp, tp = os.path.join(OUT, "navtest_logs.txt"), os.path.join(OUT, "navtest_tokens.txt.gz")
    with open(lp, "w", newline="\n") as fh:
        fh.write("\n".join(sorted(logs)) + "\n")
    with gzip.GzipFile(tp, "wb", mtime=0) as fh:                      # mtime=0: the gzip bytes (and so the sha) are reproducible
        fh.write(("\n".join(sorted(toks)) + "\n").encode())
    meta = dict(source=SRC, n_logs=len(logs), n_tokens=len(toks), logs_sha256=sha256_file(lp), tokens_sha256=sha256_file(tp))
    with open(os.path.join(OUT, "navtest_gate.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
