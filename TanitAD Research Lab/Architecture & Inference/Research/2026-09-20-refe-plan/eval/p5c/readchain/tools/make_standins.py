#!/usr/bin/env python3
"""DRY RUN only: build two stand-in "arms" from one checkpoint so that the seal's distinctness rules (C1 != C0, both != base) can pass while the
seams (read from P7's banked seam, identically for both arms) keep the identity control meaningful.
  make_standins.py SRC.pt OUT_C1.pt OUT_C0.pt     OUT_C1 = SRC + b"\\x01", OUT_C0 = SRC + b"\\x00"  (different sha256, same model for torch.load)
The stand-ins are scratch: the chain's owner deletes them (literal paths) when the dry run is finished.
"""
import os
import sys


def main():
    src, o1, o0 = sys.argv[1:4]
    for dst, tail in ((o1, b"\x01"), (o0, b"\x00")):
        if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src) + 1:
            print("exists", dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as f, open(dst, "wb") as g:
            for b in iter(lambda: f.read(1 << 22), b""):
                g.write(b)
            g.write(tail)
        print("wrote", dst, os.path.getsize(dst))


if __name__ == "__main__":
    main()
