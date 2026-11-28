"""sync_refs.py -- copy the stage-A T0 mechanism evals of the six reference arms (B_s0, U_s0, B_rep, R_s0 from block 1; UE_s0, BL_s0 from block 3; all already collected in read3/out/t0mech) into this read's out/t0mech, so that the ADE6 / latcons pairs read ONE folder (PGT.load_all takes one eval dir).
Every copied file is sha256-verified against its source; an existing target with different bytes REFUSES (exit 1, nothing overwritten).  The source is read only.  The `_proj.npz` files are not needed and not copied.
Usage: python sync_b1.py [--src DIR] [--dst DIR]"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402

NAMES = tuple("DS" + a.replace("_", "") + "A" for a in ("B_s0", "U_s0", "B_rep", "R_s0", "UE_s0", "BL_s0"))
SUFFIX = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")


def plan(src):
    return [f"eval_{n}_s0{s}.{ext}" for n in NAMES for s in SUFFIX for ext in ("json", "npz")]


def sync(src, dst):
    """-> (n_copied, n_present_equal, [problems])"""
    os.makedirs(dst, exist_ok=True)
    copied = same = 0
    bad = []
    for f in plan(src):
        s, d = f"{src}/{f}", f"{dst}/{f}"
        if not os.path.isfile(s):
            bad.append(f"source missing: {s}")
            continue
        if os.path.isfile(d):
            if L.sha256_file(d) == L.sha256_file(s):
                same += 1
            else:
                bad.append(f"REFUSED: {d} exists with different bytes than {s}")
            continue
        shutil.copyfile(s, d)
        if L.sha256_file(d) != L.sha256_file(s):
            os.remove(d)
            bad.append(f"copy of {f} failed its sha256 check (removed)")
        else:
            copied += 1
    return copied, same, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=L.B3_ROOT + "/out/t0mech")
    ap.add_argument("--dst", default=L.OUT + "/t0mech")
    a = ap.parse_args(argv)
    c, s, bad = sync(a.src, a.dst)
    print(f"sync_refs: copied {c}, already present and equal {s}, problems {len(bad)}")
    for b in bad:
        print(" ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
