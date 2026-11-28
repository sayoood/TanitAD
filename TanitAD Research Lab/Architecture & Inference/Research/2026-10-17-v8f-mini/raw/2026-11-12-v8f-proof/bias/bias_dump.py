"""bias_dump.py -- T1-kin dumps of the tags THIS job needs (the six edited checkpoints + NULL_s1, G_s1, G2_s1), through wsp_dump.run (same argv, same tree, ONE thread count: 2).
Usage: python -I bias_dump.py [TAG ...] [--out DIR] [--max-windows N]     default tags: MY_DUMP_TAGS in priority order, default --out bias/t1; done-file tags are skipped.
--max-windows N = self-test (no done file).  NULL_s0 / NULL_rep / G_s0 / G2_s0 are NOT dumped here: they come from wsp/t1 (the WSP job)."""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bias_common as B  # noqa: E402
import wsp_common as C  # noqa: E402
import wsp_dump as D  # noqa: E402

C.DUMP_TAGS.update(B.DUMP_TAGS)            # wsp_dump.make_argv reads this dict (same module object)
ORDER = ["GE2_s0", "GE1_s0", "GE0_s0", "GE2_s1", "GE1_s1", "GE0_s1", "NULL_s1", "G_s1", "G2_s1"]
assert sorted(ORDER) == sorted(B.MY_DUMP_TAGS)


def main():
    a = sys.argv[1:]
    out, mw = B.T1, 0
    if "--out" in a:
        i = a.index("--out")
        out = a[i + 1]
        del a[i:i + 2]
    if "--max-windows" in a:
        i = a.index("--max-windows")
        mw = int(a[i + 1])
        del a[i:i + 2]
    tags = a or ORDER
    assert all(t in B.MY_DUMP_TAGS for t in tags), tags
    for k, v in D.INPUTS.items():
        assert os.path.exists(v), (k, v)
    for t in tags:
        r = D.run(t, out, mw)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        if not (r["status"] == "ok" or r["status"].startswith("skip") or r["status"].startswith("SELFTEST")):
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
