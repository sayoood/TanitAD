"""Cross-tree identity of the T1-kin tool (pod-side): does the STRAT1 v8f_t1_eval.py, run with the argv the SPEEDSUB rig uses (no STRAT1 flag),
write the same bytes as the cdbff971 tool? Needed before v8f_t1_speedsub.py's TOOL_SHA256 pin moves from 61e140ed... to the STRAT1 blob.

usage: t1_identity.py build <tree> <work>      (base tree: world + tiny checkpoint + argv.json; the speedsub test's own Rig helpers)
       t1_identity.py run   <tree> <work> <label>   (runs <tree>'s v8f_t1_eval.main on argv.json into <work>/run, then renames it <work>/<label>)
       t1_identity.py cmp   <work>                  (compares <work>/base and <work>/new: every npz array bitwise, the result JSON as text)
The runs use ONE out path (<work>/run) so the JSON cannot differ by its own directory name."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def setup(tree):
    for p in (f"{tree}/stack", f"{tree}/taniteval", f"{tree}/stack/scripts", f"{tree}/stack/tests"):
        sys.path.insert(0, p)
    import tanitad
    assert os.path.realpath(tanitad.__file__).startswith(os.path.realpath(f"{tree}/stack")), tanitad.__file__


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        tree, work = sys.argv[2], Path(sys.argv[3])
        setup(tree)
        import test_v8f_t1_speedsub as TS
        work.mkdir(parents=True, exist_ok=False)
        w = TS.D.make_synthetic_world(work / "world", n_train=2, n_eval=3, T=90, seed=5)
        rig = TS.Rig({"root": work / "world", **w}, work)
        argv = rig.tool_argv("run")
        (work / "argv.json").write_text(json.dumps(argv))
        print("built", rig.ckpt_sha[:12], len(argv))
    elif cmd == "run":
        tree, work, label = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
        setup(tree)
        import torch
        torch.set_num_threads(1)
        import v8f_t1_eval as T1
        print("tool", T1.__file__, sha(T1.__file__)[:12])
        argv = json.loads((work / "argv.json").read_text())
        rc = T1.main(argv)
        assert rc == 0, rc
        os.rename(work / "run", work / label)
        print("ran", label)
    elif cmd == "cmp":
        work = Path(sys.argv[2])
        a, b = work / "base", work / "new"
        fa = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file())
        fb = sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file())
        res = {"files_base": fa, "files_new": fb, "same_file_set": fa == fb, "diff": []}
        n_arr = 0
        for f in sorted(set(fa) & set(fb)):
            pa, pb = a / f, b / f
            if f.endswith(".npz"):
                za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
                if sorted(za.files) != sorted(zb.files):
                    res["diff"].append(f"{f}: keys {sorted(set(za.files) ^ set(zb.files))}")
                for k in sorted(set(za.files) & set(zb.files)):
                    x, y = za[k], zb[k]
                    n_arr += 1
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        res["diff"].append(f"{f}:{k}")
            elif pa.read_bytes() != pb.read_bytes():
                res["diff"].append(f)
        res["n_arrays_compared"] = n_arr
        res["identical"] = res["same_file_set"] and not res["diff"]
        (work / "t1_identity_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k not in ("files_base", "files_new")}))
        print("ZZ T1ID", "IDENTICAL" if res["identical"] else "DIFFERS", "ZZ")


if __name__ == "__main__":
    main()
