"""stopdec_arm.py -- theory/stopdec/stopdec_read.py on STOPBAL arms (CPU, zero training; the instrument is unchanged, only the tree and the dump folder move).

stopdec_read.py -> speedsrc_read.py -> topk_read.py hard-code the OTBAND export (TREE) and its dump folder (T1), and read_lib.py hard-codes the same tree.  The
STOPBAL checkpoints carry config fields the OTBAND tree's V8FMiniConfig does not know, so the forward pass MUST run from the STOPBAL code-of-record export.
Without editing any of those files, this driver registers (a) read_lib_stopbal as `read_lib` and (b) topk_read.py's own source with exactly two constant lines
replaced (TREE, T1) as `topk_read`, then runs stopdec_read.main().  Both substitutions are asserted to apply exactly once.

Usage:  python stopdec_arm.py --tags BAL-NULL_s0 BAL-G_s0 ... --outdir <out/stopdec>        (env STOPBAL_TREE, STOPBAL_DUMPS as in run_dumps_stopbal.py)
        python stopdec_arm.py --selftest                                                   (substitution + import check; no model pass)
Outputs per run: <outdir>/stopdec_<tag>.npz and STOPDEC.json (rename to STOPDEC_<tag>.json per tag: stopbal_read.py reads that name; --split-json does it).
The argv file the instrument parses is <T1>/argv_<tag>_m10_causal_heads.txt, written by run_dumps_stopbal.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(os.path.dirname(HERE))
OTB = os.path.join(THEORY, "otband", "read")
TREE = os.environ.get("STOPBAL_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
T1 = os.environ.get("STOPBAL_DUMPS", "F:/TanitAD-scratch/stopbal_read/t1")
OLD_TREE, OLD_T1 = 'TREE = "F:/TanitAD-scratch/otband_read/tree"', 'T1 = "F:/TanitAD-scratch/otband_read/t1"'


def patched_topk_source(src, tree, t1):
    assert src.count(OLD_TREE) == 1 and src.count(OLD_T1) == 1, "topk_read.py no longer carries its two path constants exactly once"
    return src.replace(OLD_TREE, f'TREE = {tree!r}').replace(OLD_T1, f'T1 = {t1!r}')


def install():
    """Register read_lib (STOPBAL geometry module) and the patched topk_read in sys.modules; the instrument then imports them unchanged."""
    os.environ["STOPBAL_TREE"] = TREE
    sys.path.insert(0, HERE)
    import read_lib_stopbal
    sys.modules["read_lib"] = read_lib_stopbal
    path = os.path.join(OTB, "topk_read.py")
    src = patched_topk_source(open(path, encoding="utf-8").read(), TREE, T1)
    import types
    m = types.ModuleType("topk_read")
    m.__file__ = path
    sys.modules["topk_read"] = m
    exec(compile(src, path, "exec"), m.__dict__)
    assert m.TREE == TREE and m.T1 == T1
    return m


def attach_topk():
    """READ_NOTE_COMMIT.md section 2: the eval-only forward_topk / topk_codes of 8e9499e0 onto the STOPBAL V8FMini (topk_shim.py; no tree file edited)."""
    import topk_shim
    import tanitad.models.v8f_mini as vm
    assert os.path.abspath(vm.__file__).replace("\\", "/").startswith(TREE), vm.__file__
    rec = topk_shim.attach(vm)
    print("TOPK-SHIM", json.dumps(rec))
    return rec


def main():
    tk = install()
    shim = attach_topk()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    if "--selftest" in sys.argv:
        import speedsrc_read as SS
        assert SS.TK is tk and SS.TK.TREE == TREE and SS.TK.T1 == T1
        import tanitad
        assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
        import tanitad.models.v8f_mini as vm
        assert hasattr(vm.V8FMini, "forward_topk") and shim.get("attached") is not None
        print("SELFTEST ok: tree", TREE, "t1", T1, "tanitad", tanitad.__file__)
        return
    split = "--split-json" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--split-json"]
    import stopdec_read
    sys.argv = ["stopdec_read.py"] + argv
    stopdec_read.main()
    if split:
        outdir = argv[argv.index("--outdir") + 1]
        j = json.load(open(os.path.join(outdir, "STOPDEC.json"), encoding="utf-8"))
        for tag in j["arms"]:
            one = dict(j, arms={tag: j["arms"][tag]}, controls={tag: j["controls"][tag]})
            json.dump(one, open(os.path.join(outdir, f"STOPDEC_{tag}.json"), "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()
