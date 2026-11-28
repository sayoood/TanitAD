"""beside_obd.py -- the OTBAND read's two beside readers on the OBD arms, instrument files UNCHANGED (zero training, CPU, 2 torch threads).  No bar; both read what the card lists
beside the bars: `heads` = T's own decisions (T-LAT-F1, T-LON-F1, goal macro-F1 at tick 0 and ticks 1-2; TACTICAL family), `topk` = top-3 one-shot reads (best-of-3 minus top-1 per band,
GT manoeuvre pair in top-1 / top-3).

heads_read.py and topk_read.py hard-code OB-* names and the `otband_` bank folder.  Like theory/stopbal/read/stopdec_arm.py this driver executes the instrument's OWN source with
exact-once substitutions (asserted) and, for heads_read.py, replaces `partner` (a module-level function) after exec:
  topk_read.py  : TREE / T1 constants (to env OBD_TREE / OBD_DUMPS), the bank path `otband_{tag}` -> `obd_{tag}`, and the `pairs = [...]` list -> the OBD pairs.
  heads_read.py : NEW_DIR / TREE constants, `partner()` -> OBD pairing (ctl: BAND_sN -> CTL_sN; base: OBD-*_sN and *_rep -> PARITY_DAGOACC_sN / s0; rep: BAND_rep -> BAND_s0, NULL_rep -> NULL_s0,
                  CTL_sN -> NULL_sN for the drift row), and the two label strings.
Usage:  python beside_obd.py heads --tags OBD-NULL_s0 OBD-CTL_s0 OBD-BAND_s0 ... --out out/HEADS_OBD.json
        python beside_obd.py topk  --tags OBD-NULL_s0 OBD-CTL_s0 ... --out out/TOPK_OBD.json
        python beside_obd.py --selftest        (substitutions apply exactly once; heads_read's partner() maps the OBD tags; imports topk_read from the tree; no model pass)
topk loads checkpoints: it is a Master Mind step once the real banks exist (README_READ.md).  Tags for topk may include PARITY_DAGOACC_s0 / s1 (method control and the DAG-OACC row).
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(os.path.dirname(HERE))
OTB = os.path.join(THEORY, "otband", "read")
TREE = os.environ.get("OBD_TREE", "F:/TanitAD-scratch/otband_read/tree")
T1 = os.environ.get("OBD_DUMPS", "F:/TanitAD-scratch/otband_read/t1")
PULL = os.environ.get("OBD_PULLS", "C:/Users/Admin/t2v8f_pulls")

TOPK_SUBS = [
    ('TREE = "F:/TanitAD-scratch/otband_read/tree"', f"TREE = {TREE!r}"),
    ('T1 = "F:/TanitAD-scratch/otband_read/t1"', f"T1 = {T1!r}"),
    ('PULL = "C:/Users/Admin/t2v8f_pulls"', f"PULL = {PULL!r}"),
    ('f"{PULL}/otband_{tag}/ckpt_last.pt"', 'f"{PULL}/obd_{tag}/ckpt_last.pt"'),
    ('pairs = [("OB-BAND_s0", "OB-CTL_s0"), ("OB-BAND_s1", "OB-CTL_s1"), ("OB-BAND_rep", "OB-BAND_s0")]',
     'pairs = [("OBD-BAND_s0", "OBD-CTL_s0"), ("OBD-BAND_s1", "OBD-CTL_s1"), ("OBD-BAND_rep", "OBD-BAND_s0"), ("OBD-NULL_rep", "OBD-NULL_s0"), '
     '("OBD-CTL_s0", "PARITY_DAGOACC_s0"), ("OBD-CTL_s1", "PARITY_DAGOACC_s1"), ("OBD-NULL_s0", "PARITY_DAGOACC_s0")]'),
]
HEADS_SUBS = [
    ('TREE = "F:/TanitAD-scratch/otband_read/tree"', f"TREE = {TREE!r}"),
    ('NEW_DIR = "F:/TanitAD-scratch/otband_read/t1"', f"NEW_DIR = {T1!r}"),
    ('("ctl", "minus OB-CTL (same seed)")', '("ctl", "minus OBD-CTL (same seed)")'),
    ('("rep", "minus OB-BAND_s0")', '("rep", "minus the twin (rep -> its seed-0 arm; OBD-CTL -> OBD-NULL of the same seed)")'),
]


def partner_obd(tag, kind):
    """kind 'ctl': OBD-BAND_sN -> OBD-CTL_sN; 'base': OBD-*_sN -> PARITY_DAGOACC_sN, OBD-*_rep -> PARITY_DAGOACC_s0 (init M2b s0), PARITY_DAGOACC_sN -> DAGOACC_sN;
    'rep': OBD-BAND_rep -> OBD-BAND_s0, OBD-NULL_rep -> OBD-NULL_s0, OBD-CTL_sN -> OBD-NULL_sN (the detach row against the NULL twin; only used when that dump exists)."""
    s = tag.split("_")[-1] if "_" in tag else ""
    if kind == "ctl":
        return f"OBD-CTL_{s}" if tag.startswith("OBD-BAND_s") else None
    if kind == "base":
        if tag.startswith("OBD-"):
            return f"PARITY_DAGOACC_{s}" if s in ("s0", "s1") else ("PARITY_DAGOACC_s0" if s == "rep" else None)
        return f"DAGOACC_{s}" if tag.startswith("PARITY_DAGOACC_") else None
    if kind == "rep":
        if tag in ("OBD-BAND_rep", "OBD-NULL_rep"):
            return tag.replace("_rep", "_s0")
        return f"OBD-NULL_{s}" if tag.startswith("OBD-CTL_s") else None
    return None


def patched(src, subs):
    for old, new in subs:
        assert src.count(old) == 1, f"the instrument no longer carries exactly one of: {old}"
        src = src.replace(old, new)
    return src


def load_instrument(name, subs, post=None):
    path = os.path.join(OTB, name + ".py")
    src = patched(open(path, encoding="utf-8").read(), subs)
    sys.path.insert(0, OTB)
    m = types.ModuleType(name)
    m.__file__ = path
    m.__dict__["__name__"] = name
    sys.modules[name] = m
    exec(compile(src, path, "exec"), m.__dict__)
    if post:
        post(m)
    return m


def load_heads():
    def post(m):
        m.partner = partner_obd
    return load_instrument("heads_read", HEADS_SUBS, post)


def load_topk():
    # read_lib is the OTBAND read's (hard-codes the OTBAND export); topk_read imports it unchanged
    return load_instrument("topk_read", TOPK_SUBS)


def main():
    if "--selftest" in sys.argv:
        h = load_heads()
        assert h.partner("OBD-BAND_s1", "ctl") == "OBD-CTL_s1" and h.partner("OBD-CTL_s0", "ctl") is None
        assert h.partner("OBD-BAND_rep", "base") == "PARITY_DAGOACC_s0" and h.partner("OBD-CTL_s1", "base") == "PARITY_DAGOACC_s1"
        assert h.partner("PARITY_DAGOACC_s0", "base") == "DAGOACC_s0"
        assert h.partner("OBD-BAND_rep", "rep") == "OBD-BAND_s0" and h.partner("OBD-NULL_rep", "rep") == "OBD-NULL_s0" and h.partner("OBD-CTL_s1", "rep") == "OBD-NULL_s1"
        assert h.NEW_DIR == T1 and h.TREE == TREE
        t = load_topk()
        assert t.TREE == TREE and t.T1 == T1 and t.PULL == PULL
        assert t.tag_paths("OBD-CTL_s0")[0] == f"{PULL}/obd_OBD-CTL_s0/ckpt_last.pt", t.tag_paths("OBD-CTL_s0")
        print("SELFTEST ok: substitutions applied once; heads partner() maps the OBD tags; topk_read imports from", TREE)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    assert cmd in ("heads", "topk"), "usage: beside_obd.py heads|topk --tags ... --out ..."
    if cmd == "heads":
        h = load_heads()
        sys.exit(h.main(rest))
    t = load_topk()
    sys.argv = ["topk_read.py"] + rest
    t.main()


if __name__ == "__main__":
    main()
