#!/usr/bin/env python
"""make_names_n2.py -- writes qhead_names_n2.py from qhead_names_n1.py by asserted text edits (each old text must occur exactly once).
Note PREREG_V8F_PROOF_ADD42_N2 section 2: control h5 gates the names PER ARM TYPE, as note N1 section 4 says ("per checkpoint; outside it the arm of that checkpoint is
not named"; the names need both seeds of a type). A type with a checkpoint outside the band gets no name and no outcome number; the other type is read.
h1, h4, h6, h7, h8 gate every name as before; h2 and h3 are read on the arms of the types that are read.

  python make_names_n2.py            (writes qhead_names_n2.py next to this file and prints both sha256)
"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, DST = os.path.join(HERE, "qhead_names_n1.py"), os.path.join(HERE, "qhead_names_n2.py")

EDITS = [
    # 0: the head line of the module text
    ('"""qhead_names_n1.py -- ',
     '"""qhead_names_n2.py -- note PREREG_V8F_PROOF_ADD42_N2 section 2: control h5 gates PER ARM TYPE (a type with a checkpoint outside the band gets no name and no outcome\n'
     'number; the other type is read); h1, h4, h6, h7, h8 gate every name; h2 and h3 are read on the arms of the read types. Made from qhead_names_n1.py by make_names_n2.py.\n'
     'The text of the first form follows.\n\nqhead_names_n1.py -- '),
    # 1: which types are read is fixed from the fit records BEFORE the per-arm controls
    ('    h2 = {}\n    arms = ["%s_s%d" % (TYPES[ty]["qh"], s) for ty in TYPES for s in SEEDS] + ["LSQHI_s0"]\n',
     '    cov = {}\n'
     '    for ty in TYPES:\n'
     '        for s in SEEDS:\n'
     '            base = "%s_s%d" % (TYPES[ty]["base"], s)\n'
     '            f0 = jl(os.path.join(a.fit_dir, base + ".qhead.json")) if a.fit_dir else None\n'
     '            c0 = ((f0 or {}).get("coverage_check") or {}).get("share_q_above_label")\n'
     '            cov[base] = bool(c0 is not None and H5_LO <= c0 <= H5_HI)\n'
     '    not_named = {ty: [b for b in ("%s_s%d" % (TYPES[ty]["base"], s) for s in SEEDS) if not cov[b]] for ty in TYPES}\n'
     '    not_named = {ty: v for ty, v in not_named.items() if v}\n'
     '    read_types = [ty for ty in TYPES if ty not in not_named]\n'
     '    h2 = {}\n'
     '    arms = ["%s_s%d" % (TYPES[ty]["qh"], s) for ty in read_types for s in SEEDS] + ["LSQHI_s0"]\n'),
    # 2: h3 on the arms of the read types
    ('    h3 = {}\n    for ty in TYPES:\n', '    h3 = {}\n    for ty in read_types:\n'),
    # 3: the h5 control: per arm type
    ('    C["h5_check_coverage_in_0.15_0.25"] = {"PASS": bool(all(v["PASS"] for v in h5.values())), "per_base": h5}\n',
     '    C["h5_check_coverage_in_0.15_0.25"] = {"PASS": bool(read_types), "rule": "per arm type: a type is read only if both of its checkpoints are inside the band (note N1 section 4, note N2 section 2)",\n'
     '                                           "per_base": h5, "types_read": read_types, "types_not_named": not_named}\n'),
    # 4: names only for the read types
    ('def names(hw, q3):\n    N, B = {}, {}\n    for ty, m in TYPES.items():\n        n = {}\n',
     'def names(hw, q3, types=None):\n    N, B = {}, {}\n    for ty, m in TYPES.items():\n        if types is not None and ty not in types:\n            continue\n        n = {}\n'),
    # 5: main
    ('        N, B = names(hw, q3)\n        res["names"], res["beside"] = N, B\n',
     '        h5c = C["h5_check_coverage_in_0.15_0.25"]\n'
     '        N, B = names(hw, q3, h5c["types_read"])\n'
     '        for ty, bases in h5c["types_not_named"].items():\n'
     '            N[ty] = {"all": "NOT NAMED (h5: the coverage on the check clips is outside the band at %s; note N1 section 4, note N2 section 2); no outcome number of this type is written" % ", ".join(bases)}\n'
     '        res["names"], res["beside"] = N, B\n'),
    # 6: the record says which rule read it
    ('res = {"card": "PREREG_V8F_PROOF_ADD42_N1", "tier"',
     'res = {"card": "PREREG_V8F_PROOF_ADD42_N1", "reading_rule": "PREREG_V8F_PROOF_ADD42_N2 section 2 (h5 gates per arm type)", "tier"'),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build(src_text, edits=EDITS):
    t = src_text
    for i, (old, new) in enumerate(edits):
        assert t.count(old) == 1, "edit %d: the old text occurs %d times" % (i, t.count(old))
        t = t.replace(old, new)
    return t


def main():
    t = io.open(SRC, encoding="utf-8", newline="").read()
    assert "\r" not in t, "the source has CR line ends"
    out = build(t)
    with io.open(DST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out)
    print("src %s %s" % (sha(SRC)[:16], os.path.basename(SRC)))
    print("dst %s %s" % (sha(DST)[:16], os.path.basename(DST)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
