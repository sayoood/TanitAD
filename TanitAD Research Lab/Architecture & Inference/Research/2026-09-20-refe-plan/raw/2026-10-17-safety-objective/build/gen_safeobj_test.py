#!/usr/bin/env python3
"""gen_safeobj_test: fill test_rl_safeobj.template.py -> <PK>/refe/test_rl_safeobj.py.
SAFEOBJ_FILE_REVERSALS = patch_trainer.TRAINER_EDITS reversed ((new, old) pairs, last edit first); @CHAIN_SUITE@ = chain_suite.txt.
usage: gen_safeobj_test.py <PK dir>"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patchlib as PL  # noqa: E402
from patch_trainer import TRAINER_EDITS  # noqa: E402


def main():
    pk = PL.pk_arg()
    s = PL.read(os.path.join(HERE, "test_rl_safeobj.template.py"))
    pairs = [(new, old) for (old, new) in TRAINER_EDITS][::-1]
    lit = "SAFEOBJ_FILE_REVERSALS = (\n" + "".join(f"    ({new!r},\n     {old!r}),\n" for new, old in pairs) + ")\n"
    pat = re.compile(r"# <<<GEN SAFEOBJ_FILE_REVERSALS\n.*?# >>>GEN SAFEOBJ_FILE_REVERSALS\n", re.S)
    assert len(pat.findall(s)) == 1
    s = pat.sub(lambda m: "# <<<GEN SAFEOBJ_FILE_REVERSALS\n" + lit + "# >>>GEN SAFEOBJ_FILE_REVERSALS\n", s)
    assert s.count("@CHAIN_SUITE@") == 1
    s = s.replace("@CHAIN_SUITE@", PL.read(os.path.join(HERE, "chain_suite.txt")).rstrip("\n"))
    out = os.path.join(pk, "refe", "test_rl_safeobj.py")
    PL.write(out, s)
    print({"wrote": out, "blob": PL.git_blob(s), "reversal_pairs": len(pairs), "bytes": len(s.encode())})


if __name__ == "__main__":
    main()
