"""S1c argv from the PATCHED chain (arm_args extracted verbatim) diffed against H1-s0's BANKED argv (PASS_d257647b7858.json).  usage: argv_diff.py <patched refe dir> <banked PASS json> <out.json>"""
import json
import shlex
import sys
import tempfile
from pathlib import Path

refe, banked, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(refe))
import test_s1aux_chain as TC  # noqa: E402

text = (refe / "s1aux_pod.sh").read_text(encoding="utf-8")
bank = json.loads(banked.read_text(encoding="utf-8"))["binding"]["argv"]
tmp = Path(tempfile.mkdtemp())
res = {"banked_H1-s0": {"token": banked.name, "commit": json.loads(banked.read_text(encoding="utf-8"))["binding"]["commit"], "argv": bank}}


def flags(argv):
    d, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                d[a] = argv[i + 1]
                i += 2
                continue
            d[a] = None
        i += 1
    return d


fb = flags(bank)
for arm in ("H1-s0", "H1u10k-s0", "H1u2k-s0"):
    rc, o, err = TC.arm_args_of(text, arm, tmp)
    assert rc == 0, (arm, err)
    av = shlex.split(o)
    fa = flags(av)
    res[arm] = {"argv": av, "identical_to_banked_H1-s0": av == bank,
                "differs_from_banked": {k: {"banked": fb.get(k, "<absent>"), "this": fa.get(k, "<absent>")} for k in sorted(set(fa) | set(fb)) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}}
    print(arm, "identical to the banked H1-s0 argv:", av == bank)
    for k, v in res[arm]["differs_from_banked"].items():
        print("   ", k, v)
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
