"""Compare the xtree_dump outputs: TRAINING must be bit-identical across trees; the fix's eval must repeat under the same
seed; the old eval must NOT (RED); a mispaired control (train0 vs train1) must differ."""
import sys
from pathlib import Path

import torch

O = Path(sys.argv[1])


def differ(a, b):
    return sorted(k for k in set(a) | set(b)
                  if k not in a or k not in b or a[k].shape != b[k].shape or not torch.equal(a[k], b[k]))


def ph(r, tag):
    return next(p for p in r["phases"] if p["tag"] == tag)


bad = 0
for mode in ("off", "on", "sched"):
    old = torch.load(O / f"pgtree_{mode}.pt", weights_only=False)
    new = torch.load(O / f"detfix_{mode}.pt", weights_only=False)
    for tag in ("train0", "train1", "train2"):
        a, b = ph(old, tag), ph(new, tag)
        d_out, d_gr = differ(a["out"], b["out"]), differ(a["grads"], b["grads"])
        eq = (not d_out and not d_gr and torch.equal(a["loss"], b["loss"]) and torch.equal(a["gen"], b["gen"])
              and torch.equal(a["global"], b["global"]) and a["emit_live"] == b["emit_live"])
        bad += not eq
        print(f"{mode:5s} {tag}: outputs {len(d_out)}/{len(a['out'])} grads {len(d_gr)}/{len(a['grads'])} differ, "
              f"loss eq {torch.equal(a['loss'], b['loss'])}, gen eq {torch.equal(a['gen'], b['gen'])}, "
              f"emit {a['emit_live']}/{b['emit_live']} -> {'IDENTICAL' if eq else 'DIFFERENT'}")
    for name, r in (("old", old), ("new", new)):
        d = differ(ph(r, "eval0_s1234")["out"], ph(r, "eval2_s1234")["out"])
        s = differ(ph(r, "eval0_s1234")["out"], ph(r, "eval1_s777")["out"])
        print(f"{mode:5s} {name} eval seed 1234 twice: {len(d)} differ | seed 1234 vs 777: {len(s)} differ")
    # mispaired control: the comparator must SEE a difference between two different training steps
    mc = differ(ph(new, "train0")["grads"], ph(new, "train1")["grads"])
    print(f"{mode:5s} CONTROL train0 vs train1 grads (new): {len(mc)} differ (must be > 0)")
    bad += len(mc) == 0
print("ZZXTREE", "PASS" if bad == 0 else f"FAIL {bad}")
