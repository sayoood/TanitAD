#!/usr/bin/env python3
"""S1Z layer test (CPU, no data): s1z_layer.peel(scorer_finetune.py) is the TIP blob (62df1e64) byte for byte, and every older pin that peels the layers keeps working.  Prints ZZTEST_PASS.

  L1 peel(trainer) has git blob sha1 TIP_TRAINER_BLOB, and the layer marker is present in the trainer and absent after the peel
  L2 a trainer without the layer passes through unchanged (peel(peel(x)) == peel(x))
  L3 the chain: scorer1_layer.peel(trainer) is still the pre-SCORER1 blob (the S1Z layer is peeled first), and hnscreen_layer / carvefeed_layer see the same trainer with and without S1Z
  L4 mutation: one changed character inside a reversal's text makes peel REFUSE (a drifted trainer cannot be silently accepted)
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carvefeed_layer as CL  # noqa: E402
import hnscreen_layer as HL  # noqa: E402
import s1z_layer as ZL  # noqa: E402
import scorer1_layer as SL  # noqa: E402


def blob(src: str) -> str:
    b = src.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


raw = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8")
src = raw.replace("\r\n", "\n")                                   # a Windows checkout may carry CRLF; the blobs are LF
assert ZL.has_layer(src), "the S1Z marker is absent from scorer_finetune.py"
peeled = ZL.peel(src)
assert not ZL.has_layer(peeled) and blob(peeled) == ZL.TIP_TRAINER_BLOB, (blob(peeled), ZL.TIP_TRAINER_BLOB)
assert SL.has_layer(peeled), "the SCORER1 layer must still be there after the S1Z peel (it is the next layer down)"
print("L1 ok")
assert ZL.peel(peeled) == peeled
print("L2 ok")
pre = SL.peel(src)
assert not SL.has_layer(pre) and not ZL.has_layer(pre) and blob(pre) == SL.TIP_TRAINER_BLOB, blob(pre)
assert SL.peel(peeled) == pre
assert HL.peel(src) == HL.peel(peeled) and CL.peel(src) == CL.peel(peeled) and not ZL.has_layer(HL.peel(src)) and not ZL.has_layer(CL.peel(src))
print("L3 ok")
for k in (0, len(ZL.REVERSALS) - 1):
    new_k = ZL.REVERSALS[k][0]
    bad = src.replace(new_k, new_k.replace("e", "E", 1), 1)
    try:
        ZL.peel(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"a drifted trainer was accepted at reversal {k}")
print("L4 ok")
print("ZZTEST_PASS")
