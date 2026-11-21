#!/usr/bin/env python3
"""S1ZSM layer test (CPU, no data): s1zsm_layer.peel(scorer_finetune.py) is the TIP blob (0c19b8fd, the trainer WITH the S1Z layer) byte for byte, and every older pin that peels the layers keeps working.  Prints ZZTEST_PASS.

  L1 peel(trainer) has git blob sha1 TIP_TRAINER_BLOB, the layer marker is present in the trainer and absent after the peel, and the S1Z layer is still there (it is the next layer down)
  L2 a trainer without the layer passes through unchanged (peel(peel(x)) == peel(x))
  L3 the chain: s1z_layer.peel / scorer1_layer.peel of the trainer are the S1Z-less / pre-SCORER1 blobs (S1ZSM is peeled first), and hnscreen_layer / carvefeed_layer see the same trainer with and without S1ZSM
  L4 mutation: one changed character inside a reversal's text makes peel REFUSE (a drifted trainer cannot be silently accepted), for the first, the last and every reversal in between
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carvefeed_layer as CL  # noqa: E402
import hnscreen_layer as HL  # noqa: E402
import s1z_layer as ZL  # noqa: E402
import s1zsm_layer as ZSM  # noqa: E402
import scorer1_layer as SL  # noqa: E402


def blob(src: str) -> str:
    b = src.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


raw = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8")
src = raw.replace("\r\n", "\n")                                   # a Windows checkout may carry CRLF; the blobs are LF
assert ZSM.has_layer(src), "the S1ZSM marker is absent from scorer_finetune.py"
peeled = ZSM.peel(src)
assert not ZSM.has_layer(peeled) and blob(peeled) == ZSM.TIP_TRAINER_BLOB == "6a16d35c1818422ff514e97c4b292c32dde81795", (blob(peeled), ZSM.TIP_TRAINER_BLOB)
assert ZL.has_layer(peeled), "the S1Z layer must still be there after the S1ZSM peel (it is the next layer down)"
print("L1 ok")
assert ZSM.peel(peeled) == peeled
print("L2 ok")
nos1z = ZL.peel(src)
assert blob(nos1z) == ZL.TIP_TRAINER_BLOB and ZL.peel(peeled) == nos1z and not ZL.has_layer(nos1z) and not ZSM.has_layer(nos1z)
pre = SL.peel(src)
assert not SL.has_layer(pre) and not ZL.has_layer(pre) and not ZSM.has_layer(pre) and blob(pre) == SL.TIP_TRAINER_BLOB, blob(pre)
assert SL.peel(peeled) == pre
assert HL.peel(src) == HL.peel(peeled) and CL.peel(src) == CL.peel(peeled) and not ZSM.has_layer(HL.peel(src)) and not ZSM.has_layer(CL.peel(src))
print("L3 ok")
for k in range(len(ZSM.REVERSALS)):
    new_k = ZSM.REVERSALS[k][0]
    assert src.count(new_k) == 1, (k, src.count(new_k))
    bad = src.replace(new_k, new_k.replace("e", "E", 1) if "e" in new_k else new_k + "#", 1)
    try:
        ZSM.peel(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"a drifted trainer was accepted at reversal {k}")
print("L4 ok (%d reversals)" % len(ZSM.REVERSALS))
print("ZZTEST_PASS")
