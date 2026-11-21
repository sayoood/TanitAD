#!/usr/bin/env python3
"""SCORER1 layer test (CPU, no data): scorer1_layer.peel(scorer_finetune.py) is the TIP blob byte for byte, so every older pin that peels the layers keeps working.  Prints ZZTEST_PASS.

  P1 peel(trainer) has git blob sha1 TIP_TRAINER_BLOB (51a28163...), and the layer marker is present in the trainer and absent after the peel
  P2 a trainer without the layer passes through unchanged (peel(peel(x)) == peel(x))
  P3 the chain: hnscreen_layer.peel and carvefeed_layer.peel see the pre-SCORER1 trainer (the HN and B8 pins are unchanged by this layer)
  P4 mutation: one changed character inside a reversal's text makes peel REFUSE (a drifted trainer cannot be silently accepted)
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carvefeed_layer as CL  # noqa: E402
import hnscreen_layer as HL  # noqa: E402
import scorer1_layer as SL  # noqa: E402


def blob(src: str) -> str:
    b = src.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


raw = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8")
src = raw.replace("\r\n", "\n")                                   # a Windows checkout may carry CRLF; the blobs are LF
assert SL.has_layer(src), "the SCORER1 marker is absent from scorer_finetune.py"
peeled = SL.peel(src)
assert not SL.has_layer(peeled) and blob(peeled) == SL.TIP_TRAINER_BLOB, (blob(peeled), SL.TIP_TRAINER_BLOB)
print("P1 ok")
assert SL.peel(peeled) == peeled
print("P2 ok")
assert blob(CL.peel(src)) == blob(CL.peel(peeled)) and HL.peel(src) == HL.peel(peeled) and not SL.has_layer(HL.peel(src)) and not SL.has_layer(CL.peel(src))
print("P3 ok")
new0 = SL.SCORER1_FILE_REVERSALS[0][0]
for k in (0, len(SL.SCORER1_FILE_REVERSALS) - 1):
    bad = src.replace(SL.SCORER1_FILE_REVERSALS[k][0], SL.SCORER1_FILE_REVERSALS[k][0].replace("e", "E", 1), 1)
    try:
        SL.peel(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"a drifted trainer was accepted at reversal {k}")
print("P4 ok")
print("ZZTEST_PASS")
