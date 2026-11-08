import sys,hashlib,os
sys.path.insert(0,os.environ["REFE"])
import test_rl_s1b as T
src=open(os.path.join(os.environ["REFE"],"scorer_finetune.py"),encoding="utf-8",newline="").read()
seg=T.segments(src)["main.train_losses"]
BLK=("        if pair_w is not None or a.pair_w != 0.0:           # SCORER1 (arm B ONLY; arm A stays the BCE-only control).  Skipped entirely when OFF: the OFF path is byte-identical.\n            lB = lB + pair_term(sxB, tg, pair_w)\n")
# segment is dedented: strip 4 spaces
import textwrap
blk=textwrap.dedent(BLK) if False else "\n".join(l[4:] for l in BLK.split("\n"))
print("block count",seg.count(blk), "sig", seg.count(", pair_w=None):"))
s=seg.replace(blk,"").replace(", pair_w=None):","):")
h=hashlib.sha256(T.undo_r0c("main.train_losses",T.undo_safeobj("main.train_losses",s)).encode()).hexdigest()
print(h==T.R0_SEGMENTS["main.train_losses"],h)
