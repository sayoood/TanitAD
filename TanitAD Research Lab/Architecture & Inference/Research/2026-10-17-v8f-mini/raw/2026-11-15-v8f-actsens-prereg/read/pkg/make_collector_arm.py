"""Builds collect_latents_arm.py from the code of record collect_latents_train_R35.py (R35 phase-1 collector, sha256 recorded in the output) by replacing ONLY its R35-specific lines (paths, tag, checkpoint) with
environment-driven ones; each replaced line must occur exactly once. Same clips (clips141.json, sha-checked inside), same 17 starts, same code path, so an arm's TRAIN latents are made exactly as R35's."""
import hashlib
SRC = "F:/TanitAD-scratch/hwm_plan/collect_latents_train_R35.py"
DST = "F:/TanitAD-scratch/actsens/read/collect_latents_arm.py"
raw = open(SRC, "rb").read()
t = raw.decode("utf-8")
REP = [
    ("import wmact_lib as W", 'import sys; sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wmact")\nimport wmact_lib as W'),
    ('HERE = "F:/TanitAD-scratch/hwm_plan"', 'HERE = os.environ["COL_OUT_DIR"]; TAG = os.environ["COL_TAG"]'),
    ('TS = os.path.join(HERE, "ts")', 'TS = os.environ["COL_TS"]'),
    ('LAD = {"R35_s0": "C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt", "R35_s1": "C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt"}', 'LAD = {TAG: os.environ["COL_CKPT"]}'),
    ('cfg = cfgs["R35_s0"]', "cfg = cfgs[TAG]"),
    ('part = os.path.join(HERE, "latents_train_R35_partial.pt")', 'part = os.path.join(HERE, f"latents_train_{TAG}_partial.pt")'),
]
for a, b in REP:
    assert t.count(a) == 1, ("not exactly once", a)
    t = t.replace(a, b)
open(DST, "w", encoding="utf-8", newline="\n").write(t)
print("wrote", DST, "source sha", hashlib.sha256(raw).hexdigest()[:8], "new sha", hashlib.sha256(t.encode()).hexdigest()[:8])
