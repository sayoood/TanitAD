"""Is the OLD dumper (onpolicy_dump.py, the one that produced the main run's sets) batch-variant in the same way as rl_fan_dump.py?
Runs it at --batch 1 and at --batch 4 on the identity instrument's 24-key invariance subset (same GPU, same bank), then diffs the two.
Usage: python old_dumper_batch_probe.py <work dir> <python with torch+cuda> <out json>"""
import json
import os
import sys

W, PY, OUT = sys.argv[1].replace("\\", "/"), sys.argv[2], sys.argv[3]
EVAL = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
sys.path.insert(0, EVAL)
import rl_fan_identity as I  # noqa: E402

idd = os.path.join(W, "identity")
sub = []
for ln in open(os.path.join(idd, "keys_sub.tsv"), encoding="utf-8").read().splitlines()[1:]:
    t, l = ln.split("\t")
    sub.append((l, t))
assert len(sub) == 24, len(sub)
kset = set(sub)
probe = os.path.join(W, "old_batch_probe")
bank = os.path.join(probe, "bank")
os.makedirs(bank, exist_ok=True)
n = 0
with open(os.path.join(bank, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as g:
    for ln in open(os.path.join(idd, "bank", "targets_rank0.jsonl"), encoding="utf-8"):
        r = json.loads(ln)
        if (r["log_name"], r["token"]) in kset:
            g.write(ln if ln.endswith("\n") else ln + "\n")
            n += 1
assert n == 24, n
res = {"tool": "old_dumper_batch_probe/1", "keys": 24}
props = {}
for b in ("1", "4"):
    q = os.path.join(probe, f"queue_b{b}")
    os.makedirs(q, exist_ok=True)
    rc, sec = I.run([PY, os.path.join(I.PKG, "refe", "onpolicy_dump.py"), "--run-dir", I.FINAL, "--targets", bank, "--images", I.PIXELS,
                     "--calib", os.path.join(W, "bank_full", "calib_table.json"), "--queue", q, "--ranks", "0", "--max-pending", "1000000",
                     "--chunk", "32", "--batch", b], os.path.join(probe, f"old_b{b}.log"))
    props[b] = I.old_props(q)
    res[f"old_b{b}"] = {"rc": rc, "wall_sec": sec, "sets": len(props[b])}
res["old_b1_vs_old_b4"] = I.diff(props["1"], props["4"], "onpolicy_dump batch 1 vs batch 4, same GPU, same 24 keys")
# and the new dumper's batch-1 / batch-4 subset outputs against the old dumper's at the same batch size
for b in ("1", "4"):
    p = os.path.join(idd, f"new_b{b}", f"onpolicy_rlfan_b{b}_s0of1.jsonl")
    if os.path.exists(p):
        res[f"new_b{b}_vs_old_b{b}"] = I.diff(I.new_records(p), props[b], f"rl_fan_dump vs onpolicy_dump, both at batch {b}")
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
for k, v in res.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "first_unequal"} if isinstance(v, dict) else v)
