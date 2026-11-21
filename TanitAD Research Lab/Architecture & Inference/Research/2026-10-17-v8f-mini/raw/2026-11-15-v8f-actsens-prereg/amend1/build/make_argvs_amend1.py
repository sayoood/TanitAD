"""AMEND_ACTSENS_1 A2: add the gate-only tags REF_s0 / REF_s1 to the frozen argv file. REF_s{k} = R35_s{k}'s exact argv (ladder_R35_s{k}/argv.txt), --out = the REF gate out, `--wm-idm-gamma 100 100 100` appended.
The six run argvs are read from the landed frozen file and must come back byte-identical (the card's argvs do not change).
usage: python make_argvs_amend1.py <old argvs json> <new argvs json> <new argvs txt>"""
import json
import sys

R = "C:/Users/Admin/t2v8f_pulls/ladder_R35_s{}/argv.txt"
GOUT = "D:\\tanitad\\actsens\\gate_out\\"
old = json.load(open(sys.argv[1], encoding="utf-8"))
new = dict(old)
for k in (0, 1):
    t = open(R.format(k), encoding="utf-8").read().split()
    i = t.index("--out")
    t[i + 1] = GOUT + f"REF_s{k}"
    new[f"REF_s{k}"] = {"what": "AMEND_ACTSENS_1 A2: gate-only reference = R35_s%d's argv + the default dose named (never trains, no token)" % k, "argv": t + ["--wm-idm-gamma", "100", "100", "100"]}
json.dump(new, open(sys.argv[2], "w", encoding="utf-8", newline="\n"), indent=1)
with open(sys.argv[3], "w", encoding="utf-8", newline="\n") as f:
    f.write("# PREREG_V8F_ACTSENS argvs: <arm>\t<argv>. Control R35 s0/s1 is already trained (ladder_R35_s{k}/argv.txt). The pilot is DIV_s0 + --steps 200. REF_s0 / REF_s1 (AMEND_ACTSENS_1): gate only.\n")
    for n, v in new.items():
        f.write(f"{n}\t{' '.join(v['argv'])}\n")
for n in old:
    assert json.dumps(old[n]) == json.dumps(new[n])
print("ok", list(new))
