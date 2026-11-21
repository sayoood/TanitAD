"""Hold: RESULT_V8F_DAG_OACC (LONGITUDINAL FIXED) + the package raw/2026-10-27-dag-oacc/ (box-2 run metadata and logs, the dev-box T1-kin
read chain and reader, the rollout records; the read JSON landed with DAG-ORC A1) + register row H-V8F-DAG-OACC-1.
python mk_hold_dagoacc_result.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
B2, T1, RC = A + "v8f-dagoacc/", A + "v8f-dagoacc-t1/", A + "dagoacc_t1/"
LG = "C:/Users/Admin/box2_ship/logs/"
HERE = A + "dagland/dagoacc_result/"
H = "C:/Users/Admin/qland/hold_dagoacc_result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-dag-oacc/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()
res = rd(HERE + "RESULT_V8F_DAG_OACC.md")
assert b"\r" not in res and blob(V8 + "RESULT_V8F_DAG_OACC.md") is None
put("stage/" + V8 + "RESULT_V8F_DAG_OACC.md", res)
assert sha(blob(PK + "read/DAGOACC_READ.json")) == sha(rd(T1 + "read/DAGOACC_READ.json")) == \
    "9e6bd4f6940fb3e868d4d8aede50cf47c082a3833f014c34bb87ff201c99a894"

files = {}
for root, _, fs in os.walk(B2):
    for f in fs:
        p = os.path.join(root, f).replace("\\", "/")
        if f.endswith((".pt", ".npz")):
            continue
        assert os.path.getsize(p) < 200_000, p
        files["box2/" + p[len(B2):]] = p
for f in os.listdir(LG):
    if os.path.isfile(LG + f) and (f.startswith("v8fdagoacc") or f in ("chain_v8f_dagoacc.out", "chain_v8f_dagoacc_r2.out")):
        files["box2/logs/" + f] = LG + f
LEAD = "inputs/b1_eval_lead_block_k30.npz"
assert sha(rd(RC + LEAD)) == sha(blob(V8 + "raw/2026-10-22-dk/b1_eval_lead_block_k30.npz"))
for f in ("build_report.py", "readA.py", "rr_lib.py"):
    assert rd(RC + "readA_code/" + f).replace(b"\r\n", b"\n") == blob(V8 + "raw/2026-10-27-dag-orc-read/ref/readA_code/" + f), f
for root, _, fs in os.walk(RC):
    for f in fs:
        p = os.path.join(root, f).replace("\\", "/")
        rel = p[len(RC):]
        if rel == LEAD or rel.startswith("readA_code/"):
            continue
        files["t1read/" + rel] = p
for f in os.listdir(T1):
    if f in ("PINS.txt", "runs.json") or f.startswith("done_") or f.startswith("t1v8f_") and f.endswith(".json"):
        files["t1/" + f] = T1 + f
for sub in ("argv",):
    for f in os.listdir(T1 + sub):
        files[f"t1/{sub}/{f}"] = T1 + sub + "/" + f
files["landing/mk_hold_dagoacc_result.py"] = HERE + "mk_hold_dagoacc_result.py"
assert sha(rd(RC + "dagoacc_read.py")) == "b1ba095d98fc353b6ee9fcec74f4545e0a2c3ba35d2546ed6a36127d80dc0cce"
n = 0
for rel, src in sorted(files.items()):
    b = rd(src)
    if not b:
        continue
    assert blob(PK + rel) is None, rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    put("stage/" + PK + rel, b)
    n += 1

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
old = b"| **PRE-REGISTERED (2026-10-27), NOT RUN.** RESULT_V8F_DAG_HACC's committed next (COSTS POSITION"
assert g.count(old) == 1 and g.count(b"| **H-V8F-DAG-OACC-1**") == 1
new = ("| **LONGITUDINAL FIXED (MEASURED 2026-10-27, Amendment 1; the unamended table gives the same word; all 60 controls PASS).**"
       " PRIMARY-ACC accel MAE DAG-OACC − DAG −1.6243 [−1.9766, −1.3236] / −2.0450 [−2.5628, −1.6227] m/s² (−56 % / −58 %); KEEP ADE6"
       " −0.0356 [−0.1258, +0.0423] / −0.1363 [−0.2057, −0.0731] m; LON-CLEAR at both seeds (every longitudinal row vs P10 separated-better"
       " or not separated; accel vs P10 −0.82 / −0.60, DAG's was +0.81 / +1.44); D1 and D3 hold; D2 and the crawl rows fail at both seeds"
       " (yaw-rate, curvature, L2, M1 separated-worse vs P10); at s0 heading / yaw-rate / curvature / M1 separated-worse vs DAG. T1-kin"
       " self-action open loop (R-T0 −0.010 / −0.021 at T0); paired episode-cluster bootstrap, B = 2,000, 141 episodes. Committed next:"
       " adopted, DAG-OACC is the base; the crawl rows go to DAG-ORC re-based on DAG-OACC (its Amendment 1). `RESULT_V8F_DAG_OACC.md`,"
       " `raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json`. Was: **PRE-REGISTERED (2026-10-27), NOT RUN.** RESULT_V8F_DAG_HACC's committed"
       " next (COSTS POSITION").encode("utf-8")
g2 = g.replace(old, new)
i = g2.index(b"| **H-V8F-DAG-OACC-1**")
j = g2.index(b"\n", i)
row = g2[i:j]
END = b"| HYPOTHESIS | `Project Steering/PREREG_V8F_DAG_OACC.md` |"
assert row.endswith(END), row[-120:]
g2 = g2[:i] + row[: -len(END)] + b"| MEASURED | `Project Steering/PREREG_V8F_DAG_OACC.md` |" + g2[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in new.lower() and w not in res.lower(), w
put("mods-rw/" + GC, g2)
put("msg.txt", ("RESULT_V8F_DAG_OACC: LONGITUDINAL FIXED (H-V8F-DAG-OACC-1), read under Amendment 1 (the unamended table gives the same word),"
                " all 60 controls PASS. Accel MAE DAG-OACC - DAG -1.62 / -2.05 m/s^2 (-56 %% / -58 %%), ADE6 kept (-0.036 / -0.136 m), every"
                " longitudinal row clean against the kinematic floor at both seeds; the crawl rows (yaw-rate, curvature, L2, M1) stay open. DAG-OACC"
                " is adopted as the base; the crawl goes to DAG-ORC re-based on it. Package raw/2026-10-27-dag-oacc/ (%d files: box-2 run metadata"
                " and logs, the T1-kin read chain and reader, rollout records; dumps, ticks and checkpoints stay on E:). Register row updated.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
