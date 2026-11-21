"""Hold: RESULT_V8F_SPEEDSUB_T1 (NULL, label NULL-ONE) + the package raw/2026-10-28-speedsub-t1-result/ (the generator, the coverage check, the numbers audit, the register-row text,
the generator's mutation check and its output, this builder) + register row H-V8F-SPEEDSUB-T1-1 (mods-rw, line count unchanged).
python mk_hold_speedsub_t1_result.py <live tip> [--out <hold dir>] [--trailer-name "Claude Opus 5.5"]
The default hold dir is C:/Users/Admin/qland/hold_speedsub_t1_result/ (refused if it exists). The Master Mind lands it; this script only writes the hold, and reads the repo
only through the mirror (git show at the given tip), never through a working tree. It re-checks every in-repo source the RESULT cites (path and sha256) against the live tip, and
the register row against the text it expects, so a tip that moved under the RESULT stops the build."""
import hashlib
import os
import re
import subprocess
import sys

args = sys.argv[1:]
assert args and not args[0].startswith("--"), "usage: mk_hold_speedsub_t1_result.py <live tip> [--out dir] [--trailer-name name]"
TIP = args[0]
HOLD = "C:/Users/Admin/qland/hold_speedsub_t1_result/"
if "--out" in args:
    HOLD = args[args.index("--out") + 1].replace("\\", "/").rstrip("/") + "/"
TRAILER = "Claude Opus 5.5"
if "--trailer-name" in args:
    TRAILER = args[args.index("--trailer-name") + 1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
HERE = "E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedsub-t1-result/"
H = HOLD
assert not os.path.exists(H), H

r_ = subprocess.run(GIT + ["rev-parse", "--verify", TIP + "^{commit}"], capture_output=True)
assert r_.returncode == 0, "the tip does not resolve to a commit in the mirror"
TIPFULL = r_.stdout.decode().strip()
FORBID = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
HFTOK = re.compile(rb"hf_[A-Za-z0-9]{8,}")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIPFULL}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not HFTOK.search(b), "hf_ token in " + rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- the RESULT
res = rd(HERE + "RESULT_V8F_SPEEDSUB_T1.md")
assert b"\r" not in res and res.startswith(b"# RESULT_V8F_SPEEDSUB_T1: ") and b"Outcome **NULL**" in res and b"label NULL-ONE" in res
assert blob(V8 + "RESULT_V8F_SPEEDSUB_T1.md") is None, "RESULT already at the tip"
for w in FORBID:
    assert w not in res.lower(), w
for need in ("\u22120.1949 [\u22120.2619, \u22120.1313]", "all 158 control flags PASS", "NOT COLLAPSE", "Master Mind's ranking (beside the table)", "ESTIMATED / INFERENCE ranking, not a measured outcome",
             "+0.0008 [\u22120.0374, +0.0375]", "\u22120.0185 [\u22120.0637, +0.0260]"):
    assert need.encode("utf-8") in res, need

# every in-repo source the RESULT cites: path and sha256 must hold at the live tip
PROV = re.findall(rb"^\| [^|]+ \| `([^`]+)` \| `([0-9a-f]{64})` \| in the repo at the tip \|$", res, re.M)
assert len(PROV) == 16, len(PROV)
for path, h in PROV:
    p = path.decode("utf-8")
    p = p if p.startswith("Project Steering/") else V8 + p
    b = blob(p)
    assert b is not None and sha(b).encode() == h, ("source cited by the RESULT differs at the live tip", p, None if b is None else sha(b))
put("stage/" + V8 + "RESULT_V8F_SPEEDSUB_T1.md", res)

# ---- the package
files = {"audit/gen_result.py": HERE + "gen_result.py", "audit/audit_cover.py": HERE + "audit_cover.py", "audit/NUMBERS_AUDIT.md": HERE + "NUMBERS_AUDIT.md",
         "audit/REGISTER_ROW.md": HERE + "REGISTER_ROW.md", "audit/mutate_gen_check.py": HERE + "mutate_gen_check.py", "audit/gen_mutation_check.out": HERE + "gen_mutation_check.out",
         "landing/mk_hold_speedsub_t1_result.py": HERE + "mk_hold_speedsub_t1_result.py"}
SKIP = (".pt", ".npz", ".npy", ".pyc")
n = 0
lfn = []
sums_out = []
for rel, src in sorted(files.items()):
    assert not src.endswith(SKIP), src
    b = rd(src)
    assert b and len(b) < 400_000, (rel, len(b))
    if rel.endswith((".py", ".md", ".out")) and b"\r" in b:
        b = b.replace(b"\r\n", b"\n")
        assert b"\r" not in b, rel
        lfn.append(rel)
    assert blob(PK + rel) is None, rel
    put("stage/" + PK + rel, b)
    sums_out.append(f"{sha(b)} *{rel}")
    n += 1
if lfn:
    b = ("CRLF -> LF on banking (content otherwise unchanged):\n" + "\n".join(lfn) + "\n").encode()
    put("stage/" + PK + "LF_NORMALISED.txt", b)
    sums_out.append(f"{sha(b)} *LF_NORMALISED.txt")
    n += 1
put("stage/" + PK + "PACKAGE_SHA256SUMS.txt", ("\n".join(sums_out) + "\n").encode())
n += 1

# ---- the register row (mods-rw; one line replaced, line count unchanged)
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g is not None and b"\r" not in g
rr = rd(HERE + "REGISTER_ROW.md").decode("utf-8")
m = re.search(r"<!-- STATUS-BEGIN -->\n(.*?)\n<!-- STATUS-END -->", rr, re.S)
status = m.group(1).strip()
assert status.startswith("**NULL (label NULL-ONE; MEASURED 2026-10-28") and "\n" not in status and "|" not in status and "Was:" not in status and status.endswith("(sha256 b3b72325…).")
for w in FORBID:
    assert w not in status.lower().encode(), w
ID = b"| **H-V8F-SPEEDSUB-T1-1** (2026-10-28) |"
assert g.count(ID) == 1
i = g.index(ID)
j = g.index(b"\n", i)
row = g[i:j]
SHORT = b"| **NULL (label NULL-ONE; MEASURED 2026-10-28, T1-kin, Amendment 1 in force;"
WAS = b"Was: **PRE-REGISTERED, NOT RUN (2026-10-28).**"
TAIL = b"| MEASURED (NULL) | `Project Steering/PREREG_V8F_SPEEDSUB_T1.md` |"
assert row.count(SHORT) == 1 and row.count(WAS) == 1 and row.endswith(TAIL) and row.count(TAIL) == 1, row[-200:]
a, b_ = row.index(SHORT), row.index(WAS)
assert a < b_ and row[b_ - len(b"RESULT to follow. "):b_] == b"RESULT to follow. "
old_prefix = row[a:b_]
assert b"SPEEDSUB_T1_READ.json" in old_prefix and b"|" not in old_prefix[2:] and b"sha256 b3b72325" in old_prefix
row2 = row[:a] + b"| " + status.encode("utf-8") + b" " + row[b_:]
assert row2.startswith(ID) and row2.count(b"\n") == 0 and row2.endswith(TAIL)
assert row2.count(b"|") == row.count(b"|"), (row2.count(b"|"), row.count(b"|"))      # the pipes of the table row are unchanged
assert row2[row2.index(WAS):] == row[b_:]                                              # the Was: chain survives byte for byte
assert row2[:a] == row[:a]                                                             # the claim cell and the id are untouched
g2 = g[:i] + row2 + g[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)

# ---- commit message
MSG = ("RESULT_V8F_SPEEDSUB_T1: NULL (label NULL-ONE), H-V8F-SPEEDSUB-T1-1, T1-kin self-action open loop; one hit of four runs, at the slow-T draw REP s0; all 158 reader control flags PASS, no guard fires (Amendment 1's G-VIOL included)."
       " Outcome, hits, the four guards and the floor rule re-derived from the per-run cells by gen_result.py, which stops on any mismatch with the reader (16 of 16 mutants killed)."
       " ADE6 egoc - identity: CTRL s0 -0.0233 [-0.0629, +0.0156], CTRL s1 -0.0076 [-0.0500, +0.0349], REP s0 -0.1949 [-0.2619, -0.1313] (the hit), REP s1 -0.0333 [-0.0832, +0.0131] m, against a bar of -0.06035 m at all four runs."
       " REP s0 also moves speed, along-track, accel, heading, yaw-rate, curvature and L2 separated-better and VIOL-ANY separated-worse (+0.0421); the healthy draws show no separated ADE6 change; M1 crawl share moves at no run;"
       " egoshufc (K-REG) is separated-worse at all four runs (+3.99 to +4.49 m). Floor rule NOT COLLAPSE: the REP s0 - CTRL s0 gap on ADE6 0.4025 -> 0.2309 m."
       " Beside, deciding nothing: the logged-lane read-out moves CTRL's ADE6 by +0.0008 / -0.0185 m, not separated. Four families per run, STRATEGIC reported with S-POS across arms NOT COMPUTED (reason in the RESULT), anti-echo NOT BUILT, goal-head F1 NOT COMPUTABLE at T1-kin."
       " Committed next: PREREG section 8 (a); Master Mind's ranking (ESTIMATED / INFERENCE): (a) stays registered and is not run now, DAG-ORC's perturbation is the largest measured crawl lever and its registered route (LAUNCH-INTENT, box 2) waits on the PI's go."
       f" Package raw/2026-10-28-speedsub-t1-result/ ({n} files: generator, coverage check, numbers audit, register-row text, mutation check, builder). Register row updated.\n\n"
       f"Co-Authored-By: {TRAILER} <noreply@anthropic.com>\n")
for w in FORBID:
    assert w not in MSG.lower().encode(), w
assert MSG.endswith(f"Co-Authored-By: {TRAILER} <noreply@anthropic.com>\n")
put("msg.txt", MSG.encode("utf-8"))

# ---- last checks on everything staged
bad = []
for dp, _, fs in os.walk(H):
    for f in fs:
        p = os.path.join(dp, f)
        b = rd(p)
        if HFTOK.search(b):
            bad.append(("hf_", p))
        if p.endswith((".md", ".py", ".sh", ".out", ".txt")) and b"\r" in b:
            bad.append(("CR", p))
        if p.endswith(SKIP):
            bad.append(("binary", p))
assert not bad, bad
print("hold written", H, n, "package files; tip", TIPFULL[:8], "; register line count", len(g.split(b"\n")), "unchanged")
