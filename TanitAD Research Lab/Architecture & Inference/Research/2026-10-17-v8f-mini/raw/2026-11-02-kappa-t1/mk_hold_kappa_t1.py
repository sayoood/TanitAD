"""Hold: KAPPA-T1 (H-KAPPA-T1-1: the curvature-bounded decode composed with S, with its extractor floor fix) banked as
raw/2026-11-02-kappa-t1/, and its register row (append). Scratch on F: stays there (NOT_LANDED.txt). python mk_hold_kappa_t1.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_kappa_t1/"
S = "E:/Projects/TanitAD-artifacts/dagland/kappa_t1/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-11-02-kappa-t1/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
LIMIT = 3_000_000
M = "\u2212"


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel, b):
    assert len(b) > 0 and len(rel) <= 220, (rel, len(rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    assert len(b) < LIMIT or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


pins = {}
for line in open(S + "MANIFEST.txt", encoding="utf-8"):
    m = re.match(r"\s*(\S.*?)\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*$", line)
    if m:
        pins[m.group(1).replace(S, "")] = m.group(2)
assert pins.get("PREREG_FROZEN.md", "").startswith("9564d568") and pins.get("READER_FIX_FLOOR.md", "").startswith("e25ffb9d") and pins.get("RESULTS.md", "").startswith("0c0638db"), "manifest"
txt = b"".join(open(S + f, "rb").read() for f in ("RESULTS.md", "READER_FIX_FLOOR.md"))
n, notl, tot = 0, [], 0
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins and not rel.startswith("logs/"):
            assert sha(b) == pins[rel], rel
        why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else
               "log over 512 kB (kept on E:)" if rel.startswith("logs/") and len(b) >= 512_000 else
               "over 3 MB (rollout tick arrays / tool trees; regenerable by the chain)" if len(b) >= LIMIT else None)
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
        tot += len(b)
notl.append("F:/TanitAD-scratch/kappa_t1/ | - | - | rollout outputs out/ (8 rollouts) and tool trees (regenerable by kappa_t1_chain.sh)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_kappa_t1.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2
for q in ("+0.0115", "+0.0606", "+0.0084", "+0.0253", "-1.2441", "-1.1296", "-1.1844", "-1.1642", "1.711e-02", "+1.6946", "+1.3399",
          "+1.6201", "+1.4407", "105", "227", "HARMS", "kappa-only"):
    assert q.encode() in txt, q
ROW = ("**H-KAPPA-T1-1** (2026-11-02)",
       "After S (H-SMOOTH-T1-1), the lever selected by the pre-registered gate on the curvature excess, the curvature-bounded decode"
       " `project_feasible(a_max = 4.0, kappa_max = 0.2, clamp_entry = False)` applied after S at every replan inside the T1-kin rollout,"
       " makes v8F DOMINANT against P10 (Read A at both seeds), or MOVES the crawl rows against Y = the fresh S rerun without HARMS"
       " (ADE6 X " + M + " Y not above +0.0604 m and none of the 15 non-crawl rows separated-worse).",
       "**MEASURED 2026-11-02, T1-kin, eval-141 causal lane, DAG-OACC and SOUP at s0 / s1, paired episode-cluster bootstrap B = 2,000"
       " (another draw of episodes only); card frozen 15:44:58 (sha256 9564d568…) before any X number; gate high-|kappa| share 0.838 /"
       " 0.805 selected lever (a).** The first read was refused by the extractor's 1e-2 m floor guard against the banked baseline; the"
       " fresh floor differs from the banked one by 1.711e-02 m at most, identical in all four arm-seeds. READER_FIX_FLOOR.md (written"
       " before any X number) applies the floor and window check against the fresh Y, which the card already defines as the"
       " baseline (an extractor fix; outcome rule unchanged). All controls PASS. **Outcome: HARMS for DAG-OACC and SOUP at both"
       " seeds.** The box closes curvature MAE, L2 curvature MAE and yaw-rate MAE against P10, but heading MAE X " + M + " Y +1.6946 /"
       " +1.3399 deg (DAG-OACC), +1.6201 / +1.4407 deg (SOUP) and S-POS@30/50/80 read separated-worse (DAG-OACC s1: ADE6 +0.0606 m,"
       " along-track and heading instead), and some ticks' issued plans move by 105-227 m. ADE6 X " + M + " Y +0.0115 / +0.0606 m"
       " (DAG-OACC), +0.0084 / +0.0253 m (SOUP); X " + M + " P10 " + M + "1.2441 / " + M + "1.1296 / " + M + "1.1844 / " + M + "1.1642 m."
       " Read A against P10: D1 and D3 PASS everywhere; D2 fails only on M1 crawl share, at all four arm-seeds, and the box moves M1"
       " by " + M + "1 % to " + M + "13 % of its excess. Next (the card's committed arm for HARMS): the kappa-only projection"
       " (a_max = inf), with the accel-bounded longitudinal decode for M1 (`dagland/feas2_t1/`, its own card).",
       "MEASURED (HARMS; D2 blocked by M1 only)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-KAPPA-T1-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("KAPPA-T1 (H-KAPPA-T1-1), T1-kin: the curvature-bounded decode after S reads HARMS for DAG-OACC and SOUP at both seeds"
                " (heading MAE +1.3 to +1.7 deg and stop-position rows separated-worse against the fresh S run); it closes curvature, L2"
                " and yaw-rate against P10, and D2 is now blocked by M1 crawl share alone. Extractor floor fix disclosed"
                " (READER_FIX_FLOOR.md, written before any X number). Package raw/2026-11-02-kappa-t1/ (" + str(n) + " files, "
                + str(tot // 1000) + " kB; NOT_LANDED.txt lists " + str(len(notl)) + "); register row." + chr(10) + chr(10)
                + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "kB", tot // 1000, "not landed", len(notl))
