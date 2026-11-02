"""Hold: HISTSWAP-LF (`dagland/histswap_lf/`, H-HISTSWAP-LF-1) + DRAFT_RECOVERY_CARD_v2.md
banked as raw/2026-11-02-histswap-lf/ (the card under raw/2026-11-02-histswap/), with one register row (append). F: scratch stays (NOT_LANDED.txt).
python mk_hold_histswap_lf.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_histswap_lf/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
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


D = "E:/Projects/TanitAD-artifacts/dagland/"
S = D + "histswap_lf/"
PK = V8 + "raw/2026-11-02-histswap-lf/"
HS = V8 + "raw/2026-11-02-histswap/"
pins = {}
for line in open(S + "MANIFEST.txt", encoding="utf-8"):
    m = re.match(r"\s*(\S.*?)\s*\|\s*\d+\s*\|\s*([0-9a-f]{64})\s*$", line)
    if m:
        pins[m.group(1).replace(S, "")] = m.group(2)
assert pins.get("PREREG_HISTSWAP_LF_FROZEN.md", "").startswith("d98db980") and pins.get("RESULTS.md", "").startswith("91acd04a"), "manifest"
txt = open(S + "RESULTS.md", "rb").read()
for w in BANNED:
    assert w not in txt.lower(), w
n, tot, notl = 0, 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins and not rel.startswith("logs/"):
            assert sha(b) == pins[rel], rel
        why = ("cache" if "__pycache__" in rel or ".pytest_cache" in rel else
               "log over 512 kB (kept on E:)" if rel.startswith("logs/") and len(b) >= 512_000 else
               "over 3 MB" if len(b) >= LIMIT else None)
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
notl.append("F:/TanitAD-scratch/histswap_lf/ | - | - | rollout outputs and tool trees (regenerable by the package's chain)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED\n# path | bytes | sha256 | why\n" + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
card = open(D + "histswap/DRAFT_RECOVERY_CARD_v2.md", "rb").read()
assert sha(card).startswith("c8709eeb") and b"\r" not in card and blob(HS + "DRAFT_RECOVERY_CARD_v2.md") is None
for w in BANNED:
    assert w not in card.lower(), w
put("stage/" + HS + "DRAFT_RECOVERY_CARD_v2.md", card)
put("stage/" + PK + "mk_hold_histswap_lf.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 3
for q in ("+0.0201", "+0.0270", "0.599", "0.633", "-0.559", "-0.301", "+0.5867", "+0.6527", "-0.6165", "-0.5171", "B_LEVEL", "0.939", "0.610",
          "S-POS@30"):
    assert q.encode() in txt, q
EST = " paired episode-cluster bootstrap B = 2,000, seed 0 (another draw of episodes only)"
ROW = ("**H-HISTSWAP-LF-1** (2026-11-02)",
       "Leak-free split of H-HISTSWAP-1: the M1 crawl excess comes from O copying the TREND of its own executed history (inertia) iff a"
       " constant-velocity history at the EXECUTED speed (HCV, deployable) closes >= 0.5 of Y " + M + " P10's M1 excess at all four"
       " cells; it comes from a missing speed LEVEL iff HCV does not and a constant-velocity history at the LOGGED speed (HLVL, oracle)"
       " does.",
       "**MEASURED 2026-11-02, T1-kin, eval-141 causal lane, DAG-OACC and SOUP at s0 / s1, after S," + EST + "; card frozen 18:50:07"
       " (sha256 d98db980…) before any HCV / HLVL number; SREP5 bit-identity rerun PASS.** **Reading (B), LEVEL.** HCV: NOT CLOSES,"
       " M1 X " + M + " Y separated-worse at 4/4 cells (+0.0201 to +0.0270), share of the excess closed " + M + "0.559 to " + M + "0.301,"
       " ADE6 X " + M + " Y +0.5867 to +0.6527 m. HLVL: CLOSES at 4/4 cells, share closed 0.599 to 0.633, ADE6 X " + M + " Y " + M
       + "0.6165 to " + M + "0.5171 m. Outcome by the KAPPA / FEAS rule: HARMS for both arms on both bases (HCV: ADE6 beyond the guard"
       " plus along-track, DK and S-POS rows; HLVL: S-POS@30/50/80, STRATEGIC family, active for v8F). The trend copy is ruled out;"
       " O lacks the true speed level once the executed speed lags (HISTSWAP's full logged history closes 0.939 against HLVL's 0.610;"
       " the difference also carries logged yaw and pose rows). HLVL is an oracle and HCV removes all acceleration information, not"
       " only the trend. Next: the recovery fine-tune with re-expressed targets on box 2 (PI go pending), read with a pre-registered"
       " frame-freeze control, because T1-kin's logged frames move at the human's pace (`DRAFT_RECOVERY_CARD_v2.md`).",
       "MEASURED (reading B, LEVEL; both arms HARMS)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-HISTSWAP-LF-1**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("HISTSWAP-LF (H-HISTSWAP-LF-1), T1-kin: a constant-velocity ego history at the executed speed (deployable) does not close v8F's M1"
                " crawl excess (separated-worse at 4/4, ADE6 +0.59..+0.65 m), while one at the logged speed (oracle) closes 0.60-0.63 at 4/4:"
                " the speed LEVEL is missing, not a copied trend. Both arms HARMS. Draft recovery card v2 adds the frame-freeze control"
                " (T1-kin frames move at the human's pace). Package raw/2026-11-02-histswap-lf/ (" + str(n) + " files, " + str(tot // 1000)
                + " kB; NOT_LANDED lines " + str(len(notl)) + "); register row." + chr(10) + chr(10)
                + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "kB", tot // 1000, "not landed", len(notl))
