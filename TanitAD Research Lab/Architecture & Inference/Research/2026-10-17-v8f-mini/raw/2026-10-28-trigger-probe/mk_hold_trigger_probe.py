"""Hold: the go/stay trigger-decodability probe (H-CRAWL-06 precursor, widened; card frozen before data), banked as
raw/2026-10-28-trigger-probe/, plus one register row appended to GOALS_AND_CLAIMS.md. scratch/, caches and files over 3 MB stay
on E: and are listed with sha256 in NOT_LANDED.txt.  python mk_hold_trigger_probe.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_trigger_probe/"
S = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-trigger-probe/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
LIMIT = 3_000_000


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
    m = re.match(r"([0-9a-f]{64})\s+(\S.*?)\s+\(\d+ bytes\)\s*$", line)
    if m:
        pins[m.group(2)] = m.group(1)
assert pins.get("PREREG_FROZEN.md", "").startswith("0eb1c25c") and pins.get("RESULTS.md", "").startswith("dc25880c"), "manifest"

n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins:
            assert sha(b) == pins[rel], rel
        why = ("scratch: smoke files and feature memmaps" if rel.startswith("scratch/") else
               "cache" if "__pycache__" in rel or ".pytest_cache" in rel else
               "over 3 MB (feature / PCA / window tables; rebuilt by the package's scripts)" if len(b) >= LIMIT else None)
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
            assert b"\r" not in b, rel
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED: files of this package kept on the dev box (" + S + ")\n# path | bytes | sha256 | why\n"
                                       + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_trigger_probe.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2

SRC = "`" + PK + "RESULTS.md`"
ROW = ("**H-CRAWL-06-PROBE** (2026-10-28)",
       "Precursor of H-CRAWL-06, widened: on TRAIN, the go/stay decision (P-STOP: v0 < 0.5 m/s, GO = launch definition; P-MOVE: 0.5 <= v0"
       " < 6 m/s, label = launch definition) is linearly decodable from the frozen DINOv3 pool4x10 current-frame features beyond the ego"
       " inputs O reads (dAUC(V+E - E) >= 0.05, CI excluding 0), and the trunk state h_O carries it (AUC(Z+E) >= AUC(V+E) - 0.02): D1"
       " information present but unused; D2 the trunk discards it; D3 the features lack it (dAUC < 0.02 linear and MLP).",
       "**MEASURED 2026-10-28, zero training, TRAIN windows (canonical 306,566, digest re-verified), 5 episode folds, 12,000 windows per"
       " population (849 / 2,140 clips), episode-cluster bootstrap B = 2,000; card frozen at 22:10:08 before data (PREREG_FROZEN.md sha256"
       " 0eb1c25c\u2026).** P-MOVE, as frozen, all controls PASS: AUC E 0.731, V+E 0.784, Z+E 0.880; dAUC(V+E - E) +0.053 [0.041, 0.066]:"
       " the rule reads D1, but with a nonlinear ego baseline (MLP) the vision gain is +0.016 [0.007, 0.025]. P-STOP, as frozen: control C2"
       " (episode-shuffled labels) read 0.4629 [0.4267, 0.49993] and missed its committed 'CI covers 0.5' by 7e-5, so the frozen pooled"
       " reading is NOT EVALUABLE; a post-freeze per-fold-mean AUC (C2 0.4831 [0.465, 0.502], PASS) gives dAUC +0.054 [0.025, 0.082] and"
       " also reads D1 (exploratory). Eval-141, probes fitted on TRAIN, no refit: P-STOP (101 windows / 19 episodes) E 0.517, V+E 0.519,"
       " Z+E 0.655; P-MOVE E 0.708, V+E 0.748, Z+E 0.804. Strata (clip-level, VLM-derived traffic-light labels): the image gain over E on"
       " P-STOP is +0.099 on red-light clips, " + "\u2212" + "0.058 on green-light clips and " + "\u2212" + "0.019 where the lead vehicle"
       " departs. Confound found after the freeze (diagnostic, not a verdict): h_O also reads the nav token, which on PhysicalAI is"
       " derived from the ego's own future path (v7.2 provenance ORACLE), plus the speed limit and 4 frames; nav + limit + E alone reaches"
       " 0.719 on P-STOP (E 0.593). Reading: the rule's D1 holds by its letter, but the Z condition is carried mostly by the oracle nav,"
       " not by vision; the current-frame image features add little go/stay information over ego and none on green-light or"
       " lead-departure clips. Next zero-training arms: the model's 4 frames or frame differences (motion cues), and h_O with the nav"
       " input neutralised. Disclosed: free RAM was 2.3-2.7 GB when the V extraction started, below the brief's 3 GB guard (RSS stayed"
       " under 1 GB).",
       "MEASURED (P-MOVE thin D1 as frozen; P-STOP frozen reading NOT EVALUABLE, exploratory D1; Z advantage carried by the oracle nav)")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-CRAWL-06-PROBE**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW[:3] + (ROW[3], SRC)) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("Trigger probe (H-CRAWL-06 precursor, frozen first): P-MOVE reads a thin D1 as frozen (dAUC(V+E - E) +0.053, MLP"
                " +0.016); P-STOP's frozen reading is not evaluable (control C2 missed by 7e-5; post-freeze per-fold reading D1,"
                " exploratory). The trunk's AUC edge (0.80 / 0.88) is carried mostly by the nav token, derived from the ego's own future"
                " path; no image gain on green-light or lead-departure clips. Package raw/2026-10-28-trigger-probe/ (" + str(n)
                + " files; NOT_LANDED.txt lists " + str(len(notl)) + " kept on E:); one register row appended.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
