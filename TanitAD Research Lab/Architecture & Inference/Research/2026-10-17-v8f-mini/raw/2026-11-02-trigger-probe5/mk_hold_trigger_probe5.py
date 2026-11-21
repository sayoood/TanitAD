"""Hold: trigger probe 5 (native-resolution DINOv3 crops of the traffic-light and lead-vehicle bands, box 2) banked as
raw/2026-11-02-trigger-probe5/, and register row H-CRAWL-06-PROBE5 (append). Frames, PCA files over 3 MB and F: scratch stay off the
repo (NOT_LANDED.txt). python mk_hold_trigger_probe5.py <live tip> <first 8 hex of RESULTS.md in MANIFEST.txt>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
RES_PIN = sys.argv[2]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_trigger_probe5/"
S = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe5/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-11-02-trigger-probe5/"
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
    m = re.match(r"([0-9a-f]{64})\s+(\S.*?)\s+\((\d+) bytes\)\s*$", line)
    if m:
        pins[m.group(2).replace(S, "")] = m.group(1)
assert pins.get("PREREG_FROZEN.md", "").startswith("791e8a62") and pins.get("RESULTS.md", "").startswith(RES_PIN), "manifest"
res = open(S + "RESULTS.md", "rb").read()
for w in BANNED:
    assert w not in res.lower(), w
assert b"**STATUS: FINAL" in res and b"PENDING" not in res
for f in ("CHAIN_P5", "READOUT"):
    assert os.path.exists(S + "raw/" + f + ".done"), f

n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins and not rel.startswith(("logs/", "raw/logs/")) and rel not in ("raw/make_manifest.out", "raw/make_manifest_final.out"):
            assert sha(b) == pins[rel], rel
        why = ("aside: moved-aside files" if rel.startswith("aside/") else
               "cache" if "__pycache__" in rel or ".pytest_cache" in rel else
               "over 3 MB" if len(b) >= LIMIT else None)
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
F = "F:/TanitAD-scratch/trigger_probe5_scratch/"
for f in sorted(os.listdir(F)):
    st = os.stat(F + f)
    notl.append(f"{F}{f} | {st.st_size if os.path.isfile(F + f) else '-'} | - | probe-5 scratch: crop features, PCA components, align output (F:; box 2 holds the features, FEAT_SHA256.txt)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED: files of this package kept on the dev box (" + S + ", " + F + ")\n# path | bytes | sha256 | why\n"
                                       + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_trigger_probe5.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2

for q in (b"0.0231 [-0.0422, -0.0050]", b"0.0028 [-0.0081, +0.0024]", b"0.468 [0.446, 0.489]", b"0.4962", b"6.4 %", b"UNVERIFIED"):
    assert q in res, q
ROW = ("**H-CRAWL-06-PROBE5** (2026-11-02)",
       "Precursor of H-CRAWL-06, continued: DINOv3-L/16 at native patch resolution on crops of the traffic-light band and the"
       " lead-vehicle band (cut from the 768x1920 frame, box 2) carries more go/stay trigger than probe 1's whole-vector V+E: C+V+E "
       + M + " V+E dAUC >= 0.03 with CI > 0 in one population; DOES NOT HELP iff both cells < 0.01.",
       "**MEASURED 2026-11-02, TRAIN windows of probes 1-4 + eval-141, fold-mean AUC, episode-cluster bootstrap B = 2,000, seed 0"
       " (another draw of episodes; no readout seed spread in this run); card frozen 14:03:08 (sha256 791e8a62…) before any crop"
       " feature; align gate PASS (rel L2 median 0.0264).** Verdict by the committed rule: CROPS DO NOT HELP, controls PASS in both"
       " populations. C+V+E " + M + " V+E " + M + "0.0231 [" + M + "0.0422, " + M + "0.0050] (P-STOP, separated below V+E), "
       + M + "0.0028 [" + M + "0.0081, +0.0024] (P-MOVE). Disclosed: the crops are JPEG q95 re-encodes, which change the features by"
       " a median of 6.4 % (not gated, label-independent, so it can only bias toward NOT HELP); P-MOVE C2 LEAD+E read 0.468 [0.446,"
       " 0.489] at seed 2 and 0.4962 over five draws; the card's avg-pool known answer has no test (UNVERIFIED); C = 0.001, the grid"
       " edge, was selected in every fold. With probes 1-4 the frozen DINOv3 caps the go/stay trigger at about V+E, and native"
       " crops of the two bands do not lift it; H-GAPDECOMP-1 places no DAG-OACC crawl row as TRIGGER-BOUND, so the trigger is not"
       " the lever for DOMINANT. The perception options left (fine-tuned or other encoder, a traffic-light-state input) are"
       " architecture, a PI decision.",
       "MEASURED (CROPS DO NOT HELP)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-CRAWL-06-PROBE5**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("Trigger probe 5 (H-CRAWL-06-PROBE5): DINOv3 at native resolution on traffic-light and lead-vehicle crops (box 2)"
                " CROPS DO NOT HELP (committed rule, controls PASS): C+V+E - V+E -0.0231 [-0.0422, -0.0050] (P-STOP), -0.0028"
                " [-0.0081, +0.0024] (P-MOVE). JPEG q95 feature effect (median 6.4 %) disclosed. With probes 1-4 the frozen encoder"
                " caps the trigger at V+E. Package raw/2026-11-02-trigger-probe5/ (" + str(n) + " files; NOT_LANDED.txt lists "
                + str(len(notl)) + " kept on E: / F: / box 2); register row." + chr(10) + chr(10)
                + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
