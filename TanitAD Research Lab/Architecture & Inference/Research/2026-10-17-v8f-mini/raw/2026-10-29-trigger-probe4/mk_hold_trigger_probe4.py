"""Hold: trigger probe 4 (attention / max-pool readout over the probe-3 tokens) banked as raw/2026-10-29-trigger-probe4/, and register row
H-CRAWL-06-PROBE4 (append). Held-out attention weights stay on F:; aside/ and caches stay on E: (NOT_LANDED.txt).
python mk_hold_trigger_probe4.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_trigger_probe4/"
S = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe4/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-29-trigger-probe4/"
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
    m = re.match(r"([0-9a-f]{64})\s+\d+\s+(\S.*?)\s*$", line)
    if m:
        pins[m.group(2).replace(S, "")] = m.group(1)
assert pins.get("PREREG_FROZEN.md", "").startswith("a8f5d785") and pins.get("RESULTS.md", "").startswith("6e6f0100"), "manifest"
res = open(S + "RESULTS.md", "rb").read()
for w in BANNED:
    assert w not in res.lower(), w
assert b"**STATUS: FINAL" in res and b"PENDING" not in res
for f in ("CHAIN4",):
    assert os.path.exists(S + "raw/" + f + ".done"), f

n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins and rel not in ("raw/make_manifest.out", "raw/make_manifest_final.out"):  # its own stdout, rewritten after the hashes
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
F = "F:/TanitAD-scratch/trigger_probe4_scratch/"
for f in sorted(os.listdir(F)):
    st = os.stat(F + f)
    if os.path.isfile(F + f):
        notl.append(f"{F}{f} | {st.st_size} | - | held-out attention weights (single copy on F:)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED: files of this package kept on the dev box (" + S + ", " + F + ")\n# path | bytes | sha256 | why\n"
                                       + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_trigger_probe4.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2

ROW = ("**H-CRAWL-06-PROBE4** (2026-10-29)",
       "Precursor of H-CRAWL-06, continued: a SPATIAL readout of the frozen DINOv3-L/16 tokens (learned 4-query attention pooling over"
       " the 160 per-token PCA-16 tokens of the 8x20 grid, + E; 3 readout seeds) carries more go/stay trigger than probe 1's whole-vector"
       " V+E (dAUC >= 0.03, CI > 0, > 2x seed spread, in one population), or resolves more at 8x20 than at 4x10 under the same readout;"
       " DOES NOT HELP iff both contrasts < 0.01 in both populations.",
       "**MEASURED 2026-10-29, TRAIN 12,000 + 12,000 windows, fold-mean AUC, episode-cluster bootstrap B = 2,000 (another draw of"
       " episodes; the 3-seed spread answers readout training variance); card frozen 04:48:11 (sha256 a8f5d785…) before any"
       " real-label fit.** Verdict by the committed rule: DOES NOT HELP, every control PASS. G8att+E " + M + " V+E " + M + "0.018 ["
       + M + "0.039, +0.002] (P-STOP), " + M + "0.004 [" + M + "0.010, +0.002] (P-MOVE); G8att+E " + M + " G4att+E " + M + "0.001 ["
       + M + "0.016, +0.014], +0.002 [" + M + "0.003, +0.007]. Max-pool beside: G8max+E " + M + " V+E " + M + "0.002 / +0.001. The"
       " learned attention is near uniform (normalised entropy 0.988; largest token share 0.015 vs uniform 0.006), while the planted"
       " known answer puts 0.95 on one token and uniform attention drops it to AUC 0.568. On P-STOP green-light windows no image readout"
       " beats ego alone (G8att+E " + M + " E " + M + "0.036 [" + M + "0.110, +0.045]). With probes 1-3: five readout families and two"
       " grids all stop at or below V+E (P-STOP 0.680, P-MOVE 0.785; ego alone 0.626 / 0.732), so the frozen single-frame field caps the"
       " trigger; a 16x40 re-encode is not supported. What remains is upstream of the pooled cache (native-resolution crops, another or a"
       " fine-tuned encoder, a traffic-light-state input): perception architecture, a PI decision.",
       "MEASURED (DOES NOT HELP)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-CRAWL-06-PROBE4**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("Trigger probe 4 (H-CRAWL-06-PROBE4): an attention / max-pool readout over the frozen DINOv3 8x20 and 4x10 tokens DOES NOT"
                " HELP (committed rule, all controls PASS): G8att+E - V+E -0.018 [-0.039, +0.002] (P-STOP), -0.004 [-0.010, +0.002]"
                " (P-MOVE); 8x20 vs 4x10 under attention within +-0.002. The learned attention is near uniform. With probes 1-3 the"
                " frozen single-frame field caps the go/stay trigger at V+E; the remaining perception levers are architecture (PI)."
                " Package raw/2026-10-29-trigger-probe4/ (" + str(n) + " files; NOT_LANDED.txt lists " + str(len(notl)) + " kept on E: / F:);"
                " register row." + chr(10) + chr(10) + "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" + chr(10)).encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
