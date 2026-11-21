"""Hold: trigger probe 3 (8x20 vs 4x10 at a fixed per-token readout) banked as raw/2026-10-29-trigger-probe3/, and register row
H-CRAWL-06-PROBE3 (append). The G8 / S8 / S4 arrays stay on F:; aside/, caches and files over 3 MB stay on E: (NOT_LANDED.txt).
python mk_hold_trigger_probe3.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_trigger_probe3/"
S = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe3/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-29-trigger-probe3/"
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
assert pins.get("PREREG_FROZEN.md", "").startswith("f73b8a5a") and pins.get("RESULTS.md", "").startswith("76ef153f"), "manifest"
res = open(S + "RESULTS.md", "rb").read()
for w in BANNED:
    assert w not in res.lower(), w
assert b"**STATUS: FINAL" in res and b"PENDING" not in res
for f in ("CHAIN_linear", "CHAIN_eval", "CHAIN_mutate", "CHAIN_mlp", "CHAIN_evalmlp", "CHAIN_mlpctl", "CHAIN_pf", "CHAIN_c2band"):
    assert os.path.exists(S + "raw/" + f + ".done"), f

n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins and rel != "raw/make_manifest.out":  # its own stdout, rewritten after the hashes
            assert sha(b) == pins[rel], rel
        why = ("aside: moved-aside files" if rel.startswith("aside/") else
               "cache" if "__pycache__" in rel or ".pytest_cache" in rel else
               "over 3 MB (per-fold PCA bases; rebuilt by pca_g.py)" if len(b) >= LIMIT else None)
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
F = "F:/TanitAD-scratch/trigger_probe3_scratch/"
for f in sorted(os.listdir(F)):
    st = os.stat(F + f)
    if os.path.isfile(F + f):
        notl.append(f"F:{F}{f} | {st.st_size} | see raw/G8_sha256.txt / raw/INPUTS_SHA256.json | feature and token arrays (single copy on F:)")
put("stage/" + PK + "NOT_LANDED.txt", ("# NOT_LANDED: files of this package kept on the dev box (" + S + ", " + F + ")\n# path | bytes | sha256 | why\n"
                                       + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_trigger_probe3.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2

ROW = ("**H-CRAWL-06-PROBE3** (2026-10-29)",
       "Precursor of H-CRAWL-06, continued: at a fixed per-token PCA-16 readout (linear and 64-unit MLP), the frozen DINOv3-L/16 field"
       " at 8x20 (G8) carries more go/stay trigger than the same field at 4x10 (G4 = its exact 2x2 mean). RESOLUTION HELPS iff"
       " dAUC(G8+E " + M + " G4+E) >= 0.03 with CI > 0 in one of four cells (P-STOP / P-MOVE x linear / MLP); DOES NOT HELP iff all"
       " four < 0.01.",
       "**MEASURED 2026-10-29, TRAIN 12,000 + 12,000 windows, fold-mean AUC, episode-cluster bootstrap B = 2,000 (answers another draw"
       " of episodes; blind to training and probe-fit variance); card frozen 03:37:08 (sha256 f73b8a5a\u2026) before any 8x20 feature.**"
       " Verdict by the committed rule: INCONCLUSIVE. All four cells are negative and no CI upper bound reaches +0.01: P-STOP linear "
       + M + "0.042 [" + M + "0.060, " + M + "0.022], MLP " + M + "0.022 [" + M + "0.049, +0.002]; P-MOVE linear " + M + "0.006 ["
       + M + "0.013, +0.000], MLP " + M + "0.030 [" + M + "0.042, " + M + "0.019]. Both P-MOVE cells are excluded: the G8+E"
       " label-shuffle control reads 0.5120 [0.5017, 0.5226]. G4 = mean2x2(G8) exactly (max abs diff 0.0); the content mutation moves"
       " the number (C5b). Flat per-token readouts sit below probe 1's whole-vector V+E (G4+E " + M + " V+E " + M + "0.030 [" + M
       + "0.049, " + M + "0.010], P-STOP), and the best post-freeze flat readouts only reach it. Beside: on P-STOP green-light windows"
       " the images hurt at both grids (G4+E " + M + " E " + M + "0.116 [" + M + "0.191, " + M + "0.005]; G8+E " + M + " G4+E "
       + M + "0.000). A 16x40 probe under this readout is not worth box-2 time. Next lever, running: trigger probe 4 (attention and"
       " max-pool readouts over the same tokens, frozen 04:48:11, sha256 a8f5d785\u2026).",
       "MEASURED (INCONCLUSIVE by the rule; no cell near HELPS)",
       "`" + PK + "RESULTS.md`")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-CRAWL-06-PROBE3**" not in g and b"\r" not in g
add = ("| " + " | ".join(ROW) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("Trigger probe 3 (H-CRAWL-06-PROBE3): 8x20 vs 4x10 DINOv3 tokens at a fixed per-token readout, INCONCLUSIVE by the rule;"
                " all four cells negative, none near HELPS (P-STOP linear -0.042 [-0.060, -0.022]); the P-MOVE cells are excluded by a"
                " label-shuffle control at 0.5120. Finer tokens do not help a flat readout; the readout is the binding term. A 16x40 box-2"
                " probe is not worth running under it; probe 4 (attention / max-pool readout) is running. Package"
                " raw/2026-10-29-trigger-probe3/ (" + str(n) + " files; NOT_LANDED.txt lists " + str(len(notl)) + " kept on E: / F:);"
                " register row.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
