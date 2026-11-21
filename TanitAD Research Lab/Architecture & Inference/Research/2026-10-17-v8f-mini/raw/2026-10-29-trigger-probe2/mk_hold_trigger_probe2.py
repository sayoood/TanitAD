"""Hold: trigger probe 2 (what nav encodes; the trunk with nav neutralised; motion cues; card frozen before data), banked as
raw/2026-10-29-trigger-probe2/; register: a prefix on H-CRAWL-06-PROBE and a new row H-CRAWL-06-PROBE2 (mods-rw, superset).
scratch/, aside/, caches and files over 3 MB stay on E: (NOT_LANDED.txt).  python mk_hold_trigger_probe2.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_trigger_probe2/"
S = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe2/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-29-trigger-probe2/"
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
    m = re.match(r"([0-9a-f]{64})\s+\d+\s+(\S.*?)\s*$", line)
    if m:
        pins[m.group(2)] = m.group(1)
assert pins.get("PREREG_FROZEN.md", "").startswith("d9af82be") and pins.get("RESULTS.md", "").startswith("c1a3e615"), "manifest"

n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins:
            assert sha(b) == pins[rel], rel
        why = ("scratch: feature memmaps (6.8 GB)" if rel.startswith("scratch/") else "aside: moved-aside files" if rel.startswith("aside/") else
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
put("stage/" + PK + "mk_hold_trigger_probe2.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2


for p in ("stack/scripts/build_v9_labels.py", "stack/scripts/build_v8_speed_max_labels.py"):
    assert blob(p) is not None, p
SRC = "`" + PK + "RESULTS.md`"
M = "\u2212"
NEW = ("**H-CRAWL-06-PROBE2** (2026-10-29)",
       "Follow-up of H-CRAWL-06-PROBE on the same TRAIN windows: (1) what the nav token encodes; (2) with nav neutralised (donor nav;"
       " the model has no null-nav path), the trunk state h_O still holds go/stay information beyond ego (dAUC(Z_nn+E - E) >= 0.05:"
       " VISION-THROUGH-TRUNK; < 0.02: NAV-ONLY); (3) the model's 4 frames or the frame difference carry the trigger beyond the current"
       " frame (dAUC(max(V4, Vdiff)+E - V+E) >= 0.03: MOTION; < 0.01: none). Primary metric the mean of per-fold AUCs.",
       "**MEASURED 2026-10-29, zero training, card frozen at 23:11:02 before data (PREREG_FROZEN.md sha256 d9af82be\u2026), 5 episode folds,"
       " 12,000 windows per population, episode-cluster bootstrap B = 2,000.** (1) Nav is built from turns of the WHOLE recording on the"
       " ego's own future path (`stack/scripts/build_v9_labels.py:233`); N+E - E +0.036 [0.010, 0.062] (P-STOP), +0.014 [0.007, 0.022]"
       " (P-MOVE). The larger ego-future channel is the speed-ceiling input (declared oracle, `stack/scripts/build_v8_speed_max_labels.py`:"
       " the max of the ego's own realised speed over [t0+2 s, +6 s], one value per clip, the PI's 2026-09-10 ruling): L+E - E +0.072"
       " [0.057, 0.087] / +0.053 [0.044, 0.062]. (2) P-MOVE dAUC(Z_nn+E - E) +0.117 [0.105, 0.129]: VISION-THROUGH-TRUNK by the letter;"
       " P-STOP +0.134 [0.110, 0.155] but control C2 read 0.454 [0.418, 0.491], so P-STOP is NOT EVALUABLE as frozen. Scrambling nav removes"
       " only +0.040 / +0.032 of Z+E. Post-freeze, with nav AND the speed ceiling scrambled (Z_nl): Z_nl+E - V+E +0.006 [" + M + "0.016,"
       " 0.027] on P-STOP, and on eval-141 (no refit) Z_nl+E - E +0.072 [" + M + "0.114, 0.227] / +0.019 [" + M + "0.040, 0.080]. (3)"
       " NO-MOTION in both populations: V4+E - V+E " + M + "0.005 [" + M + "0.011, +0.001] / " + M + "0.000 [" + M + "0.003, +0.002];"
       " the frame difference is lower; eval-141 " + M + "0.003 / +0.006. Strata: the image gain appears only on red-light clips (+0.099),"
       " none on green-light or lead-departure clips. Reading: Z_nn still holds the speed-ceiling oracle, so the committed"
       " VISION-THROUGH-TRUNK label over-reads; once both ego-future channels are scrambled the trunk's go/stay gain equals the frozen"
       " image features' own small gain and is not distinguishable from 0 on eval. The go/stay trigger is weakly present in what v8F"
       " sees, and absent for green lights and departing leads: the lever is perception, and a loss-only stop/go head on this trunk would"
       " learn the oracle channels. Caveats: one checkpoint (DAG-OACC s0), TRAIN windows seen in training, eval-141 P-STOP n = 101.",
       "MEASURED (V4 NO-MOTION; Z_nn VISION-THROUGH-TRUNK by the letter on P-MOVE, P-STOP not evaluable; post-freeze the trunk's edge is"
       " the speed-ceiling and nav oracles)")
OLD_ID = "| **H-CRAWL-06-PROBE** (2026-10-28) | "
PRE = ("**Refined by H-CRAWL-06-PROBE2 (2026-10-29): most of the trunk's oracle edge is the speed-ceiling input (ego-future, declared,"
       " the PI's 2026-09-10 ruling); the nav token is the smaller part.** ")
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-CRAWL-06-PROBE2**" not in g and b"\r" not in g
lines = g.split(b"\n")
idx = [i for i, l in enumerate(lines) if l.startswith(OLD_ID.encode("utf-8"))]
assert len(idx) == 1
cells = lines[idx[0]].decode("utf-8").split(" | ")
assert cells[1].startswith("Precursor of H-CRAWL-06") and cells[2].startswith("**MEASURED 2026-10-28")
cells[2] = PRE + cells[2]
lines[idx[0]] = " | ".join(cells).encode("utf-8")
ng = b"\n".join(lines)
assert ng.replace(PRE.encode("utf-8"), b"") == g
add = ("| " + " | ".join(NEW[:3] + (NEW[3], SRC)) + " |\n").encode("utf-8")
for w in BANNED:
    assert w not in add.lower() and w not in PRE.encode("utf-8").lower(), w
assert add.count(b" | ") == 4
put("mods-rw/" + GC, ng + add)
put("msg.txt", ("Trigger probe 2 (frozen first): nav comes from turns of the whole recording on the ego's future path; the larger"
                " ego-future channel is the declared speed-ceiling input (L+E - E +0.072 / +0.053). With nav neutralised the trunk keeps"
                " its edge (P-MOVE +0.117; P-STOP not evaluable, control C2), but with nav and ceiling both scrambled its gain equals the"
                " frozen image features' and is not distinguishable from 0 on eval-141. Motion over the model's 4 frames adds nothing."
                " The lever is perception, not a stop/go head on this trunk. Package raw/2026-10-29-trigger-probe2/ (" + str(n)
                + " files; NOT_LANDED.txt lists " + str(len(notl)) + " kept on E:); register: prefix on H-CRAWL-06-PROBE, new row"
                " H-CRAWL-06-PROBE2.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
