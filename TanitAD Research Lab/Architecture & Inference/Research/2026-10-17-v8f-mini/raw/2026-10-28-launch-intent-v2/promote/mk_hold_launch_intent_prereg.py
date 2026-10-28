"""Hold: PREREG_V8F_LAUNCH_INTENT promoted (PRE-REGISTERED, NOT RUN), its v2 package raw/2026-10-28-launch-intent-v2/ (the reader with
its tests and mutation pass, the enumeration tools, the dry-run evidence, the draft it was promoted from and the promotion diff), and
register row H-V8F-LAUNCH-INTENT-1 appended after H-V8F-SPEEDSUB-T1-1 (mods; every existing byte kept).
python mk_hold_launch_intent_prereg.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/dagland/launch_intent_v2/"
H = "C:/Users/Admin/qland/hold_launch_intent_prereg/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-launch-intent-v2/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# the flag the PREREG reads is on the tip (b09ac89f), and the redesign spec is banked beside it
for rel, h in {"stack/scripts/v8f_mini_train.py": "0d708b3cb841", "stack/tanitad/models/v8f_mini.py": "b337ccecd3ab"}.items():
    assert sha(blob(rel)).startswith(h), rel
assert blob(V8 + "raw/2026-10-28-launch-intent-flag/MM_REDESIGN_2026-10-28.md") is not None

PR = "Project Steering/PREREG_V8F_LAUNCH_INTENT.md"
p = rd(D + "promote/PREREG_V8F_LAUNCH_INTENT.md")
assert sha(p).startswith("469bac1f3918") and blob(PR) is None
assert b"PRE-REGISTERED, NOT RUN (2026-10-28)" in p and b"**OPEN" not in p
put("stage/" + PR, p)

DRAFT = rd(D + "PREREG_V8F_LAUNCH_INTENT.md")
assert sha(DRAFT).startswith("aec8156a4a24")
n = 0
FILES = {"PREREG_DRAFT_v2.md": DRAFT}
for f in ("PREREG_V2_vs_V1.diff", "NOTES.md", "MANIFEST.txt", "enumerate_cells_li.out", "enumerate_cells_li_mutation.txt",
          "read/launch_intent_read.py", "read/test_launch_intent_read.py", "read/mutation_launch_intent_reader.py",
          "read/ref/rep_s0_speedcells.json", "tools/enumerate_cells_li.py", "tools/enumerate_mutation.py", "tools/enumerate_v1_vs_v2.diff",
          "evidence/dry_run.json", "evidence/mutation_reader.json", "evidence/pytest_final.txt", "evidence/static_check.txt"):
    FILES[f] = rd(D + f)
for f in ("PROMOTION_DIFF.diff", "promote_prereg.py", "static_check_promoted.txt", "pytest_promoted.txt", "mk_hold_launch_intent_prereg.py"):
    FILES["promote/" + f] = rd(D + "promote/" + f)
assert sha(FILES["read/launch_intent_read.py"]).startswith("5e950dcff299")
assert b"LAUNCH-INTENT-READ-STATIC-OK" in FILES["promote/static_check_promoted.txt"] and b"87 passed" in FILES["promote/pytest_promoted.txt"]
for f, b in FILES.items():
    assert blob(PK + f) is None and len(b) < 400_000, f
    put("stage/" + PK + f, b)
    n += 1

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"H-V8F-LAUNCH-INTENT-1" not in g
last = g.rstrip(b"\n").split(b"\n")[-1]
assert last.startswith(b"| **H-V8F-SPEEDSUB-T1-1** (2026-10-28) |"), last[:80]
ROW = ("| **H-V8F-LAUNCH-INTENT-1** (2026-10-28) | On the DAG-OACC base, O's launch-sample along-track weight (`--launch-intent-w"
       " 1.58926`: GT-launch TRAIN samples, v0 < 6 m/s and GT v(6 s) \u2265 v0 + 1 m/s, hold one third of the along-track weight on"
       " knots 1\u201312; the subset is label-side and nothing new enters at inference) moves every crawl row (yaw-rate MAE, curvature"
       " MAE, L2 curvature MAE at GT \u2265 5.2, M1 crawl share) by at least its bar at T1-kin, in both same-seed one-lever pairs"
       " (A_s \u2212 DAG-OACC_s and A_rep,s \u2212 REP_s) at both seeds, with no guard firing | **PRE-REGISTERED, NOT RUN (2026-10-28).**"
       " The LAUNCH-INTENT branch of `PREREG_V8F_DAG_ORC` section 8 (NEXT-CRAWL), routed here by its Amendment 3 after arm A read"
       " HARMS. The Master Mind redesigned the bar and the arms before promotion and before any LAUNCH-INTENT number existed"
       " (`raw/2026-10-28-launch-intent-flag/MM_REDESIGN_2026-10-28.md`): the draft's bar 2 \u00d7 max(F, RF0, RF) exceeded the whole"
       " CTRL \u2212 P10 gap on every crawl row (bar \u00f7 gap 1.3\u20139.1), so no fix could clear it. The bar is 2 \u00d7 max(F, RF0)"
       " (yaw-rate 1.1070, curvature 0.0526, L2 0.0136, M1 0.0096); RF is reported beside, and the training-run question is carried"
       " by requiring each per-seed predicate in both pairs. Arms: A_0, A_1, A_rep,0, A_rep,1, four box-2 runs (\u2248 2.8 h ESTIMATED;"
       " the PI's go, *\"Yes, queue it\"*). CTRL = DAG-OACC (`67265ec1\u2026` / `90b01474\u2026`); REP = DAG-ORC r2's replicates, reused"
       " under a licence the reader checks. Paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Outcomes: the DAG-ORC"
       " outcome table over the per-seed booleans (524,288 cells, each matching exactly one row). If the crawl does not move,"
       " NEXT-LAUNCH: M-LAUNCH moved \u2192 LOOP-STATE (zero training); not moved \u2192 LAUNCH-INTENT-HI (W 4.17852, its own PI go). Dose"
       " fallback B (W 0.726175) only after HARMS or NOT ADOPTED, its own PI go. Trainer flag landed at `b09ac89f`; reader built and"
       " self-tested (87 tests GREEN, 245 / 245 mutants killed, dry run reproduces DAG-ORC's banked tables with 0 differences), not"
       " run. | PRE-REGISTERED | `Project Steering/PREREG_V8F_LAUNCH_INTENT.md` |\n")
for w in BANNED:
    assert w not in ROW.encode("utf-8").lower(), w
put("mods/" + GC, g + ROW.encode("utf-8"))
put("msg.txt", ("PREREG_V8F_LAUNCH_INTENT promoted: PRE-REGISTERED, NOT RUN (H-V8F-LAUNCH-INTENT-1). The Master Mind's redesign before any"
                " LAUNCH-INTENT number: bar 2 x max(F, RF0) with RF beside (the draft's 2 x max(F, RF0, RF) exceeded every crawl row's whole"
                " CTRL - P10 gap), two same-seed one-lever pairs per seed (A_s - DAG-OACC_s, A_rep,s - REP_s), each per-seed predicate in both"
                " pairs. Section 0 items 1, 4, 6, 8 confirmed at promotion; nothing else changes (PROMOTION_DIFF.diff). Box 2: four runs on"
                " the PI's go. Package raw/2026-10-28-launch-intent-v2/ (%d files): the reader (87 tests GREEN on the promoted text, 245 / 245"
                " mutants killed, static check OK with the promoted sha256 pin and REFUSED with a wrong one), the enumeration tools, the dry"
                " run, the draft and the promotion diff.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode("utf-8"))
print("hold written", H, n)
