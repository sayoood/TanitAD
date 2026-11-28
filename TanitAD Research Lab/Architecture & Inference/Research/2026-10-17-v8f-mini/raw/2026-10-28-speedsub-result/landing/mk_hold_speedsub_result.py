"""Hold: RESULT_V8F_SPEEDSUB (PASS, a screening result) + the package raw/2026-10-28-speedsub-result/ (SUBREAD.json, the independent re-derivation, the ten eval JSONs, the family
JSONs / md / out, the run logs, the audit, this builder) + register row H-V8F-SPEEDSUB-1 (mods-rw, line count unchanged).
python mk_hold_speedsub_result.py <live tip> [--out <hold dir>]
The default hold dir is C:/Users/Admin/qland/hold_speedsub_result/ (refused if it exists). The Master Mind lands it; this script only writes the hold, and reads the repo
only through the mirror (git show), never through a working tree."""
import hashlib
import os
import re
import subprocess
import sys

args = sys.argv[1:]
TIP = args[0]
HOLD = "C:/Users/Admin/qland/hold_speedsub_result/"
if "--out" in args:
    HOLD = args[args.index("--out") + 1].replace("\\", "/").rstrip("/") + "/"
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
HERE = A + "dagland/speedsub_result/"
W = "F:/TanitAD-scratch/speedsub/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedsub-result/"
H = HOLD
assert not os.path.exists(H), H

FORBID = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
HFTOK = re.compile(rb"hf_[A-Za-z0-9]{8,}")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not HFTOK.search(b), "hf_ token in " + rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- the RESULT
res = rd(HERE + "RESULT_V8F_SPEEDSUB.md")
assert b"\r" not in res and res.startswith(b"# RESULT_V8F_SPEEDSUB: ") and b"Outcome **PASS**" in res
assert blob(V8 + "RESULT_V8F_SPEEDSUB.md") is None, "RESULT already at the tip"
for w in FORBID:
    assert w not in res.lower(), w
put("stage/" + V8 + "RESULT_V8F_SPEEDSUB.md", res)

# ---- the package
files = {}   # repo-relative (under PK) -> (source path, expected sha256 or None)
READ_PIN = {"subread/SUBREAD.json": ("read/SUBREAD.json", "1fe7fcb366e8123721d82891408c6d3c15948745173deb87f1ddeb5a4c53826d"),
            "subread/OUTCOME.txt": ("read/OUTCOME.txt", "178778b3294c13ac753756bc16043ac86115b43b4c08e33da5d8ec1701ba1ecc")}
for src, (dst, h) in READ_PIN.items():
    files[dst] = (W + src, h)
files["read/REDERIVE.json"] = (HERE + "REDERIVE.json", "f72573fc7eecff2d8b79a283761cc2a284d166c7e24415fa0917221f61cf8ee6")
files["read/rederive_sub.py"] = (HERE + "rederive_sub.py", None)
files["read/rederive_sub.out"] = (HERE + "rederive_sub.out", None)
for s in (0, 1):
    for m in ("own", "identity", "gt", "ego", "egoshuf"):
        files[f"eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.json"] = (W + f"eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.json", None)      # JSON only; the .npz stay on F:
    files[f"families/longlat_s{s}.json"] = (W + f"families/longlat_s{s}.json", None)
    files[f"families/longlat_s{s}.out"] = (W + f"families/longlat_s{s}.out", None)
    for a in ("ego", "gt"):
        for ext in ("json", "md", "out"):
            files[f"families/pairfam_{a}_s{s}.{ext}"] = (W + f"families/pairfam_{a}_s{s}.{ext}", None)
for f in ("run.out", "evals_s0.log", "evals_s1.log", "preflight_s0.log", "subread.log"):
    files["logs/" + f] = (W + f, None)
files["logs/speedsub_after_t1.log"] = ("F:/TanitAD-scratch/speedsub_after_t1.log", None)
files["run/run_with_taniteval.py"] = (W + "run_with_taniteval.py", None)
for f in ("NUMBERS_AUDIT.md", "gen_result.py", "audit_cover.py", "REGISTER_ROW.md", "mutate_gen_check.py", "gen_mutation_check.out"):
    files["audit/" + f] = (HERE + f, None)
files["landing/mk_hold_speedsub_result.py"] = (HERE + "mk_hold_speedsub_result.py", None)

# the originals the RESULT cites that are already in the repo: identical bytes, not re-staged
DRAFT = V8 + "raw/2026-10-28-speedsub-draft/"
for repo_rel, src, h in ((DRAFT + "run/run_speedsub.sh", A + "dagland/speedsub/run/run_speedsub.sh", "8255f69f7e094abe609083ab69fa23dd1b2359da8570243122b14e435c3413c0"),
                         (DRAFT + "run/make_tree.sh", A + "dagland/speedsub/run/make_tree.sh", "7c523217afd8e25ed9b8629c0c5489d2e0053863ee85383b3d5ecc8480f430fd"),
                         (DRAFT + "families/families_longlat_sub.py", "F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/families_longlat_sub.py", "f981ea7a44d86ee12528e29ca2e0ea67947c557e2a8aee0bf1cd16d935a316a8"),
                         ("stack/scripts/v8f_mini_eval_speedread.py", "F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/v8f_mini_eval_speedread.py", "a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca"),
                         ("Project Steering/PREREG_V8F_SPEEDSUB.md", None, "1951a240196330401abe54416c5c224026f114cb3605d8be0dce7e97d127c432"),
                         (V8 + "raw/2026-10-28-speedread/read/READ.json", "F:/TanitAD-scratch/speedread/read/READ.json", "46543c392ee412c76edab48ec95c993e11ba621cc43188ee8e25f7b966d4579f"),
                         (V8 + "raw/2026-10-28-speedread/fit/fit.json", "F:/TanitAD-scratch/speedread/fit/fit.json", "c2e51ac204e5284dac5546ed80f03a3762f569f482a3e498c9c9ae12b4f04ae7")):
    bl = blob(repo_rel)
    assert bl is not None and sha(bl) == h, (repo_rel, None if bl is None else sha(bl))
    if src is not None:
        assert sha(rd(src).replace(b"\r\n", b"\n")) == h, (repo_rel, src)

SKIP = (".pt", ".npz", ".npy", ".pyc")
n = 0
lfn = []
sums_out = []
for rel, (src, want) in sorted(files.items()):
    assert not src.endswith(SKIP), src
    b = rd(src)
    if want is not None:
        assert sha(b) == want, (rel, sha(b))
    # every staged source is cited in the RESULT's provenance table by its sha256 (the audit files, the register row and the builder are the package's own and are not)
    if not rel.startswith(("audit/", "landing/")):
        assert sha(b).encode() in res, ("source sha256 not in the RESULT's provenance table", rel)
    assert b and len(b) < 400_000, (rel, len(b))
    if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
        b = b.replace(b"\r\n", b"\n")
        assert b"\r" not in b, rel
        lfn.append(rel)
    if rel.endswith((".py", ".md")):
        assert b"\r" not in b, rel
    assert blob(PK + rel) is None, rel
    put("stage/" + PK + rel, b)
    sums_out.append(f"{sha(b)} *{rel}")
    n += 1
if lfn:
    b = ("CRLF -> LF on banking (content otherwise unchanged; the RESULT's provenance table lists the source sha256):\n" + "\n".join(lfn) + "\n").encode()
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
ev_old = re.search(r"<!-- EVID-OLD: (.*?) -->", rr).group(1)
ev_new = re.search(r"<!-- EVID-NEW: (.*?) -->", rr).group(1)
assert status.startswith("**PASS (MEASURED 2026-10-28") and status.endswith("Was:") and "\n" not in status and "|" not in status
for w in FORBID:
    assert w not in status.lower().encode(), w
ID = b"| **H-V8F-SPEEDSUB-1** (2026-10-28) |"
assert g.count(ID) == 1
i = g.index(ID)
j = g.index(b"\n", i)
row = g[i:j]
OLDHEAD = b"| **PRE-REGISTERED, NOT RUN (2026-10-28).**"
TAIL_OLD = b"| " + ev_old.encode() + b" | `Project Steering/PREREG_V8F_SPEEDSUB.md` |"
TAIL_NEW = b"| " + ev_new.encode() + b" | `Project Steering/PREREG_V8F_SPEEDSUB.md` |"
assert row.count(OLDHEAD) == 1 and row.endswith(TAIL_OLD) and row.count(TAIL_OLD) == 1, row[-160:]
row2 = row.replace(OLDHEAD, b"| " + status.encode("utf-8") + b" " + OLDHEAD[2:])
row2 = row2[: -len(TAIL_OLD)] + TAIL_NEW
assert row2.startswith(ID) and row2.count(b"\n") == 0
assert row2.count(b"|") == row.count(b"|"), (row2.count(b"|"), row.count(b"|"))      # the pipes of the table row are unchanged
# the old status text survives after "Was:" byte for byte
assert (b"Was: " + OLDHEAD[2:]) in row2
g2 = g[:i] + row2 + g[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
for w in FORBID:
    assert w not in row2[: row2.index(b"Was: ")].lower(), w
put("mods-rw/" + GC, g2)

# ---- commit message
MSG = ("RESULT_V8F_SPEEDSUB: PASS (H-V8F-SPEEDSUB-1), a screening result at T1-class open loop; all 27 reader control flags PASS and every row recomputed from the per-window dumps."
       " The banked ego-only read-out mlp_ego in the five speed cells O reads cuts X_s's own-code ADE6 by -0.1290 [-0.2107, -0.0548] m (s0) and -0.0686 [-0.1151, -0.0232] m (s1),"
       " separated-better and beyond the -0.06035 m bar at both seeds (s1 by 0.0083 m only); the egoshuf regression arm reads +6.7181 / +6.4923 m, separated-worse; the gt ceiling (T0, a control)"
       " is -0.7164 / -0.6676 m, CAPTURE 0.1801 / 0.1028. Base M2b-mixo, not DAG-OACC; paired episode-cluster bootstrap answers the episode question only, no replicate arm."
       " Four families reported per seed: LONGITUDINAL speed / along-track / accel MAE separated-better at both seeds, DK and anti-echo NOT COMPUTED (UNAVAILABLE, n = 0; work item),"
       " LATERAL and TACTICAL pointers, STRATEGIC exactly 0 by construction. Post-hoc beside rows, not registered: the plan rule-violation rate rises under ego (separated-worse at both seeds)"
       " and the 81 stop windows read worse under ego (point values). Declared open channel: the PASS was read with the logged limit lane (D-SPEEDBAND-EGO-ECHO-1, the bucket of the ego's"
       " own realised future max speed). Attribution of the read-out's skill (FINDING_1, TRAIN fold 0, mean hold-out RMSE in SD units): full 0.24751, no nav 0.26313, no limit 0.28032,"
       " p10f + frame_ok only 0.30849: the skill survives without nav, not without limit; the ADE6 attribution is unmeasured. Committed next: the T1-kin zero-training substitution on the"
       " DAG-OACC / DAG-ORC base under its own PREREG; the T1-kin lane uses --limit causal, so the follow-up refits the read-out on the causal bucket (PREREG_V8F_SPEEDSUB_T1, draft, not landed)."
       f" Package raw/2026-10-28-speedsub-result/ ({n} files: SUBREAD.json, the independent re-derivation, eval JSONs, family outputs, run logs, audit). Register row updated.\n\n"
       "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n")
for w in FORBID:
    assert w not in MSG.lower().encode(), w
assert MSG.endswith("Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n")
put("msg.txt", MSG.encode("utf-8"))

# ---- last checks on everything staged
bad = []
for dp, _, fs in os.walk(H):
    for f in fs:
        p = os.path.join(dp, f)
        b = rd(p)
        if HFTOK.search(b):
            bad.append(("hf_", p))
        if p.endswith((".md", ".py", ".sh")) and b"\r" in b:
            bad.append(("CR", p))
        if p.endswith(SKIP):
            bad.append(("binary", p))
assert not bad, bad
print("hold written", H, n, "package files; register line count", len(g.split(b"\n")), "unchanged")
