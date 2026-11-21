"""Hold: RESULT_V8F_SPEEDREAD (NOT IN INPUT) + the package raw/2026-10-28-speedread/ (the run's read JSON, fit JSON, extract record, the 14 per-mode
eval JSONs, logs, the three stops, the pull record, the run scripts) + register row H-V8F-SPEEDREAD-1 (HYPOTHESIS -> MEASURED).
python mk_hold_speedread_result.py <live tip>
The hold dir is C:/Users/Admin/qland/hold_speedread_result/ unless SPEEDREAD_HOLD_DIR names another (a proof build in scratch); the 260-character
path guard is checked against the qland path in both cases."""
import hashlib
import json
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "-c", "core.autocrlf=false"]
W = "F:/TanitAD-scratch/speedread/"
RUN = "E:/Projects/TanitAD-artifacts/speedread_run/"
HERE = "E:/Projects/TanitAD-artifacts/dagland/speedread_result/"
QL = "C:/Users/Admin/qland/hold_speedread_result/"
H = os.environ.get("SPEEDREAD_HOLD_DIR", QL)
H = H if H.endswith("/") else H + "/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedread/"
assert not os.path.exists(H), H
PINS = {"read/READ.json": "46543c392ee412c76edab48ec95c993e11ba621cc43188ee8e25f7b966d4579f",
        "fit/fit.json": "c2e51ac204e5284dac5546ed80f03a3762f569f482a3e498c9c9ae12b4f04ae7",
        "train/extract.json": "639a4a8a596a76cb12ee13d4872dd9a1481721f78a9c4285f52fced24ba2b973"}
LIMIT = 400_000


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(QL + rel) < 260 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- the result ----
res = rd(HERE + "RESULT_V8F_SPEEDREAD.md")
assert b"\r" not in res and blob(V8 + "RESULT_V8F_SPEEDREAD.md") is None
assert PINS["read/READ.json"].encode() in res and b"**NOT IN INPUT**" in res and b"raw/2026-10-28-speedread/" in res
put("stage/" + V8 + "RESULT_V8F_SPEEDREAD.md", res)

# ---- the run's pinned sources ----
for rel, h in PINS.items():
    assert sha(rd(W + rel)) == h, rel
R = json.load(open(W + "read/READ.json", encoding="utf-8"))
assert R["outcome"] == "NOT IN INPUT" and R["controls"]["C9_perm_no_skill"] is True and R["controls"]["C10_train_parity"] is True
INSTR = "fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe"
assert sha(blob("stack/scripts/v8f_mini_eval_speedread.py")) == INSTR      # the instrument at the live tip is the one the run pinned
assert sha(rd(W + "tree_f3971165/stack/scripts/v8f_mini_eval_speedread.py")) == INSTR
assert sha(rd(W + "fit/readout.pt")) == json.load(open(W + "fit/fit.json", encoding="utf-8"))["readout_sha256"]

# ---- the package ----
files = {}
for rel in ("read/READ.json", "read/OUTCOME.txt", "fit/fit.json", "train/extract.json"):
    files[rel] = W + rel
for s in (0, 1):
    for f in sorted(os.listdir(W + f"eval_s{s}")):
        if f.startswith("eval_") and f.endswith(".json"):
            files[f"eval_s{s}/{f}"] = W + f"eval_s{s}/{f}"
assert sum(1 for k in files if k.startswith("eval_s")) == 14
for f in ("run.out", "extract.log", "fit.log", "evals_s0.log", "evals_s1.log", "read.log"):
    files["logs/" + f] = W + f
for f in ("pull.out", "pull.launch.out", "verified.txt"):
    files["pull/" + f] = W + f
for stop, fs in (("stop1", ("extract_no_taniteval.log", "run_stop1.out", "run_stop1_orig.out")),
                 ("stop2", ("extract_ego23.log", "run_stop2.out", "v8f_mini_eval_speedread_d5545751.py")),
                 ("stop3", ("evals_s0_no_grid.log", "run_stop3.out"))):
    for f in fs:
        files[f"stops/{stop}/{f}"] = W + f"{stop}/{f}"
for f in ("run_speedread.sh", "run_with_taniteval.py", "pull_speedread.sh", "wake_speedread.sh", "pull_speedread.cmd", "run_speedread.cmd"):
    files["run/" + f] = RUN + f
FAM = "F:/TanitAD-scratch/speedread_result/families/"
FAMPINS = {
    "longlat_s0.json": "dbf5c339c25d9851f99c86f8a6a4eb11959e7925e3c3a5a96e3703a010b5f29d",
    "longlat_s1.json": "291364f515426871719e2a2f52370631dc42245408ddeb1599e952da3cff5a75",
    "pairfam_gt_s0.json": "b82353bcf4224d172c21aff8edb953d8f6ee7cf799cd6a2d223b985279937274",
    "pairfam_gt_s1.json": "570a6eb2cc0af786387b07258bbfc4dd407bcba10002f1dd69c2b5e1f0a8b1ee",
    "pairfam_read_s0.json": "8a56d116c924b2c6a6e7d046c126525ba60401151aa910f9c956450d84dbc24a",
    "pairfam_read_s1.json": "70051331c8f7293d57d0c368c722130b275a099067b37c6101cf8aec603a2e55",
    "subctl_own_vs_banked_s0.json": "eb3c0c3fcb14a798bf91c3e70adeb66c57081b921237a6d5ff1ea86127664615",
    "subctl_own_vs_banked_s1.json": "6421502174eb0ea4a7693b5d7b5cfb09f2a51aadb96bda4ee82548bb598b9431"
}
famfiles = sorted(f for f in os.listdir(FAM) if f.endswith((".json", ".md", ".out")))
assert len(famfiles) == 21, famfiles                          # 4 pairfam + 2 subctl runs x (json, md, out), 2 longlat json, run_families.out
for f in famfiles:
    files["families/" + f] = FAM + f
for f, h in FAMPINS.items():
    assert sha(rd(FAM + f)) == h, f
files["families/run_families.sh"] = HERE + "gen/run_families.sh"
files["families/families_longlat.py"] = HERE + "gen/families_longlat.py"
files["landing/mk_hold_speedread_result.py"] = HERE + "mk_hold_speedread_result.py"
assert sha(rd(W + "stop2/v8f_mini_eval_speedread_d5545751.py")).startswith("d5545751")
n = 0
for rel, src in sorted(files.items()):
    b = rd(src)
    assert b and not rel.endswith((".pt", ".npz", ".npy")) and len(b) < LIMIT, (rel, len(b))
    assert blob(PK + rel) is None, rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    put("stage/" + PK + rel, b)
    n += 1

# ---- the register row ----
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
ID = b"| **H-V8F-SPEEDREAD-1**"
OLD = b"| **PRE-REGISTERED, NOT RUN (2026-10-27).** Promoted from the draft"
assert g.count(ID) == 1 and g.count(OLD) == 1
i = g.index(ID)
j = g.index(b"\n", i)
row = g[i:j]
assert OLD in row


def _walk(node):
    if isinstance(node, dict):
        if "delta" in node and "separated" in node:
            yield node
            return
        for v in node.values():
            yield from _walk(v)


FAMSEP = []
for _s in (0, 1):
    _q = json.load(open(FAM + f"longlat_s{_s}.json", encoding="utf-8"))
    _p = json.load(open(FAM + f"pairfam_read_s{_s}.json", encoding="utf-8"))
    FAMSEP.append(sum(1 for c in _q["contrasts"]["read-own"].values() if c["separated"]) + sum(1 for sect in ("tactical", "strategic") for c in _walk(_p[sect]) if c["separated"]))
    assert not any(_q["contrasts"]["read-own"][k]["separated"] for k in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2"))


def f(v):
    return format(v, "+.4f").replace("-", "−")


def f2(v):
    return format(v, ".4f").replace("-", "−")


def ci(s, m):
    r = R["rows"][s][m]
    return f"{f(r['delta'])} [{f(r['lo'])}, {f(r['hi'])}]"


new = (f"| **NOT IN INPUT (MEASURED 2026-10-28, Amendment 1 in force; all 19 control flags PASS).** PRIMARY = the ridge read-out (`lin`, selected on held-out TRAIN"
       f" fold 0 before any eval batch): ADE6 read − own {ci('0', 'read')} / {ci('1', 'read')} m against −F = −{R['F_m']:.4f} m: not met, and not separated-better at either"
       f" seed. ALT (the vision + ego MLP) {ci('0', 'alt')} / {ci('1', 'alt')} m, separated-worse. CAPTURE {f2(R['rows']['0']['capture'])} / {f2(R['rows']['1']['capture'])};"
       f" the `gt` ceiling (T0) reads {f(R['rows']['0']['gt']['delta'])} / {f(R['rows']['1']['gt']['delta'])} m. T1-class open loop; paired episode-cluster bootstrap,"
       f" B = 2,000, 141 episodes. Scope: not recoverable by a ridge or a 2×256 MLP from the pooled 4-frame summary on unseen clips. Beside: the ego-only MLP has the"
       f" lowest fold-0 and eval cell RMSE and its ADE6 substitution was not run. Four families read beside, deciding nothing (RESULT §4): separated cells of the read {FAMSEP[0]} / {FAMSEP[1]} (s0 / s1);"
       f" no LONGITUDINAL row separated at either seed; head and S rows exactly +0.0000. Committed next: perception (PREREG_V8F_LORA on the A40), blocked because the A40 is not"
       f" allocated; nothing on box 2 on this line. `RESULT_V8F_SPEEDREAD.md`, `raw/2026-10-28-speedread/read/READ.json`. Was: ").encode("utf-8") + OLD[2:]
g2 = g.replace(OLD, new)
i = g2.index(ID)
j = g2.index(b"\n", i)
row = g2[i:j]
END = b"| HYPOTHESIS | `Project Steering/PREREG_V8F_SPEEDREAD.md` |"
assert row.endswith(END), row[-120:]
g2 = g2[:i] + row[: -len(END)] + b"| MEASURED | `Project Steering/PREREG_V8F_SPEEDREAD.md` |" + g2[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
assert g2.count(b"\r") == g.count(b"\r")
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in new.lower() and w not in res.lower(), w
put("mods-rw/" + GC, g2)

# ---- the commit message ----
r0, r1 = R["rows"]["0"], R["rows"]["1"]
a = lambda v: format(v, "+.4f")
put("msg.txt", (f"RESULT_V8F_SPEEDREAD: NOT IN INPUT (H-V8F-SPEEDREAD-1), all 19 control flags PASS, read under Amendment 1. The ridge read-out of T's five speed cells,"
                f" fitted on 4,512 TRAIN clips and substituted in the code O reads, moves ADE6 by {a(r0['read']['delta'])} [{a(r0['read']['lo'])}, {a(r0['read']['hi'])}] /"
                f" {a(r1['read']['delta'])} [{a(r1['read']['lo'])}, {a(r1['read']['hi'])}] m against -F = -{R['F_m']:.4f} (not met; neither separated-better); the ALT MLP"
                f" {a(r0['alt']['delta'])} / {a(r1['alt']['delta'])} m (separated-worse); the gt ceiling (T0) {a(r0['gt']['delta'])} / {a(r1['gt']['delta'])} m. Committed next,"
                f" perception on the A40, is blocked (the A40 is not allocated); nothing on box 2 on this line. The three run stops before any number (run mechanics x2,"
                f" Amendment 1) are banked with the run. The four families are read beside (paired tool for TACTICAL / STRATEGIC, committed row definitions for LONGITUDINAL / LATERAL; cells separated {FAMSEP[0]} / {FAMSEP[1]}, none longitudinal). Package raw/2026-10-28-speedread/ ({n} files: read/fit/extract JSON, the 14 per-mode eval JSONs, family outputs, logs, stops, pull"
                f" record, run scripts; npz dumps, readout.pt and the train arrays stay on F:). Register row updated.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H, n)
