"""Hold: PREREG_V8F_SPEEDSUB_T1 promoted before any T1-kin substitution rollout and before its read-out is fitted, with its evidence package
(the SPEEDSUB section 6 prerequisites, the REP s0 speed-cell diagnosis) and the register row (append).
python mk_hold_speedsub_t1_prereg.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/dagland/"
H = "C:/Users/Admin/qland/hold_speedsub_t1_prereg/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedsub-t1-prereg/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- the PREREG, promoted
d = rd(D + "speedsub_t1/PREREG_V8F_SPEEDSUB_T1.md")
assert b"\r" not in d
OLD = b"**Status: DRAFT (2026-10-28, the Master Mind). NOT RUN.**"
assert d.count(OLD) == 1
p = d.replace(OLD, ("**Status: PRE-REGISTERED, NOT RUN (2026-10-28).** Promoted by the Master Mind (draft sha256 " + sha(d)[:8]
                    + "\u2026) before any T1-kin substitution rollout and before `mlp_ego_c` is fitted. The instrument (the fit, the"
                    " `forward_infer` hook, the reader) is NOT BUILT; it is built against this text, and any amendment is appended before"
                    " the first rollout.").encode("utf-8"))
for w in BANNED:
    assert w not in p.lower(), w
PR = "Project Steering/PREREG_V8F_SPEEDSUB_T1.md"
assert blob(PR) is None
put("stage/" + PR, p)

# ---- the evidence package
n = 0
for sub in ("speedsub_t1_prep", "rep_s0_diag"):
    root = D + sub + "/"
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            pth = os.path.join(dp, f).replace("\\", "/")
            rel = sub + "/" + pth[len(root):]
            if "__pycache__" in rel or f.endswith((".pyc", ".pt", ".npz")):
                continue
            b = rd(pth)
            if rel.endswith((".py", ".sh", ".md")):
                assert b"\r" not in b, rel
            assert b and len(b) < 400_000, rel
            assert blob(PK + rel) is None, rel
            put("stage/" + PK + rel, b)
            n += 1
put("stage/" + PK + "mk_hold_speedsub_t1_prereg.py", rd(D + "speedsub_t1/mk_hold_speedsub_t1_prereg.py"))
n += 1

# ---- the register row (append after the last row, which is SPEEDSUB's)
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and g.rstrip(b"\n").split(b"\n")[-1].startswith(b"| **H-V8F-SPEEDSUB-1** (2026-10-28)")
row = ("| **H-V8F-SPEEDSUB-T1-1** (2026-10-28) | At T1-kin on the DAG-OACC base (`--limit causal`), replacing at every replan tick only"
       " the five speed cells O reads by `mlp_ego_c` (the `mlp_ego` design refitted with the T1 tool's own causal speed bucket in place of the"
       " logged one; fitted once, used as fitted) cuts ADE6 against the same run's `identity` rollout by at least F/2 = 0.06035 m, separated,"
       " at each of four runs (DAG-OACC s0 / s1 and the DAG-ORC replicates REP s0 / s1) | **PRE-REGISTERED, NOT RUN (2026-10-28).**"
       " Commissioned by `PREREG_V8F_SPEEDSUB` section 6 (SPEEDSUB read PASS on M2b-mixo, T1-class open loop). Seen before writing: the"
       " SPEEDSUB outcome, REP s0's T0 speed cells (all five 0.73\u20131.17 m/s slow against CTRL s0, separated;"
       " `raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/`), and TRAIN-fold attribution (skill survives without `nav`, not without the logged"
       " `limit`, which is the declared future-speed channel `D-SPEEDBAND-EGO-ECHO-1`; `.../speedsub_t1_prep/FINDING_1_ATTRIBUTION.md`)."
       " No T1-kin substitution number and no eval number of a causal read-out exists. Arms per run: `identity`, `egoc`, `egoshufc`"
       " (deliberate regression, must be separated-worse at all four runs); beside, the logged lane on CTRL only, a declared open channel."
       " Paired episode-cluster bootstrap, B = 2,000, 141 episodes, per run; the four runs carry the training-run question"
       " (`H-ESTIM-SEED-1`). Outcomes: PASS (a hit at all four runs, no guard) \u2192 adopt as a `forward_infer` speed sub-head and read"
       " Read A on the substituted CTRL (DOMINANT at both seeds is the dominance claim); PARTIAL / NULL \u2192 the history-aware causal"
       " read-out; HARMS \u2192 per-cell attribution at T1-kin. Beside: `H-V8F-SPEEDSUB-T1-2`, the floor rule (COLLAPSE when the REP s0 \u2212"
       " CTRL s0 gap on ADE6, yaw-rate and curvature halves under `egoc`). Zero training, dev-box CPU, about 1 h for twelve rollouts"
       " (ESTIMATED). Instrument NOT BUILT. | PRE-REGISTERED | `Project Steering/PREREG_V8F_SPEEDSUB_T1.md` |\n")
rb_ = row.encode("utf-8")
for w in BANNED:
    assert w not in rb_.lower(), w
assert rb_.count(b" | ") >= 4
put("mods/" + GC, g + rb_)
put("msg.txt", ("PREREG_V8F_SPEEDSUB_T1 promoted before any T1-kin substitution rollout and before its read-out is fitted: the ego"
                " read-out refitted on the T1 tool's causal speed bucket, substituted into O's five speed cells at every replan tick on"
                " DAG-OACC and its two replicates (zero training, dev-box CPU, T1-kin). PASS = a hit (separated, <= -F/2) at all four runs;"
                " PASS leads to the Read-A dominance read on the substituted base. Package raw/2026-10-28-speedsub-t1-prereg/ (%d files):"
                " the SPEEDSUB section 6 prerequisites (attribution refits: the logged limit carries the skill, nav does not; DAG-base"
                " pair-compatibility; the T1-kin hook site) and the REP s0 diagnosis (its T's five speed cells read 0.73-1.17 m/s slow,"
                " the source of the DAG-ORC replicate floor). Register row H-V8F-SPEEDSUB-T1-1.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n, "package files")
