"""Hold: the zero-training crawl diagnostics H-CRAWL-01 / -02 / -03 (cards frozen before data), banked as
raw/2026-10-28-crawl-diag/, plus three register rows appended to GOALS_AND_CLAIMS.md (mods, append-only).
python mk_hold_crawl_diag.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_crawl_diag/"
S = "E:/Projects/TanitAD-artifacts/dagland/crawl_diag/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-crawl-diag/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
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
    assert len(b) < 2_500_000 or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


# the agent's manifest: every banked file must still carry the sha it reported
PIN = {"PREREG_FROZEN.md": "7f4601ec9035f30d", "READINGS.md": "a97423b613569b9e", "CRAWL_DIAG.json": "e14c1f0cf584c9ea",
       "RESULTS.md": "3ebbbf59e2cb8e30", "crawl_diag.py": "a48b2eef52780222", "test_crawl_diag.py": "cecc7a998af98ecb",
       "raw/crawl01_arrays.npz": "4ed58b65a83817b7", "raw/crawl02_arrays.npz": "9096fbca555fb4a0", "raw/p10_rerun.npz": "ab9106b8890917e4"}
n = 0
for root, dirs, fs in os.walk(S):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if not b:
            continue
        if rel in PIN:
            assert sha(b).startswith(PIN[rel]), rel
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")          # disclosed in msg.txt: CRLF folded to LF, no other byte changed
            assert b"\r" not in b, rel
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
put("stage/" + PK + "mk_hold_crawl_diag.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 1

SRC = "`" + PK + "RESULTS.md`"
ROWS = [
    ("**H-CRAWL-01** (2026-10-28)",
     "On the 327 GT-launch windows (eval-141), part of v8F's launch deficit is selection on the human launching plus regression to the"
     " conditional statistic: an outcome-blind predictor built on TRAIN (speed bucket x history class, 12 cells, smallest 475 windows) reads a"
     " gap of the same order as the model's.",
     "**MEASURED 2026-10-28, T1-kin gaps, episode-cluster bootstrap B = 2,000 over 49 episodes (it answers another draw of episodes only);"
     " cards frozen at 21:21 before any data (PREREG_FROZEN.md sha256 7f4601ec\u2026).** Outcome-blind floor on the 327: median table "
     + M + "1.034 [" + M + "1.298, " + M + "0.774], mean table " + M + "0.800 [" + M + "1.071, " + M + "0.529] m/s. DAG-OACC s0 gap " + M
     + "0.875 [" + M + "1.143, " + M + "0.601]: D = +0.159 [" + M + "0.136, +0.468] (median), " + M + "0.075 [" + M + "0.377, +0.232] (mean);"
     " at the floor on the point, CI half-width 0.30, so outcome (b) holds on the point only. REP s0 gap " + M + "1.431: D " + M + "0.397 ["
     + M + "0.777, +0.010] (median), " + M + "0.631 [" + M + "1.008, " + M + "0.225] (mean): neither committed outcome. Outcome (a) fails for"
     " both runs: on GT-stay windows (stopped, the human stays; 32 windows / 10 episodes) the models move +1.272 [+1.005, +1.624] and +1.215"
     " [+1.037, +1.366] m/s. In the moving launch windows (266 / 47) DAG-OACC reads " + M + "1.008 against a floor of " + M + "0.948 (D "
     + M + "0.060 [" + M + "0.430, +0.330]); REP's extra crawl sits there (D " + M + "0.670 [" + M + "1.130, " + M + "0.196]). Reading: the"
     " smaller crawl is what a predictor blind to the launch trigger shows, and both models creep where the human stays; the lever is"
     " trigger perception or the output representation, not a copycat fix. P10 rows carry a failed bit-equality control (c6, at most 6.5 mm,"
     " thread nondeterminism) and do not enter the verdict.",
     "MEASURED (outcome (b) on the point for DAG-OACC s0; neither outcome for REP s0)"),
    ("**H-CRAWL-02** (2026-10-28)",
     "The planner's 2 s output speed tracks its input speed (S_v >= 0.6 on stopped-start windows), and the run-to-run difference in S_v"
     " tracks the 0.556 m/s launch-gap difference between two same-init runs.",
     "**MEASURED 2026-10-28, first-tick input perturbation (T0-class, no rollout), 61 stopped-start windows / 15 episodes, bootstrap as"
     " H-CRAWL-01; all 8 controls PASS (delta 0 bit-identical, v_in = 0 positive control separated).** Pooled S_v: DAG-OACC +0.736 [+0.593,"
     " +0.829], REP +0.628 [+0.501, +0.726]; difference +0.108 [+0.053, +0.167]. The worse crawler (REP) has the SMALLER S_v, so the"
     " tracking prediction is refuted by sign. Decomposition: the frozen P10 prior alone gives +0.686 [+0.557, +0.768], identical in both"
     " checkpoints; the learned residual adds +0.050 [+0.015, +0.090] (DAG-OACC) and " + M + "0.059 [" + M + "0.121, +0.012] (REP)."
     " Reading (inference): the speed sensitivity belongs to the frozen prior, so kinematic-token dropout or history truncation on the"
     " learned path (H-CRAWL-04 / -05) has no learned shortcut to remove.",
     "MEASURED (S_v >= 0.6 on the point at both runs; the run-difference prediction refuted by sign)"),
    ("**H-CRAWL-03** (2026-10-28)",
     "The 0.556 m/s run-to-run launch difference is mostly between-run (basin and data order), not checkpoint noise within a run.",
     "**NOT EVALUABLE 2026-10-28:** no intermediate checkpoint exists for either run. The trainer keeps one `ckpt_last.pt` per run,"
     " rewritten every `--ckpt-every` 500 steps (`stack/scripts/v8f_mini_train.py:203`, `:1887`); the shared init is present and its sha"
     " matches. It unblocks with a trainer option that keeps the last k checkpoints (with its G-DVB entry), at about zero box-2 cost in the"
     " next fine-tune.",
     "NOT EVALUABLE (no intermediate checkpoints)"),
]
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-CRAWL-0" not in g
crlf = g.count(b"\r\n") == g.count(b"\n")
assert crlf or b"\r" not in g
add = "".join("| " + " | ".join((i, c, e, s, SRC)) + " |\n" for i, c, e, s in ROWS).encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
last = g.rstrip(b"\r\n").split(b"\n")[-1]
assert add.count(b" | ") == 4 * len(ROWS) and last.count(b" | ") == 4, "column count differs from the table's last row"
put("mods/" + GC, g + (add.replace(b"\n", b"\r\n") if crlf else add))
put("msg.txt", ("Crawl diagnostics H-CRAWL-01 / -02 / -03 (zero training, cards frozen before data): the smaller crawl sits at the"
                " outcome-blind floor and both runs creep +1.2 m/s where the human stays; output-speed sensitivity S_v 0.736 / 0.628 is"
                " carried by the frozen P10 prior (+0.686), the learned residual adds about 0, and the worse crawler has the smaller S_v;"
                " within- vs between-run variance not evaluable (one checkpoint per run). Package raw/2026-10-28-crawl-diag/ (%d files;"
                " .py / .md CRLF folded to LF, no other byte changed); three register rows appended.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode("utf-8"))
print("hold written", H, n, "crlf register" if crlf else "lf register")
