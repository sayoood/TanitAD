"""Hold: PREREG_V8F_DAGGER Amendment 2 (BANK-CONFIG reads limit_clips with the generator's convention; post-data, control-only, before any
re-read) + the fixed reader (pinned by its post-proof final_SHA256) + the run's evidence so far (the REFUSED read, both chains' logs, the T1
manifest and pins, the banks' small records) + the register row.  python mk_hold_dag_a2.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
R = "C:/Users/Admin/dagread/"
H = "C:/Users/Admin/qland/hold_dag_a2/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RD = PK + "raw/2026-10-26-dagger-read/"
RUN = PK + "raw/2026-10-27-dagger-run/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def rd(p):
    return open(p, "rb").read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def stage(rel, b):
    assert blob(rel) is None, f"{rel} exists on the tip"
    put("stage/" + rel, b)


# 1. the reader at A2, pinned by the proof's final_SHA256 (taken after the R77 re-check)
assert os.path.exists(R + "out/finish_a2.done") and not os.path.exists(R + "out/chain_failed_a2")
fin = {ln.split(None, 1)[1].lstrip("*").strip(): ln.split()[0] for ln in rd(R + "out/final_SHA256.txt").decode().splitlines() if ln.strip()}
for f, h in fin.items():
    assert sha(rd(R + f)) == h, f"dagread/{f} moved since its proof"
for f in ("dag_read.py", "test_dag_read.py", "mutate_dag.py", "NOTES.md", "HANDOFF.md"):
    b = rd(R + f)
    assert b"\r\n" not in b, f
    if blob(RD + f) is None:
        stage(RD + f, b)
    else:
        put(("mods/" if b.startswith(blob(RD + f)) else "mods-rw/") + RD + f, b)
for root, dirs, files in os.walk(R):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", "pt")]
    for fn in files:
        p = os.path.join(root, fn).replace("\\", "/")
        rel = p[len(R):]
        if not (rel.startswith(("fixtures/generator_a2/", "scratch/", "out/", "out_pre_a2/"))) or os.path.getsize(p) == 0:
            continue
        if blob(RD + rel) is None:
            stage(RD + rel, rd(p))

# 2. Amendment 2, appended to the landed PREREG (append-only MOD)
pre = blob("Project Steering/PREREG_V8F_DAGGER.md")
a2 = rd(A + "dagland/AMEND_DAG2.md")
assert b"\n## Amendment 2" not in pre and b"\n## Amendment 1" in pre          # its own heading; the text already cites ORC's Amendment 2
put("mods/Project Steering/PREREG_V8F_DAGGER.md", pre.rstrip(b"\n") + b"\n" + a2)

# 3. the run's evidence so far (small files only; checkpoints and shard npz stay on E:, named by sha256 in their SHA256SUMS)
stage(RUN + "dag_read_refused.out", rd(A + "dagt1/logs/dag_read.out"))
stage(RUN + "chain_v8f_dag.out", rd("C:/Users/Admin/box2_ship/logs/chain_v8f_dag.out"))
stage(RUN + "dag_t1_chain.out", rd(A + "dagt1/logs/chain.out"))
for f in ("runs.json", "PINS.txt"):
    stage(RUN + "t1/" + f, rd(A + "v8f-dagger-t1/" + f))
for s in (0, 1):
    for f in ("PROVENANCE.json", "bank.json", "SHA256SUMS.txt", "bank_stage.log"):
        stage(RUN + f"bank_s{s}/" + f, rd(A + f"v8f-dagger/bank_s{s}/" + f))
for run in ("M2b_s0_dag", "M2b_s0_dag0", "M2b_s1_dag", "M2b_s1_dag0"):
    for f in ("SHA256SUMS.txt", "summary.json", "argv.json", "launch_gate_token.json"):
        p = A + f"v8f-dagger/{run}/" + f
        if os.path.exists(p):
            stage(RUN + f"{run}/" + f, rd(p))
stage(RUN + "mk_hold_dag_a2.py", rd(A + "dagland/mk_hold_dag_a2.py"))

# 4. the register row, on the LIVE tip's text
g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-DAGGER-1**")]
assert len(row) == 1
cells = row[0].split(" | ")
cells[2] = cells[2] + (
    " **2026-10-27: RUN, read REFUSED at the bank stage, Amendment 2.** Box 2 trained all four runs (05:55; swap shares 0.1013 / 0.1008, DAG-0"
    " exactly 0; COVERAGE 0.756 / 0.769; K-B5 6.07e-05 / 6.09e-05 m on 613,132 records per seed). The reader refused on BANK-CONFIG alone: its"
    " predicate read the generator's `limit_clips` 0 (no limit) as a limit. No outcome row was computed. Amendment 2 (post-data, control-only)"
    " aligns that one clause with the generator; the reader's suite and mutation proof were re-run on the final bytes. The re-read follows"
    " this landing (`raw/2026-10-27-dagger-run/`).")
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g.replace(row[0], " | ".join(cells)).encode("utf-8"))
put("msg.txt", ("PREREG_V8F_DAGGER Amendment 2 (post-data, control-only, before any re-read): BANK-CONFIG reads limit_clips as the generator writes"
                " it (0 = no limit). The reader REFUSED at 06:02 on that clause alone, at the bank stage, before any row. Fixed reader + its proof,"
                " the run's evidence (refused log, both chains' logs, T1 manifest + pins, banks' records, run records), register row\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written", H)
