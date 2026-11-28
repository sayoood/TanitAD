"""Mutation proof for PREREG_V8F_DSCALE (`--data-frac 0.5`): every mutation must turn stack/tests/test_v8f_dscale.py RED; the unmutated tree must be GREEN before and after, and every mutated file is
restored byte-identical (in a `finally`, then asserted: `read_bytes() ==` the saved original). Deletes nothing.

  python mutate_dscale.py <repo root> <python exe> [--check-anchors] [--part k/n] [--only "a||b"] [--skip-baseline]

Run it on a byte-identical COPY of the tree (it mutates source files in place; the build ran it on copies under the scratchpad), never on the only copy of the work. `--part k/n` runs every n-th
mutant, so n copies of the tree can run in parallel; `--skip-baseline` skips the full-file GREEN runs (the parts are given the baseline once, by whoever launches them).
THE NAMED MUTATIONS (the brief): (1) `a.data_frac` dropped from the `select_train_clips` call, (2) the queue's `--data-frac 0.5` dropped, (3) a reader threshold flipped. The rest take each hop of
the same path apart: the trainer (the budget sized from the half, the batches drawn from all windows, the G-DVB entry), the data module (the last half, no sha12 order, truncation not rounding),
the queue (each stage's argv, the dispatch, the eval's checkpoint), the chain (the gate's assertions, the predecessor, a seed-1 stage, the dry run) and the reader (every threshold, the decision
rule, r's direction, the manifest checks, the step / clip / argv / parameter / weight refusals, the training-loss window). Anchors must occur EXACTLY ONCE (asserted before anything runs). Each
mutant is judged by the SECTION of the test file it belongs to (`-k reach|reader|queue|chain`, test names carry their section); the baseline and the final run are the WHOLE file."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
PK = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
DATA = ROOT / "stack" / "tanitad" / "data" / "v8f_mini_data.py"
QUEUE = PK / "code" / "v8f_mini_box2_queue.cmd"
CHAIN = PK / "code" / "box2_ops" / "chain_v8f_dscale.sh"
READER = PK / "code" / "v8f_dscale_read.py"
TEST = "stack/tests/test_v8f_dscale.py"
REACH, READ, QUE, CHN, RFS = "reach", "reader and not refuses", "queue", "chain", "reader and refuses"

MUT = [
    # ---- the three NAMED mutations
    (TRAIN, "NAMED 1: a.data_frac is dropped from the select_train_clips call", "clips = D.select_train_clips(clips_all, a.data_frac)", "clips = D.select_train_clips(clips_all, 1.0)", REACH),
    (QUEUE, "NAMED 2a: traindscale drops --data-frac 0.5", '--grid pool4x10 --data-frac 0.5 --out "%OUT%" %A%"', '--grid pool4x10 --out "%OUT%" %A%"', QUE),
    (QUEUE, "NAMED 2b: gatedscale drops --data-frac 0.5", '--arm M2b --data-frac 0.5 --out "%RUNSM%\\gate_M2b-df50"', '--arm M2b --out "%RUNSM%\\gate_M2b-df50"', QUE),
    (READER, "NAMED 3: the reader's bar is 10 %, not 5 %", "BAR = 0.05 ", "BAR = 0.10 ", READ),
    # ---- the trainer
    (TRAIN, "the step budget is sized from the HALF's windows (S no longer counts all clips)", 'S = math.ceil(info["n_windows_all_clips"] / a.batch)', 'S = math.ceil(info["n_windows"] / a.batch)', REACH),
    (TRAIN, "the budget's window list is built from the kept clips", "allw = D.train_windows(store, J, clips_all, hist=cfg.hist, start=cfg.window_start)", "allw = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)", REACH),
    (TRAIN, "the batches are drawn from ALL windows (n_clips says half, the loop trains on everything)", "    return store, J, wins, clips, info\n", "    return store, J, allw, clips, info\n", REACH),
    (TRAIN, "the G-DVB entry for data_frac is removed", '    "data_frac": "data:n_clips", "steps_mult": "run",', '    "steps_mult": "run",', REACH),
    (TRAIN, "the summary does not record the data fraction", '"seed": a.seed, "data_frac": a.data_frac,', '"seed": a.seed, "data_frac": 1.0,', REACH),
    # ---- the data module
    (DATA, "select_train_clips takes the LAST n clips", "    return order[:n]\n", "    return order[-n:]\n", REACH),
    (DATA, "select_train_clips ignores the sha12 order", "    order = sorted(clips, key=sha12)\n    n = len(order) if frac >= 1.0", "    order = list(clips)\n    n = len(order) if frac >= 1.0", REACH),
    (DATA, "select_train_clips truncates instead of rounding", "max(1, int(round(frac * len(order))))", "max(1, int(frac * len(order)))", REACH),
    (DATA, "select_train_clips returns everything from 0.5 up", "n = len(order) if frac >= 1.0 else", "n = len(order) if frac >= 0.5 else", REACH),
    # ---- the queue
    (QUEUE, "traindscale runs --data-frac 0.25", '--data-frac 0.5 --out "%OUT%" %A%"', '--data-frac 0.25 --out "%OUT%" %A%"', QUE),
    (QUEUE, "evaldscale evaluates B's run dir, not Q's", 'if "%~2"=="" (echo REFUSED: evaldscale needs a SEED & exit /b 2)\r\nset "OUT=%RUNSM%\\M2b_s%~2_df50"',
     'if "%~2"=="" (echo REFUSED: evaldscale needs a SEED & exit /b 2)\r\nset "OUT=%RUNSM%\\M2b_s%~2"', QUE),
    (QUEUE, "evaldscale is never dispatched", 'if /i "%~1"=="evaldscale" goto :evaldscale\r\n', "", QUE),
    (QUEUE, "the gate line gains a second lever (--hist 10)", '--arm M2b --data-frac 0.5 --out "%RUNSM%\\gate_M2b-df50"', '--arm M2b --data-frac 0.5 --hist 10 --out "%RUNSM%\\gate_M2b-df50"', QUE),
    (QUEUE, "the E3 oracle loses its all-latcons flag", '--tag %TG%-tacorc-all-latcons --tac oracle --tac-groups all-latcons --tac-alpha 1 %EC% %ER%"\r\nif defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)\r\nif not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)\r\nif exist "%EV%\\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evaldscale_1)',
     '--tag %TG%-tacorc-all-latcons --tac oracle %EC% %ER%"\r\nif defined V8F_DRYRUN (echo %E0% & echo %E1% & echo %E2% & echo %E3% & exit /b 0)\r\nif not exist "%OUT%\\summary.json" (echo REFUSED: %TG% has not finished training & exit /b 2)\r\nif exist "%EV%\\eval_%TG%.npz" (echo [skip] eval %TG% & goto :evaldscale_1)', QUE),
    (QUEUE, "an OLD stage is edited (gatehist loses --window-start 3)", 'set "GH="%PY%" "%TRAIN%" --arm M2b --hist 10 --window-start 3 --out', 'set "GH="%PY%" "%TRAIN%" --arm M2b --hist 10 --out', QUE),
    # ---- the chain
    (CHAIN, "gate: the steps assertion is removed", 'if sc["steps_total"] != steps: bad.append', "if False: bad.append", CHN),
    (CHAIN, "gate: the built-fraction check is removed", 'if i["n_clips"] / i["n_clips_all"] != 0.5: bad.append', "if False: bad.append", CHN),
    (CHAIN, "gate: the half's identity is compared with B's whole list", 'if i["train_sha256"] != shal(half): bad.append', 'if i["train_sha256"] != shal(order): bad.append', CHN),
    (CHAIN, "gate: the parameter count may differ from B's", 'if lv["n_params_trainable"] != bl["n_params_trainable"]: bad.append', "if False: bad.append", CHN),
    (CHAIN, "gate: the budget may be sized from another window set", 'if i.get("n_windows_all_clips") != b.get("n_windows"): bad.append', "if False: bad.append", CHN),
    (CHAIN, "gate: the queue line may carry --steps", 'for flag in ("--steps", "--max-clips",', 'for flag in ("--max-clips",', CHN),
    (CHAIN, "the predecessor is the GRID chain, not HIST's", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_hist.out}", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_grid.out}", CHN),
    (CHAIN, "a seed-1 stage is added", "runone 0\ndscaleread dscale_read.json", "runone 0\nrunone 1\ndscaleread dscale_read.json", CHN),
    (CHAIN, "the dry run loses the eval stage", 'for st in "gatedscale" "traindscale 0" "evaldscale 0"; do', 'for st in "gatedscale" "traindscale 0"; do', CHN),
    # ---- the reader
    (READER, "decide: DATA-LIMITED needs r > bar, not r >= bar", 'if sep and r >= bar and d["point"] > 0:', 'if sep and r > bar and d["point"] > 0:', READ),
    (READER, "decide: EPOCH-LIMITED fires for r >= -bar", 'if sep and r <= -bar and d["point"] < 0:', 'if sep and r >= -bar and d["point"] < 0:', READ),
    (READER, "decide: the separation requirement is dropped from DATA-LIMITED", 'if sep and r >= bar and d["point"] > 0:', 'if r >= bar and d["point"] > 0:', READ),
    (READER, "decide: the sign agreement is dropped from DATA-LIMITED", 'if sep and r >= bar and d["point"] > 0:', "if sep and r >= bar:", READ),
    (READER, "D-T: r is computed the wrong way round", "r, rw = rq / rb - 1.0, wq / wb - 1.0", "r, rw = rb / rq - 1.0, wb / wq - 1.0", READ),
    (READER, "D-T: r is computed on the window-weighted RMSE", "out, out_w = decide(r, d), decide(rw, d)", "out, out_w = decide(rw, d), decide(rw, d)", READ),
    (READER, "the training-loss window is the HEAD, not the last 10 %", 'tail = [r for r in recs if r["step"] > cut]', 'tail = [r for r in recs if r["step"] <= cut]', READ),
    (READER, "the identity control no longer checks D-T", 'if dt["r"] != 0.0 or dt["r_window"] != 0.0 or dt["outcome"] != "NOT DATA-LIMITED":', "if False:", READ),
    (READER, "half_n floors instead of rounding", "return n if frac >= 1.0 else max(1, int(round(frac * n)))", "return n if frac >= 1.0 else max(1, int(frac * n))", READ),
    (READER, "the manifest's half is the LAST half", "    half = order[:n]\n", "    half = order[-n:]\n", READ),
    (READER, "the manifest is not checked against B's recorded list", 'if exp["b_sha256"] != rsm.get("train_sha256"):', "if False:", RFS),
    (READER, "Q's step count is not checked", 'if sm.get("steps") != rsm.get("steps") or sm.get("steps") != STEPS_B:', "if False:", RFS),
    (READER, "Q's clip count is not checked", 'if sm.get("n_clips") != want_n or want_n != exp["n_half"]:', "if False:", RFS),
    (READER, "Q's clips are not compared with the half", 'if sm.get("train_sha256") != exp["half_sha256"] or', "if False and", RFS),
    (READER, "Q's argv is not compared with B's", "    if fq != fb:\n", "    if False:\n", RFS),
    (READER, "the loss weights are not compared (w_diff skipped)", 'for k in w_diff(lg.get("w") or {}, lb.get("w") or {}, rfl)]', "for k in ()]", RFS),
    (READER, "the parameter groups are not compared", "            if pq.get(k) != pb.get(k):\n", "            if False:\n", RFS),
    (READER, "Q's budget may be sized from another window set", 'if info.get("n_windows_all_clips") != rsm.get("n_windows"):', "if False:", RFS),
    (READER, "Q's eval may score another checkpoint", 'if len(parts) < 2 or parts[-2] != q_dir(s) or parts[-1] != "ckpt_last.pt":', "if False:", RFS),
    (READER, "Q's built hist is not checked", '(("hist", B_HIST), ("window_start", B_START), ("hmax", b_hm))', '(("window_start", B_START), ("hmax", b_hm))', RFS),
    (READER, "B may be a data-fraction run", 'if DF_FLAG in fl and not PGT._same_float(fl[DF_FLAG], 1.0):', "if False:", RFS),
]
if "--only" in sys.argv:
    _keys = sys.argv[sys.argv.index("--only") + 1].split("||")
    MUT = [m for m in MUT if any(k in m[1] for k in _keys)]
if "--part" in sys.argv:
    _k, _n = (int(x) for x in sys.argv[sys.argv.index("--part") + 1].split("/"))
    MUT = MUT[_k - 1::_n]


def run(k=None):
    cmd = [PY, "-m", "pytest", "-q", "-x", "-rf", "-p", "no:cacheprovider", TEST] + (["-k", k] if k else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(ROOT / "stack"), str(ROOT / "taniteval")])})
    lines = r.stdout.strip().splitlines() or ["<no output>"]
    failed = [x.split("::", 1)[1].split(" - ")[0].strip() for x in lines if x.startswith(("FAILED ", "ERROR ")) and "::" in x]
    return r.returncode, lines[-1] + (f"  [{', '.join(failed)}]" if failed else "") + ("" if r.returncode == 0 or failed else "  " + " | ".join(r.stderr.strip().splitlines()[-2:]))


def apply(m, b):
    old, new = m[2].encode("utf-8"), m[3].encode("utf-8")
    assert b.count(old) == 1, f"{m[1]}: anchor {m[2][:70]!r} found {b.count(old)} times in {m[0].name}"
    assert old != new, f"{m[1]}: the mutation changes nothing"
    return b.replace(old, new)


orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
for m in MUT:
    apply(m, orig[m[0]])
print(f"anchors OK: {len(MUT)} mutants, every anchor occurs exactly once", flush=True)
if "--check-anchors" in sys.argv:
    sys.exit(0)
if "--skip-baseline" not in sys.argv:
    rc, last = run()
    print(f"BASELINE rc={rc} {last}", flush=True)
    assert rc == 0, "the unmutated tree must be GREEN"
red, survivors = 0, []
for m in MUT:
    path, name, sect = m[0], m[1], m[4]
    b = orig[path]
    path.write_bytes(apply(m, b))
    try:
        rc, last = run(sect)
    finally:
        path.write_bytes(b)
        assert path.read_bytes() == orig[path], f"{path} was not restored byte for byte"
    red += rc != 0
    if rc == 0:
        survivors.append(name)
    print(f"{'RED  ' if rc != 0 else 'GREEN'} {path.name}: {name} [-k {sect}] -> {last}", flush=True)
assert all(p.read_bytes() == orig[p] for p in orig), "a file was not restored"
if "--skip-baseline" not in sys.argv:
    rc, last = run()
    print(f"AFTER rc={rc} {last}")
else:
    rc = 0
print(f"{red} / {len(MUT)} mutations RED" + (f"; SURVIVORS: {survivors}" if survivors else ""))
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
