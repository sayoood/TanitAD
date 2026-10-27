#!/usr/bin/env python3
"""selftest_dag2_queue.py -- the proof for the box-2 DAGGER2 queue stages and chain (PREREG_V8F_DAGGER2). CPU only, no box 2, nothing launched, nothing written outside dag2q/ and the session scratch directory.

Every check is either a CHECK (must hold) or a DELIBERATE REGRESSION (a defect put in on purpose; the instrument must CATCH it; a regression that is not caught is a FAILURE of the proof). Sections:

  A  the REAL queue .cmd through cmd.exe (V8F_DRYRUN): dag2 / dag20 / bankdag2 argv; "every other flag exactly as DAG's"; the pair rule; refusals; the TEXT diff is insertions only
  S  BYTE-IDENTITY of every existing stage: the tip queue (8e121ea8) and the new queue print the same rc and lines on the stages the DAG selftest checked, bankdag/gatedag/traindag/evaldag for dag and dag0 at both seeds,
     and a sweep of EVERY dispatched stage name
  B  the REAL queue through cmd.exe, NOT dry, on a fake box (stub trainer / eval / bank generator; the real shard tool): bankdag2 fan-out into dagger_bank_dag2, the pin, byte-identity of the argv that runs, refusals,
     iteration-1 files untouched
  C  the chain's decisions on files: the DAG-read predicate (the REAL DAG_READ.json + doctored copies) and the bank judge (REAL iteration-1 banks as the format check + doctored copies)
  D  the chain script: --dry-run, --judge-only, --check-tree, the real entry's early stops, no box-2 call (ssh / scp shims), the export list covers what box 2 reads, the source order, --publish-only
  E  gate_dag2_assert.py on the REAL DAG corpus gate (selfcheck.json written by the real trainer on box 2) + doctored copies

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python selftest_dag2_queue.py [--sections A,S,B,C,D,E] [--out selftest_output.txt]
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dag2q_lib as L  # noqa: E402

CHAIN_DIR = HERE / "box2_ship"
CHAIN = Path(os.environ["DAG2Q_CHAIN"]) if os.environ.get("DAG2Q_CHAIN") else CHAIN_DIR / "chain_v8f_dag2.sh"          # the mutation runner points sections D at a MUTATED copy (its helper scripts beside it)
CHECKS_DIR = Path(os.environ["DAG2Q_CHECKS_DIR"]) if os.environ.get("DAG2Q_CHECKS_DIR") else CHAIN_DIR                    # ... and sections C / E at mutated copies of the two python helpers
sys.path.insert(0, str(CHECKS_DIR))
import v8f_dag2_chain_checks as CK  # noqa: E402
import gate_dag2_assert as GA  # noqa: E402

W = L.BT.parent                                                           # session scratch
BASH = r"C:\Program Files\Git\bin\bash.exe"
DAGART = Path("E:/Projects/TanitAD-artifacts/v8f-dagger")
DAG_READ = Path("E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json")
DAG_READ_SHA = "3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d"
TREE = Path(os.environ.get("REACH_TREE", W / "tree"))
DAG_CK = ["d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"]
M2B_CK = ["601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"]
LOG = []
N = {"check": 0, "check_fail": 0, "reg": 0, "reg_missed": 0}


def say(s):
    LOG.append(s)
    print(s, flush=True)


def check(name, cond, detail=""):
    N["check"] += 1
    if not cond:
        N["check_fail"] += 1
    say(("PASS  " if cond else "FAIL  ") + name + (f"   [{str(detail)[:400]}]" if (detail and not cond) else ""))
    return bool(cond)


def regress(name, caught, detail=""):
    N["reg"] += 1
    if not caught:
        N["reg_missed"] += 1
    say(("CAUGHT " if caught else "MISSED ") + "(deliberate regression) " + name + (f"   [{str(detail)[:400]}]" if (detail and not caught) else ""))
    return bool(caught)


def posix(p) -> str:
    return str(p).replace("\\", "/")


def sh(cmd, env=None, cwd=None, timeout=600):
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": posix(W / "tmp"), **(env or {})}
    (W / "tmp").mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL, errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1), encoding="utf-8")


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ================================================================================================ A. rendering
def names_ego0(av) -> bool:
    return av.count("--ego-perturb-p") == 1 and av[av.index("--ego-perturb-p") + 1] == "0"


def lines_for(seed, arm, sha):
    parg = " 0.1" if arm in ("dag", "dag2") else ""
    g = L.dry(f"gatedag {arm} {seed}{parg}", sha=sha)
    t = L.dry(f"traindag {arm} {seed}{parg}", sha=sha)
    e = L.dry(f"evaldag {arm} {seed}")
    return {"gate": L.trainer_line(g), "train": L.trainer_line(t), "eval": [x for x in e if "v8f_mini_eval.py" in x][0]}


def section_a():
    say("== A. the real queue through cmd.exe, V8F_DRYRUN: what DAG2 / DAG2-0 / bankdag2 would run")
    S = L.SHA_FAKE
    R = r"D:\tanitad\runs\v8f_mini_full"
    for seed in (0, 1):
        d2, d20 = lines_for(seed, "dag2", S), lines_for(seed, "dag20", S)
        d1, d10 = lines_for(seed, "dag", S), lines_for(seed, "dag0", S)
        a2, a20 = L.train_argv(d2["train"]), L.train_argv(d20["train"])
        f2, f20 = L.flags(a2), L.flags(a20)
        check(f"s{seed} DAG2: --dagger-p 0.1 once; bank = RUNSM\\dagger_bank_dag2\\s{seed} (NOT the iteration-1 dagger_bank); sha = the stage's; --dagger-seed {seed}; --ego-perturb-p 0",
              f2.get("--dagger-p") == [["0.1"]] and f2.get("--dagger-bank") == [[f"{R}\\dagger_bank_dag2\\s{seed}"]] and f2.get("--dagger-bank-sha256") == [[S]] and f2.get("--dagger-seed") == [[str(seed)]] and f2.get("--ego-perturb-p") == [["0"]],
              {k: f2.get(k) for k in ("--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed", "--ego-perturb-p")})
        check(f"s{seed} DAG2: --init-ckpt = DAG_s{seed}'s checkpoint (M2b_s{seed}_dag\\ckpt_last.pt), NOT M2b's; --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --seed {seed}",
              f2.get("--init-ckpt") == [[f"{R}\\M2b_s{seed}_dag\\ckpt_last.pt"]] and f2.get("--steps") == [["3000"]] and f2.get("--lr") == [["1e-4"]] and f2.get("--warmup") == [["100"]] and f2.get("--p-gt-start") == [["0.2"]]
              and f2.get("--p-gt-end") == [["0.2"]] and f2.get("--seed") == [[str(seed)]] and f2.get("--arm") == [["M2b"]] and f2.get("--grid") == [["pool4x10"]], {k: f2.get(k) for k in ("--init-ckpt", "--steps", "--seed")})
        check(f"s{seed} DAG2-0: --init-ckpt = DAG_s{seed}'s checkpoint; names --dagger-p 0 once and NONE of the three DAG-only flags; --ego-perturb-p 0; no ego law flag",
              f20.get("--init-ckpt") == [[f"{R}\\M2b_s{seed}_dag\\ckpt_last.pt"]] and f20.get("--dagger-p") == [["0"]] and not any(k in f20 for k in L.DAG_ONLY) and f20.get("--ego-perturb-p") == [["0"]]
              and not any(k.startswith("--ego-perturb-") and k != "--ego-perturb-p" for k in f20), sorted(f20))
        second = ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--w-acc", "--code-mix", "--o-interface", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly", "--no-consistency", "--detach-upper", "--no-resume")
        check(f"s{seed}: no second lever on either DAG2 arm", not any(k in f2 or k in f20 for k in second))
        # "every other flag exactly as DAG's argv": DAG2 vs DAG (the iteration-1 stage in the SAME queue), DAG2-0 vs DAG-0
        x2, x1 = L.drop_flag(a2, {"--out", "--init-ckpt", "--dagger-bank"}), L.drop_flag(L.train_argv(d1["train"]), {"--out", "--init-ckpt", "--dagger-bank"})
        check(f"s{seed} 'every other flag exactly as DAG's argv': DAG2 == DAG once --out, --init-ckpt and --dagger-bank are removed (same sha, same --dagger-seed, same schedule, same data flags)", x2 == x1 and len(x2) > 40, f"{len(x2)} vs {len(x1)} tokens")
        y2, y1 = L.drop_flag(a20, {"--out", "--init-ckpt"}), L.drop_flag(L.train_argv(d10["train"]), {"--out", "--init-ckpt"})
        check(f"s{seed} 'every other flag exactly as DAG-0's': DAG2-0 == DAG-0 once --out and --init-ckpt are removed", y2 == y1 and len(y2) > 40, f"{len(y2)} vs {len(y1)} tokens")
        # the pair rule
        p2, p0 = L.drop_flag(a2, {"--out", "--dagger-p"} | L.DAG_ONLY), L.drop_flag(a20, {"--out", "--dagger-p"})
        check(f"s{seed} PAIR RULE: DAG2 and DAG2-0 are identical once --out, --dagger-p and the three DAG-only flags are removed; they differ in exactly {{--out, --dagger-p}} on the shared flags, and DAG2 names exactly the three extra flags",
              p2 == p0 and set(f2) - set(f20) == L.DAG_ONLY and set(f20) <= set(f2) and {k for k in f20 if f2[k] != f20[k]} == {"--out", "--dagger-p"})
        check(f"s{seed}: run dirs are M2b_s{seed}_dag2 / _dag20 (never DAG's, DAG-0's, M2b's)", f2["--out"][0][0] == f"{R}\\M2b_s{seed}_dag2" and f20["--out"][0][0] == f"{R}\\M2b_s{seed}_dag20")
        for arm, ln in (("dag2", d2), ("dag20", d20)):
            ga, tr = L.train_argv(ln["gate"]), L.train_argv(ln["train"])
            check(f"s{seed} {arm}: GATE argv == TRAIN argv with --out and --selfcheck-only changed, --selfcheck-only exactly once; gate --out is gate_M2b-{arm}_s{seed}",
                  L.drop_flag(ga, {"--out"}, {"--selfcheck-only"}) == L.drop_flag(tr, {"--out"}) and ga.count("--selfcheck-only") == 1 and "--selfcheck-only" not in tr and L.flags(ga)["--out"][0][0] == f"{R}\\gate_M2b-{arm}_s{seed}")
            for kind in ("gate", "train"):
                av = L.train_argv(ln[kind])
                check(f"s{seed} {arm} {kind}: `--ego-perturb-p 0` NAMED exactly once with the value 0", names_ego0(av), [x for x in av if "ego" in x])
            et = L.tokens(ln["eval"])
            check(f"s{seed} {arm}: the ONE T0 eval is the standard one: tag M2b-{arm}_s{seed}, this run's checkpoint, no --ages, --project --families", "--ages" not in et and et[et.index("--tag") + 1] == f"M2b-{arm}_s{seed}"
                  and et[et.index("--ckpt") + 1] == f"{R}\\M2b_s{seed}_{arm}\\ckpt_last.pt" and et[-2:] == ["--project", "--families"], ln["eval"][:200])
        # the bank line: bankdag vs bankdag2
        b1, b2 = L.dry(f"bankdag {seed}")[0], L.dry(f"bankdag2 {seed}")[0]
        t1, t2 = L.tokens(b1), L.tokens(b2)
        g2 = L.flags(t2[t2.index("--") + 1:])
        diff_idx = [i for i, (x, y) in enumerate(zip(t1, t2)) if x != y]
        check(f"s{seed} bankdag2: the generator is DAG_s{seed}'s: --ckpt = M2b_s{seed}_dag\\ckpt_last.pt, --m2b-argv = M2b_s{seed}_dag\\argv.json, --out-dir = dagger_bank_dag2\\s{seed}",
              g2.get("--ckpt") == [[f"{R}\\M2b_s{seed}_dag\\ckpt_last.pt"]] and g2.get("--m2b-argv") == [[f"{R}\\M2b_s{seed}_dag\\argv.json"]] and L.flags(t2[2:t2.index("--")]).get("--out-dir") == [[f"{R}\\dagger_bank_dag2\\s{seed}"]], sorted(g2))
        check(f"s{seed} bankdag2 == bankdag in everything else: same token count, and the ONLY differing tokens are the three paths (--out-dir, --ckpt, --m2b-argv); same shards 4, cache, tracks, join, eval grid, batch, threads, device cuda, --expect-clips 4512 --expect-windows 306566",
              len(t1) == len(t2) and len(diff_idx) == 3 and all(t1[i - 1] in ("--out-dir", "--ckpt", "--m2b-argv") for i in diff_idx), [(t1[i], t2[i]) for i in diff_idx])
        check(f"s{seed} bankdag2: no ampersand reaches an argument; the iteration-1 bank directory '\\dagger_bank\\s' appears nowhere in the line", "&" not in b2 and f"{R}\\dagger_bank\\s" not in b2)
    # argument validation
    for args, why in (("gatedag dag2 0", "dag2 without P"), ("gatedag dag2 0 0.3", "dag2 with P = 0.3 (iteration 1's other value)"), ("gatedag dag2 0 0.2", "P = 0.2"), ("gatedag dag2 0 0", "P = 0 on dag2"), ("traindag dag2 1 0.5", "train with P 0.5"),
                      ("gatedag dag20 0 0.1", "dag20 with a P"), ("gatedag dag20 0 0", "dag20 with a P of 0 (a lever declared and inert)"), ("traindag dag20 1 0.1", "train dag20 with a P"), ("gatedag dag3 0 0.1", "an unknown arm"),
                      ("gatedag dag2 2 0.1", "seed 2"), ("gatedag dag2 0.1", "a missing seed"), ("evaldag dag2 5", "eval seed 5"), ("bankdag2 7", "bank seed 7"), ("bankdag2", "bankdag2 without a seed")):
        rc, o = L.cmd(args, env={"DAGSHA_DRY": L.SHA_FAKE})
        regress(f"`{args}` ({why}) is REFUSED: rc 2, says REFUSED, prints no trainer / bank line", rc == 2 and any("REFUSED" in x for x in o) and not any("v8f_mini_train.py" in x or "v8f_dagger_queue_tool.py" in x for x in o), f"rc {rc} {o[:2]}")
    rc, o = L.cmd("gatedag dag2 0 0.1", env={"DAGSHA_DRY": L.SHA_FAKE})
    check("gatedag dag2 0 0.1 is accepted (rc 0) and carries --dagger-p 0.1", rc == 0 and L.flags(L.train_argv(L.trainer_line(o)))["--dagger-p"] == [["0.1"]])
    # the TEXT: insertions only
    import build_queue as BQ  # noqa: E402
    tip, new = L.TIP_QUEUE.read_bytes(), L.QUEUE.read_bytes()
    tl, nl = tip.decode("utf-8").split("\r\n"), new.decode("utf-8").split("\r\n")
    j, extra = 0, 0
    for x in nl:
        if j < len(tl) and x == tl[j]:
            j += 1
        else:
            extra += 1
    check("TEXT: the new queue is the tip queue with lines INSERTED only (every tip line, in order, is in the new file; 0 removed, 0 changed)", j == len(tl) and extra == len(nl) - len(tl) and extra > 0, f"matched {j}/{len(tl)}, inserted {extra}")
    check("TEXT: the new queue re-derives byte-for-byte from build_queue.py (tip sha256 pinned) and every line ends CRLF (no lone LF / CR)", L.QUEUE == HERE / "stage" / L.PK / "code" / "v8f_mini_box2_queue.cmd" and (new == BQ.build() or os.environ.get("DAG2Q_QUEUE")) and new.count(b"\r\n") == new.count(b"\n") == new.count(b"\r") and new.endswith(b"\r\n"))
    check("TEXT: the tip queue is the pinned 8e121ea8 file", hashlib.sha256(tip).hexdigest() == BQ.TIP_SHA256)


# ================================================================================================ S. byte-identity of the existing stages
def section_s(sweep=True):
    say("== S. every EXISTING stage prints byte-identical lines from the tip queue (8e121ea8) and the new queue")
    S = L.SHA_FAKE
    cases = ["trainorc orc0 0", "trainorc orc 1", "gateorc orc 0", "gateorc orc0 1", "evalorc orc 1", "gate", "train M2b 0", "eval M2b 0", "gatehist", "trainoacc 0"]            # the 10 the DAG selftest checked
    for seed in (0, 1):
        cases += [f"bankdag {seed}", f"gatedag dag {seed} 0.1", f"gatedag dag {seed} 0.3", f"gatedag dag0 {seed}", f"traindag dag {seed} 0.1", f"traindag dag0 {seed}", f"evaldag dag {seed}", f"evaldag dag0 {seed}"]
    cases += ["gatedag dag 0", "gatedag dag 0 0.2", "gatedag dag0 0 0.1", "gatedag dagx 0 0.1", "gatedag dag 2 0.1", "bankdag 7", "evaldag dagx 0", "traindag dag0 0 0"]                       # the iteration-1 refusals, message for message
    bad = []
    for args in cases:
        r1, o1 = L.cmd(args, queue=L.TIP_QUEUE, env={"WACC": "0.1", "DAGSHA_DRY": S})
        r2, o2 = L.cmd(args, queue=L.QUEUE, env={"WACC": "0.1", "DAGSHA_DRY": S})
        if not (r1 == r2 and o1 == o2 and len(o1) > 0):
            bad.append((args, r1, r2))
    check(f"{len(cases)} stage invocations (the DAG selftest's 10 + bankdag / gatedag / traindag / evaldag for dag and dag0 at both seeds + the iteration-1 refusals): rc and every printed line identical", not bad, bad[:4])
    if not sweep:
        return
    tip_text = L.TIP_QUEUE.read_bytes().decode("utf-8")
    names = sorted(set(re.findall(r'^if /i "%~1"=="([A-Za-z0-9_]+)" goto :', tip_text, flags=re.M)))
    swroot = W / "sweep"
    env = {"REPO": str(swroot / "repo"), "PY": str(swroot / "py.exe"), "CACHEFULL": str(swroot / "cache"), "V9": str(swroot / "v9"), "RUNSM": str(swroot / "runs"), "WACC": "0.1", "DAGSHA_DRY": S}
    bad2, n = [], 0
    for nm in names:
        for tail in ("", " 0", " M2b 0", " dag 0 0.1"):
            r1, o1 = L.cmd(nm + tail, queue=L.TIP_QUEUE, env=env, timeout=60)
            r2, o2 = L.cmd(nm + tail, queue=L.QUEUE, env=env, timeout=60)
            n += 1
            if not (r1 == r2 and o1 == o2):
                bad2.append((nm + tail, r1, r2, o1[:1], o2[:1]))
    check(f"SWEEP: all {len(names)} dispatched stage names x 4 argument shapes ({n} invocations, scratch paths): rc and output identical between the tip queue and the new queue", not bad2 and len(names) > 40, bad2[:3])
    labels_tip = set(re.findall(r"^:([A-Za-z0-9_]+)\s*$", tip_text, flags=re.M))
    labels_new = set(re.findall(r"^:([A-Za-z0-9_]+)\s*$", L.QUEUE.read_bytes().decode("utf-8"), flags=re.M))
    check("labels: every tip label is still there; the new queue adds exactly :bankdag2, :dag2arm, :dag2arm_dry and :dag2arm_have", labels_tip <= labels_new and labels_new - labels_tip == {"bankdag2", "dag2arm", "dag2arm_dry", "dag2arm_have"}, sorted(labels_new - labels_tip))


# ================================================================================================ B. fake box
FAKE_TRAINER = '''import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
os.makedirs(out, exist_ok=True)
json.dump(a, open(os.path.join(out, "stub_argv.json"), "w"))
if os.environ.get("STUB_FAIL"):
    sys.exit(3)
if "--selfcheck-only" in a:
    json.dump({"declared_vs_built": "PASS"}, open(os.path.join(out, "selfcheck.json"), "w"))
else:
    json.dump({"done": True}, open(os.path.join(out, "summary.json"), "w"))
    open(os.path.join(out, "ckpt_last.pt"), "wb").write(b"x")
'''
FAKE_EVAL = '''import json, os, sys
a = sys.argv[1:]
tag = a[a.index("--tag") + 1]
d = a[a.index("--out-dir") + 1]
os.makedirs(d, exist_ok=True)
json.dump(a, open(os.path.join(d, "eval_" + tag + ".stub_argv.json"), "w"))
open(os.path.join(d, "eval_" + tag + ".npz"), "wb").write(b"x")
'''
FAKE_BANK = '''import json, os, sys, time
a = sys.argv[1:]
mode = a[0]
out = a[a.index("--out-dir") + 1]
os.makedirs(out, exist_ok=True)
if mode == "gen":
    k, n = int(a[a.index("--shard") + 1]), int(a[a.index("--n-shards") + 1])
    json.dump(a, open(os.path.join(out, "gen_argv_%d.json" % k), "w"))
    if os.environ.get("STUB_FAIL_SHARD") == str(k):
        sys.exit(1)
    stem = os.path.join(out, "shard_%03dof%03d" % (k, n))
    open(stem + ".npz", "wb").write(b"x")
    json.dump({"k": k}, open(stem + ".provenance.json", "w"))
else:
    n = len([f for f in os.listdir(out) if f.endswith(".npz")])
    json.dump({"fake": True, "n_shards": n}, open(os.path.join(out, "bank.json"), "w"), sort_keys=True)
    json.dump({"fake": True}, open(os.path.join(out, "PROVENANCE.json"), "w"))
'''


def fake_world(root: Path) -> dict:
    if root.exists():
        shutil.rmtree(root)
    repo, cache, v9, runsm = root / "repo", root / "cache", root / "v9", root / "runs"
    sc = repo / "stack" / "scripts"
    sc.mkdir(parents=True)
    (sc / "v8f_mini_train.py").write_text(FAKE_TRAINER)
    (sc / "v8f_mini_eval.py").write_text(FAKE_EVAL)
    (sc / "v8f_dagger_bank.py").write_text(FAKE_BANK)
    shutil.copy(TREE / "stack" / "scripts" / "v8f_dagger_queue_tool.py", sc / "v8f_dagger_queue_tool.py")                    # the REAL shard tool of the tip
    pk = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
    (pk / "2026-10-17-v8f-mini" / "raw").mkdir(parents=True)
    (pk / "2026-10-17-v8f-mini" / "raw" / "eval_grid_windows.json").write_text(json.dumps({"order": ["c1", "c2"]}))
    hj = pk / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"p")
    (hj / "p10_eval_pred6.npy").write_bytes(b"p")
    for d in ("train", "eval"):
        (cache / d).mkdir(parents=True)
        (cache / d / "manifest.jsonl").write_text("{}\n")
    (cache / "native_tracks.pt").write_bytes(b"t")
    v9.mkdir()
    for s in (0, 1):
        for nm in (f"M2b_s{s}", f"M2b_s{s}_dag"):
            (runsm / nm).mkdir(parents=True)
            (runsm / nm / "ckpt_last.pt").write_bytes(b"ckpt-%s" % nm.encode())
            (runsm / nm / "argv.json").write_text(json.dumps({"argv": []}))
        (runsm / f"M2b_s{s}_dag" / "summary.json").write_text(json.dumps({"done": True}))
    (runsm / "join_train.npz").write_bytes(b"j")
    return {"REPO": str(repo), "PY": L.PY, "CACHEFULL": str(cache), "V9": str(v9), "RUNSM": str(runsm)}


def tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


def section_b():
    say("== B. the real queue through cmd.exe, NOT dry, on a fake box (stub trainer / eval / bank generator; the real shard tool)")
    root = W / "fakebox"
    env = fake_world(root)
    runs = Path(env["RUNSM"])
    before_dag = {nm: tree_digest(runs / nm) for nm in ("M2b_s0_dag", "M2b_s1_dag", "M2b_s0", "M2b_s1")}

    def q(args, extra=None, dry=False):
        return L.cmd(args, env={**env, **(extra or {})}, dryrun=dry, timeout=300)
    # a half-trained DAG (no summary.json) is REFUSED, and no generator runs
    (runs / "M2b_s0_dag" / "summary.json").rename(runs / "M2b_s0_dag" / "summary.hidden")
    rc, o = q("bankdag2 0")
    regress("bankdag2 REFUSES while DAG_s0 has no summary.json (a half-trained DAG is a different generator): rc 2, no bank directory, no generator ran", rc == 2 and any("has not finished" in x for x in o) and not (runs / "dagger_bank_dag2").exists(), o[:3])
    (runs / "M2b_s0_dag" / "summary.hidden").rename(runs / "M2b_s0_dag" / "summary.json")
    for s in (0, 1):
        rc, o = q(f"bankdag2 {s}")
        bd = runs / "dagger_bank_dag2" / f"s{s}"
        check(f"bankdag2 {s}: rc 0, [ok], merged bank.json + PROVENANCE.json in dagger_bank_dag2\\s{s} (4 shards)", rc == 0 and (bd / "bank.json").exists() and (bd / "PROVENANCE.json").exists() and len(list(bd.glob("shard_*.npz"))) == 4, (rc, o[-2:]))
        gens = [json.loads((bd / f"gen_argv_{k}.json").read_text()) for k in range(4)]
        g0 = gens[0]
        check(f"bankdag2 {s}: the generators RECEIVED --ckpt = M2b_s{s}_dag\\ckpt_last.pt and --m2b-argv = M2b_s{s}_dag\\argv.json (and 4 distinct shards)",
              all(g[g.index("--ckpt") + 1] == str(runs / f"M2b_s{s}_dag" / "ckpt_last.pt") and g[g.index("--m2b-argv") + 1] == str(runs / f"M2b_s{s}_dag" / "argv.json") for g in gens) and sorted(int(g[g.index("--shard") + 1]) for g in gens) == [0, 1, 2, 3], g0[:8])
        rc, o = q(f"bankdag2 {s}")
        check(f"bankdag2 {s} again: [skip] (a finished bank is not regenerated)", rc == 0 and any("[skip]" in x for x in o), o)
    check("the iteration-1 bank directory dagger_bank was NEVER created by bankdag2", not (runs / "dagger_bank").exists())
    # gate + pin + train + eval, both arms
    for seed in (0, 1):
        sha = hashlib.sha256((runs / "dagger_bank_dag2" / f"s{seed}" / "bank.json").read_bytes()).hexdigest()
        for arm, parg in (("dag2", " 0.1"), ("dag20", "")):
            nm = f"M2b_s{seed}_{arm}"
            rc, o = q(f"gatedag {arm} {seed}{parg}")
            gate_argv = json.loads((runs / f"gate_M2b-{arm}_s{seed}" / "stub_argv.json").read_text()) if (runs / f"gate_M2b-{arm}_s{seed}" / "stub_argv.json").exists() else []
            pin = runs / "dagpin" / f"{nm}.train.argv.txt"
            check(f"gatedag {arm} {seed}: rc 0, [ok]; the trainer RECEIVED --selfcheck-only and --init-ckpt = DAG's checkpoint; the pin {nm}.train.argv.txt exists", rc == 0 and "--selfcheck-only" in gate_argv and gate_argv[gate_argv.index("--init-ckpt") + 1] == str(runs / f"M2b_s{seed}_dag" / "ckpt_last.pt") and pin.exists(), (rc, o[-2:]))
            pin_tokens = L.tokens(pin.read_text(encoding="utf-8").strip())[2:] if pin.exists() else []
            if arm == "dag2":
                check(f"gatedag dag2 {seed}: the bank sha256 in the argv is the sha256 of THIS bank.json (computed on the box), the bank dir is dagger_bank_dag2\\s{seed}", "--dagger-bank-sha256" in pin_tokens and pin_tokens[pin_tokens.index("--dagger-bank-sha256") + 1] == sha
                      and pin_tokens[pin_tokens.index("--dagger-bank") + 1] == str(runs / "dagger_bank_dag2" / f"s{seed}"), pin_tokens[-8:])
            else:
                check(f"gatedag dag20 {seed}: needs no bank and names none", not any(k in pin_tokens for k in L.DAG_ONLY) and pin_tokens[pin_tokens.index("--dagger-p") + 1] == "0")
            rc2, o2 = q(f"traindag {arm} {seed}{parg}")
            run_argv = json.loads((runs / nm / "stub_argv.json").read_text()) if (runs / nm / "stub_argv.json").exists() else []
            check(f"traindag {arm} {seed}: rc 0; the argv the trainer RECEIVED is byte-identical to the pinned tokens (it ran the PIN); summary.json written", rc2 == 0 and run_argv == pin_tokens and (runs / nm / "summary.json").exists(), (rc2, o2[-2:]))
            rc3, o3 = q(f"traindag {arm} {seed}{parg}")
            check(f"traindag {arm} {seed} again: [skip]", rc3 == 0 and any("[skip]" in x for x in o3))
            rc4, o4 = q(f"evaldag {arm} {seed}")
            ev = json.loads((runs / "eval" / f"eval_M2b-{arm}_s{seed}.stub_argv.json").read_text()) if (runs / "eval" / f"eval_M2b-{arm}_s{seed}.stub_argv.json").exists() else []
            check(f"evaldag {arm} {seed}: rc 0; the eval RECEIVED this run's checkpoint {nm}\\ckpt_last.pt and tag M2b-{arm}_s{seed}", rc4 == 0 and ev and ev[ev.index("--ckpt") + 1] == str(runs / nm / "ckpt_last.pt") and ev[ev.index("--tag") + 1] == f"M2b-{arm}_s{seed}", (rc4, o4[-2:]))
    # deliberate regressions of the pin
    s = 0
    pinf = runs / "dagpin" / "M2b_s0_dag2.train.argv.txt"
    rc, o = q("gatedag dag2 0 0.1")
    check("re-gating the SAME bank is accepted (the pin is unchanged)", rc == 0, o[-2:])
    orig = pinf.read_bytes()
    pinf.write_bytes(orig.replace(b"--steps 3000", b"--steps 3001"))
    rc, o = q("gatedag dag2 0 0.1")
    regress("a pin that differs from the line derived now (--steps 3000 -> 3001) makes gatedag REFUSE and does not replace the pin", rc == 2 and any("pins another training argv" in x for x in o) and pinf.read_bytes() != orig, o[:2])
    shutil.rmtree(runs / "M2b_s0_dag2")
    rc, o = q("traindag dag2 0 0.1")
    regress("traindag REFUSES when the pin differs from the line derived now", rc == 2 and any("differs from the one the gate tokenised" in x for x in o), o[:2])
    pinf.write_bytes(orig)
    bj = runs / "dagger_bank_dag2" / "s0" / "bank.json"
    bj_orig = bj.read_bytes()
    bj.write_bytes(bj_orig.replace(b'"n_shards"', b'"n_shards" '))
    rc, o = q("traindag dag2 0 0.1")
    regress("traindag REFUSES when bank.json changed by one byte after the gate", rc == 2 and any("differs from the one the gate tokenised" in x for x in o), o[:2])
    bj.write_bytes(bj_orig)
    pinf.unlink()
    rc, o = q("traindag dag2 0 0.1")
    regress("traindag with no pin (no gate) REFUSES", rc == 2 and any("no pinned training argv" in x for x in o), o[:2])
    pinf.write_bytes(orig)
    shutil.rmtree(runs / "eval", ignore_errors=True)
    shutil.rmtree(runs / "M2b_s1_dag20")
    rc, o = q("evaldag dag20 1")
    regress("evaldag before training REFUSES", rc == 2 and any("has not finished training" in x for x in o), o[:2])
    rc, o = q("gatedag dag2 0 0.3")
    regress("gatedag dag2 with P 0.3 REFUSES on the fake box too, and runs no trainer", rc == 2 and any("fixed at 0.1" in x for x in o), o[:2])
    rc, o = q("gatedag dag2 1 0.1", extra={"STUB_FAIL": "1"})
    regress("a failing gate (trainer rc 3) makes gatedag exit non-zero with [FAIL]", rc == 1 and any("[FAIL]" in x for x in o), o[:2])
    after = {nm: tree_digest(runs / nm) for nm in ("M2b_s0_dag", "M2b_s1_dag", "M2b_s0", "M2b_s1")}
    check("ISOLATION: the iteration-1 run directories (M2b_s0/1, M2b_s0/1_dag) are byte-for-byte what they were before the whole DAG2 session", before_dag == after)
    check("ISOLATION: no file named dag / dag0 appeared (no M2b_s?_dag0, no dagpin\\M2b_s?_dag.train..., no gate_M2b-dag_s?, no gate_M2b-dag0_s?)", not any((runs / x).exists() for x in ("M2b_s0_dag0", "M2b_s1_dag0", "gate_M2b-dag_s0", "gate_M2b-dag0_s0", "gate_M2b-dag_s1", "gate_M2b-dag0_s1")) and not any(p.name.startswith("M2b_s") and p.name.endswith("_dag.train.argv.txt") for p in (runs / "dagpin").glob("*")))


# ================================================================================================ C. decisions
def doctored_read(dst: Path, mutate):
    d = jload(DAG_READ)
    mutate(d)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return dst


def fixture_banks(root: Path, ck=(None, None), mutate=None):
    """Small-file copies of the REAL iteration-1 banks (bank.json, PROVENANCE.json, shard provenance) in root/s0, root/s1, with the generator's checkpoint sha256 optionally rewritten; ``mutate(seed, man, prov, provs)``."""
    if root.exists():
        shutil.rmtree(root)
    for s in (0, 1):
        d, src = root / f"s{s}", DAGART / f"bank_s{s}"
        d.mkdir(parents=True)
        man, prov = jload(src / "bank.json"), jload(src / "PROVENANCE.json")
        provs = [jload(src / f"shard_{k:03d}of004.provenance.json") for k in range(4)]
        if ck[s]:
            man["common"]["ckpt_sha256"] = ck[s]
        if mutate:
            mutate(s, man, prov, provs)
        jdump(d / "bank.json", man)
        jdump(d / "PROVENANCE.json", prov)
        for k, p in enumerate(provs):
            jdump(d / f"shard_{k:03d}of004.provenance.json", p)
    return root / "s0", root / "s1"


def judge(d0, d1, shas=tuple(DAG_CK)):
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):                                       # the judge prints its one-line JSON; the log keeps only verdicts
        rc, res = CK.bank_judge(type("A", (), {"seed0_dir": str(d0), "seed1_dir": str(d1), "banked_sha0": shas[0], "banked_sha1": shas[1], "expect_clips": 4512, "expect_windows": 306566}))
    return rc, res


def section_c():
    say("== C. the chain's decisions on files: the DAG-read predicate and the bank judge")
    t = W / "decide"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    # ---- the DAG-read predicate on the REAL file, both sides
    rc, msg = CK.dag_predicate(str(DAG_READ), DAG_READ_SHA)
    check("dag-predicate on the REAL DAG_READ.json: rc 0, prints `ROBUST <the pinned sha256>`", rc == 0 and msg == f"ROBUST {DAG_READ_SHA}" and sha_file(DAG_READ) == DAG_READ_SHA, msg)
    check("dag-predicate's built-in pin is the one in the brief (3393fdd5...)", CK.DAG_READ_SHA256 == DAG_READ_SHA)
    def predicate_of(mut, name, expect_pinned=True):
        p = doctored_read(t / f"{name}.json", mut)
        return CK.dag_predicate(str(p), DAG_READ_SHA if expect_pinned else sha_file(p))
    rc, msg = predicate_of(lambda d: d["controls_PASS"].__setitem__("K-B0:s0", True), "samebytes_not")                                    # re-serialised file: different bytes, same content
    regress("a re-serialised copy of the REAL read (same content, different bytes) is NOT the pinned file: rc 2, 'not the pinned DAG read'", rc == 2 and "pinned" in msg, msg[:160])
    for outcome in ("PARTIAL", "DOMINANT", "REFUSED", "HARMS", "NO EFFECT"):
        def mut(d, o=outcome):
            d["OUTCOME"]["OUTCOME"] = o
            d["OUTCOME"]["matching_rows_in_table_order"] = [o]
        rc, msg = predicate_of(mut, f"out_{outcome}", expect_pinned=False)
        regress(f"a (self-pinned) read with outcome {outcome}: rc 4, STOP, names the outcome", rc == 4 and outcome in msg and "not ROBUST" in msg, msg[:160])
    def two_rows(d):
        d["OUTCOME"]["matching_rows_in_table_order"] = ["ROBUST", "PARTIAL"]
    rc, msg = predicate_of(two_rows, "tworows", expect_pinned=False)
    regress("ROBUST with two matching table rows: rc 2 (not exactly one)", rc == 2 and "not exactly one" in msg, msg[:160])
    rc, msg = predicate_of(lambda d: d["controls_PASS"].__setitem__("K-B2:s0", False), "failctl", expect_pinned=False)
    regress("ROBUST with one failed control: rc 2 (a read with a failed control is not an outcome)", rc == 2 and "failed" in msg, msg[:160])
    rc, msg = predicate_of(lambda d: d.__setitem__("claim_row", "H-V8F-ORECOVER-1"), "foreign", expect_pinned=False)
    regress("ROBUST under another PREREG's claim row (ORC's): rc 2", rc == 2 and "claim_row" in msg, msg[:160])
    rc, msg = predicate_of(lambda d: d.pop("OUTCOME"), "nooutcome", expect_pinned=False)
    regress("no OUTCOME row: rc 2", rc == 2, msg[:160])
    rc, msg = predicate_of(lambda d: d.__setitem__("controls_PASS", {}), "emptyctl", expect_pinned=False)
    regress("an EMPTY controls_PASS (nothing was checked): rc 2", rc == 2, msg[:160])
    rc, msg = CK.dag_predicate(str(t / "missing.json"), DAG_READ_SHA)
    regress("a missing file: rc 2", rc == 2, msg[:120])
    (t / "garbage.json").write_text("{not json", encoding="utf-8")
    rc, msg = CK.dag_predicate(str(t / "garbage.json"), sha_file(t / "garbage.json"))
    regress("an unreadable (non-JSON) file: rc 2", rc == 2, msg[:120])
    # ---- the judge on the REAL iteration-1 banks: format check + K-B2 discrimination
    b0, b1 = DAGART / "bank_s0", DAGART / "bank_s1"
    rc, res = judge(b0, b1, shas=tuple(M2B_CK))
    check("judge FORMAT CHECK: the REAL iteration-1 banks pass the DAG2 judge when their own generator (M2b) is the expected one (every control + COVERAGE 0.756 / 0.769 >= 0.5): the judge reads the real bank format", rc == 0 and res["outcome"] == "PASS" and
          all(0.5 <= s["coverage"]["fraction"] for s in res["seeds"]), (rc, res["refused_because"]))
    rc, res = judge(b0, b1)
    kb2 = [b for b in res["refused_because"] if b.startswith("K-B2")]
    regress("the REAL iteration-1 banks are REFUSED as DAG2's evidence (generator = M2b, expected = DAG): rc 2, exactly the two K-B2 failures, every other control passes", rc == 2 and len(res["refused_because"]) == 2 and len(kb2) == 2, res["refused_because"])
    # ---- fixtures with the generator rewritten to DAG's checkpoint: the good case and doctored copies
    g0, g1 = fixture_banks(t / "good", ck=tuple(DAG_CK))
    rc, res = judge(g0, g1)
    check("judge: the real banks re-labelled with DAG's checkpoint sha256 (small files only) PASS, COVERAGE >= 0.5 at both seeds", rc == 0 and res["outcome"] == "PASS", res["refused_because"])
    def cases():
        def sset(seed_only, fn):
            return lambda s, man, prov, provs: fn(man, prov, provs) if s == seed_only else None
        yield "K-B4 record missing at s1", 2, "K-B4", sset(1, lambda m, p, ps: p["controls"].pop("K-B4"))
        yield "K-B5 record missing at s0", 2, "K-B5", sset(0, lambda m, p, ps: p["controls"].pop("K-B5"))
        yield "K-B4 case `loop` False at s0", 2, "K-B4", sset(0, lambda m, p, ps: p["controls"]["K-B4"]["cases"].__setitem__("loop", False))
        yield "K-B5 error 2e-4 at s1", 2, "K-B5", sset(1, lambda m, p, ps: p["controls"]["K-B5"].__setitem__("max_pose_now_err_m", 2e-4))
        yield "K-B5 PASS not True at s0", 2, "K-B5", sset(0, lambda m, p, ps: p["controls"]["K-B5"].__setitem__("PASS", 1))
        yield "K-B1 mismatch chunk at s1", 2, "K-B1", sset(1, lambda m, p, ps: ps[2]["kb1_replay"].__setitem__("mismatch_chunks", 1))
        yield "K-B0 mismatch at s0", 2, "K-B0", sset(0, lambda m, p, ps: ps[0]["kb0_tick0_identity"].__setitem__("mismatch", 3))
        yield "bank flagged partial at s1", 2, "COMPLETE", sset(1, lambda m, p, ps: m.__setitem__("partial", True))
        yield "TRAIN exclude changed at s0", 2, "TRAIN list", sset(0, lambda m, p, ps: m["common"]["train"].__setitem__("exclude", "0:60"))
        yield "rollout configuration changed at s1", 2, "ROLLOUT", sset(1, lambda m, p, ps: m["common"]["rollout"].__setitem__("executor", "greedy"))
        yield "eval overlap 1 at s0", 2, "K-B3", sset(0, lambda m, p, ps: m["common"]["eval"].__setitem__("overlap", 1))
        yield "generator checkpoint = M2b at s1 (the iteration-1 bank re-used)", 2, "K-B2", sset(1, lambda m, p, ps: m["common"].__setitem__("ckpt_sha256", M2B_CK[1]))
        def cov(m, p, ps, frac=0.49):
            n = int(frac * m["coverage"]["n_windows"])
            m["coverage"].update(n_eligible=n, blocked=True)
            p["coverage"].update(n_eligible=n)
        yield "COVERAGE 0.49 at s1 (flag consistent)", 3, "", sset(1, cov)
        def cov_flag(m, p, ps):
            m["coverage"]["blocked"] = True
        yield "COVERAGE `blocked` flag disagrees with the fraction at s0", 2, "COVERAGE", sset(0, cov_flag)
    for name, want_rc, key, mut in cases():
        d0, d1 = fixture_banks(t / "mut", ck=tuple(DAG_CK), mutate=mut)
        rc, res = judge(d0, d1)
        regress(f"judge: {name} -> rc {want_rc} ({'BLOCKED' if want_rc == 3 else 'REFUSED'}), names {key!r}" if key else f"judge: {name} -> rc {want_rc} BLOCKED", rc == want_rc and (not key or any(key in b for b in res["refused_because"])), (rc, res["refused_because"][:2]))
    def half(sd, m, p, ps):
        if sd == 1:
            n = m["coverage"]["n_windows"]
            m["coverage"].update(n_eligible=n // 2, blocked=False)
            p["coverage"].update(n_eligible=n // 2)
    d0, d1 = fixture_banks(t / "exact", ck=tuple(DAG_CK), mutate=half)
    rc, res = judge(d0, d1)
    check("judge BOUNDARY: coverage exactly 0.5 at s1 passes (inclusive)", rc == 0 and res["seeds"][1]["coverage"]["fraction"] >= 0.5 and abs(res["seeds"][1]["coverage"]["fraction"] - 0.5) < 1e-6, (rc, res["seeds"][1]["coverage"]))
    # ---- the pure helpers the chain uses are the DAG chain's (argv-sha, argv-same): sanity
    gl, tl = lines_for(0, "dag2", L.SHA_FAKE)["gate"], lines_for(0, "dag2", L.SHA_FAKE)["train"]
    check("argv-sha of the training line == an independent sha256 of the compact JSON argv; argv-same SAME for (gate, train) and DIFFERENT when --steps differs", CK.argv_sha256(CK.argv_of(tl)) == L.argv_sha256(L.train_argv(tl)) and CK.argv_same(gl, tl) and not CK.argv_same(gl, tl.replace("--steps 3000", "--steps 3001")))


# ================================================================================================ D. the chain script
SHIM = W / "shim"


def make_shims():
    SHIM.mkdir(parents=True, exist_ok=True)
    for nm in ("ssh", "scp"):
        (SHIM / nm).write_bytes(f'#!/bin/sh\necho "{nm} $@" >> "{posix(SHIM)}/calls.log"\nexit 99\n'.encode())
    (SHIM / "calls.log").write_bytes(b"")


def shim_calls() -> str:
    return (SHIM / "calls.log").read_text(errors="replace")


def bash(args, env=None, timeout=300, chain=None):
    e = {"PATH": posix(SHIM) + ":" + os.environ.get("PATH", ""), **(env or {})}
    return sh([BASH, posix(chain or CHAIN), *args], env=e, timeout=timeout)


def section_d():
    say("== D. chain_v8f_dag2.sh")
    make_shims()
    t = W / "chaind"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    src = CHAIN.read_text(encoding="utf-8")
    # ---- --dry-run
    qf = posix(L.QUEUE)
    rc, o = bash(["--dry-run"], {"V8F_QUEUE": qf, "AFTER_OUT": "/c/Users/Admin/box2_ship/logs/chain_v8f_dag.out"})
    check("chain --dry-run: rc 0, DAG read ROBUST (pinned), arm DAG2 at p 0.1, predecessor = the DAG chain's log", rc == 0 and "DAG-read predicate (rc 0): ROBUST " + DAG_READ_SHA in o and "arm DAG2 runs at p = 0.1" in o and "chain_v8f_dag.out" in o, o[:600])
    order = re.findall(r"^== (M2b_s\d_dag2?0?):", o, flags=re.M)
    check("chain --dry-run: the four runs in PREREG_V8F_DAGGER2's order after bank s0 and bank s1: DAG2 s0, DAG2-0 s0, DAG2 s1, DAG2-0 s1", order == ["M2b_s0_dag2", "M2b_s0_dag20", "M2b_s1_dag2", "M2b_s1_dag20"] and o.index("== bank s0") < o.index("== bank s1") < o.index("== M2b_s0_dag2"), order)
    check("chain --dry-run: every gate line is 'SAME' as its training line (4 of 4), prints the bank line `bankdag2` twice and never the iteration-1 `bankdag`/`dagger_bank\\s`", o.count("SAME") == 4 and "-- bankdag2 0" in o and "-- bankdag2 1" in o and "-- bankdag 0" not in o and "dagger_bank\\s" not in o and o.count("dagger_bank_dag2\\s") >= 4, o.count("SAME"))
    m = re.findall(r"^-- token argv_sha256 \(the training argv\): ([0-9a-f]{64})$", o, flags=re.M)
    exp = []
    for (s, arm) in ((0, "dag2"), (0, "dag20"), (1, "dag2"), (1, "dag20")):
        exp.append(L.argv_sha256(L.train_argv(lines_for(s, arm, "BANK_SHA256_OF_BANK_JSON")["train"])))
    check("chain --dry-run: the four planned token argv_sha256 values equal an independent sha256 of the queue's printed training argv", m == exp, (m[:1], exp[:1]))
    check("chain --dry-run touched no box: no ssh / scp call", shim_calls() == "", shim_calls()[:200])
    # ---- --check-tree: pass on a tree with the new queue, fail on the tip's
    PKC = L.PK + "/code"
    for nm, qfile in (("new", L.QUEUE), ("tip", L.TIP_QUEUE)):
        ct = W / f"chk_{nm}"
        if ct.exists():
            shutil.rmtree(ct)
        (ct / PKC).mkdir(parents=True)
        shutil.copytree(TREE / "stack", ct / "stack")
        shutil.copy(qfile, ct / PKC / "v8f_mini_box2_queue.cmd")
        rc, o = bash(["--check-tree", posix(ct)])
        if nm == "new":
            check("chain --check-tree on a tree carrying the trainer lever, the generator AND the new queue: rc 0 (all patterns found)", rc == 0 and "lever check: every one of" in o, o[-300:])
        else:
            miss = [x for x in o.splitlines() if x.startswith("MISSING")]
            regress("chain --check-tree on the tip's queue (DAG stages only): rc 1, every DAG2 queue insertion is reported MISSING (and nothing of the trainer lever is)", rc == 1 and len(miss) == 9 and all("v8f_mini_box2_queue.cmd" in x for x in miss), (rc, len(miss)))
    shutil.rmtree(W / "chk_new_nolever", ignore_errors=True)
    ct = W / "chk_new_nolever"
    shutil.copytree(W / "chk_new", ct)
    tr = ct / "stack" / "scripts" / "v8f_mini_train.py"
    tr.write_bytes(tr.read_bytes().replace(b'add_argument("--dagger-p"', b'add_argument("--dagger-pX"'))
    rc, o = bash(["--check-tree", posix(ct)])
    regress("chain --check-tree with the trainer's --dagger-p flag renamed: rc 1 and the missing pattern is named", rc == 1 and "MISSING in stack/scripts/v8f_mini_train.py" in o, o[-200:])
    # ---- --judge-only
    g0, g1 = fixture_banks(t / "good", ck=tuple(DAG_CK))
    rc, o = bash(["--judge-only", posix(g0), posix(g1)])
    check("chain --judge-only (pinned ROBUST read, both banks fine): rc 0, prints ZZDAG2-JUDGE-PASS and no stop marker", rc == 0 and "ZZDAG2-JUDGE-PASS" in o and "ZZV8FMINI-STOP" not in o, o[-400:])
    i0, i1 = DAGART / "bank_s0", DAGART / "bank_s1"
    rc, o = bash(["--judge-only", posix(i0), posix(i1)])
    regress("chain --judge-only on the REAL iteration-1 banks: rc 2, ZZDAG2-REFUSED naming K-B2, ZZV8FMINI-STOP REFUSED, no gate queued, never ZZDAG2-JUDGE-PASS", rc == 2 and "ZZDAG2-REFUSED" in o and "K-B2" in o and "ZZV8FMINI-STOP REFUSED" in o and "ZZDAG2-JUDGE-PASS" not in o, o[-500:])
    b0, b1 = fixture_banks(t / "blk", ck=tuple(DAG_CK), mutate=lambda s, m, p, ps: (m["coverage"].update(n_eligible=int(0.49 * m["coverage"]["n_windows"]), blocked=True), p["coverage"].update(n_eligible=int(0.49 * m["coverage"]["n_windows"]))) if s == 1 else None)
    rc, o = bash(["--judge-only", posix(b0), posix(b1)])
    regress("chain --judge-only with coverage 0.49 at s1: rc 3, ZZDAG2-BLOCKED and `ZZV8FMINI-STOP BLOCKED`, 'no gate was queued', never the all-gates line", rc == 3 and "ZZDAG2-BLOCKED" in o and "ZZV8FMINI-STOP BLOCKED" in o and "no gate was queued" in o and "ZZDAG2-JUDGE-PASS" not in o, o[-500:])
    # ---- the predicate stops the REAL entry before anything is touched
    for name, mut, want in (("PARTIAL", lambda d: (d["OUTCOME"].__setitem__("OUTCOME", "PARTIAL"), d["OUTCOME"].__setitem__("matching_rows_in_table_order", ["PARTIAL"])), "not ROBUST"), ("failed control", lambda d: d["controls_PASS"].__setitem__("K-B2:s0", False), "refuses")):
        p = doctored_read(t / f"read_{name.replace(' ', '_')}.json", mut)
        chain2 = t / f"chain_{name.replace(' ', '_')}.sh"
        for f in ("v8f_dag2_chain_checks.py", "gate_dag2_assert.py"):
            shutil.copy(CHECKS_DIR / f, t / f)
        chain2.write_bytes(src.replace("DAG_READ_SHA=" + DAG_READ_SHA, "DAG_READ_SHA=" + sha_file(p)).encode("utf-8"))          # a test copy whose pin is the doctored file, to reach the OUTCOME clause
        rc, o = bash([], {"DAG2_DAG_READ": posix(p)}, chain=chain2)
        regress(f"the REAL entry with a pinned-but-{name} read stops at once (rc 1, ZZV8FMINI-STOP, before AFTER_OUT / the busy marker / any box-2 call)", rc == 1 and "ZZV8FMINI-STOP" in o and want in o and "waiting for the predecessor" not in o, o[-400:])
    p = doctored_read(t / "read_unpinned.json", lambda d: d["controls_PASS"].__setitem__("K-B0:s0", True))
    rc, o = bash([], {"DAG2_DAG_READ": posix(p)})
    regress("the REAL entry with a read that is not the PINNED file stops at once: ZZV8FMINI-STOP, 'not the pinned DAG read'", rc == 1 and "ZZV8FMINI-STOP" in o and "pinned" in o and "waiting for the predecessor" not in o, o[-400:])
    rc, o = bash([])
    check("the REAL entry with the true read but no AFTER_OUT: refuses (`ZZV8FMINI-STOP AFTER_OUT is not set`), touches nothing", rc == 1 and "AFTER_OUT is not set" in o and "DAG read ROBUST" in o and "p = 0.1" in o, o[-400:])
    check("no ssh / scp call in any chain mode so far (shim log empty)", shim_calls() == "", shim_calls()[:300])
    # ---- the export list: it names every file box 2 reads (closure + data), nothing of B1, and fits
    clo = jload(HERE / "export_closure.json")
    m_exp = re.search(r"^EXPORT=\(\n(.*?)\n\)\n", src, flags=re.M | re.S)
    exported = re.findall(r'^  "(.*)"$', m_exp.group(1), flags=re.M) if m_exp else []
    need = set(clo["files"]) | {d["path"] for d in clo["data_reads"]}
    check(f"EXPORT list: exactly the closure ({len(clo['files'])} python files of the four entry scripts) + the {len(clo['data_reads'])} repo data files they open; {len(exported)} paths, no duplicates", set(exported) == need and len(exported) == len(set(exported)) == len(need), (len(exported), sorted(need - set(exported))[:3]))
    key = re.findall(r'^for f in (.*?); do$', src[src.index("KEYSHA=\"\""):], flags=re.M | re.S)[0] if "KEYSHA=\"\"" in src else ""
    keyfiles = re.findall(r'"?(\$PK[^" ]*|\$HJ[^" ]*|stack/[^" \\]+)"?', key.replace("\\\n", " "))
    resolved = [k.replace("$PK", L.PK).replace("$HJ", "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini") for k in keyfiles]
    check(f"the {len(resolved)} KEY files the chain sha-checks on box 2 are all inside the export", len(resolved) == 13 and all(k in need for k in resolved), [k for k in resolved if k not in need])
    check("the export does NOT carry the B1 lead block, the whole taniteval/ or the whole v8f-mini package (the DAG chain's 1.45 GB tar), and the size cap is 300 MB", "b1_eval_lead_block" not in src.split("EXPORT=(")[1].split(")\n")[0] and not any(p.split("/")[0] == "taniteval" and p.count("/") < 2 for p in exported)
          and re.search(r"^EXPORT_MAX_BYTES=314572800(?!\d)", src, re.M) is not None and "repo_v8fdag2_tip.tar" in src and src.count("repo_v8fdag_tip") == 1, "")
    sz = int(sh(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "cat-file", "-s", "8e121ea8687984538b7e772efb6b42f1ca1146c6:" + exported[0]])[1].strip() or 0)
    check("the export's first path exists on the tip (git cat-file)", sz > 0)
    # ---- the checkpoint assertions on box 2: DAG's three files, before the banks and after the last run
    check("the chain asserts box 2's DAG ckpt_last.pt, argv.json and summary.json (sha256) before the banks and after the last run, from the SHA256SUMS.txt values", 'ckptsha_b "before the banks"' in src and 'ckptsha_b "after the last run"' in src and "ARGV_SHA[$s]" in src and "SUMM_SHA[$s]" in src and "CKPT_SHA[$s]" in src)
    sums = [{l.split(" ", 1)[1].strip().lstrip("*"): l.split(" ", 1)[0] for l in (DAGART / f"M2b_s{s}_dag" / "SHA256SUMS.txt").read_text().splitlines()} for s in (0, 1)]
    mc = re.search(r"^CKPT_SHA=\((\w+) (\w+)\)", src, flags=re.M), re.search(r"^ARGV_SHA=\((\w+) (\w+)\)", src, flags=re.M), re.search(r"^SUMM_SHA=\((\w+) (\w+)\)", src, flags=re.M)
    check("the three sha arrays in the chain are exactly SHA256SUMS.txt's values (DAG_s0, DAG_s1), and the checkpoint values are the brief's", all(mc) and [mc[0].group(1), mc[0].group(2)] == [sums[0]["ckpt_last.pt"], sums[1]["ckpt_last.pt"]] == DAG_CK and [mc[1].group(1), mc[1].group(2)] == [sums[0]["argv.json"], sums[1]["argv.json"]]
          and [mc[2].group(1), mc[2].group(2)] == [sums[0]["summary.json"], sums[1]["summary.json"]], "")
    check("the M2b checkpoint hashes are NOT in the chain (the lineage root is not read by DAGGER2)", M2B_CK[0] not in src and M2B_CK[1] not in src)
    # ---- source order: bank -> judge -> stop -> publish -> DAG re-read -> gates -> trains
    def order_ok(s: str) -> bool:
        i_bank = s.find("for s in 0 1; do bank_one $s; done")
        i_judge = s.rfind('judge_banks "$ART/bank_s0" "$ART/bank_s1"; jr=$?')
        i_stop = s.find("if [ $jr != 0 ]; then", i_judge)
        i_stop_call = s.find('stop "$([ $jr = 3 ]', i_stop)
        i_pub = s.rfind("for s in 0 1; do publish_bank_one $s; done")
        i_dag2 = s.find('o2=$(sha256sum "$DAG_READ"', i_pub)
        i_gate = s.find('for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done')
        i_train = s.find('for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done')
        return 0 < i_bank < i_judge < i_stop < i_stop_call < i_pub < i_dag2 < i_gate < i_train and s.find("fi", i_stop_call) < i_pub
    check("chain source order: banks -> judge -> `if [ $jr != 0 ]` stop -> PUBLISH the banks -> DAG-read re-check -> ALL gates -> trains", order_ok(src))
    regress("chain source with the BLOCKED stop deleted: the order check goes RED", order_ok(src.replace('stop "$([ $jr = 3 ] && echo BLOCKED || echo REFUSED) at the bank stage (judge rc $jr): no gate was queued, nothing was trained (see $ART/DAG_OUTCOME.json)"', "true")) is False)
    regress("chain source with the judge's `if [ $jr != 0 ]` disabled: the order check goes RED", order_ok(src.replace("if [ $jr != 0 ]; then", "if false; then")) is False)
    check("the chain's bank/publish paths are the iteration-2 directory only (no `dagger_bank/s`), its logs / markers / job names are its own (no `v8fdag_`, `chain_v8f_dag.`, `ZZDAG-`, `v8f_guard_dag.sh`)", "dagger_bank/s" not in src and "v8fdag_" not in src.replace("the DAG chain's repo_v8fdag_tip.tar is left alone", "") and "ZZDAG-" not in src and "chain_v8f_dag." not in src and "v8f_guard_dag.sh" not in src and "dagger_bank_dag2/s$s" in src)
    # ---- --publish-only on a fake box that has the banks under dagger_bank_dag2
    fake = t / "fakebox"
    for s in (0, 1):
        d = fake / "dagger_bank_dag2" / f"s{s}"
        d.mkdir(parents=True)
        src_b = DAGART / f"bank_s{s}"
        man = jload(src_b / "bank.json")
        man["common"]["ckpt_sha256"] = DAG_CK[s]
        provs = []
        for k in range(4):
            stub = d / f"shard_{k:03d}of004.npz"
            stub.write_bytes(b"npz-stub-%d-%d" % (s, k))
            man["shards"][k]["file_sha256"] = sha_file(stub)
            sp = jload(src_b / f"shard_{k:03d}of004.provenance.json")
            sp["file_sha256"] = sha_file(stub)
            jdump(d / f"shard_{k:03d}of004.provenance.json", sp)
        jdump(d / "bank.json", man)
        prov = jload(src_b / "PROVENANCE.json")
        prov["bank_sha256"] = sha_file(d / "bank.json")
        jdump(d / "PROVENANCE.json", prov)
        (d / "bank_stage.log").write_text("stage log\n")
    art = t / "art"
    rc, o = bash(["--publish-only", posix(fake)], {"DAG2_ART": posix(art)})
    pub = {s: sorted(p.name for p in (art / f"bank_s{s}").glob("*")) for s in (0, 1)} if art.exists() else {}
    check("--publish-only on a stand-in box with the banks in dagger_bank_dag2\\s<S>: rc 0, ZZDAG2-PUBLISH-ONLY-DONE, 11 files + SHA256SUMS.txt per seed in <ART>/bank_s<S>", rc == 0 and "ZZDAG2-PUBLISH-ONLY-DONE" in o and all(len(v) == 12 and "SHA256SUMS.txt" in v for v in pub.values()) and len(pub) == 2, (rc, o[-300:], {k: len(v) for k, v in pub.items()}))
    old_layout = t / "oldbox"
    shutil.copytree(fake, old_layout)
    shutil.move(str(old_layout / "dagger_bank_dag2"), str(old_layout / "dagger_bank"))
    (old_layout / "dagger_bank_dag2").mkdir()
    for s in (0, 1):
        (old_layout / "dagger_bank_dag2" / f"s{s}").mkdir()
    rc, o = bash(["--publish-only", posix(old_layout)], {"DAG2_ART": posix(t / "art2")})
    regress("--publish-only on a box whose banks are in the ITERATION-1 directory (dagger_bank\\s<S>) and none in dagger_bank_dag2: rc 1 (the chain looks only in dagger_bank_dag2)", rc != 0 and "ZZDAG2-PUBLISH-ONLY-DONE" not in o, o[-300:])
    rc, o = bash(["--publish-only", posix(fake)])
    regress("--publish-only without DAG2_ART set refuses (it never writes the real artifact directory)", rc == 1 and "needs DAG2_ART" in o, o[-200:])
    check("no ssh / scp call in any chain mode (shim log empty)", shim_calls() == "", shim_calls()[:300])
    # ---- the guard, waker and launcher
    g_dag = (Path("C:/Users/Admin/box2_ship") / "v8f_guard_dag.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    g_dag2 = (CHAIN_DIR / "v8f_guard_dag2.sh").read_text(encoding="utf-8")
    check("v8f_guard_dag2.sh == the installed v8f_guard_dag.sh except the first comment line (anchored end test, bank-generator kill list unchanged)", g_dag.split("\n", 2)[2] == g_dag2.split("\n", 2)[2] and "'^(ZZV8FMINI|" not in g_dag2.split("\n", 1)[0] and "(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)" in g_dag2 and "v8f_dagger_bank|v8f_dagger_queue_tool" in g_dag2)
    cmdtxt = (CHAIN_DIR / "run_chain_v8f_dag2.cmd").read_bytes()
    check("run_chain_v8f_dag2.cmd: CRLF, AFTER_OUT = the DAG chain's log, runs chain_v8f_dag2.sh into chain_v8f_dag2.out", b"AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dag.out" in cmdtxt and b"chain_v8f_dag2.sh" in cmdtxt and b"chain_v8f_dag2.out" in cmdtxt and cmdtxt.count(b"\r\n") == cmdtxt.count(b"\n"))
    wk = (CHAIN_DIR / "wake_v8fdag2.sh").read_text(encoding="utf-8")
    check("wake_v8fdag2.sh watches chain_v8f_dag2.out, the DAG2 markers and gate lines (and bash -n is clean)", "chain_v8f_dag2.out" in wk and "ZZDAG2-BLOCKED" in wk and "DAG2 gate: PASS" in wk and "v8fdag2_guard.out" in wk and sh([BASH, "-n", posix(CHAIN_DIR / "wake_v8fdag2.sh")])[0] == 0)
    check("bash -n is clean on the chain, its template and the guard", all(sh([BASH, "-n", posix(p)])[0] == 0 for p in (CHAIN, CHAIN_DIR / "chain_v8f_dag2.sh.in", CHAIN_DIR / "v8f_guard_dag2.sh")))


# ================================================================================================ E. gate assertions on the REAL DAG corpus gate
def section_e():
    say("== E. gate_dag2_assert.py on the REAL DAG corpus gate (selfcheck.json written by the real trainer on box 2) and doctored copies")
    t = W / "gatee"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    seed = 0
    d = DAGART / f"M2b_s{seed}_dag"
    sc_path = d / "gate" / "selfcheck.json"          # the GATE's selfcheck (the run's own M2b_s0_dag/selfcheck.json is the training run's: its argv_sha is the training argv's)
    bank_json = DAGART / f"bank_s{seed}" / "bank.json"
    bsha = sha_file(bank_json)
    pin_real = (d / "gate" / "pin.txt").read_text(encoding="utf-8").strip() if (d / "gate" / "pin.txt").exists() else ""
    gl = lines_for(seed, "dag", bsha)                       # the ITERATION-1 stage lines of the new queue with the REAL bank sha: what box 2 ran for DAG
    sc = jload(sc_path)
    check("the dry-run GATE line for DAG s0 (real bank sha256) reproduces the line box 2 actually ran: sha256(' '.join(argv))[:16] == the real selfcheck.json's argv_sha", sc.get("argv_sha") == hashlib.sha256(" ".join(GA.argv_of(gl["gate"])).encode()).hexdigest()[:16], sc.get("argv_sha"))
    check("the dry-run TRAIN line for DAG s0 is the pin box 2 wrote (gate/pin.txt)", GA.argv_of(gl["train"]) == GA.argv_of(pin_real), "")
    (t / "g.txt").write_text(gl["gate"], encoding="utf-8")
    (t / "t.txt").write_text(gl["train"], encoding="utf-8")
    (t / "p.txt").write_text(pin_real, encoding="utf-8")
    expect_dag = {"bank_dir_tail": "/dagger_bank/s{seed}/bank.json", "init_tail": "/M2b_s{seed}/ckpt_last.pt"}                     # DAG's own tails: this is DAG standing in for DAG2 to feed the REAL corpus numbers
    jdump(t / "expect_dag.json", expect_dag)

    def run(selfcheck=sc_path, arm="dag2", expect=t / "expect_dag.json", init_sha=M2B_CK[0], extra=None, gate=t / "g.txt", train=t / "t.txt", pin=t / "p.txt", bsum=d / "summary.json", p="0.1", bank_sha=bsha, mode="check"):
        a = [mode, "--selfcheck", str(selfcheck), "--gate-line", str(gate), "--train-line", str(train), "--pin-line", str(pin), "--arm", arm, "--seed", str(seed), "--banked-init-sha", init_sha, "--b-summary", str(bsum)]
        if arm == "dag2":
            a += ["--p", p, "--bank-sha", bank_sha, "--bank-json", str(bank_json)]
        if expect:
            a += ["--expect", str(expect)]
        a += extra or []
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                rc = GA.main(a)
            except SystemExit as e:
                rc = e.code
        return rc, buf.getvalue()
    # the DAG2 asserter accepts the REAL DAG gate when told DAG's tails (corpus numbers, key names, B-comparison, windows 306,566, holdout sha, lru ...)
    rc, o = run()
    check("the DAG2 asserter PASSES the REAL DAG corpus gate (as DAG2 with DAG's path tails): 4,512 clips, 306,566 windows, holdout sha, K-B2 vs the init, COVERAGE, B-summary comparison, queue lines", rc == 0 and "DAG2 gate: PASS" in o, o[-500:])
    rc, o = run(expect=None)
    regress("with the DAG2 defaults (init tail ..._dag\\ckpt_last.pt, bank dir dagger_bank_dag2) the REAL iteration-1 DAG gate is REFUSED: the init path and the bank path are not DAG2's", rc == 1 and "init_ckpt path" in o and "built bank path" in o, o[-500:])
    def doctor(fn, name):
        s = jload(sc_path)
        fn(s)
        jdump(t / f"{name}.json", s)
        return t / f"{name}.json"
    cases = [
        ("init sha256 is not the expected generator's", lambda s: s["levers"]["init_ckpt"].__setitem__("sha256", "0" * 64), "does not start from"),
        ("K-B2: the bank's checkpoint differs from the init", lambda s: s["info"]["dagger"]["kb2"].__setitem__("ckpt_sha256", "1" * 64), "K-B2"),
        ("built dagger_p 0.3", lambda s: s["levers"].__setitem__("dagger_p", 0.3), "built dagger_p"),
        ("built bank sha256 is another bank's", lambda s: s["levers"].__setitem__("dagger_bank_sha256", "2" * 64), "bank sha256"),
        ("COVERAGE blocked", lambda s: s["info"]["dagger"]["coverage"].__setitem__("blocked", True), "COVERAGE"),
        ("dagger_probe failed", lambda s: s["dagger_probe"].__setitem__("pass", False), "dagger_probe"),
        ("probe control p0 loss diff not 0", lambda s: s["dagger_probe"].__setitem__("control_p0_loss_diff", 1e-9), "control_p0_loss_diff"),
        ("ego perturbation not off", lambda s: s["ego_perturb_probe"].__setitem__("ego_perturb", "on"), "ego_perturb"),
        ("steps 2999", lambda s: s.__setitem__("steps_total", 2999), "steps"),
        ("holdout set differs", lambda s: s["info"].__setitem__("excluded_sha256", "3" * 64), "holdout"),
        ("window count differs", lambda s: s["info"].__setitem__("n_windows_all_clips", 306565), "n_windows_all_clips"),
        ("a dead parameter", lambda s: s["live"].__setitem__("dead_params", ["x"]), "G-LIVE dead set"),
        ("the gate's argv_sha differs from the queue line", lambda s: s.__setitem__("argv_sha", "0" * 16), "argv_sha"),
        ("n_params_trainable differs from B's", lambda s: s["levers"].__setitem__("n_params_trainable", s["levers"]["n_params_trainable"] + 1), "n_params_trainable"),
    ]
    for name, fn, key in cases:
        rc, o = run(selfcheck=doctor(fn, "d_" + hashlib.sha256(name.encode()).hexdigest()[:6]))
        regress(f"asserter: {name} -> REFUSED (names {key!r})", rc == 1 and key in o, o[-300:])
    rc, o = run(init_sha=DAG_CK[0])
    regress("asserter: the expected init is DAG_s0's sha but the gate started from M2b (the iteration-1 run offered as DAG2's gate) -> REFUSED", rc == 1 and "does not start from" in o, o[-300:])
    rc, o = run(p="0.3")
    regress("asserter: --p 0.3 -> REFUSED (built dagger_p / the lines do not carry 0.3)", rc == 1, o[-200:])
    bad_pin = t / "pin_bad.txt"
    bad_pin.write_text(pin_real.replace("--steps 3000", "--steps 3001"), encoding="utf-8")
    rc, o = run(pin=bad_pin)
    regress("asserter: a pin that is not the planned training argv -> REFUSED", rc == 1 and "PIN" in o, o[-300:])
    rc, o = run(arm="dag20")
    regress("asserter: arm dag20 offered the DAG (bank-carrying) gate -> REFUSED", rc == 1, o[-300:])
    rc, o = run(mode="token", extra=["--commit", "0" * 40, "--queue-sha", "ab" * 32, "--key-sha-json", "{}", "--dag-read-sha", DAG_READ_SHA, "--run", "M2b_s0_dag2", "--out", str(t / "token.json")])
    tok = jload(t / "token.json") if (t / "token.json").exists() else {}
    check("the token is written after the assertions pass: prereg PREREG_V8F_DAGGER2, binds the sha256 of the training argv, carries dag_read_sha256 and the init checkpoint's sha", rc == 0 and tok.get("prereg") == "PREREG_V8F_DAGGER2" and tok.get("dag_read_sha256") == DAG_READ_SHA
          and tok["binding"]["argv_sha256"] == L.argv_sha256(GA.argv_of(pin_real)) and tok.get("init_ckpt_sha256") == M2B_CK[0] and "orc_read_sha256" not in tok, (rc, o[-200:]))
    (t / "token.json").unlink(missing_ok=True)
    rc, o = run(selfcheck=doctor(lambda s: s["levers"].__setitem__("dagger_p", 0.3), "tok_bad"), mode="token", extra=["--commit", "0" * 40, "--queue-sha", "ab" * 32, "--key-sha-json", "{}", "--dag-read-sha", DAG_READ_SHA, "--run", "x", "--out", str(t / "token.json")])
    regress("no token is written when an assertion fails", rc == 1 and not (t / "token.json").exists(), o[-200:])


# ================================================================================================ F. function-level proofs of the EDITED chain blocks (their own variables, run as bash against stubs; no box 2)
def snippet(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i) + len(end)
    return src[i:j]


def run_bash_text(text: str, env=None, name="snip"):
    f = W / "tmp" / f"{name}.sh"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    return sh([BASH, posix(f)], env=env, timeout=240)


def section_f():
    say("== F. the chain's edited blocks run as bash against stubs: box-2 checkpoint assertions, the dev-box DAG staging, the export tar")
    src = CHAIN.read_text(encoding="utf-8")
    arrays = "\n".join(re.findall(r"^(?:CKPT_SHA|ARGV_SHA|SUMM_SHA)=\(.*$", src, flags=re.M))
    sums = [{l.split(" ", 1)[1].strip().lstrip("*"): l.split(" ", 1)[0] for l in (DAGART / f"M2b_s{s}_dag" / "SHA256SUMS.txt").read_text().splitlines()} for s in (0, 1)]
    fn = snippet(src, "ckptsha_b() {", "\n}\n")
    R = "D:/tanitad/runs/v8f_mini_full"
    good = {f"{R}/M2b_s{s}_dag/{f}": sums[s][f] for s in (0, 1) for f in ("ckpt_last.pt", "argv.json", "summary.json")}

    def harness(hashes):
        tbl = "\n".join(f"{k} {v}" for k, v in hashes.items())
        return (f'R={R}\n{arrays}\nstop() {{ echo "STOP: $*"; exit 1; }}\nckset() {{ echo "CKSET $1 $2"; }}\n'
                f'cat > "$TMPDIR/hashes.txt" <<"EOT"\n{tbl}\nEOT\n'
                f'rsha() {{ grep -F -- "$1 " "$TMPDIR/hashes.txt" | head -1 | cut -d" " -f2; }}\n{fn}\nckptsha_b "TEST"\n')
    rc, o = run_bash_text(harness(good), name="ckpt_good")
    check("box-2 assertion (ckptsha_b) PASSES when DAG_s0 / DAG_s1 ckpt_last.pt, argv.json, summary.json hash to SHA256SUMS.txt's values", rc == 0 and "are the banked DAG runs'" in o and "CKSET M2b_s0_dag " in o and "CKSET M2b_s1_dag " in o, o[-300:])
    for s in (0, 1):
        for f in ("ckpt_last.pt", "argv.json", "summary.json"):
            bad = dict(good)
            bad[f"{R}/M2b_s{s}_dag/{f}"] = "0" * 64
            rc, o = run_bash_text(harness(bad), name="ckpt_bad")
            regress(f"box-2 assertion: DAG_s{s}'s {f} on box 2 differs -> STOP naming M2b_s{s}_dag/{f}", rc == 1 and f"STOP: TEST: box 2's M2b_s{s}_dag/{f}" in o, o[-200:])
    nohash = {k: v for k, v in good.items() if not k.endswith("M2b_s1_dag/argv.json")}
    rc, o = run_bash_text(harness(nohash), name="ckpt_none")
    regress("box-2 assertion: box 2 cannot be hashed for one file (no answer) -> STOP 'cannot hash'", rc == 1 and "cannot hash" in o, o[-200:])
    m2b = {k.replace("_dag/", "/"): v for k, v in good.items()}
    rc, o = run_bash_text(harness(m2b), name="ckpt_m2b")
    regress("box-2 assertion: box 2 offers M2b_s<S>/ files instead of DAG's -> STOP (the box has no M2b_s<S>_dag hash)", rc == 1 and "cannot hash" in o, o[-200:])
    # ---- the dev-box staging of DAG as the gate comparator
    blk = snippet(src, "for s in 0 1; do      # the dev box's pulled copy of DAG", 'stop "stage DAG_s$s/summary.json as the gate comparator"\ndone')
    stage_dir = W / "stage_E"

    def stage_run(arr_text, dag1=str(DAGART)):
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        return run_bash_text(f'{arr_text}\nstop() {{ echo "STOP: $*"; exit 1; }}\nE={posix(stage_dir)}\nDAG1_DIR={posix(dag1)}\n{blk}\necho STAGED\n', name="stage_blk")
    rc, o = stage_run(arrays)
    check("dev-box staging: the pulled DAG ckpt / argv / summary are hashed against SHA256SUMS.txt's values and summary.json is staged as the comparator", rc == 0 and "STAGED" in o and sha_file(stage_dir / "runs" / "M2b_s0" / "summary.json") == sums[0]["summary.json"] and sha_file(stage_dir / "runs" / "M2b_s1" / "summary.json") == sums[1]["summary.json"], o[-300:])
    tampered = re.sub(r"^SUMM_SHA=\((\w+) (\w+)\)", lambda m: f"SUMM_SHA=({m.group(1)} {'f' * 64})", arrays, flags=re.M)
    rc, o = stage_run(tampered)
    regress("dev-box staging: a pulled summary.json that is not the banked one -> STOP naming the file", rc == 1 and "summary.json sha256" in o and "STAGED" not in o, o[-300:])
    rc, o = stage_run(arrays, dag1=str(W / "nonexistent_dag"))
    regress("dev-box staging: the pulled DAG run directory is missing -> STOP", rc == 1 and "is missing" in o, o[-300:])
    # ---- the export tar, the real git archive on the real mirror
    blk2 = snippet(src, 'for xf in "${EXPORT[@]}"; do', '$SCP $B/repo_v8fdag2_tip.tar')
    blk2 = blk2[:blk2.rindex("\n")]
    exp_def = snippet(src, "EXPORT=(\n", "\n)\n")
    T = "8e121ea8687984538b7e772efb6b42f1ca1146c6"
    tdir = W / "tar_t"

    def tar_run(max_bytes=314572800, extra_path=None):
        if tdir.exists():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True)
        ed = exp_def if not extra_path else exp_def.replace("EXPORT=(\n", f'EXPORT=(\n  "{extra_path}"\n', 1)
        txt = (f'stop() {{ echo "STOP: $*"; exit 1; }}\nPK="{L.PK}"\nMIRROR=/c/Users/Admin/tanitad-push\nT={T}\nB={posix(tdir)}\nEXPORT_MAX_BYTES={max_bytes}\n{ed}\n'
               f'export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="*"\n{blk2}\necho TAR-OK\n')
        return run_bash_text(txt, name="tar_blk")
    rc, o = tar_run()
    tar = tdir / "repo_v8fdag2_tip.tar"
    sz = tar.stat().st_size if tar.exists() else 0
    names = sh(["tar", "-tf", posix(tar)])[1].split("\n") if tar.exists() else []
    check(f"export tar: the real `git archive` of the 63 paths at the tip builds a tar of {sz / 1e6:.1f} MB (<= 300 MB), named repo_v8fdag2_tip.tar (the DAG chain's repo_v8fdag_tip.tar is a different name)", rc == 0 and "TAR-OK" in o and 0 < sz <= 314572800, o[-300:])
    check("export tar: it contains the queue, the eval grid, eval141_kin / join_eval141, the thresholds file, the P10 prior, the trainer / eval / bank / tool; and NOTHING from taniteval/results, B1 or the 986 MB v8f-mini package",
          any(n.endswith("code/v8f_mini_box2_queue.cmd") for n in names) and any(n.endswith("raw/eval141_kin.npz") for n in names) and any(n.endswith("raw/join_eval141.npz") for n in names) and any(n.endswith("THRESHOLDS_FIXED.txt") for n in names)
          and any(n.endswith("p10_prior.pt") for n in names) and any(n.endswith("stack/scripts/v8f_dagger_bank.py") for n in names) and not any("taniteval/results" in n or "b1_eval_lead_block" in n for n in names), names[:5])
    rc, o = tar_run(max_bytes=1000)
    regress("export tar: a tar above the cap (1000 bytes) -> STOP 'must be 1..'", rc == 1 and "must be 1.." in o and "TAR-OK" not in o, o[-300:])
    rc, o = tar_run(extra_path="stack/scripts/no_such_file.py")
    regress("export tar: a listed path that is not on the tip -> STOP before the archive is built", rc == 1 and "the tip has no stack/scripts/no_such_file.py" in o and "TAR-OK" not in o, o[-300:])


# ================================================================================================ main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    secs = "A,S,B,C,D,E,F"
    out = HERE / "selftest_output.txt"
    sweep = True
    if "--sections" in argv:
        secs = argv[argv.index("--sections") + 1]
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    if "--no-sweep" in argv:
        sweep = False
    t0 = time.time()
    for s in secs.split(","):
        {"A": section_a, "S": lambda: section_s(sweep), "B": section_b, "C": section_c, "D": section_d, "E": section_e, "F": section_f}[s]()
    ok = N["check_fail"] == 0 and N["reg_missed"] == 0
    say(f"\nSUMMARY: {N['check'] - N['check_fail']}/{N['check']} checks pass, {N['reg'] - N['reg_missed']}/{N['reg']} deliberate regressions caught, {N['reg_missed']} missed  ({time.time() - t0:.0f} s)  -> {'ALL GREEN' if ok else 'RED'}")
    if out:
        Path(out).write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
