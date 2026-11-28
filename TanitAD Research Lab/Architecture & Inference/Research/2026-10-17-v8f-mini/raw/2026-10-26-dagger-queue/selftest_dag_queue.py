#!/usr/bin/env python3
"""selftest_dag_queue.py -- the proof for the box-2 DAgger queue stages and chain (PREREG_V8F_DAGGER section 8 + Amendment 1). CPU only, no box 2, nothing launched, nothing outside dagq/.

Every check is either a CHECK (must hold) or a DELIBERATE REGRESSION (a defect put in on purpose; the instrument must CATCH it, and a regression that is not caught is a FAILURE of the proof). Sections:

  A  the REAL queue .cmd through cmd.exe (V8F_DRYRUN): the argv of every stage: flag values, the pair rule, gate == train, p, ckpt paths, ORC-0 + --dagger-p 0, the other stages unchanged
  B  the REAL queue .cmd through cmd.exe, NOT dry, with a fake box (stub trainer / eval / bank generator): the bank fan-out, the pin, byte-identity of the argv that runs, refusals
  C  the chain's two decisions on files: the ORC predicate (both sides) and the bank judge (controls then COVERAGE; both sides, 0.5 inclusive)
  D  the chain script itself: --judge-only (BLOCKED stops before any gate), --dry-run, --check-tree (the lever greps; fails on the pre-change tip), the source order of bank -> judge -> PUBLISH -> gates
  E  gate_dag_assert.py on a REAL selfcheck.json written by the real trainer's gate (real_gate_proof.py), + mutated copies
  F  the shard tool (4 shards, a failing shard, a skip), the guard copy, the banked checkpoint hashes, the additive queue diff
  G  the bank files the chain PUBLISHES to the dev box (the four kinds the dev-box reader needs + the stage log, 11 names, SHA256SUMS.txt): bank-files / bank-verify / bank-plan, in-process, both sides
  H  the chain's own publish flow offline (--publish-only: the REAL bank_fetch_small -> judge -> publish_bank_one against a local stand-in for box 2): nothing is moved before the judge passes, every fault stops it and leaves no seal
Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python selftest_dag_queue.py     (writes selftest_output.txt)
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
sys.path.insert(0, str(HERE / "box2_ship"))
if os.environ.get("DAGQ_CHECKS_DIR"):
    sys.path.insert(0, os.environ["DAGQ_CHECKS_DIR"])        # the mutation runner points section C at a MUTATED copy of v8f_dag_chain_checks.py
import dagq_lib as L  # noqa: E402
import v8f_dag_chain_checks as CK  # noqa: E402

BT = HERE / "bt"
CHAIN = Path(os.environ["DAGQ_CHAIN"]) if os.environ.get("DAGQ_CHAIN") else HERE / "box2_ship" / "chain_v8f_dag.sh"      # the mutation runner points sections D / G at a MUTATED copy of the chain (its helper scripts beside it)
BASH = r"C:\Program Files\Git\bin\bash.exe"
SHIP = Path("C:/Users/Admin/box2_ship")
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
    """A deliberate regression: ``caught`` must be True (the instrument saw the defect)."""
    N["reg"] += 1
    if not caught:
        N["reg_missed"] += 1
    say(("CAUGHT " if caught else "MISSED ") + "(deliberate regression) " + name + (f"   [{str(detail)[:400]}]" if (detail and not caught) else ""))
    return bool(caught)


def sh(cmd, env=None, cwd=None, timeout=600):
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(BT).replace("\\", "/"), **(env or {})}
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def posix(p) -> str:
    s = str(p).replace("\\", "/")
    return s


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1), encoding="utf-8")


# ================================================================================================ A. rendering
def names_ego0(av) -> bool:
    """`--ego-perturb-p 0` named exactly once, value 0 (PREREG_V8F_DAGGER section 4)."""
    return av.count("--ego-perturb-p") == 1 and av[av.index("--ego-perturb-p") + 1] == "0"


def section_a():
    say("== A. the real queue through cmd.exe, V8F_DRYRUN: what every stage would run")
    S = L.SHA_FAKE
    lines = {}
    for seed in (0, 1):
        for arm, parg in (("dag", " 0.1"), ("dag0", "")):
            g = L.dry(f"gatedag {arm} {seed}{parg}", sha=S)
            t = L.dry(f"traindag {arm} {seed}{parg}", sha=S)
            e = L.dry(f"evaldag {arm} {seed}")
            lines[(arm, seed)] = {"gate": L.trainer_line(g), "train": L.trainer_line(t), "eval": e}
    for seed in (0, 1):
        d, d0 = lines[("dag", seed)], lines[("dag0", seed)]
        ta, t0 = L.train_argv(d["train"]), L.train_argv(d0["train"])
        fa, f0 = L.flags(ta), L.flags(t0)
        ck = f"M2b_s{seed}\\ckpt_last.pt"
        check(f"s{seed} DAG: --dagger-p 0.1 once; bank = RUNSM\\dagger_bank\\s{seed}; sha = the stage's; --dagger-seed {seed}; --ego-perturb-p 0",
              fa.get("--dagger-p") == [["0.1"]] and fa.get("--dagger-bank") == [[f"D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank\\s{seed}"]] and fa.get("--dagger-bank-sha256") == [[S]] and fa.get("--dagger-seed") == [[str(seed)]]
              and fa.get("--ego-perturb-p") == [["0"]], {k: fa.get(k) for k in ("--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed", "--ego-perturb-p")})
        check(f"s{seed} DAG: the fine-tune of ORC Amendment 1 (init = this seed's M2b, 3000 steps, lr 1e-4, warmup 100, p_gt 0.2 -> 0.2, --seed {seed})",
              fa.get("--init-ckpt") == [[f"D:\\tanitad\\runs\\v8f_mini_full\\{ck}"]] and fa.get("--steps") == [["3000"]] and fa.get("--lr") == [["1e-4"]] and fa.get("--warmup") == [["100"]]
              and fa.get("--p-gt-start") == [["0.2"]] and fa.get("--p-gt-end") == [["0.2"]] and fa.get("--seed") == [[str(seed)]] and fa.get("--arm") == [["M2b"]] and fa.get("--grid") == [["pool4x10"]], {k: fa.get(k) for k in ("--init-ckpt", "--steps", "--lr", "--warmup", "--seed")})
        check(f"s{seed} DAG-0: names --dagger-p 0 once and NONE of the three DAG-only flags (the trainer's gate refuses them at p = 0); --ego-perturb-p 0; no ego law flag",
              f0.get("--dagger-p") == [["0"]] and not any(k in f0 for k in L.DAG_ONLY) and f0.get("--ego-perturb-p") == [["0"]] and not any(k.startswith("--ego-perturb-") and k != "--ego-perturb-p" for k in f0), sorted(f0))
        second = ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--w-acc", "--code-mix", "--o-interface", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly", "--no-consistency", "--detach-upper", "--no-resume")
        check(f"s{seed}: no second lever on either arm", not any(k in fa or k in f0 for k in second))
        # the PAIR rule: same init, steps, data, seed; the argv differs ONLY in the DAgger flags and --out
        a1, a0 = L.drop_flag(ta, {"--out", "--dagger-p"} | L.DAG_ONLY), L.drop_flag(t0, {"--out", "--dagger-p"})
        check(f"s{seed} PAIR RULE: DAG and DAG-0 argv are identical once --out, --dagger-p and the three DAG-only flags are removed", a1 == a0 and len(a1) > 40, f"{len(a1)} vs {len(a0)} tokens")
        check(f"s{seed} PAIR RULE: --out differs and neither run directory is the banked M2b or an ORC directory", fa["--out"] != f0["--out"] and fa["--out"][0][0].endswith(f"M2b_s{seed}_dag") and f0["--out"][0][0].endswith(f"M2b_s{seed}_dag0"))
        check(f"s{seed} PAIR RULE: DAG names exactly the three extra flags {sorted(L.DAG_ONLY)}; every flag both name has the same value except --out and --dagger-p",
              set(fa) - set(f0) == L.DAG_ONLY and set(f0) <= set(fa) and {k for k in f0 if fa[k] != f0[k]} == {"--out", "--dagger-p"}, (sorted(set(fa) - set(f0)), sorted(k for k in f0 if fa[k] != f0[k])))
        for arm in ("dag", "dag0"):
            gl, tl = lines[(arm, seed)]["gate"], lines[(arm, seed)]["train"]
            ga, tr = L.train_argv(gl), L.train_argv(tl)
            check(f"s{seed} {arm}: GATE argv == TRAIN argv with --out and --selfcheck-only changed, --selfcheck-only exactly once",
                  L.drop_flag(ga, {"--out"}, {"--selfcheck-only"}) == L.drop_flag(tr, {"--out"}) and ga.count("--selfcheck-only") == 1 and "--selfcheck-only" not in tr)
            check(f"s{seed} {arm}: gate --out is its own gate_M2b-{arm}_s{seed}, never the run directory", L.flags(ga)["--out"][0][0].endswith(f"gate_M2b-{arm}_s{seed}") and L.flags(ga)["--out"] != L.flags(tr)["--out"])
        el = [x for x in lines[("dag", seed)]["eval"] if "v8f_mini_eval.py" in x][0]
        et = L.tokens(el)
        check(f"s{seed}: the ONE T0 eval is the standard one: tag M2b-dag_s{seed}, NO --ages (the default cycle), this run's checkpoint, --project --families (K4's fresh-age comparator is NOT run on box 2: dev-box CPU, PREREG section 5)",
              "--ages" not in et and et[et.index("--tag") + 1] == f"M2b-dag_s{seed}" and et[et.index("--ckpt") + 1].endswith(f"M2b_s{seed}_dag\\ckpt_last.pt") and et[-2:] == ["--project", "--families"], el[:200])
        # PREREG section 4: "`--ego-perturb-p 0` is named on both arms" -- as the token pair, exactly once, in the gate line and the train line of BOTH arms
        for arm in ("dag", "dag0"):
            for kind in ("gate", "train"):
                av = L.train_argv(lines[(arm, seed)][kind])
                check(f"s{seed} {arm} {kind}: PREREG section 4 -- `--ego-perturb-p 0` is NAMED, exactly once, with the value 0 (the flag's default would disable the lever by omission)", names_ego0(av), [x for x in av if "ego" in x])
        check(f"s{seed}: the ego-perturbation check itself is not blind -- an argv with the flag dropped, with a value 0.3, or with it twice FAILS it",
              not names_ego0(L.drop_flag(L.train_argv(lines[("dag", seed)]["train"]), {"--ego-perturb-p"})) and not names_ego0(L.train_argv(lines[("dag0", seed)]["train"]) + ["--ego-perturb-p", "0.3"])
              and not names_ego0(L.train_argv(lines[("dag", seed)]["train"]) + ["--ego-perturb-p", "0"]))
        regress(f"s{seed}: no `evaldagfresh` stage exists in the queue any more (O1: K4's fresh-age comparator runs on the dev-box CPU, not here) -- the queue answers `usage` with rc 2 and prints no eval line",
                (lambda r: r[0] == 2 and not any("v8f_mini_eval.py" in x for x in r[1]) and any(x.startswith("usage") for x in r[1]))(L.cmd(f"evaldagfresh dag {seed}")))
    # ORC-0 + --dagger-p 0 (PREREG section 4: "DAG-0's argv equals ORC-0's plus --dagger-p 0")
    for seed in (0, 1):
        rc, o = L.cmd(f"trainorc orc0 {seed}", queue=L.TIP_QUEUE)
        orc0 = L.train_argv(L.trainer_line(o))
        dag0 = L.train_argv(lines[("dag0", seed)]["train"])
        check(f"s{seed}: DAG-0 argv == the TIP queue's ORC-0 argv + `--dagger-p 0`, once --out is removed (the replicate-floor claim of PREREG section 4)", L.drop_flag(dag0, {"--out"}) == L.drop_flag(orc0, {"--out"}) + ["--dagger-p", "0"], f"{len(dag0)} vs {len(orc0)}")
    # p and the argument validation
    for args, why in (("gatedag dag 0", "dag without P"), ("gatedag dag 0 0.2", "P = 0.2"), ("gatedag dag 0 0", "P = 0 on dag"), ("gatedag dag0 0 0.1", "dag0 with a P"), ("gatedag dag0 0 0", "dag0 with a P of 0 (a lever declared and inert)"),
                      ("gatedag dagx 0 0.1", "an unknown arm"), ("gatedag dag 2 0.1", "seed 2"), ("gatedag dag 0.1", "a missing seed"), ("traindag dag 0 0.5", "train with P 0.5"), ("evaldag dag 5", "eval seed 5"), ("bankdag 7", "bank seed 7")):
        rc, o = L.cmd(args, env={"DAGSHA_DRY": L.SHA_FAKE})
        regress(f"`{args}` ({why}) is REFUSED, rc 2, says REFUSED, and prints no trainer line", rc == 2 and any("REFUSED" in x for x in o) and not any("v8f_mini_train.py" in x for x in o), f"rc {rc} {o[:2]}")
    for p in ("0.1", "0.3"):
        rc, o = L.cmd(f"gatedag dag 0 {p}", env={"DAGSHA_DRY": S})
        check(f"gatedag dag 0 {p} is accepted and carries --dagger-p {p}", rc == 0 and L.flags(L.train_argv(L.trainer_line(o)))["--dagger-p"] == [[p]])
    # the other stages are untouched
    for args in ("trainorc orc0 0", "trainorc orc 1", "gateorc orc 0", "gateorc orc0 1", "evalorc orc 1", "gate", "train M2b 0", "eval M2b 0", "gatehist", "trainoacc 0"):
        r1, o1 = L.cmd(args, queue=L.TIP_QUEUE, env={"WACC": "0.1"})
        r2, o2 = L.cmd(args, queue=L.QUEUE, env={"WACC": "0.1"})
        check(f"existing stage `{args}` prints byte-identical lines from the tip queue and the new queue", r1 == r2 and o1 == o2 and len(o1) > 0, f"{r1} {r2}")
    # the bank line
    for seed in (0, 1):
        o = L.dry(f"bankdag {seed}")
        t = L.tokens(o[0])
        f = L.flags(t[2:])
        gen = t[t.index("--") + 1:]
        gf = L.flags(gen)
        check(f"s{seed} bank: 4 shards into dagger_bank\\s{seed}; from M2b_s{seed}'s checkpoint and argv.json; the cache, tracks, join and eval grid as arguments; --expect-clips 4512 --expect-windows 306566; on cuda",
              f.get("--n-shards") == [["4"]] and f.get("--out-dir") == [[f"D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank\\s{seed}"]] and gf.get("--ckpt") == [[f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\ckpt_last.pt"]]
              and gf.get("--m2b-argv") == [[f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\argv.json"]] and gf.get("--expect-clips") == [["4512"]] and gf.get("--expect-windows") == [["306566"]] and gf.get("--device") == [["cuda"]]
              and all(k in gf for k in ("--cache-dir", "--tracks", "--join-train", "--eval-grid")), sorted(gf))
        check(f"s{seed} bank: no ampersand reaches an argument (the eval grid is the COPY), and the line has no --shard / --n-shards / --out-dir among the generator arguments",
              "&" not in o[0] and not any(k in gf for k in ("--shard", "--n-shards", "--out-dir")))


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
    if os.environ.get("STUB_SLOW_SHARD") == str(k):
        time.sleep(60)
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
    shutil.copy(HERE / "stage" / "stack" / "scripts" / "v8f_dagger_queue_tool.py", sc / "v8f_dagger_queue_tool.py")
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
        (runsm / f"M2b_s{s}").mkdir(parents=True)
        (runsm / f"M2b_s{s}" / "ckpt_last.pt").write_bytes(b"ckpt%d" % s)
        (runsm / f"M2b_s{s}" / "argv.json").write_text(json.dumps({"argv": []}))
    (runsm / "join_train.npz").write_bytes(b"j")
    return {"REPO": str(repo), "PY": L.PY, "CACHEFULL": str(cache), "V9": str(v9), "RUNSM": str(runsm)}


def tokens_of_file(p) -> list:
    return L.tokens(Path(p).read_text(encoding="utf-8"))


def section_b():
    say("== B. the real queue through cmd.exe, NOT dry, on a fake box (stub trainer / eval / bank generator; the real shard tool)")
    root = BT / "fakebox"
    env = fake_world(root)
    R = Path(env["RUNSM"])
    run = lambda a, extra=None: L.cmd(a, env={**env, **(extra or {})}, dryrun=False, timeout=300)
    # --- banks
    for s in (0, 1):
        rc, o = run(f"bankdag {s}")
        bd = R / "dagger_bank" / f"s{s}"
        check(f"bankdag {s}: rc 0, four shards launched with --shard 0..3 --n-shards 4, merge ran, bank.json written", rc == 0 and (bd / "bank.json").exists() and all((bd / f"shard_00{k}of004.npz").exists() for k in range(4)), f"{rc} {o}")
        gens = [jload(bd / f"gen_argv_{k}.json") for k in range(4)]
        check(f"bankdag {s}: every generator got the same arguments except --shard (and the same --out-dir), --ckpt = M2b_s{s}'s checkpoint, --device cuda, --threads 2, --batch 32",
              all(L.drop_flag(g, {"--shard"}) == L.drop_flag(gens[0], {"--shard"}) for g in gens) and [L.flags(g)["--shard"][0][0] for g in gens] == ["0", "1", "2", "3"] and L.flags(gens[0])["--ckpt"][0][0].endswith(f"M2b_s{s}\\ckpt_last.pt")
              and L.flags(gens[0])["--device"] == [["cuda"]] and L.flags(gens[0])["--threads"] == [["2"]] and L.flags(gens[0])["--n-shards"] == [["4"]], gens[0])
        check(f"bankdag {s}: eval grid was COPIED to a path without an ampersand and that copy is what the generators got", L.flags(gens[0])["--eval-grid"][0][0] == str(R / "dagpin" / "eval_grid_windows.json") and (R / "dagpin" / "eval_grid_windows.json").exists())
    rc, o = run("bankdag 0")
    check("bankdag 0 again: skipped (bank.json and PROVENANCE.json exist)", rc == 0 and any("[skip]" in x for x in o), o)
    S = {s: hashlib.sha256((R / "dagger_bank" / f"s{s}" / "bank.json").read_bytes()).hexdigest() for s in (0, 1)}
    # --- DAG s0: gate, pin, train
    rc, o = run("gatedag dag 0 0.1", {"DAGSHA_DRY": "f" * 64})
    pin = R / "dagpin" / "M2b_s0_dag.train.argv.txt"
    check("gatedag dag 0 0.1: rc 0 and the stub trainer saw the gate argv", rc == 0 and (R / "gate_M2b-dag_s0" / "stub_argv.json").exists(), o)
    pa = L.train_argv(pin.read_text())
    check("the pin carries the sha256 COMPUTED FROM bank.json on this box, and ignores an env DAGSHA_DRY (the dry-run override is never read by a real run)", L.flags(pa)["--dagger-bank-sha256"] == [[S[0]]] and "f" * 64 not in pin.read_text(), L.flags(pa).get("--dagger-bank-sha256"))
    ga = jload(R / "gate_M2b-dag_s0" / "stub_argv.json")
    check("the argv the gate ACTUALLY ran == the pinned training argv with --out and --selfcheck-only changed", L.drop_flag(ga, {"--out"}, {"--selfcheck-only"}) == L.drop_flag(pa, {"--out"}) and ga.count("--selfcheck-only") == 1)
    rc, o = run("traindag dag 0 0.1", {"DAGSHA_DRY": "e" * 64})
    ra = jload(R / "M2b_s0_dag" / "stub_argv.json")
    check("traindag dag 0 0.1: rc 0, summary.json exists", rc == 0 and (R / "M2b_s0_dag" / "summary.json").exists(), o)
    check("BYTE-IDENTITY: the argv the trainer RECEIVED in training == the pinned line's tokens == the gate's (--out / --selfcheck-only aside); the token's argv_sha256 is therefore the sha256 of what ran",
          ra == pa and L.argv_sha256(ra) == L.argv_sha256(pa) and L.drop_flag(ga, {"--out"}, {"--selfcheck-only"}) == L.drop_flag(ra, {"--out"}), (len(ra), len(pa)))
    # the chain's token binds argv_sha256 of the PINNED line = of the PLANNED line (dry-run with the pulled sha)
    planned = L.train_argv(L.trainer_line(L.dry("traindag dag 0 0.1", sha=S[0])))
    planned = [x.replace("D:\\tanitad\\runs\\v8f_mini_full", env["RUNSM"]) if isinstance(x, str) else x for x in planned]
    check("the line the chain PLANS (dry run, DAGSHA_DRY = sha of the pulled bank.json) has the same flags as the pin box 2 wrote, up to box 2's paths (the chain compares them verbatim on the real box)", L.drop_flag(planned, {"--cache-dir", "--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--out", "--init-ckpt", "--dagger-bank"}) ==
          L.drop_flag(pa, {"--cache-dir", "--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--out", "--init-ckpt", "--dagger-bank"}))
    rc, o = run("traindag dag 0 0.1")
    check("traindag again: skipped once summary.json exists", rc == 0 and any("[skip]" in x for x in o), o)
    # --- DAG-0 s0 and the PAIR rule on what actually RAN
    rc1, _ = run("gatedag dag0 0")
    rc2, _ = run("traindag dag0 0")
    r0 = jload(R / "M2b_s0_dag0" / "stub_argv.json")
    check("DAG-0 s0 gate and train ran; DAG and DAG-0 ARGV AS RECEIVED differ only in --out, --dagger-p and the three DAG-only flags", rc1 == 0 and rc2 == 0 and L.drop_flag(ra, {"--out", "--dagger-p"} | L.DAG_ONLY) == L.drop_flag(r0, {"--out", "--dagger-p"}) and L.flags(r0)["--dagger-p"] == [["0"]], (rc1, rc2))
    # --- evals
    rc, _ = run("evaldag dag 0")
    ev = R / "eval"
    check("evaldag dag 0: ran, wrote eval_M2b-dag_s0.npz, and NO `-fresh` eval file exists (O1: the fresh-age comparator is the dev box's)", rc == 0 and (ev / "eval_M2b-dag_s0.npz").exists() and not list(ev.glob("*fresh*")))
    sa = jload(ev / "eval_M2b-dag_s0.stub_argv.json")
    check("the standard eval ran WITHOUT --ages", "--ages" not in sa)
    check("PREREG section 4 on what actually RAN: the gate, the DAG training and the DAG-0 training each RECEIVED `--ego-perturb-p 0` exactly once", names_ego0(ga) and names_ego0(ra) and names_ego0(r0))
    rc, o = run("evaldag dag 0")
    check("evaldag again: skipped once its npz exists", rc == 0 and any("[skip]" in x for x in o))
    # --- the refusals (deliberate regressions)
    rc, o = run("gatedag dag 1 0.1")
    pin1 = R / "dagpin" / "M2b_s1_dag.train.argv.txt"
    good = pin1.read_bytes()
    check("s1 DAG gate ran and pinned its argv", rc == 0 and pin1.exists())
    pin1.write_bytes(good.replace(b"--steps 3000", b"--steps 3001"))
    rc, o = run("traindag dag 1 0.1")
    regress("ONE CHANGED BYTE in the pinned argv (--steps 3000 -> 3001): traindag REFUSES, rc 2, and the trainer is not executed", rc == 2 and not (R / "M2b_s1_dag" / "stub_argv.json").exists(), f"{rc} {o}")
    pin1.write_bytes(good.replace(b"--lr 1e-4", b"--lr 1e-5"))
    rc, o = run("traindag dag 1 0.1")
    regress("a changed lever in the pin (--lr 1e-4 -> 1e-5): traindag REFUSES (the line it would build now differs from the pin)", rc == 2 and not (R / "M2b_s1_dag" / "stub_argv.json").exists())
    pin1.write_bytes(good)
    bank1 = R / "dagger_bank" / "s1" / "bank.json"
    orig = bank1.read_bytes()
    bank1.write_bytes(orig + b" ")
    rc, o = run("traindag dag 1 0.1")
    regress("bank.json changed (one byte appended) AFTER the gate: traindag REFUSES, the pin is for another bank sha256", rc == 2 and not (R / "M2b_s1_dag" / "stub_argv.json").exists(), f"{rc} {o}")
    rc, o = run("gatedag dag 1 0.1")
    regress("RE-GATING after the bank changed: the stage REFUSES to replace a pin that differs (never a silent re-pin)", rc == 2 and pin1.read_bytes() == good, f"{rc} {o}")
    bank1.write_bytes(orig)
    rc, o = run("traindag dag 1 0.1")
    check("positive control: with the pin and bank.json restored byte-for-byte, traindag runs and the received argv == the pin", rc == 0 and jload(R / "M2b_s1_dag" / "stub_argv.json") == L.train_argv(good.decode()), o)
    rc, o = run("traindag dag0 1")
    regress("traindag dag0 1 WITHOUT its gate: REFUSED (no pinned argv), rc 2, nothing executed", rc == 2 and not (R / "M2b_s1_dag0" / "stub_argv.json").exists(), f"{rc} {o}")
    rc, o = run("evaldag dag0 1")
    regress("evaldag dag0 1 before training: REFUSED, rc 2", rc == 2 and not (ev / "eval_M2b-dag0_s1.npz").exists())
    rc, o = run("gatedag dag0 1")
    rc2, o2 = run("traindag dag0 1", {"STUB_FAIL": "1"})
    regress("a trainer that exits nonzero: traindag exits nonzero (the chain stops on it) and no summary.json is claimed", rc == 0 and rc2 != 0 and not (R / "M2b_s1_dag0" / "summary.json").exists(), f"{rc2} {o2}")
    (R / "M2b_s1_dag0" / "train.log").unlink(missing_ok=True)
    shutil.rmtree(root, ignore_errors=True)


# ================================================================================================ C. chain decisions
def orc_file(path, outcome, p_controls=None, claim="H-V8F-ORECOVER-1", rows=None):
    d = {"OUTCOME": {"OUTCOME": outcome, "matching_rows_in_table_order": rows if rows is not None else [outcome]}, "claim_row": claim, "controls_PASS": p_controls if p_controls is not None else {"A": True, "B": True}}
    jdump(path, d)
    return str(path)


KB4_CASES = ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")      # dag_read.py KB4_CASES


def kb_good() -> dict:
    """PROVENANCE.json `controls` as the reader's g_kb4 / g_kb5 accept them (the generator is being changed to write exactly this)."""
    return {"K-B4": {"PASS": True, "cases": {k: True for k in KB4_CASES}}, "K-B5": {"PASS": True, "max_pose_now_err_m": 3e-6, "max_rel_transform_err_m": 2e-6}}


def kb_mut(fn):
    """a `mutate` hook that edits PROVENANCE.json's controls block"""
    return lambda man, prov, provs: fn(prov["controls"], prov)


def mk_bank(d: Path, seed: int, ckpt: str, frac=0.6, n_shards=4, windows=306566, mutate=None):
    """A bank directory in the schema v8f_dagger_bank.py writes (bank.json / PROVENANCE.json / shard provenance), built from the REAL schema of rg/out/bank.json."""
    tmpl = jload(HERE / "rg" / "out" / "bank.json")
    d.mkdir(parents=True, exist_ok=True)
    elig = int(round(frac * windows)) if isinstance(frac, float) else frac
    man = copy.deepcopy(tmpl)
    man["n_shards"] = n_shards
    man["common"]["ckpt_sha256"] = ckpt
    man["common"]["train"]["n_clips"], man["common"]["train"]["n_windows"], man["common"]["train"]["exclude"] = 4512, windows, "540:600"
    man["common"]["eval"]["overlap"], man["common"]["eval"]["n_clips"] = 0, 141
    man["common"]["partial"], man["common"]["limit_clips"] = False, 0
    man["partial"] = False
    man["coverage"] = {"n_windows": windows, "n_eligible": elig, "fraction": elig / windows, "min": 0.5, "blocked": (elig / windows) < 0.5}
    man["shards"] = [{"file": f"shard_{k:03d}of{n_shards:03d}.npz", "k": k, "n_records": 10, "file_sha256": "0" * 64, "content_sha256": "1" * 64} for k in range(n_shards)]
    per = windows // n_shards
    provs = []
    for k in range(n_shards):
        nw = per + (windows - per * n_shards if k == n_shards - 1 else 0)
        provs.append({"kind": "shard", "coverage": {"n_windows": nw, "n_eligible": 1, "fraction": 0.5, "min": 0.5, "blocked": False}, "kb0_tick0_identity": {"checked": nw, "mismatch": 0},
                      "kb1_replay": {"chunks": 3, "windows": 96, "mismatch_chunks": 0, "frac": 0.005, "seed": 0}, "kb3_eval_disjoint": {"n_eval_clips": 141, "overlap": 0}, "warnings": []})
    prov = {"kind": "bank", "coverage": copy.deepcopy(man["coverage"]), "warnings": [], "controls": kb_good()}
    if mutate:
        mutate(man, prov, provs)
    jdump(d / "bank.json", man)
    jdump(d / "PROVENANCE.json", prov)
    for k, p in enumerate(provs):
        jdump(d / f"shard_{k:03d}of{n_shards:03d}.provenance.json", p)
    return d


def section_c():
    say("== C. the chain's decisions on files")
    t = BT / "chainc"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    # ---- the ORC predicate
    real = Path("E:/Projects/TanitAD-artifacts/v8f-orecover-t1/read/ORC_READ.json")
    rc, msg = CK.orc_predicate(real)
    check("ORC predicate on the REAL read (HARMS): run, p = 0.1, sha256 a50dbba9... (the value PREREG_V8F_DAGGER Amendment 1 records)", rc == 0 and msg.startswith("HARMS 0.1 a50dbba9"), msg)
    for out, want_rc, want_p in (("NO EFFECT", 0, "0.3"), ("HARMS", 0, "0.1")):
        rc, msg = CK.orc_predicate(orc_file(t / f"o_{out}.json", out))
        check(f"ORC predicate: {out} -> run at p = {want_p}", rc == want_rc and msg.split()[-2] == want_p, msg)
    for out in ("ROBUST", "PARTIAL"):
        rc, msg = CK.orc_predicate(orc_file(t / f"o_{out}.json", out))
        regress(f"ORC predicate REFUSES {out} (rc 4 = PARKED: the arm is parked, not run)", rc == 4 and "PARKED" in msg, msg)
    for out in ("REFUSED", "UNPLACED", "SOMETHING ELSE"):
        rc, msg = CK.orc_predicate(orc_file(t / f"o_{out}.json", out))
        regress(f"ORC predicate REFUSES {out} (rc 2)", rc == 2, msg)
    rc, msg = CK.orc_predicate(orc_file(t / "o_badctl.json", "HARMS", p_controls={"A": True, "B": False}))
    regress("ORC predicate REFUSES a HARMS read with one FAILED control (a read with a failed control is not an outcome)", rc == 2, msg)
    rc, msg = CK.orc_predicate(orc_file(t / "o_claim.json", "HARMS", claim="H-V8F-PGT-1"))
    regress("ORC predicate REFUSES another PREREG's read file", rc == 2, msg)
    rc, msg = CK.orc_predicate(orc_file(t / "o_two.json", "HARMS", rows=["HARMS", "NO EFFECT"]))
    regress("ORC predicate REFUSES a read matching two table rows", rc == 2, msg)
    rc, msg = CK.orc_predicate(str(t / "does_not_exist.json"))
    regress("ORC predicate REFUSES a missing file (a missing file is not a HARMS)", rc == 2, msg)
    # ---- the bank judge
    s0, s1 = CK_SHAS = (L_ck(0), L_ck(1))

    def judge(f0=None, f1=None, m0=None, m1=None, name="x"):
        d0 = mk_bank(t / f"{name}_s0", 0, s0, frac=f0 if f0 is not None else 0.62, mutate=m0)
        d1 = mk_bank(t / f"{name}_s1", 1, s1, frac=f1 if f1 is not None else 0.58, mutate=m1)
        import argparse
        a = argparse.Namespace(banked_sha0=s0, banked_sha1=s1, seed0_dir=str(d0), seed1_dir=str(d1), expect_clips=4512, expect_windows=306566)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            return CK.bank_judge(a)
    rc, res = judge(name="good")
    check("bank judge: all controls pass, coverage 0.62 / 0.58 -> PASS (rc 0)", rc == 0 and res["outcome"] == "PASS", res["refused_because"])
    rc, res = judge(f0=0.5, name="half")
    check("bank judge: coverage EXACTLY 0.5 is a PASS (the PREREG says >= 0.5, inclusive)", rc == 0, res)
    rc, res = judge(f0=153282 / 306566, name="belowhalf")
    regress("bank judge: coverage 153,282 / 306,566 (just under 0.5) -> BLOCKED (rc 3)", rc == 3 and res["blocked_seeds"] == [0], res["outcome"])
    rc, res = judge(f1=0.49, name="c049")
    regress("bank judge: coverage 0.49 at seed 1 only -> BLOCKED for the whole chain (rc 3): the decisive read needs both seeds", rc == 3 and res["blocked_seeds"] == [1], res["outcome"])
    rc, res = judge(f0=0.3, f1=0.2, name="both")
    regress("bank judge: both seeds below 0.5 -> BLOCKED", rc == 3 and res["blocked_seeds"] == [0, 1])
    flag = lambda man, prov, provs: man["coverage"].__setitem__("blocked", False)
    rc, res = judge(f0=0.3, m0=flag, name="flag")
    regress("bank judge: coverage 0.3 but the bank's own `blocked` flag says False -> REFUSED (rc 2): the fraction is recomputed, the flag must agree", rc == 2 and any("flag agrees" in b for b in res["refused_because"]), res["refused_because"][:2])
    kb1 = lambda man, prov, provs: provs[2]["kb1_replay"].__setitem__("mismatch_chunks", 1)
    rc, res = judge(f0=0.3, m0=kb1, name="kb1blocked")
    regress("bank judge: a K-B1 replay mismatch AND coverage < 0.5 -> REFUSED, not BLOCKED (a failed control takes precedence: PREREG section 7)", rc == 2 and res["blocked_seeds"] == [0] and any("K-B1" in b for b in res["refused_because"]), res["outcome"])
    SUBS = {"K-B0 mismatch": "K-B0", "K-B0 not every window checked": "K-B0", "K-B1 not run (0 chunks)": "K-B1", "K-B2: generated by another checkpoint": "K-B2", "K-B3: an eval clip in the bank": "K-B3",
            "K-B3: a shard saw an eval overlap": "K-B3", "PARTIAL bank": "COMPLETE", "TRAIN list is not 4,512 clips": "TRAIN list", "TRAIN window count differs": "TRAIN list", "rollout is not PREREG section 1's": "ROLLOUT",
            "wrong schema": "SCHEMA", "a shard provenance missing a record": "BANK-FILES", "coverage denominator is not the TRAIN window count": "COVERAGE record"}
    for nm, mut in (("K-B0 mismatch", lambda man, prov, provs: provs[0]["kb0_tick0_identity"].__setitem__("mismatch", 2)),
                    ("K-B0 not every window checked", lambda man, prov, provs: provs[1]["kb0_tick0_identity"].__setitem__("checked", 5)),
                    ("K-B1 not run (0 chunks)", lambda man, prov, provs: provs[3]["kb1_replay"].__setitem__("chunks", 0)),
                    ("K-B2: generated by another checkpoint", lambda man, prov, provs: man["common"].__setitem__("ckpt_sha256", "0" * 64)),
                    ("K-B3: an eval clip in the bank", lambda man, prov, provs: man["common"]["eval"].__setitem__("overlap", 1)),
                    ("K-B3: a shard saw an eval overlap", lambda man, prov, provs: provs[2]["kb3_eval_disjoint"].__setitem__("overlap", 1)),
                    ("PARTIAL bank", lambda man, prov, provs: man.__setitem__("partial", True)),
                    ("TRAIN list is not 4,512 clips", lambda man, prov, provs: man["common"]["train"].__setitem__("n_clips", 4511)),
                    ("TRAIN window count differs", lambda man, prov, provs: man["common"]["train"].__setitem__("n_windows", 306565)),
                    ("rollout is not PREREG section 1's", lambda man, prov, provs: man["common"]["rollout"].__setitem__("limit", "logged")),
                    ("wrong schema", lambda man, prov, provs: man.__setitem__("schema", "v8f-dagger-bank-v0")),
                    ("a shard provenance missing a record", lambda man, prov, provs: man.__setitem__("n_shards", 5)),
                    ("coverage denominator is not the TRAIN window count", lambda man, prov, provs: man["coverage"].__setitem__("n_windows", 1000))):
        rc, res = judge(m0=mut, name="m_" + re.sub(r"\W", "", nm))
        regress(f"bank judge: {nm} -> REFUSED (rc 2) and the failed control is NAMED ({SUBS[nm]})", rc == 2 and any(SUBS[nm] in b for b in res["refused_because"]), res["refused_because"][:2])
    # ---- K-B4 / K-B5 are GATED (Master Mind O6), by the reader's rules (dag_read.py g_kb4 / g_kb5 / kb_record); a MISSING record stops the chain
    def kbrefuse(nm, fn, want, seed1=False, f0=None):
        kw = {"m1": kb_mut(fn)} if seed1 else {"m0": kb_mut(fn)}
        rc, res = judge(name="kbx", f0=f0, **kw)
        regress(f"bank judge: {nm} -> REFUSED (rc 2) and the failed control is NAMED ({want})", rc == 2 and res["outcome"] == "REFUSED" and any(want in b for b in res["refused_because"]) and all(("K-B4" in b or "K-B5" in b) for b in res["refused_because"]), res["refused_because"][:2])
    rc, res = judge(name="kbx")
    check("bank judge: K-B4 (PASS True, six cases True) and K-B5 (PASS True, both errors 3e-6 / 2e-6 <= 1e-4) recorded at both seeds -> PASS, and both controls are in the verdict", rc == 0 and res["outcome"] == "PASS" and res["seeds"][0]["controls"].get("K-B4 matcher synthetic cases (PROVENANCE.json controls.K-B4: PASS True and all six cases True)") is True and res["seeds"][1]["controls"].get("K-B5 re-anchoring (PROVENANCE.json controls.K-B5: PASS True, both errors <= 1e-4 m)") is True, res["refused_because"])
    kbrefuse("a MISSING K-B4 record", lambda c, p: c.pop("K-B4"), "K-B4")
    kbrefuse("a MISSING K-B5 record", lambda c, p: c.pop("K-B5"), "K-B5")
    kbrefuse("no `controls` block in PROVENANCE.json at all (the generator that does not write K-B4 / K-B5)", lambda c, p: p.pop("controls"), "K-B4")
    kbrefuse("a MISSING K-B4 record at seed 1 only", lambda c, p: c.pop("K-B4"), "K-B4", seed1=True)
    kbrefuse("K-B4 PASS false", lambda c, p: c["K-B4"].__setitem__("PASS", False), "K-B4")
    kbrefuse("K-B5 PASS false", lambda c, p: c["K-B5"].__setitem__("PASS", False), "K-B5")
    kbrefuse("K-B4 PASS absent (a lower-case `pass` key is not the reader's key)", lambda c, p: (c["K-B4"].pop("PASS"), c["K-B4"].__setitem__("pass", True)), "K-B4")
    kbrefuse("K-B5 PASS = 1 (truthy, not the boolean True the reader requires)", lambda c, p: c["K-B5"].__setitem__("PASS", 1), "K-B5")
    kbrefuse("K-B4 record is not a dict", lambda c, p: c.__setitem__("K-B4", True), "K-B4")
    for case in KB4_CASES:
        kbrefuse(f"K-B4 case `{case}` false (PASS still True)", lambda c, p, case=case: c["K-B4"]["cases"].__setitem__(case, False), case)
    kbrefuse("K-B4 case `loop` absent", lambda c, p: c["K-B4"]["cases"].pop("loop"), "loop")
    kbrefuse("K-B4 `cases` is not a dict", lambda c, p: c["K-B4"].__setitem__("cases", [True] * 6), "K-B4")
    kbrefuse("K-B4 case recorded 1 instead of True", lambda c, p: c["K-B4"]["cases"].__setitem__("straight", 1), "straight")
    kbrefuse("K-B5 max_pose_now_err_m = 2e-4 (> 1e-4)", lambda c, p: c["K-B5"].__setitem__("max_pose_now_err_m", 2e-4), "max_pose_now_err_m")
    kbrefuse("K-B5 max_rel_transform_err_m = 2e-4 (> 1e-4)", lambda c, p: c["K-B5"].__setitem__("max_rel_transform_err_m", 2e-4), "max_rel_transform_err_m")
    kbrefuse("K-B5 max_pose_now_err_m missing", lambda c, p: c["K-B5"].pop("max_pose_now_err_m"), "max_pose_now_err_m")
    kbrefuse("K-B5 max_rel_transform_err_m is NaN", lambda c, p: c["K-B5"].__setitem__("max_rel_transform_err_m", float("nan")), "max_rel_transform_err_m")
    kbrefuse("K-B5 error recorded as a string", lambda c, p: c["K-B5"].__setitem__("max_pose_now_err_m", "0.0"), "max_pose_now_err_m")
    kbrefuse("K-B5 error recorded as the boolean True", lambda c, p: c["K-B5"].__setitem__("max_rel_transform_err_m", True), "max_rel_transform_err_m")
    kbrefuse("K-B5 error 2e-4 at seed 1 only", lambda c, p: c["K-B5"].__setitem__("max_pose_now_err_m", 2e-4), "max_pose_now_err_m", seed1=True)
    kbrefuse("a MISSING K-B4 record AND coverage 0.3 -> REFUSED, not BLOCKED (a failed control takes precedence)", lambda c, p: c.pop("K-B4"), "K-B4", f0=0.3)
    rc, res = judge(name="kbx", m0=kb_mut(lambda c, p: c["K-B5"].update({"max_pose_now_err_m": 1e-4, "max_rel_transform_err_m": 1e-4})))
    check("bank judge: K-B5 errors EXACTLY 1e-4 pass (the reader's rule is `<=`, inclusive)", rc == 0, res["refused_because"])
    rc, res = judge(name="kbx", m0=kb_mut(lambda c, p: c["K-B5"].update({"max_pose_now_err_m": 0, "max_rel_transform_err_m": 0.0})))
    check("bank judge: K-B5 errors recorded 0 (int) and 0.0 pass", rc == 0, res["refused_because"])
    rc, res = judge(name="kbx", m0=kb_mut(lambda c, p: c.__setitem__("K-B9", {"PASS": False})))
    check("bank judge: a control key BEYOND K-B4 / K-B5 in PROVENANCE.json is REPORTED, never required, never fatal", rc == 0 and res["seeds"][0]["extra_controls_in_provenance"].get("K-B9") == {"PASS": False}, res["refused_because"])
    shutil.rmtree(t, ignore_errors=True)


def L_ck(seed) -> str:
    txt = (HERE / "box2_ship" / "chain_v8f_dag.sh").read_text(encoding="utf-8")
    m = re.search(r"^CKPT_SHA=\((\w{64}) (\w{64})\)", txt, flags=re.M)
    return m.group(1 + seed)


# ================================================================================================ D. the chain script
def bash(args, env=None, timeout=300):
    return sh([BASH, posix(CHAIN), *args], env=env, timeout=timeout)


def section_d():
    say("== D. chain_v8f_dag.sh")
    t = BT / "chaind"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    harms = posix(orc_file(t / "harms.json", "HARMS"))
    robust = posix(orc_file(t / "robust.json", "ROBUST"))
    s0, s1 = L_ck(0), L_ck(1)
    # ---- --judge-only: both sides, offline
    good0, good1 = mk_bank(t / "g0", 0, s0), mk_bank(t / "g1", 1, s1)
    rc, o = bash(["--judge-only", posix(good0), posix(good1)], {"DAG_ORC_READ": harms})
    check("chain --judge-only (HARMS, both banks fine): rc 0, prints ZZDAG-JUDGE-PASS (the gates would be queued next) and no stop marker", rc == 0 and "ZZDAG-JUDGE-PASS" in o and "ZZV8FMINI-STOP" not in o, o[-500:])
    blk1 = mk_bank(t / "b1", 1, s1, frac=0.49)
    rc, o = bash(["--judge-only", posix(good0), posix(blk1)], {"DAG_ORC_READ": harms})
    regress("chain --judge-only with coverage 0.49 at s1: rc 3, prints ZZDAG-BLOCKED and `ZZV8FMINI-STOP BLOCKED`, says no gate was queued, and never prints the all-gates line", rc == 3 and "ZZDAG-BLOCKED" in o and "ZZV8FMINI-STOP BLOCKED" in o and "no gate was queued" in o and "ZZDAG-JUDGE-PASS" not in o, o[-600:])
    bad0 = mk_bank(t / "r0", 0, s0, mutate=lambda man, prov, provs: provs[1]["kb1_replay"].__setitem__("mismatch_chunks", 1))
    rc, o = bash(["--judge-only", posix(bad0), posix(good1)], {"DAG_ORC_READ": harms})
    regress("chain --judge-only with a failed K-B1: rc 2, ZZDAG-REFUSED + ZZV8FMINI-STOP REFUSED, no gate queued", rc == 2 and "ZZDAG-REFUSED" in o and "ZZV8FMINI-STOP REFUSED" in o and "ZZDAG-JUDGE-PASS" not in o, o[-600:])
    nokb = mk_bank(t / "nokb4", 1, s1, mutate=lambda man, prov, provs: prov["controls"].pop("K-B4"))
    rc, o = bash(["--judge-only", posix(good0), posix(nokb)], {"DAG_ORC_READ": harms})
    regress("chain --judge-only with a MISSING K-B4 record at seed 1: rc 2, ZZDAG-REFUSED names K-B4, ZZV8FMINI-STOP REFUSED, no gate queued, never ZZDAG-JUDGE-PASS", rc == 2 and "ZZDAG-REFUSED" in o and "K-B4" in o and "ZZV8FMINI-STOP REFUSED" in o and "ZZDAG-JUDGE-PASS" not in o, o[-500:])
    rc, o = bash(["--judge-only", posix(good0), posix(good1)], {"DAG_ORC_READ": robust})
    regress("chain --judge-only with ORC = ROBUST: the arm is PARKED before the banks are even looked at (rc 1, ZZV8FMINI-STOP, no ZZDAG-JUDGE-PASS)", rc == 1 and "ZZV8FMINI-STOP" in o and "PARKED" in o and "ZZDAG-JUDGE-PASS" not in o, o[-500:])
    # ---- the real flow parks / refuses BEFORE it touches anything (no AFTER_OUT needed, no ssh): run the real entry with a ROBUST read
    rc, o = bash([], {"DAG_ORC_READ": robust})
    regress("the REAL entry with ORC = ROBUST stops at once with `ZZV8FMINI-STOP PARKED` (before AFTER_OUT, the busy marker or any box-2 call)", rc == 1 and "ZZV8FMINI-STOP PARKED" in o and "waiting for the predecessor" not in o, o[-400:])
    rc, o = bash([], {"DAG_ORC_READ": harms})
    check("the REAL entry with ORC = HARMS but no AFTER_OUT: refuses (`ZZV8FMINI-STOP AFTER_OUT is not set`), touches nothing", rc == 1 and "AFTER_OUT is not set" in o and "ORC read HARMS" in o and "p = 0.1" in o, o[-400:])
    # ---- the source order: bank -> judge -> stop-if-nonzero -> gates -> trains
    src = CHAIN.read_text(encoding="utf-8")

    def order_ok(s: str) -> bool:
        i_bank = s.find("for s in 0 1; do bank_one $s; done")
        i_judge = s.rfind('judge_banks "$ART/bank_s0" "$ART/bank_s1"; jr=$?')       # the LAST one: the --publish-only block repeats the line earlier in the file
        i_stop = s.find("if [ $jr != 0 ]; then", i_judge)
        i_stop_call = s.find('stop "$([ $jr = 3 ]', i_stop)
        i_pub = s.rfind("for s in 0 1; do publish_bank_one $s; done")
        i_orc2 = s.find('o2=$(sha256sum "$ORC_READ"', i_pub)
        i_gate = s.find('for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done')
        i_train = s.find('for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done')
        return 0 < i_bank < i_judge < i_stop < i_stop_call < i_pub < i_orc2 < i_gate < i_train and s.find("fi", i_stop_call) < i_pub
    check("chain source order: banks -> judge -> `if [ $jr != 0 ]` stop -> PUBLISH the banks (outside that if) -> ORC re-read -> ALL gates -> trains (a BLOCKED / REFUSED judge reaches stop before any publish, gate_one or runone)", order_ok(src))
    mut = src.replace('stop "$([ $jr = 3 ] && echo BLOCKED || echo REFUSED) at the bank stage (judge rc $jr): no gate was queued, nothing was trained (see $ART/DAG_OUTCOME.json)"', "true")
    regress("chain source with the BLOCKED stop deleted: the order check goes RED", mut != src and not order_ok(mut))
    mut2 = src.replace("if [ $jr != 0 ]; then", "if false; then")
    regress("chain source with the judge's `if [ $jr != 0 ]` disabled: the order check goes RED", mut2 != src and not order_ok(mut2))
    pub_call = "for s in 0 1; do publish_bank_one $s; done"
    i_real_pub = src.rfind(pub_call)
    mut3 = src[:i_real_pub] + "true" + src[i_real_pub + len(pub_call):]
    regress("chain source with the real flow's publish call deleted (the banks would never reach the dev box): the order check goes RED", mut3 != src and not order_ok(mut3))
    i_real_judge = src.rfind('judge_banks "$ART/bank_s0" "$ART/bank_s1"; jr=$?')
    mut4 = mut3[:i_real_judge] + pub_call + "\n" + mut3[i_real_judge:]
    regress("chain source with the publish call moved BEFORE the judge (a REFUSED / BLOCKED bank would be moved): the order check goes RED", mut4 != src and not order_ok(mut4))
    def fbody(txt: str, name: str) -> str:
        m_ = re.search(r"^" + re.escape(name) + r"\(\) \{.*?^\}", txt, flags=re.M | re.S)
        return m_.group(0) if m_ else ""
    bo, fs, pb = fbody(src, "bank_one"), fbody(src, "bank_fetch_small"), fbody(src, "publish_bank_one")
    check("chain: bank_one / bank_fetch_small pull only the SMALL files the judge reads (no shard npz), assert box 2's bank.json sha256 == the pulled one, and remove an earlier attempt's SHA256SUMS.txt first",
          bool(bo) and bool(fs) and ".npz" not in bo and ".npz" not in fs and "shard_00${k}of004.provenance.json" in fs and 'rsha "$R/dagger_bank/s$s/bank.json"' in fs and 'rm -f "$d/SHA256SUMS.txt"' in fs and "bank_fetch_small $1" in bo, (bo[:60], fs[:60]))
    check("chain: publish_bank_one names the files with `bank-files`, pulls only `--kind shard_npz`, compares box 2's sha256 with the pulled file's, verifies BEFORE sealing, writes SHA256SUMS.txt with `sha256sum | tee`, re-verifies with --require-sums and removes a failed seal",
          bool(pb) and "--kind shard_npz" in pb and 'rsha "$R/dagger_bank/s$s/$f" SHA256' in pb and pb.find("bank-verify") < pb.find("sha256sum $all | tee SHA256SUMS.txt") < pb.rfind("--require-sums") and 'rm -f "$d/SHA256SUMS.txt"' in pb and "--kind all" in pb, pb[:80])
    check("chain: the token is written only after `check` passes; the pin is pulled from box 2 and given to the assertions (--pin-line)", src.index('"$(cygpath -m $ASSERT)" check') < src.index('"$(cygpath -m $ASSERT)" token') and "--pin-line" in src and "dagpin/M2b_s${s}_${arm}.train.argv.txt" in src)
    check("chain: a busy marker is only removed by a chain that took it (a refused / parked chain never deletes another chain's marker)", "BUSY_TAKEN=0" in src and 'if [ $BUSY_TAKEN = 1 ]; then echo "" > $CUR; rm -f $B/BOX2_BUSY.txt; fi' in src)
    check("chain: ORC_READ.json is re-read byte-identically (sha256) before the gates", 'o2=$(sha256sum "$ORC_READ"' in src and src.index('o2=$(sha256sum "$ORC_READ"') < src.index('gate_one $1 $2; done'))
    # ---- --dry-run
    rc, o = bash(["--dry-run"], {"V8F_QUEUE": posix(L.QUEUE), "DAG_ORC_READ": harms, "DAG_DRY_SHA0": "a1" * 32, "DAG_DRY_SHA1": "b2" * 32})
    (HERE / "chain_dry_run.txt").write_text(o, encoding="utf-8")
    check("chain --dry-run: rc 0; all four runs planned in the PREREG's order (DAG s0, DAG-0 s0, DAG s1, DAG-0 s1); each gate line is the train line with --out / --selfcheck-only changed (SAME); p = 0.1",
          rc == 0 and [m.group(0) for m in re.finditer(r"== M2b_s\d_dag0?", o)] == ["== M2b_s0_dag", "== M2b_s0_dag0", "== M2b_s1_dag", "== M2b_s1_dag0"] and o.count(": SAME") == 4 and "DIFFERENT" not in o and "p = 0.1" in o, o[-400:])
    shas = re.findall(r"token argv_sha256 \(the training argv\): (\w{64})", o)
    check("chain --dry-run: four distinct token argv_sha256 values, and the DAG ones carry the dry-run bank sha256 in the line they bind", len(shas) == 4 and len(set(shas)) == 4 and ("a1" * 32) in o and ("b2" * 32) in o, shas)
    check("chain --dry-run: for BOTH seeds the publish plan names the FOUR kinds explicitly (PROVENANCE.json, bank.json, 4 shard provenance, 4 shard npz) + the stage log, the exact 11-name SHA256SUMS.txt and the ESTIMATED size",
          o.count("kind provenance: PROVENANCE.json") == 2 and o.count("kind manifest: bank.json") == 2 and o.count("kind shard_provenance: shard_000of004.provenance.json, shard_001of004.provenance.json, shard_002of004.provenance.json, shard_003of004.provenance.json") == 2
          and o.count("kind shard_npz: shard_000of004.npz, shard_001of004.npz, shard_002of004.npz, shard_003of004.npz") == 2 and o.count("exactly 11 names") == 2 and o.count("ESTIMATED size per seed") == 2, o[-900:])
    # ---- --check-tree
    rg = HERE / "rg" / "tree"
    rc, o = bash(["--check-tree", posix(rg)])
    check("chain --check-tree on the integrated tree (rg/tree, the di/tree files + the shard tool): every lever pattern is present", rc == 0 and "every one of" in o, o[-600:])
    pre = BT / "tip_tree"
    if pre.exists():
        shutil.rmtree(pre)
    for f in ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_dagger.py", "stack/scripts/v8f_dagger_bank.py", "stack/scripts/v8f_dagger_queue_tool.py"):
        (pre / f).parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", "C:/Users/Admin/tanitad-push", "show", f"2a78f059:{f}"], capture_output=True)
        if r.returncode == 0:
            (pre / f).write_bytes(r.stdout)
    rc, o = bash(["--check-tree", posix(pre)])
    regress("chain --check-tree on the PRE-CHANGE tip (2a78f059: no DAgger trainer, no bank): FAILS and names the missing apply line", rc == 1 and "MISSING" in o and "dag\\.apply" in o and "lever check: FAILED" in o, o[-500:])
    mt = BT / "mut_tree"
    if mt.exists():
        shutil.rmtree(mt)
    for f in ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_dagger.py", "stack/scripts/v8f_dagger_bank.py", "stack/scripts/v8f_dagger_queue_tool.py"):
        (mt / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rg / f, mt / f)
    tr = mt / "stack/scripts/v8f_mini_train.py"
    txt = tr.read_text(encoding="utf-8")
    ln = [x for x in txt.split("\n") if re.match(r"^ +b = dag\.apply\(b, step\)", x)]
    check("(the loop line exists exactly once, indented, in the integrated trainer; the same text also sits in a comment)", len(ln) == 1 and txt.count("b = dag.apply(b, step)") >= 2, len(ln))
    tr.write_text(txt.replace(ln[0], ""), encoding="utf-8")
    rc, o = bash(["--check-tree", posix(mt)])
    regress("chain --check-tree on the integrated trainer with the LOOP LINE `b = dag.apply(b, step)` deleted (the comment still contains the text): FAILS", rc == 1 and "dag\\.apply" in o, o[-300:])
    shutil.rmtree(pre, ignore_errors=True)
    shutil.rmtree(mt, ignore_errors=True)
    shutil.rmtree(t, ignore_errors=True)


# ================================================================================================ E. gate assertions on a REAL selfcheck
def lines_for(sc_dir: Path, arm: str, out: Path):
    """gate / train / pin line files from the REAL gate's argv (rg/out/<dir>/gate_argv.json): the trainer's argv with a script name in front, as the queue prints it."""
    ga = jload(sc_dir / "gate_argv.json")
    ta = L.drop_flag(ga, {"--out"}, {"--selfcheck-only"})
    k = ga.index("--out")
    ta = ["--out", str(sc_dir.parent / ("run_" + arm))] + ta
    gl = "python v8f_mini_train.py " + " ".join(ga)
    tl = "python v8f_mini_train.py " + " ".join(ta)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.txt").write_text(gl)
    (out / "train.txt").write_text(tl)
    (out / "pin.txt").write_text(tl)
    return out


def section_e():
    say("== E. gate_dag_assert.py on the REAL trainer's selfcheck.json (rg/out, written by real_gate_proof.py)")
    ro = HERE / "rg" / "out"
    res = ro / "RESULT.json"
    if not res.exists():
        check("rg/out/RESULT.json exists (run real_gate_proof.py first)", False)
        return
    r = jload(res)
    check(f"real_gate_proof.py (the REAL v8f_mini_train.py --selfcheck-only on CPU, synthetic world, the queue's own tail): {r['n_pass']}/{r['n_cases']} cases pass, incl. 8 refusals of what the queue must never emit", r["all_pass"] and r["n_cases"] >= 17, f"{r['n_pass']}/{r['n_cases']}")
    t = BT / "asserte"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    ASSERT = posix(HERE / "box2_ship" / "gate_dag_assert.py")
    sc_tame = jload(ro / "dag_s0" / "selfcheck.json")
    init_sha = sc_tame["levers"]["init_ckpt"]["sha256"]
    bank_sha = hashlib.sha256((ro / "bank.json").read_bytes()).hexdigest()
    exp = {"n_clips": 6, "n_excluded": 0, "holdout_sha": None, "windows": sc_tame["info"]["n_windows_all_clips"], "bank_dir_tail": "/bank/bank.json", "init_tail": "/init_tame.pt"}
    jdump(t / "expect.json", exp)

    def run(arm, sc_path, lines, mode="check", extra=(), seed=0):
        a = ["-I", ASSERT, mode, "--selfcheck", posix(sc_path), "--gate-line", posix(lines / "gate.txt"), "--train-line", posix(lines / "train.txt"), "--pin-line", posix(lines / "pin.txt"), "--arm", arm, "--seed", str(seed), "--banked-init-sha", init_sha,
             "--expect", posix(t / "expect.json")]
        if arm == "dag":
            a += ["--p", "0.1", "--bank-sha", bank_sha, "--bank-json", posix(ro / "bank.json")]
        return sh([L.PY, *a, *extra])
    # the synthetic bank path ends with /bank/bank.json; the real one ends with /dagger_bank/s<seed>/bank.json (checked below by a mutation)
    ldag, ldag0 = lines_for(ro / "dag_s0", "dag", t / "ldag"), lines_for(ro / "dag0_s0", "dag0", t / "ldag0")
    rc, o = run("dag", ro / "dag_s0" / "selfcheck.json", ldag)
    check("gate_dag_assert check on the REAL DAG selfcheck.json: every assertion holds", rc == 0 and "DAG gate: PASS" in o, o[-900:])
    rc, o = run("dag0", ro / "dag0_s0" / "selfcheck.json", ldag0)
    check("gate_dag_assert check on the REAL DAG-0 selfcheck.json: every assertion holds", rc == 0 and "DAG gate: PASS" in o, o[-900:])

    def mutated(src, fn, name):
        sc = jload(src)
        fn(sc)
        p = t / (name + ".json")
        jdump(p, sc)
        return p
    base = ro / "dag_s0" / "selfcheck.json"
    cases = {
        "the p = 0 control reads 1e-9 instead of exactly 0.0": lambda s: s["dagger_probe"].__setitem__("control_p0_loss_diff", 1e-9),
        "the frozen-prior control reads 1e-12": lambda s: s["live"].__setitem__("control_prior_grad_sum", 1e-12),
        "the swapped batch does not move the loss (loss_diff 0)": lambda s: s["dagger_probe"].__setitem__("loss_diff", 0.0),
        "the probe did not pass": lambda s: s["dagger_probe"].__setitem__("pass", False),
        "no eligible window in the probe": lambda s: s["dagger_probe"].__setitem__("n_selected_at_p1", 0),
        "declared bank sha256 differs from the built one": lambda s: s["levers"].__setitem__("dagger_bank_sha256_declared", "0" * 64),
        "the built bank sha256 is not the pulled bank.json's": lambda s: s["levers"].__setitem__("dagger_bank_sha256", "0" * 64),
        "the init checkpoint is not the banked M2b": lambda s: s["levers"]["init_ckpt"].__setitem__("sha256", "0" * 64),
        "the init checkpoint was not applied": lambda s: s["levers"]["init_ckpt"].__setitem__("applied", False),
        "K-B2 in the gate: the bank's checkpoint is another one": lambda s: s["info"]["dagger"]["kb2"].__setitem__("ckpt_sha256", "0" * 64),
        "COVERAGE blocked in the gate": lambda s: s["info"]["dagger"]["coverage"].__setitem__("blocked", True),
        "K-B3: overlap with eval-141": lambda s: s["info"]["dagger"]["kb3"].__setitem__("overlap_with_eval", 1),
        "K-B3: window-list sha256 is not the bank's": lambda s: s["info"]["dagger"]["kb3"].__setitem__("window_list_sha256", "0" * 64),
        "steps_total is 2999": lambda s: s.__setitem__("steps_total", 2999),
        "built dagger_p is 0.3 while the run declares 0.1": lambda s: s["levers"].__setitem__("dagger_p", 0.3),
        "built dagger_seed is 1 on seed 0": lambda s: s["levers"]["dagger"].__setitem__("dagger_seed", 1),
        "a dead parameter group": lambda s: s["live"].__setitem__("dead_params", ["e_ego"]),
        "the ego-perturbation probe is not 'off' (the two levers combined)": lambda s: s["ego_perturb_probe"].__setitem__("ego_perturb", "on"),
        "the gate ran another argv (argv_sha differs)": lambda s: s.__setitem__("argv_sha", "0" * 16),
        "declared_vs_built is not PASS": lambda s: s.__setitem__("declared_vs_built", "FAIL"),
        "the P10 prior's gradient moved with the swap": lambda s: s["dagger_probe"].__setitem__("control_prior_grad_diff", 0.5),
        "the lab_s.rc frame change is off by 1e-3": lambda s: s["dagger_probe"].__setitem__("rc_max_abs_err", 1e-3),
        "bank path is not .../dagger_bank/s0/bank.json": lambda s: None,
    }
    SUBS = {'the p = 0 control reads 1e-9 instead of exactly 0.0': 'control_p0_loss_diff', 'the frozen-prior control reads 1e-12': 'prior control', 'the swapped batch does not move the loss (loss_diff 0)': 'does not reach the loss', 'the probe did not pass': 'dagger_probe', 'no eligible window in the probe': 'no eligible window', 'declared bank sha256 differs from the built one': 'bank sha256', "the built bank sha256 is not the pulled bank.json's": 'bank sha256', 'the init checkpoint is not the banked M2b': 'does not start from the banked', 'the init checkpoint was not applied': 'not applied', "K-B2 in the gate: the bank's checkpoint is another one": 'K-B2', 'COVERAGE blocked in the gate': 'COVERAGE in the gate', 'K-B3: overlap with eval-141': 'overlap with eval-141', "K-B3: window-list sha256 is not the bank's": 'window-list sha256', 'steps_total is 2999': 'steps 2999', 'built dagger_p is 0.3 while the run declares 0.1': 'built dagger_p', 'built dagger_seed is 1 on seed 0': 'built dagger_seed', 'a dead parameter group': 'dead set', "the ego-perturbation probe is not 'off' (the two levers combined)": 'ego_perturb_probe', 'the gate ran another argv (argv_sha differs)': 'ran an argv other than', 'declared_vs_built is not PASS': 'declared_vs_built', "the P10 prior's gradient moved with the swap": 'control_prior_grad_diff', 'the lab_s.rc frame change is off by 1e-3': 'frame-change error', 'bank path is not .../dagger_bank/s0/bank.json': 'built bank path'}
    for name, fn in cases.items():
        if name.startswith("bank path"):
            jdump(t / "exp_real.json", {**exp, "bank_dir_tail": "/dagger_bank/s{seed}/bank.json"})
            rc, o = sh([L.PY, "-I", ASSERT, "check", "--selfcheck", posix(base), "--gate-line", posix(ldag / "gate.txt"), "--train-line", posix(ldag / "train.txt"), "--pin-line", posix(ldag / "pin.txt"), "--arm", "dag", "--seed", "0", "--banked-init-sha", init_sha,
                           "--expect", posix(t / "exp_real.json"), "--p", "0.1", "--bank-sha", bank_sha, "--bank-json", posix(ro / "bank.json")])
        else:
            rc, o = run("dag", mutated(base, fn, "m" + str(abs(hash(name)) % 10**8)), ldag)
        regress(f"gate_dag_assert REFUSES (rc 1) for the right reason [{SUBS[name]}]: {name}", rc == 1 and "DAG gate: PASS" not in o and SUBS[name] in o, o[-300:])
    # the B-summary comparison (the banked M2b's levers: loss weights, parameter groups, window set, TRAIN list). The real banked summary has exactly these keys (checked below); here B is the synthetic world's own.
    bsum = {"levers": {"w": sc_tame["levers"]["w"], "params": sc_tame["levers"]["params"], "n_params_trainable": sc_tame["levers"]["n_params_trainable"]}, "train_sha256": sc_tame["info"]["train_sha256"], "n_windows": sc_tame["info"]["n_windows_all_clips"]}
    jdump(t / "bsum.json", bsum)
    rc, o = run("dag", base, ldag, extra=["--b-summary", posix(t / "bsum.json")])
    check("gate_dag_assert with a --b-summary equal to the built model's levers (loss weights, parameter groups, window set, TRAIN list): PASS", rc == 0 and "DAG gate: PASS" in o, o[-300:])
    for nm, fn, sub in (("B has another parameter group size", lambda b: b["levers"]["params"].__setitem__("T", 1), "parameter group"), ("B's w_pen differs", lambda b: b["levers"]["w"].__setitem__("w_pen", 9.0), "w_pen"),
                        ("B's trainable parameter count differs", lambda b: b["levers"].__setitem__("n_params_trainable", 1), "n_params_trainable"), ("B's TRAIN list differs", lambda b: b.__setitem__("train_sha256", "0" * 64), "TRAIN list"),
                        ("B's window set differs", lambda b: b.__setitem__("n_windows", 1), "window set")):
        b2 = copy.deepcopy(bsum)
        fn(b2)
        jdump(t / "bsum2.json", b2)
        rc, o = run("dag", base, ldag, extra=["--b-summary", posix(t / "bsum2.json")])
        regress(f"gate_dag_assert REFUSES [{sub}] against the banked-B comparison: {nm}", rc == 1 and sub in o, o[-250:])
    real_b = jload(SHIP / "logs" / "v8f_mini" / "runs" / "M2b_s0" / "summary.json")
    check("the REAL banked M2b_s0 summary.json has every key the --b-summary comparison reads (levers.w.{w_pen,w_pen_self,w_wm}, levers.params, levers.n_params_trainable, train_sha256, n_windows)",
          all(k in real_b["levers"]["w"] for k in ("w_pen", "w_pen_self", "w_wm")) and "params" in real_b["levers"] and "n_params_trainable" in real_b["levers"] and "train_sha256" in real_b and real_b["n_windows"] == 306566 and real_b["n_clips"] == 4512)
    # DAG-0 mutations
    base0 = ro / "dag0_s0" / "selfcheck.json"
    rcz, oz = run("dag0", mutated(base0, lambda s: s["dagger_probe"].__setitem__("dagger", "on"), "z0"), ldag0)
    regress("gate_dag_assert REFUSES a DAG-0 selfcheck whose probe is not 'off'", rcz == 1 and "should read 'off'" in oz, oz[-200:])
    rcz, oz = run("dag0", mutated(base0, lambda s: s["levers"].__setitem__("dagger_bank_sha256", "a" * 64), "z1"), ldag0)
    regress("gate_dag_assert REFUSES a DAG-0 that built a bank hash (a lever declared and inert)", rcz == 1 and "inert" in oz, oz[-200:])
    # line mutations
    gl = (ldag / "gate.txt").read_text()
    tl = (ldag / "train.txt").read_text()
    LS = {"the PIN differs": "THE PIN IS NOT", "the TRAIN line carries": "carries --selfcheck-only", "the gate line is not": "gate argv is not the training argv", "the train line names --ego": "names --ego-perturb-vmax",
          "the train line sets a second": "second lever", "the train line does not name": "does not name --dagger-bank-sha256 exactly once"}
    for name, (g2, t2, p2) in {
        "the PIN differs from the planned training line by one token (box 2 would run another argv than the token binds)": (gl, tl, tl.replace("--dagger-seed 0", "--dagger-seed 1")),
        "the TRAIN line carries --selfcheck-only": (gl, tl + " --selfcheck-only", tl + " --selfcheck-only"),
        "the gate line is not the train line (--lr changed)": (gl.replace("--lr 1e-4", "--lr 2e-4"), tl, tl),
        "the train line names --ego-perturb-vmax (a law flag at p = 0)": (gl + " --ego-perturb-vmax 8.0", tl + " --ego-perturb-vmax 8.0", tl + " --ego-perturb-vmax 8.0"),
        "the train line sets a second lever (--w-acc 0.5)": (gl, tl + " --w-acc 0.5", tl + " --w-acc 0.5"),
        "the train line does not name --dagger-bank-sha256": (gl, re.sub(r" --dagger-bank-sha256 \w+", "", tl), re.sub(r" --dagger-bank-sha256 \w+", "", tl)),
    }.items():
        d = t / ("l_" + hashlib.sha256(name.encode()).hexdigest()[:6])
        d.mkdir()
        (d / "gate.txt").write_text(g2)
        (d / "train.txt").write_text(t2)
        (d / "pin.txt").write_text(p2)
        rc, o = run("dag", base, d)
        sub = [v for k, v in LS.items() if name.startswith(k)][0]
        regress(f"gate_dag_assert REFUSES [{sub}]: {name}", rc == 1 and sub in o, o[-250:])
    # DAG-0 line that names the bank
    d = t / "l_dag0bank"
    d.mkdir()
    g0, t0 = (ldag0 / "gate.txt").read_text(), (ldag0 / "train.txt").read_text()
    for nm, extra in (("--dagger-bank", " --dagger-bank X"), ("--dagger-seed", " --dagger-seed 1")):
        (d / "gate.txt").write_text(g0 + extra)
        (d / "train.txt").write_text(t0 + extra)
        (d / "pin.txt").write_text(t0 + extra)
        rcz, oz = run("dag0", base0, d)
        regress(f"gate_dag_assert REFUSES a DAG-0 line that names {nm}", rcz == 1 and f"names {nm} on DAG-0" in oz, oz[-250:])
    # the token
    tok = t / "token.json"
    commit = "ab" * 20
    rc, o = run("dag", base, ldag, mode="token", extra=["--commit", commit, "--queue-sha", "c" * 64, "--key-sha-json", '{"x": "y"}', "--orc-read-sha", "d" * 64, "--run", "M2b_s0_dag", "--out", posix(tok)])
    tk = jload(tok) if tok.exists() else {}
    pin_argv = L.train_argv((ldag / "pin.txt").read_text())
    check("token: written after the assertions pass; kind/verdict; binding = (commit, sha256 of the PINNED training argv, the gate's train_sha256); carries the bank sha256, p and the ORC read sha256",
          rc == 0 and tk.get("kind") == "launch-gate-token" and tk.get("verdict") == "PASS" and tk["binding"]["commit"] == commit and tk["binding"]["argv_sha256"] == L.argv_sha256(pin_argv) and tk["binding"]["data_sha256"] == sc_tame["info"]["train_sha256"]
          and tk.get("bank_sha256") == bank_sha and tk.get("dagger_p") == 0.1 and tk.get("orc_read_sha256") == "d" * 64, o[-300:])
    check("token: the argv_sha256 it binds differs from the GATE argv's sha256 (it binds the training run's argv, not the gate's)", bool(tk) and tk["binding"]["argv_sha256"] != tk["gate"]["argv_sha256"])
    tok2 = t / "token2.json"
    rc, o = run("dag", mutated(base, cases["the p = 0 control reads 1e-9 instead of exactly 0.0"], "tk"), ldag, mode="token", extra=["--commit", commit, "--queue-sha", "c" * 64, "--orc-read-sha", "d" * 64, "--run", "x", "--out", posix(tok2)])
    regress("token: NOT written when an assertion fails", rc == 1 and not tok2.exists())
    rc, o = run("dag", base, ldag, mode="token", extra=["--commit", "abc", "--queue-sha", "c" * 64, "--orc-read-sha", "d" * 64, "--run", "x", "--out", posix(t / "token3.json")])
    regress("token: NOT written for a commit that is not 40 hex", rc == 1 and not (t / "token3.json").exists())
    shutil.rmtree(t, ignore_errors=True)


# ================================================================================================ F. tool, guard, shas, diff
def section_f():
    say("== F. the shard tool, the guard copy, the banked hashes, the additive queue diff")
    tool = HERE / "stage" / "stack" / "scripts" / "v8f_dagger_queue_tool.py"
    t = BT / "toolf"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    stub = t / "v8f_dagger_bank.py"
    stub.write_text(FAKE_BANK)
    gen = ["--ckpt", "c.pt", "--batch", "32"]
    out = t / "bank"
    rc, o = sh([L.PY, posix(tool), "bank", "--n-shards", "4", "--out-dir", posix(out), "--bank-script", posix(stub), "--poll-s", "0.2", "--", *gen])
    check("shard tool: 4 shards launched concurrently, merge ran, rc 0, prints the bank sha256", rc == 0 and (out / "bank.json").exists() and f"bank sha256 {hashlib.sha256((out / 'bank.json').read_bytes()).hexdigest()}" in o, o[-300:])
    rc, o = sh([L.PY, posix(tool), "bank", "--n-shards", "4", "--out-dir", posix(out), "--bank-script", posix(stub), "--poll-s", "0.2", "--", *gen])
    check("shard tool: second call skips (a merged bank exists)", rc == 0 and "[skip]" in o)
    out2 = t / "bank2"
    rc, o = sh([L.PY, posix(tool), "bank", "--n-shards", "4", "--out-dir", posix(out2), "--bank-script", posix(stub), "--poll-s", "0.2", "--", *gen], env={"STUB_FAIL_SHARD": "2", "STUB_SLOW_SHARD": "1"})
    regress("shard tool: shard 2 fails while shard 1 is still running -> rc 1, the slow shard is TERMINATED, NO merge, no bank.json", rc == 1 and not (out2 / "bank.json").exists() and "terminated" in o and "no merge" in o, o[-400:])
    shutil.rmtree(out2)
    out2.mkdir()
    for k in (0, 1):                                              # shards 0, 1 already complete from an earlier attempt: only 2 and 3 run
        (out2 / f"shard_00{k}of004.npz").write_bytes(b"x")
        (out2 / f"shard_00{k}of004.provenance.json").write_text("{}")
    rc, o = sh([L.PY, posix(tool), "bank", "--n-shards", "4", "--out-dir", posix(out2), "--bank-script", posix(stub), "--poll-s", "0.2", "--", *gen])
    check("shard tool: resume runs only the missing shards (2 and 3) and merges", rc == 0 and not (out2 / "gen_argv_0.json").exists() and (out2 / "gen_argv_2.json").exists() and (out2 / "gen_argv_3.json").exists() and (out2 / "bank.json").exists(), o[-300:])
    rc, o = sh([L.PY, posix(tool), "bank", "--n-shards", "4", "--out-dir", posix(t / "bank3"), "--bank-script", posix(stub), "--", *gen, "--shard", "1"])
    regress("shard tool REFUSES a generator argument that is set by the tool (--shard), rc 2", rc == 2)
    f = t / "f.bin"
    f.write_bytes(b"abc")
    rc, o = sh([L.PY, posix(tool), "sha", posix(f)])
    check("shard tool: `sha FILE` prints the file's sha256", rc == 0 and o.strip() == hashlib.sha256(b"abc").hexdigest())
    regress("shard tool: `sha` of a missing file is REFUSED (rc 2)", sh([L.PY, posix(tool), "sha", posix(t / "nope")])[0] == 2)
    # ---- guard copy: ONLY the process pattern (and one comment) changed
    a = (SHIP / "v8f_guard.sh").read_text(encoding="utf-8").splitlines()
    b = (HERE / "box2_ship" / "v8f_guard_dag.sh").read_text(encoding="utf-8").splitlines()
    import difflib
    ch = [x for x in difflib.unified_diff(a, b, lineterm="", n=0) if x[:1] in "+-" and not x.startswith(("---", "+++"))]
    check("guard copy: differs from v8f_guard.sh in exactly the kill pattern (1 line replaced) and one added comment line; the new pattern adds v8f_dagger_bank and v8f_dagger_queue_tool",
          len([x for x in ch if x.startswith("-")]) == 1 and "v8f_dagger_bank|v8f_dagger_queue_tool" in "\n".join(ch) and "(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)" in "\n".join(b), ch[:6])
    # ---- the banked checkpoint hashes: copied, never retyped
    orc = (SHIP / "chain_v8f_orc.sh").read_text(encoding="utf-8")
    m = re.search(r"^CKPT_SHA=\((\w{64}) (\w{64})\)", orc, flags=re.M)
    check("chain CKPT_SHA == chain_v8f_orc.sh's CKPT_SHA (s0 601cb435..., s1 6104fb15...)", (L_ck(0), L_ck(1)) == (m.group(1), m.group(2)) and L_ck(0).startswith("601cb435") and L_ck(1).startswith("6104fb15"))
    for s in (0, 1):
        p = SHIP / "logs" / "v8f_mini" / "runs" / f"M2b_s{s}" / "ckpt_last.pt"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        check(f"M2b_s{s}: sha256 of the BANKED checkpoint file on this disk == the chain's value", h == L_ck(s), h[:16])
    man = (Path("E:/Projects/TanitAD-artifacts/di/tree") / "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/full/MANIFEST.md").read_text(encoding="utf-8")
    check("raw/full/MANIFEST.md names both banked checkpoint hashes", L_ck(0)[:12] in man or L_ck(0) in man, "")
    # ---- the additive diff, re-derived
    import build_queue as BQ
    check("the staged queue is exactly tip + doc + dispatch + stages (re-derived from the additions)", BQ.build() == L.QUEUE.read_bytes())
    tip, new = L.TIP_QUEUE.read_bytes().decode().split("\r\n"), L.QUEUE.read_bytes().decode().split("\r\n")
    sm = difflib.SequenceMatcher(None, tip, new, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    check("the staged queue's diff against the tip has NO removed and NO replaced line (insertions only)", ops and all(o[0] == "insert" for o in ops), [o[0] for o in ops])
    check("the tip queue's SHA256 is the committed blob's (2a78f059) c92ca88d...", hashlib.sha256(L.TIP_QUEUE.read_bytes()).hexdigest().startswith("c92ca88d"))
    shutil.rmtree(t, ignore_errors=True)


# ================================================================================================ G. the banks the chain PUBLISHES to the dev box
def mk_published_bank(d: Path, seed: int, ckpt: str, n_shards=4, frac=0.6, mutate=None, npz_bytes=4096):
    """mk_bank (what the judge reads) + what the chain PUBLISHES: small stand-in shard npz files whose REAL sha256 is recorded in bank.json AND in each shard provenance, PROVENANCE.json naming its shard provenance files /
    manifest / bank_sha256 (as the generator's merge writes them), and bank_stage.log. The npz are stand-ins (the chain never opens them; the dev-box reader does)."""
    d = mk_bank(d, seed, ckpt, frac=frac, n_shards=n_shards, mutate=mutate)
    man, prov = jload(d / "bank.json"), jload(d / "PROVENANCE.json")
    for k in range(n_shards):
        npz = d / f"shard_{k:03d}of{n_shards:03d}.npz"
        npz.write_bytes((hashlib.sha256(f"stand-in shard npz, seed {seed}, shard {k}".encode()).digest() * (npz_bytes // 32 + 1))[:npz_bytes])
        h = hashlib.sha256(npz.read_bytes()).hexdigest()
        man["shards"][k]["file_sha256"] = h
        sp = jload(d / f"shard_{k:03d}of{n_shards:03d}.provenance.json")
        sp["file_sha256"] = h
        jdump(d / f"shard_{k:03d}of{n_shards:03d}.provenance.json", sp)
    jdump(d / "bank.json", man)
    prov["shard_provenance"] = [f"shard_{k:03d}of{n_shards:03d}.provenance.json" for k in range(n_shards)]
    prov["manifest"] = "bank.json"
    prov["bank_sha256"] = hashlib.sha256((d / "bank.json").read_bytes()).hexdigest()
    jdump(d / "PROVENANCE.json", prov)
    (d / "bank_stage.log").write_text("bank stage log (stand-in)\n", encoding="utf-8")
    return d


def write_sums(d: Path, names=None, extra=None, newline=True, bad_hash_for=None):
    """SHA256SUMS.txt over ``names`` (default: every file the chain publishes), written independently of the chain (hashlib here)."""
    names = names if names is not None else CK.bank_flat(CK.bank_names(4))
    lines = [f"{'0' * 64 if n == bad_hash_for else hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}" for n in names]
    if extra:
        lines.append(f"{'0' * 64}  {extra}")
    (d / "SHA256SUMS.txt").write_bytes(("\n".join(lines) + ("\n" if newline else "")).encode("utf-8"))


def flip_bit(p: Path):
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 1
    p.write_bytes(bytes(b))


def section_g():
    say("== G. the banks the chain publishes to the dev box (the four kinds the reader needs, SHA256SUMS.txt, nothing moved before the judge passes)")
    t = BT / "chaing"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    s0, s1 = L_ck(0), L_ck(1)
    harms = posix(orc_file(t / "harms.json", "HARMS"))
    ckpy = [sys.executable, "-I", str(HERE / "box2_ship" / "v8f_dag_chain_checks.py")]
    # ------------------------------------------------ G1: the checks (in-process, so a mutant of v8f_dag_chain_checks.py is seen)
    names = CK.bank_names(4)
    expect = {"provenance": ["PROVENANCE.json"], "manifest": ["bank.json"], "shard_provenance": [f"shard_00{k}of004.provenance.json" for k in range(4)], "shard_npz": [f"shard_00{k}of004.npz" for k in range(4)], "stage_log": ["bank_stage.log"]}
    check("bank names: the FOUR kinds the reader needs are listed explicitly (PROVENANCE.json, bank.json, 4 shard provenance, 4 shard npz) + the stage log = 11 names", names == expect and len(CK.bank_flat(names)) == 11 and len(set(CK.bank_flat(names))) == 11, names)
    good = mk_published_bank(t / "good", 0, s0)
    v = CK.bank_verify(str(good))
    check("bank-verify on a complete published bank (no SHA256SUMS.txt required): ok, 11 files, every kind non-empty", v["ok"] and v["n_files"] == 11 and all(v["bytes_by_kind"][k] > 0 for k in CK.BANK_KINDS) and v["bank_sha256"] == hashlib.sha256((good / "bank.json").read_bytes()).hexdigest(), v["bad"])
    v = CK.bank_verify(str(good), require_sums=True)
    regress("bank-verify --require-sums with NO SHA256SUMS.txt: not ok, says it is absent", (not v["ok"]) and any("SHA256SUMS.txt is absent" in b for b in v["bad"]), v["bad"])
    write_sums(good)
    v = CK.bank_verify(str(good), require_sums=True)
    check("bank-verify --require-sums on a SHA256SUMS.txt written independently (hashlib) over exactly the 11 files: ok", v["ok"], v["bad"])

    def vreg(name, edit, want, **kw):
        """edit a copy of the good bank, run bank-verify, require NOT ok and the reason to contain ``want``"""
        c = t / ("c_" + re.sub(r"\W", "", name)[:40])
        shutil.copytree(good, c)
        edit(c)
        v_ = CK.bank_verify(str(c), **kw)
        regress(f"bank-verify: {name} -> not ok, and the reason names it ({want!r})", (not v_["ok"]) and any(want in b for b in v_["bad"]), v_["bad"][:3])
        shutil.rmtree(c, ignore_errors=True)
    vreg("a shard npz that was never pulled (shard 2)", lambda c: (c / "shard_002of004.npz").unlink(), "shard_002of004.npz is absent")
    vreg("an EMPTY shard npz (shard 1)", lambda c: (c / "shard_001of004.npz").write_bytes(b""), "shard_001of004.npz is empty")
    vreg("one flipped bit in shard 3's npz (same size)", lambda c: flip_bit(c / "shard_003of004.npz"), "shard npz shard_003of004.npz hashes to")
    vreg("bank.json edited after PROVENANCE.json named its sha256", lambda c: (c / "bank.json").write_bytes((c / "bank.json").read_bytes() + b" "), "PROVENANCE.bank_sha256")
    vreg("a shard provenance file missing", lambda c: (c / "shard_002of004.provenance.json").unlink(), "shard_002of004.provenance.json is absent")
    vreg("a shard provenance that records another file_sha256", lambda c: jdump(c / "shard_001of004.provenance.json", {**jload(c / "shard_001of004.provenance.json"), "file_sha256": "0" * 64}), "records file_sha256")
    vreg("PROVENANCE.json missing", lambda c: (c / "PROVENANCE.json").unlink(), "cannot read bank.json / PROVENANCE.json")
    vreg("bank.json missing", lambda c: (c / "bank.json").unlink(), "cannot read bank.json / PROVENANCE.json")
    vreg("bank_stage.log missing", lambda c: (c / "bank_stage.log").unlink(), "bank_stage.log is absent")
    vreg("bank.json says 3 shards (the queue runs 4)", lambda c: jdump(c / "bank.json", {**jload(c / "bank.json"), "n_shards": 3}), "the queue's bankdag runs 4")
    vreg("PROVENANCE.json lists 3 of the 4 shard provenance files", lambda c: jdump(c / "PROVENANCE.json", {**jload(c / "PROVENANCE.json"), "shard_provenance": jload(c / "PROVENANCE.json")["shard_provenance"][:3]}), "shard_provenance is")
    vreg("bank.json names a shard npz that is not the generator's", lambda c: jdump(c / "bank.json", {**jload(c / "bank.json"), "shards": [{**s_, "file": "other.npz"} if s_["k"] == 2 else s_ for s_ in jload(c / "bank.json")["shards"]]}), "bank.json shards name")
    # SHA256SUMS.txt faults (each against an otherwise perfect directory)
    vreg("SHA256SUMS.txt lists 10 names (a shard npz left out)", lambda c: write_sums(c, [n for n in CK.bank_flat(names) if n != "shard_002of004.npz"]), "expected exactly 11", require_sums=True)
    vreg("SHA256SUMS.txt lists a 12th, unexpected name", lambda c: write_sums(c, extra="gen_shard_0.log"), "unexpected ['gen_shard_0.log']", require_sums=True)
    vreg("SHA256SUMS.txt carries a wrong hash for one npz", lambda c: write_sums(c, bad_hash_for="shard_001of004.npz"), "lists 000000000000 for shard_001of004.npz", require_sums=True)
    vreg("SHA256SUMS.txt without a final newline (a partial last line)", lambda c: write_sums(c, newline=False), "does not end with a newline", require_sums=True)
    vreg("SHA256SUMS.txt lists a name twice", lambda c: (c / "SHA256SUMS.txt").write_bytes((c / "SHA256SUMS.txt").read_bytes() + (c / "SHA256SUMS.txt").read_bytes().splitlines(True)[0]), "twice", require_sums=True)
    # the CLI: files named from the pulled bank.json, rc 2 on a bank that cannot be named
    rc, o = sh(ckpy + ["bank-files", posix(good), "--kind", "shard_npz"])
    check("bank-files --kind shard_npz (CLI): the 4 npz names, one per line", rc == 0 and o.split() == expect["shard_npz"], o)
    rc, o = sh(ckpy + ["bank-files", posix(good), "--kind", "small"])
    check("bank-files --kind small (CLI): every file but the npz (7 names: what the judge reads + the log)", rc == 0 and o.split() == expect["provenance"] + expect["manifest"] + expect["shard_provenance"] + expect["stage_log"], o)
    rc, o = sh(ckpy + ["bank-files", posix(t / "nowhere")])
    regress("bank-files on a directory with no bank.json: rc 2 and a reason (the chain's `names=$(...) || stop` then stops; an empty list would silently pull nothing)", rc == 2 and "cannot read bank.json" in o, o)
    rc, o = sh(ckpy + ["bank-verify", posix(good), "--require-sums"])
    check("bank-verify (CLI) on the sealed good bank: rc 0, one ZZDAG-BANKFILES line with n_files 11", rc == 0 and "ZZDAG-BANKFILES" in o and '"n_files": 11' in o, o[:200])
    est = CK.est_bank_bytes()
    check("ESTIMATED size of one seed's published bank (smoke 804 windows -> 306,566 windows; the arithmetic is bank-plan's): 0.62-0.72 GB, almost all of it the shard npz", 0.62e9 <= est["seed_total_bytes"] <= 0.72e9 and est["npz_total_bytes"] > 0.99 * est["seed_total_bytes"], est)
    rc, o = sh(ckpy + ["bank-plan"])
    check("bank-plan (CLI) prints the four kinds, `exactly 11 names` and the ESTIMATED size", rc == 0 and "exactly 11 names" in o and "ESTIMATED" in o and "kind shard_npz: shard_000of004.npz" in o, o)
    shutil.rmtree(t, ignore_errors=True)


def section_h():
    say("== H. the chain's own publish flow, offline (--publish-only: the REAL bank_fetch_small -> judge -> publish_bank_one; only pull / rsha / stop are replaced)")
    t = BT / "chainh"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    s0, s1 = L_ck(0), L_ck(1)
    harms = posix(orc_file(t / "harms.json", "HARMS"))
    names = CK.bank_names(4)

    def world(name, f0=None, f1=None, m0=None, m1=None):
        w = t / name
        mk_published_bank(w / "dagger_bank" / "s0", 0, s0, frac=f0 if f0 is not None else 0.62, mutate=m0)
        mk_published_bank(w / "dagger_bank" / "s1", 1, s1, frac=f1 if f1 is not None else 0.58, mutate=m1)
        return w

    def run_pub(w, art, env=None):
        shutil.rmtree(art, ignore_errors=True)
        return bash(["--publish-only", posix(w)], {"DAG_ART": posix(art), "DAG_ORC_READ": harms, **(env or {})})

    def npz_in(art):
        return sorted(p.name for s in (0, 1) for p in (art / f"bank_s{s}").glob("*.npz")) if art.exists() else []
    w = world("w_good")
    art = t / "art_good"
    rc, o = run_pub(w, art)
    sums_ok, why = True, []
    for s in (0, 1):
        d = art / f"bank_s{s}"
        lines = (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if (d / "SHA256SUMS.txt").exists() else []
        listed = {ln[66:]: ln[:64] for ln in lines}
        if sorted(listed) != sorted(CK.bank_flat(names)) or len(lines) != 11:
            sums_ok = False
            why.append(f"s{s}: {len(lines)} lines")
        for nm, h in listed.items():
            if not (d / nm).exists() or hashlib.sha256((d / nm).read_bytes()).hexdigest() != h or h != hashlib.sha256((w / "dagger_bank" / f"s{s}" / nm).read_bytes()).hexdigest():
                sums_ok = False
                why.append(f"s{s} {nm}")
    check("chain --publish-only (REAL fetch -> judge -> publish; a stand-in for box 2): rc 0, ZZDAG-JUDGE-PASS, then BOTH seeds publish into bank_s0 / bank_s1: 11 files each, SHA256SUMS.txt = 11 lines whose hashes equal both the pulled files' and the stand-in box's (hashlib, independent of the chain)",
          rc == 0 and "ZZDAG-JUDGE-PASS" in o and "ZZDAG-PUBLISH-ONLY-DONE" in o and sums_ok and len(npz_in(art)) == 8, (rc, why, o[-500:]))
    i_pass, i_p0, i_p1 = o.find("ZZDAG-JUDGE-PASS"), o.find("bank s0 pulled to"), o.find("bank s1 pulled to")
    check("chain --publish-only: the marker is the checkpoints' one (`bank s<S> pulled to <dir> (11 files ...)`), printed for each seed AFTER the judge's PASS and never before; the small-file fetch does not use that marker", 0 < i_pass < i_p0 < i_p1 and "11 files" in o[i_p0:i_p0 + 200] and o.count(" pulled to ") == 2, o[-600:])
    # nothing is moved before the judge passes
    wb = world("w_blk", f1=0.49)
    art = t / "art_blk"
    rc, o = run_pub(wb, art)
    regress("chain --publish-only with COVERAGE 0.49 at seed 1 (BLOCKED): stops with `ZZV8FMINI-STOP BLOCKED`, NO shard npz was pulled for either seed, NO SHA256SUMS.txt", rc == 1 and "ZZV8FMINI-STOP BLOCKED" in o and npz_in(art) == [] and not list(art.glob("bank_s*/SHA256SUMS.txt")), (rc, npz_in(art), o[-300:]))
    wr = world("w_ref", m1=lambda man, prov, provs: prov["controls"].pop("K-B4"))
    art = t / "art_ref"
    rc, o = run_pub(wr, art)
    regress("chain --publish-only with a MISSING K-B4 record at seed 1 (REFUSED): stops with `ZZV8FMINI-STOP REFUSED`, no shard npz for EITHER seed (seed 0 was fine and is still not moved), no SHA256SUMS.txt", rc == 1 and "ZZV8FMINI-STOP REFUSED" in o and npz_in(art) == [] and not list(art.glob("bank_s*/SHA256SUMS.txt")), (rc, npz_in(art), o[-300:]))
    # faults during the publish itself
    wm = world("w_miss")
    (wm / "dagger_bank" / "s1" / "shard_003of004.npz").unlink()
    art = t / "art_miss"
    rc, o = run_pub(wm, art)
    regress("chain --publish-only with shard 3's npz ABSENT on box 2 (seed 1): `ZZV8FMINI-STOP pull ...`, seed 1 gets NO SHA256SUMS.txt (never sealed), seed 0 (complete) is sealed", rc == 1 and "ZZV8FMINI-STOP pull" in o and "shard_003of004.npz" in o and not (art / "bank_s1" / "SHA256SUMS.txt").exists() and (art / "bank_s0" / "SHA256SUMS.txt").exists(), (rc, o[-300:]))
    wc = world("w_corrupt")
    flip_bit(wc / "dagger_bank" / "s0" / "shard_001of004.npz")
    art = t / "art_corrupt"
    rc, o = run_pub(wc, art)
    regress("chain --publish-only with a shard npz that does not hash to bank.json's file_sha256 (the generator's record): `nothing was sealed`, NO SHA256SUMS.txt at that seed, rc 1", rc == 1 and "ZZV8FMINI-STOP" in o and "nothing was sealed" in o and "shard_001of004.npz hashes to" in o and not (art / "bank_s0" / "SHA256SUMS.txt").exists(), (rc, o[-400:]))
    pristine = world("w_pristine")
    wt = t / "w_transfer"
    shutil.copytree(pristine, wt)
    flip_bit(wt / "dagger_bank" / "s0" / "shard_002of004.npz")
    art = t / "art_transfer"
    rc, o = run_pub(wt, art, {"FAKE_HASH_ROOT": posix(pristine)})
    regress("chain --publish-only with a TRANSFER that corrupts (box 2's own sha256 of shard 2 != the pulled file's): `ZZV8FMINI-STOP bank s0: box 2's shard_002of004.npz sha256 ... != the pulled file's`, no SHA256SUMS.txt", rc == 1 and "box 2's shard_002of004.npz sha256" in o and not (art / "bank_s0" / "SHA256SUMS.txt").exists(), (rc, o[-300:]))
    # a SHA256SUMS.txt of an EARLIER attempt must not survive into a new fetch
    wst = world("w_stale")
    (wst / "dagger_bank" / "s0" / "shard_000of004.npz").unlink()
    art = t / "art_stale"
    shutil.rmtree(art, ignore_errors=True)
    (art / "bank_s0").mkdir(parents=True)
    (art / "bank_s0" / "SHA256SUMS.txt").write_text(f"{'0' * 64}  PROVENANCE.json\n", encoding="utf-8")
    rc, o = bash(["--publish-only", posix(wst)], {"DAG_ART": posix(art), "DAG_ORC_READ": harms})
    regress("chain --publish-only over a directory that holds an EARLIER attempt's SHA256SUMS.txt, with an npz missing now: that SHA256SUMS.txt is REMOVED at the start of the fetch (a stale seal can never describe a new, incomplete pull)", rc == 1 and not (art / "bank_s0" / "SHA256SUMS.txt").exists(), (rc, list((art / "bank_s0").glob("*"))[:3], o[-200:]))
    # a SEAL that comes out short (sha256sum drops a line: a full disk, a killed pipe): the chain's own re-verification must catch it, REMOVE the file and stop. Fault injection without touching the chain:
    # a bash FUNCTION named sha256sum, defined through BASH_ENV (non-interactive bash sources it; Git bash puts /usr/bin first on PATH, so a PATH wrapper would never be found), drops the named line when the call has more than 3 arguments (the seal).
    wrap = t / "wrap"
    wrap.mkdir()
    (wrap / "env.sh").write_bytes(b'sha256sum() { if [ -n "${FAKE_DROP:-}" ] && [ $# -gt 3 ]; then command sha256sum "$@" | grep -v "[ *]${FAKE_DROP}\\$"; else command sha256sum "$@"; fi; }\n')
    benv = {"BASH_ENV": posix(wrap / "env.sh")}
    art = t / "art_short"
    rc, o = run_pub(w, art, {"FAKE_DROP": "bank_stage.log", **benv})
    regress("chain --publish-only whose sha256sum drops one line (a SHORT SHA256SUMS.txt): the post-seal `bank-verify --require-sums` catches it, `SHA256SUMS.txt does not verify (removed)`, the short file is REMOVED, rc 1",
            rc == 1 and "SHA256SUMS.txt does not verify (removed)" in o and "expected exactly 11" in o and not (art / "bank_s0" / "SHA256SUMS.txt").exists(), (rc, o[-400:]))
    rc, o = run_pub(w, t / "art_wrapctl", benv)
    check("(control for the wrapper) the same BASH_ENV sha256sum function WITHOUT FAKE_DROP seals both banks normally: the failure above came from the dropped line, not from the wrapper", rc == 0 and "ZZDAG-PUBLISH-ONLY-DONE" in o, o[-200:])
    rc, o = bash(["--publish-only", posix(w)], {"DAG_ORC_READ": harms})
    check("chain --publish-only refuses to run without DAG_ART (it never writes the real artifact directory)", rc == 1 and "needs DAG_ART" in o, o[-200:])
    src = CHAIN.read_text(encoding="utf-8")
    check("chain: the real flow calls publish_bank_one for both seeds exactly once, AFTER the judge's stop, and nowhere before it", src.count("publish_bank_one $s; done") == 2 and src.rfind("for s in 0 1; do publish_bank_one $s; done") > src.find("if [ $jr != 0 ]; then", src.rfind('judge_banks "$ART/bank_s0"')) > 0, src.count("publish_bank_one"))
    shutil.rmtree(t, ignore_errors=True)


def main():
    BT.mkdir(exist_ok=True)
    t0 = time.time()
    only = os.environ.get("DAGQ_SECTIONS", "ABCDEFGH")                       # the mutation runner runs "AB" against a mutated queue
    for fn in [f for f, k in zip((section_a, section_b, section_c, section_d, section_e, section_f, section_g, section_h), "ABCDEFGH") if k in only]:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001  a section that crashes is a FAILURE, never a silent skip
            import traceback
            N["check"] += 1
            N["check_fail"] += 1
            say(f"FAIL  section {fn.__name__} CRASHED: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    summary = (f"SELFTEST: {N['check'] - N['check_fail']}/{N['check']} checks pass ({N['check_fail']} failed); {N['reg'] - N['reg_missed']}/{N['reg']} deliberate regressions caught ({N['reg_missed']} missed); {time.time() - t0:.0f} s")
    say(summary)
    ok = N["check_fail"] == 0 and N["reg_missed"] == 0
    say("RESULT: " + ("ALL PASS" if ok else "FAILURES PRESENT"))
    (Path(os.environ["DAGQ_OUT"]) if os.environ.get("DAGQ_OUT") else HERE / "selftest_output.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
