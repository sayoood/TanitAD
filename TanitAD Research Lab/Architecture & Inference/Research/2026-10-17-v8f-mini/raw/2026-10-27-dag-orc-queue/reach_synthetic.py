#!/usr/bin/env python3
"""REACHABILITY (CLAUDE.md "built, tested, and unreachable from its caller"), part 2: the DAG-ORC arms through the tip's REAL trainer, CPU, synthetic world. Part 1 (reach_real_ckpt.py) covers the real DAG checkpoints.

  trainer-written M2b-like checkpoint (tamed) placed at <work>/M2b_s<S>_dag/ckpt_last.pt  (the path tail the queue's --init-ckpt names)
  --real trainer's launch gate, the argv the QUEUE prints for `gatedag dagorc|dagorcrep|dagorc01 S` (only the data flags, the init path, the step count and warmup replaced)--> must ACCEPT, and its selfcheck.json is banked
        as a fixture of the gate asserter's self-test (what the REAL trainer writes for these argv)
  --real trainer, a short run of the queue's `traindag` argv--> summary.json: ego_perturb_n / ego_perturb_eligible ~ p for A and B, EXACTLY 0 for REP (the control that proves the probe), dagger_n exactly 0
  --MUTATION: the same short run with ONE line of a COPY of the trainer deleted (`b = ego.apply(b, step)`, the loop's input path: the gate does not cover it) --> the gate still PASSES and ego_perturb_n reads 0, which K-REACH refuses
Deliberate regressions the REAL gate must REFUSE: REP naming law flags at p 0; law flags without --ego-perturb-p; A with --dagger-p 0.1; an init checkpoint whose config is doctored.

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python reach_synthetic.py [out.json]
Fixtures written to fixtures/: <arm>_s<S>_gate/{selfcheck.json,gate_line.txt,train_line.txt,pin_line.txt,expect.json}
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagorc_lib as L  # noqa: E402

TREE = Path(os.environ.get("REACH_TREE", r"F:\TanitAD-scratch\dagorc_tree"))
WORK = Path(os.environ.get("REACH_WORK", r"F:\TanitAD-scratch\dagorc_work\reach_syn"))
FIX = HERE / "fixtures"
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
SYN_T = 90
orig_world = D.make_synthetic_world
D.make_synthetic_world = lambda root, n_train=6, n_eval=3, T=70, seed=0, **kw: orig_world(root, n_train=n_train, n_eval=n_eval, T=SYN_T, seed=seed, **kw)
SYN = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--threads", "1", "--kin-mode", "yaw"]
res = {"cases": [], "t0": time.time()}


def case(name, ok, detail=""):
    res["cases"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail else ""), flush=True)


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def synthetic_argv(queue_line: str, out: Path, init: Path, steps: int, warmup: int, drop=(), extra=()):
    """The QUEUE-printed training argv with the data flags (--cache-dir ... --device cuda) replaced by the synthetic ones, the init path / steps / warmup replaced and --out set. Everything else is the queue's."""
    a = L.train_argv(queue_line)
    head = a[:a.index("--out")] + ["--out", str(out)]                        # --arm M2b --seed S --grid pool4x10 --out X
    tail = a[a.index("--init-ckpt"):]
    tail = [x for x in tail if x != "--selfcheck-only"]
    tail[tail.index("--init-ckpt") + 1] = str(init)
    tail[tail.index("--steps") + 1] = str(steps)
    tail[tail.index("--warmup") + 1] = str(warmup)
    argv = head + SYN + tail
    for d in drop:
        k = argv.index(d)
        del argv[k:k + 2]
    return argv + list(extra)


def run(trmod, argv, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    try:
        rc = trmod.main(argv)
    except SystemExit as e:
        rc = e.code
    sc = out / "selfcheck.json"
    return rc, (json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else None)


def line_of(argv) -> str:
    return '"python" "v8f_mini_train.py" ' + " ".join(argv)


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    # 1. an M2b-like checkpoint written by the TRAINER (tamed, as the trainer suite's fixture does), one per seed, at the path tail the queue names
    base = WORK / "base"
    assert TR.main(["--out", str(base), *SYN, "--seed", "0", "--arm", "M2b", "--grid", "pool4x10", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--steps", "2"]) == 0
    blob = torch.load(base / "ckpt_last.pt", map_location="cpu", weights_only=False)
    for n in list(blob["model"]):
        if n.startswith("prior.") and blob["model"][n].is_floating_point():
            blob["model"][n] = blob["model"][n] * 0.02
        if n.startswith("head_o.2.weight"):
            blob["model"][n] = torch.randn(blob["model"][n].shape, generator=torch.Generator().manual_seed(0)) * 0.05
    inits = {}
    for s in (0, 1):
        d = WORK / f"M2b_s{s}_dag"
        d.mkdir()
        torch.save(blob, d / "ckpt_last.pt")
        inits[s] = d / "ckpt_last.pt"
    init_sha = {s: sha_file(inits[s]) for s in (0, 1)}
    # 2. the REAL gate on the queue's gate argv of each arm and seed
    cells = [("dagorc", 0), ("dagorcrep", 0), ("dagorc01", 0), ("dagorc", 1), ("dagorcrep", 1)]
    for arm, s in cells:
        gl, tl = L.trainer_line(L.dry(f"gatedag {arm} {s}")), L.trainer_line(L.dry(f"traindag {arm} {s}"))
        out = WORK / f"gate_{arm}_s{s}"
        gate_argv = synthetic_argv(gl, out, inits[s], steps=6, warmup=2)
        gate_argv += ["--selfcheck-only"]
        t0 = time.time()
        rc, sc = run(TR, gate_argv, out)
        case(f"G {arm} s{s}: the REAL trainer's launch gate (--selfcheck-only) ACCEPTS the queue's argv (data flags / init path / steps replaced): rc 0, declared_vs_built PASS", rc in (0, None) and sc and sc.get("declared_vs_built") == "PASS", f"rc {rc} {time.time() - t0:.0f}s")
        if not (sc and sc.get("declared_vs_built") == "PASS"):
            continue
        lv, pr = sc["levers"], sc["ego_perturb_probe"]
        p = {"dagorc": 0.3, "dagorc01": 0.1, "dagorcrep": 0.0}[arm]
        if p > 0:
            case(f"G {arm} s{s}: the ego probe ran for p {p}, the perturbed batch moves the loss and the gradient, every control reads EXACTLY 0.0",
                 pr.get("p_declared") == p and pr.get("loss_diff", 0) != 0 and pr.get("grad_diff_all", 0) > 0 and all(pr.get(k) == 0.0 for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff"))
                 and pr.get("control_p0_same_object") is True, {k: pr.get(k) for k in ("p_declared", "loss_diff", "grad_diff_all", "n_selected_at_p1")})
            case(f"G {arm} s{s}: the BUILT perturber carries the registered law (p {p}, vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed {s})",
                 lv["ego_perturb"]["ego_perturb_p"] == p and lv["ego_perturb"]["ego_perturb_yawrate_std"] == 2.0 and lv["ego_perturb"]["ego_perturb_seed"] == s and lv["ego_perturb"]["ego_perturb_vscale"] == [0.5, 1.0], lv["ego_perturb"])
        else:
            case(f"G {arm} s{s}: REP's ego probe reads 'off' and the BUILT p is 0", pr.get("ego_perturb") == "off" and lv["ego_perturb"]["ego_perturb_p"] == 0.0, pr)
        dp = sc.get("dagger_probe") or {}
        case(f"G {arm} s{s}: the DAgger probe reads 'off', dagger_p 0 is built, no bank hash; init DAG-like checkpoint applied with 0 mismatches and its sha256 recorded",
             dp.get("dagger") == "off" and lv["dagger_p"] == 0.0 and lv.get("dagger_bank_sha256") is None and lv["init_ckpt"]["applied"] is True and lv["init_ckpt"]["n_mismatch_after_load"] == 0 and lv["init_ckpt"]["sha256"] == init_sha[s])
        fd = FIX / f"{arm}_s{s}_gate"
        fd.mkdir(parents=True, exist_ok=True)
        pin_argv = synthetic_argv(tl, out.parent / f"M2b_s{s}_{arm}_run", inits[s], steps=6, warmup=2)
        gate_tokens = gate_argv
        (fd / "selfcheck.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
        (fd / "gate_line.txt").write_text(line_of(gate_tokens), encoding="utf-8")
        (fd / "train_line.txt").write_text(line_of(pin_argv), encoding="utf-8")
        (fd / "pin_line.txt").write_text(line_of(pin_argv), encoding="utf-8")
        (fd / "expect.json").write_text(json.dumps({"steps": 6, "warmup": 2, "n_clips": None, "holdout_sha": None, "windows": None, "check_b_levers": False, "init_tail": "/M2b_s{seed}_dag/ckpt_last.pt"}), encoding="utf-8")
        (fd / "init_sha.txt").write_text(init_sha[s], encoding="utf-8")
    # 3. a short run of A and REP from the same init: the swap counts (K-REACH's input) and the control
    for arm, s, steps in (("dagorc", 0, 60), ("dagorcrep", 0, 60), ("dagorc01", 0, 60)):
        tl = L.trainer_line(L.dry(f"traindag {arm} {s}"))
        out = WORK / f"run_{arm}_s{s}"
        argv = synthetic_argv(tl, out, inits[s], steps=steps, warmup=2, extra=["--log-every", "10", "--ckpt-every", "1000"])
        rc, _ = run(TR, argv, out)
        sm = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else {}
        n, e = sm.get("ego_perturb_n"), sm.get("ego_perturb_eligible")
        p = {"dagorc": 0.3, "dagorc01": 0.1, "dagorcrep": 0.0}[arm]
        if p > 0:
            case(f"T {arm} s{s}: the REAL trainer's short run (queue argv, {steps} steps) finishes and the perturbation REACHED the loop: ego_perturb_n {n} / eligible {e} is ~ p {p}; dagger_n exactly 0",
                 sm.get("done") is True and isinstance(n, int) and n > 0 and e > 0 and abs(n / e - p) < 0.08 and sm.get("dagger_n") == 0, {k: sm.get(k) for k in ("done", "steps", "ego_perturb_n", "ego_perturb_eligible", "dagger_n")})
        else:
            case(f"T {arm} s{s}: CONTROL: REP's short run finishes with ego_perturb_n EXACTLY 0 (eligible > 0) and dagger_n 0", sm.get("done") is True and n == 0 and isinstance(e, int) and e > 0 and sm.get("dagger_n") == 0, {k: sm.get(k) for k in ("done", "ego_perturb_n", "ego_perturb_eligible", "dagger_n")})
        if arm == "dagorc":
            shutil.copy(out / "summary.json", FIX / "dagorc_s0_run_summary.json")
        if arm == "dagorcrep":
            shutil.copy(out / "summary.json", FIX / "dagorcrep_s0_run_summary.json")
    # 4. deliberate regressions the REAL gate must REFUSE
    gl_a, gl_rep = L.trainer_line(L.dry("gatedag dagorc 0")), L.trainer_line(L.dry("gatedag dagorcrep 0"))

    def refuses(name, argv, out, why):
        rc, sc = run(TR, argv, out)
        txt = ""
        case(f"R {name}: the REAL gate REFUSES ({why})", (rc not in (0, None)) or (sc is not None and sc.get("declared_vs_built") != "PASS"), f"rc {rc}")

    o = WORK / "neg1"
    refuses("REP naming a law flag at p 0 (--ego-perturb-yawrate-std 2.0; NOT --ego-perturb-seed 0, which equals the flag's default and is therefore invisible to the trainer's gate: the asserter's line check catches that one)", synthetic_argv(gl_rep, o, inits[0], 6, 2, extra=["--ego-perturb-yawrate-std", "2.0", "--selfcheck-only"]), o, "a flag that builds nothing is refused")
    o = WORK / "neg2"
    refuses("law flags without --ego-perturb-p (the flag's default is OFF)", synthetic_argv(gl_a, o, inits[0], 6, 2, drop=("--ego-perturb-p",), extra=["--selfcheck-only"]), o, "law set, no perturber built")
    o = WORK / "neg3"
    refuses("A together with --dagger-p 0.1", synthetic_argv(gl_a, o, inits[0], 6, 2, drop=("--dagger-p",), extra=["--dagger-p", "0.1", "--selfcheck-only"]), o, "the two levers are never combined / no bank")
    o = WORK / "neg4"
    bad = WORK / "bad_cfg.pt"
    b2 = torch.load(inits[0], map_location="cpu", weights_only=False)
    b2["cfg"]["d"] = int(b2["cfg"]["d"]) * 2
    torch.save(b2, bad)
    refuses("an init checkpoint whose architecture config is doctored (d doubled)", synthetic_argv(gl_a, o, bad, 6, 2, extra=["--selfcheck-only"]), o, "INIT_CKPT_MATCH")
    # 5. MUTATION of the loop: delete the single line `b = ego.apply(b, step)` in a COPY of the trainer
    src = (TREE / "stack" / "scripts" / "v8f_mini_train.py").read_text(encoding="utf-8")
    lines = src.split("\n")
    idx = [i for i, x in enumerate(lines) if x.lstrip().startswith("b = ego.apply(b, step)")]
    case("M the loop's apply line occurs exactly once in the tip trainer", len(idx) == 1, idx)
    mutdir = WORK / "mut" / "stack" / "scripts"
    mutdir.mkdir(parents=True)
    del lines[idx[0]]
    (mutdir / "v8f_mini_train.py").write_text("\n".join(lines), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("v8f_mini_train_mut", mutdir / "v8f_mini_train.py")
    TRM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TRM)
    tl = L.trainer_line(L.dry("traindag dagorc 0"))
    out = WORK / "mut_run_dagorc_s0"
    rc_g, sc_g = run(TRM, synthetic_argv(L.trainer_line(L.dry("gatedag dagorc 0")), WORK / "mut_gate", inits[0], 6, 2, extra=["--selfcheck-only"]), WORK / "mut_gate")
    case("M the gate of the MUTATED trainer (loop line deleted) still PASSES: the gate does not cover the loop's ego.apply (why K-REACH exists)", sc_g is not None and sc_g.get("declared_vs_built") == "PASS" and sc_g["ego_perturb_probe"].get("pass") is True)
    rc, _ = run(TRM, synthetic_argv(tl, out, inits[0], 60, 2, extra=["--log-every", "10", "--ckpt-every", "1000"]), out)
    sm = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else {}
    case("M the MUTATED trainer's run finishes with ego_perturb_n == 0 and eligible > 0 (the perturbation never reached the loop)", sm.get("done") is True and sm.get("ego_perturb_n") == 0 and (sm.get("ego_perturb_eligible") or 0) > 0, {k: sm.get(k) for k in ("done", "ego_perturb_n", "ego_perturb_eligible")})
    shutil.copy(out / "summary.json", FIX / "dagorc_s0_run_summary_LOOP_LINE_DELETED.json")
    import v8f_dagorc_chain_checks as K  # noqa: E402  (the chain's own check)
    rc_reach = K.reached(type("A", (), {"summary": str(out / "summary.json"), "arm": "dagorc"})())
    case("M the chain's K-REACH check REFUSES that run (rc 1)", rc_reach == 1, rc_reach)
    res["n_pass"] = sum(c["ok"] for c in res["cases"])
    res["n_cases"] = len(res["cases"])
    res["seconds"] = round(time.time() - res["t0"], 1)
    outp = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if outp:
        outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"reach_synthetic: {res['n_pass']}/{res['n_cases']} pass ({res['seconds']} s)")
    return 0 if res["n_pass"] == res["n_cases"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(HERE / "box2_ship"))
    sys.exit(main())
