#!/usr/bin/env python3
"""REACHABILITY, part 2 (CLAUDE.md "built, tested, and unreachable from its caller"): the WHOLE iteration-1 -> iteration-2 chain through the tip's REAL code, CPU, synthetic world.

  M2b-like init (the trainer wrote it)  --queue tool + real v8f_dagger_bank.py-->  bank 1 (generator = M2b)
  --real trainer, the argv the queue prints for `traindag dag`-->  DAG (ckpt_last.pt + argv.json that carries --init-ckpt and --dagger-*)
  --queue tool + real v8f_dagger_bank.py, --ckpt DAG --m2b-argv DAG's argv.json (the NEW queue's bankdag2 arguments)-->  bank 2 (generator = DAG)
  --real trainer's launch gate, the argv the NEW queue prints for `gatedag dag2 0 0.1` and `gatedag dag20 0`-->  must ACCEPT
  --real trainer, a short run of DAG2 (the queue's tail), then summary.json: the swap REACHED the loop (dagger_n > 0)-->

Only PATH values (--init-ckpt, --dagger-bank, --dagger-bank-sha256), the step count / warmup (so it finishes on a CPU) and the data flags (--synthetic) are replaced; the rest of the token string is what the queue prints.
Deliberate regressions the REAL gate must REFUSE: DAG2 whose --init-ckpt is the iteration-1 generator (M2b), DAG2 on the iteration-1 bank, a bank-2 trained from another checkpoint, DAG2-0 naming a bank, DAG2-0 naming a seed.

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python reach_synthetic_chain.py [out.json]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dag2q_lib as L  # noqa: E402

SCR = L.BT.parent
TREE = Path(os.environ.get("REACH_TREE", SCR / "tree"))
WORK = SCR / "reach_syn_work"
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
SYN_T = 90
orig_world = D.make_synthetic_world
D.make_synthetic_world = lambda root, n_train=6, n_eval=3, T=70, seed=0, **kw: orig_world(root, n_train=n_train, n_eval=n_eval, T=SYN_T, seed=seed, **kw)
COMMON = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--threads", "1", "--kin-mode", "yaw", "--seed", "0", "--arm", "M2b", "--grid", "pool4x10"]
res = {"cases": [], "t0": time.time()}


def case(name, ok, detail=""):
    res["cases"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail else ""), flush=True)


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_trainer(out, tail, extra=()):
    argv = ["--out", str(out), *COMMON, *tail, *extra]
    Path(out).mkdir(parents=True, exist_ok=True)
    try:
        rc = TR.main(argv)
    except SystemExit as e:
        rc = e.code
    sc = Path(out) / "selfcheck.json"
    return rc, (json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else None)


def tail_of(line_args: str, sha=None):
    """The queue-printed line -> the tokens from --init-ckpt on, minus --selfcheck-only."""
    a = L.train_argv(L.trainer_line(L.dry(line_args, sha=sha or L.SHA_FAKE)))
    t = a[a.index("--init-ckpt"):]
    return [x for x in t if x != "--selfcheck-only"]


def sub(tail, **kv):
    t = list(tail)
    for f, v in kv.items():
        t[t.index(f) + 1] = v
    return t


def make_bank(name, ckpt, argv_json, world, n_shards=2):
    tool = TREE / "stack" / "scripts" / "v8f_dagger_queue_tool.py"
    bank = WORK / name
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, str(tool), "bank", "--n-shards", str(n_shards), "--out-dir", str(bank), "--poll-s", "1", "--", "--ckpt", str(ckpt), "--m2b-argv", str(argv_json), "--cache-dir", str(world),
           "--tracks", str(world / "native_tracks.pt"), "--join-train", str(world / "join_train.npz"), "--eval-grid", str(world / "eval_grid_windows.json"), "--batch", "8", "--threads", "1", "--device", "cpu", "--replay-frac", "0.1"]
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(TREE))
    (WORK / f"{name}_tool.log").write_text(p.stdout + "\n--stderr--\n" + p.stderr, encoding="utf-8")
    return p, bank, time.time() - t


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    # 1. an M2b-like checkpoint written by the TRAINER (tamed, as the trainer suite's fixture does)
    init = WORK / "m2b"
    assert TR.main(["--out", str(init), *COMMON, "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--steps", "2"]) == 0
    world = init / "_synthetic"
    blob = torch.load(init / "ckpt_last.pt", map_location="cpu", weights_only=False)
    for n in list(blob["model"]):
        if n.startswith("prior.") and blob["model"][n].is_floating_point():
            blob["model"][n] = blob["model"][n] * 0.02
        if n.startswith("head_o.2.weight"):
            blob["model"][n] = torch.randn(blob["model"][n].shape, generator=torch.Generator().manual_seed(0)) * 0.05
    m2b = WORK / "m2b_tame.pt"
    torch.save(blob, m2b)
    m2b_argv = WORK / "m2b_argv.json"
    json.dump({"argv": ["--arm", "M2b", "--seed", "0", "--out", "x", "--kin-mode", "yaw", "--batch", "8", "--d", "32", "--n-q", "2"]}, open(m2b_argv, "w"))
    # 2. iteration-1 bank (generator = M2b), through the queue's own tool
    p, bank1, dt = make_bank("bank1", m2b, m2b_argv, world)
    case("iteration 1: bank 1 from the M2b-like checkpoint (tool, 2 shards, real generator + merge), rc 0", p.returncode == 0, f"rc {p.returncode} {dt:.0f}s {p.stdout[-200:]}")
    if p.returncode:
        return finish()
    sha1 = sha_file(bank1 / "bank.json")
    # 3. DAG: the REAL trainer on the argv the queue prints for `traindag dag 0 0.1` (paths + steps replaced), writing ckpt_last.pt + argv.json
    t_dag = sub(tail_of("traindag dag 0 0.1"), **{"--init-ckpt": str(m2b), "--dagger-bank": str(bank1), "--dagger-bank-sha256": sha1, "--steps": "6", "--warmup": "2"})
    dag_out = WORK / "dag"
    rc, _ = run_trainer(dag_out, t_dag, ["--log-every", "1", "--ckpt-every", "100"])
    dag_sum = json.loads((dag_out / "summary.json").read_text(encoding="utf-8")) if (dag_out / "summary.json").exists() else {}
    case("DAG: the real trainer fine-tunes from the M2b-like checkpoint on bank 1 (queue argv), rc 0, done, dagger swap reached the loop", rc == 0 and dag_sum.get("done") is True and (dag_sum.get("dagger_n") or 0) > 0,
         {k: dag_sum.get(k) for k in ("done", "steps", "dagger_n", "dagger_eligible")})
    dag_ck, dag_av = dag_out / "ckpt_last.pt", dag_out / "argv.json"
    av = json.loads(dag_av.read_text(encoding="utf-8"))["argv"]
    case("DAG's argv.json carries --init-ckpt and the three --dagger-* flags (it is what bankdag2 hands to --m2b-argv)", all(k in av for k in ("--init-ckpt", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")), [x for x in av if x.startswith("--dagger") or x == "--init-ckpt"])
    dag_sha = sha_file(dag_ck)
    # 4. iteration-2 bank: generator = DAG, --m2b-argv = DAG's argv.json (the arguments bankdag2 prints). The ONLY synthetic-world artifact in that file is the `--synthetic` token (the generator REFUSES a synthetic argv by design:
    #    the real DAG argv.json has none -- reach_real_ckpt.py R3); the generator is handed the same file without that one token, and the cache / tracks / join come from the --cache-dir overrides, as in iteration 1.
    gen_av = WORK / "dag_argv_for_generator.json"
    stripped = [x for x in av if x != "--synthetic"]
    json.dump({"argv": stripped}, open(gen_av, "w"))
    case("the generator's copy of DAG's argv differs from the trainer-written argv.json by the `--synthetic` token ONLY", [x for x in av if x not in stripped] == ["--synthetic"] and len(stripped) == len(av) - 1)
    p2, bank2, dt = make_bank("bank2", dag_ck, gen_av, world)
    case("iteration 2: bank 2 from the DAG checkpoint with DAG's argv.json as --m2b-argv (tool, 2 shards, real generator + merge), rc 0", p2.returncode == 0, f"rc {p2.returncode} {dt:.0f}s {p2.stdout[-300:]}")
    if p2.returncode:
        return finish()
    man2, prov2 = json.loads((bank2 / "bank.json").read_text(encoding="utf-8")), json.loads((bank2 / "PROVENANCE.json").read_text(encoding="utf-8"))
    sha2 = sha_file(bank2 / "bank.json")
    man1 = json.loads((bank1 / "bank.json").read_text(encoding="utf-8"))
    case("bank 2: the header's generating checkpoint IS the DAG checkpoint (not M2b), the window list and TRAIN list are bank 1's, not partial", man2["common"]["ckpt_sha256"] == dag_sha != man1["common"]["ckpt_sha256"]
         and man2["common"]["window_list_sha256"] == man1["common"]["window_list_sha256"] and man2["common"]["train"] == man1["common"]["train"] and not man2["partial"], {"dag": dag_sha[:12], "bank2": man2["common"]["ckpt_sha256"][:12]})
    case("bank 2: K-B4 and K-B5 PASS in PROVENANCE.json (the controls the chain's judge gates)", (prov2.get("controls") or {}).get("K-B4", {}).get("PASS") is True and (prov2.get("controls") or {}).get("K-B5", {}).get("PASS") is True, list((prov2.get("controls") or {}).keys()))
    case("bank 2: coverage clears 0.5", not man2["coverage"]["blocked"], man2["coverage"])
    shutil.copy(bank2 / "bank.json", WORK / "bank2.bank.json")
    # 5. the REAL gate on the NEW queue's lines, init = DAG, bank = bank 2
    t2 = sub(tail_of("gatedag dag2 0 0.1"), **{"--init-ckpt": str(dag_ck), "--dagger-bank": str(bank2), "--dagger-bank-sha256": sha2})
    t20 = sub(tail_of("gatedag dag20 0"), **{"--init-ckpt": str(dag_ck)})
    (WORK / "tail_dag2.json").write_text(json.dumps(t2), encoding="utf-8")
    (WORK / "tail_dag20.json").write_text(json.dumps(t20), encoding="utf-8")
    rc, sc = run_trainer(WORK / "gate_dag2", t2, ["--selfcheck-only"])
    case("REAL gate ACCEPTS the DAG2 argv (init = DAG checkpoint, bank = the DAG-generated bank): rc 0, declared_vs_built PASS", rc == 0 and sc and sc.get("declared_vs_built") == "PASS", f"rc {rc}")
    if sc:
        pr = sc.get("dagger_probe") or {}
        d = (sc["info"].get("dagger") or {})
        case("DAG2: dagger_probe passed on a swapped batch, p = 0 control exactly 0.0, levers carry p 0.1 and the declared == built bank-2 sha256", pr.get("pass") is True and pr.get("control_p0_loss_diff") == 0.0 and pr.get("control_p0_grad_diff") == 0.0
             and sc["levers"].get("dagger_p") == 0.1 and sc["levers"].get("dagger_bank_sha256") == sha2 == sc["levers"].get("dagger_bank_sha256_declared"), {k: pr.get(k) for k in ("pass", "p_declared", "n_selected_at_p1", "loss_diff")})
        case("DAG2: K-B2 reads the bank's checkpoint == --init-ckpt == the DAG checkpoint's sha256; levers.init_ckpt.sha256 is the DAG checkpoint's, applied, 0 tensors differ",
             (d.get("kb2") or {}).get("ckpt_sha256") == (d.get("kb2") or {}).get("init_ckpt_sha256") == dag_sha == sc["levers"]["init_ckpt"]["sha256"] and sc["levers"]["init_ckpt"]["applied"] and sc["levers"]["init_ckpt"]["n_mismatch_after_load"] == 0,
             {"kb2": d.get("kb2"), "init": {k: sc["levers"]["init_ckpt"].get(k) for k in ("sha256", "applied", "n_mismatch_after_load", "source_arm")}})
        case("DAG2: info.dagger records K-B3 (overlap 0) and COVERAGE (not blocked)", (d.get("kb3") or {}).get("overlap_with_eval") == 0 and (d.get("coverage") or {}).get("blocked") is False, {k: d.get(k) for k in ("coverage",)})
    rc0, sc0 = run_trainer(WORK / "gate_dag20", t20, ["--selfcheck-only"])
    case("REAL gate ACCEPTS the DAG2-0 argv (init = DAG checkpoint, --dagger-p 0 only)", rc0 == 0 and sc0 and sc0.get("declared_vs_built") == "PASS", f"rc {rc0}")
    if sc0:
        pr0 = sc0.get("dagger_probe") or {}
        case("DAG2-0: dagger_probe reads `off`; levers.dagger_p is 0.0, no bank sha256 built; init is the DAG checkpoint", pr0.get("dagger") == "off" and sc0["levers"].get("dagger_p") == 0.0 and sc0["levers"].get("dagger_bank_sha256") is None
             and sc0["levers"]["init_ckpt"]["sha256"] == dag_sha, {k: sc0["levers"].get(k) for k in ("dagger_p", "dagger_bank_sha256")})
    # 6. a short REAL training run of DAG2 on that argv: the swap must reach the loop
    rc, _ = run_trainer(WORK / "run_dag2", sub(t2, **{"--steps": "6", "--warmup": "2"}), ["--log-every", "1", "--ckpt-every", "100"])
    sm = json.loads((WORK / "run_dag2" / "summary.json").read_text(encoding="utf-8")) if (WORK / "run_dag2" / "summary.json").exists() else {}
    case("DAG2 trains (real trainer, 6 steps from the DAG checkpoint on bank 2): rc 0, done, dagger_n > 0 of dagger_eligible, init sha256 = DAG's", rc == 0 and sm.get("done") is True and (sm.get("dagger_n") or 0) > 0 and ((sm.get("levers") or {}).get("init_ckpt") or {}).get("sha256") == dag_sha,
         {k: sm.get(k) for k in ("done", "steps", "dagger_n", "dagger_eligible")})
    rc, _ = run_trainer(WORK / "run_dag20", sub(t20, **{"--steps": "6", "--warmup": "2"}), ["--log-every", "1", "--ckpt-every", "100"])
    sm0 = json.loads((WORK / "run_dag20" / "summary.json").read_text(encoding="utf-8")) if (WORK / "run_dag20" / "summary.json").exists() else {}
    case("DAG2-0 trains (6 steps from the DAG checkpoint, --dagger-p 0): rc 0, done, dagger_n exactly 0", rc == 0 and sm0.get("done") is True and sm0.get("dagger_n") == 0, {k: sm0.get(k) for k in ("done", "steps", "dagger_n", "dagger_eligible")})
    # 7. deliberate regressions the REAL gate must refuse
    other = WORK / "dag_other.pt"                                  # a DIFFERENT checkpoint of the same architecture (one bit moved): "another DAG"
    ob = torch.load(dag_ck, map_location="cpu", weights_only=False)
    k0 = next(n for n, v in ob["model"].items() if v.is_floating_point())
    ob["model"][k0] = ob["model"][k0] + 1e-3
    torch.save(ob, other)
    neg = {
        "DAG2 whose --init-ckpt is the ITERATION-1 generator (M2b), bank 2 (K-B2)": sub(t2, **{"--init-ckpt": str(m2b)}),
        "DAG2 on the ITERATION-1 bank (M2b-generated) with init = DAG (K-B2)": sub(t2, **{"--dagger-bank": str(bank1), "--dagger-bank-sha256": sha1}),
        "DAG2 whose --init-ckpt is another checkpoint than bank 2's generator (K-B2)": sub(t2, **{"--init-ckpt": str(other)}),
        "DAG2 with a wrong bank sha256 (one hex digit)": sub(t2, **{"--dagger-bank-sha256": ("0" if sha2[0] != "0" else "1") + sha2[1:]}),
        "DAG2-0 that also names the bank": t20 + ["--dagger-bank", str(bank2)],
        "DAG2-0 that also names a bank hash": t20 + ["--dagger-bank-sha256", sha2],
        "DAG2-0 that also names a non-default seed": t20 + ["--dagger-seed", "1"],
        "DAG2 together with --ego-perturb-p 0.3": sub(t2, **{"--ego-perturb-p": "0.3"}),
    }
    for name, tail in neg.items():
        rc, sc = run_trainer(WORK / ("neg_" + hashlib.sha256(name.encode()).hexdigest()[:8]), tail, ["--selfcheck-only"])
        refused = (isinstance(rc, str) and "ZZDVB-REFUSEDZZ" in rc) or (sc is not None and sc.get("declared_vs_built") not in (None, "PASS")) or (rc not in (0, None) and not sc)
        case("REFUSED by the real gate: " + name, refused and rc != 0, str(rc)[:260])
    return finish()


def finish():
    res["seconds"] = time.time() - res["t0"]
    res["n_pass"] = sum(c["ok"] for c in res["cases"])
    res["n_cases"] = len(res["cases"])
    res["all_pass"] = res["n_pass"] == res["n_cases"] and res["n_cases"] > 0
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reach_synthetic_chain.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n in ("bank2", "bank1"):                                   # the small evidence files of the banks stay (not the npz)
        pass
    print(f"reach_synthetic_chain: {res['n_pass']}/{res['n_cases']} pass in {res['seconds']:.0f}s", flush=True)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
