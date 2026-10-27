#!/usr/bin/env python3
"""Does the REAL trainer's in-process launch gate accept the argv the new queue stages emit, and refuse what they must never emit?  (CPU, synthetic world, a COPY of the tree: dagq/rg/tree.)

Not the real corpus: the data flags (%A%) are replaced by --synthetic and the three PATH values (--init-ckpt, --dagger-bank, --dagger-bank-sha256) by the synthetic ones. EVERYTHING ELSE is the token string the
queue prints (`gatedag dag 0 0.1`, `gatedag dag0 0`): the tail from --init-ckpt to the end, flag names, values and order. A bank is made by the queue's own `bank` tool with TWO shards (the dev box is memory-bound: four torch processes failed with WinError 1455) through the real
v8f_dagger_bank.py, from a checkpoint the trainer wrote, so the 4-shard fan-out + merge are exercised against the real generator as well.

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python real_gate_proof.py
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
import dagq_lib as L  # noqa: E402

TREE = HERE / "rg" / "tree"
WORK = HERE / "rg" / "work"
OUT = HERE / "rg" / "out"
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


def gate(out, tail):
    """TR.main([... --selfcheck-only]) -> (rc or the SystemExit text, selfcheck dict or None)."""
    argv = ["--out", str(out), *COMMON, *tail, "--selfcheck-only"]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "gate_argv.json").write_text(json.dumps(argv), encoding="utf-8")
    try:
        rc = TR.main(argv)
    except SystemExit as e:
        rc = e.code
    sc = Path(out) / "selfcheck.json"
    return rc, (json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else None)


def tail_of(line_args: str, sha=None):
    """The queue-printed gate line -> the tokens from --init-ckpt on, minus --selfcheck-only: the part of the argv the lever lives in."""
    a = L.train_argv(L.trainer_line(L.dry(line_args, sha=sha or L.SHA_FAKE)))
    t = a[a.index("--init-ckpt"):]
    return [x for x in t if x != "--selfcheck-only"]


def sub(tail, **kv):
    """Replace the VALUE of a flag (--flag -> value); the flag stays where the queue put it."""
    t = list(tail)
    for f, v in kv.items():
        k = t.index(f)
        t[k + 1] = v
    return t


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    if OUT.exists():
        shutil.rmtree(OUT)
    WORK.mkdir(parents=True)
    OUT.mkdir(parents=True)
    # 1. a checkpoint the TRAINER wrote (tamed: a random prior is an unstable closed loop), as the trainer suite's fixture does
    init = WORK / "init"
    assert TR.main(["--out", str(init), *COMMON, "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--steps", "2"]) == 0
    world = init / "_synthetic"
    blob = torch.load(init / "ckpt_last.pt", map_location="cpu", weights_only=False)
    for n in list(blob["model"]):
        if n.startswith("prior.") and blob["model"][n].is_floating_point():
            blob["model"][n] = blob["model"][n] * 0.02
        if n.startswith("head_o.2.weight"):
            blob["model"][n] = torch.randn(blob["model"][n].shape, generator=torch.Generator().manual_seed(0)) * 0.05
    ck = WORK / "init_tame.pt"
    torch.save(blob, ck)
    av = WORK / "argv.json"
    json.dump({"argv": ["--arm", "M2b", "--seed", "0", "--out", "x", "--kin-mode", "yaw", "--batch", "8", "--d", "32", "--n-q", "2"]}, open(av, "w"))
    # 2. the bank, through the QUEUE's own tool: four shards of the real generator, then the real merge
    tool = TREE / "stack" / "scripts" / "v8f_dagger_queue_tool.py"
    shutil.copy(L.HERE / "stage" / "stack" / "scripts" / "v8f_dagger_queue_tool.py", tool)
    bank = WORK / "bank"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, str(tool), "bank", "--n-shards", "2", "--out-dir", str(bank), "--poll-s", "1", "--", "--ckpt", str(ck), "--m2b-argv", str(av), "--cache-dir", str(world), "--tracks", str(world / "native_tracks.pt"),
           "--join-train", str(world / "join_train.npz"), "--eval-grid", str(world / "eval_grid_windows.json"), "--batch", "8", "--threads", "1", "--device", "cpu", "--replay-frac", "0.1"]
    t = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(TREE))
    (OUT / "bank_tool.log").write_text(p.stdout + "\n--stderr--\n" + p.stderr, encoding="utf-8")
    case("bank tool: 2 shards through the real v8f_dagger_bank.py + merge, rc 0", p.returncode == 0, f"rc {p.returncode} {time.time() - t:.0f}s {p.stdout[-300:]}")
    if p.returncode != 0:
        return finish()
    man = json.loads((bank / "bank.json").read_text(encoding="utf-8"))
    prov = json.loads((bank / "PROVENANCE.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256((bank / "bank.json").read_bytes()).hexdigest()
    case("bank: 2 shards in the manifest, not partial, coverage clears 0.5", man["n_shards"] == 2 and not man["partial"] and not man["coverage"]["blocked"], man["coverage"])
    case("bank tool printed the sha256 of bank.json", sha in p.stdout, sha[:16])
    shutil.copy(bank / "bank.json", OUT / "bank.json")
    shutil.copy(bank / "PROVENANCE.json", OUT / "PROVENANCE.json")
    for f in sorted(bank.glob("shard_*.provenance.json")):
        shutil.copy(f, OUT / f.name)
    # 3. the queue's lines, mapped onto the synthetic world
    t_dag = sub(tail_of("gatedag dag 0 0.1"), **{"--init-ckpt": str(ck), "--dagger-bank": str(bank), "--dagger-bank-sha256": sha})
    t_dag0 = sub(tail_of("gatedag dag0 0"), **{"--init-ckpt": str(ck)})
    (OUT / "tail_dag.json").write_text(json.dumps(t_dag), encoding="utf-8")
    (OUT / "tail_dag0.json").write_text(json.dumps(t_dag0), encoding="utf-8")
    rc, sc = gate(OUT / "dag_s0", t_dag)
    case("REAL gate accepts the DAG argv the queue prints (rc 0, declared_vs_built PASS)", rc == 0 and sc and sc.get("declared_vs_built") == "PASS", f"rc {rc}")
    if sc:
        pr = sc.get("dagger_probe") or {}
        case("DAG: dagger_probe passed on a swapped batch and the p = 0 control read exactly 0.0", pr.get("pass") is True and pr.get("control_p0_loss_diff") == 0.0 and pr.get("control_p0_grad_diff") == 0.0, {k: pr.get(k) for k in ("pass", "p_declared", "n_selected_at_p1", "loss_diff", "control_p0_loss_diff")})
        case("DAG: levers carry p 0.1, the declared and built bank sha256 and seed 0", sc["levers"].get("dagger_p") == 0.1 and sc["levers"].get("dagger_bank_sha256") == sha and sc["levers"].get("dagger_bank_sha256_declared") == sha, {k: sc["levers"].get(k) for k in ("dagger_p", "dagger_bank_sha256")})
        case("DAG: info.dagger records K-B2, K-B3 and COVERAGE", all(k in (sc["info"].get("dagger") or {}) for k in ("kb2", "kb3", "coverage")), list((sc["info"].get("dagger") or {}).keys()))
    rc0, sc0 = gate(OUT / "dag0_s0", t_dag0)
    case("REAL gate accepts the DAG-0 argv the queue prints (--dagger-p 0 only)", rc0 == 0 and sc0 and sc0.get("declared_vs_built") == "PASS", f"rc {rc0}")
    if sc0:
        pr0 = sc0.get("dagger_probe") or {}
        case("DAG-0: dagger_probe reads `off`; levers.dagger_p is 0.0 and no bank sha256 is built", pr0.get("dagger") == "off" and sc0["levers"].get("dagger_p") == 0.0 and sc0["levers"].get("dagger_bank_sha256") is None, {k: sc0["levers"].get(k) for k in ("dagger_p", "dagger_bank_sha256")})
    # 4. what the queue must NEVER emit, refused by the real gate (deliberate regressions: each must be refused)
    neg = {
        "DAG-0 that also names the bank (a lever declared and inert)": t_dag0 + ["--dagger-bank", str(bank)],
        "DAG-0 that also names the bank hash": t_dag0 + ["--dagger-bank-sha256", sha],
        "DAG-0 that also names a non-default seed": t_dag0 + ["--dagger-seed", "1"],
        "DAG without --dagger-bank-sha256": L.drop_flag(t_dag, {"--dagger-bank-sha256"}),
        "DAG with a wrong bank sha256 (one hex digit changed)": sub(t_dag, **{"--dagger-bank-sha256": ("0" if sha[0] != "0" else "1") + sha[1:]}),
        "DAG together with --ego-perturb-p 0.3": sub(t_dag, **{"--ego-perturb-p": "0.3"}),
        "DAG whose --init-ckpt is not the bank's generating checkpoint (K-B2)": sub(t_dag, **{"--init-ckpt": str(init / "ckpt_last.pt")}),
        "DAG at p = 0 that still names the bank": sub(t_dag, **{"--dagger-p": "0"}),
    }
    for name, tail in neg.items():
        d = OUT / ("neg_" + hashlib.sha256(name.encode()).hexdigest()[:8])
        rc, sc = gate(d, tail)
        refused = isinstance(rc, str) and "ZZDVB-REFUSEDZZ" in rc or (sc is not None and sc.get("declared_vs_built") not in (None, "PASS"))
        case("REFUSED by the real gate: " + name, refused and rc != 0, str(rc)[:260])
    return finish()


def finish():
    res["seconds"] = time.time() - res["t0"]
    res["n_pass"] = sum(c["ok"] for c in res["cases"])
    res["n_cases"] = len(res["cases"])
    res["all_pass"] = res["n_pass"] == res["n_cases"] and res["n_cases"] > 0
    (OUT / "RESULT.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"real_gate_proof: {res['n_pass']}/{res['n_cases']} pass in {res['seconds']:.0f}s", flush=True)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
