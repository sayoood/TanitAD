#!/usr/bin/env python3
"""dagt1_smoke.py -- plumbing test of dag_t1_chain.sh in SMOKE mode (24 windows, ONE stand-in run, outputs under dagt1/smoke).  Nothing here is a DAgger result.

    python dagt1_smoke.py setup     builds smoke/art/M2b_s0_dag: a pulled-run directory in the box-2 chain's layout (8 files + SHA256SUMS.txt), the checkpoint being a COPY of the banked
                                    ORC-0 s0 checkpoint (a stand-in: it is not a DAgger checkpoint), the standard T0 eval a copy of ORC-0's with its ckpt path rewritten, the records synthetic
    python dagt1_smoke.py check     after `DAGT1_SMOKE=1 bash dag_t1_chain.sh` (start it with a real C: >= 2 GB free, or DAGT1_TEST_CFREE_MB in SMOKE): inspects smoke/logs/chain.out, smoke/out

    DAGT1_SMOKE=1 DAGT1_RUNS=dag:0 bash dag_t1_chain.sh          (the SMOKE chain: two lanes, 24 windows each, the fresh T0 eval, then the manifest and the reader; the reader REFUSES by design on a replicated stand-in)
    The expected end: `ZZDAGT1-SMOKE-DONE 2/2 rollouts, reader rc 2 (REFUSED, nothing written)` in smoke/logs/chain.out.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dagt1_manifest as MF  # noqa: E402
import dagt1_pullcheck as PC  # noqa: E402

R = "E:/Projects/TanitAD-artifacts"
ART = os.path.join(HERE, "smoke", "art")
OUT = os.path.join(HERE, "smoke", "out")
LOG = os.path.join(HERE, "smoke", "logs")
SRC = f"{R}/v8f-orecover/M2b_s0_orc0"


def wjson(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f)


def setup():
    d = os.path.join(ART, PC.run_name("dag", 0))
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(f"{SRC}/ckpt_last.pt", os.path.join(d, "ckpt_last.pt"))
    tg = PC.eval_tag("dag", 0)
    shutil.copyfile(f"{SRC}/eval_M2b-orc0_s0.npz", os.path.join(d, f"eval_{tg}.npz"))
    with open(f"{SRC}/eval_M2b-orc0_s0.json", encoding="utf-8") as f:
        j = json.load(f)
    j["ckpt"] = f"D:\\tanitad\\runs\\v8f_mini_full\\{PC.run_name('dag', 0)}\\ckpt_last.pt"
    assert j.get("ages") == "cycle", j.get("ages")
    wjson(os.path.join(d, f"eval_{tg}.json"), j)
    with open(f"{SRC}/summary.json", encoding="utf-8") as f:
        orc = json.load(f)
    init = orc["levers"]["init_ckpt"]["sha256"]
    argv = ["--arm", "M2b", "--seed", "0", "--out", f"D:\\tanitad\\runs\\v8f_mini_full\\{PC.run_name('dag', 0)}", "--init-ckpt", "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0\\ckpt_last.pt",
            "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0.1", "--dagger-bank", "D:\\x\\bank",
            "--dagger-bank-sha256", "b" * 64, "--dagger-seed", "0"]
    lev = {"init_ckpt": {"sha256": init}, "dagger_p": 0.1, "dagger_bank_sha256": "b" * 64}
    wjson(os.path.join(d, "argv.json"), {"argv": argv, "sha": "x"})
    wjson(os.path.join(d, "summary.json"), {"done": True, "steps": 3000, "arm": "M2b", "seed": 0, "levers": lev, "dagger_n": 1003, "dagger_eligible": 10000})
    wjson(os.path.join(d, "selfcheck.json"), {"declared_vs_built": "PASS", "levers": lev})
    wjson(os.path.join(d, "launch_gate_token.json"), {"kind": "launch-gate-token", "verdict": "PASS"})
    open(os.path.join(d, "train_log.jsonl"), "w").write('{"step": 1}\n')
    lines = []
    for n in PC.base_files("dag", 0):
        h = hashlib.sha256()
        with open(os.path.join(d, n), "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {n}")
    p = os.path.join(d, "SHA256SUMS.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    t = time.time() - 60
    os.utime(p, (t, t))
    st, why, sha = PC.check(ART, "dag", 0)
    print(f"stand-in {d}: {st} {sha[:16]} {why}")
    return 0 if st == "COMPLETE" else 1


def check():
    bad = []
    chain = os.path.join(LOG, "chain.out")
    if not os.path.isfile(chain):
        print("no smoke/logs/chain.out: the SMOKE chain has not run")
        return 1
    with open(chain, encoding="utf-8", errors="replace") as f:
        text = f.read()
    last = [ln for ln in text.splitlines() if ln.startswith(("ZZDAGT1-SMOKE-DONE", "ZZDAGT1-STOP"))]
    print("last marker:", last[-1][:300] if last else "NONE")
    if not last or not last[-1].startswith("ZZDAGT1-SMOKE-DONE 2/2 rollouts"):
        bad.append("no ZZDAGT1-SMOKE-DONE 2/2 marker")
    for tag in ("SMOKE_DAG_s0_m10_causal_heads", "SMOKE_DAG_s0_m10_logged_heads"):
        for p in (f"{OUT}/t1v8f_{tag}.json", f"{OUT}/ticks/ticks_{tag}.npz", f"{OUT}/done_{tag}.txt"):
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                bad.append(f"missing {p}")
    for p in (f"{OUT}/t0fresh/eval_SMOKE_M2b-dag_s0-fresh.json", f"{OUT}/done_t0f_SMOKE_M2b-dag_s0-fresh.txt", f"{OUT}/runs.json", f"{OUT}/PINS.txt"):
        if not os.path.isfile(p):
            bad.append(f"missing {p}")
    if os.path.isfile(f"{OUT}/runs.json"):
        with open(f"{OUT}/runs.json", encoding="utf-8") as f:
            m = json.load(f)
        vb = MF.verify(m, 1, smoke=True)
        print("manifest verify (smoke):", vb or "clean")
        bad += vb
    ro = os.path.join(LOG, "dag_read.out")
    if os.path.isfile(ro):
        print("reader output:", open(ro, encoding="utf-8", errors="replace").read()[:1500])
    print("SMOKE CHECK", "OK" if not bad else f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    a = sys.argv[1:2]
    sys.exit(setup() if a == ["setup"] else check() if a == ["check"] else (print(__doc__) or 2))
