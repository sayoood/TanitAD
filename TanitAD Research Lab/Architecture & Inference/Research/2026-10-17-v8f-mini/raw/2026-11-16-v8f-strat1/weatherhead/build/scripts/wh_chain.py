"""wh_chain.py -- the chain of PREREG_V8F_WEATHERHEAD form (a): arms P1..P9 in the card's priority order, each REFUSING without its token (``wh_check.py run`` verifies it; a refusal stops the chain: fail closed).
WRITTEN, NOT LAUNCHED by the builder.  Idempotent: a step whose marker ``<runs>/state/<step>.done`` exists is skipped.  No model turn polls it: the last line of the log is the sentinel.

  python wh_chain.py all --runs R [--device cuda] [--lock-wait-h 12] [--stop-after 2]
  python wh_chain.py instrument --runs R           # assemble the instrument JSON the reader takes
  python wh_chain.py read --runs R                 # the reader on whatever has finished

Per arm: the fit (GPU: takes the shared devbox_gpu.lock itself, waits while the GPU reads >= 3,500 MiB, NEVER forces it) -> T0 eval of its checkpoint (CPU, commit-gated) -> for the H arms the fit-side logits (K-ID).
After H_s0 and H_s1: T-ISO (c) and (d) (S1 re-dump with --ctx-pool off vs S1 + head with --ctx-pool on, one environment, one thread count) and T-ISO (a) (state-dict identity).
Last line: ZZ WH CHAIN DONE ZZ | ZZ WH CHAIN FAIL ZZ <why>."""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402
import wh_check as C  # noqa: E402
import wh_run as RUN  # noqa: E402

PY = RUN.PY


def log(runs, *a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    os.makedirs(f"{runs}/state", exist_ok=True)
    with open(f"{runs}/wh_chain.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def done(runs, step):
    return os.path.exists(f"{runs}/state/{step}.done")


def mark(runs, step):
    os.makedirs(f"{runs}/state", exist_ok=True)
    open(f"{runs}/state/{step}.done", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def sub(runs, argv, step):
    """run a child; stdout / stderr to <runs>/logs/<step>.log -> rc"""
    os.makedirs(f"{runs}/logs", exist_ok=True)
    with open(f"{runs}/logs/{step}.log", "ab") as lf:
        lf.write((" ".join(argv) + "\n").encode("utf-8"))
        return subprocess.run(argv, env=RUN.child_env(), stdout=lf, stderr=subprocess.STDOUT).returncode


def fail(runs, why):
    log(runs, "ZZ WH CHAIN FAIL ZZ", why)
    open(f"{runs}/state/FAILED", "w").write(why + "\n")
    sys.exit(1)


def fit_env(device):
    e = RUN.child_env()
    if device == "cuda":
        e["CUDA_VISIBLE_DEVICES"] = "0"
    return e


def run_fit(runs, arm, device, lock_wait_h):
    t0 = time.time()
    while True:
        with open(f"{runs}/logs/fit_{arm}.log", "ab") as lf:
            lf.write((f"--- {time.strftime('%H:%M:%S')} wh_check run {arm}\n").encode())
            rc = subprocess.run([PY, f"{HERE}/wh_check.py", "run", arm, "--runs", runs, "--device", device], env=fit_env(device), stdout=lf, stderr=subprocess.STDOUT).returncode
        tail = open(f"{runs}/logs/fit_{arm}.log", "rb").read()[-3000:].decode("utf-8", "replace")
        if rc == 0:
            return 0
        if "LOCK-BUSY" in tail and time.time() - t0 < lock_wait_h * 3600:
            log(runs, f"{arm}: GPU lock busy; waiting 300 s (never forced)")
            time.sleep(300)
            continue
        return rc


def ckpt_of(runs, arm):
    return f"{runs}/fit/{arm}/ckpt_wh.pt"


def step_arm(runs, arm, device, lock_wait_h):
    s = C.ARMS[arm]
    if not done(runs, f"fit_{arm}"):
        rc = run_fit(runs, arm, device, lock_wait_h) if s["kind"] == "fit" else sub(runs, [PY, f"{HERE}/wh_check.py", "run", arm, "--runs", runs, "--device", "cpu"], f"fit_{arm}")
        if rc != 0:
            fail(runs, f"{arm}: fit / token step exited rc={rc} (see logs/fit_{arm}.log)")
        mark(runs, f"fit_{arm}")
    if not done(runs, f"t0_{arm}"):
        rc = sub(runs, [PY, f"{HERE}/wh_run.py", "t0", "--ckpt", ckpt_of(runs, arm), "--arm", arm, "--out", f"{runs}/t0", "--wait-s", "7200"], f"t0_{arm}")
        if rc != 0:
            fail(runs, f"{arm}: T0 eval rc={rc}" + (" (commit gate: retry when free commit >= 5.5 GB)" if rc == 75 else ""))
        mark(runs, f"t0_{arm}")
    if arm in ("H_s0", "H_s1") and not done(runs, f"kid_{arm}"):
        rc = sub(runs, [PY, f"{HERE}/wh_fit.py", "kid", "--ckpt", ckpt_of(runs, arm), "--out", f"{runs}/kid/kid_{arm}.npz"], f"kid_{arm}")
        if rc != 0:
            fail(runs, f"{arm}: fit-side logits rc={rc}")
        mark(runs, f"kid_{arm}")


def step_iso(runs, s):
    s1, other = f"{W.CK}/S1_s{s}/ckpt_last.pt", f"{W.CK}/S1_s{1 - s}/ckpt_last.pt"
    head = ckpt_of(runs, f"H_s{s}")
    if not done(runs, f"iso_a_s{s}"):
        import torch
        a = torch.load(s1, map_location="cpu", weights_only=False)["model"]
        b = torch.load(head, map_location="cpu", weights_only=False)["model"]
        bad = W.iso_a(a, b)
        os.makedirs(f"{runs}/iso", exist_ok=True)
        json.dump({"check": "T-ISO (a)", "n_s1_keys": len(a), "n_head_only_keys": len(set(b) - set(a)), "problems": bad[:10], "pass": not bad}, open(f"{runs}/iso/iso_a_s{s}.json", "w"), indent=1)
        mark(runs, f"iso_a_s{s}")
    for w in ("c", "d"):
        if not done(runs, f"iso_{w}_s{s}"):
            rc = sub(runs, [PY, f"{HERE}/wh_iso.py", w, "--s1", s1, "--head", head, "--s1-other", other, "--out", f"{runs}/iso", "--tag", f"s{s}"], f"iso_{w}_s{s}")
            if rc != 0:
                fail(runs, f"T-ISO ({w}) seed {s} did not pass (rc {rc}): the head reaches the planner, or the dump environment differs; nothing ships")
            mark(runs, f"iso_{w}_s{s}")


def instrument(runs):
    """the instrument JSON the reader takes: pass / fail / not-run per check; both seeds must pass"""
    def jl(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def both(files, key=None):
        rs = [jl(f) for f in files]
        if any(r is None for r in rs):
            return "not-run"
        return "pass" if all((r.get(key) if key else r).get("pass") if key else r.get("pass") for r in rs) else "fail"
    ins = {"T-ISO-a": both([f"{runs}/iso/iso_a_s{s}.json" for s in (0, 1)]), "T-ISO-c": both([f"{runs}/iso/iso_c_s{s}.json" for s in (0, 1)]), "T-ISO-d": both([f"{runs}/iso/iso_d_s{s}.json" for s in (0, 1)])}
    ins["T-ISO-b"] = both([f"{runs}/iso/iso_c_s{s}.json" for s in (0, 1)], "T-ISO-b")
    ins["T-VIS"] = both([f"{runs}/iso/iso_c_s{s}.json" for s in (0, 1)], "T-VIS")
    metas = [jl(f"{runs}/fit/{a}/wh_meta.json") for a in ("H_s0", "H_s1")]
    ins["REACH-live"] = "not-run" if any(m is None for m in metas) else ("pass" if all(m["live"]["pass"] for m in metas) else "fail")
    ins["REACH-census"] = "not-run" if any(m is None for m in metas) else ("pass" if all(m["census_pass"] for m in metas) else "fail")
    ins["DISJOINT"] = "not-run" if any(m is None for m in metas) else ("pass" if all(all(v == 0 for v in m["disjointness"].values()) for m in metas) else "fail")
    toks = [jl(C.token_path(runs, a)) for a in ("H_s0", "H_s1")]
    ins["G-EVAL"] = "not-run" if any(t is None for t in toks) else ("pass" if all(t["checks"]["G-EVAL"]["pass"] for t in toks) else "fail")
    d = [jl(f"{runs}/iso/iso_d_s{s}.json") for s in (0, 1)]
    ins["_driving"] = ({"outcome": "NO-COST-BY-IDENTITY" if all(x and x.get("structural_zero") for x in d) else "ISO-FAIL", "structural_zero_cells": "24 driving cells + 6 route-setting cells + S_CKPT read exactly 0.0 [0, 0] (an identity, not an estimate)",
                        "evidence": [x and {"n_arrays_compared": x.get("n_arrays_compared"), "differences": x.get("differences")} for x in d]} if all(x is not None for x in d) else {"outcome": "NOT-YET", "reason": "T-ISO (d) dumps not read"})
    json.dump(ins, open(f"{runs}/instrument.json", "w", encoding="utf-8"), indent=1)
    return ins


def read(runs):
    instrument(runs)
    kid = f"{runs}/kid"
    return sub(runs, [PY, f"{HERE}/wh_read.py", "--t0", f"{runs}/t0", "--out", f"{runs}/read", "--kid", kid, "--instrument", f"{runs}/instrument.json"], "read")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("all", "instrument", "read"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--lock-wait-h", type=float, default=12.0)
    ap.add_argument("--stop-after", type=int, default=9, help="the last priority P to run (1..9)")
    a = ap.parse_args(argv)
    os.makedirs(f"{a.runs}/state", exist_ok=True)
    if a.cmd == "instrument":
        print(json.dumps(instrument(a.runs), indent=1))
        return 0
    if a.cmd == "read":
        return read(a.runs)
    if os.path.exists(f"{a.runs}/state/FAILED"):
        print("a previous run FAILED; read state/FAILED, fix the cause, delete the marker")
        return 1
    arms = [k for k, v in sorted(C.ARMS.items(), key=lambda kv: (kv[1]["prio"], kv[0])) if v["prio"] <= a.stop_after]
    log(a.runs, "chain start", arms)
    for arm in arms:
        step_arm(a.runs, arm, a.device, a.lock_wait_h)
        if arm == "H_s1":
            for s in (0, 1):
                step_iso(a.runs, s)
        log(a.runs, "arm done", arm)
        if arm in ("H_s1", "RA_s1", "H_s0_rep", "K4p_s1", "FEAT_s1", "C0A_s1"):
            read(a.runs)
    log(a.runs, "ZZ WH CHAIN DONE ZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
