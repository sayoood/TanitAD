#!/usr/bin/env python3
"""CTRL-CLIP: the pre-launch gradient-clip probe of PREREG_SCORERFT_PICK.md section 3.2 (an INSTRUMENT: it keeps no weights and writes no checkpoint).

    python clip_probe.py --argv-c argv_c.json --argv-p <P's PASS-token argv> --c-metrics metrics_scorerft_c.jsonl --out clip_probe.json [--scratch DIR] [--k 8]

WHAT IT DOES.  For each form (C: C's banked argv; P: P's PASS-token argv; only `--out` is replaced, by a scratch directory; `--steps` is kept so the lr is the real run's) it runs the REAL `train.run(a)` with a
PASS-THROUGH wrapper on `torch.nn.utils.clip_grad_norm_`: the wrapper calls the original and returns its value UNCHANGED, and records N_t (the pre-clip global norm) and c_t, torch's own coefficient
clamp(max_norm / (N_t + 1e-6), max=1.0) in N_t's dtype.  Binding b_t = [c_t < 1.0], no tolerance.  K = 8 steps (steps 0-7 = the weight rule's N_MICRO): the wrapper raises Stop on the (K+1)-th clip call, i.e. before the
optimiser step of step K, so no checkpoint is ever written (C's first `ckpt` event is at step 24).  Each form's scratch directory is deleted once the JSON is written.

THE READ.  Steps are paired by index.  n_on = #{b_P = 1, b_C = 0}, n_off = #{b_P = 0, b_C = 1}.  State: CLIP-0 (no step of either form binds), CLIP-SAME occ=a/K (n_on = n_off = 0), CLIP-T on=.. off=.. occ_P=../K occ_C=../K cmin_P=.. cmin_C=.. .
Reported, deciding nothing: occ_P, occ_C, the minimum c_t of each form, N_0^P and N_0^C.
VALIDITY (any FAIL -> state CLIP-INVALID -> PL3):
  V1  a known norm of 2.0 records b = 1, a known norm of 0.5 records b = 0, the wrapped clip leaves the gradients bit-identical to the bare clip (self-check run at probe time; RED: CLIP_PROBE_MUTATE=post_clip);
  V2  each form's step-0 training and onpolicy rows pass CTRL-S0's rule (scorerft_read.ctrl_s0) against C's banked step-0 rows (the trainer's new grad_norm_pre / clip_coef keys are P-only keys, allowed by S0_RANK_ONLY);
  V3  each form's (epoch, pos, samples, batch, lr) for steps 0..K-1 equal C's banked rows;
  V4  K finite N_t per form, no `resume` event, no checkpoint written, and the trainer's own logged grad_norm_pre / clip_coef equal the wrapper's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
K_DEFAULT = 8
MAX_NORM = 1.0
MUTATE = "CLIP_PROBE_MUTATE"                       # the deliberate-regression arm: "post_clip" records the norm AFTER clipping
ORDER_KEYS = ("epoch", "pos", "samples", "batch", "lr")


class Stop(Exception):
    pass


def _coef(n, max_norm):
    import torch
    return float(torch.clamp(max_norm / (n + 1e-6), max=1.0))


def make_wrapper(orig, rec: dict, k: int):
    """pass-through wrapper on clip_grad_norm_: returns the original's value unchanged; records N_t, c_t, b_t; raises Stop on call k + 1 BEFORE calling the original."""
    import torch

    def clip(parameters, max_norm, *a, **kw):
        if len(rec["norms"]) >= k:
            raise Stop()
        params = list(parameters) if not isinstance(parameters, torch.Tensor) else [parameters]
        n = orig(params, max_norm, *a, **kw)
        if os.environ.get(MUTATE) == "post_clip":                                                        # RED: the norm of the gradients AFTER the clip
            n_rec = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params if p.grad is not None]))
        else:
            n_rec = n
        c = _coef(n_rec, max_norm)
        rec["norms"].append(float(n_rec))
        rec["coefs"].append(c)
        rec["binding"].append(bool(c < 1.0))
        return n
    return clip


def selfcheck_v1() -> dict:
    """V1: a known norm of 2.0 records b = 1, a known norm of 0.5 records b = 0, and the wrapped clip leaves the gradients bit-identical to the bare clip."""
    import torch
    out, fails = {}, []
    for gnorm, want in ((2.0, True), (0.5, False)):
        g = [torch.tensor([gnorm * 0.6, gnorm * 0.8], dtype=torch.float32)]           # |g| = gnorm exactly in the 3-4-5 triple
        ps = [torch.nn.Parameter(torch.zeros(2))]
        ps2 = [torch.nn.Parameter(torch.zeros(2))]
        ps[0].grad, ps2[0].grad = g[0].clone(), g[0].clone()
        rec = {"norms": [], "coefs": [], "binding": []}
        ret_w = make_wrapper(torch.nn.utils.clip_grad_norm_, rec, 8)(ps, MAX_NORM)
        ret_b = torch.nn.utils.clip_grad_norm_(ps2, MAX_NORM)
        same_grad = bool(torch.equal(ps[0].grad, ps2[0].grad))
        same_ret = bool(torch.equal(ret_w, ret_b))
        ok = rec["binding"] == [want] and abs(rec["norms"][0] - gnorm) < 1e-6 and same_grad and same_ret
        out[f"norm_{gnorm}"] = {"recorded_norm": rec["norms"][0], "binding": rec["binding"][0], "expected_binding": want, "grads_bit_identical": same_grad, "return_identical": same_ret, "ok": ok}
        if not ok:
            fails.append(f"known norm {gnorm}: recorded {rec['norms'][0]!r}, binding {rec['binding'][0]}, grads identical {same_grad}, return identical {same_ret}")
    return {"id": "V1", "pass": not fails, "fails": fails, "detail": out}


def with_out(argv: list, scratch: str) -> list:
    a = list(argv)
    if "--out" not in a:
        raise SystemExit("clip_probe: the argv has no --out to replace")
    a[a.index("--out") + 1] = str(scratch)
    return a


def load_argv(x) -> list:
    if isinstance(x, (list, tuple)):
        t = [str(v) for v in x]
    else:
        j = json.load(open(x, encoding="utf-8"))
        t = [str(v) for v in (j["argv"] if isinstance(j, dict) else j)]
    return t[1:] if t and not t[0].startswith("--") else t                        # the script token is dropped


def _rows(path: Path):
    train, onp, resumes = {}, {}, 0
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            ev = r.get("event")
            if ev is None and "step" in r:
                train[int(r["step"])] = r
            elif ev == "onpolicy" and "step" in r:
                onp[int(r["step"])] = r
            elif ev == "resume":
                resumes += 1
    return train, onp, resumes


def run_form(argv: list, scratch, k: int = K_DEFAULT, refe: Path | None = None) -> dict:
    """run the real train.run(a) on `argv` (its --out replaced by `scratch`) under the pass-through wrapper until the (k+1)-th clip call; return the form's record."""
    sys.path.insert(0, str(refe or HERE))
    import torch
    import train as T
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    argv = with_out(argv, str(scratch))
    a = T.build_parser().parse_args(argv)
    rec = {"norms": [], "coefs": [], "binding": []}
    orig = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = make_wrapper(orig, rec, k)
    stopped, rc = False, None
    try:
        rc = T.run(a)
    except Stop:
        stopped = True
    finally:
        torch.nn.utils.clip_grad_norm_ = orig
    train, onp, resumes = _rows(scratch / "metrics.jsonl")
    ck = sorted(str(p.relative_to(scratch)) for p in scratch.rglob("*.pt"))
    return {"argv": argv, "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(), "rank_w": a.rank_w, "k": k, "stopped_before_step_k": stopped, "rc": rc,
            "norms": rec["norms"], "coefs": rec["coefs"], "binding": rec["binding"],
            "logged_norm": [train[s].get("grad_norm_pre") for s in range(k) if s in train], "logged_coef": [train[s].get("clip_coef") for s in range(k) if s in train],
            "order": {str(s): {kk: train[s].get(kk) for kk in ORDER_KEYS} for s in range(k) if s in train}, "n_train_rows": len([s for s in train if s < k]), "resume_events": resumes,
            "checkpoints_written": ck, "metrics_path": str(scratch / "metrics.jsonl"), "onpolicy_step0": onp.get(0)}


def judge(form_c: dict, form_p: dict, c_metrics: str, k: int = K_DEFAULT, check_c_sha: bool = True, v1: dict | None = None, read_dir: str | None = None) -> dict:
    """the probe's JSON: V1-V4, the paired read and the state.  Pure over the two form records and C's banked metrics (a path)."""
    rd = Path(read_dir or (HERE.parent / "eval" / "scorerft"))
    sys.path.insert(0, str(rd))
    import scorerft_read as SR
    v1 = v1 or selfcheck_v1()
    v2 = {"id": "V2", "pass": True, "fails": [], "detail": {}}
    for lab, f in (("C", form_c), ("P", form_p)):
        r = SR.ctrl_s0(c_metrics, f["metrics_path"], rank_w=(f["rank_w"] if lab == "P" else None), step=0, check_c_sha=check_c_sha, require_rank_L=(lab == "P"))
        v2["detail"][lab] = {"pass": r["pass"], "fails": r["fails"], "n_keys_compared": r.get("n_keys_compared"), "p_only_keys": r.get("p_only_keys")}
        if not r["pass"]:
            v2["pass"] = False
            v2["fails"] += [f"{lab}-form: {x}" for x in r["fails"]]
    tc, _, _ = SR._rows(c_metrics)
    v3 = {"id": "V3", "pass": True, "fails": [], "detail": {}}
    for lab, f in (("C", form_c), ("P", form_p)):
        bad = []
        for s in range(k):
            want = {kk: tc[s].get(kk) for kk in ORDER_KEYS} if s in tc else None
            got = f["order"].get(str(s))
            if want is None or got is None or any(not SR._same(want[kk], got[kk]) for kk in ORDER_KEYS):
                bad.append(s)
        v3["detail"][lab] = {"steps_differing": bad}
        if bad:
            v3["pass"] = False
            v3["fails"].append(f"{lab}-form: (epoch, pos, samples, batch, lr) differs from C's banked rows at steps {bad}")
    v4 = {"id": "V4", "pass": True, "fails": [], "detail": {}}
    for lab, f in (("C", form_c), ("P", form_p)):
        fails = []
        if len(f["norms"]) != k or not all(math.isfinite(x) for x in f["norms"]):
            fails.append(f"{len(f['norms'])} norms, finite {sum(math.isfinite(x) for x in f['norms'])} (need {k} finite)")
        if f["resume_events"]:
            fails.append(f"{f['resume_events']} resume events")
        if f["checkpoints_written"]:
            fails.append(f"checkpoints written: {f['checkpoints_written']}")
        if not f["stopped_before_step_k"]:
            fails.append("the run was not stopped by the wrapper (it ended on its own)")
        if f["logged_norm"] and (f["logged_norm"] != f["norms"][:len(f["logged_norm"])] or f["logged_coef"] != f["coefs"][:len(f["logged_coef"])]):
            fails.append("the trainer's logged grad_norm_pre / clip_coef differ from the wrapper's")
        v4["detail"][lab] = {"fails": fails}
        if fails:
            v4["pass"] = False
            v4["fails"] += [f"{lab}-form: {x}" for x in fails]
    valid = all(v["pass"] for v in (v1, v2, v3, v4))
    bC, bP = form_c["binding"], form_p["binding"]
    n_on = sum(1 for b_p, b_c in zip(bP, bC) if b_p and not b_c)
    n_off = sum(1 for b_p, b_c in zip(bP, bC) if not b_p and b_c)
    occ_c, occ_p = sum(bC), sum(bP)
    cmin_c, cmin_p = (min(form_c["coefs"]) if form_c["coefs"] else None), (min(form_p["coefs"]) if form_p["coefs"] else None)
    if not valid:
        state = "CLIP-INVALID"
    elif n_on + n_off > 0:
        state = f"CLIP-T on={n_on} off={n_off} occ_P={occ_p}/{k} occ_C={occ_c}/{k} cmin_P={cmin_p:.6g} cmin_C={cmin_c:.6g}"
    elif occ_c == 0 and occ_p == 0:
        state = "CLIP-0"
    else:
        state = f"CLIP-SAME occ={occ_p}/{k}"
    return {"instrument": "clip_probe.py", "card": "PREREG_SCORERFT_PICK.md section 3.2", "k": k, "max_norm": MAX_NORM, "forms": {"C": form_c, "P": form_p}, "validity": {"V1": v1, "V2": v2, "V3": v3, "V4": v4},
            "valid": valid, "state": state, "n_on": n_on, "n_off": n_off, "occ_P": occ_p, "occ_C": occ_c, "cmin_P": cmin_p, "cmin_C": cmin_c,
            "N0_P": (form_p["norms"][0] if form_p["norms"] else None), "N0_C": (form_c["norms"][0] if form_c["norms"] else None),
            "scope": f"steps 0-{k - 1} only; P's full-run clip path is UNMEASURED", "pl": ("PL3" if not valid else ("PL4" if n_on + n_off > 0 else None))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--argv-c")
    ap.add_argument("--argv-p")
    ap.add_argument("--c-metrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--form", choices=("c", "p"), default=None, help="(internal) run ONE form in this process and write its record to --out")
    ap.add_argument("--argv", default=None, help="(internal, with --form) the argv file of the form")
    a = ap.parse_args(argv)
    if a.form:
        rec = run_form(load_argv(a.argv), a.scratch, a.k)
        Path(a.out).write_text(json.dumps(rec, default=float), encoding="utf-8")
        return 0
    if not (a.argv_c and a.argv_p and a.c_metrics and a.scratch):
        ap.error("--argv-c, --argv-p, --c-metrics and --scratch are required")
    scratch = Path(a.scratch)
    recs = {}
    for lab, f_ in (("c", a.argv_c), ("p", a.argv_p)):
        fo = scratch / f"form_{lab}.json"
        scratch.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--form", lab, "--argv", f_, "--scratch", str(scratch / f"run_{lab}"), "--k", str(a.k), "--out", str(fo)],
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0 or not fo.exists():
            print(f"clip_probe: the {lab.upper()}-form run failed (rc {r.returncode}); no JSON written (PL3: the probe did not run)", file=sys.stderr)
            return 4
        recs[lab] = json.loads(fo.read_text(encoding="utf-8"))
    res = judge(recs["c"], recs["p"], a.c_metrics, a.k)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    print("ZZCLIP_PROBE " + res["state"] + f" valid={res['valid']}")
    for lab in ("c", "p"):
        shutil.rmtree(str(scratch / f"run_{lab}"), ignore_errors=True)                  # the probe's scratch runs are deleted once the JSON is written
    return 0 if res["valid"] else 5


if __name__ == "__main__":
    sys.exit(main())
