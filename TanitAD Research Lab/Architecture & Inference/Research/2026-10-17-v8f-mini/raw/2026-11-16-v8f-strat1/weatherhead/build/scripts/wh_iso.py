"""wh_iso.py -- T-ISO (c) and (d) of PREREG_V8F_WEATHERHEAD (card 4.3): the pooled head feeds neither the planner nor any goal path.

  c   ``forward_infer`` at batch 64 on ALL eval windows, S1 (``--ctx-pool off``) vs S1 + head (``--ctx-pool on``), through the T0 evaluator's own loop: every CTX_ISOLATED_OUTPUTS key, ``traj60``, ``traj30`` and
      the shipped ``ctx_logits`` must be ``torch.equal``.  Regression arm: S1 seed 0 vs S1 seed 1 (different planner weights) must read DIFFERENT.
  d   the T1-kin dumps (``--limit causal``, batch 32, as the read) of S1 re-dumped and of S1 + head, in ONE environment and one thread count: every array of the tick file except ``ctx_pool_logits`` byte-equal.
      Regression arm: the S1 seed 0 dump vs the S1 seed 1 dump must read DIFFERENT.
When (d) reads byte-equal at a seed, the 24 driving cells and the 6 route-setting cells of that seed read exactly 0.0 [0, 0]: a STRUCTURAL zero (an identity, not an estimate)."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402
import wh_run as RUN  # noqa: E402

M = W.M
KEYS_C = M.CTX_ISOLATED_OUTPUTS + ("traj60", "traj30", "ctx_logits")


def _record_run(argv_ckpt, tag, ctx_pool, out_dir, probe=None):
    """run the T0 evaluator IN-PROCESS with ``forward_infer`` wrapped to record the isolated outputs of every batch -> (list of {key: tensor}, rc)"""
    import importlib.util
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("v8f_mini_eval_wh", W.TREE + "/stack/scripts/v8f_mini_eval.py")
    EV = importlib.util.module_from_spec(spec)
    sys.path.insert(0, W.TREE + "/stack/scripts")
    spec.loader.exec_module(EV)
    rec = []
    orig = M.V8FMini.forward_infer

    def wrapped(self, batch):
        if probe is not None and not probe and self.ctx_pool_head is not None:       # the FIRST real batch of a pooled-head run: T-ISO (b) and T-VIS on the REAL weights and real data
            probe["T-ISO-b"] = M.ctx_pool_isolation_problems(self, batch)
            try:
                probe["T-VIS"] = {"pass": True, **M.assert_ctx_pool_vision_only(self, batch)}
            except AssertionError as e:
                probe["T-VIS"] = {"pass": False, "problem": str(e)[:300]}
        out = orig(self, batch)
        rec.append({k: out[k].detach().clone() for k in KEYS_C if k in out})
        return out
    M.V8FMini.forward_infer = wrapped
    try:
        av = RUN.t0_argv(argv_ckpt, tag, ctx_pool, out_dir)[2:]
        EV.main(av)
    finally:
        M.V8FMini.forward_infer = orig
    return rec


def compare_runs(a, b):
    import torch
    bad, n = [], 0
    if len(a) != len(b):
        return ["different batch counts"], 0
    for i, (x, y) in enumerate(zip(a, b)):
        for k in x:
            if k not in y:
                continue
            n += 1
            if not torch.equal(x[k], y[k]):
                bad.append(f"batch {i} {k} differs")
    return bad, n


def iso_c(s1, head, s1_other, out_dir, tag):
    import time
    t0 = time.time()
    while W.commit_headroom() < W.COMMIT_MARGIN and time.time() - t0 < 3600:           # the commit gate (>= 5.5 GB free commit); a script waits, no model turn does
        time.sleep(30)
    ref = _record_run(s1, f"WH-ISOC-{tag}-s1", "off", out_dir)
    probe = {}
    hd = _record_run(head, f"WH-ISOC-{tag}-head", "on", out_dir, probe)
    bad, n = compare_runs(ref, hd)
    out = {"check": "T-ISO (c)", "n_batches": len(ref), "n_windows_batches_batch": 64, "n_compared": n, "keys": list(KEYS_C), "differences": bad[:5], "pass_identity": not bad and n > 0}
    if s1_other:
        oth = _record_run(s1_other, f"WH-ISOC-{tag}-other", "off", out_dir)
        rb, rn = compare_runs(ref, oth)
        out["regression_arm"] = {"what": "S1 seed 0 vs S1 seed 1 (different planner weights)", "n_differences": len(rb), "must_read": "DIFFERENT", "pass": len(rb) > 0}
    out["T-ISO-b"] = {"problems": probe.get("T-ISO-b"), "pass": probe.get("T-ISO-b") == []}
    out["T-VIS"] = probe.get("T-VIS", {"pass": False})
    out["pass"] = bool(out["pass_identity"] and out.get("regression_arm", {"pass": True})["pass"] and out["T-ISO-b"]["pass"] and out["T-VIS"]["pass"])
    return out


def _t1(ckpt, arm, ctx_pool, out, max_windows):
    rc = RUN.run_argv(RUN.t1_argv(ckpt, f"WH-{arm}_m10_causal_heads", ctx_pool, out, max_windows), f"{out}/log_WH-{arm}_t1.txt", wait_s=3600)
    return rc, f"{out}/ticks/ticks_WH-{arm}_m10_causal_heads.npz"


META = ("ckpt", "tag")                    # 0-d string scalars the dump carries about ITSELF (its checkpoint path and tag): metadata, not a planner array; they are listed in the record, never compared


def compare_ticks(p, q, skip=("ctx_pool_logits",)):
    za, zb = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
    meta = {k for k in META if k in za.files and za[k].ndim == 0 and za[k].dtype.kind in "US"}
    skip = tuple(skip) + tuple(meta)
    ka, kb = set(za.files) - set(skip), set(zb.files) - set(skip)
    bad = []
    if ka != kb:
        bad.append(f"key sets differ: {sorted(ka ^ kb)[:6]}")
    for k in sorted(ka & kb):
        a, b = za[k], zb[k]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            bad.append(f"{k} differs")
    return bad, len(ka & kb), sorted((set(za.files) ^ set(zb.files)) - meta)


def iso_d(s1, head, s1_other, out_dir, tag, max_windows=0):
    os.makedirs(out_dir, exist_ok=True)
    r0, p0 = _t1(s1, f"ISOD-{tag}-s1", "off", out_dir, max_windows)
    r1, p1 = _t1(head, f"ISOD-{tag}-head", "on", out_dir, max_windows)
    if r0 or r1:
        return {"check": "T-ISO (d)", "pass": False, "reason": f"dump rc {r0} / {r1}"}
    bad, n, only = compare_ticks(p0, p1)
    out = {"check": "T-ISO (d)", "n_arrays_compared": n, "only_in_one": only, "differences": bad[:6], "pass_identity": not bad and n > 0 and only == ["ctx_pool_logits"], "max_windows": max_windows,
           "environment": {"threads": 2, "device": "cpu", "batch": 32, "limit": "causal", "replan_steps": 10}}
    if s1_other:
        r2, p2 = _t1(s1_other, f"ISOD-{tag}-other", "off", out_dir, max_windows)
        rb = compare_ticks(p0, p2)[0] if r2 == 0 else ["dump failed"]
        out["regression_arm"] = {"what": "S1 seed 0 dump vs S1 seed 1 dump", "n_differences": len(rb), "must_read": "DIFFERENT", "pass": len(rb) > 0}
    out["pass"] = bool(out["pass_identity"] and out.get("regression_arm", {"pass": True})["pass"])
    out["structural_zero"] = bool(out["pass_identity"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("c", "d"))
    ap.add_argument("--s1", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--s1-other", default="", help="the regression arm's other S1 seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-windows", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    r = (iso_c if a.which == "c" else (lambda *x: iso_d(*x, max_windows=a.max_windows)))(a.s1, a.head, a.s1_other, a.out, a.tag)
    json.dump(r, open(f"{a.out}/iso_{a.which}_{a.tag}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: r[k] for k in r if k in ("check", "pass", "pass_identity", "differences", "regression_arm")}))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
