"""The G0 wrapper control (SPEC.md §2; the refcv6 battery's AMENDMENT A2 form, adopted from the start).

One batch of 4 (the first 4 windows of in-run batch 1), DDIM eps zeroed in every arm. Conditions
P1 as_run / P2 cudnn_det / P3 fp32_det; arms plain / repeat / permuted / micro[1,3] / micro[3,1];
Floor = max rel(|plain-repeat|, |plain-permuted|); Wrapper = max rel(|plain-micro13|, |plain-micro31|).
Clause: under P3 every term Wrapper <= max(3*Floor, 1e-5), AND the wrapper regressions W1 (0-dim
merged as part 0) and W2 (parts concatenated in reverse) each exceed that bar on >= 1 term.
Ported from the refcv6 package's `wrapper_probe.py`; called in-process by `g0_refcv7.py`.
"""
from __future__ import annotations

import time

import torch

from microbatch import MicroBatchForward

CONDITIONS = {
    "P1_as_run": {"cudnn_tf32": True, "matmul_tf32": False, "deterministic": False,
                  "benchmark": False, "trunk_bf16": True, "trunk_nhwc": True},
    "P2_cudnn_det": {"cudnn_tf32": False, "matmul_tf32": False, "deterministic": True,
                     "benchmark": False, "trunk_bf16": True, "trunk_nhwc": True},
    "P3_fp32_det": {"cudnn_tf32": False, "matmul_tf32": False, "deterministic": True,
                    "benchmark": False, "trunk_bf16": False, "trunk_nhwc": False},
}


class W1MergeFirstOnly(MicroBatchForward):
    """DELIBERATE REGRESSION W1: 0-dim / float outputs taken from micro-part 0 only."""

    def _merge(self, parts, path):
        first = parts[0]
        if (torch.is_tensor(first) and first.dim() == 0) or (
                isinstance(first, float) and not isinstance(first, bool)):
            self.rules[path] = "W1-first-only"
            return first
        return super()._merge(parts, path)


class W2ReversedCat(MicroBatchForward):
    """DELIBERATE REGRESSION W2: batch tensors concatenated with the micro-parts REVERSED."""

    def _merge(self, parts, path):
        first = parts[0]
        if torch.is_tensor(first) and first.dim() >= 1 and all(
                torch.is_tensor(p) and p.dim() >= 1 and p.shape[0] == m
                for p, m in zip(parts, self.sizes)):
            self.rules[path] = "W2-reversed-cat"
            return torch.cat(list(reversed(parts)), 0)
        return super()._merge(parts, path)


def _scalars(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.ndim == 0:
            out[k] = float(v.detach())
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and not str(k).startswith("_"):
            out[k] = float(v)
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(a), 1e-12)


def _set_condition(c, trunk, chunk=None):
    torch.backends.cudnn.allow_tf32 = c["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = c["matmul_tf32"]
    torch.backends.cudnn.deterministic = c["deterministic"]
    torch.backends.cudnn.benchmark = c["benchmark"]
    trunk.memory_levers["bf16"] = c["trunk_bf16"]
    trunk.memory_levers["channels_last"] = c["trunk_nhwc"]
    if chunk is not None:
        trunk.memory_levers["chunk_ckpt"] = int(chunk)


def _run_arms(tr, model, sb, sb_rev, device, mode, abl, arms, sizes):
    res = {}
    for name in arms:
        torch.manual_seed(0)
        with torch.no_grad():
            if name in ("plain", "repeat"):
                d = tr.compute_losses_v3(model, sb, device, mode=mode, ablate_frames=abl)
            elif name == "permuted":
                d = tr.compute_losses_v3(model, sb_rev, device, mode=mode, ablate_frames=abl)
            else:
                cls = {"micro13": MicroBatchForward, "micro31": MicroBatchForward,
                       "W1": W1MergeFirstOnly, "W2": W2ReversedCat}[name]
                sz = list(reversed(sizes)) if name == "micro31" else list(sizes)
                w = cls(model, sz).install()
                try:
                    d = tr.compute_losses_v3(model, sb, device, mode=mode, ablate_frames=abl)
                finally:
                    w.remove()
        res[name] = _scalars(d)
        del d
        torch.cuda.synchronize()
    return res


def _analyse(res, include_mutations):
    terms = sorted(k for k, v in res["plain"].items() if abs(v) > 1e-6 and k != "goal_gate_grad"
                   and all(k in res[a] for a in ("repeat", "permuted", "micro13", "micro31")))
    rows, worst_w, worst_f, bad = {}, 0.0, 0.0, []
    for k in terms:
        p = res["plain"][k]
        floor = max(_rel(p, res["repeat"][k]), _rel(p, res["permuted"][k]))
        wrap = max(_rel(p, res["micro13"][k]), _rel(p, res["micro31"][k]))
        bar = max(3.0 * floor, 1e-5)
        r = {"plain": p, "floor": floor, "wrapper": wrap, "bar": bar, "ok": wrap <= bar}
        if include_mutations:
            for m in ("W1", "W2"):
                if m in res and k in res[m]:
                    r[f"{m}_rel"] = _rel(p, res[m][k])
                    r[f"{m}_detected"] = r[f"{m}_rel"] > bar
        rows[k] = r
        worst_w, worst_f = max(worst_w, wrap), max(worst_f, floor)
        if not r["ok"]:
            bad.append(k)
    out = {"n_terms": len(terms), "max_wrapper_rel": worst_w, "max_floor_rel": worst_f,
           "terms_over_bar": bad, "terms": rows}
    if include_mutations:
        for m in ("W1", "W2"):
            det = [k for k, r in rows.items() if r.get(f"{m}_detected")]
            out[f"{m}_detected_terms"] = det[:40]
            out[f"{m}_detected"] = bool(det)
    return out


def probe(tr, model, first_batch, device, mode, abl, trunk, *, sub_batch_fn, n=4):
    lev0 = dict(trunk.memory_levers)
    bk0 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    sizes = [1, n - 1]
    sb = sub_batch_fn(first_batch, n)
    sb_rev = sub_batch_fn(first_batch, n, rows=list(range(n))[::-1])
    rec = {"batch": f"first {n} windows of in-run batch 1", "ddim_eps": "zeros (every arm)",
           "micro": sizes, "trunk_levers_as_built": lev0, "conditions": {}}
    orig = torch.randn_like
    torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
    try:
        for cname, c in CONDITIONS.items():
            arms = ["plain", "repeat", "permuted", "micro13", "micro31"]
            with_mut = cname in ("P3_fp32_det", "P2_cudnn_det")
            if with_mut:
                arms += ["W1", "W2"]
            chunk_used = lev0.get("chunk_ckpt")
            t0 = time.time()
            try:
                try:
                    _set_condition(c, trunk)
                    res = _run_arms(tr, model, sb, sb_rev, device, mode, abl, arms, sizes)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk_used = 4
                    _set_condition(c, trunk, chunk=4)
                    res = _run_arms(tr, model, sb, sb_rev, device, mode, abl, arms, sizes)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rec["conditions"][cname] = {"settings": c, "status": f"OOM: {str(exc)[:200]}"}
                trunk.memory_levers.update(lev0)
                continue
            an = _analyse(res, include_mutations=with_mut)
            rec["conditions"][cname] = {"settings": c, "chunk_ckpt_used": chunk_used,
                                        "wall_s": round(time.time() - t0, 1),
                                        "raw": res, "analysis": an}
            trunk.memory_levers.update(lev0)
            print(f"[wrapper] {cname}: max wrapper rel {an['max_wrapper_rel']:.3e}, max floor rel "
                  f"{an['max_floor_rel']:.3e}, over bar {len(an['terms_over_bar'])}"
                  + (f", W1 detected {an['W1_detected']}, W2 detected {an['W2_detected']}"
                     if with_mut else ""), flush=True)
    finally:
        torch.randn_like = orig
        trunk.memory_levers.update(lev0)
        (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = bk0
    use = "P3_fp32_det" if "analysis" in rec["conditions"].get("P3_fp32_det", {}) else "P2_cudnn_det"
    an = rec["conditions"].get(use, {}).get("analysis")
    if an is None:
        rec.update(clause="VOID", clause_condition=None, reasons=["neither P3 nor P2 ran"])
        return rec
    reasons = ([f"merge defect: {k} wrapper {an['terms'][k]['wrapper']:.3e} > bar "
                f"{an['terms'][k]['bar']:.3e}" for k in an["terms_over_bar"][:20]]
               + ([] if an["W1_detected"] else ["W1 (0-dim first-only) NOT detected: no power"])
               + ([] if an["W2_detected"] else ["W2 (reversed cat) NOT detected: no power"]))
    clause = ("PASS" if (not an["terms_over_bar"] and an["W1_detected"] and an["W2_detected"])
              else "VOID" if not (an["W1_detected"] and an["W2_detected"]) else "FAIL")
    rec.update(clause=clause, clause_condition=use, reasons=reasons)
    return rec
