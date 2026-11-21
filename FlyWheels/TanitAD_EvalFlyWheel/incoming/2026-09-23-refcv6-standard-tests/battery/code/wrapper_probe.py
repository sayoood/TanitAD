"""AMENDMENT A2 (SPEC.md): the wrapper control, re-defined to isolate the MERGE from the arithmetic.

    python wrapper_probe.py --ckpt <ckpt> --config <config.json> --out <json> [--g0-a1-json <json>]

One batch of 4 (the first 4 windows of the first in-run eval batch), DDIM eps zeroed in every arm.
Precision conditions P1 as_run / P2 cudnn_det / P3 fp32_det; arms plain / repeat / permuted /
micro[1,3] / micro[3,1]; Floor = max rel(|plain-repeat|, |plain-permuted|); Wrapper = max rel(|plain -
micro13|, |plain - micro31|). A2 clause: under P3 every term Wrapper <= max(3*Floor, 1e-5), AND the
deliberate wrapper regressions W1 (0-dim merged as part 0) and W2 (parts concatenated in reverse)
each exceed that bar on >= 1 term. The bar is written in SPEC.md BEFORE this ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402
import reproduce_inrun_eval as G  # noqa: E402
import gpu_gate  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402

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
    """DELIBERATE REGRESSION W2: batch tensors concatenated with the micro-parts in REVERSED order."""

    def _merge(self, parts, path):
        first = parts[0]
        if torch.is_tensor(first) and first.dim() >= 1 and all(
                torch.is_tensor(p) and p.dim() >= 1 and p.shape[0] == m
                for p, m in zip(parts, self.sizes)):
            self.rules[path] = "W2-reversed-cat"
            return torch.cat(list(reversed(parts)), 0)
        return super()._merge(parts, path)


def reverse_rows(eb, n):
    idx = list(range(n))[::-1]
    o = {}
    for k, v in eb.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            o[k] = v[idx]
        elif isinstance(v, G.LawOnly):
            o[k] = G.LawOnly(v.u8[idx], v.law_idx)
        elif isinstance(v, (list, tuple)) and len(v) == n:
            o[k] = type(v)(v[i] for i in idx)
        else:
            o[k] = v
    return o


def scalars(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.ndim == 0:
            out[k] = float(v.detach())
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def rel(a, b):
    return abs(a - b) / max(abs(a), 1e-12)


def set_condition(c, trunk, chunk=None):
    torch.backends.cudnn.allow_tf32 = c["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = c["matmul_tf32"]
    torch.backends.cudnn.deterministic = c["deterministic"]
    torch.backends.cudnn.benchmark = c["benchmark"]
    trunk.memory_levers["bf16"] = c["trunk_bf16"]
    trunk.memory_levers["channels_last"] = c["trunk_nhwc"]
    if chunk is not None:
        trunk.memory_levers["chunk_ckpt"] = int(chunk)


def run_arms(tr, model, sb, n, mode, abl, arms):
    res = {}
    for name in arms:
        torch.manual_seed(0)
        with torch.no_grad():
            if name in ("plain", "repeat"):
                d = tr.compute_losses_v3(model, sb, "cuda", mode=mode, ablate_frames=abl)
            elif name == "permuted":
                d = tr.compute_losses_v3(model, reverse_rows(sb, n), "cuda", mode=mode,
                                         ablate_frames=abl)
            else:
                cls = {"micro13": MicroBatchForward, "micro31": MicroBatchForward,
                       "W1": W1MergeFirstOnly, "W2": W2ReversedCat}[name]
                sizes = [3, 1] if name == "micro31" else [1, 3]
                w = cls(model, sizes).install()
                try:
                    d = tr.compute_losses_v3(model, sb, "cuda", mode=mode, ablate_frames=abl)
                finally:
                    w.remove()
        res[name] = scalars(d)
        torch.cuda.synchronize()
    return res


def analyse(res, *, include_mutations):
    terms = sorted(k for k, v in res["plain"].items() if abs(v) > 1e-6 and k != "goal_gate_grad")
    rows, worst_w, worst_f, bad = {}, 0.0, 0.0, []
    for k in terms:
        p = res["plain"][k]
        floor = max(rel(p, res["repeat"][k]), rel(p, res["permuted"][k]))
        wrap = max(rel(p, res["micro13"][k]), rel(p, res["micro31"][k]))
        bar = max(3.0 * floor, 1e-5)
        r = {"plain": p, "floor": floor, "wrapper": wrap, "bar": bar, "ok": wrap <= bar}
        if include_mutations:
            for m in ("W1", "W2"):
                if m in res and k in res[m]:
                    r[f"{m}_rel"] = rel(p, res[m][k])
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
            out[f"{m}_detected_terms"] = det
            out[f"{m}_detected"] = bool(det)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g0-a1-json", default=None, help="to compose G0-A2 for this checkpoint")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    if not a.no_gate:
        while True:
            g = gpu_gate.gate(self_pid=None)
            if g["ok"]:
                break
            print(f"[A2] GPU gate WAIT {json.dumps(g)}", flush=True)
            time.sleep(60)
    tr = L.trainer()
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, "cuda")
    e_ds, _eps, _drec = L.build_eval_dataset(
        model, cfg, args, config, with_perception_targets=True,
        dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), int(args.batch))
    law_idx = int(tr.LAW_AHEAD) - 1
    first = next(iter(G.iter_batches(e_ds, perm, int(args.batch), 1, law_idx)))
    n = 4
    sb = G.sub_batch(first, n)
    G.patch_frames_to_device(tr)
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    lev0 = dict(trunk.memory_levers)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "wrapper_probe.py (SPEC AMENDMENT A2)",
           "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(), "ckpt": a.ckpt,
           "ckpt_md5": L.md5_file(a.ckpt), "step": mrec["state_dict"]["step"],
           "batch": "first 4 windows of in-run eval batch 1", "ddim_eps": "zeros (every arm)",
           "trunk_levers_as_built": lev0, "conditions": {}}
    orig_randn_like = torch.randn_like
    torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
    try:
        for cname, c in CONDITIONS.items():
            arms = ["plain", "repeat", "permuted", "micro13", "micro31"]
            if cname == "P3_fp32_det":
                arms += ["W1", "W2"]
            chunk_used = lev0.get("chunk_ckpt")
            t0 = time.time()
            try:
                set_condition(c, trunk)
                res = run_arms(tr, model, sb, n, mode, abl, arms)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chunk_used = 4
                set_condition(c, trunk, chunk=4)
                res = run_arms(tr, model, sb, n, mode, abl, arms)
            an = analyse(res, include_mutations=(cname == "P3_fp32_det"))
            rec["conditions"][cname] = {"settings": c, "chunk_ckpt_used": chunk_used,
                                        "wall_s": round(time.time() - t0, 1),
                                        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                                        "raw": res, "analysis": an}
            print(f"[A2] {cname}: max wrapper rel {an['max_wrapper_rel']:.3e}, max floor rel "
                  f"{an['max_floor_rel']:.3e}, over bar {an['terms_over_bar']}"
                  + (f", W1 detected {len(an['W1_detected_terms'])}, W2 detected "
                     f"{len(an['W2_detected_terms'])}" if cname == "P3_fp32_det" else ""),
                  flush=True)
            trunk.memory_levers.update(lev0)
    finally:
        torch.randn_like = orig_randn_like
        trunk.memory_levers.update(lev0)
    p3 = rec["conditions"]["P3_fp32_det"]["analysis"]
    clause = ("PASS" if (not p3["terms_over_bar"] and p3["W1_detected"] and p3["W2_detected"])
              else "VOID" if not (p3["W1_detected"] and p3["W2_detected"]) else "FAIL")
    rec["A2_wrapper_clause"] = clause
    rec["A2_wrapper_reasons"] = (
        ([f"merge defect: {k} wrapper {p3['terms'][k]['wrapper']:.3e} > bar {p3['terms'][k]['bar']:.3e}"
          for k in p3["terms_over_bar"]]) +
        ([] if p3["W1_detected"] else ["W1 (0-dim first-only) NOT detected: no power"]) +
        ([] if p3["W2_detected"] else ["W2 (reversed cat) NOT detected: no power"]))
    if a.g0_a1_json:
        a1 = json.load(open(a.g0_a1_json, encoding="utf-8"))
        g0 = json.load(open(a1["g0_json"], encoding="utf-8")) if Path(a1["g0_json"]).exists() else None
        non_wrap = [r for r in (g0["verdict"]["reasons"] if g0 else [])
                    if not r.startswith("M1") and not r.startswith("wrapper control")]
        detected = [m for m, d in a1["detection"].items() if d["detected"]]
        if g0 and g0.get("ckpt_md5") != rec["ckpt_md5"]:
            raise SystemExit("[A2] the G0 artifact is for another checkpoint")
        rec["G0_A2"] = ("PASS" if (not non_wrap and detected and clause == "PASS") else
                        "VOID" if (clause == "VOID" or not detected) else "FAIL")
        rec["G0_A2_reasons"] = non_wrap + rec["A2_wrapper_reasons"] + (
            [] if detected else ["no M2-M4 mutation detected"])
        rec["G0_A1_as_registered"] = a1["G0_A1"]
    rec["wall_s"] = round(time.time() - t_all, 1)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: rec.get(k) for k in ("A2_wrapper_clause", "A2_wrapper_reasons",
                                               "G0_A1_as_registered", "G0_A2", "G0_A2_reasons")},
                     indent=1))


if __name__ == "__main__":
    main()
