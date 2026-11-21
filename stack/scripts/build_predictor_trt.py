#!/usr/bin/env python3
"""Export + build + VERIFY the operative predictor's TensorRT engine.

⛔ **Why this file exists at all.** The engine shipped to Thor on 2026-08-02 was
**batch-1 static**, while the deployed :class:`TacticalSelector` fans over
``TacticalConfig.n_maneuvers = 9`` candidates. MEASURED on Thor 2026-08-03: the
9-candidate fan serialised through a batch-1 engine costs **244 % of the 100 ms
budget**; through a batch-9 engine, **56 %**. That is a **4.3x deployment
requirement**, not an optimisation — and it was invisible because every published
tick rolled one candidate.

The second reason is procedural: the script that produced the original engine was
a transient heredoc and **is not on disk**, so what it exported cannot be
inspected — which is precisely why the 0.726 MHA-fastpath claim could never be
settled. An engine that decides a deployment gets a checked-in builder.

Usage (on the TRT box; TensorRT's python bindings are SYSTEM packages)::

    PYTHONPATH=/usr/lib/python3.12/dist-packages:$HOME/TanitAD/stack \\
      python stack/scripts/build_predictor_trt.py \\
        --ckpt ~/models/flagship-v1-speedjerk/ckpt.pt \\
        --out  ~/trt_deploy/v1_pred_dyn1-9_fp16 \\
        --max-batch 9 --fp16

Verification is by **loading the engine and executing it**, never by exit code:
``trtexec`` and ``scp`` have both exited 0 on this programme while producing
nothing usable. The builder deserialises the plan, reads its optimisation
profile back, runs it at min and max batch, and compares against eager.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


class PredWrap(torch.nn.Module):
    """(states, actions[, intent]) -> z_next. The engine's IO contract, bound BY NAME.

    ⛔ **The intent input is not optional for the flagship.** The D-030 operative
    predictor is FiLM-conditioned on the tactical intent token — that conditioning
    IS the hierarchy seam. An engine exported with two inputs silently computes
    the UNCONDITIONED prediction, and nothing raises: `TacticalSelector` passes
    `intent=` as a keyword, a two-input wrapper accepts and ignores it, and the
    only symptom is that the decision changes. Measured 2026-08-03 while building
    this: an intent-less engine moved the selector's score by a **median 3.5
    units** against a 1.9-unit decision margin and flipped **13.5 %** of
    selections — which reads exactly like an fp16 precision failure and is not one.
    """

    def __init__(self, predictor, with_intent=False):
        super().__init__()
        self.p = predictor
        self.with_intent = with_intent

    def forward(self, states, actions, intent=None):
        if self.with_intent:
            return self.p(states, actions, intent=intent)[1]
        return self.p(states, actions)[1]


def export_onnx(predictor, path, *, batch, window, state_dim, action_dim,
                device="cuda", opset=17, dynamic=True, intent_dim=None):
    """Export the predictor to ONNX with a dynamic batch axis.

    ⛔ ``set_fastpath_enabled(False)`` stays. Two independent 2026-08-03 probes
    (10 cells, two weight distributions) found the flag INERT on this path and
    could not reproduce the 0.726 error the runbook attributes to it — but it
    costs 5.1e-7 in eager and ~1.6 % of engine latency, and the original failing
    script is not on disk, so the mechanism is unmapped rather than disproved.
    Cheap insurance stays until someone can inspect what actually failed.
    """
    torch.backends.mha.set_fastpath_enabled(False)
    st = torch.randn(batch, window, state_dim, device=device)
    ac = torch.randn(batch, window, action_dim, device=device)
    names = ["states", "actions"]
    args = (st, ac)
    if intent_dim:
        args = (st, ac, torch.randn(batch, intent_dim, device=device))
        names.append("intent")
    wrap = PredWrap(predictor, with_intent=bool(intent_dim)).eval()
    kw = {}
    if dynamic:
        kw["dynamic_axes"] = {n: {0: "B"} for n in names + ["z_next"]}
    t0 = time.perf_counter()
    torch.onnx.export(wrap, args, path, input_names=names,
                      output_names=["z_next"], opset_version=opset,
                      dynamo=False, **kw)
    return {"path": path, "export_s": round(time.perf_counter() - t0, 2),
            "MB": round(os.path.getsize(path) / 1e6, 1), "opset": opset,
            "dynamic_batch": bool(dynamic), "fastpath": "OFF",
            "inputs": names, "intent_dim": intent_dim}


def build_engine(onnx_path, plan_path, *, window, state_dim, action_dim,
                 min_batch=1, opt_batch=9, max_batch=9, fp16=True,
                 dynamic=True, timeout_s=3600, intent_dim=None):
    """Run trtexec. Returns a dict; **existence of the plan is the evidence**,
    not the exit code."""
    cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={plan_path}",
           "--skipInference"]
    if fp16:
        cmd.append("--fp16")
    if dynamic:
        def shp(b):
            s = f"states:{b}x{window}x{state_dim},actions:{b}x{window}x{action_dim}"
            return s + (f",intent:{b}x{intent_dim}" if intent_dim else "")
        cmd += [f"--minShapes={shp(min_batch)}", f"--optShapes={shp(opt_batch)}",
                f"--maxShapes={shp(max_batch)}"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_s)
    ok = os.path.exists(plan_path)
    return {"ok": ok, "cmd": " ".join(cmd), "build_s": round(time.perf_counter() - t0, 1),
            "MB": round(os.path.getsize(plan_path) / 1e6, 1) if ok else None,
            "returncode": r.returncode,
            "tail": None if ok else (r.stderr or r.stdout)[-1200:]}


class TRTPredictor(torch.nn.Module):
    """Drop-in for ``WorldModel.predictor``: returns ``(None, z_next)``.

    ⚠️ Subclasses ``nn.Module`` deliberately — ``model.predictor = <plain object>``
    raises ``TypeError: cannot assign ... as child module``, so a bare-class
    wrapper cannot actually be dropped in where deployment needs it.
    ⛔ Binds IO **by name**, never by index.
    """

    def __init__(self, plan_path, device="cuda"):
        super().__init__()
        import tensorrt as trt
        with open(plan_path, "rb") as f:
            self.engine = trt.Runtime(
                trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"failed to deserialize {plan_path}"
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        for n in ("states", "actions", "z_next"):
            assert n in self.names, f"engine IO {self.names} lacks {n!r}"
        self.has_intent = "intent" in self.names
        self.device = device
        self.plan_path = plan_path

    def profile_shapes(self, name="states", profile=0):
        """min/opt/max the engine will actually accept — read from the PLAN."""
        return [tuple(s) for s in self.engine.get_tensor_profile_shape(name, profile)]

    def forward(self, states, actions, intent=None):
        # ⛔ FAIL LOUD instead of silently computing the unconditioned prediction.
        # A wrapper that accepted `intent` and ignored it cost a whole gate on
        # 2026-08-03: it looked exactly like an fp16 precision failure (13.5 % of
        # tactical selections flipped) and was a dropped hierarchy input.
        if intent is not None and not self.has_intent:
            raise ValueError(
                f"engine {os.path.basename(self.plan_path)} has inputs {self.names} "
                "and cannot consume an intent token, but one was passed. Rebuild "
                "with --intent-dim, or the hierarchy seam is silently severed.")
        if intent is None and self.has_intent:
            raise ValueError(
                f"engine {os.path.basename(self.plan_path)} REQUIRES an intent "
                "token and none was passed.")
        st = states.detach().contiguous().float()
        ac = actions.detach().contiguous().float()
        self.ctx.set_input_shape("states", tuple(st.shape))
        self.ctx.set_input_shape("actions", tuple(ac.shape))
        keep = [st, ac]
        if self.has_intent:
            it = intent.detach().contiguous().float()
            if it.shape[0] == 1 and st.shape[0] > 1:
                it = it.expand(st.shape[0], -1).contiguous()
            self.ctx.set_input_shape("intent", tuple(it.shape))
            keep.append(it)
        out = torch.empty(tuple(self.ctx.get_tensor_shape("z_next")),
                          device=self.device, dtype=torch.float32)
        self.ctx.set_tensor_address("states", st.data_ptr())
        self.ctx.set_tensor_address("actions", ac.data_ptr())
        if self.has_intent:
            self.ctx.set_tensor_address("intent", keep[-1].data_ptr())
        self.ctx.set_tensor_address("z_next", out.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream), \
            "TRT execute_async_v3 returned False"
        torch.cuda.current_stream().synchronize()
        return (None, out)

    __call__ = torch.nn.Module.__call__


def verify_engine(plan_path, predictor, *, window, state_dim, action_dim,
                  batches=(1, 9), device="cuda", intent_dim=None):
    """VERIFY BY LOADING AND EXECUTING — the runbook's rule.

    Returns per-batch rel-err against eager plus the profile read back from the
    plan. A batch the profile does not cover fails here, loudly, instead of at
    deployment. When the engine carries an ``intent`` input, the reference is the
    eager predictor **with the same intent** — comparing a conditioned engine
    against an unconditioned reference is the wiring bug this file exists to stop.
    """
    eng = TRTPredictor(plan_path, device=device)
    rep = {"io_names": eng.names, "profile_states": eng.profile_shapes("states"),
           "profile_actions": eng.profile_shapes("actions"),
           "has_intent": eng.has_intent, "per_batch": {}}
    if eng.has_intent:
        rep["profile_intent"] = eng.profile_shapes("intent")
    for b in batches:
        st = torch.randn(b, window, state_dim, device=device)
        ac = torch.randn(b, window, action_dim, device=device)
        it = torch.randn(b, intent_dim, device=device) if eng.has_intent else None
        with torch.no_grad():
            ref = (predictor(st, ac, intent=it)[1] if eng.has_intent
                   else predictor(st, ac)[1])
            got = eng(st, ac, intent=it)[1]
        rep["per_batch"][b] = {
            "shape": tuple(got.shape),
            "rel_err": round(float((ref - got).norm() / ref.norm()), 8),
            "finite": bool(torch.isfinite(got).all()),
        }
    if eng.has_intent:
        # ⛔ The intent must MATTER: an engine that took the input and ignored it
        # would pass every check above.
        st = torch.randn(2, window, state_dim, device=device)
        ac = torch.randn(2, window, action_dim, device=device)
        i1 = torch.randn(2, intent_dim, device=device)
        with torch.no_grad():
            a_ = eng(st, ac, intent=i1)[1]
            b_ = eng(st, ac, intent=i1 * 0)[1]
        rep["intent_is_live_rel_change"] = round(
            float((a_ - b_).norm() / a_.norm()), 6)
        assert rep["intent_is_live_rel_change"] > 1e-5, \
            "engine has an 'intent' input but its output does not depend on it"
    # A row must not depend on its neighbours: batch-9 row 0 == batch-1 output.
    st1 = torch.randn(1, window, state_dim, device=device)
    ac1 = torch.randn(1, window, action_dim, device=device)
    it1 = torch.randn(1, intent_dim, device=device) if eng.has_intent else None
    nb = max(batches)
    stn, acn = st1.expand(nb, -1, -1).contiguous(), ac1.expand(nb, -1, -1).contiguous()
    itn = it1.expand(nb, -1).contiguous() if eng.has_intent else None
    with torch.no_grad():
        o1 = eng(st1, ac1, intent=it1)[1]
        on = eng(stn, acn, intent=itn)[1]
    rep["row_independence_rel_err"] = round(
        float((on - o1.expand_as(on)).norm() / o1.norm()), 8)
    return eng, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.onnx/.plan/.json)")
    ap.add_argument("--config", default="flagship4b", help="config factory name")
    ap.add_argument("--action-dim", type=int, default=3)
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=None, help="default = max")
    ap.add_argument("--max-batch", type=int, default=9)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--static", action="store_true", help="static batch = max-batch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--intent-dim", type=int, default=None,
                    help="export the D-030 intent input (flagship: "
                         "cfg.tactical_policy.d_intent = 256). ⛔ omit ONLY for a "
                         "model whose predictor is genuinely unconditioned")
    # ⛔ Geometry. The default config is 256x256 SQUARE; a 429-token arm (v5f,
    # 176x624 cylindrical) needs these, and the STRICT load below will REFUSE the
    # checkpoint without them (`encoder.pos` 429 vs 256) rather than quietly
    # resizing. That refusal is the feature — never feed an arm a raster it was
    # not trained at.
    ap.add_argument("--v2-subframe", default=None, help="e.g. 176x624")
    ap.add_argument("--frame-h", type=int, default=256)
    ap.add_argument("--frame-w", type=int, default=640)
    ap.add_argument("--frame-hfov", type=float, default=120.0)
    ap.add_argument("--projection", default="cylindrical")
    a = ap.parse_args(argv)

    from tanitad import config as C
    from tanitad.models.fourbrain import WorldModel

    cfg = getattr(C, f"{a.config}_config")()
    if a.v2_subframe:
        from types import SimpleNamespace

        from train_flagship_v4 import resolve_v2_frames
        resolve_v2_frames(SimpleNamespace(frame_h=a.frame_h, frame_w=a.frame_w,
                                          frame_hfov=a.frame_hfov,
                                          projection=a.projection,
                                          v2_subframe=a.v2_subframe, f_ref=None),
                          cfg, label="trt_build")
        cfg.speed_input = True
    object.__setattr__(cfg.predictor, "action_dim", a.action_dim)
    if getattr(cfg, "tactical_pred", None) is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", a.action_dim)
    model = WorldModel(cfg)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(sd)                      # STRICT — never strict=False
    model = model.to(a.device).eval()

    W, S, A = cfg.predictor.window, model.state_dim, a.action_dim
    opt_b = a.opt_batch or a.max_batch
    onnx_p, plan_p = f"{a.out}.onnx", f"{a.out}.plan"
    Path(onnx_p).parent.mkdir(parents=True, exist_ok=True)

    rep = {"ckpt": a.ckpt, "step": int(ck.get("step", -1)) if isinstance(ck, dict) else -1,
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "window": W, "state_dim": S, "action_dim": A,
           "batch_profile": ("static %d" % a.max_batch if a.static
                             else f"dynamic {a.min_batch}..{a.max_batch} (opt {opt_b})")}
    rep["onnx"] = export_onnx(model.predictor, onnx_p, batch=opt_b, window=W,
                              state_dim=S, action_dim=A, device=a.device,
                              opset=a.opset, dynamic=not a.static,
                              intent_dim=a.intent_dim)
    rep["engine"] = build_engine(onnx_p, plan_p, window=W, state_dim=S, action_dim=A,
                                 min_batch=a.min_batch, opt_batch=opt_b,
                                 max_batch=a.max_batch, fp16=a.fp16,
                                 dynamic=not a.static, intent_dim=a.intent_dim)
    if not rep["engine"]["ok"]:
        print(json.dumps(rep, indent=2))
        sys.exit(2)
    batches = (a.max_batch,) if a.static else (a.min_batch, a.max_batch)
    _, rep["verify"] = verify_engine(plan_p, model.predictor, window=W, state_dim=S,
                                     action_dim=A, batches=batches, device=a.device,
                                     intent_dim=a.intent_dim)
    with open(f"{a.out}.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
