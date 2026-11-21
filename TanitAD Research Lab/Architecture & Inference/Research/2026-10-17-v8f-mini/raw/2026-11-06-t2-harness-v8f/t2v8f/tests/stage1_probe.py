"""Isolated probe for tests/test_t2v8f_stage1.py: runs ONE stage-1 job (`actor`) from an extracted package in its own process, with an optional deliberate fault, and prints one line
`PROBE_JSON {...}`. Isolation matters: the stage-1 module imports `t2v8f_run`, whose module state (kit, tree, renderer) must not leak into the pytest process.

    python -I stage1_probe.py --pkg P --kit K --pull D [--mut none|roi_shift|empty_draws_lead]
"""
import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--kit", required=True)
ap.add_argument("--pull", required=True)
ap.add_argument("--mut", default="none", choices=("none", "roi_shift", "empty_draws_lead"))
a = ap.parse_args()
sys.path.insert(0, f"{a.pkg}/code")
import t2v8f_stage1 as S1  # noqa: E402

ctx = S1.Ctx(argparse.Namespace(pkg=a.pkg, kit=a.kit, hf=f"{a.pkg}/hf_unused", pull=a.pull, stub=True, matrix_inproc=True, min_free_gb=0.0, cmd="all"))
if a.mut == "roi_shift":                                       # the ROI the falsifier reads is 400 px to the right of where the lead is drawn
    _orig = S1.lead_roi
    S1.lead_roi = lambda *p, **k: (lambda b: (b[0] + 400, b[1], b[2] + 400, b[3]))(_orig(*p, **k))
if a.mut == "empty_draws_lead":                                # the `empty` condition renders the lead too
    def _bad_empty(ctx_, sid, c2ns, ts_us):
        r, tr, _, _, _, _ = S1.build_renderer(ctx_, sid, objects=False)
        out = []
        for (f, (lx, ly)), c2n, t in zip(sorted(ctx_.manifest["actor_leads"][sid].items(), key=lambda kv: int(kv[0])), c2ns, ts_us):
            d = S1.ActorDouble(r, S1.stub_truth_roi(r, lx, ly), 0.0, 1e18)
            d.set_actors_enabled(True)
            out.append(d.render(c2n, actor_time_us=t)[0])
        return out
    S1.render_empty = _bad_empty
res = S1.actor(ctx)
print("PROBE_JSON " + json.dumps(res, default=str))
