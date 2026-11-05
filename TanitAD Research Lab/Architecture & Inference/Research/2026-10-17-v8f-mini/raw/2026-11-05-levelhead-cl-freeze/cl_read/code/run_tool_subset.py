#!/usr/bin/env python3
"""run_tool_subset.py -- run the RI tool IN-PROCESS on the FIRST N windows of the eval grid (the banked rollouts' own batches: window k sits in the same 32-window batch as in a full run).

The tool's own ``--max-windows`` takes an EVENLY SPACED subset, so every window has different batch-mates than in the full rollout; a batched CPU kernel may round a row differently by its position in the
batch, and a 1e-5 m difference at tick 0 can be amplified by a discrete rule of the rollout (MEASURED, KA_ATTR_*: one window, 0.6-0.8 m at tick 2).  This wrapper keeps the tool's code untouched and truncates ONLY
``D.eval_grid_windows``' return value, so the first N windows are rolled in the batches (and batch positions) they have in a full run.
``--probe-level OUT.json`` additionally wraps ``V8FMini._level`` with a pass-through COUNTER (same return values, nothing changed): the number of calls the tool's own rollout makes to the level head and the
range of v_hat, written to OUT.json at exit (0 calls = the checkpoint has no head).
usage: python -I run_tool_subset.py --tool <v8f_t1_eval_ri.py> --first N [--probe-level OUT.json] -- run <the tool's own arguments>"""
import importlib.util
import json
import sys

sys.dont_write_bytecode = True


def main():
    a = sys.argv[1:]
    assert a[0] == "--tool" and a[2] == "--first", a[:4]
    tool, n, rest, probe = a[1], int(a[3]), a[4:], None
    if rest[0] == "--probe-level":
        probe, rest = rest[1], rest[2:]
    assert rest[0] == "--", rest[:1]
    rest = rest[1:]
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri_subset", tool)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    orig = mod.D.eval_grid_windows
    mod.D.eval_grid_windows = lambda grid, J, _o=orig, _n=n: _o(grid, J)[:_n]
    st = {"level_calls": 0, "v_hat_min": None, "v_hat_max": None, "v_hat_first_call": None, "v_in_abs_max": 0.0, "n_nonfinite": 0}
    if probe:
        import torch
        ol = mod.M.V8FMini._level

        def counted(self, b, adapt, last, _ol=ol):
            v_hat, v_in = _ol(self, b, adapt, last)
            st["level_calls"] += 1
            vh = v_hat.detach().float()
            st["n_nonfinite"] += int((~torch.isfinite(vh)).sum())
            lo, hi = float(vh.min()), float(vh.max())
            st["v_hat_min"] = lo if st["v_hat_min"] is None else min(st["v_hat_min"], lo)
            st["v_hat_max"] = hi if st["v_hat_max"] is None else max(st["v_hat_max"], hi)
            st["v_in_abs_max"] = max(st["v_in_abs_max"], float(v_in.detach().abs().max()))
            if st["v_hat_first_call"] is None:
                st["v_hat_first_call"] = [round(float(x), 5) for x in vh[:8]]
            return v_hat, v_in
        mod.M.V8FMini._level = counted
    rc = mod.main(rest)
    if probe:
        json.dump(st, open(probe, "w", encoding="utf-8", newline="\n"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
