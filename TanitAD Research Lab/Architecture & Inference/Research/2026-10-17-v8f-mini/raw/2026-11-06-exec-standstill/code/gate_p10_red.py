#!/usr/bin/env python3
"""gate_p10_red.py -- the RED arm of G-P10: the exec tool with a MUTATED P10 floor (the prior's output + 1e-3 m, at every tick), same subset, same flags as G_R2_eps1e-3.

The mutation is a monkeypatch of ``floor_planner`` in this harness only (the tool file is untouched). gates_exec.py then requires the P10 floor's tick-0 plan
(its first 10 executed knots) to differ from the unmutated run's: if the check could not see a changed prior, the gate would be blind.

usage: python -I gate_p10_red.py --out-root C:/Users/Admin/exec_standstill_scratch/gate_red
"""
import argparse
import importlib.util
import os
import sys

TOOL = "C:/Users/Admin/exec_standstill_scratch/tree/stack/scripts/v8f_t1_eval_exec.py"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CK = f"{ART}/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"
GRID = "C:/Users/Admin/exec_standstill_scratch/subset/grid_subset.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("exec_tool_red", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    orig = tool.floor_planner

    def mutated(model, device):
        base = orig(model, device)

        def plan(batch):
            return base(batch) + 1e-3                                   # the mutation: a changed prior output
        return plan
    tool.floor_planner = mutated
    tag = "G_P10_RED"
    out = os.path.join(a.out_root, tag)
    os.makedirs(out, exist_ok=True)
    argv = ["run", "--ckpt", CK, "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
            "--eval-grid", GRID, "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic",
            "--limit", "causal", "--dump-ticks", f"{out}/ticks", "--dump-heads", "--device", "cpu", "--smooth", "sg72", "--scene", "logged", "--no-analyze",
            "--dump-exec-ch", f"{out}/exec_ch_{tag}.npz", "--exec-eps", "1e-3"]
    return tool.main(argv)


if __name__ == "__main__":
    sys.exit(main())
