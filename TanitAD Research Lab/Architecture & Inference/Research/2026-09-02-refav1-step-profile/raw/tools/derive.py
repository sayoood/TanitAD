"""Derived numbers from profile.json: phase families, component-sum cross-check,
bs-8 dev-box estimate, lever arithmetic. Prints markdown; writes derived.json beside profile.json."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_probe/profile_out/profile.json")
d = json.loads(p.read_text(encoding="utf-8"))
out = {}

FAMILY = {
    "op_rollout (fwd+bwd)": ["P/op_rollout"],
    "op loss: to_enc + mse": ["P/to_enc", "P/loss_mse"],
    "target path (adapter on future/ext, tac_field x30, str_subspace) [grad-recorded, detached]":
        ["P/std_future", "P/adapter_future", "P/tac_field_targets", "P/std_str_ext", "P/adapter_str_ext"],
    "tactical (tac_field_init + rollout)": ["P/tac_field_init", "P/tac_rollout"],
    "strategic (subspace + rollout)": ["P/str_subspace", "P/str_rollout"],
    "encode window (std + adapter)": ["P/encode", "P/std_window", "P/adapter_window"],
    "brains (strategic+tactical policies)": ["P/brains"],
    "instruments (participation eig)": ["P/participation_eig", "P/participation_ratio"],
    "heads/proposal/label CE": ["P/heads", "P/proposal", "P/loss_ce"],
    "forward glue (chan_std, mean, stack, slices)": ["P/forward"],
    "optimizer + clip + zero_grad": ["P/opt_step", "P/clip_grad", "P/zero_grad"],
    "h2d": ["P/h2d"],
    "loader": ["P/loader_fetch"],
}


def fam(name):
    for k, v in FAMILY.items():
        if name in v:
            return k
    return f"other:{name}"


for key in [k for k in d if k.startswith("step_")]:
    S = d[key]; P = S.get("profiler")
    if not P:
        continue
    ba = P["trace"].get("backward_attribution", {})
    n = max(P["trace"]["n_profiler_steps"], 1)
    rows = {}
    if ba.get("status") == "ok":
        for ph, ms in ba["forward_kernel_ms_by_phase"].items():
            r = rows.setdefault(fam(ph), {"fwd": 0.0, "bwd": 0.0}); r["fwd"] += ms / n
        for ph, ms in ba["backward_kernel_ms_by_phase"].items():
            r = rows.setdefault(fam(ph), {"fwd": 0.0, "bwd": 0.0}); r["bwd"] += ms / n
        una = ba["unattributed_kernel_ms"] / n
        tot = sum(r["fwd"] + r["bwd"] for r in rows.values()) + una
        print(f"\n## {key}: kernel time by family (ms/step; total {tot:.0f} incl. unattributed {una:.0f}); "
              f"profiler kernel total {P['kernel_time_ms_per_step']:.0f} ms")
        print("| family | fwd ms | bwd ms | total ms | % of kernel time |\n|---|---|---|---|---|")
        for k, r in sorted(rows.items(), key=lambda kv: -(kv[1]["fwd"] + kv[1]["bwd"])):
            t = r["fwd"] + r["bwd"]
            print(f"| {k} | {r['fwd']:.0f} | {r['bwd']:.0f} | {t:.0f} | {100*t/tot:.1f} |")
        print(f"| unattributed | | | {una:.0f} | {100*una/tot:.1f} |")
        out[key + "_families"] = {"rows": rows, "unattributed_ms": una, "total_ms": tot}

# component-sum cross-check + bs8 estimate
for key in [k for k in d if k.startswith("components_")]:
    C = d[key]; it = C["items"]
    ks = sorted(int(n.split("_K")[1].split("[")[0]) for n in it if n.startswith("op_rollout_K") and "oom" not in it[n])
    pts = [(K, it[f"op_rollout_K{K}[+to_enc+mse]"]["fwd_ms"], it[f"op_rollout_K{K}[+to_enc+mse]"]["bwd_ms"]) for K in ks]
    print(f"\n## {key}: op-rollout K-ladder (fwd/bwd ms)")
    for K, f_, b_ in pts:
        print(f"  K={K}: fwd {f_:.0f} bwd {b_:.0f} total {f_+b_:.0f}  per-step {(f_+b_)/K:.1f} ms")
    est30 = None
    if len(pts) >= 2:
        # least squares total = a*K + c on the fitted points
        xs = [K for K, _, _ in pts]; ys = [f_ + b_ for _, f_, b_ in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9)
        c = my - a * mx
        ss_res = sum((y - (a * x + c)) ** 2 for x, y in zip(xs, ys)); ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        est30 = a * 30 + c
        have30 = 30 in ks
        print(f"  linear fit total = {a:.1f}*K + {c:.0f} ms, R2 {r2:.4f}, n {len(xs)}, window K in [{min(xs)},{max(xs)}] -> "
              f"{'MEASURED' if have30 else 'ESTIMATED (' + f'{30/max(xs):.2f}x beyond window)'} K=30: {est30:.0f} ms")
        if have30:
            est30 = it["op_rollout_K30[+to_enc+mse]"]["fwd_ms"] + it["op_rollout_K30[+to_enc+mse]"]["bwd_ms"]
    others = {n: (r.get("fwd_ms", 0) or 0) + (r.get("bwd_ms", 0) or 0) for n, r in it.items()
              if not n.startswith("op_rollout_K") and "oom" not in r and ("fwd_ms" in r)}
    fixed = {n: r["ms"] for n, r in it.items() if "ms" in r}
    print("  other components (fwd+bwd ms): " + ", ".join(f"{n}={v:.0f}" for n, v in others.items()))
    print("  fixed (ms): " + ", ".join(f"{n}={v:.1f}" for n, v in fixed.items()))
    out[key + "_derived"] = {"ladder": pts, "op_rollout_K30_total_ms": est30, "others_ms": others, "fixed_ms": fixed}

# eig / loader fixed costs
E = d.get("eig", {}); L = d.get("loader", {})
if E:
    fixed_instr = E["live_gpu_fp64"]["median_ms"] + E["chan_std_op"]["median_ms"] + E["chan_std_tac"]["median_ms"] + E["chan_std_str"]["median_ms"]
    print(f"\ninstruments per step: eig {E['live_gpu_fp64']['median_ms']:.1f} + chan_std {E['chan_std_op']['median_ms']+E['chan_std_tac']['median_ms']+E['chan_std_str']['median_ms']:.1f} = {fixed_instr:.1f} ms; fp32 eig variant {E['fp32_variant_gpu']['median_ms']:.1f} ms")
    out["instruments_ms"] = fixed_instr
if L:
    print(f"loader per batch(8): hot {L['batch_lru_hot_s']['median']*1000:.0f} ms; live-state (8 misses) est {L['miss_cost_warm_pagecache_s']['per_batch_of_8_est_s']*1000 + L['slice_dequant_per_window_s']['per_batch_of_8_est_s']*1000:.0f} ms (warm page cache)")

(p.parent / "derived.json").write_text(json.dumps(out, indent=1, default=str))
