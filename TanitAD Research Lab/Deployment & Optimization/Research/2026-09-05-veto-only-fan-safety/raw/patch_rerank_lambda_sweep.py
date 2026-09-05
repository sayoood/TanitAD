#!/usr/bin/env python3
"""Add the OFFSET-SHRINK sweep to the re-rank probe -- the design curve §6 asks for.

§6 measured that refcv3's fan is 8.56x the anchor bank's peak friction load and that the
offset head creates the whole blow-up. The obvious question is then quantitative rather than
rhetorical: **how much of the offset can be given back, and what does the fan lose?**

The sweep evaluates `path(lambda) = bank + lambda * offset` for lambda in [0, 1] on the SAME
forward pass the re-rank probe already pays for, and reports at each lambda:

  fan_envelope / fan_kamm_over / fan_off_reach / fan_peak_g   -- the feasibility it buys
  oracle_ade_m  (min over the 128 candidates of ADE to the human's logged 2 s future)
                                                              -- the fan QUALITY it costs
  fan_spread_m  (std of the 2 s endpoints)                    -- the coverage it costs
  sel_* under the model's own unchanged sel_idx               -- what the car would drive

lambda = 1 is the shipped fan (an identity control that must reproduce the probe's own
numbers), lambda = 0 is the raw bank. A curve with a knee is a design; a curve without one
says the offset is not compressible and the fix has to be structural.
"""
import io
import os

P = os.environ.get("RERANK_PATH", r"C:\Users\Admin\veto_wp\raw\rl_fan_rerank_probe.py")
LF = chr(10)

s = io.open(P, encoding="utf-8").read().replace(chr(13) + LF, LF)
if "LAMBDAS" in s:
    raise SystemExit("[patch] lambda sweep already present")

# 1. the constant
old = 'GATE_KS = (2, 4, 8, 16, 32, 128)'
new = ('GATE_KS = (2, 4, 8, 16, 32, 128)\n'
       '#: offset-shrink sweep. 1.0 is the shipped fan (identity control), 0.0 the raw bank.\n'
       'LAMBDAS = (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)')
assert old in s, "GATE_KS anchor"
s = s.replace(old, new)

# 2. collect per-window lambda rows inside the batch loop
old2 = """                row["agrees_model__kin_only"] = float(pick["kin_only"] == pick["model"])"""
new2 = """                row["agrees_model__kin_only"] = float(pick["kin_only"] == pick["model"])
                # ---- the OFFSET-SHRINK sweep, same forward, no extra GPU ------- #
                for lam in LAMBDAS:
                    q = lam_sc[lam]
                    for f in ("envelope", "kamm_over", "off_reach", "infeasible"):
                        row[f"lam{lam}__fan_{f}"] = float(q[f][j].float().mean())
                    row[f"lam{lam}__fan_peak_g"] = float(q["peak_g"][j].mean())
                    row[f"lam{lam}__sel_envelope"] = float(q["envelope"][j, pick["model"]])
                    row[f"lam{lam}__sel_peak_g"] = float(q["peak_g"][j, pick["model"]])
                    row[f"lam{lam}__oracle_ade_m"] = float(lam_ade[lam][j].min())
                    row[f"lam{lam}__sel_ade_m"] = float(lam_ade[lam][j, pick["model"]])
                    row[f"lam{lam}__fan_spread_m"] = float(lam_spread[lam][j])"""
assert old2 in s, "agrees_model anchor"
s = s.replace(old2, new2)

# 3. compute the lambda tensors once per batch, before the per-window loop
old3 = """            order = rank.argsort(dim=1, descending=True)                 # [B, N]
"""
new3 = """            order = rank.argsort(dim=1, descending=True)                 # [B, N]
            # ---- the OFFSET-SHRINK sweep: path(lambda) = bank + lambda * offset ---- #
            # `out["offset"]` is exactly what the decoder added to the bank, so the
            # interpolation is EXACT, not a reconstruction: at lambda = 1 the paths are
            # bitwise the emitted fan (asserted below), at lambda = 0 they are the bank.
            offs = out["offset"]                                         # [B, N, 8, 2]
            bank8 = fan - offs                                           # [B, N, 8, 2]
            lam_sc, lam_ade, lam_spread = {}, {}, {}
            for lam in LAMBDAS:
                p8 = bank8 + lam * offs
                p5 = D.with_origin(p8[..., :D.N_REWARD_SLOTS, :])
                lam_sc[lam] = FS.score_paths(p5, b["v0"], lead5,
                                             lead_len_m=D.LEAD_LEN_DEFAULT_M)
                lam_ade[lam] = (p8[..., :D.N_REWARD_SLOTS, :]
                                - gt[:, None]).norm(dim=-1).mean(dim=-1)
                lam_spread[lam] = p8[..., D.N_REWARD_SLOTS - 1, :].std(dim=1).norm(dim=-1)
            # identity control: lambda = 1 must reproduce the emitted fan EXACTLY.
            _d = float((lam_ade[1.0] - ade).abs().max())
            if _d > 1e-5:
                raise RuntimeError(f"lambda=1 identity control failed: max|diff| {_d:.3e} m "
                                   "- the sweep is not interpolating the shipped fan")
"""
assert old3 in s, "order anchor"
s = s.replace(old3, new3)

# 4. aggregate + print the curve
old4 = """    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)"""
new4 = """    res["lambda_sweep"] = {}
    for lam in LAMBDAS:
        res["lambda_sweep"][str(lam)] = {
            m: boot([r[f"lam{lam}__{m}"] for r in rows], eids, n_boot=a.n_boot, seed=a.seed)
            for m in ("fan_envelope", "fan_kamm_over", "fan_off_reach", "fan_infeasible",
                      "fan_peak_g", "sel_envelope", "sel_peak_g", "oracle_ade_m",
                      "sel_ade_m", "fan_spread_m")}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)"""
assert old4 in s, "makedirs anchor"
s = s.replace(old4, new4)

old5 = """    print(f"[rerank] -> {a.out}")"""
new5 = """    print("\\n=== OFFSET-SHRINK SWEEP  path(lambda) = bank + lambda * offset ===")
    print(f"  {'lambda':>7s} {'fan_env':>8s} {'fan_peak_g':>11s} {'fan_offreach':>13s} "
          f"{'oracle_ade':>11s} {'sel_ade':>8s} {'sel_env':>8s} {'spread_m':>9s}")
    for lam in LAMBDAS:
        q = res["lambda_sweep"][str(lam)]
        print(f"  {lam:7.2f} {q['fan_envelope']['mean']:8.4f} {q['fan_peak_g']['mean']:11.4f} "
              f"{q['fan_off_reach']['mean']:13.4f} {q['oracle_ade_m']['mean']:11.4f} "
              f"{q['sel_ade_m']['mean']:8.4f} {q['sel_envelope']['mean']:8.4f} "
              f"{q['fan_spread_m']['mean']:9.4f}")
    print(f"[rerank] -> {a.out}")"""
assert old5 in s, "print anchor"
s = s.replace(old5, new5)

io.open(P, "w", encoding="utf-8", newline=LF).write(s)
print("[patch] lambda sweep added to %s" % P)
