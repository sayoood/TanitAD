"""Compute / IO cost model of the SH5 GPU stages (ESTIMATED; the first 20 chunk-steps of `fit` print the measured rate as ZZTHROUGHPUT).  python -B sh5_cost.py"""
import json

P, D_CTX, D, FF = 7680, 256, 256, 512
P_SUB, N_PTS, LAYERS, PASSES = 32, 9, 2, 12


def per_token_gf(sets_per_token):
    kv = LAYERS * 2 * 2 * P * D_CTX * D                       # K and V projections of the whole ctx, per layer
    q = P_SUB * N_PTS
    attn = LAYERS * 4 * q * P * D                             # QK^T and AV
    other = LAYERS * (q * (2 * D * D * 2 + 2 * 2 * D * FF)) + P_SUB * 2 * (N_PTS * D) * D
    fwd = kv + sets_per_token * (attn + other)
    return 3 * fwd / 1e9                                      # fwd + bwd ~ 3x fwd


def cost(T, sets, learners, tflops=(3, 6, 10)):
    gf = per_token_gf(sets / T)
    visits = 0.8 * T * PASSES
    pf = learners * visits * gf / 1e6
    return {"GF_per_token_visit": round(gf, 1), "PF": round(pf, 1), "hours": {str(t): round(pf * 1e3 / t / 3600, 2) for t in tflops}}


out = {}
for name, T, sets in (("LOCAL_816_tokens", 816, 1444), ("FULL_2782_tokens", 2782, 4941)):
    out[name] = {"A1_per_seed_20_learners": cost(T, sets, 20), "A2_per_seed_5_learners": cost(T, sets, 5), "B_seed0_25_learners": cost(T, sets, 25), "B_seed1_15_learners": cost(T, sets, 15),
                 "PRE_1_learner": cost(T, sets, 1), "ctx_gib": round(T * P * D_CTX * 2 / 2 ** 30, 2), "ctx_build_min_at_1.66s_per_token": round(T * 1.66 / 60, 1),
                 "ctx_read_min_per_pass_at_24MiBps": round(T * P * D_CTX * 2 / 2 ** 20 / 24 / 60, 1)}
print(json.dumps(out, indent=1))
