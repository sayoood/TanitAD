"""P2a: claims from the WEIGHTS ALONE. Tensor arithmetic on ckpt['model']; NO model is instantiated, NO input is created, NO forward pass.
Checkpoint: C0 s0 (M2b_s0_recft_C0). Plan speed := (x30[11] - x30[7]) / 0.8 s = the 0.8 s secant at +2.0 s (the P1 definition).
plan30[j] = p30[j] + res60[2j+1]  (subsample_02 takes the odd indices), p30 = base + P10net(standardise(f8)) + ym, res60 = basis @ (head_o(h) * 2.0).
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch

CK = sys.argv[1] if len(sys.argv) > 1 else "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/P2A_WEIGHTS.json"
ck = torch.load(CK, map_location="cpu", weights_only=False)
sd = {k: v.detach().double().numpy() for k, v in ck["model"].items()}
cfg = ck["cfg"]
GELU_MAX = 1.1289  # max slope of exact GELU (at x ~ 0.75); min slope -0.1289
J_HI, J_LO, DTAU = 11, 7, 0.8
R = {"ckpt": CK, "cfg": {k: cfg[k] for k in ("d", "heads", "ff", "nb", "res_scale_m", "depth_o", "n_q", "o_interface", "code_mix", "kin_mode") if k in cfg}}
nrm = np.linalg.norm
smax = lambda M: float(np.linalg.svd(M, compute_uv=False)[0])

# ---------------- readouts
r_p = np.zeros(60); r_p[2 * J_HI] = 1 / DTAU; r_p[2 * J_LO] = -1 / DTAU
basis = sd["basis"]
r_res = cfg["res_scale_m"] * (basis[2 * J_HI + 1] - basis[2 * J_LO + 1]) / DTAU          # [12]
R["readout"] = {"r_res": r_res.tolist(), "norm_r_res": float(nrm(r_res))}

# ---------------- PRIOR path (ego-only: no vision, no code)
W1, b1, W2, b2, ym, xm, xs = (sd[k] for k in ("prior.net.0.weight", "prior.net.0.bias", "prior.net.2.weight", "prior.net.2.bias", "prior.ym", "prior.xm", "prior.xs"))
rW2 = r_p @ W2                                   # [256]
g_lin_z = rW2 @ W1                               # [8] gain wrt STANDARDISED f8, GELU slope 1 (linear limit)
R["prior"] = {
    "std_xs": xs.tolist(), "mean_xm": xm.tolist(),
    "feature_names": ["v", "a", "k", "v^2", "v*a", "v*k", "v^2*k", "a*k"],
    "g_lin_z": g_lin_z.tolist(), "norm_g_lin_z": float(nrm(g_lin_z)),
    "g_lin_raw_units": (g_lin_z / xs).tolist(),
    "norm_rW2": float(nrm(rW2)), "sigma_max_W1": smax(W1), "sigma_max_W2": smax(W2),
    "bound_gain_z": float(nrm(rW2) * smax(W1) * GELU_MAX),
    "frob_W1": float(nrm(W1)), "frob_W2": float(nrm(W2)),
    "offset_r_p_dot_ym": float(r_p @ ym), "offset_r_p_dot_b2": float(r_p @ b2),
    "svd_W1": np.linalg.svd(W1, compute_uv=False).tolist(),
}
sv = np.linalg.svd(W2[[2 * J_HI, 2 * J_LO]], compute_uv=False)
R["prior"]["svd_W2_speed_rows"] = sv.tolist()

# ---------------- RESIDUAL head
Wo1, bo1, Wo2, bo2 = sd["head_o.0.weight"], sd["head_o.0.bias"], sd["head_o.2.weight"], sd["head_o.2.bias"]
Wo2x = Wo2[:12]                                  # channel-major: rows 0..11 = coefficients of channel 0 (x = longitudinal)
g_h = r_res @ Wo2x                               # [512]
a_h = g_h @ Wo1                                  # [1024] gain wrt reader output h (GELU slope 1)
gam, bet = sd["read_o.norm.weight"], sd["read_o.norm.bias"]
a_q = a_h.reshape(4, 256)
a_prime = a_q * gam[None]                        # gradient wrt LN INPUT up to the common 1/sigma_z factor and the projection
Hmax = float(np.sqrt(4) * (16.0 * np.abs(gam).max() + nrm(bet)))
sv_x = np.linalg.svd(Wo2x, compute_uv=False)
sv_all = np.linalg.svd(Wo2, compute_uv=False)
coef = Wo2x @ np.zeros(512)
R["residual"] = {
    "norm_g_h": float(nrm(g_h)), "norm_a_h": float(nrm(a_h)),
    "per_query_norm_a_h": [float(nrm(a_q[q])) for q in range(4)],
    "per_query_norm_a_prime": [float(nrm(a_prime[q])) for q in range(4)],
    "sigma_max_Wo1": smax(Wo1), "frob_Wo1": float(nrm(Wo1)),
    "svd_Wo2_xrows": sv_x.tolist(), "svd_Wo2_all": sv_all.tolist(),
    "effective_rank_Wo2x": float((sv_x.sum() ** 2) / (sv_x ** 2).sum()),
    "r_res_in_left_singular_basis": None,
    "readout_alignment_with_top_left_sv": None,
    "H_max_norm_reader_output_bound": Hmax,
    "bound_abs_res_speed": float(abs(r_res @ bo2[:12]) + nrm(g_h) * (smax(Wo1) * Hmax + nrm(bo1))),
    "linear_limit_swing_bound": float(nrm(a_h) * Hmax),
    "const_offset_r_res_dot_bo2": float(r_res @ bo2[:12]),
    "ln_gamma_mean": float(gam.mean()), "ln_gamma_max": float(np.abs(gam).max()), "ln_beta_norm": float(nrm(bet)),
}
U, S, Vt = np.linalg.svd(Wo2x, full_matrices=False)
proj = U.T @ r_res                               # components of r_res along each left singular vector
R["residual"]["r_res_in_left_singular_basis"] = proj.tolist()
R["residual"]["readout_alignment_with_top_left_sv"] = float(abs(proj[0]) / nrm(r_res))
R["residual"]["effective_gain_r_res_Wo2x"] = float(nrm(g_h))
# basis: how much of the speed readout does each of the 12 coefficients carry
R["residual"]["abs_r_res"] = np.abs(r_res).tolist()

# ---------------- INPUT-COLUMN BLOCKS into the memory of read_o (cond tokens + scene tokens)
blocks = {"e_ego": ("e_ego.weight", "e_ego.bias"), "e_nav": ("e_nav.weight", "e_nav.bias"), "e_lim": ("e_lim.weight", "e_lim.bias"),
          "e_ck": ("e_ck.weight", "e_ck.bias"), "e_code": ("e_code.weight", "e_code.bias"), "e_age": ("e_age.weight", "e_age.bias"),
          "adapter": ("adapter.1.weight", "adapter.1.bias")}
R["blocks"] = {}
for nm, (wk, bk) in blocks.items():
    Wm, bm = sd[wk], sd[bk]
    R["blocks"][nm] = {"shape": list(Wm.shape), "frob": float(nrm(Wm)), "sigma_max": smax(Wm), "bias_norm": float(nrm(bm)),
                       "frob_per_input_col_rms": float(nrm(Wm) / np.sqrt(Wm.shape[1])),
                       "stable_rank": float(nrm(Wm) ** 2 / smax(Wm) ** 2)}
Wc = sd["e_code.weight"]
R["blocks"]["e_code_lon_cols_27_34"] = {"frob": float(nrm(Wc[:, 27:34])), "cols": 7}
R["blocks"]["e_code_lat_cols_22_27"] = {"frob": float(nrm(Wc[:, 22:27])), "cols": 5}
R["blocks"]["e_code_goal_cols_0_22"] = {"frob": float(nrm(Wc[:, 0:22])), "cols": 22}
R["blocks"]["e_code_cons_cols_34_60"] = {"frob": float(nrm(Wc[:, 34:60])), "cols": 26}
R["blocks"]["pos_frob_per_token_rms"] = float(nrm(sd["pos"]) / np.sqrt(40))
R["blocks"]["time_frob_per_frame_rms"] = float(nrm(sd["time"]) / np.sqrt(4))
R["blocks"]["adapter_ln_gamma_rms"] = float(np.sqrt((sd["adapter.0.weight"] ** 2).mean()))

# ---------------- reader O value path / skip path capacities (weight-only).  a_prime: [4, 256] gradient wrt the final-LN input (per query), up to 1/sigma_z.
# The stream starts z0_q = q_q + mean(cond tokens)  (SKIP path: a cond token reaches every query with weight 1/6, scene tokens never do);
# each decoder layer l adds  sum_t alpha_qt (Wo Wv) m_t  (attention path; alpha = unknown, bounded in [0,1], sums to 1 per head).
dec = {}
for l in range(cfg["depth_o"]):
    pre = f"read_o.dec.layers.{l}.multihead_attn."
    Win = sd[pre + "in_proj_weight"]; Wv = Win[512:768]; Wo = sd[pre + "out_proj.weight"]
    dec[l] = {"Wov": Wo @ Wv, "Wk": Win[256:512], "Wq": Win[:256]}
    R.setdefault("value_path", {})[f"layer{l}"] = {"sigma_max_WoWv": smax(Wo @ Wv), "frob_WoWv": float(nrm(Wo @ Wv)), "sigma_max_Wv": smax(Wv), "sigma_max_Wo": smax(Wo),
                                                    "norm_bv": float(nrm(Win[512:768] @ np.zeros(256) + sd[pre + "in_proj_bias"][512:768]))}
N_COND, N_SCENE = 6, 160
def grad_at_memory(a_p, alpha_vec_fn):
    """sum over queries of the memory-token gradient vectors [256] in the token space: skip (1/6 for cond) + attention paths of both layers."""
    return None
# cond-token gradient vectors in token space (256), summed over the 4 queries as a Frobenius mass: sqrt(sum_q ||.||^2)
def mass(vs): return float(np.sqrt(sum(float(nrm(v)) ** 2 for v in vs)))
res = {}
for gname in ("e_ego", "e_nav", "e_lim", "e_ck", "e_code", "e_age"):
    Wg = sd[blocks[gname][0]]
    cap = {}
    # skip path: (1/6) a'_q for each query
    cap["skip"] = mass([Wg.T @ (a_prime[q] / N_COND) for q in range(4)])
    # attention path, uniform alpha = 1/166 per head (all heads sum to Wo Wv)
    for tag, alpha in (("attn_uniform", 1.0 / (N_COND + N_SCENE)), ("attn_alpha1", 1.0)):
        cap[tag] = mass([Wg.T @ sum(alpha * dec[l]["Wov"].T @ a_prime[q] for l in dec) for q in range(4)])
    res[gname] = cap
Wa = sd["adapter.1.weight"]
cap = {}
cap["skip"] = 0.0
# scene tokens: 160 tokens; gradient wrt each token's LN-output coordinates [1024]: Wa^T (alpha Wov^T a').  Block mass over 160 tokens at alpha uniform = sqrt(160)*alpha*|.|
for tag, alpha, ntok in (("attn_uniform", 1.0 / (N_COND + N_SCENE), N_SCENE), ("attn_alpha1_one_token", 1.0, 1)):
    per_tok = [Wa.T @ sum(alpha * dec[l]["Wov"].T @ a_prime[q] for l in dec) for q in range(4)]
    cap[tag] = float(np.sqrt(ntok)) * mass(per_tok)
res["adapter_scene"] = cap
R["capacity_weight_only"] = res
R["capacity_notes"] = ("mass = sqrt(sum_q ||.||^2) over the 4 query streams; gradient wrt the group's INPUT coordinates in the units its embedding was built for "
                       "(e_ego/adapter: unit-variance coordinates by construction of the standardise / LayerNorm; e_nav, e_lim, e_ck, e_code, e_age: raw input units). "
                       "common factor 1/sigma_z (final-LN row std) and the LN projection are dropped for ALL groups alike, so group ratios are exact within this approximation; "
                       "GELU slope in head_o taken as 1. Attention alpha is data-dependent and unknown without a forward pass: reported at uniform (1/166) and at 1 on one token.")
cp = res
R["ratios"] = {
    "ego_skip_over_vision_uniform": cp["e_ego"]["skip"] / cp["adapter_scene"]["attn_uniform"],
    "ego_skip_over_vision_one_token": cp["e_ego"]["skip"] / cp["adapter_scene"]["attn_alpha1_one_token"],
    "ego_skip_plus_attn_uniform_over_vision_uniform": (cp["e_ego"]["skip"] + cp["e_ego"]["attn_uniform"]) / cp["adapter_scene"]["attn_uniform"],
    "code_skip_over_ego_skip": cp["e_code"]["skip"] / cp["e_ego"]["skip"],
    "nav_skip_over_ego_skip": cp["e_nav"]["skip"] / cp["e_ego"]["skip"],
}
# the self-attention and FF share in the stream are not modelled; the prior path is separate (ego-only).
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({k: R[k] for k in ("prior", "residual", "ratios")}, indent=1)[:6000])
print(json.dumps(R["blocks"], indent=1)[:3000])
print(json.dumps(R["capacity_weight_only"], indent=1))
print(json.dumps(R["value_path"], indent=1))
