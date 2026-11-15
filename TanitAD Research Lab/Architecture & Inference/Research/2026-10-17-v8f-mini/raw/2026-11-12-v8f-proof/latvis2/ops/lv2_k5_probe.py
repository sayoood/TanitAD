"""lv2_k5_probe.py ARM CKPT : why control k5 of card PREREG_V8F_PROOF_ADD38 reads above its bar. Pod CPU, latent states only (no token, no D).
(a) the tool's rollout function applied to the BANKED start states and actions against the banked rollout (same checkpoint);
(b) the states assembled from the pod state file (pod token cache) against the banked start and future states (dev-box state file): size of the difference;
(c) frame alignment: for each window, which pod frame is nearest to the banked start state (must be frame 4) and to the banked future state (frame 14);
(d) the banked actions of the R3g file against those of the window list the tool uses.
Tier T0, a world-model diagnostic on logged frames."""
import json
import sys

import numpy as np

W = "/workspace/v8proof"
sys.path.insert(0, W + "/probe")
import imag_dump as ID  # noqa: E402

arm, ckpt = sys.argv[1], sys.argv[2]
torch, M, tree = ID.import_stack(W + "/tree_leadsup")
torch.set_num_threads(4)
cfg, model, st, ck = ID.load_model(torch, M, ckpt, require_head=False)
ctx = ID.predictor_context(st)
b = np.load(f"{W}/probe/lv2/banked/LATENTS_CTX_{arm}_R3g.npz", allow_pickle=True)
Wn = np.load(f"{W}/probe/imag_in/LATENTS_CTX_G_s0_R1.npz", allow_pickle=True)
eid, g0 = Wn["eid"].astype(str), Wn["g0"].astype(np.int64)
same_list = bool(np.array_equal(eid, b["eid"].astype(str)) and np.array_equal(g0, b["g0"].astype(np.int64)))
z0b = np.asarray(b["z0"], np.float32); tb = np.asarray(b["true_2s"], np.float32); pb = np.asarray(b["pred_2s"], np.float32)
Ab = np.asarray(b["act_scaled"], np.float32)
out = {"arm": arm, "ckpt": ckpt, "predictor_context": int(ctx), "n": int(len(eid)), "same_window_list": same_list,
       "d_actions_max_abs": float(np.abs(Ab - np.asarray(Wn["act_scaled"], np.float32)).max())}
# (a) rollout function on the banked inputs
R = []
for i in range(0, len(z0b), 256):
    R.append(ID.roll_states(torch, st, torch.as_tensor(z0b[i:i + 256]), torch.as_tensor(Ab[i:i + 256]), ctx)[:, 9].numpy())
R = np.concatenate(R)
out["a_rollout_fn_on_banked_inputs_max_abs"] = float(np.abs(R - pb).max())
# (b) pod states against banked states
Z = ID.assemble_states(torch, f"{W}/probe/states/{arm}_eval.npz", eid, g0).numpy()     # [n, 16, 640]


def cmp(x, y):
    d = x - y
    num = (x * y).sum(1); den = np.sqrt((x * x).sum(1) * (y * y).sum(1)) + 1e-12
    xc, yc = x - x.mean(0), y - y.mean(0)
    r = (xc * yc).sum(0) / (np.sqrt((xc * xc).sum(0) * (yc * yc).sum(0)) + 1e-12)
    return {"max_abs": float(np.abs(d).max()), "rms_diff": float(np.sqrt((d * d).mean())), "rms_state": float(np.sqrt((y * y).mean())),
            "rms_centred_state": float(np.sqrt((yc * yc).mean())), "cosine_mean": float((num / den).mean()), "cosine_min": float((num / den).min()),
            "corr_per_dim_median": float(np.median(r)), "corr_per_dim_min": float(r.min()),
            "share_of_values_above_1e-2": float((np.abs(d) > 1e-2).mean()), "share_of_values_above_1e-1": float((np.abs(d) > 1e-1).mean())}


out["b_start_state_pod_vs_banked"] = cmp(Z[:, 4], z0b)
out["b_future_state_pod_vs_banked"] = cmp(Z[:, 14], tb)
# (c) alignment
d0 = np.stack([((Z[:, f] - z0b) ** 2).mean(1) for f in range(16)], 1)
d1 = np.stack([((Z[:, f] - tb) ** 2).mean(1) for f in range(16)], 1)
out["c_nearest_frame_to_banked_start_is_4_share"] = float((d0.argmin(1) == 4).mean())
out["c_nearest_frame_to_banked_future_is_14_share"] = float((d1.argmin(1) == 14).mean())
out["c_mse_at_right_frame_over_mse_at_neighbour_median"] = float(np.median(d0[:, 4] / np.minimum(d0[:, 3], d0[:, 5])))
# the rollout from the pod start states, against the banked rollout and against the size of the motion
Rp = []
for i in range(0, len(Z), 256):
    Rp.append(ID.roll_states(torch, st, torch.as_tensor(Z[i:i + 256, 4]), torch.as_tensor(Ab[i:i + 256]), ctx)[:, 9].numpy())
Rp = np.concatenate(Rp)
out["rollout_pod_vs_banked"] = cmp(Rp, pb)
out["rms_true_change_2s_pod"] = float(np.sqrt(((Z[:, 14] - Z[:, 4]) ** 2).mean()))
print(json.dumps(out))
