import json, numpy as np, os, sys
W = "C:/Users/Admin/t1heads/smoke"
def arrays(d):
    return {f: {k: np.load(os.path.join(d, f))[k] for k in np.load(os.path.join(d, f)).files} for f in sorted(os.listdir(d))}
A = {n: arrays(f"{W}/{n}/dump_SM") for n in ("tip", "off", "heads", "tip2")}
def same(x, y):
    return x.keys() == y.keys() and all(set(x[f]) == set(y[f]) and all(x[f][k].dtype == y[f][k].dtype and x[f][k].shape == y[f][k].shape and x[f][k].tobytes() == y[f][k].tobytes() for k in x[f]) for f in x)
print("episode dump files:", len(A["off"]))
print("dumps bit-identical  tip==off:", same(A["tip"], A["off"]), " off==heads:", same(A["off"], A["heads"]), " tip==tip2:", same(A["tip"], A["tip2"]))
jb = {n: open(f"{W}/{n}/t1v8f_SM.json", "rb").read() for n in ("tip", "off", "heads", "tip2")}
print("result JSON bytes    tip==off:", jb["tip"] == jb["off"], " tip==tip2:", jb["tip"] == jb["tip2"], f"({len(jb['off'])} bytes)")
jh, jo = json.loads(jb["heads"]), json.loads(jb["off"])
h = jh["t1v8f"]; tf, th = h.pop("ticks_file"), h.pop("ticks_heads")
print("heads JSON == off JSON after dropping ticks_file/ticks_heads:", jh == jo, "| ticks_heads:", th)
z = np.load(tf); z = {k: z[k] for k in z.files}
print("ticks keys:", len(z), "N rows:", len(z["tick"]), "= 24 windows x", len(z["tick"]) // 24, "ticks")
for k in ("s_ckpt", "s_ckpt_now", "s_ckpt_w0", "s_pose", "t_goal_prob", "t_lat_logits", "t_lon_logits", "t_cons_pred", "t_pose", "pose_w0"):
    print(f"  {k:13s} {str(z[k].dtype):8s} {z[k].shape!s:12s} finite={bool(np.isfinite(z[k]).all())}")
r0 = z["tick"] == 0
print("tick-0 s_ckpt x per arc (30/50/80 m):", np.round(z["s_ckpt"][r0][:, :, 0].mean(0), 2))
print("tick-0 s_ckpt == s_ckpt_now == s_ckpt_w0 within 1e-4:", np.allclose(z["s_ckpt"][r0], z["s_ckpt_now"][r0], atol=1e-4) and np.allclose(z["s_ckpt"][r0], z["s_ckpt_w0"][r0], atol=1e-4))
print("later ticks: mean |s_ckpt_now - s_ckpt| =", float(np.abs(z["s_ckpt_now"] - z["s_ckpt"])[~r0].mean()), " mean |s_ckpt_w0 - s_ckpt_now| =", float(np.abs(z["s_ckpt_w0"] - z["s_ckpt_now"])[~r0].mean()))
print("ages (age_t, age_s) seen:", sorted({(int(a), int(b)) for a, b in zip(z["age_t"], z["age_s"])}))
print("goal_prob in [0,1]:", bool(((z["t_goal_prob"] >= 0) & (z["t_goal_prob"] <= 1)).all()), "| lat argmax counts:", np.bincount(z["t_lat_logits"].argmax(-1), minlength=5).tolist(), "| lon argmax counts:", np.bincount(z["t_lon_logits"].argmax(-1), minlength=7).tolist())
d = {(int(c), int(t)): ol for f in A["heads"] for c, t, ol in zip(A["heads"][f]["clip_index"], A["heads"][f]["ws"], A["heads"][f]["ol"])}
print("tick-0 plan rows == dump ol (C-T2-style):", all(np.array_equal(z["plan"][i], d[(int(z["ci"][i]), int(z["t0"][i]))]) for i in np.where(r0)[0]))
