import json, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_common as C
import l2_oracle_s2_rows as S2R
st = C.stack(); RS = st["RS"]
P = S2R.load_panel(0); tab = C.deploy_table()
full_bat = RS.retime_plan(np.asarray(P["plan8"], np.float32), np.asarray(P["prior8"], np.float32), C.ALL_T, tab, cast=True)
a = bool(np.array_equal(full_bat[:, :4], np.asarray(P["stage_banked"])))
zb = np.load("C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s0.npz", allow_pickle=False)
Wb = int(json.loads(str(zb["meta_json"]))["window_W"])
kb = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(zb["e_i"], zb["t"].astype(np.int64) + Wb - 1))}
jb = np.array([kb.get((int(c), int(w)), -1) for c, w in zip(P["clip"], P["ws"])])
os_plan = zb["traj_own"][jb]
full_os = RS.retime_plan(os_plan.astype(np.float32), np.asarray(P["prior8"], np.float32), C.ALL_T, tab, cast=True)
d_plan = np.abs(os_plan - np.asarray(P["plan8"])).reshape(len(jb), -1).max(1)
d_stage = np.abs(full_os[:, :4] - np.asarray(P["stage_banked"])).reshape(len(jb), -1).max(1)
print("ZZDIAG (a) stage(battery plan8) == stage_banked:", a)
print("ZZDIAG joined", int((jb >= 0).sum()), "of", len(jb))
print("ZZDIAG (b) os-only plan != battery plan on", int((d_plan > 1e-6).sum()), "windows; staged diff on", int((d_stage > 1e-6).sum()), "; max staged diff m %.3f" % d_stage.max())
print("ZZDIAG (c) staged equal where plans equal:", bool((d_stage[d_plan <= 1e-6] <= 1e-6).all()))
