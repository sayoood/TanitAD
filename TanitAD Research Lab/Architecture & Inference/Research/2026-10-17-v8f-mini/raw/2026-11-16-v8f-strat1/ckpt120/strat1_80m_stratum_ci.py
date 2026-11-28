import sys, json, importlib.util, numpy as np
sys.path.insert(0, "F:/TanitAD-scratch/strat1/read")
import strategic_cells as SC
spec = importlib.util.spec_from_file_location("ci", sys.argv[1]); CI = importlib.util.module_from_spec(spec); spec.loader.exec_module(CI)
RD = "F:/TanitAD-scratch/strat1/read"
J = np.load(RD + "/inputs/join_eval141.npz"); NT = np.load(RD + "/inputs/navtok_table.npz", allow_pickle=True); NX = np.load(RD + "/inputs/navtext_nav.npz", allow_pickle=True)
G = json.load(open(RD + "/inputs/eval_grid_windows.json")); clips = list(J["clip"])
ci = np.array([clips.index(c) for c in G["order"] for _ in G["windows"][c]]); t = np.array([int(x) for c in G["order"] for x in G["windows"][c]])
r = J["off"][ci] + t; sd, th = SC.side_table(NT["vocab"]); cls = NT["eval_cls_id"][r].astype(int); tok = NT["eval_flag"][r] == 0
side = np.where(tok, sd[cls], 0).astype(np.int8); theta = np.where(tok, th[cls], np.nan); d = np.where(side != 0, NX["eval_nav_d"][r].astype(float), np.nan)
on = (NX["eval_nav"][r, 1] + NX["eval_nav"][r, 2]) > 0; rc_ok = J["rc_ok"][r].astype(bool)
out = {}
for arm in ("S1_s0", "S1_s1", "N1_s0", "N1_s1"):
    z = np.load(RD + "/out/t0/eval_STRAT1-%s_real.npz" % arm, allow_pickle=True)
    cell = SC.ckpt_cell(np.asarray(z["ckpt"], float)[..., 2], (30, 50, 80), side, theta, d, tok, rc_ok)[:, 2]
    for nm, m in (("token_on", np.isfinite(cell) & on), ("all", np.isfinite(cell))):
        s = side[m]; maj = 1 if (s == 1).sum() >= (s == -1).sum() else -1; bc = (s == maj).astype(float)
        res = CI.paired_episode_cluster_bootstrap(cell[m], bc, ci[m], n_boot=2000, seed=0)
        out[arm + ":" + nm] = {k: res[k] for k in ("delta", "lo", "hi", "separated")}
print(json.dumps(out))
