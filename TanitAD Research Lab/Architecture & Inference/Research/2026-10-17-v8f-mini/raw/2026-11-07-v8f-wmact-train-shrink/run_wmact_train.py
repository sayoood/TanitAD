"""PREREG_V8F_WMACT, TRAIN-window read: statistics and controls from latents_train.pt. Copy of run_wmact.py; the differences are marked DIFF. Writes RESULT_WMACT_TRAIN.json."""
import json, time, numpy as np, torch
import wmact_lib as W

t0 = time.time()
L = torch.load("latents_train.pt", weights_only=False)                 # DIFF: the 141 streamed TRAIN clips
Le = torch.load("latents.pt", weights_only=False)                      # DIFF: eval-141 latents, only for the shared mean-action constant (card log: "the mean action stays the eval-141 constant")
eid_all, ok_all = np.array(L["eid"]), L["ok"].all(1).numpy()
sel = np.flatnonzero(ok_all)                                           # windows whose 16 frames are all real (start = window frame 0)
eid = eid_all[sel]
uniq, Wt = W.boot_weights(eid)
W.check_draws_equal_ci(eid, uniq, Wt)
n, n_ep = len(sel), len(uniq)
res = {"windows_total": int(L["N"]), "windows_used": int(n), "episodes_used": int(n_ep), "controls": {}, "rows": {}, "sens": {}, "tier": "T0",
       "tanitad_file": L["tanitad_file"], "start_frame": "window frame 0 of the 16-frame training window; windows with all 16 frames real",
       "split": "TRAIN (141 clips, smallest sha256 among the 4,512 M2b trained on; 17 starts per clip)", "clips_sha256": L["clips_sha256"]}
raw_mean = {}
R = lambda es, C, sign: W.ratio_row(es, C, sign, Wt, n, n_ep)
ref = json.load(open("RESULT_WMACT.json"))["action_mean_raw_eval141_a_yaw"]
for tag in ("M2b_s0", "M2b_s1"):
    m, cfg, ck = W.load_model(W.CKPT[tag])
    Z1, A_raw = L["Z1"][tag][sel], L["A_raw"][sel]
    A_sc = m.wm.scale_actions(A_raw)[:, :-1]
    mean_sc, mean_raw = W.mean_action_scaled(m.wm, Le["A_raw"], Le["ok"])    # DIFF: the eval-141 constant, recomputed exactly as the eval read did
    assert [float(x) for x in mean_raw] == ref[tag], (tag, [float(x) for x in mean_raw], ref[tag])
    raw_mean[tag] = [float(x) for x in mean_raw]
    cols = W.model_columns(m.wm, Z1, A_sc, eid, mean_sc)
    perm = cols.pop("perm")
    es = W.ep_sums(cols, eid, uniq)
    rows = {}
    for l, ks in W.KS.items():
        for k in ks:
            C, S = es[f"C/{l}/{k}"], f"/{l}/{k}"
            et = es["err_true" + S]
            row = {"SKILL": R(es, C, {"C" + S: 1.0, "err_true" + S: -1.0}),
                   "ACTION-USE": R(es, C, {"err_donor" + S: 1.0, "err_true" + S: -1.0}),
                   "ACTION-USE-MEAN": R(es, C, {"err_mean" + S: 1.0, "err_true" + S: -1.0}),
                   "CTRL3_SKILL_startdonor": R(es, C, {"C" + S: 1.0, "err_startdonor" + S: -1.0}),
                   "CTRL1_identity_ACTION-USE": R(es, C, {"err_self" + S: 1.0, "err_true" + S: -1.0})}
            if l == 1:
                for nm in ("acc", "yaw"):
                    row[f"ACTION-USE[{nm} swapped to donor]"] = R(es, C, {f"err_donor_{nm}" + S: 1.0, "err_true" + S: -1.0})
                    row[f"ACTION-USE-MEAN[{nm} set to mean]"] = R(es, C, {f"err_mean_{nm}" + S: 1.0, "err_true" + S: -1.0})
            rows[f"L{l}/k{k}"] = row
    res["rows"][tag] = rows
    res["sens"][tag] = W.sens_block(m.wm, Z1, A_sc, eid, perm)
    sid = W.sens_block(m.wm, Z1, A_sc, eid, torch.arange(n))                       # control 1: donor = the window itself
    c = res["controls"]
    c.setdefault("1_identity_sens_max_abs", {})[tag] = max(abs(v) for v in sid.values())
    c.setdefault("1_identity_actionuse_max_abs", {})[tag] = max(abs(r["CTRL1_identity_ACTION-USE"]["point"]) for r in rows.values())
    m2 = W.zero_act_embed(m)                                                         # control 2: level-1 action embedding zeroed
    c2 = W.model_columns(m2.wm, Z1, A_sc, eid, mean_sc); c2.pop("perm")
    e2 = W.ep_sums(c2, eid, uniq)
    c.setdefault("2_zeroed_act_embed", {})[tag] = {
        f"k{k}": {"SKILL": R(e2, e2[f"C/1/{k}"], {f"C/1/{k}": 1.0, f"err_true/1/{k}": -1.0}),
                  "ACTION-USE_point": float(R(e2, e2[f"C/1/{k}"], {f"err_donor/1/{k}": 1.0, f"err_true/1/{k}": -1.0})["point"])} for k in W.KS[1]}
    c["2_zeroed_act_embed"][tag]["sens_L1_max_abs"] = max(abs(v) for kk, v in W.sens_block(m2.wm, Z1, A_sc, eid, perm).items() if kk.startswith("L1/"))
    # control 4: DIFF -- computed during the stream on the first 64 TRAIN windows (collect_latents_train.py:control4, the expressions of run_wmact.py)
    c.setdefault("4_reproduction", {})[tag] = L["ctrl4"][tag]
    print(tag, "done", round(time.time() - t0, 1), flush=True)
res["action_mean_raw_eval141_a_yaw"] = raw_mean
res["note_mean"] = "the eval-141 mean (real cached frames), shared with the eval read, as the card log fixes"
res["wall_s_stats"] = time.time() - t0
res["wall_s_collect"] = float(L["wall_s"])
json.dump(res, open("RESULT_WMACT_TRAIN.json", "w"), indent=1, default=float)
print("wrote RESULT_WMACT_TRAIN.json", round(time.time() - t0, 1))
