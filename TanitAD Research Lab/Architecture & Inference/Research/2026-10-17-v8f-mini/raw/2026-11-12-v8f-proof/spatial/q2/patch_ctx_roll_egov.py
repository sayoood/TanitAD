"""patch_ctx_roll_egov.py -- one-off edit of ctx_roll.py for the EGOV arms of card PREREG_V8F_PROOF_ADD19 (control e1: a non-EGOV arm must read the same).
Keeps the earlier file as ctx_roll_pre_egov.py.  Every replacement must match exactly once."""
import io
import shutil

P = "ctx_roll.py"
s = io.open(P, encoding="utf-8").read()
shutil.copyfile(P, "ctx_roll_pre_egov.py")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("    return torch, st, info\n",
    "    info[\"l1_action_dim\"] = int(st.cfg[0].action_dim)\n"
    "    info[\"wm_speed_scale\"] = float(getattr(M, \"WM_SPEED_SCALE\", 0.0))\n"
    "    return torch, st, info\n")
rep("    ap.add_argument(\"--width\", type=int, default=640)\n",
    "    ap.add_argument(\"--width\", type=int, default=640)\n"
    "    ap.add_argument(\"--speed-npz\", default=None, help=\"EGOV arms (card ADD19): SPEED_WINDOWS.npz of collect_speed.py, the logged speed of every frame\")\n")
rep("    assert A.shape[1] == FRAMES - 1, tuple(A.shape)\n",
    "    assert A.shape[1] == FRAMES - 1, tuple(A.shape)\n"
    "    egov = None\n"
    "    if info[\"l1_action_dim\"] == 3:      # EGOV arm (card ADD19, note N1): third action channel = the logged speed of the START frame of each transition over the scale\n"
    "        if not a.speed_npz:\n"
    "            print(\"REFUSED: the level-1 predictor of this arm takes three action channels; give --speed-npz\")\n"
    "            return 3\n"
    "        Sp = np.load(a.speed_npz, allow_pickle=True)\n"
    "        rws = np.asarray(rows, np.int64)\n"
    "        if Sp[\"V\"].shape[0] != Zall.shape[0]:\n"
    "            print(\"REFUSED: the speed file holds %d windows, the state file %d\" % (Sp[\"V\"].shape[0], Zall.shape[0]))\n"
    "            return 3\n"
    "        same = bool(np.array_equal(Sp[\"eid\"].astype(str)[rws], np.asarray(Lw[\"eid\"]).astype(str)) and np.array_equal(np.asarray(Sp[\"t\"])[rws], np.asarray(Lw[\"t\"])))\n"
    "        Vw = torch.as_tensor(np.asarray(Sp[\"V\"], np.float32)[rws][:, :FRAMES - 1]) / float(info[\"wm_speed_scale\"])\n"
    "        acc = np.asarray(Sp[\"A_raw\"], np.float64)[rws][:, :FRAMES - 1, 0]\n"
    "        dv = np.diff(np.asarray(Sp[\"V\"], np.float64)[rws], axis=1)\n"
    "        corr = float(np.corrcoef(dv.ravel(), acc.ravel())[0, 1])\n"
    "        corr_shift = float(np.corrcoef(dv[:, 1:].ravel(), acc[:, :-1].ravel())[0, 1])\n"
    "        egov = {\"windows_identical\": same, \"scale\": float(info[\"wm_speed_scale\"]), \"corr_speed_change_with_logged_acceleration\": corr,\n"
    "                \"same_with_a_one_frame_shift\": corr_shift, \"corr_bar\": SPEED_CORR_BAR,\n"
    "                \"pass\": bool(same and float(info[\"wm_speed_scale\"]) == 10.0 and corr >= SPEED_CORR_BAR)}\n"
    "        A = torch.cat([A, Vw[..., None]], -1)                               # [n, 15, 3]\n"
    "    elif info[\"l1_action_dim\"] != 2:\n"
    "        print(\"REFUSED: level-1 action width %d\" % info[\"l1_action_dim\"])\n"
    "        return 3\n")
rep("        o3 = torch.cat(chunked(lambda sl: pred(Z[sl, S0 - ctx + 1:S0 + 1], e[sl, S0 - ctx + 1:S0 + 1])[:, -1], n))\n",
    "        o3 = torch.cat(chunked(lambda sl: pred(Z[sl, S0 - ctx + 1:S0 + 1], e[sl, S0 - ctx + 1:S0 + 1])[:, -1], n))\n"
    "        o1s = None\n"
    "        if egov is not None:               # control e2 of card ADD19: the speed channel shuffled across windows (fixed seed) must raise the one-step error\n"
    "            pm = torch.randperm(n, generator=torch.Generator().manual_seed(11))\n"
    "            As = A.clone()\n"
    "            As[:, :, 2] = A[pm][:, :, 2]\n"
    "            es = st.act_embed(As)\n"
    "            o1s = torch.cat(chunked(lambda sl: pred(Z[sl, S0:S0 + 1], es[sl, S0:S0 + 1])[:, -1], n))\n")
rep("    one[\"ratio_context_%d_over_copy\" % ctx] = one[\"context_%d\" % ctx] / one[\"copy\"]\n",
    "    one[\"ratio_context_%d_over_copy\" % ctx] = one[\"context_%d\" % ctx] / one[\"copy\"]\n"
    "    if egov is not None:\n"
    "        e2 = mse(o1s, nxt)\n"
    "        egov[\"e2_one_step_mse_speed_shuffled\"] = e2\n"
    "        egov[\"e2_ratio_shuffled_over_true_speed\"] = e2 / one[\"context_1\"]\n"
    "        egov[\"e2_pass\"] = bool(e2 > one[\"context_1\"])\n")
rep("f0_R1=pw(r0, Z[:, K]))\n", "f0_R1=pw(r0, Z[:, K]), **({\"step_c1_shuf\": pw(o1s, nxt)} if egov is not None else {}))\n")
rep("    ok = all(v[\"pass\"] for v in controls.values())\n",
    "    if egov is not None:\n"
    "        controls[\"x5_speed_file\"] = egov\n"
    "    ok = all(v[\"pass\"] for v in controls.values())\n")
rep("act_scaled=A[:, S0:S0 + K].numpy())", "act_scaled=A[:, S0:S0 + K, :2].numpy())")
rep("act_scaled=A.numpy(), **ident)", "act_scaled=A[..., :2].numpy(), **ident)")
rep("FAMILIES = ", "SPEED_CORR_BAR = 0.5   # x5: the change of the logged speed between two frames against the logged acceleration of that transition (alignment of the speed file)\nFAMILIES = ")
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("patched ctx_roll.py")
