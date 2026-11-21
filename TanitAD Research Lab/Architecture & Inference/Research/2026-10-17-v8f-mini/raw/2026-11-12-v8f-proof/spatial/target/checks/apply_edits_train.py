"""Edits of stack/scripts/v8f_mini_train.py (flags, gate reads, G-LIVE terms) and stack/tanitad/data/v8f_mini_data.py (wm_speed). Run once on tree_patch."""
root = "C:/Users/Admin/v8proof_c/target/tree_patch/stack/"


def edit(path, pairs):
    s = open(root + path, encoding="utf-8", newline="").read()
    assert "\r\n" not in s
    for old, new, n in pairs:
        assert s.count(old) == n, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(root + path, "w", encoding="utf-8", newline="").write(s)


edit("scripts/v8f_mini_train.py", [
    ('''"wm_l1_diag": "lever:wm_l1_diag",   # MOTION:''',
     '''"wm_l1_diag": "lever:wm_l1_diag",
    "wm_l1_act": "lever:wm_l1_act", "wm_l1_futtok_k": "lever:wm_l1_futtok_k", "w_wm_futtok": "lever:w_wm_futtok",     # TARGET: EGOV from the BUILT level-1 action embedding width, FUTTOK from the built head and the stack's loss config, see bad_wm_state
    # MOTION:''', 1),
    ('''    ap.add_argument("--wm-dim", dest="wm_dim", type=int, default=128,''',
     '''    ap.add_argument("--wm-l1-act", dest="wm_l1_act", choices=M.WM_ACT_MODES, default="acc_yaw",
                    help="TARGET lever EGOV: level-1 action of the WM auxiliary. acc_yaw (default, every earlier arm bit for bit) = (acceleration, ch1); speed = + the logged ego speed of the start frame / WM_SPEED_SCALE (10 m/s) as a third channel")
    ap.add_argument("--wm-l1-futtok-k", dest="wm_l1_futtok_k", type=int, default=0,
                    help="TARGET lever FUTTOK: K, the rollout length of the token-read term (a per-cell linear head reads the rolled-out states of steps 1..K against the frozen cached token of the same cell at the same frame, layer-normalised, no gradient). 0 (default) = off; else 2 <= K <= hmax - 1, needs --wm-state grid and --w-wm-futtok > 0")
    ap.add_argument("--w-wm-futtok", dest="w_wm_futtok", type=float, default=0.0, help="TARGET lever FUTTOK: W, the weight of the token-read term in the level-1 loss (0.0 exactly when --wm-l1-futtok-k is 0)")
    ap.add_argument("--wm-dim", dest="wm_dim", type=int, default=128,''', 1),
    ('''w_wm_roll=a.w_wm_roll, wm_l1_diag=a.wm_l1_diag,''',
     '''w_wm_roll=a.w_wm_roll, wm_l1_diag=a.wm_l1_diag, wm_l1_act=a.wm_l1_act, wm_l1_futtok_k=a.wm_l1_futtok_k, w_wm_futtok=a.w_wm_futtok,''', 1),
    ('''"wm_l1_residual", "wm_l1_roll_k", "w_wm_roll",        # MOTION:''',
     '''"wm_l1_residual", "wm_l1_roll_k", "w_wm_roll", "wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok",        # TARGET (EGOV / FUTTOK, same rule); MOTION:''', 1),
    ('''"wm_l1_residual", "wm_l1_roll_k", "w_wm_roll", "wm_l1_diag"):''',
     '''"wm_l1_residual", "wm_l1_roll_k", "w_wm_roll", "wm_l1_diag", "wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok"):''', 1),
    ('''    if built.get("wm_l1_diag") != a.wm_l1_diag:
        bad.append(f"lever wm_l1_diag: declared {a.wm_l1_diag} but built {built.get('wm_l1_diag')}")
''', '''    if built.get("wm_l1_diag") != a.wm_l1_diag:
        bad.append(f"lever wm_l1_diag: declared {a.wm_l1_diag} but built {built.get('wm_l1_diag')}")
    # TARGET levers, each read back from the BUILT model: EGOV from the input width of the level-1 action embedding, FUTTOK from the head that was built and the K / W the stack's loss reads
    if built.get("wm_l1_act") != a.wm_l1_act:
        bad.append(f"lever wm_l1_act: declared {a.wm_l1_act} but built {built.get('wm_l1_act')} (read from the input width of the level-1 action embedding: 3 = speed)")
    if built.get("wm_l1_futtok_k") != a.wm_l1_futtok_k:
        bad.append(f"lever wm_l1_futtok_k: declared {a.wm_l1_futtok_k} but the built stack's loss reads {built.get('wm_l1_futtok_k')}")
    if built.get("w_wm_futtok") != a.w_wm_futtok:
        bad.append(f"lever w_wm_futtok: declared {a.w_wm_futtok} but the built stack's loss reads {built.get('w_wm_futtok')}")
    if not ((a.wm_l1_futtok_k == 0 and a.w_wm_futtok == 0.0) or (a.wm_l1_futtok_k >= 2 and a.w_wm_futtok > 0.0 and math.isfinite(a.w_wm_futtok))):
        bad.append(f"wm_l1_futtok_k {a.wm_l1_futtok_k} with w_wm_futtok {a.w_wm_futtok}: need (0, 0.0) or (K >= 2, W > 0): a lever that is declared and does nothing")
    if a.wm_l1_futtok_k > 0 and a.wm_state != "grid":
        bad.append(f"wm_l1_futtok_k {a.wm_l1_futtok_k} needs --wm-state grid (declared {a.wm_state}): the token read is per cell")
''', 1),
    ('''        expect.add("wm_roll")                                            # MOTION lever B: a declared rollout term that never appears is a launch blocker
''', '''        expect.add("wm_roll")                                            # MOTION lever B: a declared rollout term that never appears is a launch blocker
    if cfg.wm_l1_futtok_k > 0:
        expect.add("wm_futtok")                                          # TARGET lever FUTTOK: a declared token term that never appears is a launch blocker
''', 2),
])

edit("tanitad/data/v8f_mini_data.py", [
    ('''              "act": np.stack([np.nan_to_num(d["a"][idc]), d["ch1"][idc]], -1).astype(np.float32),
''', '''              "act": np.stack([np.nan_to_num(d["a"][idc]), d["ch1"][idc]], -1).astype(np.float32),
              "wm_speed": np.nan_to_num(d["v"][idc]).astype(np.float32),     # TARGET lever EGOV: the LOGGED ego speed [m/s] of every cached frame (the cache's `v`); read only when --wm-l1-act speed
''', 1),
    ('''    for k in ("p10f", "act", "nav", "pose",''', '''    for k in ("p10f", "act", "wm_speed", "nav", "pose",''', 1),
])
print("ok")
