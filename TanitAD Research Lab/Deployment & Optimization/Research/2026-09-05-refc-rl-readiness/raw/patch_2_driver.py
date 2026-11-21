"""Patch 2 (REF-C RL-readiness WP, 2026-09-05): rl_refcv3_min.py — moving-lead reward
context (--lead-mode track), oracle-in-fan readout, --mode humanflag, a verdict
that reads paired_openloop.py's REAL record schema; and the test-number fix.
Run: python patch_2_driver.py C:\\Users\\Admin\\refcv4b_repo"""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()


def patch(rel, old, new, count=1):
    path = os.path.join(ROOT, rel)
    s = open(path, encoding="utf-8").read()
    n = s.count(old)
    assert n == count, f"{rel}: expected {count} match(es), found {n}: {old[:70]!r}"
    s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", rel, "x", count)


# ---- 0. the test numbers (sampled check: a waypoint must land inside r) -----------
patch("stack/tests/test_rl_rewards.py",
      "    # lead 8 m ahead (centre-to-centre), same speed: the gap stays 8 m > r = 2 m\n"
      "    lead_far = torch.stack([8.0 + v * t, torch.zeros_like(t)], dim=-1)\n",
      "    # lead 9 m ahead (centre-to-centre), same speed: the gap stays 9 m > r = 2 m.\n"
      "    # (9, not 8: the static check is SAMPLED at the waypoints 0/5/10/15/20 m, so\n"
      "    # the held-at-t0 lead must sit within r of a sample — 10 m hits 9 m by 1 m.)\n"
      "    lead_far = torch.stack([9.0 + v * t, torch.zeros_like(t)], dim=-1)\n")

D = "stack/scripts/rl_refcv3_min.py"

# ---- 1. constants --------------------------------------------------------------------
patch(D,
      "LEAD_LEN_DEFAULT_M = 4.5     # rewards.py default; per-row size_x is RECORDED, not fed (scalar contract)\n"
      "EXPECT_BASE_STEP = 40284\n",
      "LEAD_LEN_DEFAULT_M = 4.5     # rewards.py default; per-row size_x is RECORDED, not fed (scalar contract)\n"
      "EXPECT_BASE_STEP = 40284\n"
      "#: HOW THE LEAD ENTERS THE REWARD CONTEXT (set from --lead-mode in main()).\n"
      "#:   \"track\"  — the lead agent's OWN 10-sample track (obstacle.offline join,\n"
      "#:              build_lead_block_b1.py: t0 frame, 0.2–2.0 s) resampled onto the\n"
      "#:              0.5 s reward grid; contact, headway and the TTC veto are\n"
      "#:              TIME-ALIGNED. DEFAULT since 2026-09-05. It is privileged (not an\n"
      "#:              inference input) but it is ANOTHER agent's motion, never the\n"
      "#:              ego's — the preflight proves the ctx is invariant to the ego future.\n"
      "#:   \"static\" — the predecessor's legacy: the first sample held fixed over the\n"
      "#:              horizon. Kept ONLY as the recorded comparison: `--mode humanflag`\n"
      "#:              measures how often it flags the human driver's own future.\n"
      "LEAD_MODE = \"track\"\n"
      "GRID_S = (0.0, 0.5, 1.0, 1.5, 2.0)   # the 5-point reward prefix (origin + 4 slots)\n")

# ---- 2. lead_track() after lead_row() ------------------------------------------------
patch(D,
      "def build_batch(corp, lead, wis, device, *, with_gt: bool):\n",
      "def resample_track(track: np.ndarray, ts_rel: np.ndarray) -> np.ndarray:\n"
      "    \"\"\"[K, 2] lead samples at ``ts_rel`` (0.2..2.0 s) -> [5, 2] at GRID_S. t = 0 is\n"
      "    a linear extrapolation from the first two samples (one 0.2 s step back).\"\"\"\n"
      "    out = np.zeros((len(GRID_S), 2))\n"
      "    for i, t in enumerate(GRID_S):\n"
      "        if t < ts_rel[0]:\n"
      "            slope = (track[1] - track[0]) / (ts_rel[1] - ts_rel[0])\n"
      "            out[i] = track[0] + slope * (t - ts_rel[0])\n"
      "        else:\n"
      "            out[i, 0] = np.interp(t, ts_rel, track[:, 0])\n"
      "            out[i, 1] = np.interp(t, ts_rel, track[:, 1])\n"
      "    return out\n"
      "\n"
      "\n"
      "def lead_track(corp, lead, wi) -> torch.Tensor:\n"
      "    \"\"\"The lead's track on the reward grid, [5, 2] (t0 frame). No lead -> the far\n"
      "    sentinel held over the horizon (CONSTANT per window -> cancels in both advantages).\"\"\"\n"
      "    has, xy, _ln = lead_row(corp, lead, wi)\n"
      "    if not has or lead is None or lead.ts_rel is None:\n"
      "        return torch.tensor([[FAR_LEAD_X_M, 0.0]] * len(GRID_S), dtype=torch.float32)\n"
      "    e_i, t = corp.ds.index[wi]\n"
      "    row = lead.idx.get((corp.clip_ids[e_i], int(t + corp.W - 1 + corp.raw_off)))\n"
      "    track = np.asarray(lead.leads[row], dtype=np.float64)\n"
      "    if not np.all(np.isfinite(track)):\n"
      "        return torch.tensor([[xy[0], xy[1]]] * len(GRID_S), dtype=torch.float32)\n"
      "    return torch.tensor(resample_track(track, lead.ts_rel), dtype=torch.float32)\n"
      "\n"
      "\n"
      "def build_batch(corp, lead, wis, device, *, with_gt: bool):\n")

# ---- 3. load_lead_block carries ts_rel ----------------------------------------------
patch(D,
      "    blk, idx, meta = ARM.ra.load_lead_block_rows(path)\n"
      "    leads = np.asarray(blk[\"leads\"], dtype=np.float64)          # [R, K, 2]\n"
      "    has = np.asarray(blk[\"has_lead\"]).astype(bool).reshape(-1)\n"
      "    lens = np.asarray(blk[\"lead_lens\"], dtype=np.float64).reshape(-1)\n"
      "    return SimpleNamespace(leads=leads, has=has, lens=lens, idx=idx,\n"
      "                           clips={c for (c, _f) in idx}), meta, {\"n_rows\": int(leads.shape[0])}\n",
      "    blk, idx, meta = ARM.ra.load_lead_block_rows(path)\n"
      "    leads = np.asarray(blk[\"leads\"], dtype=np.float64)          # [R, K, 2]\n"
      "    has = np.asarray(blk[\"has_lead\"]).astype(bool).reshape(-1)\n"
      "    lens = np.asarray(blk[\"lead_lens\"], dtype=np.float64).reshape(-1)\n"
      "    raw = np.load(path, allow_pickle=True)\n"
      "    ts_rel = (np.asarray(raw[\"ts_rel_s\"], dtype=np.float64).reshape(-1)\n"
      "              if \"ts_rel_s\" in raw.files else None)\n"
      "    gt_tg = (np.asarray(raw[\"gt_time_gap_min_s\"], dtype=np.float64).reshape(-1)\n"
      "             if \"gt_time_gap_min_s\" in raw.files else None)\n"
      "    return SimpleNamespace(leads=leads, has=has, lens=lens, idx=idx, ts_rel=ts_rel,\n"
      "                           gt_time_gap=gt_tg,\n"
      "                           clips={c for (c, _f) in idx}), meta, {\"n_rows\": int(leads.shape[0])}\n")

# ---- 4. build_batch carries the track ----------------------------------------------
patch(D,
      "         \"lead_xy\": torch.tensor(xy, dtype=torch.float32, device=device),   # [B, 2]\n"
      "         \"lead_len_m\": list(ln), \"wis\": list(wis)}\n",
      "         \"lead_xy\": torch.tensor(xy, dtype=torch.float32, device=device),   # [B, 2]\n"
      "         \"lead_track\": torch.stack([lead_track(corp, lead, wi) for wi in wis]).to(device),  # [B, 5, 2]\n"
      "         \"lead_len_m\": list(ln), \"wis\": list(wis)}\n")

# ---- 5. reward_ctx: the two lead models ----------------------------------------------
patch(D,
      "    B = batch[\"v0\"].shape[0]\n"
      "    lead = batch[\"lead_xy\"]                                             # [B, 2]\n"
      "    ctx = {\n"
      "        \"dt\": DT_REWARD_S,\n"
      "        \"v0\": batch[\"v0\"].reshape(B, 1, 1),                             # [B, 1, 1]\n"
      "        \"obstacles\": lead.reshape(B, 1, 1, 1, 2),                       # [B,1,1,K=1,2]\n"
      "        \"lead_path\": lead.reshape(B, 1, 1, 1, 2).expand(B, 1, 1, S5, 2),  # static lead\n"
      "        \"lead_len_m\": LEAD_LEN_DEFAULT_M,\n"
      "    }\n",
      "    B = batch[\"v0\"].shape[0]\n"
      "    lead = batch[\"lead_xy\"]                                             # [B, 2]\n"
      "    ctx = {\"dt\": DT_REWARD_S,\n"
      "           \"v0\": batch[\"v0\"].reshape(B, 1, 1),                          # [B, 1, 1]\n"
      "           \"lead_len_m\": LEAD_LEN_DEFAULT_M}\n"
      "    if LEAD_MODE == \"track\":\n"
      "        if S5 != len(GRID_S):\n"
      "            raise ValueError(f\"track mode scores the {len(GRID_S)}-point prefix, got S5={S5}\")\n"
      "        # MOVING lead, time-aligned; NO static `obstacles` key, so `collision`\n"
      "        # takes rewards._collision's lead_path branch (per-step contact).\n"
      "        ctx[\"lead_path\"] = batch[\"lead_track\"].reshape(B, 1, 1, S5, 2)\n"
      "    else:\n"
      "        ctx[\"obstacles\"] = lead.reshape(B, 1, 1, 1, 2)                   # [B,1,1,K=1,2]\n"
      "        ctx[\"lead_path\"] = lead.reshape(B, 1, 1, 1, 2).expand(B, 1, 1, S5, 2)  # static\n")

# ---- 6. readout: oracle-in-fan --------------------------------------------------------
patch(D,
      "        reach = (fan - bank[None]).norm(dim=-1).mean(dim=(1, 2))        # [B]\n",
      "        reach = (fan - bank[None]).norm(dim=-1).mean(dim=(1, 2))        # [B]\n"
      "        oracle = (fan[..., :N_REWARD_SLOTS, :] - gt[:, None]).norm(dim=-1).mean(dim=-1).min(dim=1).values  # [B] oracle-in-fan 2 s\n")
patch(D,
      "                         \"R_REACH\": float(reach[j]), \"sel_idx\": int(sel[j])})\n",
      "                         \"R_REACH\": float(reach[j]), \"R_ORACLE\": float(oracle[j]),\n"
      "                         \"sel_idx\": int(sel[j])})\n")
patch(D, "(\"R1\", \"R2\", \"R3\", \"R_FAN\", \"R_REACH\")", "(\"R1\", \"R2\", \"R3\", \"R_FAN\", \"R_REACH\", \"R_ORACLE\")", count=2)
patch(D,
      "    \"\"\"R1 composed DEFAULT reward · R2 fan collision (lead-only) · R3 sel-ADE 2 s ·\n"
      "    R_FAN endpoint spread · R_REACH mean |fan - bank| · sel_idx per window.\n",
      "    \"\"\"R1 composed DEFAULT reward · R2 fan collision (lead-only) · R3 sel-ADE 2 s ·\n"
      "    R_FAN endpoint spread · R_REACH mean |fan - bank| · R_ORACLE oracle-in-fan ADE 2 s\n"
      "    (the fan-QUALITY readout the echo gate needs: an echo arm improves it while\n"
      "    R_FAN collapses) · sel_idx per window.\n")

# ---- 7. preflight records the lead mode + provenance -----------------------------
patch(D,
      "    rep[\"model\"] = {\"ckpt\": a.ckpt, \"step\": prov.get(\"step\"), \"arm\": prov.get(\"arm\"),\n",
      "    rep[\"lead_mode\"] = LEAD_MODE\n"
      "    rep[\"lead_path_provenance\"] = (\n"
      "        \"track: the lead AGENT's own obstacle.offline positions at t0+0.2..2.0 s in the \"\n"
      "        \"ego t0 frame (build_lead_block_b1.py), resampled to the 0.5 s grid — privileged, \"\n"
      "        \"NOT derived from the ego future (proved below by permutation)\"\n"
      "        if LEAD_MODE == \"track\" else\n"
      "        \"static: the lead's FIRST sample (t0+0.2 s) held over the horizon — LEGACY\")\n"
      "    rep[\"model\"] = {\"ckpt\": a.ckpt, \"step\": prov.get(\"step\"), \"arm\": prov.get(\"arm\"),\n")

# ---- 8. --mode humanflag + the verdict rewrite ------------------------------------
OLD_VERDICT_HEAD = "def mode_verdict(a) -> int:\n"
NEW_HUMANFLAG = '''def _human_future(corp, wi):
    """The logged ego 2 s future in the t0 frame, [1, 1, 1, 5, 2] (fan-shaped), and v0.
    Poses only — no frame decode — via the SAME waypoint_targets the reg_echo arm uses."""
    e_i, t = corp.ds.index[wi]
    poses = corp.ds.episodes[e_i].poses
    T = int(poses.shape[0])
    t0 = t + corp.W - 1
    idx = torch.arange(t0 + 1, t0 + 1 + 60).clamp(max=T - 1)
    fut = poses[idx].float()[None]
    pose_last = poses[t0].float()[None]
    gt8 = refb_labels.waypoint_targets(pose_last, fut, list(ARM_HORIZONS))
    return with_origin(gt8[:, :N_REWARD_SLOTS]).reshape(1, 1, 1, 5, 2), float(pose_last[0, 3])


def mode_humanflag(a) -> int:
    """⛔ THE H-RL-THRESH-1 CHECK, BEFORE ANY ARM: does the reward's scene context flag
    the HUMAN DRIVER'S OWN FUTURE? Scores the logged 2 s future, the hold-v0 straight
    path (the `ha0` floor) and a frozen path under BOTH lead models on every RL-fit
    window with a lead. PASS iff, under the ACTIVE mode, collision and the TTC veto
    each fire on the human on <= --human-flag-max of windows (THRESHOLD_CALIBRATION's
    design band: ~0.05-0.15 doing its job, > 0.25 mis-calibrated). Zero GPU."""
    global LEAD_MODE
    device = "cpu"
    _model, cfg, _targs, prov = load(a, device)
    corp = open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, linfo = load_lead_block(a.lead_block)
    if lead is None or lead.ts_rel is None:
        raise SystemExit("[humanflag] ⛔ a lead block with ts_rel_s is required")
    spec = RewardSpec(weights=dict(DEFAULT_WEIGHTS), dt=DT_REWARD_S)
    wis = scoreable_windows(corp, lead)
    rows = []
    active = LEAD_MODE
    for wi in wis:
        has, xy, _ln = lead_row(corp, lead, wi)
        if not has:
            continue
        e_i, t = corp.ds.index[wi]
        row = lead.idx.get((corp.clip_ids[e_i], int(t + corp.W - 1 + corp.raw_off)))
        human, v0 = _human_future(corp, wi)
        xs = torch.tensor([v0 * s for s in GRID_S], dtype=torch.float32)
        hold = torch.stack([xs, torch.zeros_like(xs)], dim=-1).reshape(1, 1, 1, 5, 2)
        frozen = torch.zeros(1, 1, 1, 5, 2)
        batch = {"v0": torch.tensor([v0]), "lead_xy": torch.tensor([xy], dtype=torch.float32),
                 "lead_track": lead_track(corp, lead, wi)[None]}
        rec = {"wi": int(wi), "clip": corp.clip_ids[e_i], "eid": int(e_i), "v0": v0,
               "gt_time_gap_min_s": (float(lead.gt_time_gap[row])
                                     if lead.gt_time_gap is not None else float("nan"))}
        for mode in ("static", "track"):
            LEAD_MODE = mode
            ctx = reward_ctx(batch, S5=5)
            for name, traj in (("human", human), ("hold_v0", hold), ("frozen", frozen)):
                parts = spec.per_component(traj, ctx)
                r = {k: float(v.reshape(-1)[0]) for k, v in parts.items()}
                r["composed"] = float(spec(traj, ctx).reshape(-1)[0])
                r["ttc_veto"] = bool(RW.ttc_violation(traj, {**ctx, "ttc_min_s": 1.5}).reshape(-1)[0])
                r["collision_fired"] = r["collision"] < 0
                rec[f"{name}_{mode}"] = r
        rows.append(rec)
    LEAD_MODE = active
    n = len(rows)

    def rate(sub, key, pred=bool):
        v = [pred(r[sub][key]) for r in rows]
        return {"rate": float(np.mean(v)) if v else float("nan"), "n": len(v)}

    def mean(sub, key):
        v = [r[sub][key] for r in rows]
        return float(np.mean(v)) if v else float("nan")

    summary = {}
    for mode in ("static", "track"):
        h = f"human_{mode}"
        summary[mode] = {
            "collision_fires_on_human": rate(h, "collision_fired"),
            "ttc_veto_fires_on_human": rate(h, "ttc_veto"),
            "headway_below_0.5_on_human": rate(h, "headway", lambda v: v < 0.5),
            "headway_mean_on_human": mean(h, "headway"),
            "composed_mean": {"human": mean(h, "composed"),
                              "hold_v0": mean(f"hold_v0_{mode}", "composed"),
                              "frozen": mean(f"frozen_{mode}", "composed")},
            "hold_v0_scores_at_least_human": {
                "rate": float(np.mean([r[f"hold_v0_{mode}"]["composed"] >= r[h]["composed"]
                                       for r in rows])) if rows else float("nan"), "n": n},
            "frozen_scores_at_least_human": {
                "rate": float(np.mean([r[f"frozen_{mode}"]["composed"] >= r[h]["composed"]
                                       for r in rows])) if rows else float("nan"), "n": n}}
    by_tg = {}
    for lo, hi in ((0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)):
        sel = [r for r in rows if lo <= r["gt_time_gap_min_s"] < hi]
        if sel:
            by_tg[f"[{lo},{hi if hi < 1e9 else 'inf'})"] = {
                "n": len(sel),
                "static_collision": float(np.mean([r["human_static"]["collision_fired"] for r in sel])),
                "static_ttc_veto": float(np.mean([r["human_static"]["ttc_veto"] for r in sel])),
                "track_collision": float(np.mean([r["human_track"]["collision_fired"] for r in sel])),
                "track_ttc_veto": float(np.mean([r["human_track"]["ttc_veto"] for r in sel]))}
    act = summary[active]
    worst = max(act["collision_fires_on_human"]["rate"], act["ttc_veto_fires_on_human"]["rate"])
    out = {"_what": "human-future flag rates under the RL reward's scene context, static vs track lead",
           "_evidence_class": "MEASURED (ours)",
           "_tier": "instrument probe, T0, NON-PARITY RL-fit clips",
           "active_lead_mode": active, "human_flag_max": float(a.human_flag_max),
           "n_scoreable_windows": len(wis), "n_lead_windows_scored": n,
           "n_episodes": len({r["eid"] for r in rows}), "reward_weights": dict(DEFAULT_WEIGHTS),
           "dt_s": DT_REWARD_S, "grid_s": list(GRID_S),
           "summary": summary, "by_human_time_gap": by_tg,
           "PASS": bool(n > 0 and worst <= float(a.human_flag_max)),
           "per_window": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    _p(f"[humanflag] {'PASS' if out['PASS'] else 'FAIL'} under lead_mode={active}: "
       f"collision fires on the human {act['collision_fires_on_human']['rate']:.3f}, "
       f"TTC veto {act['ttc_veto_fires_on_human']['rate']:.3f} (n={n}, max {a.human_flag_max}); "
       f"static: {summary['static']['collision_fires_on_human']['rate']:.3f} / "
       f"{summary['static']['ttc_veto_fires_on_human']['rate']:.3f} -> {a.out}")
    return 0 if out["PASS"] else 1


def _fam_rows(rec: dict) -> dict:
    """paired_openloop.py's record -> {metric_key: {family, delta, lo, hi, separated,
    lower_is_better}}. The record's shape is rec['families'][FAMILY]['metrics'][KEY]
    = {delta, lo, hi, separated, verdict, ...}; direction is (B-floor)-(A-floor), so
    NEGATIVE favours B (the arm) on lower-is-better metrics."""
    lib = {mk: bool(l) for mk, _f, l, _u in PAIRED.TRAJ_METRICS}
    rows = {}
    for fam, blk in (rec.get("families") or {}).items():
        for mk, r in ((blk or {}).get("metrics") or {}).items():
            if isinstance(r, dict) and r.get("delta") is not None:
                rows[mk] = {"family": fam, "delta": float(r["delta"]), "lo": r.get("lo"),
                            "hi": r.get("hi"), "separated": bool(r.get("separated")),
                            "lower_is_better": lib.get(mk, True)}
    return rows


def mode_verdict(a) -> int:
'''
patch(D, OLD_VERDICT_HEAD, NEW_HUMANFLAG)

OLD_EXIT = '''    out["void"] = void
    if void:
        out["exit"] = "VOID"
    else:
        rl = arms["rl"]
        fan = rl["readout_deltas_paired"]["R_FAN"]
        base_fan = float(J(os.path.join(a.out_dir, "rl", "readout_before.json"))["aggregate"]["R_FAN"])
        rl_fan_ok = not (fan["sep"] and fan["delta"] < 0 and abs(fan["delta"]) >= 0.15 * base_fan)
        pr = pairs.get("rl", {})
        fam = pr.get("families_paired_deltas", {})       # filled by the chain's pairing step
        ade = fam.get("ade_2s", {})
        ade_guard_ok = not (ade.get("lo", 0.0) > 0.02)
        improved = [k for k, v in fam.items() if k != "ade_2s" and v.get("sep")
                    and v.get("better_is") == "lower" and v.get("delta", 0) < 0]
        osel = fam.get("oracle_sel_ade_2s", {})
        fan_quality_worse = bool(osel.get("sep") and osel.get("delta", 0) > 0)
        out["rl"] = {"fan_delta": fan, "G_FAN_ok": rl_fan_ok, "ade_guard_ok": ade_guard_ok,
                     "families_improved_separated": improved, "fan_quality_worse": fan_quality_worse}
        if not rl_fan_ok:
            out["exit"] = "FAIL-COLLAPSE — the RL arm collapsed its own fan"
        elif not ade_guard_ok:
            out["exit"] = "FAIL-GUARD — ADE_2s degraded beyond +0.02 m with separation"
        elif fan_quality_worse:
            out["exit"] = "FAIL-FAN — the oracle-selected (fan-quality) arm got worse"
        elif improved and not fan_quality_worse:
            out["exit"] = f"PASS — RL improved {improved} with paired separation under the guards"
        elif fam.get("os_ade_2s", {}).get("sep") and fam["os_ade_2s"]["delta"] < 0 and not \\
                (osel.get("sep") and osel.get("delta", 0) < 0):
            out["exit"] = "REJECT-SELECTOR — the selected path improved but the fan did not (DDv2 defect)"
        else:
            out["exit"] = "NULL — no family moved with separation; RL stage inert on this base"
'''
NEW_EXIT = '''    # the ECHO SIGNATURE on the regression arm, recorded: oracle-in-fan IMPROVES while
    # the spread collapses — an ADE-only eval would call that arm a win.
    if reg is not None:
        out["reg_echo_fan"]["R_ORACLE_delta"] = reg["readout_deltas_paired"].get("R_ORACLE")
        out["reg_echo_fan"]["R3_delta"] = reg["readout_deltas_paired"].get("R3")
    pr = pairs.get("rl") or {}
    if pr.get("void"):
        void.append(f"V3: the paired record for `rl` is VOID: {pr.get('void_reasons')}")
    out["void"] = void
    if void:
        out["exit"] = "VOID"
    else:
        rl = arms["rl"]
        fan = rl["readout_deltas_paired"]["R_FAN"]
        base_fan = float(J(os.path.join(a.out_dir, "rl", "readout_before.json"))["aggregate"]["R_FAN"])
        rl_fan_ok = not (fan["sep"] and fan["delta"] < 0 and abs(fan["delta"]) >= 0.15 * base_fan)
        fam = _fam_rows(pr)                              # paired_openloop.py's real schema
        ade = fam.get("ade_m", {})
        ade_guard_ok = not (ade.get("lo") is not None and float(ade["lo"]) > 0.02)
        improved = [k for k, v in fam.items() if v["family"] != "ADE" and v["separated"]
                    and ((v["delta"] < 0) if v["lower_is_better"] else (v["delta"] > 0))]
        worsened = [k for k, v in fam.items() if v["family"] != "ADE" and v["separated"]
                    and ((v["delta"] > 0) if v["lower_is_better"] else (v["delta"] < 0))]
        osel = rl["readout_deltas_paired"].get("R_ORACLE", {})   # oracle-in-fan, T0 readout, paired
        fan_quality_worse = bool(osel.get("sep") and osel.get("delta", 0) > 0)
        fan_quality_better = bool(osel.get("sep") and osel.get("delta", 0) < 0)
        ade_better = bool(ade.get("separated") and ade.get("delta", 0) < 0)
        out["rl"] = {"fan_delta": fan, "G_FAN_ok": rl_fan_ok, "ade_guard_ok": ade_guard_ok,
                     "ade_m_paired": ade, "families_improved_separated": improved,
                     "families_worsened_separated": worsened,
                     "oracle_in_fan_delta": osel, "fan_quality_worse": fan_quality_worse,
                     "n_family_metrics_read": len(fam)}
        if not fam:
            out["exit"] = ("VOID — the paired record carries no family metrics "
                           "(schema mismatch); nothing is decidable")
        elif not rl_fan_ok:
            out["exit"] = "FAIL-COLLAPSE — the RL arm collapsed its own fan"
        elif not ade_guard_ok:
            out["exit"] = "FAIL-GUARD — ADE degraded beyond +0.02 m with separation"
        elif fan_quality_worse:
            out["exit"] = "FAIL-FAN — oracle-in-fan (fan quality) got worse with separation"
        elif improved and not worsened:
            out["exit"] = f"PASS — RL improved {improved} with paired separation under the guards"
        elif improved and worsened:
            out["exit"] = f"SPLIT — improved {improved} but worsened {worsened}; no verdict, report per family"
        elif ade_better and not fan_quality_better:
            out["exit"] = "REJECT-SELECTOR — the selected path improved but the fan did not (DDv2 defect)"
        else:
            out["exit"] = "NULL — no family moved with separation; RL stage inert on this base"
'''
patch(D, OLD_EXIT, NEW_EXIT)

# ---- 9. main: the new mode + --lead-mode ----------------------------------------------
patch(D,
      "    ap.add_argument(\"--mode\", required=True, choices=(\"preflight\", \"fitlist\", \"arm\", \"verdict\"))\n",
      "    ap.add_argument(\"--mode\", required=True,\n"
      "                    choices=(\"preflight\", \"fitlist\", \"arm\", \"verdict\", \"humanflag\"))\n"
      "    ap.add_argument(\"--lead-mode\", choices=(\"track\", \"static\"), default=\"track\",\n"
      "                    help=\"how the lead enters the reward ctx (see LEAD_MODE)\")\n"
      "    ap.add_argument(\"--human-flag-max\", type=float, default=0.15,\n"
      "                    help=\"humanflag PASS ceiling on the human-flag rate under the active mode\")\n")
patch(D,
      "    a = ap.parse_args(argv)\n"
      "    torch.manual_seed(int(a.seed))\n"
      "    return {\"preflight\": mode_preflight, \"fitlist\": mode_fitlist,\n"
      "            \"arm\": mode_arm, \"verdict\": mode_verdict}[a.mode](a)\n",
      "    a = ap.parse_args(argv)\n"
      "    global LEAD_MODE\n"
      "    LEAD_MODE = a.lead_mode\n"
      "    torch.manual_seed(int(a.seed))\n"
      "    return {\"preflight\": mode_preflight, \"fitlist\": mode_fitlist,\n"
      "            \"arm\": mode_arm, \"verdict\": mode_verdict,\n"
      "            \"humanflag\": mode_humanflag}[a.mode](a)\n")

# ---- 10. docstring: the mode list --------------------------------------------------------
patch(D,
      "  --mode fitlist     the RL-fit clip list (train-split ids NOT in the eval split) → txt\n",
      "  --mode fitlist     the RL-fit clip list (train-split ids NOT in the eval split) → txt\n"
      "  --mode humanflag   ⛔ the H-RL-THRESH-1 check: how often the reward's own scene\n"
      "                     context flags the HUMAN driver's future (static vs track lead),\n"
      "                     hold-v0 and frozen references, PASS/FAIL at --human-flag-max. 0 GPU.\n")
print("patch 2 complete")
