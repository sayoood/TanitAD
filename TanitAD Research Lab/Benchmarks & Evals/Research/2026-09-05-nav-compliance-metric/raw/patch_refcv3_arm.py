"""Apply the ADDITIVE nav-compliance edits to taniteval/tools/refcv3_arm.py.

Exact-match anchors; each must occur exactly once, else refuse. Writes the
patched text to the path given as argv[1] (read from argv[1] too). LF line
endings preserved. Idempotent: refuses if the marker is already present.
"""
import sys

MARK = "NAV-COMPLIANCE sidecar"
path = sys.argv[1]
src = open(path, encoding="utf-8", newline="").read()
assert "\r\n" not in src, "refcv3_arm.py is LF; refusing to touch a CRLF copy"
if MARK in src:
    raise SystemExit("already patched")

edits = []

# ---- (1) sidecar doc ------------------------------------------------------ #
edits.append((
    '''    "ha_controls": "the held (a, channel-1) actually integrated, in the run's "
                   "declared action units",
}
''',
    '''    "ha_controls": "the held (a, channel-1) actually integrated, in the run's "
                   "declared action units",
    # ⭐ NAV-COMPLIANCE sidecar (taniteval.nav_compliance, 2026-09-05) — the
    # BEHAVIOURAL readouts per conditioning. The old strategic metric scored a
    # HEAD against a label that is a bijection of the fed token; these are what
    # the model DOES, and a path cannot be produced by copying a token.
    "plan_full_{cond}": "[S, 2] the emitted plan out['traj'] at EVERY model slot "
                        "(not the dump grid) under each conditioning",
    "gstr_{cond}": "[3] the strategic goal g_str = (cos, sin, dist_pref) (hier only)",
    "sel_idx_{cond}": "the SELECTED anchor under each conditioning",
    "sel_bank_{cond}": "[S, 2] the selected anchor's UNREFINED bank path "
                       "(out['anchor_bank'][sel_idx]) — the selection surface",
    "fan_term_heading_{cond}": "[N_anchors] terminal heading of every REFINED "
                               "candidate (out['anchor_traj']) — fan coverage",
    "reach_keep_{cond}": "[N_anchors] the S2 reach mask (1.0 = survivor)",
    "gt_future_ext": "[60, 4] the recorded future poses (WORLD frame), clamped at "
                     "the episode end; gt_future_valid_ext [60] says where",
    "pose_last": "[4] (x, y, yaw, v) at t0 (world frame)",
    "ego_t0": "[4] (v0, a_long, yaw_rate, curvature) at t0 — "
              "refc_v3.ego_state_at_t0 without the keep bit",
    "ep_poses": "[T, 4] the episode's PROVIDER poses, once per file (for the "
                "label time-base control)",
}
'''))

# ---- (2) import + ego-state in run_dump ----------------------------------- #
edits.append((
    '''    t_start = time.time()
    tr = trainer()
    dev = a.device
    model, cfg, targs, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
''',
    '''    t_start = time.time()
    tr = trainer()
    from tanitad.refs import refc_v3 as v3mod
    dev = a.device
    model, cfg, targs, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
    # ⭐ REF-C v4 (E11'): a build with `ego_state_inject` TRAINED with the measured
    # ego block on every row (keep drawn at ego_dropout) and the trainer's own
    # eval feeds it with keep = 1. Calling such a build WITHOUT the block hands
    # the goal path nothing while the core still sees v0 — a regime the model
    # never saw. So the block is fed here exactly as `compute_losses_v3` feeds it
    # (`v3.ego_state_from_batch`, the OBSERVED window only). A v3 build
    # (`ego_state_inject` False) is untouched: ego_state=None as before.
    feed_ego = bool(getattr(cfg, "ego_state_inject", False))
    if feed_ego:
        _p("[ego] ego_state_inject build: feeding the MEASURED ego block "
           "(v0, a_long, yaw_rate, curvature, keep=1) at t0 to every forward — "
           "refc_v3.ego_state_from_batch, the trainer's own entry")
'''))

# ---- (2b) the OPTIONAL nav-FLIP arm (--with-navflip): left <-> right --------- #
edits.append((
    '''ARM_MEANING = {
''',
    '''ARM_MEANING = {
    "os_navflip": "T1 (RULING OPEN) — as os with the nav token FLIPPED "
                  "(left<->right) on every window: the sharpest nav "
                  "intervention for taniteval.nav_compliance. OPTIONAL "
                  "(--with-navflip); shuffle + zero stay the REQUIRED controls.",
'''))

edits.append((
    '''    conds_fed = ["nav_true"] + (["nav_shuffled"] if "os_navshuf" in arms else [])
''',
    '''    conds_fed = ["nav_true"] + (["nav_shuffled"] if "os_navshuf" in arms else [])
    # ⭐ --with-navflip (2026-09-05, harvested from the predecessor's draft): the
    # SHARPEST nav intervention — left <-> right, so the fed token disagrees
    # with the truth on EVERY commanded window (the shuffle feeds `follow` on
    # most of them). OPTIONAL and default OFF: the two REQUIRED controls stay
    # shuffle + zero (BACKLOG R39) and a banked dump's arm list is unchanged.
    nav_flip = None
    if nav_on and getattr(a, "with_navflip", False):
        nav_flip = nav_true.copy()
        nav_flip[nav_true == refb_labels.NAV_LEFT] = refb_labels.NAV_RIGHT
        nav_flip[nav_true == refb_labels.NAV_RIGHT] = refb_labels.NAV_LEFT
        conds_fed.append("nav_flipped")
        arms.append("os_navflip")
        _p(f"[nav-flip] os_navflip: left<->right on {int((nav_flip != nav_true).sum())}"
           f"/{len(nav_true)} windows — the sharpest intervention (optional arm)")
'''))

edits.append((
    '''            nav_vals = {"nav_true": int(nav_true[i]),
                        "nav_shuffled": int(nav_shuf[i])}
''',
    '''            nav_vals = {"nav_true": int(nav_true[i]),
                        "nav_shuffled": int(nav_shuf[i])}
            if nav_flip is not None:
                nav_vals["nav_flipped"] = int(nav_flip[i])
'''))

edits.append((
    '''            if "os_navzero" in arms:
                acc["os_navzero"].append(
                    out_z["traj"].float()[0:1, slots].cpu().numpy())
''',
    '''            if "os_navzero" in arms:
                acc["os_navzero"].append(
                    out_z["traj"].float()[0:1, slots].cpu().numpy())
            if "os_navflip" in arms:
                r = conds_fed.index("nav_flipped")
                acc["os_navflip"].append(traj[r:r + 1, slots].cpu().numpy())
'''))

edits.append((
    '''            dec.setdefault("nav_cmd_shuf", []).append(int(nav_shuf[i]))
''',
    '''            dec.setdefault("nav_cmd_shuf", []).append(int(nav_shuf[i]))
            if nav_flip is not None:
                dec.setdefault("nav_cmd_flip", []).append(int(nav_flip[i]))
'''))

edits.append((
    '''ARM_TIERS = {"os": "T1", "os_navshuf": "T1", "os_navzero": "T1", "ha": "T1",
             "ha0": "T1", "oracle_sel": "T0"}
''',
    '''ARM_TIERS = {"os": "T1", "os_navshuf": "T1", "os_navzero": "T1", "ha": "T1",
             "ha0": "T1", "oracle_sel": "T0", "os_navflip": "T1"}
'''))

edits.append((
    '''    ap.add_argument("--with-oracle-sel", action="store_true",
''',
    '''    ap.add_argument("--with-navflip", action="store_true",
                    help="ALSO roll the nav-FLIP arm (left<->right on every "
                         "window): the sharpest nav intervention for the "
                         "nav-compliance metric (taniteval.nav_compliance). "
                         "Optional; shuffle + zero remain the REQUIRED controls.")
    ap.add_argument("--with-oracle-sel", action="store_true",
'''))

edits.append((
    '''            v0_t = torch.full((b,), v0, dtype=torch.float32, device=dev)
            tf = time.time()
            with torch.no_grad():
                out = model(fr_b, nav_cmd=nav_t if nav_on else None,
                            v0=v0_t, steps=steps)
''',
    '''            v0_t = torch.full((b,), v0, dtype=torch.float32, device=dev)
            ego_b = ego_z = None
            if feed_ego:
                _es = v3mod.ego_state_from_batch(
                    {"pose_last": item["pose_last"].float()[None],
                     "actions": item["actions"].float()[None]}, device=dev)
                ego_b = _es.expand(b, -1).contiguous() if b > 1 else _es
                ego_z = _es
            tf = time.time()
            with torch.no_grad():
                out = model(fr_b, nav_cmd=nav_t if nav_on else None,
                            v0=v0_t, steps=steps, ego_state=ego_b)
'''))

edits.append((
    '''                out_z = None
                if do_navzero:
                    out_z = model(fr, nav_cmd=None,
                                  v0=v0_t[:1], steps=steps)
''',
    '''                out_z = None
                if do_navzero:
                    out_z = model(fr, nav_cmd=None,
                                  v0=v0_t[:1], steps=steps, ego_state=ego_z)
'''))

# ---- (3) the sidecar rows -------------------------------------------------- #
edits.append((
    '''            dec.setdefault("ha_controls", []).append(
                hold[None].expand(n_f, 2).float().cpu().numpy()[None])
            ws.append(int(t0))
            n_done += 1
''',
    '''            dec.setdefault("ha_controls", []).append(
                hold[None].expand(n_f, 2).float().cpu().numpy()[None])
            # ⭐ NAV-COMPLIANCE sidecar (taniteval.nav_compliance): the
            # BEHAVIOURAL readouts per conditioning — the emitted plan at every
            # model slot, the strategic goal, the SELECTED vocabulary element
            # (the selection surface, decided at t=0) and the refined fan's
            # terminal headings with the reach mask. Additive keys only.

            def _navcomp_row(o, r, cname):
                dec.setdefault(f"plan_full_{cname}", []).append(
                    o["traj"].float()[r:r + 1].cpu().numpy())             # [1, S, 2]
                if "g_str" in o:
                    dec.setdefault(f"gstr_{cname}", []).append(
                        o["g_str"].float()[r:r + 1].cpu().numpy())        # [1, 3]
                si = int(o["sel_idx"][r])
                dec.setdefault(f"sel_idx_{cname}", []).append(si)
                bank = o["anchor_bank"].float()[r]                          # [N, S, 2]
                dec.setdefault(f"sel_bank_{cname}", []).append(
                    bank[si:si + 1].cpu().numpy())                          # [1, S, 2]
                fan_ = o["anchor_traj"].float()[r]                          # [N, S, 2]
                d_ = fan_[:, -1] - fan_[:, -2]
                th = torch.atan2(d_[:, 1], d_[:, 0])
                th = torch.where(torch.hypot(d_[:, 0], d_[:, 1]) < 0.05,
                                 torch.zeros_like(th), th)
                dec.setdefault(f"fan_term_heading_{cname}", []).append(
                    th[None].cpu().numpy())                                 # [1, N]
                rk = o.get("reach_keep")
                dec.setdefault(f"reach_keep_{cname}", []).append(
                    (rk[r:r + 1].float() if rk is not None
                     else torch.ones(1, fan_.shape[0])).cpu().numpy())     # [1, N]

            for ci_, cname in enumerate(conds_fed):
                _navcomp_row(out, ci_, cname)
            if do_navzero:
                _navcomp_row(out_z, 0, "nav_zero")
            dec.setdefault("gt_future_ext", []).append(
                item["future_poses_ext"].float()[None].cpu().numpy())       # [1, 60, 4]
            dec.setdefault("gt_future_valid_ext", []).append(
                fv.float()[None].cpu().numpy())                             # [1, 60]
            dec.setdefault("pose_last", []).append(
                pose_last[None].cpu().numpy())                              # [1, 4]
            dec.setdefault("ego_t0", []).append(
                v3mod.ego_state_at_t0(
                    pose_last[None, None].expand(1, item["actions"].shape[0], 4),
                    item["actions"].float()[None])[:, :4].cpu().numpy())    # [1, 4]
            ws.append(int(t0))
            n_done += 1
'''))

edits.append((
    '''        np.savez_compressed(
            os.path.join(a.dump_dir, "decisions",
                         f"ep{len(episodes_manifest):03d}.npz"),
            ws=np.array(ws), **dec_np)
''',
    '''        np.savez_compressed(
            os.path.join(a.dump_dir, "decisions",
                         f"ep{len(episodes_manifest):03d}.npz"),
            ws=np.array(ws), ep_poses=ep.poses.float().cpu().numpy(), **dec_np)
'''))

# ---- (4) manifest --------------------------------------------------------- #
edits.append((
    '''        "head_conditionings": conds,
        "fed_conditionings": conds_fed,
        "sidecar_schema": _SIDECAR_DOC,
''',
    '''        "head_conditionings": conds,
        "fed_conditionings": conds_fed,
        "sidecar_schema": _SIDECAR_DOC,
        "ego_state_fed": feed_ego,
        "navcomp_sidecar": True,
'''))

# ---- (5) the analysis block ----------------------------------------------- #
edits.append((
    '''def analyze_refcv3(dump_dir: str, *, n_boot: int = 2000, seed: int = 0,
                   dt: float | None = None, tiers: dict | None = None,
                   lead_block: str | None = None) -> dict:
''',
    '''def analyze_refcv3(dump_dir: str, *, n_boot: int = 2000, seed: int = 0,
                   dt: float | None = None, tiers: dict | None = None,
                   lead_block: str | None = None,
                   labels: str | None = None) -> dict:
'''))

edits.append((
    '''    ref["strategic"] = strat

''',
    '''    # ---- ⭐ NAV-COMPLIANCE: the BEHAVIOURAL strategic metric (PI 2026-09-04:
    # "we are evaluating the fact that the model is following the nav command in
    # consistency to the strategic goals"). Decided by the intervention pair on
    # the same windows, never by a rate; a dump without the sidecar keys gets a
    # REFUSAL with its reason.
    _nc = None
    try:
        from taniteval import nav_compliance as _nc
        strat["nav_compliance"] = _nc.from_refcv3_dump(
            dump_dir, labels_path=labels, n_boot=n_boot, seed=seed)
    except Exception as ex:      # noqa: BLE001 — a refusal, not a crash
        _why = f"nav_compliance did not run: {type(ex).__name__}: {str(ex)[:300]}"
        strat["nav_compliance"] = (_nc.unavailable_block(_why, N) if _nc is not None
                                   else _refused(_why, "T1", N))
    ref["strategic"] = strat

'''))

edits.append((
    '''    rec = analyze_refcv3(dump_dir, n_boot=a.n_boot, seed=a.seed,
                         tiers=t1._parse_tiers(a.tiers), lead_block=lead_path)
''',
    '''    rec = analyze_refcv3(dump_dir, n_boot=a.n_boot, seed=a.seed,
                         tiers=t1._parse_tiers(a.tiers), lead_block=lead_path,
                         labels=a.labels)
'''))

for i, (old, new) in enumerate(edits):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"edit {i}: anchor occurs {n} times (need exactly 1)")
    src = src.replace(old, new)

open(path, "w", encoding="utf-8", newline="").write(src)
print(f"patched {path}: {len(edits)} edits, {src.count(MARK)} marker(s)")
