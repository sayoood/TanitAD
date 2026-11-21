"""Apply the X10 hooks to the lander-tip refc_v3_train.py / declared_vs_built.py (CRLF-preserving) and write the
result under code/fix/.  Refuses if a marker is already present (i.e. the base is not the tip).

    python make_trainer_patch.py            # reads agent/arch-inf-20260803 from C:/Users/Admin/tanitad-push/.git
"""
import os
import subprocess

PK = "D:/Projects/TanitAD/FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing"
GD = "C:/Users/Admin/tanitad-push/.git"
TIP = "agent/arch-inf-20260803"


def tip_bytes(path):
    return subprocess.run(["git", "--git-dir", GD, "show", f"{TIP}:{path}"], capture_output=True, check=True).stdout


def tip_blob(path):
    return subprocess.run(["git", "--git-dir", GD, "rev-parse", f"{TIP}:{path}"], capture_output=True,
                          check=True).stdout.decode().strip()


def crlf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\n", "\r\n")


def must_replace(txt, old, new, count=1):
    if chr(13) in txt:               # the base is CRLF (1056 of 1057 stack .py blobs); declared_vs_built.py is LF
        old, new = crlf(old), crlf(new)
    assert txt.count(old) == count, (txt.count(old), old[:80])
    return txt.replace(old, new)


# ------------------------------------------------------------------ refc_v3_train.py
P = "stack/scripts/refc_v3_train.py"
txt = tip_bytes(P).decode("utf-8")
assert "pose_sync" not in txt, "base already carries pose_sync -- not the tip"

# (1) class attributes
txt = must_replace(txt, '''    ego_history: bool = False
''', '''    ego_history: bool = False
    #: ⭐ refcv8 X10 (PI data audit D3): the POSE-TO-IMAGE timing correction, ``tanitad/data/pose_sync.py``.
    #: ``None`` = OFF, the DEFAULT: ``__getitem__`` is then byte-for-byte the pre-X10 path (no new key, no new
    #: branch taken). Set by :meth:`enable_pose_sync` from ``--pose-sync-sidecar``. It re-samples the WINDOW's ego
    #: state at the NOW image's capture instant and touches ONLY the fields listed in :meth:`_pose_sync_apply`.
    #: ⛔ It does not move a row index, the label clock, or any ``(sid, k)`` join.
    pose_sync = None
    #: sids the sidecar did not cover (counted at enable time, capped there); they are read UNSHIFTED.
    pose_sync_uncovered_sids: frozenset = frozenset()
''')

# (2) enable_pose_sync + _pose_sync_apply, inserted before V3Dataset.__getitem__
txt = must_replace(txt, '''    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        T = ep.poses.shape[0]
        idx = torch.arange(t + w, t + w + MAX_H_EXT)
''', '''    def enable_pose_sync(self, sidecar_path: str, *, sign: float = 1.0,
                         max_uncovered_frac: float = 0.01) -> dict:
        """⭐ refcv8 X10 -- OPT-IN pose-to-image timing correction. Returns the census ``config.json`` stamps.

        ``sidecar_path`` (``--pose-sync-sidecar``): JSON-lines ``{sid, dt_s, n_rows, delta_us}`` built by
        ``scripts/build_pose_sync_sidecar.py`` from each clip's camera ``timestamps.parquet``.

        ⛔ REFUSES, at launch and never mid-run: a sidecar that covers NONE of this split's clips; one that covers
        fewer than ``1 - max_uncovered_frac`` of them; and any COVERED clip whose row count is not
        ``len(poses) + raw_offset`` (a sidecar for another timeline). ``sign`` exists only for the tests'
        deliberate-regression arm."""
        from tanitad.data import pose_sync as _ps
        ps = _ps.read_pose_sync_sidecar(str(sidecar_path), sign=sign)
        n_cov = 0
        unc = set()
        bad = []
        for ep in self.episodes:
            sid = int(ep.episode_id)
            row = ps.table.get(sid)
            if row is None:
                unc.add(sid)
                continue
            n_cov += 1
            want = int(ep.poses.shape[0]) + self._raw_offset(ep)
            if row.delta_s.shape[0] != want:
                bad.append((sid, int(row.delta_s.shape[0]), want))
        n_all = n_cov + len(unc)
        if n_cov == 0:
            raise SystemExit(f"[v3] ⛔ --pose-sync-sidecar {sidecar_path}: it covers NONE of this split's {n_all} "
                             f"clips -- a sidecar for another corpus. Refusing rather than running every clip "
                             f"unshifted while config.json names the sidecar.")
        if bad:
            raise SystemExit(f"[v3] ⛔ --pose-sync-sidecar {sidecar_path}: {len(bad)} covered clip(s) disagree on the "
                             f"row count (sid, sidecar rows, needed) e.g. {bad[:3]} -- it was built on another "
                             f"timeline than this cache.")
        if len(unc) > max_uncovered_frac * n_all:
            raise SystemExit(f"[v3] ⛔ --pose-sync-sidecar {sidecar_path}: {len(unc)}/{n_all} clips are uncovered "
                             f"(cap {max_uncovered_frac:.1%}).")
        self.pose_sync = ps
        self.pose_sync_uncovered_sids = frozenset(unc)
        return {"sidecar": ps.path, "sidecar_rows": ps.n_rows, "sidecar_meta": ps.meta, "sign": ps.sign,
                "n_clips": n_all, "n_covered": n_cov, "n_uncovered_read_unshifted": len(unc),
                "rule": "window re-sampled at t_pose + delta[NOW raw row]; rows, label clock and (sid,k) joins unchanged",
                "fields_shifted": ["pose_last", "future_poses", "future_poses_ext", "pose_hist", "goal_tac",
                                   "actions", "future_actions"]}

    def _pose_sync_apply(self, item: dict, ep, t: int) -> None:
        """Re-sample this window's ego state at the NOW image's capture instant (X10). In place.

        Shifted (one common shift, the NOW row's): ``pose_last`` (hence ``v0 = pose_last[:, 3]``),
        ``future_poses``, ``future_poses_ext`` (hence ``waypoint_targets`` and the tactical factored labels),
        ``pose_hist`` when ``ego_history``, ``goal_tac``, ``actions`` and ``future_actions``.
        NOT shifted: anything that reads ``ep.poses`` through another dataset layer (``nav_cmd``, route labels,
        ``lan``, the agent-future transform) -- those are coarse/oracle labels on the pose clock the agent, v9 and
        map joins are keyed on."""
        from tanitad.data import pose_sync as _ps
        sid = int(ep.episode_id)
        if sid in self.pose_sync_uncovered_sids:
            return
        w = self.window
        now = t + w - 1
        got = _ps.window_shifted_tracks(self.pose_sync, sid, now, self._raw_offset(ep), ep.poses, ep.actions)
        if got is None:                                   # covered at enable time, so this is a bug: FAIL LOUD
            raise RuntimeError(f"[v3] pose-sync: sid {sid} NOW raw row {now + self._raw_offset(ep)} is outside "
                               f"the sidecar although enable_pose_sync validated its length")
        P, A, _s = got
        pv = torch.from_numpy(P)
        av = torch.from_numpy(A)
        T = pv.shape[0]
        idx = torch.arange(t + w, t + w + MAX_H_EXT)
        item["pose_last"] = pv[now]
        item["future_poses"] = pv[t + w:t + w + self.max_horizon]
        item["future_poses_ext"] = pv[idx.clamp(max=T - 1)]
        if self.ego_history:
            item["pose_hist"] = pv[t:t + w]
        g, gv = refb_labels.goal_tac_targets(pv, now, v3.GOAL_TAU_STEPS)
        item["goal_tac"] = g
        item["goal_tac_valid"] = gv
        item["actions"] = av[t:t + w]
        item["future_actions"] = av[t + w:t + w + self.max_horizon]

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        T = ep.poses.shape[0]
        idx = torch.arange(t + w, t + w + MAX_H_EXT)
''')

# (3) the hook, right after goal_tac_valid
txt = must_replace(txt, '''        item["goal_tac"] = g                                        # [K, 4]
        item["goal_tac_valid"] = gv                                 # [K] bool
        # ---- v7.2 tactical labels (PI 2026-09-02: MANDATORY) --------------
''', '''        item["goal_tac"] = g                                        # [K, 4]
        item["goal_tac_valid"] = gv                                 # [K] bool
        # ---- ⭐ refcv8 X10: pose-to-image timing (OPT-IN; None = this line is a no-op) ----
        if self.pose_sync is not None:
            self._pose_sync_apply(item, ep, t)
        # ---- v7.2 tactical labels (PI 2026-09-02: MANDATORY) --------------
''')

# (4) train(): enable on the train and eval datasets
txt = must_replace(txt, '''    nav_stats = eval_nav_stats = v7_manifest = None
    tac_goal_stats = None                       # D-TACGOAL
''', '''    nav_stats = eval_nav_stats = v7_manifest = None
    tac_goal_stats = None                       # D-TACGOAL
    pose_sync_stats = eval_pose_sync_stats = None            # refcv8 X10 (opt-in)
    if getattr(args, "pose_sync_sidecar", None):
        pose_sync_stats = ds.enable_pose_sync(args.pose_sync_sidecar)
        print(f"[v3] pose-sync: {pose_sync_stats['n_covered']}/{pose_sync_stats['n_clips']} clips shifted "
              f"({pose_sync_stats['n_uncovered_read_unshifted']} unshifted)", flush=True)
''')
txt = must_replace(txt, '''        e_ds = dcls(e_eps, **kw)
        e_ds.u8_frames = u8     # the eval decodes in the MAIN process: 4x less there too
''', '''        e_ds = dcls(e_eps, **kw)
        e_ds.u8_frames = u8     # the eval decodes in the MAIN process: 4x less there too
        if getattr(args, "pose_sync_sidecar", None):             # refcv8 X10: the eval GT rides the SAME clock
            eval_pose_sync_stats = e_ds.enable_pose_sync(args.pose_sync_sidecar)
''')

# (5) config.json stamp
txt = must_replace(txt, '''        "label_clock": ({"train": clip_clock_stats, "eval": eval_clip_clock_stats}
                        if clip_clock_stats is not None else None),
''', '''        "label_clock": ({"train": clip_clock_stats, "eval": eval_clip_clock_stats}
                        if clip_clock_stats is not None else None),
        # ⭐ refcv8 X10: None = the pose-to-image offset is UNCORRECTED (every refcv<=7 run). Never a silent default.
        "pose_sync": ({"train": pose_sync_stats, "eval": eval_pose_sync_stats}
                      if pose_sync_stats is not None else None),
''')

# (6) argparse
txt = must_replace(txt, '''    ap.add_argument("--clip-clock-sidecar", default=None,
''', '''    ap.add_argument("--pose-sync-sidecar", default=None,
                    help="⭐ refcv8 X10 (OPT-IN, default OFF): JSON-lines {sid, dt_s, n_rows, delta_us} built by "
                         "`scripts/build_pose_sync_sidecar.py` from each clip's camera timestamps. Re-samples the "
                         "TRAINING window's ego state (pose_last/v0, future poses, pose_hist, goal_tac, actions) at "
                         "the instant the NOW image was captured (0-34 ms later than its stored pose; MEASURED mean "
                         "16.5 ms). No row index, label clock or (sid,k) join moves. Adds NO inference input. "
                         "Stamped in config.json `pose_sync`.")
    ap.add_argument("--clip-clock-sidecar", default=None,
''')
out = f"{PK}/code/fix/{P}"
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "wb").write(txt.encode("utf-8"))
print("wrote", out, "base blob", tip_blob(P), "len", len(txt))

# ------------------------------------------------------------------ declared_vs_built.py
P2 = "stack/tanitad/train/declared_vs_built.py"
t2 = tip_bytes(P2).decode("utf-8")
assert "pose_sync" not in t2
t2 = must_replace(t2, '''        ("clip_clock_sidecar", "the label CLOCK source; G3 checks the clock it produces"),
''', '''        ("clip_clock_sidecar", "the label CLOCK source; G3 checks the clock it produces"),
        ("pose_sync_sidecar", "the X10 pose-to-image timing sidecar (training ego-state clock; stamped `pose_sync`)"),
''')
out2 = f"{PK}/code/fix/{P2}"
os.makedirs(os.path.dirname(out2), exist_ok=True)
open(out2, "wb").write(t2.encode("utf-8"))
print("wrote", out2, "base blob", tip_blob(P2))


# ------------------------------------------------------------------ test_declared_vs_built.py (the pinned registry size)
P3 = "stack/tests/test_declared_vs_built.py"
t3 = tip_bytes(P3).decode("utf-8")
assert "X10" not in t3
t3 = must_replace(t3, """    # --r8-no-rc), registered by `tanitad.train.refcv8_train._register_gdvb` (226 -> 256)
    assert len(dvb.REGISTRY) == 256
""", """    # --r8-no-rc), registered by `tanitad.train.refcv8_train._register_gdvb` (226 -> 256)
    # refcv8 X10 (pose-to-image timing): +1 = --pose-sync-sidecar, "data", registered in declared_vs_built itself (256 -> 257)
    assert len(dvb.REGISTRY) == 257
""")
out3 = f"{PK}/code/fix/{P3}"
os.makedirs(os.path.dirname(out3), exist_ok=True)
open(out3, "wb").write(t3.encode("utf-8"))
print("wrote", out3, "base blob", tip_blob(P3))
