"""One-shot editor: adds --dump-ticks (and the train-window mode it needs) to the COPY of v8f_t1_eval.py. Without the new flags every code path is the original's."""
p = "C:/Users/Admin/hlat/code/stack/scripts/v8f_t1_eval.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- docstring: one line on the new mode
rep('''Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)

PREREG_V8F_HACC_LAT adds ``--dump-ticks DIR``: one row per (window, tick) of the MODEL's rollout -- O's plan before the post-map (tau), the prior (P), the plan the model issued, the measured ego speed
v0 at the tick, the logged future in the tick's EXECUTED frame (GT, "rejoin the human's path") and the window / episode / clip ids. Absent, the tool's output is unchanged bit for bit (control C-T).
``--windows train`` runs the same rollout on the training windows of HBLEND's 141 train clips (tick samples only: no floor rollout, no analysis, nothing of the eval-141 world).
"""''')

# ---- model_planner: optional aux capture
rep('''def model_planner(model, device):
    import torch

    def plan(batch):
        with torch.no_grad():
            out = model.forward_infer(batch)
        plan.keys_read = set(out["_keys_read"])
        return out["traj30"].float().cpu().numpy()
    plan.keys_read = set()
    return plan
''', '''def model_planner(model, device, keep_aux: bool = False):
    """The model's plan. ``keep_aux`` (PREREG_V8F_HACC_LAT ``--dump-ticks``) additionally keeps, per call, O's plan before the post-map (``tau``), the prior (``P``) and the gate flags; it changes no output."""
    import torch

    def plan(batch):
        with torch.no_grad():
            out = model.forward_infer(batch)
        plan.keys_read = set(out["_keys_read"])
        if keep_aux:
            plan.aux = {"tau": out.get("traj30_o", out["traj30"]).double().cpu().numpy(), "P": out["p10"].double().cpu().numpy(),
                        "fired": None if "hacc_gate_fired" not in out else out["hacc_gate_fired"].cpu().numpy()}
        return out["traj30"].float().cpu().numpy()
    plan.keys_read = set()
    plan.aux = None
    return plan
''')

# ---- the recorder, before the rollout section
rep('''# ------------------------------------------------------------------------------------------------ the rollout
def rollout(''', '''# ------------------------------------------------------------------------------------------------ the tick recorder (--dump-ticks)
def gt_in_frame(pose_grid, pose_now, j: int):
    """The logged future ``pose_grid[j+1 .. j+30]`` in the frame of ``pose_now`` = (x, y, yaw, ...): the SAME arithmetic as ``floors_kin.gt_waypoints`` / ``v8f_mini_data.gt_future`` (rotation by
    ``-yaw``), with the origin pose replaced by the EXECUTED pose. At tick 0 the executed pose IS the logged pose, so this equals the store's ``gt[t0]`` (control C-T2). ``ok`` False (NaN GT) when the logged clip
    ends before ``j + 30`` (the same ``t <= T - 31`` rule as ``gt_future``)."""
    pg = np.asarray(pose_grid, np.float64)
    if j + H > len(pg) - 1:
        return np.full((H, 2), np.nan), False
    d = pg[j + 1:j + 1 + H, :2] - np.asarray(pose_now, np.float64)[None, :2]
    c, s = np.cos(-pose_now[2]), np.sin(-pose_now[2])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=-1), True


class TickRecorder:
    """The ``hook`` of :func:`rollout` that keeps the tick samples. ``hook(n, batch, plan, states)`` runs BEFORE the plan is committed, so ``st.j`` and ``st.d['pose'][st.j]`` are the tick's executed state."""

    FIELDS = ("tau", "P", "plan", "gt", "gt_ok", "v0", "pose", "fired", "win", "ci", "t0", "tick", "j", "age_t", "age_s")

    def __init__(self, planner, store, J):
        self.planner, self.store, self.J = planner, store, J
        self.rows = {k: [] for k in self.FIELDS}
        self.clip = []
        self.base = 0                                                    # the index of the chunk's first window in the run's window list

    def __call__(self, n, batch, plan, states):
        aux = self.planner.aux
        if aux is None:
            raise RuntimeError("--dump-ticks needs a planner built with keep_aux=True")
        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()             # the measured ego speed the prior (and the gate) read: exactly p10f[:, -1, 0]
        for b, st in enumerate(states):
            clip = str(self.J["clip"][st.ci])
            pose_now = np.array(st.d["pose"][st.j], np.float64)
            gt, ok = gt_in_frame(self.store.get(clip)["pose"], pose_now, st.j)
            r = self.rows
            r["tau"].append(aux["tau"][b]); r["P"].append(aux["P"][b]); r["plan"].append(np.asarray(plan[b], np.float64))
            r["gt"].append(gt); r["gt_ok"].append(ok); r["v0"].append(v0[b]); r["pose"].append(pose_now[:4])
            r["fired"].append(False if aux["fired"] is None else bool(aux["fired"][b]))
            r["win"].append(self.base + b); r["ci"].append(st.ci); r["t0"].append(st.t0); r["tick"].append(n); r["j"].append(st.j)
            r["age_t"].append(int(batch["a_t"][b])); r["age_s"].append(int(batch["a_s"][b]))
            self.clip.append(clip)

    def save(self, path, extra: dict | None = None):
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        out["clip"] = np.asarray(self.clip)
        for k, v in (extra or {}).items():
            out[k] = np.asarray(v)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **out)
        return out


# ------------------------------------------------------------------------------------------------ the rollout
def rollout(''')

# ---- parser
rep('''    p.add_argument("--no-analyze", action="store_true")
    p = sub.add_parser("analyze")''', '''    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--dump-ticks", default="", help="PREREG_V8F_HACC_LAT: a directory; writes ticks_<tag>.npz = one row per (window, tick) of the model's rollout (tau, P, plan, v0, GT in the executed frame, ids)")
    p.add_argument("--windows", choices=("eval", "train"), default="eval", help="train: HBLEND's 141 train clips (needs --dump-ticks, --train-cache, --train-join, --fit-clips); tick samples only")
    p.add_argument("--train-cache", default="", help="--windows train: the cache root that holds <root>/train/clips")
    p.add_argument("--train-join", default="", help="--windows train: join_train.npz")
    p.add_argument("--fit-clips", default="", help="--windows train: fit_clips.json (the 141 train clips)")
    p = sub.add_parser("analyze")''')

# ---- cmd_run: windows, recorder, no floor in train mode
rep('''    tracks = D.load_native_tracks(a.tracks)
    J = D.load_join(a.eval_join)
    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(a.eval_grid, J)
    if a.max_windows''', '''    tracks = D.load_native_tracks(a.tracks)
    train_mode = a.windows == "train"
    if train_mode:
        if not (a.dump_ticks and a.train_cache and a.train_join and a.fit_clips):
            raise SystemExit("REFUSED: --windows train needs --dump-ticks, --train-cache, --train-join and --fit-clips (it produces tick samples only)")
        J = D.load_join(a.train_join)
        store = D.ClipStore(a.train_cache, "train", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
        wins = D.train_windows(store, J, json.load(open(a.fit_clips, encoding="utf-8"))["clips"], hist=cfg.hist, start=cfg.window_start)    # the trainer's own builder, as HBLEND's cmd_fit
    else:
        J = D.load_join(a.eval_join)
        store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
        wins = D.eval_grid_windows(a.eval_grid, J)
    if a.max_windows''')

rep('''    for b0 in range(0, len(wins), a.batch):
        chunk = wins[b0:b0 + a.batch]
        mp = model_planner(model, dev)
        st_m = make_states(store, tracks, J, chunk, a.nav)
        plan0, exe = rollout(mp, st_m, ds, proxy, **kw)
        keys |= mp.keys_read
        st_f = make_states(store, tracks, J, chunk, a.nav)
        _, exe_f = rollout(floor_planner(model, dev), st_f, ds, proxy, **kw)
        n_clip["cl"] += sum(s.clipped > 0 for s in st_m)
        n_clip["fl"] += sum(s.clipped > 0 for s in st_f)
        cl += exe
        fl += exe_f
        ol.append(plan0)
        for ci, t0, _, _ in chunk:''', '''    rec = None
    for b0 in range(0, len(wins), a.batch):
        chunk = wins[b0:b0 + a.batch]
        mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks))
        st_m = make_states(store, tracks, J, chunk, a.nav)
        if a.dump_ticks:
            rec = rec or TickRecorder(mp, store, J)
            rec.planner, rec.base = mp, b0
            plan0, exe = rollout(mp, st_m, ds, proxy, hook=rec, **kw)
        else:
            plan0, exe = rollout(mp, st_m, ds, proxy, **kw)
        keys |= mp.keys_read
        n_clip["cl"] += sum(s.clipped > 0 for s in st_m)
        if train_mode:
            print(f"[{b0 + len(chunk)}/{len(wins)}] {time.time() - t_start:.0f}s", flush=True)
            continue
        st_f = make_states(store, tracks, J, chunk, a.nav)
        _, exe_f = rollout(floor_planner(model, dev), st_f, ds, proxy, **kw)
        n_clip["fl"] += sum(s.clipped > 0 for s in st_f)
        cl += exe
        fl += exe_f
        ol.append(plan0)
        for ci, t0, _, _ in chunk:''')

rep('''    v0 = np.asarray(v0, np.float64)
    arms = {"cl"''', '''    if train_mode:
        tick_path = os.path.join(a.dump_ticks, f"ticks_{a.tag}.npz")
        out = rec.save(tick_path, {"split": "train", "tag": a.tag, "ckpt": a.ckpt, "replan_steps": a.replan_steps, "n_windows": len(wins),
                                   "clip_names": np.asarray(J["clip"])})
        hd = {"tag": a.tag, "ckpt": a.ckpt, "split": "train", "tier": "T1 (self-action OPEN loop; ego state = own executed motion), TRAIN windows, tick samples only",
              "replan_steps": a.replan_steps, "schedule": a.schedule, "scene": a.scene, "nav": a.nav, "limit": a.limit, "n_windows": len(wins),
              "n_tick_samples": int(len(out["tau"])), "n_gt_ok": int(out["gt_ok"].sum()), "ticks_file": tick_path, "model_keys_read": sorted(keys),
              "windows_with_envelope_clip": n_clip["cl"], "seconds": time.time() - t_start}
        os.makedirs(a.out_dir, exist_ok=True)
        json.dump(hd, open(os.path.join(a.out_dir, f"t1v8f_{a.tag}.json"), "w"), indent=1)
        print("wrote", tick_path, "samples", len(out["tau"]), "gt_ok", int(out["gt_ok"].sum()))
        return 0
    v0 = np.asarray(v0, np.float64)
    arms = {"cl"''')

# header: the gate (only when set) and the tick file (only when dumped)
rep('''    header["open_channels"] = [c for c in header["open_channels"] if c]
    os.makedirs(a.out_dir, exist_ok=True)''', '''    header["open_channels"] = [c for c in header["open_channels"] if c]
    if model.hacc_gate_active:
        header["hacc_gate_v"] = float(model.hacc_gate_v)
    if a.dump_ticks:
        tick_path = os.path.join(a.dump_ticks, f"ticks_{a.tag}.npz")
        rec.save(tick_path, {"split": "eval", "tag": a.tag, "ckpt": a.ckpt, "replan_steps": a.replan_steps, "n_windows": len(wins), "clip_names": np.asarray(J["clip"])})
        header["ticks_file"] = tick_path
    os.makedirs(a.out_dir, exist_ok=True)''')
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n"))
print("ok")
