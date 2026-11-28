"""Apply the LEVEL-FLOOR patch (``--level-floor on``, PREREG_LEVELFLOOR_A3 sha256 977a785e...) to the REINDEXED tool (sha256 0417c838..., ``v8f_t1_eval_ri.py``). One-shot; each replacement must match exactly once.
Usage: python -I apply_lf_patch.py <RI v8f_t1_eval_ri.py> <out v8f_t1_eval_lf.py>     (the patched tool keeps the name v8f_t1_eval_ri.py inside the read tree; see build_lf_tree.py)
``--level-floor off`` (the default; the flag absent) is the source tool's behaviour bit for bit: every new code path is behind ``floor is not None`` / ``a.level_floor == "on"`` and ``level_floor.py`` is not even imported.
``--level-floor on --floor-kappa K --floor-ckpt PATH``: the floor operator F of ``level_floor.py`` (next to the tool) runs at every replan inside ``model_planner``'s ``plan(batch)``: raw -> S (``--smooth``) -> F -> kbox
(``--feasible``), the floor heads (v_hat, presence, gap_hat) read from the SECOND checkpoint's own encode of the SAME tick batch. The patched tool's sha256 is regenerable: run this script on the pinned source."""
import hashlib
import sys

SRC_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the RI tool 0417c838"
assert b"\r" not in b, "the source tool must be LF"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''
Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''
``--level-floor {off,on}`` (LEVEL-FLOOR, PREREG_LEVELFLOOR_A3 sha256 977a785e..., default off; this file is the RI tool + ``apply_lf_patch.py``): ``on`` applies the floor operator F (``level_floor.py``, next to this tool)
to the plan at EVERY replan, inside ``model_planner``'s ``plan(batch)``: raw -> S (``--smooth``) -> F -> the projection (``--feasible``). F runs AFTER S in every branch (the floor branch sits before the unpatched early returns; A3_RULING_1): ``--smooth sg72`` with no ``--feasible`` (the banked bases' flags = the bare arm) is raw -> S -> F; ``--smooth sg72 --feasible kbox`` (the composite) is raw -> S -> F -> kbox; ``--smooth off --feasible off`` is raw -> F. ``--floor-kappa K`` (float in [0, 1], required with on, refused with off) is the dose;
``--floor-ckpt PATH`` (required with on, refused with off) is the head checkpoint (the LLR checkpoint: its cfg must have ``level_head`` AND ``lead_head`` on and its data interface equal to the planner's): a SECOND model instance
encodes the SAME tick batch itself (``floor_heads``: the adapter, ``lvl_head`` and ``lead_head`` only; its O / S / T are not run; the first replan is checked against that model's own ``forward_infer`` outputs
``level_v`` / ``lead_logit`` / ``lead_gap``). F reads v_hat (m/s), pi = sigmoid(presence logit), g = 40 * gap_hat (m) from the floor heads and v0 = ``batch["p10f"][:, -1, 0]`` (the measured ego speed) and nothing else;
no nav, route, limit, label or situation-classifier output enters it. With ``--dump-ticks`` the ticks file carries ``lf_bind`` / ``lf_cap`` (bool) and ``lf_vf`` (NaN = no floor) / ``lf_vhat`` / ``lf_pi`` / ``lf_gap`` (m) /
``lf_lift`` (the 0-3 s mean plan speed after F minus before, m/s) per replan, plus ``lf_bind_k10`` (instrument: F changed a knot among the first 10 = the 2 s the read executes), and ``plan_raw`` (the model's plan before S and F) even when ``--smooth off --feasible off``. The result JSON header carries ``level_floor``,
``floor_kappa``, ``floor_ckpt``, ``floor_ckpt_sha256`` and ``level_floor_detail`` (the operator, its constants, the bind / cap counts, the floor model's batch keys). ``--histswap`` != off is refused with the floor
(F's v0 would be the logged speed).

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''')

# 2. helpers (after HEAD_KEYS) -------------------------------------------------------------------------------------
rep('HEAD_KEYS = ("ckpt", "goal_prob", "lat_logits", "lon_logits", "cons_pred")\n', '''HEAD_KEYS = ("ckpt", "goal_prob", "lat_logits", "lon_logits", "cons_pred")


# ------------------------------------------------------------------------------------------------ LEVEL-FLOOR (PREREG_LEVELFLOOR_A3, sha256 977a785e...): the floor operator F
LF_MATCH = ("hist", "hmax", "max_age_t", "max_age_s", "kin_mode", "n_tok", "d_feat")        # the planner's batch is built for ITS cfg: the floor model must read the same interface
LF_PARITY_TOL = 1e-6                                                                           # the first replan's light-path heads vs the floor model's own forward_infer (expected: 0.0)
_LF_MOD = None


def _lf():
    """``level_floor.py`` (the pure operator, next to this tool), loaded BY PATH and only under ``--level-floor on`` (``off`` never imports it)."""
    global _LF_MOD
    if _LF_MOD is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "level_floor.py")
        if not os.path.isfile(path):
            raise SystemExit(f"REFUSED: --level-floor on needs {path} next to this tool")
        spec = importlib.util.spec_from_file_location("level_floor", path)
        _LF_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_LF_MOD)
    return _LF_MOD


def lf_check_args(a):
    """The flag refusals, before anything is loaded: ``on`` needs ``--floor-kappa`` in [0, 1] and an existing ``--floor-ckpt``; ``off`` takes neither; no ``--histswap`` with the floor."""
    if a.level_floor == "on":
        if a.floor_kappa is None:
            raise SystemExit("REFUSED: --level-floor on needs --floor-kappa K (a dose is never defaulted)")
        if not (math.isfinite(float(a.floor_kappa)) and 0.0 <= float(a.floor_kappa) <= 1.0):
            raise SystemExit(f"REFUSED: --floor-kappa must be a finite number in [0, 1], got {a.floor_kappa!r}")
        if not a.floor_ckpt:
            raise SystemExit("REFUSED: --level-floor on needs --floor-ckpt PATH (the head checkpoint)")
        if not os.path.isfile(a.floor_ckpt):
            raise SystemExit(f"REFUSED: --floor-ckpt {a.floor_ckpt} does not exist")
        if a.histswap != "off":
            raise SystemExit(f"REFUSED: --level-floor on is not defined with --histswap {a.histswap} (F's v0 would be the logged speed, an oracle channel)")
    elif a.floor_kappa is not None or a.floor_ckpt:
        raise SystemExit("REFUSED: --floor-kappa / --floor-ckpt belong to --level-floor on; with the floor off they would silently do nothing")


def lf_check_cfg(fcfg, fgrid, cfg, grid):
    """Refuse a floor checkpoint whose cfg lacks ``level_head`` on AND ``lead_head`` on, or whose data interface differs from the planner's."""
    if fcfg.level_head != "on" or fcfg.lead_head != "on":
        raise SystemExit(f"REFUSED: --floor-ckpt's cfg must have level_head on AND lead_head on (it has level_head={fcfg.level_head!r}, lead_head={fcfg.lead_head!r})")
    for k in LF_MATCH + ("window_start_effective",):
        if getattr(fcfg, k) != getattr(cfg, k):
            raise SystemExit(f"REFUSED: --floor-ckpt's {k}={getattr(fcfg, k)!r} differs from the planner's {getattr(cfg, k)!r}: the two models would not read the same batch")
    if fgrid != grid:
        raise SystemExit(f"REFUSED: --floor-ckpt's grid {fgrid!r} differs from the planner's {grid!r}")


def floor_heads(fm, batch):
    """(v_hat [B], presence logit [B], gap_hat [B]) float32 numpy and the batch keys read, from the floor model's OWN encode of ``batch``: ``_encode`` (the adapter on ``feats``), then ``_level`` and ``_lead`` at the
    current frame (the exact calls ``V8FMini._forward`` makes for ``level_v`` / ``lead_logit`` / ``lead_gap``). The model's O / S / T are not run. ``batch`` is read through the model's key guard."""
    import torch
    g = M._KeyGuard(batch)
    with torch.no_grad():
        adapt = fm._encode(g)
        last_now = torch.full((adapt.shape[0],), g["feats"].shape[1] - 1, dtype=torch.long, device=adapt.device)
        v_hat = fm._level(g, adapt, last_now)[0]
        logit, gap_hat = fm._lead(g, adapt, last_now)[:2]
    return v_hat.float().cpu().numpy(), logit.float().cpu().numpy(), gap_hat.float().cpu().numpy(), set(g.read)


class LevelFloor:
    """The floor of one run: the floor model (a second instance, ``--floor-ckpt``), the dose and the running counts. ``__call__(batch, plan, v0)`` -> ``(plan_out, info)`` (``level_floor.apply_floor_batch``)."""

    def __init__(self, fm, kappa: float, ckpt_path: str, ckpt_sha256: str):
        self.fm, self.kappa, self.ckpt_path, self.ckpt_sha256 = fm, float(kappa), ckpt_path, ckpt_sha256
        self.op = _lf()
        self.keys_read = set()
        self.parity = {"checked_replans": 0, "max_abs_diff": None}
        self.stats = {"replans": 0, "band": 0, "bind": 0, "bind_k10": 0, "cap": 0, "dir_fallback_binds": 0, "short_last_seg_binds": 0, "lift_sum_ms": 0.0}

    def heads(self, batch):
        v_hat, logit, gap_hat, read = floor_heads(self.fm, batch)
        self.keys_read |= read
        if self.parity["checked_replans"] == 0:                                                      # the first replan: the light path equals the model's own forward_infer outputs
            import torch
            with torch.no_grad():
                o = self.fm.forward_infer(batch)
            d = max(float(np.abs(v_hat - o["level_v"].float().cpu().numpy()).max()), float(np.abs(logit - o["lead_logit"].float().cpu().numpy()).max()),
                    float(np.abs(gap_hat - o["lead_gap"].float().cpu().numpy()).max()))
            if not d <= LF_PARITY_TOL:
                raise RuntimeError(f"level floor: the floor heads' light path differs from the floor model's forward_infer by {d:.3e} (> {LF_PARITY_TOL})")
            self.parity = {"checked_replans": 1, "max_abs_diff": d}
        return v_hat, logit, gap_hat

    def __call__(self, batch, plan, v0):
        v_hat, logit, gap_hat = self.heads(batch)
        out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        st = self.stats
        st["replans"] += len(info["bind"])
        st["band"] += int(info["band"].sum())
        st["bind"] += int(info["bind"].sum())
        st["cap"] += int(info["cap"].sum())
        st["bind_k10"] += int(info["bind_k10"].sum())
        st["dir_fallback_binds"] += int((info["dir_fallback"] & info["bind"]).sum())
        st["short_last_seg_binds"] += int((info["short_last"] & info["bind"]).sum())
        st["lift_sum_ms"] += float(info["lift"][info["bind"]].sum())
        return out, info

    def meta(self):
        return {"level_floor": "on", "floor_kappa": self.kappa, "floor_ckpt_sha256": self.ckpt_sha256}

    def detail(self):
        st = dict(self.stats)
        st["lift_mean_ms_where_bound"] = (st["lift_sum_ms"] / st["bind"]) if st["bind"] else None
        return {"operator": self.op.OPERATOR_TEXT, "constants": self.op.CONSTANTS, "prereg": "dagland/motion_c/a3_prereg/PREREG_LEVELFLOOR_A3.md (sha256 977a785e...), H-LEVELFLOOR-1",
                "applied": "at every replan inside model_planner's plan(batch): raw -> S (--smooth) -> F -> the projection (--feasible); the P10 floor is not floored",
                "vision_only": "v_hat, presence and gap_hat come from the floor model's adapter + lvl_head + lead_head on the tick's frames; v0 = batch['p10f'][:, -1, 0] (measured ego speed); nothing else enters",
                "floor_model_batch_keys_read": sorted(self.keys_read), "heads_parity_vs_forward_infer": self.parity, "counts": st}


def load_level_floor(a, cfg, ck, dev):
    """The floor of ``--level-floor on``: the head checkpoint is loaded into its OWN V8FMini (refused unless its cfg has both heads on and reads the planner's interface)."""
    import hashlib
    import torch
    fk_ = torch.load(a.floor_ckpt, map_location="cpu", weights_only=False)
    fcfg = M.V8FMiniConfig(**fk_["cfg"])
    lf_check_cfg(fcfg, fk_.get("grid"), cfg, ck.get("grid"))
    fm = M.V8FMini(fcfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    fm.load_state_dict(fk_["model"])
    fm.to(dev).eval()
    probs = M.level_head_source_problems() + M.lead_head_source_problems()
    if probs:
        raise SystemExit(f"REFUSED: the floor model's heads are not vision-only by source: {probs}")
    h = hashlib.sha256()
    with open(a.floor_ckpt, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return LevelFloor(fm, a.floor_kappa, a.floor_ckpt, h.hexdigest())
''')

# 3. model_planner ---------------------------------------------------------------------------------------------------
rep('def model_planner(model, device, keep_aux: bool = False, keep_heads: bool = False, smooth: str = "off", feasible: str = "off"):',
    'def model_planner(model, device, keep_aux: bool = False, keep_heads: bool = False, smooth: str = "off", feasible: str = "off", floor=None):')
rep('; ``off`` returns the plan untouched."""', '''; ``off`` returns the plan untouched.
    ``floor`` (``--level-floor on``, a :class:`LevelFloor`): after S the floor operator F runs on the plan at every replan (raw -> S -> F -> ``feasible``); ``plan.floor_info`` holds what F did (None without a floor)."""''')
rep('''        raw = out["traj30"].float().cpu().numpy()
        if smooth == "off" and feasible == "off":
            return raw
''', '''        raw = out["traj30"].float().cpu().numpy()
        if floor is not None:                                            # LEVEL-FLOOR: raw -> S -> F -> projection; the floor heads read the SAME batch in the floor model's own encode
            plan.plan_raw = raw
            plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()          # the measured ego speed (== TickRecorder's ``v0``)
            plan_f, plan.floor_info = floor(batch, plan_s, v0)
            if feasible != "off":
                plan.plan_s = plan_s                                     # the plan after S, before F and the projection
                return feasible_plan(plan_f, feasible, v0=v0)
            return plan_f
        if smooth == "off" and feasible == "off":
            return raw
''')
rep('''    plan.plan_s = None
    return plan
''', '''    plan.plan_s = None
    plan.floor_info = None
    return plan
''')

# 4. TickRecorder ----------------------------------------------------------------------------------------------------
rep('    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")',
    '    LF_FIELDS = ("lf_bind", "lf_cap", "lf_vf", "lf_vhat", "lf_pi", "lf_gap", "lf_lift", "lf_bind_k10")                                              # LEVEL-FLOOR: per replan, only with the floor on\n'
    '    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")')
rep('feasible: str = "off", histswap: str = "off"):\n        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible, self.histswap = planner, store, J, bool(heads), smooth, feasible, histswap\n',
    'feasible: str = "off", histswap: str = "off", level_floor: bool = False):\n        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible, self.histswap = planner, store, J, bool(heads), smooth, feasible, histswap\n        self.level_floor = bool(level_floor)\n')
rep('(("plan_raw",) if (smooth != "off" or feasible != "off") else ())', '(("plan_raw",) if (smooth != "off" or feasible != "off" or level_floor) else ())')
rep('+ (("v0_in",) if histswap != "off" else ())}', '+ (("v0_in",) if histswap != "off" else ()) + (self.LF_FIELDS if level_floor else ())}')
rep('            if self.smooth != "off" or self.feasible != "off":', '            if self.smooth != "off" or self.feasible != "off" or self.level_floor:')
rep('r["age_t"].append(int(batch["a_t"][b])); r["age_s"].append(int(batch["a_s"][b]))\n', '''r["age_t"].append(int(batch["a_t"][b])); r["age_s"].append(int(batch["a_s"][b]))
            if self.level_floor:                                         # LEVEL-FLOOR: what F did at this replan (its inputs from the floor heads, the lift, whether F / the lead cap bound)
                fi = self.planner.floor_info
                r["lf_bind"].append(bool(fi["bind"][b])); r["lf_cap"].append(bool(fi["cap"][b])); r["lf_bind_k10"].append(bool(fi["bind_k10"][b]))
                for k_ in ("vf", "vhat", "pi", "gap", "lift"):
                    r["lf_" + k_].append(np.float32(fi[k_][b]))
''')

# 5. argparse -----------------------------------------------------------------------------------------------------
rep('    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS', '''    p.add_argument("--level-floor", choices=("off", "on"), default="off", help="LEVEL-FLOOR (PREREG_LEVELFLOOR_A3): on = the floor operator F (level_floor.py) lifts the plan's along-track progress to kappa * v_hat from the floor heads, at every replan, after --smooth and before --feasible; off (default) = unchanged bit for bit")
    p.add_argument("--floor-kappa", type=float, default=None, help="LEVEL-FLOOR: the dose kappa in [0, 1]; required with --level-floor on, refused with off")
    p.add_argument("--floor-ckpt", default="", help="LEVEL-FLOOR: the head checkpoint (cfg: level_head on AND lead_head on); required with --level-floor on, refused with off")
    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS''')

# 6. cmd_run ---------------------------------------------------------------------------------------------------------
rep('        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")\n',
    '        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")\n    lf_check_args(a)\n')
rep('    model.to(dev).eval()\n', '    model.to(dev).eval()\n    floor = load_level_floor(a, cfg, ck, dev) if a.level_floor == "on" else None\n')
rep('mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks), keep_heads=bool(a.dump_heads), smooth=a.smooth, feasible=a.feasible)',
    'mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks), keep_heads=bool(a.dump_heads), smooth=a.smooth, feasible=a.feasible, floor=floor)')
rep('rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible, histswap=a.histswap)',
    'rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible, histswap=a.histswap, level_floor=floor is not None)')
rep('''                             **({"histswap": a.histswap} if a.histswap != "off" else {}),
''', '''                             **({"histswap": a.histswap} if a.histswap != "off" else {}), **(floor.meta() if floor is not None else {}),
''')
rep('    os.makedirs(a.out_dir, exist_ok=True)\n    if a.no_analyze:\n', '''    if floor is not None:                                                # only then: --level-floor off / absent => the header (and the result JSON) is byte-identical to the RI tool's
        header["level_floor"], header["floor_kappa"], header["floor_ckpt"], header["floor_ckpt_sha256"] = "on", floor.kappa, floor.ckpt_path, floor.ckpt_sha256
        header["level_floor_detail"] = floor.detail()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.no_analyze:
''')

assert b"\r" not in s.encode("utf-8")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("patched", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())
