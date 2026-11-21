#!/usr/bin/env python3
"""v8f_t1_eval_rule.py -- the DISTRULE wrapper (card PREREG_V8F_PROOF_ADD26) around the pinned T1-kin tool `v8f_t1_eval.py`.

Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance).

  python v8f_t1_eval_rule.py --tree <repo root or stack root> [--dist-rule {off,label,head,never,push,plumb}] [--rule-log NPZ] run <the argv of the pinned tool's `run`>

The pinned tool is imported from --tree by file path and is NOT edited (sha256 recorded in the rule log). Two of its module globals are replaced for the
call of its own `cmd_run`: `model_planner` (returns a RulePlanner = the model planner followed by the rule) and `rollout` (binds the window states to the
RulePlanner); `make_states` is wrapped to attach the logged speed row of each clip.  `--dist-rule off` (default) installs nothing.
Where the rule sits, the sources and every open choice of the card: DISTRULE_DEFS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dist_rule as R  # noqa: E402

PINNED_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"
MODES = ("off", "label", "head", "never", "push", "plumb")
#: mode -> (source, keyword arguments of dist_rule.distance_rule); `never` is the constant set of control r2 as written in the card, `plumb` and `push` see DISTRULE_DEFS E5
MODE_CFG = {"label": ("label", {}),
            "head": ("head", {}),
            "never": ("label", {"D": -1e9, "tau": 0.0}),
            "push": ("label", {"push_mps": R.PUSH_MPS}),
            "plumb": ("label", {"use_cap": False, "use_floor": False})}
GAP_MAX_M = 60.0
STEP_MAX_M = 3.0
RUN_MAX, RUN_MIN = 5, 3
HEAD_P_MIN = 0.5
LEAD_GAP_SCALE_M, LEAD_CLOS_SCALE_MS = 50.0, 10.0   # v8f_mini.LEAD_SUP_GAP_SCALE_M / LEAD_SUP_CLOS_SCALE_MS (asserted against the tree at run time)
FLOORS = ("card", "cap-only")      # card = the projection as written; cap-only = builder option, DISTRULE_DEFS E3b (needs a card amendment before it is read)
LOG_FIELDS = ("ci", "t0", "j", "k", "tick", "active", "present", "src_bad", "sat", "sat_exec", "lift", "changed",
              "g_src", "c_src", "p_src", "v0", "v_l", "v_L", "a_l", "a_m", "g0")


def sha256_file(path, blk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================================ the pinned tool
def find_pinned(tree):
    for rel in ("stack/scripts/v8f_t1_eval.py", "scripts/v8f_t1_eval.py"):
        p = os.path.join(tree, rel)
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise SystemExit(f"REFUSED: no v8f_t1_eval.py under --tree {tree} (looked for stack/scripts/ and scripts/)")


def load_pinned(tree):
    """Import the pinned tool by file path from the tree; assert that `tanitad` was taken from that same tree."""
    path = find_pinned(tree)
    spec = importlib.util.spec_from_file_location("v8f_t1_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval"] = mod
    spec.loader.exec_module(mod)
    root = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(path), "..", "..")))
    got = os.path.normcase(os.path.abspath(mod.M.__file__))
    if not got.startswith(root):
        raise SystemExit(f"REFUSED: tanitad was imported from {got}, not from the tree {root} (a stale editable install shadows it)")
    return mod, path


# ============================================================================================ sources
class LabelSource:
    """Privileged source: presence, gap and closing speed from the label join at the tick frame, rows at or before the tick only (DISTRULE_DEFS E4)."""

    def __init__(self, J):
        self.off = np.asarray(J["off"], np.int64)
        self.n = np.asarray(J["n"], np.int64)
        self.valid = np.asarray(J["lead_valid"])
        self.gap = np.asarray(J["lead_gap"], np.float64)

    def _ok(self, r):
        g = self.gap[r]
        return bool(self.valid[r] == 1 and np.isfinite(g) and 0.0 <= g <= GAP_MAX_M)

    def read(self, ci, j):
        """-> (present, gap, closing).  closing = LS slope of the trailing run (<= 5 frames, steps <= 3.0 m, >= 3 frames) else 0.0."""
        o, n = int(self.off[ci]), int(self.n[ci])
        if j < 0 or j >= n:
            return False, float("nan"), 0.0
        r = o + int(j)
        if not self._ok(r):
            return False, float("nan"), 0.0
        idx = [r]
        f = r - 1
        while f >= o and len(idx) < RUN_MAX and self._ok(f) and abs(self.gap[f + 1] - self.gap[f]) <= STEP_MAX_M:
            idx.append(f)
            f -= 1
        g = float(self.gap[r])
        if len(idx) < RUN_MIN:
            return True, g, 0.0
        idx = idx[::-1]
        G = self.gap[idx]
        t = np.arange(len(idx), dtype=np.float64) * R.DT
        tb = t.mean()
        return True, g, float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())


def head_values(model, feats, frame_ok=None, hist=None):
    """The LEADSUP head on the tokens of the last `hist` frames of the batch (frames j-4 .. j).  -> p [B], gap_m [B], clos_ms [B], ok [B] (all hist frames real).
    The formula of lead_dump.head_outputs (kind feats): adapter + pos -> z1 -> head; the single end frame."""
    import torch
    hist = int(model.wm.lead_head.hist if hist is None else hist)
    with torch.no_grad():
        f = feats[:, -hist:]
        z1 = model.wm.z1(model.adapter(f.float()) + model.pos[None, None])
        o = model.wm.lead_head(z1)[:, 0].float().cpu().numpy()
    p = (1.0 / (1.0 + np.exp(-o[:, 0]))).astype(np.float32)
    ok = np.ones(len(o), bool) if frame_ok is None else frame_ok[:, -hist:].all(-1).cpu().numpy()
    return p, (o[:, 1] * LEAD_GAP_SCALE_M).astype(np.float32), (o[:, 2] * LEAD_CLOS_SCALE_MS).astype(np.float32), ok


# ============================================================================================ the planner
class RuleLog:
    def __init__(self):
        self.rows = {k: [] for k in LOG_FIELDS}

    def add(self, **cols):
        for k in LOG_FIELDS:
            self.rows[k].append(np.asarray(cols[k]))

    def arrays(self):
        out = {}
        for k in LOG_FIELDS:
            v = self.rows[k]
            out[k] = np.concatenate(v) if v else np.zeros(0)
        return out

    def save(self, path, meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.arrays(), **{k: np.asarray(v) for k, v in meta.items()})


def attach_logged(states, store, J):
    """Attach the logged cache speed row of each window's clip (`base["v"]`, the array `wm_speed` is built from) to its state.  A state's own `d["v"]` holds own-executed rows after t0."""
    for st in states:
        st.logged_v = np.asarray(store.get(str(J["clip"][st.ci]))["v"], np.float64)
    return states


class RulePlanner:
    """The model planner followed by the distance rule.  `base` is the pinned model planner; it keeps its own `keys_read`, `aux`, `heads`."""

    def __init__(self, base, model, mode, m, fk, label=None, log=None, floor="card"):
        if mode not in MODE_CFG:
            raise ValueError(f"mode must be one of {tuple(MODE_CFG)}")
        self.base, self.model, self.mode, self.m, self.fk = base, model, mode, int(m), fk
        self.source, kw = MODE_CFG[mode]
        if floor not in FLOORS:
            raise ValueError(f"floor must be one of {FLOORS}")
        self.kw = dict(kw, floor_lifts=(floor == "card"))
        self.label = label
        self.log = log if log is not None else RuleLog()
        self.states = None
        if self.source == "label" and label is None:
            raise ValueError("label source needs a LabelSource")
        if self.source == "head" and getattr(getattr(model, "wm", None), "lead_head", None) is None:
            raise SystemExit("REFUSED: --dist-rule head needs a checkpoint trained with lead_sup on (no wm.lead_head in this model)")

    keys_read = property(lambda self: self.base.keys_read)
    aux = property(lambda self: self.base.aux)
    heads = property(lambda self: self.base.heads)

    def bind(self, states):
        self.states = states

    def __call__(self, batch):
        plan = self.base(batch)
        st = self.states
        if st is None or len(st) != plan.shape[0]:
            raise RuntimeError("RulePlanner not bound to the states of this batch (rollout shim missing)")
        B = len(st)
        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()
        ci = np.array([s.ci for s in st], np.int64)
        t0 = np.array([s.t0 for s in st], np.int64)
        j = np.array([s.j for s in st], np.int64)
        k = j - t0
        g_log = np.stack([np.asarray(s.d["gt"][s.t0], np.float64) for s in st])
        pos = np.stack([self.fk.gt_waypoints(s.track[0::2], np.array([s.t0]), np.array([int(kk)]))[0, 0] for s, kk in zip(st, k)])[:, None, :]
        a_m = np.where(k > 0, R.progress(g_log, pos)[:, 0], 0.0)           # a_m(0) = a_l(0) = 0 by definition (HEADWAY D4); the projection of the origin carries 1e-13 of rounding
        a_l = np.where(k > 0, R.logged_arc(g_log)[np.arange(B), np.clip(k - 1, 0, 29)], 0.0)
        v_l = np.array([s.logged_v[int(s.j)] if int(s.j) < len(s.logged_v) else np.nan for s in st], np.float64)
        p_src = np.full(B, np.nan)
        if self.source == "label":
            rd = [self.label.read(int(c), int(jj)) for c, jj in zip(ci, j)]
            present = np.array([r[0] for r in rd], bool)
            g_src = np.array([r[1] for r in rd], np.float64)
            c_src = np.array([r[2] for r in rd], np.float64)
        else:
            p, g_h, c_h, ok = head_values(self.model, batch["feats"], batch["frame_ok"])
            p_src, g_src, c_src = p.astype(np.float64), g_h.astype(np.float64), c_h.astype(np.float64)
            present = (p >= HEAD_P_MIN) & ok
        with np.errstate(invalid="ignore"):
            src_bad = present & ~(np.isfinite(g_src) & np.isfinite(c_src) & np.isfinite(v_l))
            present = present & ~src_bad
            v_L = np.maximum(0.0, v_l + c_src)
            g0 = g_src + a_l - a_m
        out = R.distance_rule(plan, v0, present, g_src, g0, v_L, m=self.m, **self.kw)
        self.log.add(ci=ci, t0=t0, j=j, k=k, tick=k // max(self.m, 1), active=out.active, present=present, src_bad=src_bad, sat=out.n_sat,
                     sat_exec=out.n_sat_exec, lift=out.n_lift, changed=out.changed, g_src=g_src, c_src=c_src, p_src=p_src, v0=v0, v_l=v_l,
                     v_L=v_L, a_l=a_l, a_m=a_m, g0=g0)
        return out.plan


# ============================================================================================ main
def install(T, mode, a, J=None, log=None, floor="card"):
    """Replace model_planner / rollout / make_states of the pinned module T for one call of T.cmd_run.  Returns (planners list, restore function)."""
    orig = {k: getattr(T, k) for k in ("model_planner", "rollout", "make_states")}
    made = []
    label = LabelSource(J) if MODE_CFG[mode][0] == "label" else None
    if MODE_CFG[mode][0] == "head":
        assert abs(T.M.LEAD_SUP_GAP_SCALE_M - LEAD_GAP_SCALE_M) == 0 and abs(T.M.LEAD_SUP_CLOS_SCALE_MS - LEAD_CLOS_SCALE_MS) == 0

    def model_planner(model, device, keep_aux=False, keep_heads=False):
        base = orig["model_planner"](model, device, keep_aux=keep_aux, keep_heads=keep_heads)
        rp = RulePlanner(base, model, mode, a.replan_steps, T.fk, label=label, log=log, floor=floor)
        made.append(rp)
        return rp

    def make_states(store, tracks, J_, wins, nav_mode):
        out = orig["make_states"](store, tracks, J_, wins, nav_mode)
        return attach_logged(out, store, J_)

    def rollout(planner, states, *args, **kw):
        if isinstance(planner, RulePlanner):
            planner.bind(states)
        return orig["rollout"](planner, states, *args, **kw)

    T.model_planner, T.rollout, T.make_states = model_planner, rollout, make_states

    def restore():
        for k, v in orig.items():
            setattr(T, k, v)
    return made, restore


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--dist-rule", choices=MODES, default="off")
    ap.add_argument("--rule-log", default="")
    ap.add_argument("--floor", choices=FLOORS, default="card", help="card (default) = the projection as written; cap-only = DISTRULE_DEFS E3b, not the card")
    mine, rest = ap.parse_known_args(argv)
    T, path = load_pinned(mine.tree)
    a = T.build_parser().parse_args(rest)
    if a.mode != "run":
        if mine.dist_rule != "off":
            raise SystemExit("REFUSED: --dist-rule applies to the `run` mode of the pinned tool only")
        return T.main(rest)
    if mine.dist_rule == "off":
        if mine.rule_log:
            raise SystemExit("REFUSED: --rule-log with --dist-rule off would write nothing (a flag that silently writes nothing is worse than none)")
        return T.cmd_run(a)
    pin = sha256_file(path)
    if pin != PINNED_SHA256:
        print(f"WARNING: pinned tool sha256 {pin} != {PINNED_SHA256}", file=sys.stderr, flush=True)
    src = MODE_CFG[mine.dist_rule][0]
    J = T.D.load_join(a.eval_join) if src == "label" else None
    log = RuleLog()
    made, restore = install(T, mine.dist_rule, a, J, log, mine.floor)
    try:
        rc = T.cmd_run(a)
    finally:
        restore()
        if mine.rule_log:
            meta = {"mode": mine.dist_rule, "source": src, "D": R.D_M, "TAU": R.TAU_S, "B_MAX": R.B_MAX, "G_MAX": R.G_MAX_M, "replan_steps": a.replan_steps,
                    "pinned_path": path, "pinned_sha256": pin, "ckpt": a.ckpt, "tag": a.tag, "eval_join": a.eval_join,
                    "join_sha256": sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "mode_kwargs": json.dumps(MODE_CFG[mine.dist_rule][1]), "floor": mine.floor}
            log.save(mine.rule_log, meta)
            print("wrote", mine.rule_log, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
