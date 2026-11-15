#!/usr/bin/env python
"""v8f_t1_eval_rule_mix.py -- card PREREG_V8F_PROOF_ADD29 (RULEATTR).

The distance rule of card ADD26 (note N1) with a MIXED lead source: presence, gap and closing speed each from `label` or `head`
(closing speed also `zero`). The wrapper v8f_t1_eval_rule.py is imported and not edited; the pinned eval tool stays pinned.

  python v8f_t1_eval_rule_mix.py --tree TREE --mix-presence {label,head} --mix-gap {label,head} --mix-closing {label,head,zero}
         [--rule-log NPZ] [--floor {card,cap-only}] run <the argv of the pinned tool's `run`>

Rule for a missing value: on a tick that the presence part calls present, a gap or closing speed whose chosen source has no finite value
is taken from the other source; the counts go into the meta of the rule log.
Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v8f_t1_eval_rule as W  # noqa: E402

R = W.R
MIX = {"presence": "label", "gap": "label", "closing": "label"}
COUNT = {"ticks": 0, "present": 0, "gap_from_other": 0, "closing_from_other": 0, "head_present": 0, "label_present": 0, "both_present": 0}
STATE = {"label": None}


class MixPlanner(W.RulePlanner):
    """RulePlanner with source `mix`; every other mode runs the unchanged parent code."""

    def __init__(self, base, model, mode, m, fk, label=None, log=None, floor="card"):
        super().__init__(base, model, mode, m, fk, label=label, log=log, floor=floor)
        if self.source == "mix":
            self.label = STATE["label"]
            if self.label is None:
                raise ValueError("mix source needs a LabelSource")
            if getattr(getattr(model, "wm", None), "lead_head", None) is None:
                raise SystemExit("REFUSED: the mixed source needs a checkpoint trained with lead_sup on (no wm.lead_head in this model)")

    def __call__(self, batch):
        if self.source != "mix":
            return super().__call__(batch)
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
        a_m = np.where(k > 0, R.progress(g_log, pos)[:, 0], 0.0)
        a_l = np.where(k > 0, R.logged_arc(g_log)[np.arange(B), np.clip(k - 1, 0, 29)], 0.0)
        v_l = np.array([s.logged_v[int(s.j)] if int(s.j) < len(s.logged_v) else np.nan for s in st], np.float64)
        # ---- the two sources, read exactly as the parent reads them
        rd = [self.label.read(int(c), int(jj)) for c, jj in zip(ci, j)]
        lp = np.array([r[0] for r in rd], bool)
        lg = np.array([r[1] for r in rd], np.float64)
        lc = np.array([r[2] for r in rd], np.float64)
        p, g_h, c_h, ok = W.head_values(self.model, batch["feats"], batch["frame_ok"])
        p_src = p.astype(np.float64)
        hg, hc = g_h.astype(np.float64), c_h.astype(np.float64)
        hp = (p >= W.HEAD_P_MIN) & ok
        # ---- the mix
        present = lp.copy() if MIX["presence"] == "label" else hp.copy()
        lab_g, lab_c = np.isfinite(lg), np.isfinite(lc)
        head_g, head_c = np.isfinite(hg), np.isfinite(hc)
        if MIX["gap"] == "label":
            sub_g = present & ~lab_g & head_g
            g_src = np.where(sub_g, hg, lg)
        else:
            sub_g = present & ~head_g & lab_g
            g_src = np.where(sub_g, lg, hg)
        if MIX["closing"] == "label":
            sub_c = present & ~lab_c & head_c
            c_src = np.where(sub_c, hc, lc)
        elif MIX["closing"] == "head":
            sub_c = present & ~head_c & lab_c
            c_src = np.where(sub_c, lc, hc)
        else:
            sub_c = np.zeros(B, bool)
            c_src = np.zeros(B, np.float64)
        COUNT["ticks"] += int(B)
        COUNT["gap_from_other"] += int(sub_g.sum())
        COUNT["closing_from_other"] += int(sub_c.sum())
        COUNT["head_present"] += int(hp.sum())
        COUNT["label_present"] += int(lp.sum())
        COUNT["both_present"] += int((hp & lp).sum())
        # ---- the unchanged tail of the parent
        with np.errstate(invalid="ignore"):
            src_bad = present & ~(np.isfinite(g_src) & np.isfinite(c_src) & np.isfinite(v_l))
            present = present & ~src_bad
            v_L = np.maximum(0.0, v_l + c_src)
            g0 = g_src + a_l - a_m
        COUNT["present"] += int(present.sum())
        out = R.distance_rule(plan, v0, present, g_src, g0, v_L, m=self.m, **self.kw)
        self.log.add(ci=ci, t0=t0, j=j, k=k, tick=k // max(self.m, 1), active=out.active, present=present, src_bad=src_bad, sat=out.n_sat,
                     sat_exec=out.n_sat_exec, lift=out.n_lift, changed=out.changed, g_src=g_src, c_src=c_src, p_src=p_src, v0=v0, v_l=v_l,
                     v_L=v_L, a_l=a_l, a_m=a_m, g0=g0)
        return out.plan


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--mix-presence", choices=("label", "head"), required=True)
    ap.add_argument("--mix-gap", choices=("label", "head"), required=True)
    ap.add_argument("--mix-closing", choices=("label", "head", "zero"), required=True)
    ap.add_argument("--rule-log", default="")
    ap.add_argument("--floor", choices=W.FLOORS, default="card")
    mine, rest = ap.parse_known_args(argv)
    T, path = W.load_pinned(mine.tree)
    a = T.build_parser().parse_args(rest)
    if a.mode != "run":
        raise SystemExit("REFUSED: the mixed source applies to the `run` mode of the pinned tool only")
    pin = W.sha256_file(path)
    if pin != W.PINNED_SHA256:
        raise SystemExit(f"REFUSED: pinned tool sha256 {pin} != {W.PINNED_SHA256}")
    assert abs(T.M.LEAD_SUP_GAP_SCALE_M - W.LEAD_GAP_SCALE_M) == 0 and abs(T.M.LEAD_SUP_CLOS_SCALE_MS - W.LEAD_CLOS_SCALE_MS) == 0
    MIX.update(presence=mine.mix_presence, gap=mine.mix_gap, closing=mine.mix_closing)
    W.MODE_CFG["mix"] = ("mix", {})
    W.RulePlanner = MixPlanner
    J = T.D.load_join(a.eval_join)
    STATE["label"] = W.LabelSource(J)
    log = W.RuleLog()
    made, restore = W.install(T, "mix", a, None, log, mine.floor)
    try:
        rc = T.cmd_run(a)
    finally:
        restore()
        if mine.rule_log:
            meta = {"mode": "mix", "source": "mix", "mix": json.dumps(MIX, sort_keys=True), "mix_counts": json.dumps(COUNT, sort_keys=True),
                    "D": R.D_M, "TAU": R.TAU_S, "B_MAX": R.B_MAX, "G_MAX": R.G_MAX_M, "replan_steps": a.replan_steps,
                    "pinned_path": path, "pinned_sha256": pin, "ckpt": a.ckpt, "tag": a.tag, "eval_join": a.eval_join,
                    "join_sha256": W.sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "mode_kwargs": "{}", "floor": mine.floor}
            log.save(mine.rule_log, meta)
            print("wrote", mine.rule_log, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
