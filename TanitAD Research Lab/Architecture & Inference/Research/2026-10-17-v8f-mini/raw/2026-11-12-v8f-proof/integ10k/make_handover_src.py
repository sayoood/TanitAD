"""make_handover_src.py -- writes v8f_t1_eval_reader_rule_rzgfhqs.py from v8f_t1_eval_reader_rule_rzgfhq.py by asserted text edits (each old text exactly once).
Study "handover-src": which INPUT of the state hand-over at the first tick of an aged window carries the state part of the track error. One new option, --t0-state
own|logged|speed_logged|path_logged (design: HANDOVER_SRC_DESIGN.md). The parent, the pinned tool v8f_t1_eval.py and the distrule files are not changed. Standard library only.
Usage: python make_handover_src.py (in this folder). Refuses to overwrite the output (set HS_OVERWRITE_OWN=1 to rewrite it)."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhq.py")
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqs.py")
PARENT_SHA16 = "92a8034cc0f7b0d2"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhq.py -- written by make_longroll10k_b.py from v8f_t1_eval_reader_rule_rzgfhp.py (the first build of the study "longroll"): three repairs, all for P above 0.\n'

HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqs.py -- written by make_handover_src.py from v8f_t1_eval_reader_rule_rzgfhq.py (study "handover-src"): one more parameter.
  --t0-state own|logged|speed_logged|path_logged (default own; accepted only with --preroll-steps above 0, else REFUSED)
  After a pre-roll the model reads, at the FIRST tick of the scored window, the own executed history. This parameter replaces, at that tick only and only in the batch the planner is
  handed, one group of that state by what a fresh window at the same t0 reads (the logged cache). The pre-roll itself, the executor (it continues from the OWN pose), the own state
  that later ticks read, the rule and its geometry (own position, a_m0), and the way the scored track is written (the frame of the logged pose at t0) are not changed.
    own           nothing is replaced. No new statement changes a value: the dumps equal those of the parent bit for bit (the build_item patch is not installed).
    logged        everything the model reads at that tick (the ego history rows of the batch: p10f, base, pose_now / pose_t / pose_s, the route channel nav, and the speed-bucket
                  input d["v"]) is the logged cache of the window: the item is the one a fresh window at the same t0 builds.
    speed_logged  the speed group from the logged cache (v, a = dv/dt, the speed-bucket input d["v"], pose column 3); positions and headings (pose x, y, yaw), the yaw-rate channel
                  and the route channel as own; kappa = clamp(omega / max(v, 2), +-0.3) is recomputed from the own omega and the logged v; p10f and base follow from that kin3.
    path_logged   positions and headings (pose x, y, yaw) and the yaw-rate channel from the logged cache; the speed group and the route channel as own; kappa from the logged
                  omega and the own v; p10f and base follow.
  Rows replaced: the ego history rows the batch reads at that tick (the last hmax rows, up to t0) and, for the speed-bucket input, every row up to t0. The tick-0 ages are 0.
  Mechanism: the pinned build_item is wrapped (T.build_item, restored in finish_preroll) for the main-phase call only (PrerollRun.main_kind); the item is built from a view of the
  state whose arrays are the chosen mixture (HandoverView, handover_shadow). The own item is built once more for the comparison (no counters touched) and the largest absolute
  difference per column group is kept per planner kind: positions (pose x, y), headings (pose yaw, wrapped), speed (p10f column 0), and p10f other columns, base, nav, limit.
  Log: one line per planner kind, "ZZ T0-STATE mode=... kind=...", and the keys t0_state / t0_state_first_tick_max_abs_diff / t0_state_note in preroll_log_<run>.json and t0_state in
  the meta of rule_log_pre_<run>.npz. The tick file is written by the pinned tool and carries no wrapper meta; its tick-0 rows (v0, s_pose, t_pose, plan) show what was handed over.
  The pre-roll offsets file (preroll_offsets_<run>.npz) is identical across the four modes: nothing before the first scored tick is touched.
Text of the parent (rzgfhq) follows.
v8f_t1_eval_reader_rule_rzgfhq.py -- written by make_longroll10k_b.py from v8f_t1_eval_reader_rule_rzgfhp.py (the first build of the study "longroll"): three repairs, all for P above 0.
'''

BLOCK = r'''# ==== HANDOVER BEGIN (study handover-src) ==========================================================================================================
T0_STATES = ("own", "logged", "speed_logged", "path_logged")
HO_EGO = ("pose", "kin3", "p10f", "base", "a", "ch1", "v")      # the ego arrays of the pinned WindowState (WindowState.EGO)


class HandoverView:
    """What the pinned build_item reads of a window state: ci, j, T, d, nav, scene0 (and nothing else)."""

    def __init__(self, st, d, nav):
        self.ci, self.j, self.T, self.d, self.nav, self.scene0 = st.ci, st.j, st.T, d, nav, st.scene0


def _nanmax_abs(x):
    x = np.abs(np.asarray(x, np.float64)).reshape(-1)
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else 0.0


def _wrap_pi(x):
    return (np.asarray(x, np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def handover_shadow(mode, own_d, lg, own_nav, lg_nav, t0, hmax, kappa, kin_feat, kdx):
    """-> (d, nav, info): the ego arrays and the route array the first scored tick of an aged window reads under `mode` (not own). own_d = the state's d (own executed rows up to t0), lg = the logged
    cache dict of the clip (unmodified arrays), own_nav / lg_nav = the state's nav and the logged nav rows of the clip. Arrays are copies; own_d and lg are not changed.
    kappa(omega, v) = the programme's _pose_kappa0 (clamp(omega / max(v, 2), +-0.3)); kin_feat(kin3 [n, 3]) -> [n, 8] (floors_kin.kin_features); kdx(kin3 [n, 3]) -> [n, K] (kdx_base, yaw units).
    info: for the two partial modes the largest |kappa(own omega, own v) - own kin3 kappa| and the same for the logged arrays over the history rows (the fidelity of the recomposition)."""
    if mode not in T0_STATES[1:]:
        raise ValueError("t0 state mode %r" % (mode,))
    t0 = int(t0)
    d = dict(own_d)
    for k in HO_EGO:
        d[k] = np.array(own_d[k], copy=True)
    nav = np.array(own_nav, copy=True)
    sel = slice(0, t0 + 1)
    if mode == "logged":
        for k in HO_EGO:
            d[k][sel] = np.asarray(lg[k])[sel]
        nav[sel] = np.asarray(lg_nav)[sel]
        return d, nav, {}
    hist = np.arange(max(1, t0 - int(hmax) + 1), t0 + 1)
    info = {}
    if mode == "speed_logged":
        d["v"][sel] = np.asarray(lg["v"])[sel]
        d["a"][sel] = np.asarray(lg["a"])[sel]
        d["kin3"][sel, 0:2] = np.asarray(lg["kin3"])[sel, 0:2]
        d["pose"][sel, 3] = np.asarray(lg["pose"])[sel, 3]
        omega = np.asarray(own_d["ch1"], np.float64)
    else:
        d["pose"][sel, 0:3] = np.asarray(lg["pose"])[sel, 0:3]
        d["ch1"][sel] = np.asarray(lg["ch1"])[sel]
        omega = np.asarray(lg["ch1"], np.float64)
    info["kappa_recompose_max_abs_own"] = _nanmax_abs(kappa(np.asarray(own_d["ch1"], np.float64)[hist], np.asarray(own_d["kin3"], np.float64)[hist, 0]) - np.asarray(own_d["kin3"], np.float64)[hist, 2])
    info["kappa_recompose_max_abs_logged"] = _nanmax_abs(kappa(np.asarray(lg["ch1"], np.float64)[hist], np.asarray(lg["kin3"], np.float64)[hist, 0]) - np.asarray(lg["kin3"], np.float64)[hist, 2])
    k3 = np.asarray(d["kin3"], np.float64)
    d["kin3"][hist, 2] = kappa(omega[hist], k3[hist, 0])
    d["p10f"][hist] = np.asarray(kin_feat(np.asarray(d["kin3"], np.float64)[hist])).astype(d["p10f"].dtype)
    d["base"][t0] = np.asarray(kdx(np.asarray(d["kin3"], np.float64)[t0:t0 + 1]))[0].astype(d["base"].dtype)
    return d, nav, info


def item_diff(a, b):
    """the largest absolute difference between two items of the pinned build_item, per column group: positions (pose x, y: the history rows and pose_now / pose_t / pose_s), headings
    (pose yaw, wrapped), speed (p10f column 0), and the others (p10f columns 1.., base, nav, limit, frame_ok flags: the number of rows that differ)."""
    pos, head = 0.0, 0.0
    for key in ("pose", "pose_now", "pose_t", "pose_s"):
        pa, pb = np.atleast_2d(np.asarray(a[key], np.float64)), np.atleast_2d(np.asarray(b[key], np.float64))
        pos = max(pos, _nanmax_abs(pa[:, :2] - pb[:, :2]))
        head = max(head, _nanmax_abs(_wrap_pi(pa[:, 2] - pb[:, 2])))
    fa, fb = np.asarray(a["p10f"], np.float64), np.asarray(b["p10f"], np.float64)
    return {"positions_m": pos, "headings_rad": head, "speed_ms": _nanmax_abs(fa[:, 0] - fb[:, 0]), "p10f_other": _nanmax_abs(fa[:, 1:] - fb[:, 1:]),
            "base": _nanmax_abs(np.asarray(a["base"], np.float64) - np.asarray(b["base"], np.float64)), "nav": _nanmax_abs(np.asarray(a["nav"], np.float64) - np.asarray(b["nav"], np.float64)),
            "limit": _nanmax_abs(np.asarray(a["limit"], np.float64) - np.asarray(b["limit"], np.float64)),
            "frame_ok_rows": float(np.sum(np.asarray(a["frame_ok"], bool) != np.asarray(b["frame_ok"], bool)))}


def ho_stats_add(stats, kind, diff, info):
    s = stats.setdefault(kind, {"states": 0, "states_item_differs": 0, "max_abs": {k: 0.0 for k in diff}, "kappa_recompose_max_abs_own": 0.0, "kappa_recompose_max_abs_logged": 0.0})
    s["states"] += 1
    s["states_item_differs"] += int(any(v > 0.0 for v in diff.values()))
    for k, v in diff.items():
        s["max_abs"][k] = max(s["max_abs"][k], float(v))
    for k in ("kappa_recompose_max_abs_own", "kappa_recompose_max_abs_logged"):
        if k in info:
            s[k] = max(s[k], float(info[k]))


def make_handover_build_item(orig_bi, T, runner, mode, stats):
    """The build_item the pinned rollout calls while --t0-state is not own. Only the first tick of the MAIN phase of a re-based state is touched (runner.main_kind is set by PrerollRun while its
    main call runs); every other call (the pre-roll segments, later ticks) goes to orig_bi unchanged."""
    cache = {}

    def kappa(om, v):
        if "f" not in cache:
            import torch
            from tanitad.models.kinematic_prior import _pose_kappa0
            cache["f"] = (torch, _pose_kappa0)
        torch_, pk = cache["f"]
        return pk(torch_.from_numpy(np.array(om, np.float64)), torch_.from_numpy(np.array(v, np.float64))).numpy()

    def kin_feat(k3):
        return T.fk.kin_features(k3)

    def kdx(k3):
        return T.D.kdx_base(k3, T.D.KIN_UNITS["yaw"])

    def build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats_=None):
        kind = runner.main_kind
        if kind is None or int(tick) != 0 or getattr(st, "frame_pose5", None) is None or not hasattr(st, "lr_base") or int(st.j) != int(st.lr_t0):
            return orig_bi(ds, proxy, st, tick, a_t, a_s, scene, limit, stats_)
        d, nav, info = handover_shadow(mode, st.d, st.lr_base, st.nav, st.lr_nav, st.lr_t0, ds.hmax, kappa, kin_feat, kdx)
        view = HandoverView(st, d, nav)
        it = orig_bi(ds, proxy, view, tick, a_t, a_s, scene, limit, stats_)
        st.scene0 = view.scene0
        ref = orig_bi(ds, proxy, HandoverView(st, st.d, st.nav), tick, a_t, a_s, scene, limit, None)     # the own item, for the comparison only (no counters)
        ho_stats_add(stats, kind, item_diff(it, ref), info)
        return it
    return build_item
# ==== HANDOVER END ===================================================================================================================================


'''


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


ANCHORS = []


def main():
    if os.path.exists(OUT) and os.environ.get("HS_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % OUT)
    raw = open(SRC, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha[:16] == PARENT_SHA16, "parent sha256 %s" % sha[:16]
    t = raw.decode("utf-8")
    assert "\r" not in t
    n0 = t.count("\n")

    def E(old, new, name):
        nonlocal t
        t = once(t, old, new)
        ANCHORS.append(name)

    E(OLD_FIRST, HEADER, "A1 header docstring")
    E("def _tracked():\n", BLOCK + "def _tracked():\n", "A2 HANDOVER block before def _tracked")
    E("        self.hook_rows, self.hook_max_shift = 0, 0.0\n", "        self.hook_rows, self.hook_max_shift = 0, 0.0\n        self.main_kind = None                          # handover-src: the planner kind while the main-phase call runs, else None\n",
      "A3 PrerollRun.__init__ main_kind")
    E("        plan0, exe = orig(planner, states, *args, **kw_main)\n",
      "        self.main_kind = kind\n        try:\n            plan0, exe = orig(planner, states, *args, **kw_main)\n        finally:\n            self.main_kind = None\n",
      "A4 PrerollRun.__call__ main call")
    E('            st.lr_track = np.asarray(tracks[str(J["clip"][st.ci])], np.float64)\n',
      '            st.lr_track = np.asarray(tracks[str(J["clip"][st.ci])], np.float64)\n'
      '            if STATE["t0state"] != "own":              # handover-src: the logged cache of the clip and its nav rows (read-only references / a copy)\n'
      '                st.lr_base = store.get(str(J["clip"][st.ci]))\n'
      '                st.lr_nav = np.array(J["nav"][int(J["off"][st.ci]):int(J["off"][st.ci]) + int(J["n"][st.ci])], np.float32)\n',
      "A5 make_states: lr_base / lr_nav")
    E('    STATE["runner"] = None\n',
      '    STATE["runner"] = None\n    STATE["t0state"] = str(getattr(mine, "t0_state", "own"))\n    STATE["ho_stats"], STATE["ho_orig_bi"] = {}, None\n',
      "A6 install_preroll: STATE t0state")
    E('        if mine.preroll_exec != "plan":\n            raise SystemExit("REFUSED: --preroll-exec goes with --preroll-steps above 0")\n',
      '        if mine.preroll_exec != "plan":\n            raise SystemExit("REFUSED: --preroll-exec goes with --preroll-steps above 0")\n'
      '        if STATE["t0state"] != "own":\n            raise SystemExit("REFUSED: --t0-state %s goes with --preroll-steps above 0 (there is no state hand-over without a pre-roll)" % STATE["t0state"])\n',
      "A7 install_preroll: refusal for P = 0")
    E("    R.distance_rule = make_g0_shim(R.distance_rule, STATE)\n",
      "    R.distance_rule = make_g0_shim(R.distance_rule, STATE)\n"
      '    if STATE["t0state"] != "own":                      # handover-src: not installed for own, so own is the parent\n'
      '        STATE["ho_orig_bi"] = T.build_item\n'
      '        T.build_item = make_handover_build_item(T.build_item, T, runner, STATE["t0state"], STATE["ho_stats"])\n',
      "A8 install_preroll: build_item patch")
    E('    T.make_states, T.rollout = orig_ms, orig_ro\n    Rm, orig_dr = STATE["pre_shim"]\n',
      '    T.make_states, T.rollout = orig_ms, orig_ro\n    if STATE.get("ho_orig_bi") is not None:\n        T.build_item = STATE["ho_orig_bi"]\n    Rm, orig_dr = STATE["pre_shim"]\n',
      "A9 finish_preroll: restore build_item")
    E('        meta["preroll_steps"] = np.asarray(runner.P)\n',
      '        meta["preroll_steps"] = np.asarray(runner.P)\n        meta["t0_state"] = np.asarray(STATE.get("t0state", "own"))\n',
      "A10 finish_preroll: rule_log_pre meta")
    E('    path = os.path.join(folder, "preroll_log_" + run + ".json")\n',
      '    t0m, hs = STATE.get("t0state", "own"), STATE.get("ho_stats") or {}\n'
      '    rec["t0_state"] = t0m\n'
      '    rec["t0_state_first_tick_max_abs_diff"] = hs\n'
      '    rec["t0_state_note"] = "largest absolute difference, over the states of the run and the history rows, between the batch handed to the model at the first scored tick and the own batch; empty for own (nothing replaced)"\n'
      '    if t0m == "own":\n'
      '        print("ZZ T0-STATE mode=own nothing replaced: the frames handed to the model are the own frames (all differences 0.0 by construction)", flush=True)\n'
      '    for kk in sorted(hs):\n'
      '        mx = hs[kk]["max_abs"]\n'
      '        print("ZZ T0-STATE mode=%s kind=%s states=%d states_item_differs=%d max_abs_diff positions_m=%.9g headings_rad=%.9g speed_ms=%.9g | p10f_other=%.9g base=%.9g nav=%.9g limit=%.9g frame_ok_rows=%d | kappa_recompose own=%.3g logged=%.3g"\n'
      '              % (t0m, kk, hs[kk]["states"], hs[kk]["states_item_differs"], mx["positions_m"], mx["headings_rad"], mx["speed_ms"], mx["p10f_other"], mx["base"], mx["nav"], mx["limit"], int(mx["frame_ok_rows"]),\n'
      '                 hs[kk]["kappa_recompose_max_abs_own"], hs[kk]["kappa_recompose_max_abs_logged"]), flush=True)\n'
      '    path = os.path.join(folder, "preroll_log_" + run + ".json")\n',
      "A11 finish_preroll: preroll log keys and the log line")
    E('    ap.add_argument("--preroll-exec", choices=("plan", "logged"), default="plan")\n',
      '    ap.add_argument("--preroll-exec", choices=("plan", "logged"), default="plan")\n    ap.add_argument("--t0-state", choices=T0_STATES, default="own")\n',
      "A12 argparse --t0-state")
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    out_sha = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
    print("wrote", OUT, out_sha, t.count("\n"), "lines (parent %d lines, sha256 %s)" % (n0, sha[:16]))
    print("anchors (%d):" % len(ANCHORS))
    for a in ANCHORS:
        print("  ", a)


if __name__ == "__main__":
    main()
