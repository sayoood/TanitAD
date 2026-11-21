"""check_rule_every.py -- tool check of the parameter --rule-every of v8f_t1_eval_reader_rule_rzf.py (card PREREG_V8F_PROOF_ADD62, controls k4, k5 and the wrapper part of k1, k2, k3, k6),
no model, no GPU, no data of any card: synthetic windows on circular roads, a lead on the same road, a plan that follows the road. The real ReaderPlanner of the wrapper is driven
(tick rule through X.MixPlanner, then the later looks), with the real LabelSource, the real pinned tool functions (plan_to_world, envelope, floors_kin) and a fake token reader.
Printed: what each check measured. Run on the pod: CUDA_VISIBLE_DEVICES=-1 DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python check_rule_every.py <tree> [wrapper.py]
Last line: ZZ CHECK-RULE-EVERY all hold: True/False   (True only if every check holds, the planted fault included: the fault run must FAIL the direction check)."""
import hashlib
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("DISTRULE_DIR", HERE))
sys.path.insert(0, HERE)
import v8f_t1_eval_rule as W  # noqa: E402
import v8f_t1_eval_rule_mix as X  # noqa: E402
import dist_rule as R  # noqa: E402

WRAP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "v8f_t1_eval_reader_rule_rzf.py")
spec = importlib.util.spec_from_file_location("v8f_t1_eval_reader_rule_rzf", WRAP)
Z = importlib.util.module_from_spec(spec)
sys.modules["v8f_t1_eval_reader_rule_rzf"] = Z
spec.loader.exec_module(Z)
import torch  # noqa: E402

T, path = W.load_pinned(sys.argv[1])
Z.STATE["T"] = T
WSHA = hashlib.sha256(open(WRAP, "rb").read()).hexdigest()
print("wrapper rzf sha256 %s" % WSHA)
print("pinned tool %s" % os.path.basename(path))

DT, M, N, TT, T0 = 0.2, 10, 30, 60, 12
OK = []


def verdict(name, cond, detail):
    OK.append(bool(cond))
    print("%-62s %s | %s" % (name, "holds" if cond else "DIFFERENT", detail))


# ---------------------------------------------------------------- synthetic world
class Road:
    """circular road, signed curvature kap, start (x0, y0), heading th0; arc coordinate s from the start"""

    def __init__(self, x0, y0, th0, kap):
        self.x0, self.y0, self.th0, self.kap = x0, y0, th0, kap
        self.Rr = 1.0 / kap
        self.cx, self.cy = x0 - self.Rr * math.sin(th0), y0 + self.Rr * math.cos(th0)

    def xy(self, s):
        s = np.asarray(s, np.float64)
        th = self.th0 + self.kap * s
        return np.stack([self.x0 + self.Rr * (np.sin(th) - math.sin(self.th0)), self.y0 - self.Rr * (np.cos(th) - math.cos(self.th0))], -1)

    def th(self, s):
        return self.th0 + self.kap * np.asarray(s, np.float64)

    def arc_of(self, p):
        p = np.asarray(p, np.float64)
        th = np.arctan2((p[..., 0] - self.cx) / self.Rr, -(p[..., 1] - self.cy) / self.Rr)
        return ((th - self.th0 + math.pi) % (2 * math.pi) - math.pi) / self.kap


def to_ego(world, pose):
    d = np.asarray(world, np.float64) - np.asarray(pose[:2], np.float64)
    c, s = math.cos(-pose[2]), math.sin(-pose[2])
    return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], -1)


class State:
    pass


def build(kind, B, seed, with_tokens=False):
    rng = np.random.default_rng(seed)
    roads, sts, gaps, sl_all, ve_all, vp_all = [], [], [], [], [], []
    for b in range(B):
        kap = rng.choice([-1.0, 1.0]) * rng.uniform(1 / 600, 1 / 100)
        rd = Road(rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-3, 3), kap)
        ve = rng.uniform(8, 20)
        vp = ve * rng.uniform(0.9, 1.0)
        gap0 = 1.2 * vp + rng.uniform(1.5, 3.0)
        decel = 0.0 if kind == "hold" else rng.uniform(5.0, 8.0)
        f = np.arange(TT)
        se = ve * DT * (f - T0)
        sl = np.zeros(TT)
        sl[:T0 + 1] = gap0 + se[:T0 + 1]
        for ff in range(T0 + 1, TT):
            sl[ff] = sl[ff - 1] + max(0.0, ve - decel * DT * (ff - T0)) * DT
        gap = sl - se
        s = State()
        s.ci, s.t0, s.j = b, T0, T0
        gt = np.zeros((TT, N, 2))
        gt[T0] = to_ego(rd.xy(ve * DT * np.arange(1, N + 1)), (rd.x0, rd.y0, rd.th0))
        pose = np.zeros((TT, 3))
        pose[T0] = (rd.x0, rd.y0, rd.th0)
        p10f = np.full((TT, 3), np.nan)
        p10f[:T0 + 1] = 1.0
        s.d = {"gt": gt, "pose": pose, "p10f": p10f}
        if with_tokens:
            s.d["feats"] = (rng.normal(0.0, 0.5, (TT, 2, 3))).astype(np.float16)
        s.track = np.zeros((2 * TT, 4))
        s.track[2 * T0] = (rd.x0, rd.y0, rd.th0, ve)
        s.logged_v = np.full(TT, ve)
        roads.append(rd)
        sts.append(s)
        gaps.append(gap)
        sl_all.append(sl)
        ve_all.append(ve)
        vp_all.append(vp)
    J = {"off": np.arange(B) * TT, "n": np.full(B, TT), "lead_valid": np.ones(B * TT, np.int8), "lead_gap": np.concatenate(gaps)}
    plan = np.zeros((B, N, 2), np.float32)
    for b in range(B):
        a = vp_all[b] * DT * np.arange(1, N + 1)
        plan[b] = to_ego(roads[b].xy(a), (roads[b].x0, roads[b].y0, roads[b].th0)).astype(np.float32)
    return {"roads": roads, "states": sts, "J": J, "plan": plan, "sl": sl_all, "ve": np.array(ve_all), "vp": np.array(vp_all), "B": B}


class FakeReader:
    hist = 5

    def read(self, tokens):
        t = tokens.float().numpy() if hasattr(tokens, "float") else np.asarray(tokens, np.float32)
        s = t.reshape(len(t), -1).mean(1)
        return (1.0 / (1.0 + np.exp(-30.0 * s))).astype(np.float32), (15.0 + 100.0 * np.abs(s)).astype(np.float32)


def make_planner(sc, form, every, sts=None):
    sts = sc["states"] if sts is None else sts
    Z.STATE.update(form=form, every=every, reader=None, xs=None, ys=None)
    Z.STATE["mid"] = {"rows": {f: [] for f in Z.MID_FIELDS}, "rd": {k: [] for k in ("p", "g_read", "g_used", "ok")}, "p1": {"ci": [], "t0": [], "plan": []}, "env": {"plans": 0, "hits": 0}}
    Z.STATE["cols"] = {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}
    if form == "RLc":
        mix = ("label", "label", "label")
        W.head_values = Z.head_values_none
    else:
        mix = ("head", "head", "zero")
        W.head_values = Z.head_values_reader
        Z.STATE["reader"], Z.STATE["xs"], Z.STATE["ys"] = FakeReader(), np.array([0.0, 100.0]), np.array([0.0, 100.0])
        Z.reader_closing = lambda rd, tok: np.zeros(len(tok), np.float32)
    W.HEAD_P_MIN = Z.P_MIN
    X.MIX.update(presence=mix[0], gap=mix[1], closing=mix[2])
    W.MODE_CFG["mix"] = ("mix", {})
    X.STATE["label"] = W.LabelSource(sc["J"])
    plan0 = sc["plan"]
    base = lambda batch: np.array(plan0, copy=True)  # noqa: E731
    pl = Z.ReaderPlanner(base, None, "mix", M, T.fk, label=None, log=W.RuleLog(), floor="cap-only")
    pl.bind(sts)
    return pl


def batch_of(sts, v0):
    B = len(sts)
    p10f = torch.zeros((B, 10, 3), dtype=torch.float64)
    p10f[:, -1, 0] = torch.tensor(np.asarray(v0, np.float64))
    feats = torch.zeros((B, 10, 2, 3), dtype=torch.float16)
    fo = torch.ones((B, 10), dtype=torch.bool)
    for b, s in enumerate(sts):
        if "feats" in s.d:
            feats[b] = torch.tensor(np.stack([s.d["feats"][int(s.j) - 9 + i] for i in range(10)]))
    return {"p10f": p10f, "feats": feats, "frame_ok": fo}


def call(sc, form, every, sts=None, v0=None):
    sts = sc["states"] if sts is None else sts
    pl = make_planner(sc, form, every, sts)
    out = pl(batch_of(sts, sc["vp"] if v0 is None else v0))
    return pl, out


def mid_rows(q=None):
    M_ = Z.STATE["mid"]["rows"]
    d = {f: (np.concatenate(M_[f]) if M_[f] else np.zeros(0)) for f in Z.MID_FIELDS}
    if q is not None:
        m = d["q"] == q
        d = {f: v[m] for f, v in d.items()}
    return d


def knot_arcs(sc, plan):
    out = np.zeros((sc["B"], N))
    for b in range(sc["B"]):
        rd = sc["roads"][b]
        pose = (rd.x0, rd.y0, rd.th0)
        out[b] = rd.arc_of(T.plan_to_world(np.asarray(plan[b], np.float64), pose))
    return out


def end_gap(sc, plan, step=10):
    a = knot_arcs(sc, plan)
    return np.array([sc["sl"][b][T0 + step] - a[b, step - 1] for b in range(sc["B"])])


# ---------------------------------------------------------------- k5: the rate is live and acts in the known direction (brake scenario), equal plans when the lead holds
B = 48
sc_b = build("brake", B, 62)
sc_h = build("hold", B, 63)
pl_t, p_tick = call(sc_b, "RLc", 10)                              # the rule at the tick only
pl_5, p_5 = call(sc_b, "RLc", 5)
rows5 = mid_rows()
g_t, g_5 = end_gap(sc_b, p_tick), end_gap(sc_b, p_5)
n_far = int((g_5 > g_t + 1e-6).sum())
verdict("k5 brake: rule every 5 ends the block farther from the lead", bool((g_5 >= g_t - 1e-9).all() and n_far >= int(0.9 * B)),
        "windows %d, strictly farther %d, none closer %s, median gain %.3f m, max gain %.3f m" % (B, n_far, bool((g_5 >= g_t - 1e-9).all()), float(np.median(g_5 - g_t)), float((g_5 - g_t).max())))
pl_th, p_th = call(sc_h, "RLc", 10)
pl_h5, p_h5 = call(sc_h, "RLc", 5)
eq_hold = bool(np.array_equal(p_th, p_h5))
verdict("k5 hold: lead holding its speed, the two plans are equal", eq_hold, "windows %d, bit-identical plans %s, later looks that changed a plan %d" % (B, eq_hold, int(mid_rows()["changed"].sum())))

# the planted fault: the later look is fed the tick's own inputs; the same direction check must FAIL
orig_dr = R.distance_rule
saved = {}


def faulty(plan, v0, present, g_src, g0, v_L, m=10, **kw):
    if m == M:
        saved.update(v0=v0, present=present, g_src=g_src, g0=g0, v_L=v_L)
    else:
        v0, present, g_src, g0, v_L = saved["v0"], saved["present"], saved["g_src"], saved["g0"], saved["v_L"]
    return orig_dr(plan, v0, present, g_src, g0, v_L, m=m, **kw)


R.distance_rule = faulty
try:
    pl_f, p_f = call(sc_b, "RLc", 5)
finally:
    R.distance_rule = orig_dr
g_f = end_gap(sc_b, p_f)
n_far_f = int((g_f > g_t + 1e-6).sum())
fault_pass = bool((g_f >= g_t - 1e-9).all() and n_far_f >= int(0.9 * B))
verdict("k5 planted fault (look fed the tick's own inputs) FAILS the direction check", not fault_pass,
        "strictly farther %d of %d (the real run: %d); the direction check on the fault run reads %s" % (n_far_f, B, n_far, fault_pass))

# ---------------------------------------------------------------- k4: cap only
ex = rows5["excess"]
arcs_t, arcs_5 = knot_arcs(sc_b, p_tick), knot_arcs(sc_b, p_5)
geo = float((arcs_5 - arcs_t).max())
verdict("k4 cap only: logged excess of every later look (float64 rule output)", bool(len(ex) == B and ex.max() <= 1e-9), "looks %d, largest excess %.3e m (limit 1e-9)" % (len(ex), float(ex.max())))
verdict("k4 cap only: final plan never farther along the road than the tick plan", geo <= 1e-3, "largest excess %.3e m by an independent road-arc measure (chord sag of the plan path and float32 knots allow 1e-3)" % geo)
pl_2, p_2 = call(sc_b, "RLc", 2)
rows2 = mid_rows()
arcs_2 = knot_arcs(sc_b, p_2)
dd = arcs_2[:, :M] - arcs_5[:, :M]
ahead = float(dd.max())
n_ahead = int((dd.max(axis=1) > 1e-3).sum())
print("INFO (not a check of the tool; the card's second clause of control k4 is read on the real dumps): rule every 2 against rule every 5 at the first block, synthetic brake windows: "
      "largest excess over steps 1 to 10 %.3e m, windows ahead (> 1e-3 m) at some step %d of %d, at step 10 %d; per-step largest excess %s" % (
          ahead, n_ahead, B, int((dd[:, M - 1] > 1e-3).sum()), " ".join("%.3f" % float(dd[:, i].max()) for i in range(M))))
verdict("k4 cap only, every 2: logged excess of every later look", bool(rows2["excess"].max() <= 1e-9), "looks %d, largest excess %.3e m" % (len(rows2["excess"]), float(rows2["excess"].max())))

# ---------------------------------------------------------------- k2 (wrapper part): steps up to q never change; the first look is the old rule
keep5 = bool(np.array_equal(np.asarray(p_5)[:, :5], np.asarray(p_tick)[:, :5]))
keep2 = bool(np.array_equal(np.asarray(p_2)[:, :2], np.asarray(p_tick)[:, :2]))
unch = np.nonzero(~rows5["changed"].astype(bool))[0]
keepu = bool(np.array_equal(np.asarray(p_5)[unch], np.asarray(p_tick)[unch]))
verdict("k2 steps up to q are not changed; a look that changes nothing leaves the plan bit for bit", keep5 and keep2 and keepu,
        "steps 1-5 equal (every 5) %s, steps 1-2 equal (every 2) %s, windows with an unchanged look %d: plan equal %s" % (keep5, keep2, len(unch), keepu))
tr = pl_5.log.arrays()
tr_t = pl_t.log.arrays()
same_tick = all(np.array_equal(np.asarray(tr[c]), np.asarray(tr_t[c]), equal_nan=True) for c in W.LOG_FIELDS)
verdict("k2 the tick rows of the log do not depend on the parameter", same_tick, "one row per window and tick: %d rows in each, columns %d" % (len(tr["ci"]), len(W.LOG_FIELDS)))

# ---------------------------------------------------------------- (g) every 10: nothing new runs
pl_x = make_planner(sc_b, "RLc", 10)
plan_a = pl_x(batch_of(sc_b["states"], sc_b["vp"]))
pl_y = make_planner(sc_b, "RLc", 10)
plan_b = X.MixPlanner.__call__(pl_y, batch_of(sc_b["states"], sc_b["vp"]))
none_rows = sum(len(v) for v in Z.STATE["mid"]["rows"].values()) + len(Z.STATE["mid"]["p1"]["ci"])
verdict("k1 (wrapper part): --rule-every 10 is the rule at the tick only", bool(np.array_equal(plan_a, plan_b) and none_rows == 0),
        "plan equal to the unextended tick path %s, later-look and p1 rows written %d" % (bool(np.array_equal(plan_a, plan_b)), none_rows))
pl_z, plan_c = call(sc_b, "RLc", 12)                               # above the replan interval: also nothing
verdict("k1 (wrapper part): --rule-every above the replan interval is the rule at the tick only", bool(np.array_equal(plan_c, plan_b)), "equal %s" % bool(np.array_equal(plan_c, plan_b)))

# ---------------------------------------------------------------- k6 (counts): later-look rows = windows x ticks x (10 / h - 1)
n5, n2 = len(rows5["q"]), len(rows2["q"])
verdict("k6 counts of the later-look rows (one tick)", n5 == B * (M // 5 - 1) and n2 == B * (M // 2 - 1) and sorted(set(rows2["q"].astype(int).tolist())) == [2, 4, 6, 8],
        "every 5: %d rows (expected %d), every 2: %d rows (expected %d), q values %s" % (n5, B * 1, n2, B * 4, sorted(set(rows2["q"].astype(int).tolist()))))

# ---------------------------------------------------------------- k3 (wrapper part): the look reads what a real tick on that frame reads
def second_tick(sc, plan_after, q):
    sts2 = []
    for b, s in enumerate(sc["states"]):
        rd = sc["roads"][b]
        pose1 = np.asarray(s.d["pose"][int(s.j)], np.float64)
        wxy = T.plan_to_world(np.asarray(plan_after[b][q - 1:q], np.float64), pose1)[0]
        yaw = float(rd.th(rd.arc_of(wxy)))
        s2 = State()
        s2.ci, s2.t0, s2.j = s.ci, s.t0, int(s.j) + q
        s2.d = {k: np.array(v, copy=True) for k, v in s.d.items()}
        s2.d["pose"][s2.j] = (wxy[0], wxy[1], yaw)
        s2.d["p10f"][:s2.j + 1] = 1.0
        s2.track = np.array(s.track, copy=True)
        s2.track[2 * s2.j] = (wxy[0], wxy[1], yaw, 1.0)
        s2.logged_v = s.logged_v
        sts2.append(s2)
    return sts2


def cmp_rows(mid, tick, cols):
    worst, same_int = 0.0, True
    for c in cols:
        x, y = np.asarray(mid[c], np.float64), np.asarray(tick[c], np.float64)
        if len(x) != len(y) or not np.array_equal(np.isfinite(x), np.isfinite(y)):
            return float("inf"), False
        f = np.isfinite(x)
        worst = max(worst, float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0)
    return worst, same_int


for kind, sc in (("brake", sc_b), ("hold", sc_h)):
    plX, outX = call(sc, "RLc", 5)
    mid5 = mid_rows(5)
    sts2 = second_tick(sc, outX, 5)
    pl2, _ = call(sc, "RLc", 10, sts2, v0=sc["vp"])
    tick2 = pl2.log.arrays()
    worst, _ = cmp_rows(mid5, tick2, ("j", "k", "present", "g_src", "c_src", "v_l", "a_l", "a_m", "g0"))
    verdict("k3 label look at step 5 equals the real tick at frame j + 5 (%s)" % kind, worst <= 1e-6, "windows %d, largest difference of (frame, label read, logged speed, logged arc, progress, corrected gap) %.3e (limit 1e-6)" % (B, worst))
scr = build("brake", 24, 64, with_tokens=True)
plR, outR = call(scr, "RR", 5)
midR = mid_rows(5)
rdm = {k: np.concatenate(v) for k, v in Z.STATE["mid"]["rd"].items()}
sts2 = second_tick(scr, outR, 5)
pl2R, _ = call(scr, "RR", 10, sts2, v0=scr["vp"])
tick2R = pl2R.log.arrays()
rdt = {k: np.concatenate(v) for k, v in Z.STATE["cols"].items()}
worst, _ = cmp_rows(midR, tick2R, ("j", "k", "present", "g_src", "c_src", "v_l", "a_l", "a_m", "g0"))
wr = max(float(np.max(np.abs(np.asarray(rdm[c], np.float64) - np.asarray(rdt[c], np.float64)))) for c in ("p", "g_read", "g_used"))
okr = bool(np.array_equal(rdm["ok"], rdt["ok"]))
verdict("k3 reader look at step 5 equals the real tick at frame j + 5 (fake token reader)", worst <= 1e-6 and wr <= 1e-6 and okr,
        "windows %d, present %d, rule inputs worst %.3e, reader reads (p, gap read, gap used) worst %.3e, full-history flags equal %s" % (len(midR["q"]), int(midR["present"].sum()), worst, wr, okr))

# ---------------------------------------------------------------- the wrapper text: the parameter, its default and its refusals
src = open(WRAP, encoding="utf-8").read()
txt_ok = all(t in src for t in ('ap.add_argument("--rule-every", type=int, default=10)', 'STATE["every"] = int(mine.rule_every)', "save_mid(mine.rule_log, meta, a.replan_steps)",
                                "if int(STATE[\"every\"]) < int(self.m):", "REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc or RR",
                                "REFUSED: --rule-every below the replan interval needs --rule-log"))
verdict("the wrapper has the parameter (default 10), its refusals and the save of the later-look log", txt_ok, "text anchors present %s" % txt_ok)
print("ZZ CHECK-RULE-EVERY all hold: %s" % all(OK))
