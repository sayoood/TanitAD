"""Tests for read_scorer1p6.py / draw_e2.py / build_e2_table.py (CPU only).

    CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -m pytest -q eval/test_read_scorer1p6.py --basetemp F:/TanitAD_p6eval/pt/run

Every outcome O1..O6 and Unnamed is produced by the synthetic arms built for it (no real score is read); the deliberate-regression controls must flip the verdict or turn a gate RED;
bars are tested at exact boundaries; the real-data self-check runs P6 as a fake arm against itself.
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import read_scorer1p6 as R  # noqa: E402

M = 64


# ------------------------------------------------------------------------------------------------ synthetic world
class Scn:
    """12 logs x 24 tokens. Proposal k of token i plans a constant speed r_k * v0 (so v4/v0 = r_k exactly); the human drives v0. Truth: EP rises with r up to 1.1; TTC falls past r = 1.25;
    NC zeros are frequent for r > 1.25; the decoy (k = 63, r = 1.35) has EP 1, TTC 0.8, NC = DAC = 1. P6 picks the slowest proposal. The oracle picks the best PDMS (ties: slowest)."""

    def __init__(self, seed=3, lateral_cost=False, n_logs=12, per_log=24):
        rng = np.random.default_rng(seed)
        N = n_logs * per_log
        self.N = N
        tok = [f"t{i:05d}" for i in range(N)]
        log = [f"log{i // per_log:02d}" for i in range(N)]
        cruise = (np.arange(N) % 2 == 0)
        v0 = np.where(cruise, 10.0 + 0.5 * (np.arange(N) % 5), 3.0)
        j = np.arange(1, 9)
        human = np.zeros((N, 8, 3))
        human[:, :, 0] = v0[:, None] * 0.5 * j[None]
        self.world = R.World(tok, log, human, v0, np.zeros(N), np.ones(N, int), [f"fp{i}" for i in range(N)])
        r = np.empty((N, M))
        for i in range(N):
            r[i, :61] = (np.round(np.linspace(0.5, 1.3, 61) * 256) / 256)[rng.permutation(61)]
            r[i, 61] = 0.25       # the dud: clean (NC = DAC = TTC = 1) but far too slow, PDMS ~0.68   # dyadic: float32-exact plans, so accel errors are exactly 0
            r[i, 62] = 1.125      # the TTC trap: not an overspeed (<= 1.15), EP 1, but TTC 0.8
            r[i, 63] = 1.34375    # the decoy: overspeed, EP 1, TTC 0.8
        self.r = r
        props = np.zeros((N, M, 8, 3), np.float32)
        props[..., 0] = (r * v0[:, None])[:, :, None] * 0.5 * j[None, None]
        if lateral_cost:
            props[..., 1] = (0.6 * (r > 1.05))[:, :, None] * (j[None, None] / 8.0)
        self.props = props
        nc = np.where(rng.random((N, M)) < np.where(r > 1.25, 0.5, 0.04), 0.0, 1.0)
        dac = np.where(rng.random((N, M)) < 0.04, 0.0, 1.0)
        nc[:, 61:] = dac[:, 61:] = 1.0
        ep = np.clip(r / 1.1, 0, 1)
        ttc = 1 - 0.6 * np.clip((r - 1.25) / 0.1, 0, 1)
        ttc[:, 62:] = 0.8
        sub = np.stack([nc, dac, ep, ttc, np.ones((N, M)), np.ones((N, M))], -1)
        self.sub = sub
        self.pdms = nc * dac * (5 * ep + 5 * ttc + 2.0) / 12
        self.slowest = r[:, :61].argmin(1)
        self.oracle = np.array([min(np.where(self.pdms[i] == self.pdms[i].max())[0], key=lambda k: r[i, k]) for i in range(N)])
        self.oracle_fast = np.array([max(np.where(self.pdms[i] == self.pdms[i].max())[0], key=lambda k: r[i, k]) for i in range(N)])   # same PDMS, but overspeeds (r > 1.15)
        self.decoy = np.full(N, 63)
        self.u = rng.random(N)
        self.p6_pick = np.where(rng.random(N) < 0.4, self.oracle, self.slowest)      # P6: right on 40 % of tokens, over-cautious (slowest) elsewhere; a random pick is WORSE than P6
        self.cruise = cruise
        e2 = np.arange(0, N, 3)
        self.e2_idx = e2
        self.e2_tokens = [tok[i] for i in e2]

    # ---- pick vectors
    def mix(self, frac, shift=0.0, base=None, other=None):
        base = self.p6_pick if base is None else base
        other = self.oracle if other is None else other
        use = ((self.u + shift) % 1.0) < frac
        return np.where(use, other, base)

    def e2_sabotage(self):
        """oracle everywhere except the E2 tokens, where the arm picks the dud (clean but slow, PDMS ~0.68): E3 gains, E2 regret rises, G1 and G3 untouched"""
        pick = self.oracle.copy()
        pick[self.e2_idx] = 61
        return pick

    def trap_cruise(self):
        return np.where(self.cruise, 62, self.p6_pick)

    def decoy_cruise(self):
        return np.where(self.cruise, self.decoy, self.p6_pick)

    # ---- logits: head signal -beta*(r-1) + noise, +8 on the chosen proposal (all heads) => the planner's pick is `pick` by construction
    def logits(self, pick, beta, seed):
        rng = np.random.default_rng(seed)
        sig = -beta * (self.r - 1.0)[..., None] + 0.05 * rng.standard_normal((self.N, M, 6)) + (0.3 * rng.standard_normal((self.N, M, 1)) if beta < 3 else 0.0)
        sig[np.arange(self.N), pick, :] += 8.0
        return sig.astype(np.float32)

    def arm(self, name, pick, beta, seed, scored=True):
        L = self.logits(pick, beta, seed)
        assert (R.pick_logits(L) == pick).all(), f"{name}: the construction must reproduce the intended pick"
        rows = {}
        if scored:
            for i in np.where(pick != self.p6_pick)[0]:
                rows[self.world.token[i]] = (True, float(self.pdms[i, pick[i]]), self.sub[i, pick[i]].copy())
        return R.Arm(name, self.world.token, L, self.props.copy(), rows=rows)

    def inputs(self, arms, tactical=False):
        w = self.world
        p6L = self.logits(self.p6_pick, 6.0, 100)
        rows = {w.token[i]: (True, float(self.pdms[i, self.p6_pick[i]]), self.sub[i, self.p6_pick[i]].copy()) for i in range(self.N)}
        e = self.e2_idx
        red = np.array([rows[w.token[i]][1] - 0.1 for i in e])
        tab = dict(token=np.array(self.e2_tokens), pdms=self.pdms[e], sub=self.sub[e], valid=np.ones((len(e), M), bool), red=red)
        return R.Inputs(w, p6L, self.props, rows, self.e2_tokens, tab, arms, self.p6_pick.copy(), None, None, tactical, {})

    def base_arms(self, c0=None, c0r=None, s1e=None, s1=None):
        """each given as (pick, beta, seed); defaults = P6-like arms"""
        d = {"C0": (self.p6_pick, 6.0, 11), "C0r": (self.p6_pick, 6.0, 12), "S1E": (self.p6_pick, 6.0, 13)}
        if c0:
            d["C0"] = c0
        if c0r:
            d["C0r"] = c0r
        if s1e:
            d["S1E"] = s1e
        if s1:
            d["S1"] = s1
        return {k: self.arm(k, *v) for k, v in d.items()}


@pytest.fixture(scope="module")
def S():
    return Scn()


def read(S, **kw):
    return R.run_read(S.inputs(S.base_arms(**kw)))


# ------------------------------------------------------------------------------------------------ outcomes O1..O6 and Unnamed, each from the input built for it
def test_O1_goal_met(S):
    res = read(S, s1e=(S.oracle, 0.2, 21))
    v = res["verdict"]
    assert res["status"] == "READ" and v["adopted_arm"] == "S1E" and v["outcome"]["primary"] == "O1", v["outcome"]
    assert v["goal_status_adopted"] == "GOAL MET" and v["verdict"] == "ADOPT"
    assert res["endpoints"]["S1E"]["M1_holds"] is True and v["outcome"]["matches"] == ["O1"]      # M1 holds but an arm is adopted: O6 must not match


def test_O2_gain_below_goal(S):
    res = read(S, s1e=(S.mix(0.5), 0.2, 22))
    v = res["verdict"]
    assert v["adopted_arm"] == "S1E" and v["outcome"]["primary"] == "O2" and v["goal_status_adopted"] == "BELOW GOAL", v["outcome"]


def test_O2_adopts_C0_when_S1E_does_nothing(S):
    res = read(S, c0=(S.mix(0.5), 6.0, 23), c0r=(S.mix(0.5), 6.0, 24))
    v = res["verdict"]
    assert v["adopted_arm"] == "C0" and v["outcome"]["primary"] == "O2"


def test_O3_harm(S):
    res = read(S, s1e=(S.decoy_cruise(), 0.2, 25))
    v = res["verdict"]
    assert v["adopted_arm"] is None and v["outcome"]["primary"] == "O3", v["outcome"]
    g = res["endpoints"]["S1E"]["guards"]
    assert g["all_guards_ok"] is False and not g["G1_each_head_ge_-0.5"]["TTC"] and g["G3_vs_P6_ci_hi_lt_0.10"] is False
    assert "S1" in v["outcome"]["next_levers"]["O3"]


def test_G3_alone_blocks_adoption(S):
    res = read(S, s1e=(S.oracle_fast, 0.2, 26))
    e = res["endpoints"]["S1E"]
    g = e["guards"]
    assert e["P1_clears"] is True and all(g["G1_each_head_ge_-0.5"].values()) and g["G2_ci_hi_lt_0.5"] is True      # PDMS gain, G1 and G2 fine ...
    assert g["G3_vs_P6_ci_hi_lt_0.10"] is False and e["G3_overspeed_diff_vs_P6"]["lo"] > 0.5                      # ... only the overspeed guard fails
    v = res["verdict"]
    assert v["adopted_arm"] is None and "O3" in v["outcome"]["matches"] and v["outcome"]["primary"] == "O3"
    ok = read(S, s1e=(S.oracle, 0.2, 26))                                                                         # same gain with the slower tie: adopted
    assert ok["verdict"]["adopted_arm"] == "S1E"


def test_G1_alone_blocks_adoption(S):
    res = read(S, s1e=(S.trap_cruise(), 0.2, 27))
    e = res["endpoints"]["S1E"]
    g = e["guards"]
    assert e["P1_clears"] is True and g["G3_vs_P6_ci_hi_lt_0.10"] is True and g["G2_ci_hi_lt_0.5"] is True
    assert g["G1_each_head_ge_-0.5"] == {"TTC": False, "NC": True, "DAC": True} or not g["G1_each_head_ge_-0.5"]["TTC"]
    assert res["verdict"]["adopted_arm"] is None and res["verdict"]["outcome"]["primary"] == "O3"


def test_G2_alone_blocks_adoption(S):
    res = read(S, s1e=(S.e2_sabotage(), 0.2, 30))
    e = res["endpoints"]["S1E"]
    g = e["guards"]
    assert e["P1_clears"] is True and all(g["G1_each_head_ge_-0.5"].values()) and g["G3_vs_P6_ci_hi_lt_0.10"] is True
    assert g["G2_ci_hi_lt_0.5"] is False and e["G2_delta_regret_vs_P6"]["lo"] > 0
    assert res["verdict"]["adopted_arm"] is None


def test_G3_alone_blocks_C0_too(S):
    res = read(S, c0=(S.oracle_fast, 6.0, 28), c0r=(S.oracle_fast, 6.0, 29))
    c0 = res["endpoints"]["C0"]
    assert c0["P1_vs_P6"]["clears"] is True and c0["guards"]["G3_vs_P6_ci_hi_lt_0.10"] is False and res["verdict"]["adopted_arm"] is None


def test_O4_null(S):
    res = read(S)
    v = res["verdict"]
    assert v["adopted_arm"] is None and v["outcome"]["primary"] == "O4" and v["verdict"] == "NO ADOPTION", v["outcome"]


def test_O5_swamped(S):
    res = read(S, c0=(S.mix(0.10), 6.0, 31), c0r=(S.mix(0.30, shift=0.5), 6.0, 32), s1e=(S.mix(0.10), 6.0, 33))
    e = res["endpoints"]["C0"]["E3_pdms_vs_P6"]
    assert e["lo"] > 0, "built to have CI-lo > 0"
    F = res["verdict"]["F_pdms_E3"]
    assert not abs(e["point"]) > 2 * F, "built to sit inside 2F"
    v = res["verdict"]
    assert v["adopted_arm"] is None and v["outcome"]["primary"] == "O5", v["outcome"]


def test_O6_mechanism_only(S):
    res = read(S, s1e=(S.p6_pick, 0.2, 41))
    v = res["verdict"]
    assert res["endpoints"]["S1E"]["M1_holds"] is True
    assert v["adopted_arm"] is None and v["outcome"]["primary"] == "O6", v["outcome"]


def test_Unnamed(S):
    res = read(S, c0=(S.decoy_cruise(), 6.0, 51), c0r=(S.decoy_cruise(), 6.0, 52))
    v = res["verdict"]
    assert res["endpoints"]["C0"]["P1_vs_P6"]["clears"] is True and res["endpoints"]["C0"]["guards"]["all_guards_ok"] is False
    assert v["adopted_arm"] is None and v["outcome"]["primary"] == "Unnamed" and v["outcome"]["matches"] == ["Unnamed"], v["outcome"]


def test_every_outcome_label_is_reachable_and_distinct(S):
    got = {read(S, s1e=(S.oracle, 0.2, 1))["verdict"]["outcome"]["primary"], read(S, s1e=(S.mix(0.5), 0.2, 2))["verdict"]["outcome"]["primary"],
           read(S, s1e=(S.decoy_cruise(), 0.2, 3))["verdict"]["outcome"]["primary"], read(S)["verdict"]["outcome"]["primary"],
           read(S, c0=(S.mix(0.10), 6.0, 4), c0r=(S.mix(0.30, shift=0.5), 6.0, 5), s1e=(S.mix(0.10), 6.0, 6))["verdict"]["outcome"]["primary"],
           read(S, s1e=(S.p6_pick, 0.2, 7))["verdict"]["outcome"]["primary"],
           read(S, c0=(S.decoy_cruise(), 6.0, 8), c0r=(S.decoy_cruise(), 6.0, 9))["verdict"]["outcome"]["primary"]}
    assert got == {"O1", "O2", "O3", "O4", "O5", "O6", "Unnamed"}


# ------------------------------------------------------------------------------------------------ family clause
def test_family_clause_adopt_vs_gain_with_cost():
    clean = Scn(lateral_cost=False)
    res = read(clean, s1e=(clean.oracle, 0.2, 61))
    assert res["verdict"]["verdict"] == "ADOPT" and res["verdict"]["adverse_components"] == {}
    cost = Scn(lateral_cost=True)
    res = read(cost, s1e=(cost.oracle, 0.2, 62))
    v = res["verdict"]
    assert v["adopted_arm"] == "S1E" and v["verdict"] == "PDMS GAIN WITH COST", v
    assert "cross_mae_m" in v["adverse_components"] and v["adverse_components"]["cross_mae_m"]["lo"] > 0
    assert "never described as the better driver" in v["cost_note"]


def test_strategic_is_paused_never_a_pass(S):
    res = read(S)
    assert res["strategic"].startswith("PAUSED") and "ITEM 25" in res["strategic"] and "pass" in res["strategic"].lower()
    md = R.to_markdown(res)
    assert "STRATEGIC: PAUSED" in md and "T1-class open loop" in md and "MEASURED" in md and "episode" not in md.split("Estimator")[0]


# ------------------------------------------------------------------------------------------------ deliberate-regression controls
def test_shuffled_logits_arm_is_not_adopted(S):
    rng = np.random.default_rng(5)
    good = S.arm("S1E", S.oracle, 0.2, 71)
    sh = np.stack([good.e3_logits[i][rng.permutation(M)] for i in range(S.N)])
    pick = R.pick_logits(sh)
    rows = {S.world.token[i]: (True, float(S.pdms[i, pick[i]]), S.sub[i, pick[i]].copy()) for i in np.where(pick != S.p6_pick)[0]}
    bad = R.Arm("S1E", S.world.token, sh, S.props.copy(), rows=rows)
    arms = S.base_arms()
    arms["S1E"] = bad
    res = R.run_read(S.inputs(arms))
    assert res["status"] == "READ" and res["verdict"]["adopted_arm"] != "S1E" and res["verdict"]["adopted_arm"] is None
    assert res["endpoints"]["S1E"]["E3_pdms_vs_P6"]["point"] < 0
    arms["S1E"] = good
    assert R.run_read(S.inputs(arms))["verdict"]["adopted_arm"] == "S1E"          # the un-shuffled arm IS adopted: the control moves the verdict


def test_shifted_proposal_fails_G0_E3(S):
    arms = S.base_arms(s1e=(S.oracle, 0.2, 81))
    P = arms["S1E"].e3_proposals.copy()
    P[17, 5, 3, 0] += np.float32(1e-3)
    arms["S1E"].e3_proposals = P
    res = R.run_read(S.inputs(arms))
    assert res["status"] == "READ STOPPED" and res["gates"]["G0-E3[S1E]"]["PASS"] is False and res["gates"]["G0-E3[S1E]"]["n_tokens_not_bit_identical"] == 1
    assert "verdict" not in res and "no verdict" in res["why"]
    assert res["gates"]["G0-E3[C0]"]["PASS"] is True
    md = R.to_markdown(res)
    assert "No verdict is written" in md


def test_swapped_labels_change_the_sign(S):
    B = R.BootSet(S.world.log)
    a, b = np.random.default_rng(1).random(S.N), np.random.default_rng(2).random(S.N)
    p1, r1, _ = B.diff(a, b)
    p2, r2, _ = B.diff(b, a)
    x, y = R.rec(p1, r1), R.rec(p2, r2)
    assert x["point"] == pytest.approx(-y["point"], abs=1e-12) and x["lo"] == pytest.approx(-y["hi"], abs=1e-12) and x["hi"] == pytest.approx(-y["lo"], abs=1e-12)
    assert x["point"] != 0
    res = read(S, s1e=(S.oracle, 0.2, 91))
    sw = R.run_read(S.inputs(S.base_arms(s1e=(S.oracle, 0.2, 91))), swap_labels=True)
    m, ms = res["endpoints"]["S1E"]["M1_dRho_level"]["point"], sw["endpoints"]["S1E"]["M1_dRho_level"]["point"]
    assert m == pytest.approx(-ms, abs=1e-12) and m != 0
    assert res["endpoints"]["S1E"]["M1_vs_C0"]["point"] == pytest.approx(-sw["endpoints"]["S1E"]["M1_vs_C0"]["point"], abs=1e-12)
    assert res["endpoints"]["S1E"]["M1_holds"] != sw["endpoints"]["S1E"]["M1_holds"] or res["endpoints"]["S1E"]["M1_vs_C0"]["point"] > 0


def test_E2_gate_red_when_table_disagrees_with_the_csv(S):
    inp = S.inputs(S.base_arms())
    inp.e2_table["pdms"] = inp.e2_table["pdms"].copy()
    inp.e2_table["pdms"][4, S.p6_pick[S.e2_idx[4]]] += 1e-12
    res = R.run_read(inp)
    assert res["status"] == "READ STOPPED" and res["gates"]["G0-E2"]["PASS"] is False and res["gates"]["G0-E2"]["n_pdms_bit_equal"] == len(S.e2_idx) - 1


def test_E2_gate_red_control_needs_a_shifted_column_that_differs(S):
    inp = S.inputs(S.base_arms())
    csv_pd = np.array([inp.p6_rows[t][1] for t in inp.e2_tokens])
    inp.e2_table["red"] = csv_pd.copy()                    # a "shifted" column identical to the csv = the control cannot see: RED
    res = R.run_read(inp)
    assert res["gates"]["G0-E2"]["PASS"] is False and res["gates"]["G0-E2"]["RED_shifted_pick_n_differ"] == 0


def test_missing_scores_stop_as_incomplete_and_emit_changed_names_them(S, tmp_path):
    arms = S.base_arms(c0=(S.mix(0.3), 6.0, 5))
    arms["C0"].rows = {}
    inp = S.inputs(arms)
    res = R.run_read(inp)
    assert res["status"] == "INCOMPLETE" and "C0" in res["why"]
    out = R.emit_changed(inp, ["C0"], str(tmp_path))
    n_changed = int((R.pick_logits(arms["C0"].e3_logits) != S.p6_pick).sum())
    z = np.load(out["C0"]["seam"])
    assert out["C0"]["n_needed"] <= n_changed and z["poses"].shape == (out["C0"]["n_needed"], 8, 3) and z["poses"].dtype == np.float32
    # tokens that fall in the E2 table are already scored there: not asked for again
    assert out["C0"]["n_needed"] == n_changed - sum(1 for i in S.e2_idx if R.pick_logits(arms["C0"].e3_logits[i:i + 1])[0] != S.p6_pick[i])


def test_score_pool_conflict_stops_the_read(S):
    arms = S.base_arms(c0=(S.mix(0.5), 6.0, 5))
    t = next(t for t in arms["C0"].rows if S.world.index[t] in set(S.e2_idx))
    v, s, sb = arms["C0"].rows[t]
    arms["C0"].rows[t] = (v, s + 0.01, sb)
    res = R.run_read(S.inputs(arms))
    assert res["status"] == "READ STOPPED" and "conflicting" in res["why"]


def test_ENS3_is_read_after_and_cannot_change_the_verdict(S):
    arms = S.base_arms(s1e=(S.oracle, 0.2, 101))
    res = R.run_read(S.inputs(arms))
    ens = res["ENS3"]
    assert ens["read_after_verdict"] is True and "no replicate" in ens["replicate_floor"]
    # the members' logits are averaged: with C0 = C0r = P6-like and S1E = oracle the ensemble picks are a defined function of them
    assert ens["status"] in ("READ", ) or ens["status"].startswith("NOT READ: harness scores missing")
    res2 = R.run_read(S.inputs(arms), need_ens3=False)
    assert res2["verdict"] == {**res["verdict"]} or json.dumps(res2["verdict"], default=R.jdefault) == json.dumps(res["verdict"], default=R.jdefault)


# ------------------------------------------------------------------------------------------------ bars at exact boundaries (pure functions)
def test_boundaries_P1():
    assert R.p1_clears(1e-9, 1.0, 0.4) is True                       # lo > 0, |d| = 1.0 > 0.8
    assert R.p1_clears(0.0, 1.0, 0.4) is False                       # lo == 0 is not > 0
    assert R.p1_clears(0.1, 0.8, 0.4) is False                       # |d| == 2F is not > 2F
    assert R.p1_clears(0.1, 0.8000001, 0.4) is True
    assert R.p1_clears(0.1, -2.0, 0.4) is True                       # the 2F rule is on |d| (the CI-lo > 0 rule carries the sign)
    assert R.p1_clears(0.1, 1.0, None) is False                      # no replicate floor: never cleared


def test_boundaries_M1():
    assert R.m1_clears(0.5, 0.01, -1.0, 0.1) is True                 # d == 0.5 * |dRho_C0|
    assert R.m1_clears(0.4999999, 0.01, -1.0, 0.1) is False
    assert R.m1_clears(0.5, 0.0, -1.0, 0.1) is False                 # lo == 0
    assert R.m1_clears(0.5, 0.01, -1.0, 0.25) is False               # |d| == 2F
    assert R.m1_clears(0.5, 0.01, -1.0, None) is False


def test_boundaries_guards_and_goal():
    assert R.g1_ok(-0.5) is True and R.g1_ok(-0.5000001) is False and R.g1_ok(-0.49) is True
    assert R.g2_ok(0.4999999) is True and R.g2_ok(0.5) is False
    assert R.g3_ok(0.0999999) is True and R.g3_ok(0.10) is False
    assert R.goal_status(94.55, 94.9) == "GOAL MET" and R.goal_status(94.5499, 94.55) == "GOAL NOT EXCLUDED"
    assert R.goal_status(94.5499, 94.5499) == "BELOW GOAL" and R.goal_status(94.549, 94.7) == "GOAL NOT EXCLUDED"
    assert R.adverse("lower", 1e-9, 2.0) is True and R.adverse("lower", 0.0, 2.0) is False
    assert R.adverse("higher", -2.0, -1e-9) is True and R.adverse("higher", -2.0, 0.0) is False


def test_decide_outcome_table():
    base = dict(adopted=None, adopted_point=None, s1e_p1=False, s1e_m1=False, s1e_guards_ok=True, c0_vs_p6=False, s1e_vs_p6=False, swamped=False, m1_holds=False, guard_caused=False)
    assert R.decide_outcome({**base, "adopted": "S1E", "adopted_point": 94.55})["primary"] == "O1"
    assert R.decide_outcome({**base, "adopted": "C0", "adopted_point": 94.5499})["primary"] == "O2"
    assert R.decide_outcome({**base, "s1e_m1": True, "s1e_guards_ok": False, "m1_holds": True})["primary"] == "O3"
    assert R.decide_outcome(base)["primary"] == "O4"
    assert R.decide_outcome({**base, "swamped": True})["primary"] == "O5"
    assert R.decide_outcome({**base, "m1_holds": True, "s1e_m1": True})["primary"] == "O6"
    assert R.decide_outcome({**base, "c0_vs_p6": True})["primary"] == "Unnamed"
    both = R.decide_outcome({**base, "swamped": True})
    assert both["matches"] == ["O5", "O4"] or set(both["matches"]) == {"O4", "O5"}      # overlap listed, primary by the fixed order


# ------------------------------------------------------------------------------------------------ estimator and ranks
def test_ties_use_average_ranks_not_argsort_of_argsort(S, monkeypatch):
    seen = []
    orig = R.SA.per_token_spearman
    monkeypatch.setattr(R.SA, "per_token_spearman", lambda X, Y, Mk, method="tie": (seen.append(method), orig(X, Y, Mk, method))[1])
    read(S)
    assert seen and set(seen) == {"tie"}
    x = np.array([[1.0, 1.0, 1.0, 2.0, 3.0, 3.0]])
    y = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    m = np.ones_like(x, bool)
    assert orig(x, y, m, "tie")[0] != pytest.approx(orig(x, y, m, "legacy")[0])


def test_bootstrap_is_the_programme_estimator(S):
    B = R.BootSet(S.world.log)
    ul, li, draws = R.sc.log_boot_idx(S.world.log, 10000, 20260927)
    v = np.random.default_rng(0).random(S.N)
    pt, lo, hi = R.sc.boot_mean(v, np.ones(S.N, bool), li, len(ul), draws)
    p2, r2, _ = B.stat(v)
    q = R.rec(p2, r2)
    assert (pt, lo, hi) == (q["point"], q["lo"], q["hi"])


# ------------------------------------------------------------------------------------------------ real data: P6 as a fake arm against itself
def test_real_data_P6_against_itself_is_exactly_zero():
    if not (os.path.exists("F:/TanitAD_p6read_work/run/dump/dump_p6.npz") and os.path.exists(R.DEFAULT_P6_CSV)):
        pytest.skip("real inputs absent")
    res = R.run_read(R.inputs_selfcheck())
    assert res["status"] == "READ" and all(g["PASS"] for g in res["gates"].values())
    zd = R.zero_difference_report(res)
    assert zd["n_difference_records"] > 60 and zd["n_nonzero_or_missing"] == 0, zd
    v = res["verdict"]
    assert v["adopted_arm"] is None and v["verdict"] == "NO ADOPTION" and v["F_pdms_E3"] == 0.0 and v["outcome"]["primary"] == "O4"
    for nm in ("C0", "C0r", "S1E"):
        e = res["endpoints"][nm]
        assert e["E3_pdms_vs_P6"] == {"point": 0.0, "lo": 0.0, "hi": 0.0, "n": e["E3_pdms_vs_P6"]["n"]} and e["E3_pick_changed_vs_P6"] == 0
        assert not (e["P1_vs_P6"].get("clears") or e.get("P1_clears"))
    assert res["ENS3"]["status"] == "READ" and res["ENS3"]["BAR_CLEARED"] is False and res["ENS3"]["diff_vs_base"]["point"] == 0.0
    # the same run with one arm perturbed is NOT zero: the zero is not an artifact of the check
    inp = R.inputs_selfcheck()
    L = inp.arms["S1E"].e3_logits.copy()
    L[:, :, 3] += np.linspace(0, 4, M)[None, :].astype(np.float32)
    inp.arms["S1E"].e3_logits = L
    pick = R.pick_logits(L)
    p6pick = R.pick_logits(inp.p6_logits)
    assert (pick != p6pick).any()
    tab_pos = {t: i for i, t in enumerate(inp.e2_table["token"])}
    inp.arms["S1E"].rows = {t: (True, float(inp.e2_table["pdms"][tab_pos[t], pick[i]]), inp.e2_table["sub"][tab_pos[t], pick[i]]) for i, t in enumerate(inp.world.token) if pick[i] != p6pick[i]}
    res2 = R.run_read(inp)
    assert res2["status"] == "READ" and R.zero_difference_report(res2)["n_nonzero_or_missing"] > 0


# ------------------------------------------------------------------------------------------------ draw_e2 and build_e2_table
def test_draw_e2_is_deterministic_disjoint_and_the_banked_list_matches():
    import draw_e2 as D
    allt = [f"{i:016x}" for i in range(300)]
    c3 = allt[:40]
    a, pool = D.draw(allt, c3, 50, 20261108)
    b, _ = D.draw(list(reversed(allt)), c3, 50, 20261108)
    assert a == b == sorted(a) and len(set(a)) == 50 and not set(a) & set(c3) and pool == 260
    assert D.draw(allt, c3, 50, 7)[0] != a
    p = HERE.parent / "e2" / "e2_tokens.json"
    if p.exists():
        d = json.load(open(p))
        c3r = json.load(open(D.DEFAULT_C3))["tokens"]
        assert len(d["tokens"]) == 2000 and len(set(d["tokens"])) == 2000 and not set(d["tokens"]) & set(c3r) and d["tokens"] == sorted(d["tokens"])
        sha = open(HERE.parent / "e2" / "e2_tokens.sha256").read().split()[0]
        assert sha == D.sha256_file(p)


def test_draw_e2_never_opens_a_score():
    src = (HERE / "draw_e2.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"""')))
    for forbidden in ("table_p6", "refe_p6final", ".csv", "pdms", "proposals_p6final", "np.load"):
        assert forbidden not in code.split('"""')[-1], forbidden


def _fake_harness(tmp):
    """a fake NAVSIM harness: the score of a pose set is a deterministic function of the plan (no real score is used)."""
    import selc_score as S

    def score_of(poses):
        x = poses[:, :, 0].mean(1)
        y = poses[:, :, 1].mean(1)
        ep = np.clip(x / 40.0, 0, 1) * np.where(np.abs(y) < 1.5, 1.0, 0.5)
        one = np.ones_like(ep)
        sub = np.stack([one, one, ep, one, one, one], 1)
        s = sub[:, 0] * sub[:, 1] * (5 * sub[:, 2] + 5 * sub[:, 3] + 2 * sub[:, 4]) / 12
        return s, sub

    def fake_score_one(job, logdir, retries_p):
        label, seam, tj, n = job
        z = np.load(seam)
        s, sub = score_of(z["poses"].astype(np.float64))
        p = S.csv_of(label)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("," + "token,valid," + ",".join(S.SUB) + ",score\n")
            for i, t in enumerate(z["token"]):
                f.write(f"{i},{t},True," + ",".join(repr(float(v)) for v in sub[i]) + f",{float(s[i])!r}\n")
        os.makedirs(logdir, exist_ok=True)
        return label, {"status": "PASS", "csv_valid_rows": n}

    return fake_score_one, score_of


def test_build_e2_table_with_a_fake_harness_passes_G0_and_fails_when_the_csv_disagrees(tmp_path):
    import build_e2_table as BT
    rng = np.random.default_rng(0)
    n = 20
    tok = [f"e{i:03d}" for i in range(n)]
    logs = {t: f"L{i // 10}" for i, t in enumerate(tok)}
    props = np.zeros((n, M, 8, 3), np.float32)
    props[..., 0] = (rng.random((n, M)) * 20 + 5)[:, :, None] * 0.5 * np.arange(1, 9)[None, None]
    props[..., 1] = (rng.random((n, M)) * 3 - 1.5)[:, :, None]
    logits = rng.standard_normal((n, M, 6)).astype(np.float32)
    pick = R.pick_logits(logits)
    npz = tmp_path / "props.npz"
    np.savez(npz, token=np.array(tok), fingerprint=np.array([f"fp{i}" for i in range(n)]), proposals=props, logits=logits, pick=pick)
    tj = tmp_path / "tok.json"
    json.dump({"tokens": tok, "token_log": logs}, open(tj, "w"))
    fake, score_of = _fake_harness(tmp_path)
    s, sub = score_of(props[np.arange(n), pick].astype(np.float64))
    csvp = tmp_path / "p6.csv"

    def write_csv(s_, sub_):
        with open(csvp, "w", encoding="utf-8") as f:
            f.write("," + "token,valid," + ",".join(BT.S.SUB) + ",score\n")
            for i, t in enumerate(tok):
                f.write(f"{i},{t},True," + ",".join(repr(float(v)) for v in sub_[i]) + f",{float(s_[i])!r}\n")
    write_csv(s, sub)

    class A:
        proposals, tokens, csv, out, work, workers, columns = str(npz), str(tj), str(csvp), str(tmp_path / "out"), str(tmp_path / "work"), 1, 0

    BT.S.MIN_RAM_GIB = 0.1
    assert BT.build(A, score_one=fake) == 0
    z = np.load(Path(A.out) / "e2_table_p6final.npz")
    assert z["pdms"].shape == (n, M) and bool(z["valid"].all()) and json.load(open(Path(A.out) / "e2_g0.json"))["PASS"] is True
    assert z["pdms"][np.arange(n), pick].tolist() == s.tolist()
    # RED: a csv that disagrees with the harness on one token => G0-E2 FAIL (exit 4), the table is not trusted
    s2 = s.copy()
    s2[3] += 1e-9
    write_csv(s2, sub)
    A.out, A.work = str(tmp_path / "out2"), str(tmp_path / "work2")
    assert BT.build(A, score_one=fake) == 4
    assert json.load(open(Path(A.out) / "e2_g0.json"))["PASS"] is False
    # the proposals file is absent (P6's second pass not finished): exit 3, nothing written but the log
    A.proposals, A.out = str(tmp_path / "absent.npz"), str(tmp_path / "out3")
    assert BT.build(A, score_one=fake) == 3


def test_cli_file_layout_end_to_end_on_real_C3_standin(tmp_path):
    """the file layout the arms must deliver, through main(): read (P6 clones), then a changed arm with no scores -> INCOMPLETE -> emit-changed writes the harness seam"""
    dump, table = "F:/TanitAD_p6read_work/run/dump/dump_p6.npz", "F:/TanitAD_p6read_work/run/score/table_p6.npz"
    if not (os.path.exists(dump) and os.path.exists(table) and os.path.exists(R.DEFAULT_P6_CSV)):
        pytest.skip("real inputs absent")
    Z = np.load(dump, allow_pickle=False)
    for nm in ("C0", "C0r", "S1E"):
        d = tmp_path / nm
        d.mkdir()
        np.savez(d / "e3_logits.npz", token=Z["token"], logits=Z["logits"], proposals=Z["props8"])
    tok = [str(t) for t in Z["token"]]
    ej = tmp_path / "e2_tokens.json"
    json.dump({"tokens": tok, "token_log": {t: str(l) for t, l in zip(tok, Z["log"])}}, open(ej, "w"))
    base = ["--p6-proposals", dump, "--world-dump", dump, "--e2-tokens", str(ej), "--e2-table", table, "--no-tactical"]
    arms = [f"--arm={n}={tmp_path / n}" for n in ("C0", "C0r", "S1E")]
    rc = R.main(["read", *base, *arms, "--out", str(tmp_path / "out")])
    assert rc == 0
    res = json.load(open(tmp_path / "out" / "scorer1p6_read.json"))
    assert res["status"] == "READ" and res["verdict"]["outcome"]["primary"] == "O4" and (tmp_path / "out" / "scorer1p6_read.md").exists()
    assert res["inputs"]["p6_csv"][1] == R.sha256_file(R.DEFAULT_P6_CSV)
    # change one arm's scorer so that picks move on tokens that the table does not cover: no scores -> INCOMPLETE
    L = np.array(Z["logits"]).copy()
    L[:, :, 3] += np.linspace(0, 4, M)[None, :].astype(np.float32)
    np.savez(tmp_path / "S1E" / "e3_logits.npz", token=Z["token"], logits=L, proposals=Z["props8"])
    arms2 = [f"--arm={n}={tmp_path / n}" for n in ("C0", "C0r", "S1E")]
    # E2 table covers all tokens here (stand-in), so shrink the E2 set to force missing E3 scores
    json.dump({"tokens": tok[:200], "token_log": {t: str(l) for t, l in zip(tok[:200], Z["log"][:200])}}, open(ej, "w"))
    rc = R.main(["read", *base, *arms2, "--out", str(tmp_path / "out2")])
    res2 = json.load(open(tmp_path / "out2" / "scorer1p6_read.json"))
    assert rc == 2 and res2["status"] == "INCOMPLETE" and "S1E" in res2["why"] and "verdict" not in res2
    rc = R.main(["emit-changed", *base[:2], *base[2:], *arms2, "--out", str(tmp_path / "emit")])
    assert rc == 0
    need = json.load(open(tmp_path / "emit" / "needs_scoring.json"))["S1E"]
    z = np.load(need["seam"])
    assert need["n_needed"] > 0 and z["poses"].shape == (need["n_needed"], 8, 3) and set(json.load(open(need["tokens"]))["tokens"]) == {str(t) for t in z["token"]}
