"""REWARD-2 (PREREG_REWARD_2.md) pure logic on top of reward1.py: seed, the frozen exclusion list, the guarded draw, the free-road class (B3-F), the kinematic crawl share.
Kept free of simulation so tests can mutate each rule and require RED. Evidence tier: ENVIRONMENT validity only."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g
import reward1 as r1

SEED_V6 = 20261111                 # PREREG_REWARD_2 section 2, fixed before drawing; no seed search
CURVE_MIN_SEG_M = 0.05             # PREREG_REWARD_2 section 1, the single change
FLOOR = r1.FLOOR                   # 0.20, unchanged
HERE = os.path.dirname(os.path.abspath(__file__))
P6 = os.path.dirname(HERE)         # pilot_v6
P5 = os.path.join(os.path.dirname(P6), "pilot_v5")
STATES_V6 = os.path.join(P6, "PREREG_REWARD_2_STATES.json")
EXCL_V6 = os.path.join(P6, "PREREG_REWARD_2_EXCLUDE.json")
MIN_FREE_V0 = 1.0                  # B3-F: launch speed >= 1.0 m/s
CRAWL_RATIO = 0.10                 # B3's KNOWN crawl bar


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ section 0: derivation-only states (never validation)
def excluded_logs_v6():
    """Section 0: the logs of ALL states seen so far = M-GYM-2 main + stall sets (incl. stalls 7, 10, 11, 15), the 20 M-GYM-4 states (both via r1.excluded_logs())
    and every one of the 240 states of the seed-20261110 list (PREREG_REWARD_1_STATES.json, simulated or not)."""
    v5 = r1.load_states_v5()
    assert len(v5) == 240
    return sorted(set(r1.excluded_logs()) | {s["log"] for s in v5})


def exclusion_record():
    ex = excluded_logs_v6()
    v5 = {s["log"] for s in r1.load_states_v5()}
    return dict(prereg="PREREG_REWARD_2.md section 0", seed_to_draw=SEED_V6, n_logs=len(ex), n_reward1_240_logs=len(v5),
                n_m_gym_2_logs=len(g.used_logs_m_gym_2()), n_m_gym_4_logs=len({s["log"] for s in json.load(open(r1.V4_STATES, encoding="utf-8"))}), logs=ex)


def freeze_exclusion(path=EXCL_V6):
    """Writes the exclusion list + its own .sha256 BEFORE any rollout and before the draw; an existing file is only verified, never rewritten."""
    shaf = path.replace(".json", ".sha256")
    if os.path.exists(path):
        load_exclusion(path)
        return "REUSED"
    rec = exclusion_record()
    b = json.dumps(rec, indent=1).encode()
    open(path, "wb").write(b)
    open(shaf, "w", encoding="utf-8").write(sha_bytes(b) + "  " + os.path.basename(path) + "\n")
    return "FROZEN"


def load_exclusion(path=EXCL_V6):
    raw = open(path, "rb").read()
    want = open(path.replace(".json", ".sha256"), encoding="utf-8").read().split()[0]
    assert sha_bytes(raw) == want, "frozen exclusion list was modified"
    return json.loads(raw)["logs"]


# ------------------------------------------------------------------ section 2: the draw (guarded by draw_gate through gym_env)
def select_states_v6(seed=SEED_V6, exclude_logs=None):
    """Same rule as PREREG_REWARD_1 section 2 (select_states_v5 = one fixed ordered list of 240, idx 0-19 full-arm, 20-159 ladder-only, blocks 160-199, 200-239).
    With seed == SEED_V6 the gate in gym_env.select_states_v4 refuses unless called from the chain's draw stage after R1-R4 + G-REPRO."""
    ex = list(exclude_logs) if exclude_logs is not None else load_exclusion()
    return r1.select_states_v5(seed=seed, exclude_logs=ex)


def load_states_v6():
    raw = open(STATES_V6, "rb").read()
    want = open(STATES_V6.replace(".json", ".sha256"), encoding="utf-8").read().split()[0]
    assert sha_bytes(raw) == want, "pinned REWARD-2 state list was modified"
    return json.loads(raw.decode())


# ------------------------------------------------------------------ section 3 secondary: free road (B3-F) and the kinematic crawl share
def classify_free(rec):
    """B3-F class of one state record, from state metadata only (never a return): free road iff launch speed >= 1.0 m/s AND no lead or conflict agent in any
    cycle of any of the six constant-speed rungs (the 'any rung' definition of REWARD_2_COUNTERFACTUALS Q2: the log-replayed agents seen along any rung's path)."""
    v0 = float(rec["v0"])
    agent = any(c["lead_gap"] is not None or c["conf_ttc"] is not None for v in rec["ladder"].values() for c in v["logs"])
    return dict(free=bool(v0 >= MIN_FREE_V0 and not agent), v0=v0, agent_in_window=bool(agent))


DT, CYC = 0.1, 5
CT = DT * CYC


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def kin_progress(v0, vt, limit):
    """Sum over 17 cycles of the normalised progress of a rung that follows the ladder rule alone (gym_env const_traj: accel <= 2, decel <= 3 m/s^2, 0.1 s steps),
    against the brake baseline (REWARD_2_COUNTERFACTUALS.py Q4, q = 1, no reward effects)."""
    vs = v0; tot = 0.0; norm = limit * CT; v00 = max(v0, 0.0)
    for cyc in range(17):
        ds = 0.0
        for _ in range(CYC):
            vs = vs + _clip(vt - vs, -3.0 * DT, 2.0 * DT); ds += vs * DT

        def s(t):
            tt = min(t, v00 / 2.0)
            return v00 * tt - 0.5 * 2.0 * tt * tt
        tot += _clip((ds - (s(cyc * CT + CT) - s(cyc * CT))) / norm, 0.0, 1.5)
    return tot


def kin_crawl_over_best(v0, limit):
    """crawl/best that the rung definitions and the window give from kinematics alone (q = 1)."""
    pc = kin_progress(v0, 1.5, limit)
    pb = max(kin_progress(v0, 0.9 * limit, limit), kin_progress(v0, limit, limit))
    return pc / pb


def kin_share(rows, bar=CRAWL_RATIO):
    """rows: eligible rows with 'kin_crawl_over_best'. Share of eligible states whose kinematic crawl/best already exceeds the 0.10 bar (the bar cannot be met by progress alone)."""
    kin = [r["kin_crawl_over_best"] for r in rows]
    return dict(n=len(kin), n_kin_above_bar=int(sum(k > bar for k in kin)), share_above_bar=(sum(k > bar for k in kin) / len(kin)) if kin else None,
                mean=(sum(kin) / len(kin)) if kin else None, min=min(kin) if kin else None, max=max(kin) if kin else None)


def b3f_verdict(rows, free_of=lambda r: r["free"]):
    """B3-F = B3's KNOWN rule on the F states among the ELIGIBLE rows. PASS iff >= 5 F states and all KNOWN; FAIL iff >= 5 and any not KNOWN; fewer than 5 -> UNDETERMINED_LT5."""
    f_rows = [r for r in rows if r1.eligible(r["informative"], r["limit"]) and free_of(r)]
    v = r1.b3_verdict(f_rows, at_cap=True)
    if v["verdict"] == "UNDETERMINED":
        v["verdict"] = "UNDETERMINED_LT5"
    v["n_free_eligible"] = len(f_rows)
    return v
