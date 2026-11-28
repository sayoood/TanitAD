"""REWARD-1 (PREREG_REWARD_1.md) pure logic: the fresh-set ordering, eligibility, the B3 verdict and the block-extension rule.
Kept free of simulation so tests can mutate each rule and require RED. Evidence tier: ENVIRONMENT validity only."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g

SEED_V5 = 20261110
FLOOR = 0.20                    # PREREG_REWARD_1 section 1
MIN_LIMIT = 4.0                 # eligibility: lane limit >= 4.0 m/s (section 3)
MIN_ELIGIBLE = 5                # B3 needs >= 5 eligible informative states
N_FULL, N_FIRST, BLOCK, CAP = 20, 160, 40, 240
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES_V5 = os.path.join(HERE, "PREREG_REWARD_1_STATES.json")
V4_STATES = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/PREREG_M_GYM_4_STATES.json"


def excluded_logs():
    """Section 0: M-GYM-2 main + stall sets (incl. stalls 7, 10, 11, 15) and the 20 M-GYM-4 states."""
    v4 = {s["log"] for s in json.load(open(V4_STATES, encoding="utf-8"))}
    return sorted(set(g.used_logs_m_gym_2()) | v4)


def select_states_v5(seed=SEED_V5, exclude_logs=None):
    """ONE fixed ordered list of CAP=240 states. idx 0..19 = exactly select_states_v4(seed, 10, 5, excl) (10 random + 10 stop-and-go/junction, 5 logs x 2);
    20..159 = ladder-only extension (70 random, then 70 stop-and-go/junction); 160..199 and 200..239 = blocks 1 and 2 (20 random then 20 stall each).
    Random and stall sub-samplers never share a log. Prefix-nesting is asserted, so the list for a smaller cap is a prefix of this one."""
    excl = set(exclude_logs) if exclude_logs is not None else set(excluded_logs())
    base = g.select_states_v4(seed=seed, n_random=10, n_stall_logs=5, exclude_logs=sorted(excl))
    r0 = g.select_states_v4(seed=seed, n_random=10, n_stall_logs=0, exclude_logs=sorted(excl))
    S = g.select_states_v4(seed=seed, n_random=0, n_stall_logs=60, exclude_logs=sorted(excl | {x["log"] for x in r0}))
    R = g.select_states_v4(seed=seed, n_random=120, n_stall_logs=0, exclude_logs=sorted(excl | {x["log"] for x in S}))
    assert len(R) == 120 and len(S) == 120, ("not enough candidate logs", len(R), len(S))
    assert R[:10] == r0 and S[:10] == base[10:] and base[:10] == r0
    assert not ({x["log"] for x in R} & {x["log"] for x in S})
    order = base + R[10:80] + S[10:80]
    for b in range(2):
        order += R[80 + 20 * b: 100 + 20 * b] + S[80 + 20 * b: 100 + 20 * b]
    assert len(order) == CAP and len({x["token"] for x in order}) == CAP
    out = []
    for i, s in enumerate(order):
        blk = 0 if i < N_FIRST else 1 + (i - N_FIRST) // BLOCK
        out.append(dict(s, idx=i, tier="full" if i < N_FULL else "ladder", block=blk))
    return out


def load_states_v5():
    raw = open(STATES_V5, "rb").read().decode()
    want = open(STATES_V5.replace(".json", ".sha256"), encoding="utf-8").read().split()[0]
    assert hashlib.sha256(raw.encode()).hexdigest() == want, "pinned REWARD-1 state list was modified"
    return json.loads(raw)


def eligible(informative, limit, min_limit=MIN_LIMIT):
    """Section 3: eligible = informative AND lane limit >= 4.0 m/s."""
    return bool(informative) and float(limit) >= min_limit


def b3_verdict(rows, at_cap, min_eligible=MIN_ELIGIBLE, elig=eligible):
    """rows: dicts with informative (bool), known (bool or None), limit (m/s), idx. Judged on ELIGIBLE states only.
    PASS iff >= min_eligible eligible and all KNOWN; FAIL iff >= min_eligible and any not KNOWN;
    UNDETERMINED iff fewer than min_eligible eligible and at_cap; CONTINUE (not a verdict) if fewer and not yet at the cap."""
    el = [r for r in rows if elig(r["informative"], r["limit"])]
    n_known = sum(bool(r["known"]) for r in el)
    if len(el) >= min_eligible:
        v = "PASS" if n_known == len(el) else "FAIL"
    else:
        v = "UNDETERMINED" if at_cap else "CONTINUE"
    return dict(verdict=v, n_eligible=len(el), n_known=n_known, eligible_idx=[r["idx"] for r in el],
                not_known_idx=[r["idx"] for r in el if not r["known"]])


def next_block(n_eligible, n_reached, min_eligible=MIN_ELIGIBLE, cap=CAP, block=BLOCK):
    """Block-extension rule: None if enough eligible states or the cap is reached, else the state index range (lo, hi) of the next block of 40."""
    if n_eligible >= min_eligible or n_reached >= cap:
        return None
    return (n_reached, min(n_reached + block, cap))
