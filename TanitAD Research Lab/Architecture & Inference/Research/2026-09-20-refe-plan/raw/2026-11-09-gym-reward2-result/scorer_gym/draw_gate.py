"""The seed-20261111 gate (PREREG_REWARD_2 section 2/4: "the seed 20261111 states are spent once read").
`require_open(seed)` is called by gym_env at the two places the seed is consumed (select_states_v4, _h_seed). For a guarded seed it passes ONLY when
 (a) the process is not in a dry/test mode (MGYM6_DRY unset),
 (b) the process is the chain's draw stage (MGYM6_DRAW_STAGE == "1", set by chain_v6 for draw_v6.py only), and
 (c) DRAW_GATE.json exists, written by chain_v6 after R1-R4 and G-REPRO passed, and binds the sha256 of the frozen prereg, CODE_PIN.json and the exclusion list.
Any other seed passes through unchanged. A test or dry run therefore cannot reach the draw."""
import hashlib, json, os

GUARDED_SEEDS = frozenset({20261111})
P6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # pilot_v6
GATE_PATH = r"F:\tanitad_scratch\mgym6\DRAW_GATE.json"
BOUND = {"prereg_sha": os.path.join(P6, "PREREG_REWARD_2.md"), "code_pin_sha": os.path.join(P6, "CODE_PIN.json"),
         "exclusion_sha": os.path.join(P6, "PREREG_REWARD_2_EXCLUDE.json")}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def require_open(seed, gate_path=None, bound=None):
    if seed not in GUARDED_SEEDS:
        return True
    assert not os.environ.get("MGYM6_DRY"), "dry/test mode (MGYM6_DRY): the seed 20261111 states are spent once read, this process may not draw"
    assert os.environ.get("MGYM6_DRAW_STAGE") == "1", "only the chain's draw stage may consume seed 20261111"
    gp = gate_path or GATE_PATH
    assert os.path.exists(gp), "no DRAW_GATE.json: R1-R4 / G-REPRO have not passed in the chain"
    gate = json.load(open(gp, encoding="utf-8"))
    assert gate.get("r_all_pass") is True and all(gate.get(k) is True for k in ("R1", "R2", "R3", "R4", "G_REPRO")), "gate does not record R1-R4 + G-REPRO PASS"
    for k, p in (bound or BOUND).items():
        assert gate.get(k) == sha_file(p), f"gate is bound to a different {k}"
    return True
