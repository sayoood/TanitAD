"""The fresh draw of PREREG_REWARD_1 section 2 (seed 20261110). Run ONLY after G-REPRO passed. Writes PREREG_REWARD_1_STATES.json (+ .sha256 of the FILE bytes). Refuses to overwrite."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward1 as r1
assert not os.path.exists(r1.STATES_V5), "already drawn: a draw is never repeated"
sts = r1.select_states_v5(seed=r1.SEED_V5)
j = json.dumps(sts, indent=1)
open(r1.STATES_V5, "wb").write(j.encode())
sha = hashlib.sha256(open(r1.STATES_V5, "rb").read()).hexdigest()
open(r1.STATES_V5.replace(".json", ".sha256"), "w", encoding="utf-8").write(sha + "  PREREG_REWARD_1_STATES.json\n")
print(sha, len(sts), sum(s["sampler"] == "random" for s in sts[:160]), len({s["log"] for s in sts}))
