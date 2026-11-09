"""The fresh draw of PREREG_REWARD_2 section 2 (seed 20261111). Run ONLY by chain_v6's draw stage, after R1-R4 + G-REPRO passed (the chain writes DRAW_GATE.json and sets MGYM6_DRAW_STAGE=1 for this process).
  draw_v6.py freeze   : writes PREREG_REWARD_2_EXCLUDE.json + .sha256 (section 0 exclusion list). Does not touch the seed.
  draw_v6.py draw     : writes PREREG_REWARD_2_STATES.json (+ .sha256 of the FILE bytes). Refuses to overwrite and refuses in dry/test mode.
The seed itself is consumed only inside gym_env.select_states_v4 / _h_seed, both behind draw_gate.require_open."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward2 as r2


def main(argv, select=r2.select_states_v6):
    mode = argv[0]
    if mode == "freeze":
        print(r2.freeze_exclusion(), r2.EXCL_V6)
        print(len(r2.load_exclusion()))
        return 0
    assert mode == "draw"
    assert not os.environ.get("MGYM6_DRY"), "draw_v6 refuses in dry/test mode"
    assert not os.path.exists(r2.STATES_V6), "already drawn: a draw is never repeated"
    ex = r2.load_exclusion()
    sts = select(seed=r2.SEED_V6, exclude_logs=ex)
    assert len(sts) == 240 and not ({s["log"] for s in sts} & set(ex)), "exclusion violated"
    j = json.dumps(sts, indent=1)
    open(r2.STATES_V6, "wb").write(j.encode())
    sha = hashlib.sha256(open(r2.STATES_V6, "rb").read()).hexdigest()
    open(r2.STATES_V6.replace(".json", ".sha256"), "w", encoding="utf-8").write(sha + "  PREREG_REWARD_2_STATES.json\n")
    print(sha, len(sts), sum(s["sampler"] == "random" for s in sts[:160]), len({s["log"] for s in sts}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
