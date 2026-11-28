"""rehearse_pl.py -- card ADD114, lead's rehearsal of the planted-join part of control e7 on BANKED runs only (stand-ins): the banked label-form runs RLc of K0 and K30,
checked by the reader's own pl_check against the evaluation join (the join those runs received). It confirms on real files the open definition of the reader kit
(a logged step (ci, j) is the join row off[ci] + j) and that `present` of an RLc rule log is the rule's own test for a join. No new run of the card is opened.

  cd <podread root>; python rehearse_pl.py <stage dir with read_presattr10k.py> <t1 dir> <evaluation join> <model reader>
"""
import json
import sys

stage, pull, join, model = sys.argv[1:5]
sys.path.insert(0, stage)
import read_presattr10k as R  # noqa: E402

R.bind(model)
JL = R.load_join_arrays(join)
print("join keys", sorted(JL))
ok = True
for S, run in (("K0", "RLcKF5Hb20"), ("K30", "RLcKF5P30Hb20")):
    for m in (0, 1):
        logs = R.load_logs_setting(pull, run, m, S)
        if m == 0:
            print(S, run, "log keys", sorted(logs[0]))
        r = R.pl_check(R.rows4(logs), JL)
        ok = ok and r["PASS"]
        print("REHEARSE-PL", S, run, "s%d" % m, json.dumps(r))
print("ZZ REHEARSE-PL all_pass %s" % ok)
