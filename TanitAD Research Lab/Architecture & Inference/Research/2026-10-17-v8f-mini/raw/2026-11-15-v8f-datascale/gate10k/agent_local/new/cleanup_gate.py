"""cleanup_gate.py (pod): deletes the gate agent's OWN scratch data by literal absolute path and logs every file (path, size) in /root/dscale10k/gate/deletions_gate.log.
Never touches: tokens, evidence, item records, argv, dry records, banks of the study, checkpoints of the pilot, anything of STRAT1 or another session, merged/, the cache, the union view.
What goes: the resume smoke's runs and logs' checkpoints (re-creatable by gate/resume_smoke.py), the mint test scratch, the CPU build scratch of the init / probe items (re-creatable)."""
import os
import shutil
import time

D = "/root/dscale10k"
LOG = D + "/gate/deletions_gate.log"
TARGETS = [D + "/gate/resume_smoke", D + "/tmp/mint_test_U", D + "/tmp/init_B_s0", D + "/tmp/init_U_s0", D + "/tmp/init_B_rep", D + "/tmp/probe2_B_s0.A", D + "/tmp/probe2_B_rep.A",
           D + "/tmp/pt", D + "/tmp/pt2", D + "/tmp/pt3", D + "/gate/sb/U_smoke_A"]
tot = 0
with open(LOG, "a") as lg:
    for t in TARGETS:
        assert t.startswith(D + "/") and ".." not in t
        if os.path.isdir(t):
            for dp, dn, fn in os.walk(t):
                for f in fn:
                    p = os.path.join(dp, f)
                    n = os.path.getsize(p) if os.path.isfile(p) and not os.path.islink(p) else 0
                    tot += n
                    lg.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{p}\t{n}\n")
            shutil.rmtree(t, ignore_errors=True)
print("deleted bytes", tot, "log", LOG)
