"""gen_pins.py <queue file>: print the pins test_v8f_orc_chain.py embeds, computed from a queue file -- sha256[:16] of "\n" + each label's block text (the programme's convention), and the sha256[:16] of the dry-run
output (lines containing .py) of a fixed list of other stages under a fixed environment. Run on the 98713259 queue and on the edited one: the label pins of the first must all reappear in the second."""
import hashlib
import json
import os
import re
import subprocess
import sys

q = sys.argv[1]
text = open(q, "rb").read().decode("utf-8").replace("\r\n", "\n")
parts = re.split(r"(?m)^(:[A-Za-z0-9_]+)$", text)
blocks = {parts[i][1:]: parts[i + 1] for i in range(1, len(parts), 2)}
pins = {k: hashlib.sha256(("\n" + v).encode()).hexdigest()[:16] for k, v in blocks.items()}
STAGES = ["gate", "train M2b 0", "train M2b 1 pool8x20", "eval M2b 0", "floors", "compare", "tac", "tacgrp", "tacgrp2", "tacabs", "gatearm M2b", "tacabsarm M2b 0", "gatewc", "trainwc 0", "evalwc 0", "gatepgt", "trainpgt 0",
          "evalpgt 0", "gategrid M2b", "traingrid 0 M2b", "evalgrid 0 M2b", "gatehist", "trainhist 1", "evalhist 0", "gatedscale", "traindscale 0", "evaldscale 0", "kind", "gatemixo", "trainmixo 0", "trainmixo 1", "evalmixo 0",
          "gateflat", "trainflat 1", "evalflat 0", "waccoacc", "gateoacc", "trainoacc 0", "trainoacc 1", "evaloacc 1", "fitoacc 0"]
env = dict(os.environ, V8F_DRYRUN="1", REPO=r"Z:\r", PY=r"Z:\py.exe", RUNSM=r"Z:\runsm", CACHEFULL=r"Z:\cf", V9=r"Z:\v9", WACC="0.5")
dry = {}
for st in STAGES:
    r = subprocess.run(f'cmd /c ""{q}" {st}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    lines = [x for x in r.stdout.splitlines() if ".py" in x]
    assert r.returncode == 0 and lines, (st, r.returncode, r.stdout[:200])
    dry[st] = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
print(json.dumps({"labels": pins, "dry": dry}, indent=0))
