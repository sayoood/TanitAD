#!/usr/bin/env python3
"""Log-disjoint TRAIN / HELDOUT split, written BEFORE any P1 diagnostic or rule score is computed.
Stratified by whether a log carries a CRUISE token, so both halves carry CRUISE logs. Seed 20261107, numpy default_rng,
logs sorted ascending by name first (order-independent). Half of each stratum (floor) goes to TRAIN."""
import hashlib
import json

import numpy as np

import sel_common as sc

D = sc.load()
logs = np.unique(D["log"])
cl = set(D["log"][D["cruise"]])
rng = np.random.default_rng(20261107)
train = []
for grp in (sorted(l for l in logs if l in cl), sorted(l for l in logs if l not in cl)):
    perm = rng.permutation(len(grp))
    train += [grp[i] for i in perm[: len(grp) // 2]]
train = sorted(train)
held = sorted(set(logs) - set(train))
m_tr = np.isin(D["log"], train)
out = {"seed": 20261107, "rule": "stratified by has-CRUISE-token; floor(n/2) of each stratum to TRAIN", "train_logs": train, "heldout_logs": held,
       "n_train_logs": len(train), "n_heldout_logs": len(held), "n_train_tokens": int(m_tr.sum()), "n_heldout_tokens": int((~m_tr).sum()),
       "n_train_cruise": int((m_tr & D["cruise"]).sum()), "n_heldout_cruise": int((~m_tr & D["cruise"]).sum())}
s = json.dumps(out, indent=1, sort_keys=True)
open(sc.OUT + "/split.json", "w").write(s)
print({k: v for k, v in out.items() if not k.endswith("_logs")}, "sha256", hashlib.sha256(s.encode()).hexdigest())
