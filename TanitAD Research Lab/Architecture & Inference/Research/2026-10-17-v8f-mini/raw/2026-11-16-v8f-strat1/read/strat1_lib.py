"""strat1_lib.py -- shared constants, hashing and window mapping for the STRAT1 read (PREREG_V8F_STRAT1, frozen card; READ_PLAN.md).
Pure numpy; no torch.  Evidence class of every number the readers print: MEASURED (ours, dumps / T0 evals, zero training); tier stamped per block."""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMIT = "e81bde01c03a73e811734c088320f101dcec5452"          # code of record: the commit every arm trained from (gate_token.json "commit")
TREE = os.environ.get("STRAT1_TREE", HERE + "/tree")          # git -c core.autocrlf=false archive e81bde01 stack taniteval
INP = os.environ.get("STRAT1_INPUTS", HERE + "/inputs")
CK = os.environ.get("STRAT1_CKPTS", HERE + "/ckpts")
OUTD = os.environ.get("STRAT1_OUT", HERE + "/out")
ARMS = ("C0_s0", "N1_s0", "S1_s0", "C0_s1", "N1_s1", "S1_s1", "C0_rep")
TEXT_ARMS = ("N1_s0", "S1_s0", "N1_s1", "S1_s1")              # cfg.nav_source == "text"
CTX_ARMS = ("S1_s0", "S1_s1")                                 # cfg.ctx_heads == "on"
PINS = {                                                      # sha256 of the pinned input files (inputs.sha256 on the pod, hashes re-checked on load)
    "navtok_table.npz": "2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992",
    "navtext_nav.npz": "03bd75bc3a3de17928681bedb8e46a48a891304de45d395c49cc876acf6e6160",
    "strat_labels_table.npz": "8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff",
    "join_eval141.npz": "f0ea722f59c1a4d8ba81ce7f9b204dd3a28950fc1c16778f63c2962786e5206a",
    "eval_grid_windows.json": "659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c",
}
EVAL_N_EPISODES, EVAL_N_WINDOWS = 141, 2399
NB, SEED = 2000, 0


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pinned(name, root=None, check=True):
    """path of a pinned input; its sha256 is re-hashed against PINS (a mismatch raises, nothing is read)."""
    p = os.path.join(root or INP, name)
    if check and name in PINS:
        got = sha256_file(p)
        if got != PINS[name]:
            raise RuntimeError(f"pinned input {name} sha256 {got[:12]} != {PINS[name][:12]}")
    return p


def grid_windows(grid_path, join_clip):
    """-> (ci [N], t [N]) of the eval grid in the evaluators' order (episode order of the join, then the grid's t list).  Refuses a clip order that differs from the join's."""
    g = json.load(open(grid_path, encoding="utf-8"))
    if list(g["order"]) != [str(c) for c in join_clip]:
        raise RuntimeError("eval grid episode order differs from the join's clip order")
    ci, t = [], []
    for i, key in enumerate(g["order"]):
        for tt in g["windows"][key]:
            ci.append(i)
            t.append(int(tt))
    return np.asarray(ci, np.int64), np.asarray(t, np.int64)


def rows_of(off, ci, t):
    """table row of window (ci, t): eval_off[ci] + t  (the NOW frame; BUILD_SPEC section 5, READ_PLAN 2.4)"""
    return np.asarray(off, np.int64)[np.asarray(ci, np.int64)] + np.asarray(t, np.int64)


def paired(CI, a, b, eid, nb=NB):
    """paired episode-cluster bootstrap of a - b over the windows finite in both -> dict, or n/a with n.  Estimator: 'another draw of EPISODES' only."""
    a, b, eid = (np.asarray(x, np.float64) if k < 2 else np.asarray(x) for k, x in enumerate((a, b, eid)))
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0, "n_episodes": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], eid[m], n_boot=nb, seed=SEED)
    return {"status": "OK", "delta": float(a[m].mean() - b[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def load_ci(tree=None):
    """taniteval/ci.py (pure numpy) loaded from its file in the read tree"""
    import importlib.util
    p = (tree or TREE) + "/taniteval/taniteval/ci.py"
    spec = importlib.util.spec_from_file_location("strat1_ci", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.SOURCE = p
    return m
