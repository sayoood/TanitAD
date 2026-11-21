"""Does the PATCHED trainer accept the WP-C gate's OWN argv at pin time?

⛔ The argv is copied from `run_gate.sh::base_args()` verbatim, not retyped from
prose. Data paths are pod paths and are never opened -- `_pin_trainer_cfg` runs
before any dataset is built, which is exactly the stage a launch dies at when a
flag combination is refused.
"""
from __future__ import annotations
import importlib.util, json, os, sys, traceback

STACK = os.environ["TANIT_STACK"]
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))
_spec = importlib.util.spec_from_file_location(
    "refc_v3_train_gateargv", os.path.join(STACK, "scripts",
                                           "refc_v3_train.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

BASE = """--arm hier --size base
--v2-cache /root/data/train
--v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz
--eval-cache /root/data/eval
--eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz
--eval-every 500 --eval-batches 8
--image-hw 256 640
--steps 6000 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
--lr 1e-4 --warmup 2000
--log-every 50 --save-every 500
--u8-batches
--nav-from-v7
--ego-state-inject --ego-dropout 0.5
--anchors /workspace/experiments/wpc-oracle-gate/anchors.pt
--n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat
--sel-accel-max 2.0
--sampler ddim --w-u0 0.5
--sel-refined --sel-score-emitted
--goal-str
--tac-goal-tok-head
--agents oracle --agent-join /workspace/joins/b1_train_plus_eval_agents.jsonl.xz --w-agent 0
""".split()

ARMS = {
    "B0":      ["--seed", "0", "--out", "/tmp/B0", "--wp-index", "off"],
    "B1":      ["--seed", "0", "--out", "/tmp/B1", "--wp-index", "on"],
    "B0r":     ["--seed", "1", "--out", "/tmp/B0r", "--wp-index", "off"],
    "B1r":     ["--seed", "1", "--out", "/tmp/B1r", "--wp-index", "on"],
    "B1const": ["--seed", "0", "--out", "/tmp/B1c", "--wp-index", "on",
                "--wp-index-mode", "const"],
    "B1shuf":  ["--seed", "0", "--out", "/tmp/B1s", "--wp-index", "on",
                "--wp-index-mode", "shuffle"],
}
#: ⛔ THE SAME-BREATH CONTROL: the gate's argv with the join REMOVED must be
#: REFUSED by the branch this task added. A pin that accepts everything proves
#: nothing about the pins that accepted the six arms.
res = {}
for name, extra in ARMS.items():
    try:
        args = T.build_parser().parse_args(BASE + extra)
        cfg = T._pin_trainer_cfg(T.v3.refc_v3_sized_config("base", hier=True),
                                 args)
        st = T._seam_stamp(cfg, args)
        res[name] = {"pin": "OK",
                     "agents_enable": bool(cfg.core.agents.enable),
                     "agents_oracle": bool(cfg.core.agents.oracle),
                     "cross_agent": bool(cfg.core.decoder.cross_agent),
                     "wp_index_built": getattr(cfg.core.decoder, "wp_index",
                                               None) is not None,
                     "stamp_agents_oracle": st["agents"]["oracle"]}
    except BaseException as e:                              # noqa: BLE001
        res[name] = {"pin": "REFUSED", "exc": type(e).__name__,
                     "msg": str(e)[:300],
                     "tb": traceback.format_exc()[-300:]}
print(json.dumps(res, indent=2))
