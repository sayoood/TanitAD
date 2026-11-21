"""Diff the SIX gate arms as PARSED NAMESPACES, not as intent.

`one_variable`: B1 - B0 must differ in EXACTLY ONE key. A past sweep was invalidated
because an arm silently changed the effective lambda alongside its named lever, so the
comparison is made on argparse output, never on the command strings.
"""
import importlib.util, sys, json, itertools

spec = importlib.util.spec_from_file_location(
    "rt", "/workspace/TanitAD/stack/scripts/refc_v3_train.py")
m = importlib.util.module_from_spec(spec)
sys.modules["rt"] = m
spec.loader.exec_module(m)

mk = None
for cand in ["build_parser", "make_parser", "_build_parser", "get_parser", "_parser"]:
    if hasattr(m, cand):
        mk = getattr(m, cand); print("ZZ_PARSER_FN", cand); break
if mk is None:
    print("ZZ_PARSER_FN NONE")
    print("ZZ_CANDIDATES", [n for n in dir(m) if "pars" in n.lower()][:20])
    sys.exit(3)

ROOT = "/workspace/experiments/wpc-oracle-gate"
JOIN = "/workspace/joins/b1_train_plus_eval_agents.jsonl.xz"
ANCH = ROOT + "/anchors.pt"

BASE = ("--arm hier --size base "
        "--v2-cache /root/data/train "
        "--v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz "
        "--eval-cache /root/data/eval "
        "--eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz "
        "--eval-every 500 --eval-batches 8 "
        "--image-hw 256 640 "
        "--steps 6000 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 "
        "--lr 1e-4 --warmup 2000 "
        "--log-every 50 --save-every 500 "
        "--u8-batches "
        "--nav-from-v7 "
        "--ego-state-inject --ego-dropout 0.5 "
        "--anchors " + ANCH + " "
        "--n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat "
        "--sel-accel-max 2.0 "
        "--sampler ddim --w-u0 0.5 "
        "--sel-refined --sel-score-emitted "
        "--goal-str --tac-goal-tok-head "
        "--agents oracle --agent-join " + JOIN + " --w-agent 0").split()

ARMS = {
    "B0":      BASE + ["--seed", "0", "--out", ROOT + "/B0", "--wp-index", "off"],
    "B1":      BASE + ["--seed", "0", "--out", ROOT + "/B1", "--wp-index", "on"],
    "B0r":     BASE + ["--seed", "1", "--out", ROOT + "/B0r", "--wp-index", "off"],
    "B1r":     BASE + ["--seed", "1", "--out", ROOT + "/B1r", "--wp-index", "on"],
    "B1const": BASE + ["--seed", "0", "--out", ROOT + "/B1const", "--wp-index", "on",
                       "--wp-index-mode", "const"],
    "B1shuf":  BASE + ["--seed", "0", "--out", ROOT + "/B1shuf", "--wp-index", "on",
                       "--wp-index-mode", "shuffle"],
}

ns = {}
for k, argv in ARMS.items():
    ns[k] = vars(mk().parse_args(argv))

def diff(a, b, ignore=("out",)):
    ka, kb = set(ns[a]), set(ns[b])
    d = {}
    for k in ka | kb:
        if k in ignore: continue
        va, vb = ns[a].get(k, "<ABSENT>"), ns[b].get(k, "<ABSENT>")
        if va != vb: d[k] = [va, vb]
    return d

out = {}
for a, b in [("B0","B1"), ("B0r","B1r"), ("B0","B0r"), ("B1","B1r"),
             ("B1","B1const"), ("B1","B1shuf")]:
    d = diff(a, b)
    out["%s_vs_%s" % (a, b)] = d
    print("ZZ_DIFF %s_vs_%s n=%d %s" % (a, b, len(d), json.dumps(d, default=str)))

# same-breath control: an arm against ITSELF must differ in ZERO keys, or the
# comparator is not measuring what it claims.
print("ZZ_CTRL B0_vs_B0 n=%d" % len(diff("B0","B0")))
json.dump({k: {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in out.items()},
          open("/workspace/ns_diff.json","w"), indent=1, sort_keys=True)
print("ZZ_WROTE /workspace/ns_diff.json")
