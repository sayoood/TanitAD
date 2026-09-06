"""RED<->GREEN MUTATION TEST on the v7.2 tactical labels.

⛔ THE QUESTION: does flipping the GT traffic-light colour on every labelled
window move ANYTHING the trainer consumes?

⭐ THE DISCRIMINATING CONTROL (same breath, same records, same code path):
a mutation of `a_tac.lon` -- a field that MUST move the supervised target.
A null on the target is only informative if the control moves.

ASCII-only output (cp1252 dev box).
"""
import gzip, json, sys, copy, dataclasses

sys.path.insert(0, sys.argv[2])
from tanitad.data import v7_labels as V

SRC = sys.argv[1]
TMP = sys.argv[3]

with gzip.open(SRC, "rt", encoding="utf-8") as fh:
    base = [json.loads(l) for l in fh if l.strip()]

FLIP = {"TRAFFIC_LIGHT_REACT_RED": "TRAFFIC_LIGHT_REACT_GREEN",
        "TRAFFIC_LIGHT_REACT_GREEN": "TRAFFIC_LIGHT_REACT_RED"}
STATE_FLIP = {"red": "green", "green": "red"}


def mutate_target(recs):
    """Flip RED<->GREEN: the token key AND the `state` field.

    ⛔ SINGLE-PASS REBUILD, not an in-place two-way swap. An in-place swap
    over {RED:GREEN, GREEN:RED} re-visits the entry it just inserted and
    flips it BACK -- self-cancelling on exactly the RED records, which
    silently halves the mutation while still reporting a flip count.
    (Measured here first: 1115 "flips" over 779 records, of which only the
    363 GREEN ones actually moved.)
    """
    out = copy.deepcopy(recs)
    n = 0
    for r in out:
        g_tac = r.get("g_tac") or {}
        goals = g_tac.get("goals") or {}
        if any(k in FLIP for k in goals):
            new_goals = {}
            for k, rec in goals.items():
                if k in FLIP:
                    if isinstance(rec, dict) and rec.get("state") in STATE_FLIP:
                        rec["state"] = STATE_FLIP[rec["state"]]
                    new_goals[FLIP[k]] = rec
                    n += 1
                else:
                    new_goals[k] = rec
            g_tac["goals"] = new_goals
        # also flip inside a_tac.serves_goals.goals_checked (the audit echo)
        sg = (r.get("a_tac") or {}).get("serves_goals") or {}
        gc = sg.get("goals_checked")
        if isinstance(gc, list):
            sg["goals_checked"] = [FLIP.get(t, t) for t in gc]
        # and the CoT referent, so the mutation is total
        sem = r.get("semantics") or {}
        refs = sem.get("referents")
        if isinstance(refs, list):
            sem["referents"] = [{"TRAFFIC_LIGHT_RED": "TRAFFIC_LIGHT_GREEN",
                                 "TRAFFIC_LIGHT_GREEN": "TRAFFIC_LIGHT_RED"}.get(t, t)
                                for t in refs]
    return out, n


def mutate_control(recs):
    """CONTROL: change `a_tac.lon` on exactly the SAME records the target
    mutation touched. This MUST move the supervised tac_lon target."""
    out = copy.deepcopy(recs)
    n = 0
    for r in out:
        goals = (r.get("g_tac") or {}).get("goals") or {}
        if not any("TRAFFIC_LIGHT" in k for k in goals):
            continue
        at = r.get("a_tac") or {}
        cur = at.get("lon")
        alt = "ACCELERATE" if cur != "ACCELERATE" else "BRAKE_TO"
        at["lon"] = alt
        n += 1
    return out, n


def write(recs, path):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return path


def load(path):
    labels, man = V.load_v7_labels(path, allow_oracle_nav=True)
    return labels, man


def supervised_view(labels):
    """EXACTLY the fields the trainer consumes as supervision + the goal signal."""
    return [(l.clip_id, l.tac_lat, l.tac_lon, l.str_action, l.str_goal,
             json.dumps(l.tac_anchor, sort_keys=True)) for l in labels]


def audit_view(labels):
    return [json.dumps(l.audit.get("goal_flags"), sort_keys=True) for l in labels]


print("=" * 74)
print("RED<->GREEN MUTATION TEST -- v7.2 tactical labels")
print("=" * 74)
print("source blob:", SRC)
print("n_records  :", len(base))

tl_recs = sum(1 for r in base
              if any("TRAFFIC_LIGHT" in k
                     for k in ((r.get("g_tac") or {}).get("goals") or {})))
print("records carrying a TRAFFIC_LIGHT_* tactical goal:", tl_recs)
print()

tgt, n_t = mutate_target(base)
ctl, n_c = mutate_control(base)
print("TARGET  mutation: flipped RED<->GREEN on %d goal entries" % n_t)
print("CONTROL mutation: changed a_tac.lon on %d records (same record set)" % n_c)

# ⛔ POSITIVE ASSERTION THAT EACH MUTATION LANDED IN THE JSON ITSELF.
# Without this, "0 changed downstream" is indistinguishable from "the
# mutation never happened" -- the same failure family as a grep whose file
# could not be read.
def _n_tok(recs, tok):
    return sum(1 for r in recs
               if tok in ((r.get("g_tac") or {}).get("goals") or {}))

def _n_lon(recs, tok):
    return sum(1 for r in recs if (r.get("a_tac") or {}).get("lon") == tok)

print()
print("MUTATION LANDED? (positive assertion on the mutated JSON)")
print("   RED  count  base=%4d target=%4d   (must swap)"
      % (_n_tok(base, "TRAFFIC_LIGHT_REACT_RED"), _n_tok(tgt, "TRAFFIC_LIGHT_REACT_RED")))
print("   GREEN count base=%4d target=%4d   (must swap)"
      % (_n_tok(base, "TRAFFIC_LIGHT_REACT_GREEN"), _n_tok(tgt, "TRAFFIC_LIGHT_REACT_GREEN")))
print("   a_tac.lon=ACCELERATE base=%4d control=%4d   (must rise)"
      % (_n_lon(base, "ACCELERATE"), _n_lon(ctl, "ACCELERATE")))
assert _n_tok(tgt, "TRAFFIC_LIGHT_REACT_RED") == _n_tok(base, "TRAFFIC_LIGHT_REACT_GREEN"), \
    "target mutation did not land"
print()

p0 = write(base, TMP + "/base.jsonl.gz")
p1 = write(tgt, TMP + "/target.jsonl.gz")
p2 = write(ctl, TMP + "/control.jsonl.gz")

L0, M0 = load(p0)
L1, M1 = load(p1)
L2, M2 = load(p2)

s0, s1, s2 = supervised_view(L0), supervised_view(L1), supervised_view(L2)
a0, a1, a2 = audit_view(L0), audit_view(L1), audit_view(L2)

d_t = sum(1 for x, y in zip(s0, s1) if x != y)
d_c = sum(1 for x, y in zip(s0, s2) if x != y)
da_t = sum(1 for x, y in zip(a0, a1) if x != y)
da_c = sum(1 for x, y in zip(a0, a2) if x != y)

print("-" * 74)
print("RESULT -- what moved in the SUPERVISED fields")
print("   (V7Label.tac_lat / tac_lon / str_action / str_goal / tac_anchor)")
print("-" * 74)
print("  TARGET  (red<->green flip) : %d / %d records changed" % (d_t, len(s0)))
print("  CONTROL (a_tac.lon change) : %d / %d records changed" % (d_c, len(s0)))
print()
print("  [audit-only channel, for reference]")
print("  TARGET  audit goal_flags   : %d / %d records changed" % (da_t, len(a0)))
print("  CONTROL audit goal_flags   : %d / %d records changed" % (da_c, len(a0)))
print()

print("-" * 74)
print("VERDICT")
print("-" * 74)
if d_c == 0:
    print("  INCONCLUSIVE -- the CONTROL did not move either. Dead rig; the")
    print("  null about the traffic light is NOT interpretable.")
elif d_t == 0:
    print("  The control moved %d records; the target moved 0." % d_c)
    print("  => The rig is LIVE and the red<->green flip is INVISIBLE to every")
    print("     field the trainer consumes as supervision.")
    print("  => The GT traffic-light colour DOES NOT REACH TRAINING.")
else:
    print("  Target moved %d records -- the colour DOES reach the loader's")
    print("  supervised output. Trace onward to the loss." % d_t)
print()

print("-" * 74)
print("WHY: the four supervised heads and their class counts")
print("-" * 74)
for k, v in V.HEADS.items():
    print("  %-12s %2d classes" % (k, len(v)))
from tanitad.models.vocab_v7 import TACTICAL_GOAL_TOKENS_V7 as TG
print("  TACTICAL_GOAL_TOKENS_V7 has %d tokens and is NOT among HEADS." % len(TG))
tl_tok = [t for t in TG if "TRAFFIC_LIGHT" in t]
print("  traffic-light tokens in that unsupervised tuple:", tl_tok)
print()
print("  V7Label dataclass fields (all the loader exposes):")
print("   ", [f.name for f in dataclasses.fields(V.V7Label)])
print("  -> no field carries g_tac.goals; it is routed to `audit`, which the")
print("     module docstring declares 'audit-only, NEVER a training input'.")
