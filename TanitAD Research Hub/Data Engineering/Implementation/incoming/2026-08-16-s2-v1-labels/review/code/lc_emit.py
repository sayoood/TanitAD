"""The emission-rate change from the §LC ruling, MEASURED on the real
Engine A recompute for all 801 labeled clips (201 aug120 + 600 w120val).

Re-derives with the NEW s2_derive and compares against the BANKED v1 labels.
"""
import json
import os
import sys
from collections import Counter

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
import s2_derive                                            # noqa: E402

assert s2_derive.check_vocab_drift() == "checked", "vocab pins drifted"
print("vocab drift check: OK (pins == the real v6 module)")

banked = {}
for f in ("s2_labels_aug120.jsonl", "s2_labels_w120val.jsonl"):
    for line in open(os.path.join(PKG, "labels", f), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            banked[r["clip_id"]] = r

new_g, new_a, old_g, old_a = Counter(), Counter(), Counter(), Counter()
n = 0
changed = []
for f in ("engine_a_aug120.jsonl", "engine_a_w120val.jsonl"):
    for line in open(os.path.join(PKG, "labels", f), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        cid, ea = r["clip_id"], r["engine_a"]
        b = banked.get(cid)
        if b is None:
            continue
        # The banked record does not persist the raw VLM symbols, but it
        # persists EXACTLY the two fields s2_derive reads, inside the
        # corroboration block. Reconstruct them so the re-derivation is
        # faithful — passing {} would silently disable the ROUTE_TO gate and
        # manufacture a delta my change did not cause. (It did, first run.)
        gc = (b.get("g_str") or {}).get("corroboration") or {}
        ac = (b.get("a_str") or {}).get("corroboration") or {}
        sym = {"goal_kind": gc.get("vlm_goal_kind"),
               "actions": [{"verb": v.get("verb"),
                            "direction": v.get("direction")}
                           for v in (ac.get("vlm_verbs") or [])]}
        g = s2_derive.derive_g_str(ea, sym)
        a = s2_derive.derive_a_str(ea, sym)
        n += 1
        new_g[g["token"]] += 1
        new_a[a["token"]] += 1
        ob, ab = b.get("g_str") or {}, b.get("a_str") or {}
        old_g[ob.get("token")] += 1
        old_a[ab.get("token")] += 1
        if ob.get("token") != g["token"] or ab.get("token") != a["token"]:
            changed.append((cid, ob.get("token"), g["token"],
                            ab.get("token"), a["token"]))

print(f"\nre-derived {n} records\n")
w = 22
print(f"{'g_str token':<{w}}{'BANKED v1':>11}{'NEW':>8}{'delta':>8}")
print("-" * (w + 27))
for t in s2_derive.STRATEGIC_GOAL_TOKENS:
    o, ne = old_g.get(t, 0), new_g.get(t, 0)
    if o or ne:
        print(f"{t:<{w}}{o:>11}{ne:>8}{ne-o:>+8}")
print()
print(f"{'a_str token':<{w}}{'BANKED v1':>11}{'NEW':>8}{'delta':>8}")
print("-" * (w + 27))
for t in s2_derive.STRATEGIC_ACTION_TOKENS:
    o, ne = old_a.get(t, 0), new_a.get(t, 0)
    if o or ne:
        print(f"{t:<{w}}{o:>11}{ne:>8}{ne-o:>+8}")

print(f"\nrecords whose token changed: {len(changed)}/{n} "
      f"= {100*len(changed)/n:.2f}%")
print(f"LANE_TARGET        {old_g.get('LANE_TARGET',0)} -> "
      f"{new_g.get('LANE_TARGET',0)}   "
      f"({100*old_g.get('LANE_TARGET',0)/n:.2f}% -> "
      f"{100*new_g.get('LANE_TARGET',0)/n:.2f}%)")
print(f"PREPARE_LANE_CHANGE {old_a.get('PREPARE_LANE_CHANGE',0)} -> "
      f"{new_a.get('PREPARE_LANE_CHANGE',0)}  "
      f"({100*old_a.get('PREPARE_LANE_CHANGE',0)/n:.2f}% -> "
      f"{100*new_a.get('PREPARE_LANE_CHANGE',0)/n:.2f}%)")
dest = Counter((o_g, n_g) for _, o_g, n_g, _, _ in changed)
print("\nwhere the removed g_str labels went:")
for (o, ne), c in dest.most_common():
    print(f"   {o} -> {ne}: {c}")
dest_a = Counter((o_a, n_a) for _, _, _, o_a, n_a in changed)
print("where the removed a_str labels went:")
for (o, ne), c in dest_a.most_common():
    print(f"   {o} -> {ne}: {c}")
json.dump({"n": n, "old_g": dict(old_g), "new_g": dict(new_g),
           "old_a": dict(old_a), "new_a": dict(new_a),
           "changed": changed},
          open(os.path.join(
              r"C:\Users\Admin\AppData\Local\Temp\claude"
              r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
              r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad",
              "lc_emit.json"), "w", encoding="utf-8"), indent=1)
