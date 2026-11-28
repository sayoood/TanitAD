#!/usr/bin/env python3
"""BAR-4, at the ARM level rather than the dataclass level: the head loader's
guards, each shown to REFUSE a real corrupted bundle and to ACCEPT the same
bundle once the datum it objects to is restored.

⛔ WHY THE ARM-LEVEL DEMO IS SEPARATE FROM THE UNIT TESTS. The unit tests pin the
VOCABULARY; these pin the LOADER, which is the thing that stands between an
operator's flag and a banked dump. A guard that exists in the dataclass and not on
the path the operator actually takes protects nothing.

Each case is a PAIR -- refuse, then accept with the field restored -- so a refusal
cannot be an unrelated error on the way past the guard.
"""
import copy
import json
import os
import sys

import torch

R = r"C:\Users\Admin\dkhead_run"
sys.path.insert(0, os.path.join(R, "stack"))
sys.path.insert(0, os.path.join(R, "taniteval"))
sys.path.insert(0, os.path.join(R, "taniteval", "tools"))

import refav1_arm as ra                                          # noqa: E402

BUNDLE = os.path.join(R, "head", "gap_head_bundle_recal.pt")
TMP = os.path.join(R, "guard_tmp.pt")
good = torch.load(BUNDLE, map_location="cpu", weights_only=False)
CK_SHA = good["ckpt_sha256"]
CK_STEP = int(good["ckpt_step"])
D_STATE = int(good["d_state"])

cases = []


def try_load(b, **kw):
    torch.save(b, TMP)
    args = dict(ckpt_sha=CK_SHA, ckpt_step=CK_STEP, d_state=D_STATE)
    args.update(kw)
    try:
        ra.load_gap_head(TMP, **args)
        return None
    except SystemExit as e:
        return str(e)


def case(name, mutate, **kw):
    b = copy.deepcopy(good)
    mutate(b)
    err = try_load(b, **kw)
    ok = try_load(copy.deepcopy(good), **({} if kw else {}))
    cases.append({"guard": name,
                  "refused_the_bad_bundle": err is not None,
                  "message": (err or "")[:150],
                  "accepted_the_good_bundle": ok is None})
    print("%-42s refused=%-5s accepted_good=%-5s  %s"
          % (name, err is not None, ok is None, (err or "")[:88]))


case("wrong ckpt_sha256 (another trunk)",
     lambda b: b.__setitem__("ckpt_sha256", "0" * 64))
case("wrong ckpt_step",
     lambda b: b.__setitem__("ckpt_step", 12345))
case("wrong d_state",
     lambda b: b.__setitem__("d_state", 512))
case("wrong pooling geometry",
     lambda b: b.__setitem__("pool", [16, 20]))
case("wrong token grid",
     lambda b: b.__setitem__("grid", [8, 20]))
case("not a head bundle (kind)",
     lambda b: b.__setitem__("kind", "something_else"))
case("missing the `present` target",
     lambda b: b["targets"].pop("present"))

# the cross-fit selector: a clip with no out-of-fold head must REFUSE, and the
# same call on a known clip must succeed -- the same-breath control.
known = sorted(good["fold_of_clip"])[0]
try:
    ra._dk_head_for(good, "a-clip-that-was-never-in-the-fold-map",
                    "gap_all_lead")
    unknown_refused = False
except SystemExit:
    unknown_refused = True
h, f = ra._dk_head_for(good, known, "gap_all_lead")
cases.append({"guard": "clip absent from the fold map (leak guard)",
              "refused_the_bad_bundle": unknown_refused,
              "accepted_the_good_bundle": h is not None,
              "message": "an unknown clip has no OUT-OF-FOLD head; guessing one "
                         "would leak that clip's own labels into its prediction"})
print("%-42s refused=%-5s accepted_good=%-5s  (known clip -> fold %d)"
      % ("clip absent from fold map", unknown_refused, h is not None, f))

os.remove(TMP)
allpass = all(c["refused_the_bad_bundle"] and c["accepted_the_good_bundle"]
              for c in cases)
out = {"task": "D-REFAV1-DK-DECODED BAR-4 -- the ARM-LEVEL head-loader guards",
       "evidence_class": "MEASURED (ours)",
       "bundle": BUNDLE, "n_guards": len(cases), "ALL_PASS": allpass,
       "note": ("each guard is a PAIR: the mutated bundle is REFUSED and the "
                "unmutated one ACCEPTED in the same breath, so a refusal cannot "
                "be an unrelated error on the way past the guard"),
       "cases": cases}
p = os.path.join(R, "pkg", "raw", "guard_demo.json")
json.dump(out, open(p, "w"), indent=1)
print("\n%d/%d guards refuse the bad bundle AND accept the good one -> %s"
      % (sum(1 for c in cases if c["refused_the_bad_bundle"]
             and c["accepted_the_good_bundle"]), len(cases),
         "PASS" if allpass else "FAIL"))
print("wrote", p)
