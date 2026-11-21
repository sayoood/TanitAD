import sys
p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s

old = '''        rows = scorer_dvb_rows(ns, data, profile)
        if profile.get("schedule_reach"):
            rows = rows + scorer_schedule_rows(ns, data, arm)
'''
assert s.count(old) == 1
s = s.replace(old, old + '''        if profile.get("data_expect"):
            rows = rows + scorer_expect_rows(profile, ns, data, arm)
''')

old2 = '''def check_dvb_scorer(argv: list, pf, profile: dict, arm=None, rehearsal=False) -> dict:'''
assert s.count(old2) == 1
fn = '''def _dotted(d, path):
    """data["a.b"] -> d["a"]["b"], or None when any hop is missing."""
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def scorer_expect_rows(profile: dict, ns, data: dict, arm=None) -> list:
    """SFT-4R (profile keys `data_expect`, `data_expect_min`, `data_expect_updates`): the trainer's OWN `data` event against the registered input counts, and the schedule re-derived from it.
    SFT-4 launched on 146,185 of 174,912 PDM labels because "complete" was never a computed fact; under --require-pdm `train_sets` is the number of served sets that HAVE a label, so asserting
    it equal to the served count (174,912) IS the completeness check, made on what the trainer will train on and not on a log line.  A floor (`data_expect_min`) is a registered minimum.
    The re-derived total is scorer_finetune.py:1363-1365 (`ceil(train_sets / batch x epochs / accum)`, capped by --max-updates when given) against `data_expect_updates`.
    Deliberate-regression arm `data_incomplete`: the same preflight with 83.6 % of the training sets and PDM labels (SFT-4's 146,185 of 174,912) -- every count row and the schedule row go RED."""
    d = dict(data)
    if arm_is("data_incomplete", arm):
        d = {**d, "train_sets": int(d.get("train_sets", 0) * 0.836), "lane_labels": {**(d.get("lane_labels") or {}), "pdm_train": int((_dotted(d, "lane_labels.pdm_train") or 0) * 0.836)}}
    rows = []
    for path, want in profile.get("data_expect", {}).items():
        got = _dotted(d, path)
        rows.append({"dest": "data_expect", "flag": "data." + path, "declared": want, "built": got, "how": f"data event: {path} (registered {want})", "ok": _close(want, got)})
    for path, floor in profile.get("data_expect_min", {}).items():
        got = _dotted(d, path)
        rows.append({"dest": "data_expect", "flag": "data." + path, "declared": f">= {floor}", "built": got,
                     "how": f"data event: {path} (a registered floor)", "ok": isinstance(got, int) and not isinstance(got, bool) and got >= floor})
    want_updates = profile.get("data_expect_updates")
    if want_updates is not None:
        n = d.get("train_sets")
        derived = None
        if isinstance(n, int) and n > 0:
            derived = int(math.ceil(n / ns.batch * ns.epochs / ns.accum))
            if ns.max_updates:
                derived = min(derived, ns.max_updates)
        rows.append({"dest": "data_expect", "flag": "schedule", "declared": want_updates, "built": derived,
                     "how": f"re-derived: ceil(data.train_sets {n} / --batch {ns.batch} x --epochs {ns.epochs} / --accum {ns.accum}) = the updates the run ends at (scorer_finetune.py:1363-1365)",
                     "ok": _close(want_updates, derived)})
    return rows


'''
s = s.replace(old2, fn + old2)
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
