#!/usr/bin/env python3
"""Apply the B8 G-DVB edits to the P6-fixed gate_checks.py / launch_gate.py in a tree.  usage: apply_gdvb.py <refe dir>"""
import sys, os

refe = sys.argv[1]


def edit(path, pairs):
    s = open(path, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, f"{os.path.basename(path)}: snippet occurs {s.count(old)} times: {old[:80]!r}"
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(s)


GC = os.path.join(refe, "gate_checks.py")
LG = os.path.join(refe, "launch_gate.py")

edit(GC, [
    # docstring: the arm list
    ("aux_lever_ignored, con_temp_ignored); G-PREFLIGHT", "aux_lever_ignored, con_temp_ignored, carve_feed_ignored); G-PREFLIGHT"),
    # (a) the registry
    ('    "epo_con_temp": "built",           # PICKCON:',
     '    # B8 (raw/2026-10-18-r0e-carvefeed): the dual\'s FEED.  epo_con_feed / epo_carve_sha256 / epo_carve_micro are `built`: observed through the data event\'s `epo_con_carve` block, which exists iff the carve loader was BUILT.\n'
     '    # epo_carve_logs is `data` (the list\'s BYTES): launch_gate.RLS1_DATA lists --epo-carve-logs, so the PASS token fingerprints the file, and the BUILT block\'s sha256 is a second binding.\n'
     '    "epo_con_feed": "built", "epo_carve_logs": "data", "epo_carve_sha256": "built", "epo_carve_micro": "built",\n'
     '    "epo_con_temp": "built",           # PICKCON:'),
    # (b) the block's keys
    ('SCORER_EPO_CON_COMPS = {"g5": ("ttc", "nc")}',
     'SCORER_EPO_CARVE_KEYS = ("feed", "logs", "sets", "sha256", "micro", "train_rows_in_carve_logs", "carve_rows_in_heldout_logs")     # B8: the keys of the data event\'s `epo_con_carve` block (written iff --epo-con-feed carve)\n'
     'SCORER_EPO_CON_COMPS = {"g5": ("ttc", "nc")}'),
    # (c) the observed rows
    ('                add("epo_con_" + k, getattr(ns, "epo_con_" + k), blk.get(k), f"data event: epo_con.{k}" + why)\n',
     '                add("epo_con_" + k, getattr(ns, "epo_con_" + k), blk.get(k), f"data event: epo_con.{k}" + why)\n'
     '        # B8: the dual\'s FEED against the BUILT carve block (present iff the carve loader exists; absent = the training micro-batches feed the dual, every trainer before B8)\n'
     '        cblk = data.get("epo_con_carve") if isinstance(data.get("epo_con_carve"), dict) else None\n'
     '        cb = cblk or {}\n'
     '        feed = getattr(ns, "epo_con_feed", "train")\n'
     '        add("epo_con_feed", feed, "carve" if cb.get("feed") == "carve" else "train", "data event: epo_con_carve.feed (BUILT: the block exists iff the carve loader was built; absent = the training micro-batches feed the dual)")\n'
     '        add("epo_carve_sha256", getattr(ns, "epo_carve_sha256", ""), cb.get("sha256", ""), "data event: epo_con_carve.sha256 (the sha256 of the bytes the trainer READ; empty with no carve)")\n'
     '        add("epo_carve_micro", getattr(ns, "epo_carve_micro", 1), cb.get("micro", 1), "data event: epo_con_carve.micro (carve micro-batches per update; 1 with no carve)")\n'
     '        if feed == "carve":\n'
     '            add("epo_carve_logs", [0, 0, True], [cb.get("train_rows_in_carve_logs"), cb.get("carve_rows_in_heldout_logs"), bool(cb.get("logs", 0) > 0 and cb.get("sets", 0) > 0)],\n'
     '                "data event: no training row in a carve log, no carve row in an RL-val log, and the carve is non-empty")\n'),
    # (d) the regression arm
    ('        rows = scorer_dvb_rows(ns, data, profile)\n',
     '        if arm_is("carve_feed_ignored", arm) and isinstance(data.get("epo_con"), dict):          # B8: the trainer built the OPPOSITE feed to the argv\'s (a declared carve with no carve block; a declared train with one)\n'
     '            data = ({k: v for k, v in data.items() if k != "epo_con_carve"} if isinstance(data.get("epo_con_carve"), dict)\n'
     '                    else dict(data, epo_con_carve={"feed": "carve", "logs": 1, "sets": 1, "sha256": "0" * 64, "micro": 1, "train_rows_in_carve_logs": 0, "carve_rows_in_heldout_logs": 0}))\n'
     '        rows = scorer_dvb_rows(ns, data, profile)\n'),
    # (e) the G7 carve re-judge
    ('def judge_g13(g13, ns) -> list:',
     'def judge_g7_carve(g7, ns) -> list:\n'
     '    """B8: G7 (heldout_disjoint) re-judged from its RAW carve fields against the argv; [] = proved.  With the default feed G7 must carry NO carve field (a carve measurement on a train-fed arm would be another\n'
     '    measurement).  With --epo-con-feed carve: the list the trainer READ is the registered one (sha256), no training row sits in a carve log, no carve row in an RL-val log, the carve holds >= one batch, and one REAL\n'
     '    carve forward through the dual\'s own feed returned a finite violation for every constrained component and as many micro-batches as the argv asks."""\n'
     '    r = []\n'
     '    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)          # noqa: E731\n'
     '    g7 = g7 if isinstance(g7, dict) else {}\n'
     '    carve_keys = sorted(k for k in g7 if k.startswith("carve_") or k == "train_rows_in_carve_logs")\n'
     '    if getattr(ns, "epo_con_feed", "train") != "carve":\n'
     '        return [f"G7: carve fields {carve_keys} on an argv whose --epo-con-feed is train"] if carve_keys else []\n'
     '    if g7.get("carve_sha256") != getattr(ns, "epo_carve_sha256", None):\n'
     '        r.append(f"G7: the carve list the trainer read ({str(g7.get(\'carve_sha256\'))[:16]}..) is not the registered --epo-carve-sha256")\n'
     '    if g7.get("train_rows_in_carve_logs") != 0:\n'
     '        r.append(f"G7: {g7.get(\'train_rows_in_carve_logs\')!r} training row(s) sit in a carve log (must be exactly 0): the carve is not held out")\n'
     '    if g7.get("carve_rows_in_heldout_logs") != 0:\n'
     '        r.append(f"G7: {g7.get(\'carve_rows_in_heldout_logs\')!r} carve row(s) sit in an RL-val log (must be exactly 0)")\n'
     '    if not (fin(g7.get("carve_sets")) and g7["carve_sets"] >= getattr(ns, "batch", 1) and fin(g7.get("carve_logs")) and g7["carve_logs"] > 0):\n'
     '        r.append(f"G7: the carve is smaller than one batch or has no log: sets {g7.get(\'carve_sets\')!r}, logs {g7.get(\'carve_logs\')!r}")\n'
     '    if g7.get("carve_probe_n") != getattr(ns, "epo_carve_micro", None):\n'
     '        r.append(f"G7: the carve probe scored {g7.get(\'carve_probe_n\')!r} micro-batches, not the argv\'s --epo-carve-micro {getattr(ns, \'epo_carve_micro\', None)!r}")\n'
     '    want = list(SCORER_EPO_CON_COMPS.get(getattr(ns, "epo_con", "off"), ()))\n'
     '    pg = g7.get("carve_probe_g") if isinstance(g7.get("carve_probe_g"), dict) else {}\n'
     '    if sorted(pg) != sorted(want) or not all(fin(v) for v in pg.values()):\n'
     '        r.append(f"G7: the carve probe\'s violation is not finite for every constrained component {want}: {pg}")\n'
     '    return r\n'
     '\n\ndef judge_g13(g13, ns) -> list:'),
])

edit(LG, [
    ('"--teacher-lane-train", "--teacher-lane-heldout", "--epo-terms-train", "--epo-terms-heldout", "--aux-drv-labels")',
     '"--teacher-lane-train", "--teacher-lane-heldout", "--epo-terms-train", "--epo-terms-heldout", "--aux-drv-labels", "--epo-carve-logs")'),
    ('"--epo-terms-train", "--epo-terms-heldout", "--epo-ref")\n# --aux-drv-shuffle-seed',
     '"--epo-terms-train", "--epo-terms-heldout", "--epo-ref",\n                   "--epo-con-feed", "--epo-carve-logs", "--epo-carve-sha256", "--epo-carve-micro")          # B8: the carve feed of the dual; a non-EPO arm never carries it\n# --aux-drv-shuffle-seed'),
    ('    g12 = next((v for k, v in gates.items() if prefix[k] == "G12"), None)\n',
     '    g7 = next((v for k, v in gates.items() if prefix[k] == "G7"), None)\n'
     '    if g7 is not None:                                       # G7\'s carve fields (B8) are re-judged from their raw values too\n'
     '        reasons += C.judge_g7_carve(g7, ns)\n'
     '    g12 = next((v for k, v in gates.items() if prefix[k] == "G12"), None)\n'),
])
print("applied")
