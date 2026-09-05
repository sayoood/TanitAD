"""Patch kin_gate_eval.py: rank on the TRUE hier ranking, keep an as-published control."""
import io
import os

os.chdir("C:/Users/Admin/kingate/raw")
p = "kin_gate_eval.py"
s = io.open(p, encoding="utf-8").read()

s = s.replace('"route_logits", "nav_cmd", "has_lead", "eid", "wi"',
              '"sel_score_v3", "route_logits", "nav_cmd", "has_lead", "eid", "wi"')

a = '            acc["sel_score"].append(out["sel_score"].detach().float().cpu())'
b = ('            acc["sel_score"].append(out["sel_score"].detach().float().cpu())\n'
     '            sv3 = out.get("sel_score_v3")\n'
     '            acc["sel_score_v3"].append(\n'
     '                sv3.detach().float().cpu() if sv3 is not None\n'
     '                else out["sel_score"].detach().float().cpu())')
assert a in s
s = s.replace(a, b)

a = ('                    "sel_idx_identical":\n'
     '                        bool(torch.equal(out["sel_idx"], out2["sel_idx"])),')
b = ('                    "sel_idx_identical":\n'
     '                        bool(torch.equal(out["sel_idx"], out2["sel_idx"])),\n'
     '                    "sel_score_v3_present": bool("sel_score_v3" in out),')
assert a in s
s = s.replace(a, b)

a = ('    rank = bk["sel_score"].clone().float()\n'
     '    rank = rank.masked_fill(~bk["reach"], float("-inf"))\n'
     '    order = rank.argsort(dim=1, descending=True)\n'
     '    return {"fan2": fan2, "r_kin": r_kin, "sc": sc, "ade": ade, "order": order,\n'
     '            "c_feas": comp["feasibility"], "c_comf": comp["comfort"]}')
b = ('    # THE RANKING THE DEPLOYED MODEL ACTUALLY USES.\n'
     '    # refcv3 is the `hier` arm (config argv `--arm hier`), and refc.py:1763 argmaxes\n'
     '    # `rank`, which on that arm is `sel_score_v3` (the goal-seam-grafted score) masked\n'
     '    # by reach_keep -- NOT `sel_score`. MEASURED: reconstructing the ranking from\n'
     '    # `sel_score` disagrees with the model\'s own `sel_idx` on 35 of 400 windows\n'
     '    # (8.75 %), so a gate built on it ranks over a candidate set that does not contain\n'
     '    # the model\'s own pick on ~1 window in 11. `order` is the TRUE ranking; `order_ap`\n'
     '    # reproduces the as-published probe so the defect is measured, not asserted.\n'
     '    rank = bk["sel_score_v3"].clone().float().masked_fill(~bk["reach"], float("-inf"))\n'
     '    order = rank.argsort(dim=1, descending=True)\n'
     '    rank_ap = bk["sel_score"].clone().float().masked_fill(~bk["reach"], float("-inf"))\n'
     '    order_ap = rank_ap.argsort(dim=1, descending=True)\n'
     '    return {"fan2": fan2, "r_kin": r_kin, "sc": sc, "ade": ade, "order": order,\n'
     '            "order_ap": order_ap,\n'
     '            "c_feas": comp["feasibility"], "c_comf": comp["comfort"]}')
assert a in s
s = s.replace(a, b)

a = ('    for k in GATE_KS:\n'
     '        idx = torch.empty(W, dtype=torch.long)\n'
     '        for j in range(W):\n'
     '            cand = S["order"][j, :min(k, S["order"].shape[1])]\n'
     '            idx[j] = cand[S["r_kin"][j][cand].argmax()]\n'
     '        pick["gate%d" % k] = idx\n'
     '    pick["kin_only"] = S["r_kin"].argmax(dim=1)')
b = ('    for tag, key in (("gate", "order"), ("gateAP", "order_ap")):\n'
     '        for k in GATE_KS:\n'
     '            idx = torch.empty(W, dtype=torch.long)\n'
     '            for j in range(W):\n'
     '                cand = S[key][j, :min(k, S[key].shape[1])]\n'
     '                idx[j] = cand[S["r_kin"][j][cand].argmax()]\n'
     '            pick["%s%d" % (tag, k)] = idx\n'
     '    pick["kin_only"] = S["r_kin"].argmax(dim=1)')
assert a in s
s = s.replace(a, b)

a = ('    m_idx, g1_idx = pick["model"], pick["gate1"]\n'
     '    n_dis = int((m_idx != g1_idx).sum())\n'
     '    metric_keys = ["ade_m", "ade8_m", "fde_m", "peak_g"] + list(FLAGS)\n'
     '    maxdiff = {k: float(np.abs(rows["gate1"][k] - rows["model"][k]).max())\n'
     '               for k in metric_keys}\n'
     '    goff = {\n'
     '        "object_assert_selected_index_identical": bool(n_dis == 0),\n'
     '        "n_windows_disagreeing_on_index": n_dis,\n'
     '        "n_windows": int(len(m_idx)),\n'
     '        "disagreement_rate": float(n_dis) / max(1, len(m_idx)),\n'
     '        "metrics_max_abs_diff": maxdiff,\n'
     '        "metrics_all_exactly_zero": bool(all(v == 0.0 for v in maxdiff.values())),\n'
     '    }\n'
     '    goff["PASS"] = bool(goff["object_assert_selected_index_identical"]\n'
     '                        and goff["metrics_all_exactly_zero"])')
b = ('    metric_keys = ["ade_m", "ade8_m", "fde_m", "peak_g"] + list(FLAGS)\n'
     '    m_idx = pick["model"]\n'
     '\n'
     '    def _goff(name, ranking):\n'
     '        gi = pick[name]\n'
     '        nd = int((m_idx != gi).sum())\n'
     '        md = {k: float(np.abs(rows[name][k] - rows["model"][k]).max())\n'
     '              for k in metric_keys}\n'
     '        d = {"gate_off_arm": name, "ranking_used": ranking,\n'
     '             "object_assert_selected_index_identical": bool(nd == 0),\n'
     '             "n_windows_disagreeing_on_index": nd,\n'
     '             "n_windows": int(len(m_idx)),\n'
     '             "disagreement_rate": float(nd) / max(1, len(m_idx)),\n'
     '             "metrics_max_abs_diff": md,\n'
     '             "metrics_all_exactly_zero": bool(all(v == 0.0 for v in md.values()))}\n'
     '        d["PASS"] = bool(d["object_assert_selected_index_identical"]\n'
     '                         and d["metrics_all_exactly_zero"])\n'
     '        return d\n'
     '\n'
     '    goff = _goff("gate1", "sel_score_v3 (the ranking refc.py:1763 argmaxes)")\n'
     '    goff["as_published_control"] = _goff(\n'
     '        "gateAP1", "sel_score (the AS-PUBLISHED probe\'s ranking)")')
assert a in s
s = s.replace(a, b)

a = '        for rule in ["model"] + ["gate%d" % k for k in GATE_KS] + ["kin_only", "oracle"]:'
b = ('        for rule in (["model"] + ["gate%d" % k for k in GATE_KS]\n'
     '                     + ["gateAP1", "gateAP2"] + ["kin_only", "oracle"]):')
assert a in s
s = s.replace(a, b)

a = ('        print("  G-OFF(%s) %s  index_identical=%s disagree=%d/%d  metrics_all_zero=%s"\n'
     '              % (tag, "PASS" if g["PASS"] else "FAIL",\n'
     '                 g["object_assert_selected_index_identical"],\n'
     '                 g["n_windows_disagreeing_on_index"], g["n_windows"],\n'
     '                 g["metrics_all_exactly_zero"]))')
b = (a + '\n'
     '        ga = g["as_published_control"]\n'
     '        print("    as-published control (ranking = sel_score): %s  disagree=%d/%d'
     ' (%.2f pct)"\n'
     '              % ("PASS" if ga["PASS"] else "FAIL",\n'
     '                 ga["n_windows_disagreeing_on_index"], ga["n_windows"],\n'
     '                 100.0 * ga["disagreement_rate"]))')
assert a in s
s = s.replace(a, b)

a = '    for rule in ("model", "gate1", "gate2", "gate4", "kin_only", "oracle"):'
assert a in s
s = s.replace(a, '    for rule in ("model", "gate1", "gate2", "gate4", "gateAP2", '
                 '"kin_only", "oracle"):')
a = '    for rule in ("gate1", "gate2", "gate4", "kin_only", "oracle"):'
assert a in s
s = s.replace(a, '    for rule in ("gate1", "gate2", "gate4", "gateAP2", "kin_only", '
                 '"oracle"):')

io.open(p, "w", encoding="utf-8").write(s)
print("patched OK; sel_score_v3 refs:", s.count("sel_score_v3"))
