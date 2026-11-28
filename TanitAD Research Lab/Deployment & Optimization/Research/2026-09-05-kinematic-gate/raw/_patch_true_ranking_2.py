"""Apply the same TRUE-RANKING correction to the two offline readers.

`gate_share_of_gap.py` and `longitudinal_lead.py` both rebuilt the gate's candidate order from
`sel_score`. On the `hier` arm that is the PRE-GRAFT core score; refc.py:1763 argmaxes
`sel_score_v3` masked by `reach_keep`, and the two disagree with the model's own `sel_idx` on
35 of 400 windows (8.75 %). Both readers now prefer `sel_score_v3` when the bank carries it and
say so in the artifact, so `gate1` is the model by construction.
"""
import io
import os

os.chdir("C:/Users/Admin/kingate/raw")

OLD = ('    rank = bk["sel_score"].float().masked_fill(~bk["reach"].bool(), float("-inf"))\n'
       '    order = rank.argsort(dim=1, descending=True)')
NEW = ('    # THE RANKING THE DEPLOYED MODEL ACTUALLY USES: on the `hier` arm refc.py:1763\n'
       '    # argmaxes `sel_score_v3` (goal-seam-grafted) masked by reach_keep, NOT `sel_score`.\n'
       '    _rk = "sel_score_v3" if "sel_score_v3" in bk else "sel_score"\n'
       '    rank = bk[_rk].float().masked_fill(~bk["reach"].bool(), float("-inf"))\n'
       '    order = rank.argsort(dim=1, descending=True)')

for p in ("gate_share_of_gap.py", "longitudinal_lead.py"):
    s = io.open(p, encoding="utf-8").read()
    assert OLD in s, p
    s = s.replace(OLD, NEW)
    # record which ranking was used, and assert gate1 == model
    if p == "gate_share_of_gap.py":
        a = '        "rules": rows,\n    }'
        b = ('        "rules": rows,\n'
             '        "ranking_key_used": _rk,\n'
             '        "gate1_equals_model_by_construction":\n'
             '            bool(int((pick["gate1"] != pick["model"]).sum()) == 0),\n'
             '        "gate1_index_disagreements": int((pick["gate1"] != pick["model"]).sum()),\n'
             '    }')
        assert a in s, p
        s = s.replace(a, b)
        a = ('    print("=== P4: WHAT SHARE OF THE 8.56x DOES THE GATE CLOSE? (T0, %dw/%dep) ==="\n'
             '          % (W, out["n_episodes"]))')
        b = (a + '\n'
             '    print("  ranking used: %s   gate1==model on all windows: %s (disagree %d)"\n'
             '          % (_rk, out["gate1_equals_model_by_construction"],\n'
             '             out["gate1_index_disagreements"]))')
        assert a in s, p
        s = s.replace(a, b)
    else:
        a = '           "abs": res, "paired_vs_model": paired,'
        b = ('           "ranking_key_used": _rk,\n'
             '           "gate1_index_disagreements": int((pick["gate1"] != pick["model"]).sum()),\n'
             '           "abs": res, "paired_vs_model": paired,')
        assert a in s, p
        s = s.replace(a, b)
        a = ('    print("  lead windows %d of %d (%d episodes); dt 0.50 s; no-lead windows masked'
             ' to NaN"\n          % (out["n_lead_windows"], W, out["n_lead_episodes"]))')
        b = (a + '\n'
             '    print("  ranking used: %s ; gate1 index disagreements with model: %d"\n'
             '          % (_rk, out["gate1_index_disagreements"]))')
        assert a in s, p
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)
    print("patched", p)
