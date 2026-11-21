p = "selection_quality.py"
s = open(p, encoding="utf-8", newline="").read()

def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new)

rep('''    python eval/selection_quality.py read   --table table_N.npz --arm-csv ... --out selection_quality_N.json
''', '''    python eval/selection_quality.py read   --table table_N.npz --arm-csv ... --out selection_quality_N.json
    python eval/selection_quality.py level  --arm-csv <the arm's harness csv> [--out level_N.json]   # the arm's OWN level CI (the PREREG goal-bar test): csv only, no 64 columns, seconds
''')
rep('''def read_table(table_npz, out=None, arm_csv=None) -> dict:''', '''def level_from_csv(arm_csv, token_log: dict, n: int = N_BOOT, seed: int = SEED) -> dict:
    """the arm's level (mean PDMS x100) with the registered paired log-cluster bootstrap CI, from its harness csv alone. This is the number PREREG_P6's goal-bar test (94.55) needs;
    it does NOT depend on the 64-column selection-quality table (4 to 6.4 h). Every csv token must have a log; an unknown token is an error, never a silent drop."""
    rows = PE.read_csv(arm_csv)
    unknown = [t for t in rows if t not in token_log]
    if not rows or unknown:
        raise ValueError(f"{arm_csv}: {len(rows)} tokens, {len(unknown)} without a log (e.g. {unknown[:3]}): not a level")
    per = {t: float(r["score"]) * 100.0 for t, r in rows.items()}
    mu, lo, hi = PE.boot(per, {t: token_log[t] for t in per}, n=n, seed=seed)
    return {"arm_csv": str(arm_csv), "n_tokens": len(per), "n_logs": len({token_log[t] for t in per}), "level_x100": {"mean": mu, "ci95": [lo, hi]},
            "estimator": ESTIMATOR, "tier": TIER, "evidence_class": "MEASURED on the arm's own harness csv"}


def read_table(table_npz, out=None, arm_csv=None) -> dict:''')
rep('''    r = sp.add_parser("read")
''', '''    lv_ = sp.add_parser("level")
    lv_.add_argument("--arm-csv", required=True)
    lv_.add_argument("--out", default=None)
    r = sp.add_parser("read")
''')
rep('''    res = read_table(a.table, a.out)
''', '''    if a.cmd == "level":
        import gzip
        exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
        res = level_from_csv(a.arm_csv, {t: v["log_name"] for t, v in exp.items()})
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        lv = res["level_x100"]
        print("ZZSELQ_LEVEL", f"{lv['mean']:.4f} [{lv['ci95'][0]:.4f}, {lv['ci95'][1]:.4f}] n={res['n_tokens']} logs={res['n_logs']}")
        return 0
    res = read_table(a.table, a.out)
''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
