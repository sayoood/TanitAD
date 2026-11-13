"""Add top-level tokens/logs to a families_run.py --per-window JSON made by the PRE-2207ef03 families_run.py.
Exactly what families_run.py at 2207ef03 writes (out["tokens"], out["logs"] = toks, logs over read_rows(--rows) filtered
by arm, sorted by token). Reads the pinned families_run.read_rows; writes a NEW file, never the input."""
import importlib.util, json, sys
fr_path, rows_glob, arm, src, dst = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("fr", fr_path); fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
rows = [r for r in fr.read_rows([rows_glob]) if arm in r["arms"]]
j = json.load(open(src, encoding="utf-8"))
assert j["arm"] == arm and j["n"] == len(rows), (j["arm"], j["n"], len(rows))
assert j["n_logs"] == len({r["log"] for r in rows})
j["tokens"] = [r["token"] for r in rows]; j["logs"] = [r["log"] for r in rows]
j["per_window_schema"] = "taniteval.per_window/1: blocks carry `_per_window`; arrays are in `tokens` order (read_rows sorts by token)"
j["_tokens_injected"] = {"by": "fam_tokens_inject.py (MM 2026-11-13)", "why": "pinned families_run.py predates 2207ef03, which writes tokens/logs under --per-window", "rows": rows_glob}
# per-window n must equal the token count in every frame that exported _per_window
for fr_name, f in j["frames"].items():
    pw = (f.get("block") or {}).get("_per_window")
    if isinstance(pw, dict) and "n" in pw:
        assert int(pw["n"]) == len(rows), (fr_name, pw["n"], len(rows))
        if pw.get("tokens") is not None: assert list(pw["tokens"]) == j["tokens"], fr_name
json.dump(j, open(dst, "w", encoding="utf-8"), indent=1)
print("ZZINJECT_OK", arm, len(rows), j["n_logs"])
