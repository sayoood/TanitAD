"""Mutation proof for PREREG_V8F_HIST Amendment 1 (`--window-start`): every mutation must turn stack/tests/test_v8f_window_start.py RED; the unmutated tree must be GREEN before
and after, and every mutated file is restored byte-identical (in a `finally`, then asserted: `read_bytes() ==` the saved original). Deletes nothing.

  python mutate_a1.py <repo root> <python exe> [--check-anchors] [--part k/n] [--only "a||b"]

Run it on a byte-identical COPY of the tree (it mutates source files in place; the build ran it in `scratchpad/hist_m1`), never on the only copy of the work.
The five mutations the brief names are the first five groups: the flag never reaches train_windows, the held-age room is still computed from hist, built_levers omits window_start,
the G-DVB entry is removed, the default is changed to 3. The rest take each hop of the same path apart (parser, config, dataset, G-DVB's three read-backs, the effective-value rule,
the range check) so that a green run cannot hide a hop that does not run. The last ten ("EVAL FIX") take the evaluators apart: each of `v8f_mini_eval.py` / `v8f_t1_eval.py` drops the
keyword, hard-codes 3, passes None, passes hist - 1, or gains a second unused construction without the keyword. Anchors must occur EXACTLY ONCE (asserted before anything runs)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
DATA = ROOT / "stack" / "tanitad" / "data" / "v8f_mini_data.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval.py"
T1E = ROOT / "stack" / "scripts" / "v8f_t1_eval.py"
TEST = "stack/tests/test_v8f_window_start.py"

MUT = [
    # ---- the five named mutations
    (TRAIN, "NAMED 1a: the flag never reaches train_windows (the all-clips call, which sizes the budget)", "clips_all, hist=cfg.hist, start=cfg.window_start)", "clips_all, hist=cfg.hist)"),
    (TRAIN, "NAMED 1b: the flag never reaches train_windows (the trained clips' call)", "J, clips, hist=cfg.hist, start=cfg.window_start)", "J, clips, hist=cfg.hist)"),
    (DATA, "NAMED 2: the held-age room is still computed from hist", "        room = max(t - self.window_start, 0) ", "        room = max(t - (self.hist - 1), 0) "),
    (MODEL, "NAMED 3: built_levers omits window_start", '                "window_start": cfg.window_start_effective,                  # Amendment 1: the EFFECTIVE first-window step (hist - 1 when the flag is absent)\n', ""),
    (TRAIN, "NAMED 4: the G-DVB entry for the flag is removed", '    "window_start": "lever:window_start",', "    "),
    (TRAIN, "NAMED 5: the flag's default is 3", 'ap.add_argument("--window-start", dest="window_start", type=int, default=-1,', 'ap.add_argument("--window-start", dest="window_start", type=int, default=3,'),
    # ---- the path, hop by hop
    (TRAIN, "make_config does not hand the flag to the config", "hist=a.hist, hmax=hmax, window_start=a.window_start, n_q=a.n_q,", "hist=a.hist, hmax=hmax, n_q=a.n_q,"),
    (TRAIN, "the trainer builds its dataset without the flag (the clamp stays hist - 1)", "max_age_s=cfg.max_age_s,\n                          window_start=cfg.window_start)", "max_age_s=cfg.max_age_s)"),
    (DATA, "V8FMiniWindows ignores its window_start argument", "self.window_start = effective_window_start(self.hist, window_start)", "self.window_start = effective_window_start(self.hist, None)"),
    (DATA, "train_windows ignores its start argument", "    first = effective_window_start(hist, start)\n", "    first = hist - 1\n"),
    (DATA, "the absent window start is 0, not hist - 1", "    if window_start is None or int(window_start) == WINDOW_START_ABSENT:\n        return hist - 1\n",
     "    if window_start is None or int(window_start) == WINDOW_START_ABSENT:\n        return 0\n"),
    (DATA, "effective_window_start accepts a start above hist - 1 (no range check)", "    if not 0 <= ws <= hist - 1:\n", "    if False:\n"),
    (MODEL, "the config accepts an out-of-range window_start", "        if self.window_start != -1 and not 0 <= self.window_start <= self.hist - 1:\n", "        if False:\n"),
    (MODEL, "the config's window_start defaults to 3", "    window_start: int = -1 ", "    window_start: int = 3 "),
    (MODEL, "the effective window start ignores the flag", "        return self.hist - 1 if self.window_start == -1 else self.window_start\n", "        return self.hist - 1\n"),
    (MODEL, "built_levers reports the raw field (-1 when absent), not the effective step", '"window_start": cfg.window_start_effective,', '"window_start": cfg.window_start,'),
    (MODEL, "built_levers omits hist and hmax", '                "hist": int(self.time.shape[0]), "hmax": cfg.hmax,           #', '                #'),
    (MODEL, "built_levers reports a hist one too small", '"hist": int(self.time.shape[0]),', '"hist": int(self.time.shape[0]) - 1,'),
    # ---- G-DVB
    (TRAIN, "G-DVB: the built hist / hmax / window_start lever check is off", '        if built.get(k) != want:\n            bad.append(f"lever {k}: declared {want} but built {built.get(k)}")\n',
     '        if False:\n            bad.append(f"lever {k}: declared {want} but built {built.get(k)}")\n'),
    (TRAIN, "G-DVB: the window list's first t is not read back (the dataset's origin is checked twice instead)", '("window_t_min", "the first t of the training window list")',
     '("ds_window_start", "the first t of the training window list")'),
    (TRAIN, "G-DVB: the dataset's clamp origin is not read back (the window list's first t is checked twice instead)", '("ds_window_start", "the dataset\'s held-age clamp origin")',
     '("window_t_min", "the dataset\'s held-age clamp origin")'),
    (TRAIN, "G-DVB: a declared flag needs no read-back from the data", "        if k in data_info or a.window_start != -1:\n", "        if k in data_info:\n"),
    (TRAIN, "G-DVB: the absent flag is held to 3, not to hist - 1", "ws_decl = a.hist - 1 if a.window_start == -1 else a.window_start", "ws_decl = 3 if a.window_start == -1 else a.window_start"),
    (TRAIN, "build_data records the asked-for start as the 'measured' first t (the lie hides)", '"window_t_min": min((w[1] for w in allw), default=None)})',
     '"window_t_min": D.effective_window_start(cfg.hist, cfg.window_start)})'),
    (TRAIN, "the dataset's clamp origin is recorded from the config, not read back from the dataset", 'info["ds_window_start"] = ds.window_start', 'info["ds_window_start"] = D.effective_window_start(cfg.hist, cfg.window_start)'),
    (TRAIN, "the trainer does not record the dataset's clamp origin at all", '    info["ds_window_start"] = ds.window_start ', "    pass "),
    # ---- the evaluators (eval-side fix): the checkpoint's window_start must reach the dataset the EVAL builds
    (EVAL, "EVAL FIX: the mini evaluator builds its dataset WITHOUT the checkpoint's window_start (the clamp stays hist - 1)", 'max_age_s=cfg.max_age_s,\n                          window_start=cfg.window_start)', "max_age_s=cfg.max_age_s)"),
    (EVAL, "EVAL FIX: the mini evaluator hard-codes window_start=3", "window_start=cfg.window_start)", "window_start=3)"),
    (EVAL, "EVAL FIX: the mini evaluator passes window_start=None (absent)", "window_start=cfg.window_start)", "window_start=None)"),
    (EVAL, "EVAL FIX: the mini evaluator passes hist - 1 instead of the checkpoint's value", "window_start=cfg.window_start)", "window_start=cfg.hist - 1)"),
    (EVAL, "EVAL FIX: the mini evaluator gains a SECOND, unused construction without the keyword (only the static check can see it)", '    tot_hmax = cfg.hmax\n', '    tot_hmax = cfg.hmax\n    _unused = lambda: D.V8FMiniWindows(None, None, [], hist=cfg.hist)\n'),
    (T1E, "EVAL FIX: the T1 evaluator builds its dataset WITHOUT the checkpoint's window_start (the clamp stays hist - 1)", 'max_age_s=cfg.max_age_s,\n                          window_start=cfg.window_start)', "max_age_s=cfg.max_age_s)"),
    (T1E, "EVAL FIX: the T1 evaluator hard-codes window_start=3", "window_start=cfg.window_start)", "window_start=3)"),
    (T1E, "EVAL FIX: the T1 evaluator passes window_start=None (absent)", "window_start=cfg.window_start)", "window_start=None)"),
    (T1E, "EVAL FIX: the T1 evaluator passes hist - 1 instead of the checkpoint's value", "window_start=cfg.window_start)", "window_start=cfg.hist - 1)"),
    (T1E, "EVAL FIX: the T1 evaluator gains a SECOND, unused construction without the keyword (only the static check can see it)", '    proxy = _OneClipStore()\n', '    proxy = _OneClipStore()\n    _unused = lambda: D.V8FMiniWindows(None, None, [], hist=cfg.hist)\n'),
]
if "--only" in sys.argv:
    _keys = sys.argv[sys.argv.index("--only") + 1].split("||")
    MUT = [m for m in MUT if any(k in m[1] for k in _keys)]
if "--part" in sys.argv:
    _k, _n = (int(x) for x in sys.argv[sys.argv.index("--part") + 1].split("/"))
    MUT = MUT[_k - 1::_n]


def run():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-rf", "-p", "no:cacheprovider", TEST], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(ROOT / "stack"), str(ROOT / "taniteval")])})
    lines = r.stdout.strip().splitlines() or ["<no output>"]
    failed = [x.split("::", 1)[1].split(" - ")[0].strip() for x in lines if x.startswith("FAILED ") and "::" in x]
    return r.returncode, lines[-1] + (f"  [{', '.join(failed)}]" if failed else "") + ("" if r.returncode == 0 or failed else "  " + " | ".join(r.stderr.strip().splitlines()[-2:]))


def apply(m, b):
    olds, news = (m[2], m[3]) if isinstance(m[2], list) else ([m[2]], [m[3]])
    seg = b
    for old, new in zip(olds, news):
        o, n = old.encode(), new.encode()
        assert seg.count(o) == 1, f"{m[1]}: anchor {old[:70]!r} found {seg.count(o)} times in {m[0].name}"
        assert o != n, f"{m[1]}: the mutation changes nothing"
        seg = seg.replace(o, n)
    return seg


orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
for m in MUT:
    apply(m, orig[m[0]])
print(f"anchors OK: {len(MUT)} mutants, every anchor occurs exactly once", flush=True)
if "--check-anchors" in sys.argv:
    sys.exit(0)
rc, last = run()
print(f"BASELINE rc={rc} {last}", flush=True)
assert rc == 0, "the unmutated tree must be GREEN"
red, survivors = 0, []
for m in MUT:
    path, name = m[0], m[1]
    b = orig[path]
    path.write_bytes(apply(m, b))
    try:
        rc, last = run()
    finally:
        path.write_bytes(b)
        assert path.read_bytes() == orig[path], f"{path} was not restored byte for byte"
    red += rc != 0
    if rc == 0:
        survivors.append(name)
    print(f"{'RED  ' if rc != 0 else 'GREEN'} {path.name}: {name} -> {last}", flush=True)
assert all(p.read_bytes() == orig[p] for p in orig), "a file was not restored"
rc, last = run()
print(f"AFTER rc={rc} {last}")
print(f"{red} / {len(MUT)} mutations RED" + (f"; SURVIVORS: {survivors}" if survivors else ""))
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
