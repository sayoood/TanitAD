"""Quick (unmutated + selected mutations) run of eval/test_safeobj_outcome.py from the patched_full tree. NOT part of the package; the full run is the lane."""
import importlib.util, sys, tempfile, time
from pathlib import Path
ROOT = Path("C:/Users/Admin/pickcon/patched_full/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("tso", ROOT / "test_safeobj_outcome.py")
m = importlib.util.module_from_spec(spec); sys.modules["tso"] = m; spec.loader.exec_module(m)
only = sys.argv[1:]            # substrings of mutation names to run
src = m.TOOL_PATH.read_text(encoding="utf-8")
t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    W, A = m.build_world(td)
    print("world built", round(time.time() - t0), "s", flush=True)
    bad = m.run_suite(src, W, A, td, "base", e2e=True)
    print("UNMUTATED failing checks:", bad, round(time.time() - t0), "s", flush=True)
    names = {nm for nm, _ in m.CHECKS}
    print("n_checks", len(m.CHECKS), "n_mut", len(m.MUTATIONS))
    for i, (name, old, new, expect) in enumerate(m.MUTATIONS):
        if not any(s in name for s in only):
            continue
        assert src.count(old) == 1, ("anchor", name, src.count(old))
        unknown = {x for x in expect if x not in names and not x.startswith("e2e:")}
        assert not unknown, unknown
        got = m.run_suite(src.replace(old, new), W, A, td, f"m{i}", e2e=any(x.startswith("e2e:") for x in expect))
        print("MUT", name[:70], "RED" if got else "GREEN!!", "expected_subset=", expect <= set(got), "n_red=", len(got), got[:4], flush=True)
