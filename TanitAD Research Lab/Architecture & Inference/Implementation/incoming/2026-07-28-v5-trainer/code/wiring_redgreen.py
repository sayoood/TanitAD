"""RED/GREEN — prove each new guard FAILS when its one wiring line is removed.

⛔ A GUARD THAT CANNOT FAIL IS WORSE THAN NONE (class C13). A green test suite
proves the tests pass; it does not prove they would notice the defect. This probe
removes each guard in turn, re-runs the test that owns it, and records that the
test goes RED — then restores the file and records that it goes GREEN again.

Every mutation is a single, surgical string replacement in the deployed source.
The originals are saved BEFORE any edit and restored in a ``finally``; the run
ends by re-asserting the sha256 of every touched file, so a crash cannot leave a
mutated trainer on disk unnoticed.

⛔ No pod is contacted. Everything runs against the synthetic fixtures in
``stack/tests/test_v5_trainer_v2_val.py``.

Run:  python wiring_redgreen.py --out ../raw/wiring_redgreen_<date>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
STACK = REPO / "stack"
V4 = STACK / "scripts" / "train_flagship_v4.py"
PARITY = STACK / "tanitad" / "data" / "parity.py"
PY = sys.executable
TESTS = "tests/test_v5_trainer_v2_val.py"

#: (name, file, old, new, the test that MUST go red, why it matters)
MUTATIONS = [
    ("gate probes the TRAIN half instead of the held-out one", V4,
     "        val_eps = build_v2_providers(a.v2_val_cache, lru_size=a.v2_lru)",
     "        val_eps = build_v2_providers(a.v2_train_cache, lru_size=a.v2_lru)",
     "test_the_gate_probes_the_VAL_providers_not_the_train_ones",
     "a val loop fed the training clips reports health forever — the exact "
     "shape of the defect being fixed, and nothing downstream would notice"),

    ("the v2 parity guard is skipped", V4,
     "    provenance = (_assert_parity_v2(a.v2_train_cache, a.v2_val_cache,\n"
     "                                    require=a.require_parity) if use_v2",
     "    provenance = ({'train_parity': {}, 'val_parity': {}} if use_v2",
     "test_the_parity_guard_runs_BEFORE_the_v2_loader",
     "an unregistered v2 cache would train under --require-parity"),

    # ⚠️ BOTH calls must go: the first draft removed only the TRAIN binding and
    # the test stayed GREEN, because the VAL binding refused the same run. That
    # is a real property worth recording (the guard fires from either half) and
    # it would have made this row a probe that proves nothing.
    ("the geometry binding is removed (BOTH halves)", V4,
     "        provenance[\"geometry_binding\"] = parity.assert_v2_geometry_matches(\n"
     "            provenance[\"train_parity\"], frame, label=\"--v2-train-cache\",\n"
     "            providers=train_eps)\n"
     "        provenance[\"geometry_binding_val\"] = parity.assert_v2_geometry_matches(\n"
     "            provenance[\"val_parity\"], frame, label=\"--v2-val-cache\",\n"
     "            providers=val_eps)",
     "        provenance[\"geometry_binding\"] = {\"checked_shape\": False}\n"
     "        provenance[\"geometry_binding_val\"] = {\"checked_shape\": False}",
     "test_a_wide_cache_read_with_DEFAULT_flags_is_refused",
     "a 256x640 cache read with the default 256x256 flags trains happily and "
     "voids every number — membership never sees pixels"),

    ("the train/val leak check is removed", V4,
     "    leak = parity.assert_v2_splits_disjoint(train_dirs, val_dirs,\n"
     "                                            label=\"v2 train/val\")",
     "    leak = {'disjoint': True, 'train_clips': 0, 'val_clips': 0, "
     "'overlap': 0, 'label': 'v2 train/val'}",
     "test_a_train_val_LEAK_is_refused",
     "a leaked val clip makes the early-stop unable to fire, which is worse "
     "than no early-stop because it is believed"),

    ("the missing-manifest hint is removed", PARITY,
     "                    *_missing_entry_lines(dirs, None, manifest_path),\n",
     "",
     "test_an_unregistered_v2_cache_NAMES_the_missing_manifest_entry",
     "runbook step 3 (git add the manifest) is the step that gets forgotten, "
     "and the refusal named nothing"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(node: str) -> dict:
    r = subprocess.run(
        [PY, "-m", "pytest", "-q", f"{TESTS}::{node}", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=str(STACK), capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    tail = (r.stdout or "").strip().splitlines()
    return {"returncode": r.returncode,
            "passed": r.returncode == 0,
            "summary": tail[-1] if tail else "",
            "reason": next((ln.strip() for ln in reversed(tail)
                            if ln.strip().startswith(("E ", "assert"))), None)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    originals = {p: p.read_bytes() for p in {V4, PARITY}}
    before = {str(p.relative_to(REPO)).replace("\\", "/"): _sha(p)
              for p in originals}
    rows = []
    try:
        for name, path, old, new, node, why in MUTATIONS:
            src = path.read_text(encoding="utf-8")
            if old not in src:
                rows.append({"mutation": name, "APPLIED": False,
                             "error": "anchor string not found — the source "
                                      "moved; this probe must be updated"})
                continue
            green = _run(node)
            path.write_text(src.replace(old, new, 1), encoding="utf-8")
            red = _run(node)
            path.write_bytes(originals[path])
            restored = _run(node)
            rows.append({
                "mutation": name, "APPLIED": True,
                "file": str(path.relative_to(REPO)).replace("\\", "/"),
                "test": node, "why_it_matters": why,
                "green_before": green, "RED_when_removed": red,
                "green_after_restore": restored,
                "PASS": bool(green["passed"] and not red["passed"]
                             and restored["passed"]),
            })
    finally:
        for p, b in originals.items():
            p.write_bytes(b)

    after = {str(p.relative_to(REPO)).replace("\\", "/"): _sha(p)
             for p in originals}
    rec = {
        "what": "each guard removed in turn; the test that owns it must go RED",
        "host": "dev box (no pod contacted)",
        "cases": rows,
        "files_restored_bit_exact": before == after,
        "sha256_before": before, "sha256_after": after,
        "ALL_PASS": bool(all(r.get("PASS") for r in rows) and before == after),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0 if rec["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
