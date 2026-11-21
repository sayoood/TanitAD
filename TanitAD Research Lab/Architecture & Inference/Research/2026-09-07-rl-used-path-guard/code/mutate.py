"""Mutation proof for the refcv3_adapter conditioning contract.

⛔ EVERY MUTATION IS A COMPLETE, SELF-CONSISTENT EDIT. A sibling's first attempt
went RED for a `NameError` from a dangling reference, which proves nothing about
the property under test — a mutation that breaks the module is not evidence that
a test is load-bearing. So each patch below leaves the module importable and every
name defined, and the harness REFUSES a mutation whose failure set includes a
collection error.

Run:  python mutate.py <mirror_stack_dir>
"""
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

STACK = Path(sys.argv[1])
ADAPTER = STACK / "tanitad" / "rl" / "refcv3_adapter.py"
TESTFILE = "tests/test_rl_refcv3_used_path_guard.py"
PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ENV = {
    "PYTHONPATH": f"{STACK};{STACK.parent / 'taniteval'}",
    "PYTHONIOENCODING": "utf-8",
    "OMP_NUM_THREADS": "6",
    "SYSTEMROOT": r"C:\Windows",
    "PATH": r"C:\Windows\System32",
}

# ---------------------------------------------------------------------------
# The mutations. (name, what property it removes, find, replace)
# ---------------------------------------------------------------------------
MUTATIONS = [
    (
        "M1_once_only_latch",
        "the PER-BATCH property: assert on batch 1 only, exactly the optimisation "
        "refc_adapter carried before 4fc462c",
        """        self.batches += 1
        missing = _missing_channels(self._required, batch)""",
        """        self.batches += 1
        if self.batches > 1:            # MUTATION M1: the once-only latch
            return self._req
        missing = _missing_channels(self._required, batch)""",
    ),
    (
        "M2_check_after_sampling",
        "the ORDERING property: it still refuses on batch 2, but only AFTER the "
        "model has already been run on the mis-conditioned batch",
        """        if contract is not None:
            contract.check(batch)
        frames = batch["frames"]
        kw = _forward_kwargs(batch, cfg_in)
        out = model(frames, **kw)""",
        """        frames = batch["frames"]
        kw = _forward_kwargs(batch, cfg_in)
        out = model(frames, **kw)
        if contract is not None:        # MUTATION M2: refuse AFTER sampling
            contract.check(batch)""",
    ),
    (
        "M3_over_assert_v0",
        "the OVER-ASSERTION guard: require v0 unconditionally, refusing builds "
        "that were never trained with it",
        """    return {c: (_resolve_any(cfg, by_channel[c].predicates)
                if by_channel[c].predicates else False)
            for c in channels}""",
        """    return {c: (True if c == "v0" else          # MUTATION M3
                (_resolve_any(cfg, by_channel[c].predicates)
                 if by_channel[c].predicates else False))
            for c in channels}""",
    ),
    (
        "M4_inherit_refc_adapters_predicates",
        "the VERDICT itself: root v0's predicates at `core.` the way refc_adapter "
        "does — i.e. copy a sibling's contract onto a different model family",
        """            predicates=("anchors.v0_conditioned", "core.anchors.v0_conditioned",
                        "sel_reach_clamp", "core.sel_reach_clamp"),""",
        """            predicates=("core.anchors.v0_conditioned",   # MUTATION M4
                        "core.sel_reach_clamp"),""",
    ),
    (
        "M5_drop_the_plumbing_check",
        "the WIRING half: a REQUIRED channel this adapter never forwards passes "
        "silently, so the guard is green while the channel arrives as None",
        """            unplumbed = [k for k in required if k not in self._plumbed]""",
        """            unplumbed = []                  # MUTATION M5""",
    ),
    (
        "M6_ignore_undeclared_channels",
        "the DRIFT guard: a forward that grows a channel is silently narrowed "
        "out of scope instead of refusing",
        """    undeclared = [c for c in channels if c not in declared]""",
        """    undeclared = []                     # MUTATION M6""",
    ),
]


def run() -> tuple[int, int, list[str], bool, list[str]]:
    r = subprocess.run(
        [PY, "-m", "pytest", TESTFILE, "-q", "-p", "no:cacheprovider",
         "--tb=line", "-rf"],
        cwd=STACK, env=ENV, capture_output=True, text=True)
    out = r.stdout + r.stderr
    broke = ("error" in out.lower() and "errors" in out.lower()
             and re.search(r"\d+ errors?\b", out) is not None)
    m = re.search(r"(?:(\d+) failed,?\s*)?(\d+) passed", out)
    failed = int(m.group(1) or 0) if m else -1
    passed = int(m.group(2)) if m else -1
    names = sorted(set(re.findall(r"FAILED \S+::(\S+)", out)))
    # ⭐ WHY it went red, not only THAT it did — two mutations can kill the same
    # test for different reasons, and if they do, both halves are load-bearing.
    why = [ln.strip()[:230] for ln in out.splitlines()
           if re.match(r"^[A-Za-z]:.*\.py:\d+: ", ln.strip())]
    return failed, passed, names, broke, why


def main() -> int:
    src = io.open(ADAPTER, encoding="utf-8").read()
    sha = hashlib.sha256(src.encode()).hexdigest()[:12]
    print(f"baseline adapter sha256[:12] = {sha}\n")

    f, p, names, broke, _why = run()
    print(f"BASELINE            failed={f} passed={p} broke={broke}")
    assert f == 0 and not broke, "baseline must be fully GREEN before mutating"
    baseline_passed = p
    results = [{"mutation": "BASELINE", "removes": "-", "failed": f,
                "passed": p, "red_tests": [], "verdict": "GREEN"}]

    for name, removes, find, repl in MUTATIONS:
        assert src.count(find) == 1, f"{name}: anchor not unique ({src.count(find)})"
        io.open(ADAPTER, "w", encoding="utf-8", newline="\n").write(
            src.replace(find, repl, 1))
        try:
            f, p, red, broke, why = run()
        finally:
            io.open(ADAPTER, "w", encoding="utf-8", newline="\n").write(src)
        assert hashlib.sha256(
            io.open(ADAPTER, encoding="utf-8").read().encode()).hexdigest()[:12] == sha, \
            f"{name}: adapter not restored"

        if broke:
            verdict = "⛔ INVALID — the module did not import; proves nothing"
        elif f == 0:
            verdict = "⛔ SURVIVED — no test is load-bearing for this property"
        elif f + p != baseline_passed:
            verdict = f"⚠️ RED but test count moved ({f + p} vs {baseline_passed})"
        else:
            verdict = "✅ RED"
        print(f"{name:38s} failed={f} passed={p}  {verdict}")
        for t in red:
            print(f"      RED: {t}")
        for w in why:
            print(f"      WHY: {w}")
        results.append({"mutation": name, "removes": removes, "failed": f,
                        "passed": p, "red_tests": red, "why": why,
                        "verdict": verdict})

    # final: the file is byte-identical to where we started
    f, p, _, _, _ = run()
    print(f"\nRESTORED            failed={f} passed={p}")
    assert f == 0, "the suite must be GREEN again after every mutation is reverted"

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("mutation_result.json")
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
