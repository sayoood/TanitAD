"""The one-variable check, run on the ACTUAL launch commands (SPEC §1 / skill §1).

⛔ "Diff the actual launch commands, not the intent." MEASURED 2026-08-22: a row-bank
arm changed `n` AND silently multiplied the effective lambda, and two sweeps were
invalidated, because the panel was audited against what it MEANT to vary.

Reads each arm's `config.json["argv"]` — what the trainer actually received — turns it
into a flag->value map, and prints the symmetric difference against A0. Also asserts the
HELD-CONSTANT set is byte-identical across arms, which is the half that actually catches
drift (a lever nobody named is invisible in a diff of the levers you expected).

usage: python onevar_check.py [arms_dir]
"""
import json
import sys
from pathlib import Path

ARMS = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\run_wbank\arms")
A0 = "A0_fixed"

# the lever each arm is ALLOWED to move, committed in SPEC.md before the panel ran
EXPECTED = {
    "A1_pred":   {"--withheld-bank", "--withheld-bank-warmup"},
    "A2_random": {"--withheld-bank", "--withheld-bank-warmup"},
    "A3_drop25": {"--ego-dropout"},
    "A4_none":   {"--withheld-bank"},
    "A5_regress": {"--ablate-frames"},
}
# flags whose VALUE must be identical everywhere; the rig itself
HELD = ["--arm", "--size", "--data-root", "--episodes", "--steps", "--batch", "--lr",
        "--warmup", "--seed", "--log-every", "--anchors", "--n-anchors",
        "--anchor-control-units", "--sel-accel-max", "--device"]


def as_map(argv):
    """flag -> value ('' for a bare store_true), preserving order."""
    m, i = {}, 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            m[tok] = argv[i + 1]
            i += 2
        else:
            m[tok] = ""
            i += 1
    return m


def load(name):
    p = ARMS / name / "config.json"
    if not p.exists():
        return None
    return as_map(json.load(open(p, encoding="utf-8"))["argv"])


def main():
    base = load(A0)
    if base is None:
        raise SystemExit(f"no {A0}/config.json — nothing to diff against")
    names = [d.name for d in sorted(ARMS.iterdir()) if d.is_dir()]
    print(f"# one-variable check vs {A0} (argv as the trainer received it)\n")
    verdicts = {}
    for n in names:
        if n == A0:
            continue
        m = load(n)
        if m is None:
            print(f"{n}: NO CONFIG (not run)")
            continue
        moved = set()
        for k in set(base) | set(m):
            if base.get(k) != m.get(k):
                moved.add(k)
        exp = EXPECTED.get(n, set())
        extra = moved - exp
        missing = exp - moved
        ok = not extra and not missing
        verdicts[n] = ok
        print(f"{n}: {'PASS' if ok else '⛔ FAIL'} — moved {sorted(moved)}")
        for k in sorted(moved):
            print(f"    {k}: {base.get(k, '<absent>')!r} -> {m.get(k, '<absent>')!r}")
        if extra:
            print(f"    ⛔ UNEXPECTED levers moved: {sorted(extra)}")
        if missing:
            print(f"    ⛔ expected lever did NOT move: {sorted(missing)}")
    print("\n# held-constant set (must be identical on every arm)")
    bad_held = []
    for k in HELD:
        vals = {n: (load(n) or {}).get(k, "<absent>") for n in names if load(n)}
        uniq = set(vals.values())
        flag = "ok" if len(uniq) == 1 else "⛔ DRIFT"
        if len(uniq) != 1:
            bad_held.append(k)
        print(f"  {k:26s} {flag}  {sorted(uniq) if len(uniq) != 1 else next(iter(uniq))!r}")
    # A1 vs A2 must differ in EXACTLY the speed source
    a1, a2 = load("A1_pred"), load("A2_random")
    if a1 and a2:
        d = {k for k in set(a1) | set(a2) if a1.get(k) != a2.get(k)}
        print(f"\n# A1 vs A2 (must be ONLY --withheld-bank): moved {sorted(d)} -> "
              f"{'PASS' if d == {'--withheld-bank'} else '⛔ FAIL'}")
        verdicts["A1_vs_A2"] = (d == {"--withheld-bank"})
    overall = all(verdicts.values()) and not bad_held
    print(f"\nONEVAR_VERDICT={'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
