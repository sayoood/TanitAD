"""⛔ THE MUTATION PROOF for the RL channel VALUE-flow guard.

An assertion that cannot fail proves nothing. Each mutation below REINTRODUCES one real
defect in the real file and REQUIRES a red; the two controls require a green, because a
check that always fails proves nothing either.

⚠️ Deliberately NOT the shape the sibling found in `test_rl_channel_guard.py`: no
expectation here is "whatever the code does" — every row states RED or GREEN up front.

Run: python mutate.py [mirror_stack_dir]   (off-Drive; CPU only; no GPU, no pod)
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

STACK = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else r"C:\Users\Admin\tanitad-chanval\stack")
ADAPTER = STACK / "tanitad" / "rl" / "refc_adapter.py"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
SUITES = ["tests/test_rl_channel_value_flow.py",
          "tests/test_rl_refc_adapter_robust.py",
          "tests/test_rl_forward_keys_cover_signature.py",
          "tests/test_rl_channel_guard.py"]

M1_OLD = '    obj, parts = cfg, dotted.split(".")'
M1_NEW = ('    return bool(getattr(cfg, dotted, False))   # MUTATION: the flat read\n'
          '    obj, parts = cfg, dotted.split(".")')

M2_OLD = '        predicates=("core.anchors.v0_conditioned", "core.sel_reach_clamp"),'
M2_NEW = '        unblock="a placeholder unblock long enough to pass the floor",'

M3_OLD = '        predicates=("core.graft_lan",),'
M3_NEW = '        unblock="a placeholder unblock long enough to pass the floor",'

M4_OLD = ('        channel="nav_known",\n'
          '        owner="E1 nav companion-bit seam",')
M4_NEW = ('        channel="nav_known",\n'
          '        owner="E1 nav companion-bit seam",\n'
          '        predicates=("core.nav_known_channel",),')

M5_OLD = (
    '        unblock="\u26d4 nothing should. This is a decision, not a gap: a per-arm '
    "'this run \"\n"
    '                "intends to supply nav\' declaration could carry it, but the '
    'guard would "\n'
    '                "then be asserting the caller\'s intent, not the checkpoint\'s '
    'training.",')
M5_NEW = '        unblock="tbd",'

M6_OLD = '        channel="withheld_speed",'
M6_NEW = '        channel="withheld_speed_TYPO",'

M7_OLD = '    if missing:'
M7_NEW = '    if missing and False:'

#: (id, description, expectation, old, new)
MUTATIONS = [
    ("M1", "the FLAT getattr is restored -- THE ORIGINAL DEFECT: a nested predicate "
           "reads False forever instead of raising", "RED", M1_OLD, M1_NEW),
    ("M2", "v0 loses its predicates -> back to unasserted", "RED", M2_OLD, M2_NEW),
    ("M3", "lan loses its predicate -> back to unasserted", "RED", M3_OLD, M3_NEW),
    ("M4", "nav_known is OVER-asserted -> would refuse the programme's own "
           "nav_cmd=None eval arm", "RED", M4_OLD, M4_NEW),
    ("M5", "an unasserted channel's unblock becomes the placeholder 'tbd'", "RED",
     M5_OLD, M5_NEW),
    ("M6", "withheld_speed drops out of the records -> a plumbed channel nobody "
           "ruled on", "RED", M6_OLD, M6_NEW),
    ("M7", "assert_conditioning stops raising -> the guard is inert", "RED",
     M7_OLD, M7_NEW),
]


def run() -> tuple[bool, str]:
    p = subprocess.run([PY, "-m", "pytest", *SUITES, "-q", "--no-header",
                        "--tb=no"],
                       cwd=STACK, capture_output=True, text=True,
                       env={"PYTHONPATH": str(STACK), "SYSTEMROOT": r"C:\Windows"})
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    return p.returncode == 0, (lines[-1] if lines else "?")


def main() -> int:
    original = ADAPTER.read_text(encoding="utf-8")
    rows = []

    ok, tail = run()
    rows.append(("--", "CONTROL, unmutated", "GREEN", "GREEN" if ok else "RED", tail))
    print(f"CONTROL unmutated: {'GREEN' if ok else 'RED'}  ({tail})")

    try:
        for mid, desc, want, old, new in MUTATIONS:
            n = original.count(old)
            if n != 1:
                rows.append((mid, desc, want, f"ANCHORx{n}", "mutation NOT applied"))
                print(f"{mid}: \u26d4 ANCHOR MISS ({n} matches) -- {desc}")
                continue
            ADAPTER.write_text(original.replace(old, new, 1), encoding="utf-8",
                               newline="")
            ok, tail = run()
            got = "GREEN" if ok else "RED"
            rows.append((mid, desc, want, got, tail))
            print(f"{mid}: want {want}, got {got}  "
                  f"{'OK' if got == want else 'MISMATCH'}  ({tail})")
            ADAPTER.write_text(original, encoding="utf-8", newline="")
    finally:
        ADAPTER.write_text(original, encoding="utf-8", newline="")

    ok, tail = run()
    rows.append(("--", "CONTROL, after restore", "GREEN",
                 "GREEN" if ok else "RED", tail))
    print(f"CONTROL restored: {'GREEN' if ok else 'RED'}  ({tail})")

    met = sum(1 for r in rows if r[2] == r[3])
    print(f"\n{met}/{len(rows)} expectations met")

    out = STACK.parent / "mutation_proof.txt"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("MUTATION PROOF -- RL channel VALUE flow guard "
                 "(refc_adapter.CHANNEL_REQUIREMENTS)\n")
        fh.write(f"stack: {STACK}\nsuites: {' '.join(SUITES)}\n")
        fh.write("CPU only, off-Drive mirror. No GPU, no pod.\n\n")
        fh.write(f"{'id':<5}{'want':<7}{'got':<9}description\n")
        fh.write("-" * 96 + "\n")
        for mid, desc, want, got, tail in rows:
            fh.write(f"{mid:<5}{want:<7}{got:<9}{desc}\n")
            fh.write(f"{'':<21}{tail}\n")
        fh.write(f"\n{met}/{len(rows)} expectations met\n")
    print(f"wrote {out}")
    return 0 if met == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
