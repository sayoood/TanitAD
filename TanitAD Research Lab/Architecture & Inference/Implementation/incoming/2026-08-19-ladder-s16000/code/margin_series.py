"""Margin-vs-C-V0 series across v6 S-W checkpoints.

⛔ THE SIGN CONVENTION, pinned from LADDER_3SEED.md:237 —
   K1B is NEGATIVE-IS-BETTER, so `margin = arm_K1B - cv0_K1B` and
   **POSITIVE MEANS THE ONE-NUMBER EGO-SPEED SCALAR WINS.**
Reading K1B itself as the margin inverts the conclusion; that is the mistake
this file exists to make impossible to repeat.

⚠️ REPRODUCTION GATE FIRST. Before any new checkpoint is reported, the four
margins banked in LADDER_3SEED.md §6a must be reproduced from the JSON on disk.
An analyser that cannot re-derive the known answer may not be trusted with the
unknown one.
"""
import json, sys
from pathlib import Path
from statistics import mean

DIRS = [Path(a) for a in sys.argv[1:]]
D = DIRS[0]          # the banked dir holds the C-V0 control


def find(name: str) -> Path:
    """⚠️ New checkpoints are fitted into their OWN incoming dir, never written
    into the banked one — a re-run must not overwrite the artifact its own
    reproduction gate is checked against."""
    for d in DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)
SEEDS = ("0", "1", "2")
BANKED = {"s09000": +0.262, "s09250": +0.243, "s10000": +0.217, "s11250": +0.211}


def k1b(path: Path, target: str):
    d = json.loads(path.read_text(encoding="utf-8"))
    t = d["targets"][target]
    return [t["per_seed"][s]["k1_guard"]["K1B_delta"] for s in SEEDS], d.get("step")


def margins(arm: str, target: str):
    a, step = k1b(find(f"ll3_{arm}.json"), target)
    c, _ = k1b(find("ll3_proxyv0.json"), target)
    per = [x - y for x, y in zip(a, c)]
    return per, mean(per), a, c, step


print("=" * 78)
print("REPRODUCTION GATE — n_agents_all, route A, vs LADDER_3SEED.md §6a")
print("=" * 78)
ok = True
for arm, want in BANKED.items():
    try:
        p = find(f"ll3_{arm}.json")
    except FileNotFoundError:
        p = D / f"ll3_{arm}.json"
    if not p.exists():
        print(f"  {arm:10s} MISSING {p.name}"); ok = False; continue
    per, m, a, c, step = margins(arm, "n_agents_all")
    hit = abs(m - want) <= 0.001
    ok &= hit
    print(f"  {arm:10s} margin {m:+.4f}  banked {want:+.3f}  "
          f"{'OK' if hit else 'MISMATCH'}   per-seed {[round(x,3) for x in per]}")
print(f"\nGATE: {'PASS' if ok else 'FAIL — do not report new points'}\n")
if not ok:
    sys.exit(1)

TARGETS = ["n_agents_all", "n_agents_grid", "nearest_any", "lead_gap",
           "lead_present", "ego_v0"]
arms = sorted({p.stem[4:] for d in DIRS for p in d.glob("ll3_s*.json")},
              key=lambda a: int(a[1:]) if a[1:].isdigit() else 0)
print("=" * 78)
print("MARGIN SERIES  (positive = the 1-feature ego-speed scalar WINS)")
print("=" * 78)
hdr = f"{'target':<15}" + "".join(f"{a:>11}" for a in arms)
print(hdr); print("-" * len(hdr))
for tg in TARGETS:
    row = f"{tg:<15}"
    for a in arms:
        try:
            _, m, *_ = margins(a, tg)
            row += f"{m:>+11.4f}"
        except (KeyError, FileNotFoundError):
            row += f"{'—':>11}"
    print(row)
print()
print("per-seed sign stability on n_agents_all:")
for a in arms:
    try:
        per, m, *_ = margins(a, "n_agents_all")
        sg = "".join("+" if x > 0 else "-" for x in per)
        print(f"  {a:<10} mean {m:+.4f}  seeds {[round(x,3) for x in per]}  signs {sg}"
              f"{'  ⚠️ FLIPS' if len(set(sg)) > 1 else ''}")
    except (KeyError, FileNotFoundError):
        pass
