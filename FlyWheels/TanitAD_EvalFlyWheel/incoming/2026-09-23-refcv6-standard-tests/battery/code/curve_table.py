"""Across checkpoints: the headline cells of every banked battery tag, side by side (copied from JSONs).
usage: python curve_table.py <raw_dir> tag1 [tag2 ...]      e.g. raw step5000 step30000 final"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CELLS = [("S2", "os_minus_ha0ext", "os − ha0_ext (echo)"), ("S2", "os_minus_ha", "os − ha (hold)"),
         ("S2", "os_minus_ha0__CV", "os − ha0 (CV)"), ("S2", "os_minus_refcv4b", "os − refcv4b"),
         ("S2", "os_minus_refcv5v2_s0", "os − refcv5-v2 s0"),
         ("S2", "navzero_minus_os", "nav withheld − os"), ("S2", "vmaxzero_minus_os", "max-speed withheld − os"),
         ("S6", "os_minus_ha0ext", "S6: os − ha0_ext"), ("S6", "os_minus_refcv4b", "S6: os − refcv4b")]


def cell(root, tag, seed, surface, name):
    f = root / tag / (f"cross_paired_s{seed}.json" if surface == "S2" else f"cross_paired_s6_s{seed}.json")
    if not f.exists():
        return "—"
    c = ((json.load(open(f, encoding="utf-8"))["pairs"].get(name) or {}).get("families") or {})
    x = (c.get("ADE") or {}).get("ade_m")
    if not x:
        return "—"
    return f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' **sep**' if x['separated'] else ''}"


def _fmt(x):
    if not x or "delta" not in x:
        return "—"
    return f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' **sep**' if x['separated'] else ''}"


def lever(root, tag, seed, which):
    d = root / tag / "levers"
    if which == "L3":
        f = d / "eps0.json"
        return _fmt(((json.load(open(f, encoding="utf-8")).get("cells") or {}).get("os_eps0_minus_ha0_ext"))
                    if f.exists() else None)
    f = d / "levers.json"
    if not f.exists():
        return "—"
    o = json.load(open(f, encoding="utf-8"))
    if which == "L1":
        return _fmt((o.get("L1_seed_average") or {}).get("os_avg_minus_ha0_ext"))
    key = "L2_causal_hold_blend" if which == "L2" else "L2e_echo_blend_DIAGNOSTIC"
    return _fmt(((o.get(key) or {}).get(f"seed{seed}") or {}).get("blend_minus_ha0_ext"))


def main():
    root = Path(sys.argv[1])
    tags = sys.argv[2:]
    steps = {}
    for t in tags:
        f = root / t / "battery_summary.json"
        steps[t] = json.load(open(f, encoding="utf-8")).get("step") if f.exists() else None
    print("### refcv6 across checkpoints — a TABLE, not a fit\n")
    print("⛔ **Two experiments (SPEC A5).** Checkpoints at step ≤ 34,500 are pre-switch: *F3 detach-only, F4 on "
          "the last layer only; tactical labels ~0.37 s early*. A later checkpoint (the FINAL) is the post-switch "
          "*hybrid: F3 cascade loss + true label clock from step 34,500* (resumed on 82c2331). A difference across "
          "the switch MIXES training time with the fix and is never attributed to the fix alone; no learning-curve "
          "fit spans step 34,500. TACTICAL is not in this table: each tag's `TACTICAL_CLOCKS.md` carries it under "
          "both label clocks.\n")
    print("Steps: " + ", ".join(f"{t} = {steps[t]}" + (" (post-switch hybrid)" if (steps[t] or 0) > 34500 else "")
                                for t in tags) + "\n")
    for seed in (0, 1):
        print(f"\n#### ADE paired cells (m, b − a, T1), inference seed {seed}\n")
        print("| cell | " + " | ".join(tags) + " |")
        print("|---|" + "---|" * len(tags))
        for surf, nm, lab in CELLS:
            print(f"| {lab} | " + " | ".join(cell(root, t, seed, surf, nm) for t in tags) + " |")
        # SPEC A4 levers (reported, no bar; each is a DIFFERENT planner from the registered arm)
        print(f"| A4 L2 blend(os, ha) − ha0_ext | " + " | ".join(lever(root, t, seed, "L2") for t in tags) + " |")
        print(f"| A4 L2e blend(os, echo) − ha0_ext (diag.) | "
              + " | ".join(lever(root, t, seed, "L2e") for t in tags) + " |")
        if seed == 0:
            print(f"| A4 L1 seed-average − ha0_ext | " + " | ".join(lever(root, t, 0, "L1") for t in tags) + " |")
            print(f"| A4 L3 eps0 − ha0_ext | " + " | ".join(lever(root, t, 0, "L3") for t in tags) + " |")
    print("\n#### Levels and gates\n")
    print("| tag | step | G0-A2 | os ADE 0–2 s [CI] | os ADE 1–6 s | bars (BAR-R6-1..5) | T-FLIP | OBEDIENCE |")
    print("|---|---|---|---|---|---|---|---|")
    for t in tags:
        d = root / t
        if not (d / "battery_summary.json").exists():
            print(f"| {t} | — | — | — | — | — | — | — |")
            continue
        s = json.load(open(d / "battery_summary.json", encoding="utf-8"))
        g0 = (s.get("stages") or {}).get("g0") or {}
        a2 = (json.load(open(d / "analysis_s0.json", encoding="utf-8"))
              if (d / "analysis_s0.json").exists() else None)
        a6 = (json.load(open(d / "analysis_s6_s0.json", encoding="utf-8"))
              if (d / "analysis_s6_s0.json").exists() else None)
        ade = (a2["arms"]["os"]["intervals"]["metrics"]["ade_dense_m"] if a2 else None)
        ade6 = (a6["arms"]["os"]["intervals"]["metrics"]["ade_dense_m"]["mean"] if a6 else None)
        acc = (json.load(open(d / "acceptance_s0.json", encoding="utf-8"))
               if (d / "acceptance_s0.json").exists() else {})
        bars = "/".join((b.get("verdict") or "?")[:4] for b in s.get("bars", []))
        print(f"| {t} | {s.get('step')} | {g0.get('G0_A2')} | "
              + (f"{ade['mean']:.4f} [{ade['lo']:.4f}, {ade['hi']:.4f}]" if ade else "—")
              + f" | {ade6 if ade6 is None else f'{ade6:.4f}'} | {bars} | "
              f"{(acc.get('tflip') or {}).get('verdict')} | {(acc.get('obedience') or {}).get('verdict')} |")


if __name__ == "__main__":
    main()
