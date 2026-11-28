"""Render §5.3 / §6 of PSEUDO_SIMULATION.md straight from the artifact JSON.

Exists so the report's tables are **machine-derived, not hand-typed**. This
program has retracted numbers for transcription reasons more than once; a table
that is generated from the artifact cannot drift from it.

Reads ``artifacts/pseudosim_v4_30k.json`` and prints markdown. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ART = _HERE.parent / "artifacts"


def _ci(d, nd=4):
    if not isinstance(d, dict) or d.get("mean") is None:
        return "—"
    return f"**{d['mean']:.{nd}f}** [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def _delta(d, nd=4):
    if not isinstance(d, dict) or d.get("delta") is None:
        return "— (not estimable)"
    sep = "⭐ **SEP**" if d.get("separated") else "n.s."
    return f"{d['delta']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}] {sep}"


def main(path=None):
    p = Path(path) if path else (_ART / "pseudosim_v4_30k.json")
    r = json.loads(p.read_text(encoding="utf-8"))
    arms = list(r["arms"])

    print("### 5.3 — discriminative range, MEASURED (per arm)\n")
    print("| component | arm | n | n NaN | min | max | mean | IQR | "
          "ceiling ≥0.999 | floor ≤0.001 | between-arm spread | **admissible** |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for name in ("ego_progress", "recovery", "comfort", "no_collision", "ttc"):
        for a in arms:
            n = r["arms"][a]["component_discriminative_range"].get(name, {})
            if n.get("n") is None:
                print(f"| `{name}` | (all) | — | — | — | — | — | — | — | — | — | "
                      f"⛔ {n.get('reason', 'n/a')} |")
                break
            ok = "✅" if n.get("admissible") else f"⛔ {n.get('reason', '')}"
            print(f"| `{name}` | `{a}` | {n['n']} | {n['n_nan']} | "
                  f"{n['min']:.4f} | {n['max']:.4f} | {n['mean']:.4f} | "
                  f"{n['iqr']:.4f} | {n['ceiling_frac_ge_0p999']:.4f} | "
                  f"{n['floor_frac_le_0p001']:.4f} | "
                  f"{n.get('between_arm_spread', '—')} | {ok} |")

    print("\n### 6 — arm scores\n")
    print("| arm | n evals | n eps | `ego_progress` | `recovery` | `comfort` | "
          "**PSS_recovery_progress** |")
    print("|---|---:|---:|---|---|---|---|")
    for a in arms:
        n = r["arms"][a]
        c = n.get("composite", {})
        pss = ("⛔ REFUSED" if "REFUSED_TO_EMIT" in c else _ci(c.get("ci")))
        print(f"| `{a}` | {n['n_evaluations']} | {n['n_episodes']} | "
              f"{_ci(n['components']['ego_progress']['ci'])} | "
              f"{_ci(n['components']['recovery']['ci'])} | "
              f"{_ci(n['components']['comfort']['ci'])} | {pss} |")

    print("\n**Paired (episode-cluster bootstrap, B=2000, identical rows):**\n")
    print("| contrast | `ego_progress` | `recovery` | **PSS** |")
    print("|---|---|---|---|")
    for k, v in r.get("paired", {}).items():
        print(f"| `{k.replace('__minus__', ' − ')}` | "
              f"{_delta(v.get('ego_progress'))} | {_delta(v.get('recovery'))} | "
              f"{_delta(v.get('PSS_recovery_progress'))} |")

    iv = r.get("INSTRUMENT_VALIDITY", {})
    print(f"\n**INSTRUMENT VALIDITY (sighted − blind): "
          f"{_delta(iv.get('sighted_minus_blind_PSS'))} → separated="
          f"{iv.get('separated')}**")
    print(f"\nrecovery defined fraction: {r.get('_recovery_defined_fraction')}")
    print(f"ego_progress mean: {r.get('_ego_progress_mean')}")
    ep = r.get("envelope_proof", {})
    print(f"\nenvelope: steps_any={ep.get('EXTRAPOLATION_frac_steps_any')} "
          f"windows={ep.get('EXTRAPOLATION_frac_windows_any_step_out_of_envelope')} "
          f"traffic_mode={r.get('traffic_mode')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
