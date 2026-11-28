"""Read the O13 pilot the way the PRE-REGISTRATION says to: BLOCK MEDIANS.

⛔ WHY NOT SINGLE ROWS. At batch 4 the per-step estimate swings enormously —
this pilot read o13_excess **+0.3614 at step 800 and −0.1624 at step 1500**. I
reported "the pilot is positive" off steps 100–300 and the trend then reversed.
Either reading, quoted from a single row, is an artefact of the batch. The
pre-registration commits to a **200-step median**, and this script is that rule
applied rather than restated.

⭐ THE PAIRED COMPARISON IS THE POINT. `o13_excess` alone cannot distinguish
"the term works" from "the term is degenerate": O11 also produced a positive
excess while degrading o5 by +18.7 %. The only separator is o5 on a MATCHED arm —
same corpus, same seed, same steps, differing ONLY in the o13 weight.

Prints, per 500-step block:
    o13_excess       median · > 0 means the predicted latent beats the exact
                     no-information floor of 1.0
    beats_passthrough median · > 0 means zhat beats the PRESENT latent z_t
    shuffled         median · MUST stay near 1.0 — if it falls with the loss,
                     the term is fitting something other than the pairing
    o5 (o13 arm)     vs o5 (control arm), and the RELATIVE change, which is the
                     degeneracy test
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

SPD = pathlib.Path(__file__).resolve().parent
BLOCK = 500


def rows(tag):
    f = SPD / f"o13{tag}" / "train_log.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "run_start" in line[:20]:
            continue
        r = json.loads(line)
        if "step" in r and "o5_loss" in r:
            out.append(r)
    return out


def blocks(rs, key):
    b = {}
    for r in rs:
        if key in r:
            b.setdefault(r["step"] // BLOCK, []).append(float(r[key]))
    return {k: st.median(v) for k, v in sorted(b.items()) if len(v) >= 2}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    P, C = rows("pilot"), rows("ctrl")
    print(f"\n  O13 PILOT — block medians ({BLOCK} steps), NON-PARITY 24 eps")
    print(f"  pilot rows {len(P)}  ·  control rows {len(C)}"
          f"{'  ⚠️ CONTROL NOT FINISHED' if len(C) < len(P) else ''}\n")
    if not P:
        print("  no pilot rows yet")
        return 0
    ex, bp, sh, o5p = (blocks(P, "o13_excess"), blocks(P, "o13_beats_passthrough"),
                       blocks(P, "o13_shuffled"), blocks(P, "o5_loss"))
    o5c = blocks(C, "o5_loss")
    print(f"  {'steps':<14}{'excess':>10}{'beats_pt':>11}{'shuffled':>11}"
          f"{'o5(o13)':>10}{'o5(ctrl)':>10}{'o5 rel':>9}")
    print("  " + "-" * 76)
    for k in sorted(ex):
        rel = ""
        if k in o5c and o5c[k] > 0:
            rel = f"{100.0 * (o5p[k] - o5c[k]) / o5c[k]:>+8.1f}%"
        cc = f"{o5c[k]:>10.4f}" if k in o5c else f"{'--':>10}"
        print(f"  {k * BLOCK:>6}-{(k + 1) * BLOCK:<7}{ex[k]:>+10.4f}"
              f"{bp.get(k, float('nan')):>+11.4f}{sh.get(k, float('nan')):>11.4f}"
              f"{o5p[k]:>10.4f}{cc}{rel:>9}")
    print()
    last = sorted(ex)[-1]
    e, b = ex[last], bp.get(last, 0.0)
    # ⛔ COMPARE ONLY AT A BLOCK BOTH ARMS REACHED. The first version indexed the
    # control at the PILOT's last block and died with KeyError — which was the
    # honest failure, because the silent alternative (comparing step 3000 of one
    # arm against step 700 of the other) would have produced a confident number
    # from mismatched training stages. A paired statistic must be paired.
    shared = sorted(set(o5p) & set(o5c))
    if not o5c:
        print("  ⚠️ NO VERDICT YET — the matched control has not run. `o13_excess`\n"
              "     alone cannot separate 'works' from 'degenerate': O11 showed a\n"
              "     POSITIVE excess while degrading o5 by +18.7 %.")
    elif not shared:
        print("  ⚠️ NO VERDICT — no block reached by BOTH arms yet.")
    elif shared[-1] < max(o5p):
        print(f"  ⚠️ NO VERDICT YET — the control has only reached block "
              f"{shared[-1] * BLOCK}-{(shared[-1] + 1) * BLOCK} of the pilot's "
              f"{max(o5p) * BLOCK}+.\n"
              f"     Early blocks are where two o5 curves differ MOST; the arm is\n"
              f"     judged at the END. Partial-control o5 rel at the shared "
              f"blocks:\n"
              f"     " + ", ".join(
                  f"{k * BLOCK}-{(k + 1) * BLOCK}: "
                  f"{100.0 * (o5p[k] - o5c[k]) / max(o5c[k], 1e-12):+.1f}%"
                  for k in shared))
    else:
        last = shared[-1]
        e, b = ex[last], bp.get(last, 0.0)
        d = 100.0 * (o5p[last] - o5c[last]) / max(o5c[last], 1e-12)
        if e > 0 and b > 0 and d < 10:
            v = ("TRAINING-SET POSITIVE with prediction intact — the gradient path "
                 "works. ⚠️ This is NOT generalisation; the pre-registered read is "
                 "HELD-OUT on the parity arm.")
        elif e > 0 and d >= 10:
            v = ("DEGENERATE SIGNATURE — excess up, o5 degraded. This is O11's "
                 "failure repeating.")
        elif e <= 0:
            v = ("NOT LEARNED — the predicted latent does not beat the exact 1.0 "
                 "floor even on the TRAINING set of 24 episodes. That is a stronger "
                 "negative than a held-out null and it points at the REFUTED branch.")
        else:
            v = "UNCLASSIFIED — state the numbers, do not round them to a claim."
        print(f"  => {v}")
    print(f"\n  ⚠️ NON-PARITY, 24 episodes, TRAINING SET. This pair answers 'can the\n"
          f"     term move, and does it cost prediction?' — NOT 'does it generalise?'\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
