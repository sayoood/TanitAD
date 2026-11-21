# -*- coding: utf-8 -*-
"""THE GUARD THAT WOULD HAVE SAVED A GPU HOUR: refuse `--jerk-seam a0` when
`W_JERK == 0`, because the repaired term is then multiplied by zero and the arm
is INERT BY CONSTRUCTION. Anchored, idempotent."""
import io
import sys
import time

PATH = sys.argv[1]


def rd(p, tries=30):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit("INCONCLUSIVE read")


def wr(p, t, tries=30):
    for _ in range(tries):
        try:
            with io.open(p, "w", encoding="utf-8", newline="") as f:
                f.write(t)
            if len(rd(p)) == len(t):
                return
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit("INCONCLUSIVE write")


t = rd(PATH)
if "INERT BY CONSTRUCTION" in t:
    print("ALREADY PATCHED")
    raise SystemExit(0)
EOL = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")
n0 = len(t)

OLD = '''    _p(f"[cost] metric={cost_metric} weights={used_w} "
       f"({'CLI override' if cost_weights else 'shipped module constants'}); "
       f"COST_METRICS={_R.COST_METRICS} from {_R.__file__}")
'''
NEW = u'''    _p(f"[cost] metric={cost_metric} weights={used_w} "
       f"({'CLI override' if cost_weights else 'shipped module constants'}); "
       f"COST_METRICS={_R.COST_METRICS} from {_R.__file__}")

    # ⛔⛔ THE JERK SEAM IS MULTIPLIED BY `W_JERK`. REFUSE AN ARM THAT IS
    # INERT BY CONSTRUCTION (D-REFAV1-LON-SEAM-INERT, 2026-09-05).
    # MEASURED, and it cost a GPU hour before this guard existed: `lonseam` was
    # launched as `--jerk-seam a0` against the `wk15` baseline triple
    # `(0.0, 15.11245, 64.29715042415070)`. The cost line is
    # `c = c + w_jerk * jerk.pow(2).mean(-1)`, so with `w_jerk = 0.0` REPAIRING
    # `jerk` cannot change `c` BY A SINGLE BIT. The arm ran to completion, the
    # reached-it guard passed (the flag DID reach `plan()`; the record carries
    # `jerk_seam: "a0"`), and it produced `+0.0000 [+0.0000, +0.0000]` on all
    # ten family metrics with emitted controls bit-identical to `wk15` --
    # a null that is pure arithmetic and says NOTHING about the seam.
    # ⚠ A null arm whose lever is multiplied by zero is not a null about the
    # lever; it is a null about a term that was switched off. Same family as
    # M23 (4) ("a NULL arm with a live optimiser is not a null") and as counting
    # on the wrong predicate: the experiment was uninformative BEFORE it ran,
    # and the only thing that catches that is a refusal at flag-parse time.
    # ⇒ The informative pair is `w_jerk > 0` with the seam OFF vs the SAME
    # `w_jerk` with the seam ON -- two arms, one variable between them.
    if getattr(a, "jerk_seam", "off") != "off" and float(used_w["W_JERK"]) == 0.0:
        raise SystemExit(
            "[refav1_arm] ⛔ --jerk-seam %s with W_JERK = 0.0 is INERT BY "
            "CONSTRUCTION: the cost adds `w_jerk * mean(jerk^2)`, so repairing "
            "`jerk` while `w_jerk` is zero cannot change the objective by a "
            "single bit, and the arm would bank a meaningless +0.0000. Pass a "
            "non-zero W_JERK in --cost-weights (and compare against a baseline "
            "carrying the SAME W_JERK with --jerk-seam off, so the seam is the "
            "one variable). Refusing before the rollout rather than after it."
            % getattr(a, "jerk_seam", "off"))
'''
c = t.count(OLD)
if c != 1:
    raise SystemExit("ANCHOR occurs %d (need 1) -- ABORT" % c)
t = t.replace(OLD, NEW, 1)
if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK %s  %d -> %d chars" % (PATH, n0, len(t)))
