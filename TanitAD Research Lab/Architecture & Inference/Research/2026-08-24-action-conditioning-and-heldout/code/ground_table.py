"""Assemble the pixel-groundedness table across arms.

⛔ THE VERDICT USES `t > 2`, NOT `|t| > 2`. My first version tested the ABSOLUTE t
and so reported `splitfrz10k` (t = −2.17) as *"a real residual — that arm is the
lead"*. A NEGATIVE t means the action makes the residual prediction WORSE; it is a
negative outlier, not a lead. That was the SIXTH auto-verdict in one night to fire
on the wrong quantity, which is why every verdict here is a COMPUTED FIELD printed
beside its inputs, never a conclusion standing on its own.

⚠️ AND A NOTE ON HOW THIS FILE WAS WRITTEN. Three attempts tonight to generate a
script through a bash heredoc were destroyed by the same trap: a quoted heredoc
consumes one backslash level, so a generated `\\n` becomes a REAL newline and
lands inside an f-string. It is in the project's own memory notes. ⇒ Author files
with the Write tool; the shell is for running them, not for containing them.
"""
from __future__ import annotations

import json
import pathlib
import sys

SP = pathlib.Path(__file__).resolve().parent
FILES = {"splitp30k": "egodom_split.json", "rdw8p30k": "egodom.json",
         "ro128p30k": "egodom_ro128.json", "champ30k": "egodom_champ30k.json",
         "scale1": "egodom_scale1.json", "rdw8s30k": "egodom_rdw8s30k.json",
         "splitfrz10k": "egodom_splitfrz10k.json",
         "postrain10k": "egodom_postrain10k.json"}
# ⚠️ the init class is a FACT ABOUT THE ARM, recorded here so the grouping is not
# reconstructed from memory each time it is read.
DISTILLED = {"splitp30k", "splitfrz10k", "postrain10k"}


def main() -> int:
    rows = []
    for a, f in FILES.items():
        p = SP / f
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        for k, A in r.get("arms", {}).items():
            z, res = A.get("z_t", {}), A.get("residual", {})
            px = z.get("RAW PIXELS 8x8")
            ac = res.get("action [steer,accel,v0]")
            if px:
                rows.append((k, px["true_minus_shuffled"], px["t"],
                             ac["true_minus_shuffled"] if ac else 0.0,
                             ac["t"] if ac else 0.0))
    rows.sort(key=lambda r: -r[1])
    print()
    print(f"  {'arm':<14}{'pixels->z_t':>13}{'t':>7}"
          f"{'residual:action':>18}{'t':>7}   init")
    print("  " + "-" * 72)
    for k, px, t, ad, at in rows:
        cls = "DISTILLED" if k in DISTILLED else "scratch"
        print(f"  {k:<14}{px:>+13.4f}{t:>7.2f}{ad:>+18.4f}{at:>7.2f}   {cls}")

    d = [r for r in rows if r[0] in DISTILLED]
    s = [r for r in rows if r[0] not in DISTILLED]
    if d and s:
        dmin, dmax = min(x[1] for x in d), max(x[1] for x in d)
        smin, smax = min(x[1] for x in s), max(x[1] for x in s)
        print()
        print(f"  distilled inits : n={len(d)}  groundedness {dmin:+.4f} .. {dmax:+.4f}")
        print(f"  scratch-trained : n={len(s)}  groundedness {smin:+.4f} .. {smax:+.4f}")
        sep = "CLEAN, no overlap" if dmin > smax else "OVERLAPPING"
        print(f"  SEPARATION: min(distilled) {dmin:+.4f} vs max(scratch) "
              f"{smax:+.4f}  ->  {sep}")

    pos = [r for r in rows if r[4] > 2.0]        # ⛔ t > 2, NOT |t| > 2
    neg = [r for r in rows if r[4] < -2.0]
    print()
    print(f"  arms whose drift-removed residual POSITIVELY carries the action "
          f"(t > 2): {len(pos)} of {len(rows)}")
    if neg:
        print(f"  (and {len(neg)} with t < -2, i.e. the action makes the residual "
              f"prediction WORSE — a negative outlier, NOT a lead)")
    print(f"  => {'the residual is NOISE in every arm measured' if not pos else 'a real residual exists — investigate that arm'}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
