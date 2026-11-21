"""build_deploy_gains_r8.py -- the DEPLOYED re-timing gain table `RETIME_DEPLOY` (SHIP-R8 AMENDMENT 1, 2026-10-19), derived from the FROZEN train file only.

NEW module (never edits an existing file).  Pure numpy/json/hashlib; no stack imports; CPU; no fit.

    python build_deploy_gains_r8.py            # -> raw/retime_deploy/FROZEN_GAINS_RETIME_deploy.json + FROZEN_GAINS_RETIME_deploy.sha256

THE FUNCTION (AMENDMENT 1, quoted from Project Steering/GOALS_AND_CLAIMS.md): knots (0.5 s, 0.967), (1.0, 0.619), (1.5, 0.551), (2.0, 0.531), (3.0, 0.655),
(4.0, 0.699), (5.0, 0.722), (6.0, 0.745); piecewise-linear in time between knots; held constant before 0.5 s and after 6.0 s.  The exact floats are the frozen
train file's `surfaces.S2.all.RETIME` at S2's instants (0.5, 1.0, 1.5, 2.0 s) and `surfaces.S6.all.RETIME` at S6's 3, 4, 5, 6 s instants; S6's own 1.0 s and
2.0 s gains (0.639 / 0.597) are NOT knots -- the deployed function takes the S2 values there.  Consequences (asserted below): on S2 the per-surface table equals the
train table EXACTLY (every S2 instant is a knot); on S6 only the 1.0 s and 2.0 s gains change.

Source: raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json, sha256 b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee (pinned below, checked against the
.sha256 file beside it as well).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TF = PKG / "raw" / "retime_trainfit"
OUT = PKG / "raw" / "retime_deploy"
SOURCE = TF / "FROZEN_GAINS_RETIME_train.json"
SOURCE_SHA256_FILE = TF / "FROZEN_GAINS_RETIME_train.sha256"
SOURCE_SHA256 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
DEPLOY_JSON = OUT / "FROZEN_GAINS_RETIME_deploy.json"
DEPLOY_SHA256_FILE = OUT / "FROZEN_GAINS_RETIME_deploy.sha256"

#: (knot time s, source surface).  The value of knot k is surfaces[src].all.RETIME at that surface's instant == t.
KNOT_PLAN = ((0.5, "S2"), (1.0, "S2"), (1.5, "S2"), (2.0, "S2"), (3.0, "S6"), (4.0, "S6"), (5.0, "S6"), (6.0, "S6"))
#: AMENDMENT 1's own 3-decimal rendering of the knots -- an independent authority for the exact floats (checked by round(x, 3)).
REGISTER_KNOTS_3DP = (0.967, 0.619, 0.551, 0.531, 0.655, 0.699, 0.722, 0.745)
SURFACES = ("S2", "S6")
INTERPOLATION = "piecewise-linear in time between knots; held constant before the first knot (0.5 s) and after the last (6.0 s)"


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def knots_from_frozen(fz: dict) -> tuple[list, list]:
    """(times, gains) of the 8 knots from the frozen TRAIN file.  A knot time that is not one of the source surface's instants is an error (no interpolation to make a knot)."""
    ts, gs = [], []
    for t, surf in KNOT_PLAN:
        node = fz["surfaces"][surf]
        inst = [float(x) for x in node["instants_s"]]
        tab = node["all"]["RETIME"]
        if len(inst) != len(tab):
            raise SystemExit(f"[deploy-gains] {surf}: {len(inst)} instants but {len(tab)} gains")
        hit = [i for i, x in enumerate(inst) if x == t]
        if len(hit) != 1:
            raise SystemExit(f"[deploy-gains] knot {t} s is not an instant of {surf} ({inst})")
        ts.append(float(t))
        gs.append(float(tab[hit[0]]))
    return ts, gs


def gain_at(t: float, knots_t, knots_g) -> float:
    """The deployed gain function g(t): exact at a knot, linear between knots, constant outside.  Hand-written (no np.interp) so the knot-exactness is by construction."""
    t = float(t)
    if t <= knots_t[0]:
        return float(knots_g[0])
    if t >= knots_t[-1]:
        return float(knots_g[-1])
    for i in range(len(knots_t) - 1):
        t0, t1 = knots_t[i], knots_t[i + 1]
        if t == t0:
            return float(knots_g[i])
        if t0 < t < t1:
            w = (t - t0) / (t1 - t0)
            return float(knots_g[i] + w * (knots_g[i + 1] - knots_g[i]))
    return float(knots_g[-1])      # t == knots_t[-1] is handled above; unreachable otherwise


def surface_table(instants, knots_t, knots_g) -> list:
    return [gain_at(t, knots_t, knots_g) for t in instants]


def build(fz: dict) -> dict:
    """The deploy record from a frozen-train dict (pure).  Does NOT check the source hash (main() does; the tests feed synthetic dicts)."""
    kt, kg = knots_from_frozen(fz)
    surfaces = {}
    for surf in SURFACES:
        inst = [float(x) for x in fz["surfaces"][surf]["instants_s"]]
        train = [float(x) for x in fz["surfaces"][surf]["all"]["RETIME"]]
        dep = surface_table(inst, kt, kg)
        surfaces[surf] = {"instants_s": inst, "RETIME_DEPLOY": dep, "RETIME_TRAIN": train, "identical_to_train": dep == train,
                          "changed_vs_train": [{"t_s": t, "train": a, "deploy": b} for t, a, b in zip(inst, train, dep) if a != b]}
    return {"tool": "build_deploy_gains_r8.py", "generated": time.strftime("%FT%T"),
            "purpose": "SHIP-R8 AMENDMENT 1: the deployed re-timing stage is ONE gain function of time, derived from the frozen train gains (no new fit)",
            "source_file": "raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json", "source_sha256": SOURCE_SHA256,
            "knots": {"t_s": kt, "gain": kg, "source_surface": [s for _t, s in KNOT_PLAN], "register_3dp": list(REGISTER_KNOTS_3DP)},
            "interpolation": INTERPOLATION, "surfaces": surfaces}


def check_deploy(d: dict) -> list:
    """Every structural claim of AMENDMENT 1 as a list of failure strings ([] = all hold).  The expectations are LITERALS (the 3-dp register values; the knot times;
    the S2-identity / S6-1-and-2 s substitution), not a re-derivation of the build."""
    bad = []
    kt, kg = d["knots"]["t_s"], d["knots"]["gain"]
    if kt != [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
        bad.append(f"knot times {kt}")
    if [round(g, 3) for g in kg] != list(REGISTER_KNOTS_3DP):
        bad.append(f"knot gains at 3 dp {[round(g, 3) for g in kg]} != register {list(REGISTER_KNOTS_3DP)}")
    s2, s6 = d["surfaces"]["S2"], d["surfaces"]["S6"]
    if s2["instants_s"] != [0.5, 1.0, 1.5, 2.0] or s6["instants_s"] != [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        bad.append("surface instants")
    if s2["RETIME_DEPLOY"] != s2["RETIME_TRAIN"] or not s2["identical_to_train"] or s2["changed_vs_train"]:
        bad.append("S2 deploy table is not identical to the S2 train table")
    if s6["RETIME_DEPLOY"] != [s2["RETIME_TRAIN"][1], s2["RETIME_TRAIN"][3]] + s6["RETIME_TRAIN"][2:]:
        bad.append("S6 deploy table != [S2@1.0, S2@2.0, S6@3, S6@4, S6@5, S6@6]")
    if [c["t_s"] for c in s6["changed_vs_train"]] != [1.0, 2.0]:
        bad.append(f"S6 changed instants {[c['t_s'] for c in s6['changed_vs_train']]} != [1.0, 2.0]")
    if [round(g, 3) for g in s6["RETIME_TRAIN"][:2]] != [0.639, 0.597]:
        bad.append("S6 train 1 s / 2 s gains are not the 0.639 / 0.597 AMENDMENT 1 quotes")
    return bad


def main() -> int:
    got = sha256_file(SOURCE)
    if got != SOURCE_SHA256:
        raise SystemExit(f"[deploy-gains] REFUSING: {SOURCE} hashes to {got}, not the pinned {SOURCE_SHA256}")
    if SOURCE_SHA256_FILE.read_text().strip() != SOURCE_SHA256:
        raise SystemExit("[deploy-gains] REFUSING: the .sha256 file beside the frozen gains does not carry the pinned hash")
    fz = json.loads(SOURCE.read_text(encoding="utf-8"))
    if fz.get("fit_source") != "train" or fz.get("fit_guard", {}).get("n_overlap_with_eval139") != 0:
        raise SystemExit("[deploy-gains] REFUSING: the source is not a train-fitted file")
    d = build(fz)
    bad = check_deploy(d)
    if bad:
        raise SystemExit("[deploy-gains] REFUSING: the built table fails AMENDMENT 1's own checks: " + "; ".join(bad))
    OUT.mkdir(parents=True, exist_ok=True)
    DEPLOY_JSON.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    h = sha256_file(DEPLOY_JSON)
    DEPLOY_SHA256_FILE.write_text(h + "\n")
    # assert on the ARTIFACT: re-read from disk, re-check
    d2 = json.loads(DEPLOY_JSON.read_text(encoding="utf-8"))
    if check_deploy(d2) or sha256_file(DEPLOY_JSON) != DEPLOY_SHA256_FILE.read_text().strip():
        raise SystemExit("[deploy-gains] the written artifact does not re-read clean")
    print(f"WROTE {DEPLOY_JSON} sha256 {h}")
    for s in SURFACES:
        print(s, [round(x, 4) for x in d["surfaces"][s]["RETIME_DEPLOY"]], "identical_to_train" if d["surfaces"][s]["identical_to_train"] else f"changed {d['surfaces'][s]['changed_vs_train']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
