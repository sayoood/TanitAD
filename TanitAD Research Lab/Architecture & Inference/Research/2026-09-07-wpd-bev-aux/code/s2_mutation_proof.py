"""WP-D s2 — THE MUTATION PROOF: can the gate actually go RED?

⛔ A test suite that has never been shown to FAIL is a suite whose green is
uninformative. CLAUDE.md `e4af94f` records four measured cases in one night of a
check that shared the defect it checked for and was therefore *"green forever"*,
and names the strongest discriminator: **a mutation that reintroduces the real
historical defect and must go RED.**

This script does that mechanically. For each mutation it: copies the stack to a
throwaway tree, applies ONE textual edit that re-introduces a defect this
programme actually suffered, runs the WHOLE ``test_bev_aux.py`` against the
mutant, and records which tests failed.

⛔ **A MUTANT THAT LEAVES EVERYTHING GREEN IS A FAILURE OF THIS SCRIPT'S
SUBJECT, NOT OF THE MUTANT** — it means the suite does not test what it claims —
and it is reported as ``SURVIVED``, with a non-zero exit.

⚠️ The edits are asserted to have APPLIED (the old text was present exactly once
and the new text is present afterwards). A mutation that silently did not apply
would produce a green run that reads exactly like a surviving mutant.

CPU only. No GPU, no network.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: (id, file, old, new, why). Each `why` names the real defect.
MUTATIONS = [
    ("M1_MIRRORED_WORLD", "tanitad/data/bev_aux.py",
     "        dx = px - cx\n        dy = py - cy\n",
     "        dx = px - cx\n        dy = py + cy          # MUTANT: +y RIGHT\n",
     "The ego-frame convention is +x fwd / +y LEFT, MEASURED by the parked-car "
     "experiment. E-DEC-18's build: 'a sign error here does not crash and does "
     "not show in a loss curve, it teaches a MIRRORED world.'"),
    ("M2_PINHOLE_FOV", "tanitad/data/bev_aux.py",
     "    hfov_deg: float = 120.0\n",
     "    hfov_deg: float = 92.641      # MUTANT: the pinhole formula\n",
     "The retracted 2026-08-21 error: 2*atan((W/2)/f) gives 92.641 deg on a "
     "CYLINDRICAL corpus whose true field is 120 deg (rig name "
     "camera_front_wide_120fov). It silently deletes every agent between "
     "46.32 and 60 deg of bearing."),
    ("M3_TWO_STATE_MERGE", "tanitad/data/bev_aux.py",
     "    if occlusion == \"mask\":\n        mask &= ~shadow_mask(occ, spec)\n",
     "    if occlusion == \"mask\":\n        pass       # MUTANT: no third state\n",
     "WP-A on WP-D's critical path: 'a scored negative still MERGES "
     "seen-and-empty with agent-occluded'. Supervising 'empty' on an "
     "occupied-but-occluded cell teaches the trunk that occluded space is free. "
     "MEASURED cost on B1 EVAL: 27.96 % of occupied cells are occluded."),
    ("M4_NO_LABEL_IS_CLEAR", "tanitad/data/bev_aux.py",
     "        return np.zeros(spec.shape, dtype=np.float32), \\\n"
     "            np.zeros(spec.shape, dtype=bool)\n",
     "        return np.zeros(spec.shape, dtype=np.float32), \\\n"
     "            np.ones(spec.shape, dtype=bool)     # MUTANT: NO_LABEL = clear\n",
     "The join's own documentation: 'An ABSENT (clip, frame) line is NO_LABEL "
     "..., never road clear'. The labels span ~20 s while egomotion runs "
     "48-140 s, so most frames of a long clip carry no label at all."),
    ("M5_TIE_BLIND_AP", "tanitad/refs/refc_bev_aux.py",
     "    ends = np.flatnonzero(np.r_[np.diff(s) != 0.0, True])\n"
     "    prec = tp[ends] / (ends + 1).astype(np.float64)\n"
     "    d_rec = np.diff(np.r_[0.0, tp[ends]]) / n_pos\n"
     "    return float((prec * d_rec).sum())\n",
     "    prec = tp / np.arange(1, y.size + 1, dtype=np.float64)   # MUTANT\n"
     "    return float((prec * y).sum() / n_pos)\n",
     "The per-sample AP breaks ties by array order, so a CONSTANT control reads "
     "0.010236 against a base rate of 0.008333 (+22.8 %) and every arm would be "
     "ranked against an inflated floor. This defect was REAL in this module "
     "until its own control caught it on the first run."),
    ("M6_HEAD_NOT_LAST", "tanitad/refs/refc.py",
     "        self.route_head = nn.Linear(feat, N_ROUTE)\n",
     "        self.route_head = nn.Linear(feat, N_ROUTE)\n"
     "        if getattr(cfg, 'bev_aux', None) is not None:\n"
     "            _mutant_early = nn.Linear(7, 7)   # MUTANT: an RNG draw ...\n"
     "            del _mutant_early                 # ... in the AUX ARM ONLY\n",
     "Module construction draws from the global RNG. A head constructed before "
     "the end of __init__ silently changes every SUBSEQUENT module's INITIAL "
     "WEIGHTS in the aux arm only, so the aux-on/aux-off A/B differs in the "
     "SEED as well as in the lever — a one-variable violation invisible in "
     "every log. ⚠️ The FIRST version of this mutant inserted an unconditional "
     "draw and SURVIVED, correctly: shifting both arms equally is not the "
     "defect. The mutant has to be arm-conditional to be the real one."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, help="clean stack snapshot")
    ap.add_argument("--work", required=True, help="scratch dir for mutants")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = Path(a.stack)
    work = Path(a.work)
    results = []

    # --- baseline: the clean tree MUST be green, or nothing below means anything
    base = subprocess.run(
        [a.python, "-m", "pytest", "tests/test_bev_aux.py", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=str(src), env={**__import__("os").environ, "PYTHONPATH": str(src)},
        capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    base_ok = base.returncode == 0
    print(f"[s2] BASELINE {'GREEN' if base_ok else 'RED'}  "
          f"{base.stdout.strip().splitlines()[-1] if base.stdout else ''}",
          flush=True)
    if not base_ok:
        print(base.stdout[-3000:])
        raise SystemExit("[s2] the clean tree is not green — fix that first")

    for mid, rel, old, new, why in MUTATIONS:
        dst = work / mid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__",
                                                      ".pytest_cache"))
        p = dst / rel
        txt = p.read_text(encoding="utf-8")
        n_hits = txt.count(old)
        # ⚠️ EXACTLY ONCE. A mutation that did not apply produces a green run
        # indistinguishable from a surviving mutant; one that applied twice is a
        # different experiment from the one described.
        assert n_hits == 1, f"{mid}: anchor found {n_hits} times in {rel}"
        p.write_text(txt.replace(old, new), encoding="utf-8")
        assert new in p.read_text(encoding="utf-8"), f"{mid}: edit did not land"

        r = subprocess.run(
            [a.python, "-m", "pytest", "tests/test_bev_aux.py", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=str(dst),
            env={**__import__("os").environ, "PYTHONPATH": str(dst)},
            capture_output=True, text=True,
        encoding="utf-8", errors="replace")
        failed = sorted({ln.split("::")[1].split()[0].split(" ")[0]
                         for ln in r.stdout.splitlines()
                         if ln.startswith("FAILED tests/")})
        killed = r.returncode != 0
        results.append({"mutation": mid, "file": rel, "why": why,
                        "killed": killed, "n_failed": len(failed),
                        "failed_tests": failed,
                        "summary": (r.stdout.strip().splitlines()[-1]
                                    if r.stdout.strip() else "")})
        # ⚠️ ASCII on purpose. This console is cp1252 and an emoji here CRASHED
        # the script mid-report on the first run — the "truncated artifact that
        # reads like a complete one" trap, in a script whose whole job is to
        # report a failure.
        print(f"[s2] {mid:24s} {'KILLED' if killed else 'SURVIVED!!':12s} "
              f"{len(failed)} test(s): {', '.join(failed[:4])}", flush=True)
        shutil.rmtree(dst, ignore_errors=True)

    survived = [r["mutation"] for r in results if not r["killed"]]
    out = {"baseline_green": base_ok, "n_mutations": len(MUTATIONS),
           "n_killed": sum(1 for r in results if r["killed"]),
           "survived": survived, "results": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[s2] {out['n_killed']}/{len(MUTATIONS)} mutants KILLED; "
          f"wrote {a.out}")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
