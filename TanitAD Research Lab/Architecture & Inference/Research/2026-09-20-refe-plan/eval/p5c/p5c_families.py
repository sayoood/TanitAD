#!/usr/bin/env python3
"""P5-C read instrument B9: the four metric families on BOTH frames for C1, C0, base, and paired C1 - C0 (also C0 - base, C1 - base) via fam_ci.

Thin wrapper, nothing re-implemented: `families_run.py --per-window` per arm (rows from transp_score), then `fam_ci.py` per pair.
Frames (card 8.2): executed_tracked (PRIMARY, the only frame G2 reads), executed_transparent (reported, NEVER a guard), reference (diagnostic).
STRATEGIC = PAUSED (PI 2026-09-19, item 25): printed as PAUSED, never a pass, never omitted.
Pairing direction: fam_ci's diff = B - A; the pair "C1-C0" runs --a C0 --b C1 so every difference reads x - y.

  python p5c_families.py --rows "<rows_s*.jsonl>" --arms C1 C0 base --pairs C1:C0 C0:base C1:base --out-dir D [--n-boot 2000]
Outputs  D/fam_<arm>.json, D/famci_<x>_vs_<y>.json (x - y), D/families_summary.json (per pair x frame: separated-adverse list with role),
         D/tokens.txt (`log:token`, sorted by token = the arrays' order).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.abspath(os.path.join(HERE, "..", ".."))
FAMILIES_RUN = os.path.join(HERE, "..", "instruments", "families_run.py")
FAM_CI = os.path.join(PK, "raw", "2026-11-12-instruments", "fam_ci", "fam_ci.py")
REPO = os.path.abspath(os.path.join(PK, "..", "..", "..", ".."))    # PK is 4 levels below the repo root
ROLE = {"executed_tracked": "PRIMARY: the only frame G2 guards on", "executed_transparent": "REPORTED, NEVER A GUARD (inflated by construction on a pre-distorted plan)",
        "reference": "diagnostic"}
sys.path.insert(0, HERE)
from p5c_read import entry_adverse, fam_direction, STRATEGIC     # noqa: E402  (the same polarity rule G2 uses)


def run_cmd(argv: list, log: str) -> int:
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def arm_families(rows_glob: str, arm: str, out_dir: str, n_boot: int, python: str, repo: str) -> str:
    out = os.path.join(out_dir, f"fam_{arm}.json")
    rc = run_cmd([python, FAMILIES_RUN, "--rows", rows_glob, "--arm", arm, "--label", arm, "--out", out, "--n-boot", str(n_boot),
                  "--per-window", "--tanitad-repo", repo], os.path.join(out_dir, f"fam_{arm}.log"))
    if rc != 0 or not os.path.exists(out):
        raise SystemExit(f"families_run failed for {arm} (rc {rc}); see {out_dir}/fam_{arm}.log")
    raised = raised_frames(json.load(open(out, encoding="utf-8")))
    if raised:       # a frame whose block RAISED has no components: fam_ci would write empty families and G2 would read a vacuous PASS
        raise SystemExit(f"families_run for {arm}: frames {raised} RAISED (first: {raised[0][1]}); the --repo tree is probably older than the instrument (per_window); see {out}")
    return out


def raised_frames(fam: dict) -> list:
    """[(frame, error)] for every frame of a families_run JSON whose block did not compute."""
    out = []
    for fr, blk in (fam.get("frames") or {}).items():
        b = blk.get("block") if isinstance(blk, dict) else None
        if isinstance(b, dict) and b.get("status") == "RAISED":
            out.append((fr, b.get("error")))
    return out


def write_tokens(rows_glob: str, out: str) -> int:
    pairs = {}
    for fp in sorted(glob.glob(rows_glob)):
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                if "error" not in r:
                    pairs[r["token"]] = r["log"]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for t in sorted(pairs):
            f.write(f"{pairs[t]}:{t}\n")
    return len(pairs)


def pair_ci(fx: str, fy: str, x: str, y: str, tokens: str, out_dir: str, python: str) -> str:
    """x - y: fam_ci's B - A with B = x, A = y."""
    out = os.path.join(out_dir, f"famci_{x}_vs_{y}.json")
    rc = run_cmd([python, FAM_CI, "--a", fy, "--b", fx, "--tokens", tokens, "--out", out], os.path.join(out_dir, f"famci_{x}_vs_{y}.log"))
    if rc != 0 or not os.path.exists(out):
        raise SystemExit(f"fam_ci failed for {x} - {y} (rc {rc}); see {out_dir}/famci_{x}_vs_{y}.log")
    return out


def summarize(famci: dict) -> dict:
    """per frame: every LONGITUDINAL / LATERAL component separated adverse (C1-side worse) with the frame's role; STRATEGIC PAUSED."""
    out = {"strategic": STRATEGIC, "frames": {}}
    for fr, role in ROLE.items():
        blk = (famci.get("frames") or {}).get(fr)
        if not blk:
            out["frames"][fr] = {"role": role, "status": "ABSENT"}
            continue
        adv, ok, other = [], 0, 0
        for fam in ("longitudinal", "lateral"):
            for k, e in (blk["families"].get(fam) or {}).items():
                r = entry_adverse(k, e)
                if r is None:
                    other += 1
                    continue
                ok += 1
                if r:
                    adv.append({"family": fam, "component": k, "direction": fam_direction(k), "diff": e["diff_B_minus_A"]})
        out["frames"][fr] = {"role": role, "guard": fr == "executed_tracked", "components_compared": ok, "components_undecided": other,
                             "separated_adverse": adv, "tactical_components": len((blk["families"].get("tactical") or {}))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="glob of transp_score rows")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="x:y  -> x - y")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000, help="families_run's own per-arm CI draws (fam_ci always uses its 10,000)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--repo", default=REPO, help="dir with taniteval/ and stack/")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tokens = os.path.join(a.out_dir, "tokens.txt")
    n = write_tokens(a.rows, tokens)
    fam = {arm: arm_families(a.rows, arm, a.out_dir, a.n_boot, a.python, a.repo) for arm in a.arms}
    summ = {"n_tokens": n, "pairs": {}}
    for p in a.pairs:
        x, y = p.split(":")
        fp = pair_ci(fam[x], fam[y], x, y, tokens, a.out_dir, a.python)
        summ["pairs"][f"{x}-{y}"] = {"file": fp, **summarize(json.load(open(fp, encoding="utf-8")))}
    with open(os.path.join(a.out_dir, "families_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summ, f, indent=1)
    for k, v in summ["pairs"].items():
        t = v["frames"]["executed_tracked"]
        print("ZZP5CR_FAM", k, "tracked-frame compared", t.get("components_compared"), "adverse", [x["component"] for x in t.get("separated_adverse", [])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
