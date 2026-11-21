"""Build a Thor suite-gate plan for `thor_gate_runner2.py` (dev box side; plain git, no worktree read).

    python make_thor_suite_plan.py --git-dir <repo>/.git --tip <sha> --cand <sha> --out <dir>

Writes into <dir>: `tree_tip.tar` (git archive of TIP's stack/ taniteval/ tools/, committed bytes: core.autocrlf
off), `overlay.tar` + `overlay/` (CAND's version of every stack/ or taniteval/ file that differs from TIP),
`plan_thor.json` (the runner's schema, with `tip` and `cand` recorded), and copies of the runner and the verdict
script. Ship <dir> to Thor, then run `nohup python3 thor_gate_runner2.py <dir> <pytest side folder> &`.

TIP and CAND each run every `stack/tests/test_*.py` of their own commit plus the 7 taniteval files of the pinned
protocol (`TANITEVAL_PROTOCOL`; `--taniteval-all` for every taniteval test file). ⛔ A CAND
that DELETES or RENAMES a file under stack/ or taniteval/ is REFUSED: the runner can only overlay files, so the CAND
tree would still carry TIP's copy and the suite would test a tree that is not the commit. Use TIP = the launch
commit's parent (or the commit itself) to keep the overlay small."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOTS = ("stack", "taniteval")
HERE = Path(__file__).resolve().parent
#: ⛔ The taniteval scope of the Thor protocol the PIN was derived on: every Thor suite gate since
#: `qland/work/thorgate_b2_0201` (the source of `launch_gate_thor_env_failures.json`: verdict sha256 70c66cba…, tip
#: 06380de7) ran ALL stack tests and exactly these 7 taniteval files. A wider set would meet failures the pin never
#: classified, so it is opt-in (--taniteval-all) and changes the protocol.
TANITEVAL_PROTOCOL = ("tests/test_bench_suite_contract.py", "tests/test_bench_suite_gpu_gap.py",
                      "tests/test_bench_suite_internal_t1.py", "tests/test_refav1_components_yaw_mask.py",
                      "tests/test_refcv3_arm_astar_geometry.py", "tests/test_refcv6_perception_rebuild.py",
                      "tests/test_render_refcv3_video.py")


def git(gd: str, *a: str) -> bytes:
    return subprocess.run(["git", "--git-dir", gd, "-c", "core.autocrlf=false", *a], check=True,
                          capture_output=True).stdout


def tests_of(gd: str, sha: str, taniteval_all: bool) -> dict[str, list[str]]:
    out = {}
    for root in ROOTS:
        names = git(gd, "ls-tree", "-r", "--name-only", "-z", sha, f"{root}/tests").decode().split("\0")
        out[root] = sorted(n[len(root) + 1:] for n in names
                           if n.rsplit("/", 1)[-1].startswith("test_") and n.endswith(".py"))
    if not out["stack"]:
        sys.exit(f"{sha[:12]}: zero stack test files listed -- an empty listing is not an empty suite")
    if not taniteval_all:
        miss = [f for f in TANITEVAL_PROTOCOL if f not in out["taniteval"]]
        if miss:
            sys.exit(f"{sha[:12]}: the protocol's taniteval files are not all in the commit: {miss}")
        out["taniteval"] = list(TANITEVAL_PROTOCOL)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taniteval-all", action="store_true",
                    help="every taniteval test file (NOT the pinned protocol; see TANITEVAL_PROTOCOL)")
    a = ap.parse_args(argv)
    gd = a.git_dir
    tip, cand = (git(gd, "rev-parse", "--verify", f"{x}^{{commit}}").decode().strip() for x in (a.tip, a.cand))
    for x in (tip, cand):
        if len(x) != 40:
            sys.exit(f"not a 40-char commit: {x!r}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    status = git(gd, "diff", "--name-status", "--no-renames", "-z", tip, cand, "--", *ROOTS).decode().split("\0")
    pairs = list(zip(status[0::2], status[1::2]))
    gone = [p for s, p in pairs if s.startswith("D")]
    if gone:
        sys.exit(f"REFUSED: CAND deletes {len(gone)} file(s) under {ROOTS} (first: {gone[:3]}) -- the runner "
                 f"cannot remove a file from the TIP archive; pick a TIP closer to CAND")
    changed = sorted(p for s, p in pairs if s[:1] in ("A", "M", "T"))
    (out / "tree_tip.tar").write_bytes(git(gd, "archive", "--format=tar", tip, *ROOTS, "tools"))
    md5 = {}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in changed:
            b = git(gd, "show", f"{cand}:{p}")
            md5[p] = hashlib.md5(b).hexdigest()
            ti = tarfile.TarInfo(f"overlay/{p}")
            ti.size, ti.mode = len(b), 0o644
            tf.addfile(ti, io.BytesIO(b))
            d = out / "overlay" / p
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(b)
    (out / "overlay.tar").write_bytes(buf.getvalue())
    tt, tc = tests_of(gd, tip, a.taniteval_all), tests_of(gd, cand, a.taniteval_all)
    plan = {"archive": "tree_tip.tar", "tip": tip, "cand": cand, "overlay_md5": md5,
            "taniteval_scope": "all" if a.taniteval_all else "protocol-7",
            "stages": {"TIP": {"overlay": [], **tt}, "CAND": {"overlay": changed, **tc}}}
    (out / "plan_thor.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    for f in ("thor_gate_runner2.py", "gate_verdict.py"):
        shutil.copyfile(HERE / f, out / f)
    print(f"plan: TIP {tip[:12]} ({len(tt['stack'])}+{len(tt['taniteval'])} test files), CAND {cand[:12]} "
          f"({len(tc['stack'])}+{len(tc['taniteval'])}), overlay {len(changed)} file(s) -> {out}")
    print("ZZPLAN-OKZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
