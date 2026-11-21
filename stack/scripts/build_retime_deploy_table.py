"""Build the SHIP-R8 deployed re-timing table ``RETIME_DEPLOY`` from the FROZEN train-fit gains -- no new fit, no retyped floats.

    python build_retime_deploy_table.py --src <...>/raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json \
        --out <stack>/tanitad/configs/refcv8_retime_deploy_gains.json            # write json + .sha256
    python build_retime_deploy_table.py --src ... --out ... --check              # regenerate in memory, compare to the file on disk

SHIP-R8 AMENDMENT 1 (2026-10-19, ``Project Steering/GOALS_AND_CLAIMS.md``): knots = the S2 gains at 0.5 / 1.0 / 1.5 / 2.0 s and
the S6 gains at 3 / 4 / 5 / 6 s, from ``surfaces.S2.all.RETIME`` and ``surfaces.S6.all.RETIME``; piecewise-linear in time between
knots, held constant outside them (``tanitad.retime_stage.gain_at``).  The S6 gains at 1 s and 2 s are NOT used: they disagree with
the S2 gains at the shared instants, which is exactly why the amendment exists.

The floats are copied from the parsed frozen file and serialised with ``json`` (shortest round-trip repr), so the deployed table
holds the identical IEEE-754 doubles.  The output is deterministic (no wall-clock): a regeneration is byte-identical, and ``--check``
proves it.  The source is refused unless its bytes hash to its ``.sha256`` sidecar AND to the digest pinned below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: the frozen train file's digest (D-R8-SHIP-RETIME: "frozen gains UNCHANGED (sha256 b208c9c1e4236075...)")
SOURCE_SHA256 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
SOURCE_REPO_PATH = ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/retime_trainfit/"
                    "FROZEN_GAINS_RETIME_train.json")
GAC_ROWS = ("D-R8-SHIP-RETIME", "SHIP-R8 AMENDMENT 1")
REGISTERED = "2026-10-19"
#: which source cells make the knots: (surface, instants kept)
KNOT_SOURCES = (("S2", (0.5, 1.0, 1.5, 2.0)), ("S6", (3.0, 4.0, 5.0, 6.0)))


_CRLF = bytes((13, 10))
_LF = bytes((10,))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_lf(b: bytes) -> str:
    """sha256 over the LF-canonical form (CRLF folded to LF) -- what the deployed table's sidecar and ``tanitad.retime_stage``
    use, so a CRLF checkout (``core.autocrlf=true`` on the dev box) and an LF checkout (a pod) verify the same file."""
    return hashlib.sha256(b.replace(_CRLF, _LF)).hexdigest()


def read_frozen(src: Path) -> tuple[dict, str]:
    """The frozen train file was WRITTEN ON WINDOWS (text-mode ``open``): its 409 newlines are CRLF and the registered digest
    b208c9c1... is over THOSE bytes (MEASURED 2026-10-19: the LF-folded form hashes 0c46aa1d...).  Accept the registered digest
    over the raw bytes, or over the CRLF form of the content (a repo checkout that normalised it to LF), and nothing else."""
    raw = Path(src).read_bytes()
    got = sha256_bytes(raw)
    crlf = sha256_bytes(raw.replace(_CRLF, _LF).replace(_LF, _CRLF))
    side = Path(str(src).rsplit(".json", 1)[0] + ".sha256")
    if side.is_file():
        want = side.read_text(encoding="utf-8").strip().split()[0].lower()
        if want != SOURCE_SHA256:
            raise SystemExit(f"[deploy-table] {side.name} {want} != the pinned frozen digest {SOURCE_SHA256}")
    if SOURCE_SHA256 not in (got, crlf):
        raise SystemExit(f"[deploy-table] {src}: sha256 {got} (CRLF form {crlf}) != the frozen digest {SOURCE_SHA256}; refusing")
    return json.loads(raw.decode("utf-8")), SOURCE_SHA256


def build(frozen: dict, src_sha: str) -> dict:
    knots, gains, srcs = [], [], []
    for surf, keep in KNOT_SOURCES:
        s = frozen["surfaces"][surf]
        inst, g = s["instants_s"], s["all"]["RETIME"]
        if len(inst) != len(g):
            raise SystemExit(f"[deploy-table] {surf}: {len(inst)} instants vs {len(g)} gains")
        for t in keep:
            i = [j for j, x in enumerate(inst) if x == t]
            if len(i) != 1:
                raise SystemExit(f"[deploy-table] {surf}: instant {t} s not found exactly once in {inst}")
            knots.append(float(inst[i[0]]))
            gains.append(g[i[0]])                      # the parsed double itself: never rounded, never retyped
            srcs.append(f"surfaces.{surf}.all.RETIME[{i[0]}]")
    return {
        "name": "RETIME_DEPLOY",
        "doc": ("refcv8 frozen post-planner re-timing stage, deployed gain table (SHIP-R8 AMENDMENT 1): piecewise-linear in time "
                "between knots, held constant before the first and after the last knot. tanitad.retime_stage.gain_at."),
        "knots_s": knots,
        "gains": gains,
        "knot_sources": srcs,
        "source": {"path": SOURCE_REPO_PATH, "sha256": src_sha, "fit_source": frozen.get("fit_source"),
                   "prior": frozen.get("prior"), "tree": frozen.get("tree"), "frozen_generated": frozen.get("generated")},
        "gac_rows": list(GAC_ROWS),
        "date": REGISTERED,
        "generated_by": "stack/scripts/build_retime_deploy_table.py",
    }


def dumps(tab: dict) -> bytes:
    return (json.dumps(tab, indent=1, ensure_ascii=True) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="regenerate in memory and compare with --out and its sidecar; write nothing")
    a = ap.parse_args(argv)
    frozen, sha = read_frozen(Path(a.src))
    blob = dumps(build(frozen, sha))
    out = Path(a.out)
    side = Path(str(out) + ".sha256")
    digest = sha256_lf(blob)
    if a.check:
        ok = out.is_file() and out.read_bytes() == blob and side.is_file() and side.read_text(encoding="utf-8").split()[0] == digest
        print(f"[deploy-table] check: {'IDENTICAL' if ok else 'MISMATCH'} sha256 {digest}")
        return 0 if ok else 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    side.write_text(f"{digest}  {out.name}\n", encoding="utf-8", newline="\n")
    print(f"[deploy-table] wrote {out} ({len(blob)} B) sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
