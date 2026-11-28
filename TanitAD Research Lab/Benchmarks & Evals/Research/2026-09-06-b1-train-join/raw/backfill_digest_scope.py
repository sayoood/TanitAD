"""Give a legacy join sidecar the digest-scope block its CONSUMER requires.

MEASURED 2026-09-06: `refc_v3_train.py --agent-join-verify auto` calls
`join_meta.read_digest_scope`, which requires a TOP-LEVEL `digest_scope` block
and REFUSES a sidecar that declares nothing -- deliberately, with no fallback.
`build_b1_agent_join.py` wrote only `summary.digest_scope`, a STRING nested
inside `summary`, which that reader never looks at. Both the B1 EVAL join and
the B1 TRAIN join therefore carried a digest the trainer could not use.

`join_meta.backfill` MEASURES which artifact the recorded digest actually covers
(it hashes both candidates and refuses if neither -- or both -- match), so the
declaration is evidence, not an assumption.

Every print is ASCII.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
for _p in (_REPO / "stack", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.data import join_meta as jm  # noqa: E402


def do(join_path: str) -> dict:
    jp = Path(join_path)
    side = jm.sidecar_path(jp)
    if side is None:
        raise SystemExit("no sidecar beside %s" % jp)
    meta = json.loads(io.open(side, encoding="utf-8").read())
    if isinstance(meta.get("digest_scope"), dict):
        sc = jm.read_digest_scope(meta, where=str(side))
        print("  %s ALREADY declares: %s(%s of %s)"
              % (side.name, sc.algo, sc.scope, sc.filename))
        return {"path": str(jp), "action": "already-declared",
                "scope": sc.scope}
    new, ev = jm.backfill(meta, jp, algo="md5")
    io.open(side, "w", encoding="utf-8", newline="\n").write(
        json.dumps(new, indent=1))
    sc = jm.read_digest_scope(new, where=str(side))
    # positive re-read: the consumer's own verify must now pass
    v = jm.verify(jp, new, where=str(side))
    print("  %s BACKFILLED: recorded=%s matched=%s -> %s(%s of %s); "
          "consumer verify OK: %s"
          % (side.name, ev["recorded"][:12], ev["matched"], sc.algo, sc.scope,
             sc.filename, v.get("digest", "")[:12]))
    return {"path": str(jp), "action": "backfilled", "scope": sc.scope,
            "evidence": ev, "verified": True}


if __name__ == "__main__":
    out = [do(p) for p in sys.argv[1:]]
    print(json.dumps(out, indent=1)[:1200])
