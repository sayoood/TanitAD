"""⚠️ RUNBOOK STEP 3 — put the pod's registration into the repo's manifest.

**This is the step that gets forgotten**, and forgetting it is worse than
stranding work: the cache reads NON-PARITY on every host but the one that built
it, and under ``--require-parity`` the trainer refuses to start
(``parity.py`` §9 ``_sibling_candidate_key``, ``V5_TRAINER.md`` §5.1).

WHY A MERGE AND NOT AN ``scp`` OF THE WHOLE FILE
------------------------------------------------
``register_v2_sibling.py --write-manifest`` rewrites the manifest on the POD.
Copying that file back would (a) clobber any entry another stream added to the
repo copy in the meantime and (b) rewrite every line, because the repo's working
copy is CRLF and the pod's is LF — a 1,700-line whitespace diff with the one real
change buried in it. So only the ENTRY travels, and it is inserted here.

WHAT THIS REFUSES
-----------------
* an entry whose ``corpus_key`` is not ``--new-key``;
* a key that already exists (unless ``--force``) — a truncated cache must never
  overwrite a good manifest;
* **any diff other than the addition of exactly that one key** — checked by
  re-parsing both sides and comparing the corpora dicts, so a stray edit or a
  reformat cannot ride along.

🔒 The entry carries counts and sha256 digests only; no clip ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MANIFEST = (Path(__file__).resolve().parents[6]
                    / "stack" / "tanitad" / "data" / "parity_manifest.json")


def _newline_of(path: Path) -> str:
    """Preserve the file's own line ending so the diff is the change alone."""
    raw = path.read_bytes()
    return "\r\n" if b"\r\n" in raw else "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True,
                    help="the entry JSON written by register_v2_sibling.py")
    ap.add_argument("--new-key", required=True)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    mp = Path(a.manifest)
    nl = _newline_of(mp)
    before = json.loads(mp.read_text(encoding="utf-8"))
    ent = json.loads(Path(a.entry).read_text(encoding="utf-8"))

    if ent.get("corpus_key") != a.new_key:
        raise SystemExit(
            f"REFUSING: the entry's corpus_key is {ent.get('corpus_key')!r}, "
            f"not {a.new_key!r}. An entry filed under the wrong key is an inert "
            f"registration — corpus_key_of() would never resolve to it.")
    if a.new_key in before.get("corpora", {}) and not a.force:
        prev = before["corpora"][a.new_key]
        raise SystemExit(
            f"REFUSING: {a.new_key} is ALREADY in {mp} "
            f"({prev.get('episode_count')} episodes, digest "
            f"{str(prev.get('episode_uid_sha256'))[:12]}…). Pass --force only "
            f"if you have verified the cache that should own this key.")

    after = json.loads(json.dumps(before))
    after.setdefault("corpora", {})[a.new_key] = ent

    added = set(after["corpora"]) - set(before.get("corpora", {}))
    removed = set(before.get("corpora", {})) - set(after["corpora"])
    changed = {k for k in before.get("corpora", {})
               if json.dumps(before["corpora"][k], sort_keys=True)
               != json.dumps(after["corpora"][k], sort_keys=True)}
    if added != {a.new_key} or removed or changed:
        raise SystemExit(
            f"REFUSING: the merge would change more than the one new key. "
            f"added={sorted(added)} removed={sorted(removed)} "
            f"changed={sorted(changed)}")
    for k in ("schema", "notes"):
        if json.dumps(before.get(k), sort_keys=True) != \
                json.dumps(after.get(k), sort_keys=True):
            raise SystemExit(f"REFUSING: top-level {k!r} would change.")

    txt = json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    print(f"[merge] {mp}")
    print(f"[merge] adds exactly 1 corpus key: {a.new_key}")
    print(f"[merge] episode_count={ent.get('episode_count')} "
          f"uid_kind={ent.get('uid_kind')} "
          f"clip_sha256={str(ent.get('episode_uid_sha256'))[:16]}…")
    if a.dry_run:
        print("[merge] --dry-run: nothing written")
        return 0
    with open(mp, "w", encoding="utf-8", newline=nl) as fh:
        fh.write(txt)
    print(f"[merge] WROTE {mp} (line ending preserved: "
          f"{'CRLF' if nl == chr(13) + chr(10) else 'LF'})")
    print("[merge] ⚠️ NOW STAGE IT:  git add "
          "stack/tanitad/data/parity_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
