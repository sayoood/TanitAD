"""Emit v7 labels over the WHOLE 26.3 h Alpamayo corpus.

⭐ THE MORNING RUNNER (PI /goal, 2026-08-23): *"prepare the pipeline to run
tomorrow early in the whole 26 h corpus"*.

    python stack/scripts/s2_run_corpus.py --out <dir>            # everything
    python stack/scripts/s2_run_corpus.py --out <dir> --limit 200
    python stack/scripts/s2_run_corpus.py --out <dir> --resume   # after a stop

Scope = the 4,729 Alpamayo clips INTERSECTED with available egomotion. Both
sides are UUID-keyed, so there is no legacy-id join and none of the 20.5 %
wrong-episode failure that invalidated the first validation (C140).

## What it refuses to do

* **It never silently labels a clip it cannot fully read.** A clip whose
  recording is too short for the 6 s plan is REFUSED with a reason, not
  emitted with a truncated window — a short label is indistinguishable from a
  real one downstream.
* **It never invents a token.** Every emitted token passes
  `vocab_v7.assert_frozen`; the vocabulary is frozen (PI 2026-08-23) and an
  unknown token raises rather than producing a corpus with an unmappable class.
* **It reports coverage per FAMILY**, because a missing family is a work item
  and a pooled count hides it.

⚠️ Writes JSONL incrementally and flushes per shard, so a kill costs one shard
rather than the run. `--resume` skips clips already present in the output.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tanitad.data import alpamayo_records as AR          # noqa: E402
from tanitad.data import egomotion_source as ES          # noqa: E402

import s2_geom_emit_v7 as EMIT                           # noqa: E402


def scope() -> list[str]:
    """Clips we can label: Alpamayo augmentation AND egomotion, both by UUID."""
    return sorted(AR.available() & ES.available())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=int, default=250)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = out / "s2_labels_v7.jsonl"

    clips = scope()
    if a.limit:
        clips = clips[:a.limit]

    done: set[str] = set()
    if a.resume and labels.exists():
        for line in labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["clip_id"])
                except Exception:                        # noqa: BLE001
                    pass
        clips = [c for c in clips if c not in done]

    print(f"corpus: {len(AR.available())} alpamayo | {len(ES.available())} egomotion "
          f"| scope {len(scope())} | to emit {len(clips)}"
          + (f" (resuming, {len(done)} done)" if done else ""), flush=True)

    t0 = time.time()
    n_ok = 0
    refused: list[dict] = []
    fam = collections.Counter()
    tok_g_str, tok_a_str, tok_g_tac = (collections.Counter() for _ in range(3))
    tok_lat, tok_lon, tok_nav = (collections.Counter() for _ in range(3))
    agree_lat = [0, 0]
    agree_lon = [0, 0]
    grounded = [0, 0]
    buf: list[str] = []

    mode = "a" if (a.resume and labels.exists()) else "w"
    with labels.open(mode, encoding="utf-8") as fh:
        for i, c in enumerate(clips):
            try:
                r = EMIT.emit_one(c)
            except Exception as e:                        # noqa: BLE001
                refused.append({"clip_id": c, "reason": f"{type(e).__name__}: {e}"})
                continue

            n_ok += 1
            tok_g_str[r["g_str"]["token"]] += 1
            tok_a_str[r["a_str"]["token"]] += 1
            for g in r["g_tac"]["goals"]:
                tok_g_tac[g] += 1
            tok_lat[r["a_tac"]["lat"]] += 1
            tok_lon[r["a_tac"]["lon"]] += 1
            tok_nav[r["nav_command"]["token"]] += 1

            al = r.get("alpamayo") or {}
            if (al.get("lateral") or {}).get("agree") is not None:
                agree_lat[1] += 1
                agree_lat[0] += bool(al["lateral"]["agree"])
            if (al.get("longitudinal") or {}).get("agree") is not None:
                agree_lon[1] += 1
                agree_lon[0] += bool(al["longitudinal"]["agree"])
            for g, args in r["g_tac"]["goals"].items():
                if isinstance(args, dict) and "grounded" in args:
                    grounded[1] += 1
                    grounded[0] += bool(args["grounded"])

            # the four binding metric families, per clip
            fam["longitudinal"] += any(k in r["g_tac"]["goals"] for k in
                                       ("SPEED_BAND", "STOP_POINT"))
            fam["lateral"] += r["a_tac"]["lat"] != "LANE_KEEP"
            fam["tactical"] += bool(r["g_tac"]["goals"])
            fam["strategic"] += r["g_str"]["token"] != "FOLLOW_ROUTE"

            buf.append(json.dumps(r, ensure_ascii=False))
            if len(buf) >= a.shard:
                fh.write("\n".join(buf) + "\n")
                fh.flush()
                buf.clear()
                el = time.time() - t0
                print(f"  [{i+1}/{len(clips)}] ok={n_ok} refused={len(refused)} "
                      f"{(i+1)/max(el,1e-9):.0f} clips/s "
                      f"ETA {(len(clips)-i-1)/max((i+1)/max(el,1e-9),1e-9)/60:.1f} min",
                      flush=True)
        if buf:
            fh.write("\n".join(buf) + "\n")

    # ⛔ EMISSION CENSUS — catches a token silently falling to ZERO.
    # MEASURED 2026-08-28: `FOLLOW_LANE` went 1,281 -> 0 when SPEED_BAND became
    # unconditional (the `if not goals` fallback could never fire again), and
    # NOTHING caught it. `test_vocab_reachability` checks the DECLARED
    # unreachable list, not what the corpus actually produces — so a regression
    # that empties a class is invisible to it. This census is the check that
    # binds, and it runs on every corpus build.
    from tanitad.models import vocab_v7 as _V7
    _seen = collections.Counter()
    for _c in (tok_g_str, tok_a_str, tok_g_tac, tok_lat, tok_lon, tok_nav):
        _seen.update(_c)
    silent = sorted(t for t in _V7.ALL_V7_TOKENS
                    if not _seen.get(t) and t not in _V7.NOT_YET_EXTRACTABLE)
    if silent:
        print(f"  ⚠️ {len(silent)} frozen token(s) emitted ZERO times and NOT "
              f"declared unreachable: {silent}", flush=True)

    summary = {
        "emitted": n_ok,
        "vocab_emission_census": {
            "frozen": len(_V7.ALL_V7_TOKENS),
            "emitted_at_least_once": sum(1 for t in _V7.ALL_V7_TOKENS if _seen.get(t)),
            "declared_unreachable": len(_V7.NOT_YET_EXTRACTABLE),
            "silent_and_undeclared": silent,
        },
        "refused": len(refused),
        "scope": len(scope()),
        "alpamayo_clips": len(AR.available()),
        "egomotion_clips": len(ES.available()),
        "seconds": round(time.time() - t0, 1),
        "hours_of_driving": round(n_ok * 20.0 / 3600.0, 1),
        "families_present": dict(fam),
        "g_str": dict(tok_g_str.most_common()),
        "a_str": dict(tok_a_str.most_common()),
        "g_tac": dict(tok_g_tac.most_common()),
        "a_tac_lat": dict(tok_lat.most_common()),
        "a_tac_lon": dict(tok_lon.most_common()),
        "nav": dict(tok_nav.most_common()),
        "alpamayo_lateral_agreement": (
            f"{agree_lat[0]}/{agree_lat[1]}"
            + (f" = {agree_lat[0]/agree_lat[1]:.1%}" if agree_lat[1] else "")),
        "alpamayo_longitudinal_agreement": (
            f"{agree_lon[0]}/{agree_lon[1]}"
            + (f" = {agree_lon[0]/agree_lon[1]:.1%}" if agree_lon[1] else "")),
        "cot_tokens_grounded_by_box": (
            f"{grounded[0]}/{grounded[1]}"
            + (f" = {grounded[0]/grounded[1]:.1%}" if grounded[1] else "")),
        "done": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "refused.json").write_text(json.dumps(refused, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
