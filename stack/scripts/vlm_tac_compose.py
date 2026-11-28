"""Compose banked VLM generations into tactical/strategic labels + a census.

⭐ THIS IS WHERE EVERY RULE IS APPLIED, AND IT RUNS HERE ON PURPOSE. The remote
side emits raw text and decides nothing; parsing, the strict verdict
vocabulary, the admissible-set prior, the referent requirement, the ego gate
and the composition all live in tested code (``vlm_tac_prompts.parse_verdict`` /
``parse_referent``, ``tac_str_labels.compose``). A second implementation of the
rules next to the GPU would be the one nobody tests.

⛔ THE ECHO-FREE PASS IS THE LABEL. Only ``no_ego`` generations compose. The
``with_ego`` pass exists as a CONTROL — to measure whether showing ego numbers
changes the verdict — and a control that is allowed to become the label is not a
control. (The programme has been burned by this shape before: flagship v1's
route head scored 1.0000 as an exact bijection of the nav it was fed.)

⚠️ THE CENSUS IS PART OF THE DELIVERABLE, NOT A NICETY. A label set is only
usable if you can say how many labels exist, how many abstained, and WHY. The
counts that matter most are the ones that mean "do not trust this yet":
``hit_cap`` (truncated mid-reasoning — a statement about the operator's cap, not
the model), ``referent_echoed_prompt`` (the model may be parroting the prompt's
own example), and the disagreement flags.

Usage:
    python stack/scripts/vlm_tac_compose.py --raw raw.jsonl --payload payload.json \
        --out labels.jsonl --census census.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

import vlm_tac_prompts as P  # noqa: E402
from tanitad.lake.tac_str_labels import ABSTAIN, compose  # noqa: E402

LANE_REL = ("LEFT", "CURRENT", "RIGHT")


def load_raw(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "clip_id" in r and "kind" in r:
                out[(r["clip_id"], r["kind"])] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--vocab", default="v6.1")
    args = ap.parse_args(argv)

    raw = load_raw(Path(args.raw))
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    clips = {c["clip_id"]: c for c in payload["clips"]}

    cen: dict = {
        "n_clips_in_payload": len(clips),
        "generations": Counter(), "verdicts": {}, "flags": Counter(),
        "labels": {"lat": Counter(), "lon": Counter(), "g_str": Counter()},
        "quality": Counter(),
        "parity_gate": payload.get("_parity_gate"),
    }
    rows = []
    for cid, clip in clips.items():
        verdicts: dict[str, str] = {}
        referent = None
        echoed = False
        for kind in ("lon", "lane", "sign"):
            rec = raw.get((cid, kind))
            if rec is None:
                cen["generations"][f"{kind}/absent"] += 1
                continue
            if "error" in rec:
                cen["generations"][f"{kind}/error"] += 1
                continue
            cen["generations"][f"{kind}/ok"] += 1
            complete = not rec.get("hit_cap", False)
            if not complete:
                cen["quality"]["hit_cap"] += 1
            if rec.get("closed_think"):
                cen["quality"]["closed_think"] += 1
            v, _think = P.parse_verdict(rec["raw"], kind, complete=complete)
            verdicts[kind] = v
            cen["verdicts"].setdefault(kind, Counter())[v] += 1
            if kind == "lon":
                referent, echoed = P.parse_referent(rec["raw"], complete=complete)
                if referent:
                    cen["quality"]["referent_named"] += 1
                if echoed:
                    cen["quality"]["referent_echoed_prompt"] += 1

        vlon = verdicts.get("lon", ABSTAIN)
        vlane = verdicts.get("lane", ABSTAIN)
        lab = compose(
            clip_id=cid,
            alpamayo_lane=clip["alpamayo"]["lane"],
            alpamayo_lon=clip["alpamayo"]["longitudinal"],
            # ⭐ the two fields the pipeline used to discard entirely. `cot` is
            # 100 % populated and names the referent on 88.7 % of clips;
            # `lateral` ("Go Straight", 53 %) is the ROUTE-level signal that
            # makes FOLLOW_MAIN_ROAD / STRAIGHT_THROUGH reachable.
            alpamayo_lateral=clip["alpamayo"].get("lateral"),
            alpamayo_cot=clip["alpamayo"].get("cot"),
            vlm_lon=None if vlon == ABSTAIN else vlon,
            vlm_referent=referent,
            vlm_lane_target_rel=vlane if vlane in LANE_REL else ABSTAIN,
            vocab_version=args.vocab)
        d = lab.to_dict()
        # ⚠️ the echo flag is not a compose() concern (it is about the PROMPT,
        # not the label algebra) but it must survive into the artifact, or a
        # parroted referent becomes indistinguishable from an observed one.
        if echoed:
            d.setdefault("flags", []).append("REFERENT_ECHOED_PROMPT_EXAMPLE")
        rows.append(d)
        for f in d.get("flags", []):
            cen["flags"][f] += 1
        cen["labels"]["lat"][d["a_tac_lat"]["value"]] += 1
        cen["labels"]["lon"][d["a_tac_lon"]["value"]] += 1
        cen["labels"]["g_str"][(d.get("g_str") or {}).get("value")] += 1

    with Path(args.out).open("w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _plain(o):
        if isinstance(o, Counter):
            return dict(o)
        if isinstance(o, dict):
            return {k: _plain(v) for k, v in o.items()}
        return o

    cen = _plain(cen)
    Path(args.census).write_text(json.dumps(cen, indent=1), encoding="utf-8")

    print(f"labels {len(rows)} -> {args.out}")
    print(f"generations   {cen['generations']}")
    for k, v in cen["verdicts"].items():
        print(f"  verdicts {k:5s} {v}")
    print(f"quality       {cen['quality']}")
    print(f"flags         {cen['flags']}")
    for axis, c in cen["labels"].items():
        print(f"  labels {axis:6s} {c}")
    # ⛔ THE ABSTAIN DECOMPOSITION. A single ABSTAIN column conflates three
    # completely different states, and reporting them together made a
    # half-finished run look like a refusing model (MEASURED 2026-08-19: of six
    # ABSTAINs, FOUR were generations that had never run and ZERO were genuine
    # refusals). Never print an abstain total without this breakdown.
    cap = cen["quality"].get("hit_cap", 0)
    tot = sum(v for k, v in cen["generations"].items() if k.endswith("/ok"))
    absent = sum(v for k, v in cen["generations"].items() if k.endswith("/absent"))
    err = sum(v for k, v in cen["generations"].items() if k.endswith("/error"))
    cen["abstain_decomposition"] = {
        "generation_never_ran": absent, "generation_errored": err,
        "truncated_at_cap": cap, "ran_to_EOS": max(0, tot - cap),
        "_read": "only ran_to_EOS abstentions are statements about the MODEL; "
                 "the rest are statements about the RUN or the operator's cap",
    }
    Path(args.census).write_text(json.dumps(cen, indent=1), encoding="utf-8")
    print(f"\nABSTAIN decomposition — never-ran {absent} · errored {err} · "
          f"truncated-at-cap {cap} · ran-to-EOS {max(0, tot - cap)}")
    if absent:
        print(f"⛔ {absent} generation(s) NEVER RAN. Those are NOT model "
              f"abstentions and must not be read as capability.")
    if cap:
        print(f"⚠️ {cap}/{tot} hit the token cap and were truncated mid-reasoning "
              f"— a fact about the CAP, not the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
