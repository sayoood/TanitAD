#!/usr/bin/env python3
"""Mark the 2.0 s SEAM sheet SUPERSEDED — in place, without destroying it.

⛔ WHY NOT JUST DELETE IT. C89's own rule: *"do not delete the old number; mark
it as the seam value so the shape of the error stays legible."* The same applies
to the artefact. The 2.0 s sheet is kept beside the band sheet as the record of
what was measured and how it was wrong.

⚠️ WHY A BANNER AND NOT A RENAME. The filename is referenced from
`TACTICAL_REVIEW.md` and the review-selection JSON. Renaming would break those
paths and, worse, a stale link would 404 rather than explain itself. A banner
travels WITH the artefact: whoever opens the file — from any link, at any future
date — is told in the first line that it reads the wrong horizon.

⚠️ The localStorage key of the old sheet is deliberately NOT touched. Any
verdicts already stored against it stay recoverable.

This script is IDEMPOTENT: re-running it does not stack banners.

Usage:
  python mark_seam_sheet_superseded.py [--sheet <html>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]

MARK = "data-superseded-banner=\"c89\""

BANNER = f"""<div {MARK} style="background:#8b1a1a;color:#fff;padding:12px 16px;
 border-radius:8px;margin:0 0 16px;font:14.5px/1.55 -apple-system,Segoe UI,
 Roboto,sans-serif">
 ⛔ <b>SUPERSEDED — THIS SHEET READS THE WRONG HORIZON (retraction C89).</b><br>
 Every label below is read at <b>2.0&nbsp;s</b>, which is the <b>SEAM</b> where
 the operative band ends and the tactical band begins — not the tactical band.
 The binding spec is
 <code style="background:rgba(255,255,255,.16);padding:1px 5px;border-radius:3px">
 TAC_BAND_S = (2.0, 6.0)</code> in
 <code style="background:rgba(255,255,255,.16);padding:1px 5px;border-radius:3px">
 stack/tanitad/models/v6.py:136-140</code>. 2.0&nbsp;s was chosen because it was
 the <b>argmax of κ</b>, then described as "the v6 tactical band", which it is
 not.<br>
 ⭐ <b>Do not adjudicate this sheet.</b> The rebuilt one is
 <b>TACTICAL_VISUAL_REVIEW_BAND_2_6S.html</b>, read over
 <b>(2.0,&nbsp;6.0]&nbsp;s</b>. Verdicts are stored under a different key, so
 the two sheets cannot contaminate each other.<br>
 ⚠️ The κ figures printed below (<b>LON 0.3655 / LAT 0.4694</b>) are
 <b>SEAM</b> values, and they are also read at the threshold cell that
 <i>maximised</i> κ. At the true band the agreement is materially
 <b>worse</b> — see <code>raw/b1_band_agreement.json</code>.<br>
 <span style="opacity:.85">Kept, not deleted, so the shape of the error stays
 legible.</span>
</div>"""

OLD_TITLE = "<title>TanitAD — tactical label review (2.0 s horizon)</title>"
NEW_TITLE = ("<title>[SUPERSEDED · seam 2.0 s] TanitAD — tactical label "
             "review</title>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(
        _PKG / "review" / "TACTICAL_VISUAL_REVIEW.html"))
    a = ap.parse_args()
    p = Path(a.sheet)
    html = p.read_text(encoding="utf-8")

    if MARK in html:
        print(f"ALREADY_MARKED (idempotent no-op) -> {p}")
        return 0

    if html.count("<body>") != 1:
        print(f"⛔ REFUSING: expected exactly one <body>, found "
              f"{html.count('<body>')}", file=sys.stderr)
        return 2
    html = html.replace("<body>", "<body>\n" + BANNER, 1)

    n_title = html.count(OLD_TITLE)
    if n_title == 1:
        html = html.replace(OLD_TITLE, NEW_TITLE, 1)
    else:
        # ⚠️ FAIL LOUD rather than silently leaving a title that says the sheet
        # is fine. The banner still lands; the mismatch is reported.
        print(f"⚠️ title not matched exactly (found {n_title}); banner applied, "
              f"title left as-is", file=sys.stderr)

    p.write_text(html, encoding="utf-8")
    print(f"MARKED_SUPERSEDED banner+title -> {p} "
          f"({p.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
