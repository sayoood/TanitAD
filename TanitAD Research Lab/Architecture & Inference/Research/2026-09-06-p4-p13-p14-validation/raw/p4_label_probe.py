"""P4 -- the 15-token STRATEGIC vocabulary: what can actually be supervised.

ZERO GPU. Reads the banked v7 label sample and answers, with controls:
  (a) per-class coverage of g_str (8) and a_str (7)  -- NOT pooled accuracy
  (b) the MAJORITY-CLASS control: what a constant predictor scores
  (c) provenance / grounded share for the STRATEGIC tokens specifically
  (d) the t0_s band -> what fraction of usable frames carry strategic GT
  (e) THE LEAK TEST: is g_str a bijection of nav_command?  (the route-head
      echo defect: 369/369 exact, scored 1.0000, read as skill)

ASCII ONLY in print() -- cp1252 dev box.
"""
from __future__ import annotations

import collections
import json
import math
import sys

# Frozen vocabulary, transcribed from stack/tanitad/models/vocab_v7.py and
# ASSERTED against that file below (a hardcoded list nobody checks is a
# claim about my memory, not about the source).
G_STR = ("FOLLOW_ROUTE", "TURN_LEFT_FOLLOW_ROUTE", "TURN_RIGHT_FOLLOW_ROUTE",
         "STOP_AT_FOLLOW_ROUTE", "EXIT_LEFT_FOLLOW_ROUTE",
         "EXIT_RIGHT_FOLLOW_ROUTE", "LANE_CHANGE_L_FOLLOW_ROUTE",
         "LANE_CHANGE_R_FOLLOW_ROUTE")
A_STR = ("HOLD_MAIN_ROAD", "PREPARE_TURN_L_FOLLOW_ROUTE",
         "PREPARE_TURN_R_FOLLOW_ROUTE", "PREPARE_STOP_FOLLOW_ROUTE",
         "PREPARE_EXIT_FOLLOW_ROUTE", "PREPARE_LANE_CHANGE_FOLLOW_ROUTE",
         "RESUME_CRUISE_FOLLOW_ROUTE")


def assert_vocab(vocab_py: str) -> None:
    """Control: every token must be present in the source file, and the
    counts must be 8 / 7. A probe whose vocabulary drifted from source
    would report coverage for classes that do not exist."""
    with open(vocab_py, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    if len(src) < 1000:
        raise SystemExit("INCONCLUSIVE: vocab_v7.py read short (%d bytes) "
                         "-- mount flap, not a result" % len(src))
    missing = [t for t in G_STR + A_STR if ('"%s"' % t) not in src]
    if missing:
        raise SystemExit("VOCAB DRIFT: not in source: %s" % missing)
    assert len(G_STR) == 8 and len(A_STR) == 7
    print("[control] vocab asserted against source: 8 goal + 7 action = 15 "
          "tokens, all present in vocab_v7.py (%d bytes read)" % len(src))


def per_class(rows, field, tokens):
    """Per-class support + the majority-class control."""
    cnt = collections.Counter()
    prov = collections.Counter()
    n_present = 0
    for r in rows:
        d = r.get(field) or {}
        tok = d.get("token")
        if tok is None:
            continue
        n_present += 1
        cnt[tok] += 1
        prov[str(d.get("provenance"))] += 1
    return cnt, prov, n_present


def report_head(name, cnt, prov, n_present, tokens, n_rows):
    print("")
    print("=" * 74)
    print("HEAD %s -- n_rows=%d  n_with_token=%d (%.2f%%)  d=%d classes"
          % (name, n_rows, n_present, 100.0 * n_present / max(n_rows, 1),
             len(tokens)))
    print("=" * 74)
    unseen = []
    print("%-36s %7s %8s" % ("class", "n", "share"))
    for t in tokens:
        c = cnt.get(t, 0)
        if c == 0:
            unseen.append(t)
        print("%-36s %7d %7.3f%%"
              % (t, c, 100.0 * c / max(n_present, 1)))
    extra = [t for t in cnt if t not in tokens]
    for t in extra:
        print("%-36s %7d %7.3f%%   <-- OFF-VOCABULARY" % (t, cnt[t],
              100.0 * cnt[t] / max(n_present, 1)))
    pop = len(tokens) - len(unseen)
    print("-" * 74)
    print("POPULATED CLASSES: %d of %d   UNSEEN: %s"
          % (pop, len(tokens), unseen if unseen else "none"))
    # ---- the controls that must read a KNOWN value --------------------- #
    if n_present:
        maj_tok, maj_n = cnt.most_common(1)[0]
        maj = maj_n / n_present
        print("[CONTROL majority-class] constant predictor '%s' scores "
              "POOLED accuracy %.4f -- and MACRO per-class recall exactly "
              "%.4f (= 1/%d), which is the no-information value."
              % (maj_tok, maj, 1.0 / len(tokens), len(tokens)))
        print("[CONTROL uniform-random]  macro recall %.4f ; pooled acc %.4f"
              % (1.0 / len(tokens), 1.0 / len(tokens)))
        # entropy: how much there is to learn at all
        h = -sum((c / n_present) * math.log(c / n_present, 2)
                 for c in cnt.values() if c)
        print("[scale] label entropy %.4f bits of a possible %.4f "
              "(log2 %d) -- headroom ratio %.3f"
              % (h, math.log(len(tokens), 2), len(tokens),
                 h / math.log(len(tokens), 2)))
    print("[provenance] %s" % dict(prov))


def leak_test(rows, field, nav_field="nav_command"):
    """THE ECHO TEST. If g_str is a function of nav_command, a head fed nav
    at inference reports the label's own source as skill (MEASURED on the
    flagship route head: 369/369 bijection, scored 1.0000)."""
    pair = collections.Counter()
    nav_cnt = collections.Counter()
    tok_cnt = collections.Counter()
    n = 0
    for r in rows:
        d = r.get(field) or {}
        tok = d.get("token")
        nav = r.get(nav_field)
        if isinstance(nav, dict):
            nav = nav.get("token") or nav.get("command") or json.dumps(nav)
        if tok is None or nav is None:
            continue
        n += 1
        pair[(str(nav), tok)] += 1
        nav_cnt[str(nav)] += 1
        tok_cnt[tok] += 1
    if not n:
        print("[LEAK %s] INCONCLUSIVE: no (nav, token) pairs present" % field)
        return
    # functional determinism: for each nav value, how often is the modal
    # token right?  1.0000 == exact bijection == echo.
    best = collections.defaultdict(int)
    for (nav, tok), c in pair.items():
        best[nav] = max(best[nav], c)
    det = sum(best.values()) / n
    # mutual information, in bits
    hi = 0.0
    for (nav, tok), c in pair.items():
        p = c / n
        hi += p * math.log(p / ((nav_cnt[nav] / n) * (tok_cnt[tok] / n)), 2)
    h_tok = -sum((c / n) * math.log(c / n, 2) for c in tok_cnt.values() if c)
    print("")
    print("[LEAK TEST %s vs %s]  n=%d" % (field, nav_field, n))
    print("  nav->token DETERMINISM  %.4f   (1.0000 = exact bijection = ECHO)"
          % det)
    print("  mutual information      %.4f bits of H(token)=%.4f  -> %.1f%% "
          "of the label is already in nav" % (hi, h_tok,
                                              100.0 * hi / max(h_tok, 1e-9)))
    print("  distinct nav values %d ; distinct tokens %d"
          % (len(nav_cnt), len(tok_cnt)))
    if det >= 0.999:
        print("  ==> VERDICT: BIJECTION. Feeding nav at inference makes this "
              "head an ECHO. Supervise it, but nav must NOT be an input.")
    elif hi / max(h_tok, 1e-9) > 0.5:
        print("  ==> VERDICT: MAJORITY OF THE LABEL IS IN NAV. Any nav-fed "
              "arm needs a nav-ablated control before the number is quotable.")
    else:
        print("  ==> VERDICT: nav does NOT determine the token; a vision head "
              "has something real to extract.")


def band(rows):
    """The supervisable fraction. One record per clip at t0_s, valid +/- the
    tolerance -> what share of the usable horizon carries strategic GT."""
    t0 = collections.Counter()
    avail = []
    per_clip = collections.Counter()
    for r in rows:
        t0[r.get("t0_s")] += 1
        h = (r.get("horizon") or {}).get("available_s")
        if h is not None:
            avail.append(float(h))
        per_clip[r.get("clip_id")] += 1
    print("")
    print("[BAND] t0_s histogram: %s" % dict(t0))
    print("[BAND] records per clip: %s"
          % dict(collections.Counter(per_clip.values())))
    if avail:
        avail.sort()
        med = avail[len(avail) // 2]
        print("[BAND] available_s  min %.1f  median %.1f  max %.1f  n=%d"
              % (avail[0], med, avail[-1], len(avail)))
        for tol in (2.0,):
            covered = 2.0 * tol
            print("[BAND] one record per clip, valid +/- %.1f s => %.1f s of "
                  "a median %.1f s usable horizon = %.2f%% SUPERVISABLE ; "
                  "%.2f%% carries NO strategic GT"
                  % (tol, covered, med, 100.0 * covered / med,
                     100.0 * (1.0 - covered / med)))


def main(path, vocab_py):
    assert_vocab(vocab_py)
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    print("[read] %s -> %d rows (%d unparseable)" % (path, len(rows), bad))
    if len(rows) < 10:
        raise SystemExit("INCONCLUSIVE: read %d rows -- mount flap or wrong "
                         "file, not a result" % len(rows))

    for field, toks in (("g_str", G_STR), ("a_str", A_STR)):
        cnt, prov, npres = per_class(rows, field, toks)
        report_head(field, cnt, prov, npres, toks, len(rows))

    # CONTROL: the tactical head on the same rows must read NON-ZERO, or a
    # zero strategic count is a claim about my parser, not about the labels.
    tac = sum(1 for r in rows if (r.get("a_tac") or {}).get("lat"))
    print("")
    print("[CONTROL parser] a_tac.lat present on %d/%d rows (must be > 0)"
          % (tac, len(rows)))

    for field in ("g_str", "a_str"):
        leak_test(rows, field)
    band(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
