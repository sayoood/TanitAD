"""Batch 2 / D3 -- the per-head gradient-reach row (`ga_*`) reaches metrics.jsonl at the log
cadence, the run DECLARES what it logs, and the first log row is held against the declaration.

    python patch_d3.py <fix2 root>

Edits (each anchored, each anchor must occur exactly once):
  stack/scripts/refc_v3_train.py (CRLF, preserved):
    E1 helpers `_logged_after`, `_grad_reach_declared`, `_grad_reach_declaration` before
       `_grad_probe_row`
    E2 config.json key `grad_reach_logging` after the `declared_vs_built` block
    E3 `_gp_row` uses `_logged_after` (same semantics, one definition)
    E4 the `ga_*` fill uses `_logged_after` (THE FIX) and `_grad_reach_declared`
    E5 `_d3_checked` initialised beside `_bev_parity_checked`
    E6 the first regular log row is held against the declaration (SystemExit on mismatch)
  stack/tanitad/train/declared_vs_built.py (LF):
    E7 `check_logged_rows(config, rows) -> list[Mismatch]`
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def edit(s: str, old: str, new: str, tag: str) -> str:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch_d3] {tag}: anchor found {n} times")
    return s.replace(old, new)


# ------------------------------------------------------------------ the trainer (CRLF)
tp = ROOT / "stack" / "scripts" / "refc_v3_train.py"
raw = tp.read_bytes()
assert raw.count(b"\r\n") == raw.count(b"\n"), "refc_v3_train.py is expected to be pure CRLF"
s = raw.decode("utf-8").replace("\r\n", "\n")

s = edit(s, r'''def _grad_probe_row(model, names, log_every_hit: bool = True) -> dict:
''', r'''def _logged_after(step: int, log_every: int, steps: int) -> bool:
    """True when the step ABOUT TO RUN (the PRE-increment ``step``) is one the loop writes to
    metrics.jsonl. The loop logs AFTER ``step += 1``, on ``step % log_every == 0 or
    step == steps``, so a row FILLED before the increment has to test ``step + 1``.

    ⛔⛔ D3 (2026-09-26, MEASURED by the map-signal audit): the per-head gradient-reach row was
    filled on ``step % log_every == 0`` -- the pre-increment step -- and written after the
    increment, so fill and write never met: **0 `ga_*` keys in refcv6-r101-s0's 4,621 metrics
    rows** (only a run's FINAL row, via the ``steps`` clause, could ever carry one). Every
    instrument that fills a row before the increment uses THIS one rule, so there is a single
    definition of "this step is logged" (pinned: tests/test_grad_reach_logged.py).
    """
    n = int(step) + 1
    return n % max(1, int(log_every)) == 0 or n >= int(steps)


def _grad_reach_declared(model) -> bool:
    """Whether the run REPORTS per-head gradient reach (``ga_*``): a perception branch or the
    refcv6 tactical decoder is built (PI RULING 2026-09-17 R3). ONE definition, used by the loop
    AND by the config.json declaration, so the two cannot drift apart."""
    return (getattr(model, "_perception", None) is not None
            or getattr(model, "tac_decoder_v6", None) is not None)


def _grad_reach_declaration(model, args) -> dict:
    """⛔ D3: what the run DECLARES it will log, written into config.json BEFORE step 1, so its
    metrics can be held against it (`declared_vs_built.check_logged_rows`; the loop runs that
    check itself on its FIRST log row and refuses). The key set is read off the BUILT model --
    the parts `grad_reach_report` names -- never written down here."""
    on = _grad_reach_declared(model)
    keys = sorted(k for part in (_perc.grad_reach_report(model) if on else {})
                  for k in (f"ga_{part}", f"ga_{part}_n"))
    return {"declared": bool(on), "keys": keys,
            "cadence": "every metrics row: step % log_every == 0 or step == steps",
            "log_every": int(args.log_every),
            "source": "refcv6_perception_branch.grad_reach_report, read off the BUILT model"}


def _grad_probe_row(model, names, log_every_hit: bool = True) -> dict:
''', "E1 helpers")

s = edit(s, r'''                              "module": "tanitad/train/declared_vs_built.py"},
''', r'''                              "module": "tanitad/train/declared_vs_built.py"},
        # ⛔ D3: what this run declares it LOGS -- held against metrics.jsonl at the first row
        "grad_reach_logging": _grad_reach_declaration(model, args),
''', "E2 declaration")

s = edit(s, r'''            log_every_hit=bool(_gp_names) and (
                (step + 1) % args.log_every == 0
                or (step + 1) >= args.steps))
''', r'''            log_every_hit=bool(_gp_names) and _logged_after(
                step, args.log_every, args.steps))
''', "E3 gp cadence")

s = edit(s, r'''        _ga_on = (getattr(model, "_perception", None) is not None
                  or getattr(model, "tac_decoder_v6", None) is not None)
        if _ga_on and (
                step % max(1, args.log_every) == 0 or step + 1 >= args.steps):
''', r'''        _ga_on = _grad_reach_declared(model)
        # ⛔⛔ D3 (2026-09-26): `_logged_after` tests the POST-increment step -- the one the log
        # below writes. The old `step % log_every == 0` filled pre-steps 0, 50, 100 ... that are
        # written as steps 1, 51, 101 ... -- never a log step: 0 `ga_*` keys in refcv6's 4,621 rows.
        if _ga_on and _logged_after(step, args.log_every, args.steps):
''', "E4 ga cadence (THE FIX)")

s = edit(s, r'''    _bev_parity_checked = False        # WP-D: the E-DEC-18b gate fires once
''', r'''    _bev_parity_checked = False        # WP-D: the E-DEC-18b gate fires once
    _d3_checked = False                # D3: declared-vs-LOGGED, held at the first log row
''', "E5 d3 flag")

s = edit(s, r'''            if _cd_row:
                row.update(_cd_row)
            log.write(json.dumps(row) + "\n")
            log.flush()
''', r'''            if _cd_row:
                row.update(_cd_row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            # ⛔⛔ D3: a DECLARED instrument must WRITE. The first log row is held against
            # config.json's own `grad_reach_logging` declaration (the row is written first, so
            # the evidence stays in metrics.jsonl), and a dead instrument costs `log_every`
            # steps instead of a run (refcv6-r101-s0: 0 `ga_*` keys in 4,621 rows).
            if not _d3_checked:
                _d3_checked = True
                _d3_bad = _dvb.check_logged_rows(_run_config, [row])
                if _d3_bad:
                    raise SystemExit(
                        "[v3] ⛔ D3: a declared instrument did not reach metrics.jsonl at "
                        "the first log row (step %d):\n  - %s"
                        % (step, "\n  - ".join(str(m) for m in _d3_bad)))
''', "E6 first-row check")

tp.write_bytes(s.replace("\n", "\r\n").encode("utf-8"))

# ------------------------------------------------------------------ G-DVB (LF)
dp = ROOT / "stack" / "tanitad" / "train" / "declared_vs_built.py"
d = dp.read_text(encoding="utf-8")
assert "\r\n" not in d
d = edit(d, r'''            f"D-REFCV6-F3-WHITELIST, D-REFCV6-CONFIG-BUILD):\n  - "
            + "\n  - ".join(str(b) for b in bad))
''', r'''            f"D-REFCV6-F3-WHITELIST, D-REFCV6-CONFIG-BUILD):\n  - "
            + "\n  - ".join(str(b) for b in bad))


# ============================================================================
# D3 (2026-09-26): declared vs LOGGED -- the same defect class, one step later
# ============================================================================

def check_logged_rows(config: dict, rows) -> list[Mismatch]:
    """-> a Mismatch per key the run DECLARED it logs (config.json ``grad_reach_logging``) that a
    regular metrics row (one carrying ``loss``) does NOT carry, and per ``ga_*`` key in any row of
    a run that declared none ([] == the rows carry exactly what config.json declared).

    ⛔⛔ An instrument that is BUILT and DECLARED but never WRITES is the declared-vs-built defect
    one step later. MEASURED 2026-09-26 by the map-signal audit: refcv6-r101-s0 reports per-head
    gradient reach, and **0 of its 4,621 metrics rows carry a `ga_*` key** -- the row was filled
    on the pre-increment step and written on the post-increment one. Run this on a launch
    smoke's ``metrics.jsonl``; the trainer runs it itself on its FIRST log row and refuses.
    ⚠️ A config.json with NO declaration (a pre-D3 trainer) is itself a Mismatch: nothing in it
    says what should have been logged, so nothing can be shown to have been.
    """
    dec = (config or {}).get("grad_reach_logging")
    if not isinstance(dec, dict):
        return [Mismatch("grad_reach_logging", "a declaration in config.json", None,
                         "config.json", "pre-D3 trainer: nothing says what the run logs")]
    keys = [str(k) for k in (dec.get("keys") or [])]
    regular = [r for r in rows if isinstance(r, dict) and "loss" in r]
    out: list[Mismatch] = []
    if dec.get("declared"):
        if not keys:
            out.append(Mismatch("grad_reach_logging.keys", "the ga_* keys of the built heads",
                                [], "config.json", "declared ON with an empty key list"))
        if not regular:
            out.append(Mismatch("grad_reach_logging", "at least one metrics row", 0,
                                "metrics.jsonl",
                                "no regular (loss-carrying) row to hold the declaration against"))
        for r in regular:
            miss = [k for k in keys if k not in r]
            if miss:
                out.append(Mismatch("grad_reach_logging", keys, [k for k in keys if k in r],
                                    f"metrics.jsonl row step={r.get('step')}",
                                    f"declared ga_* keys MISSING: {miss}"))
    else:
        for r in rows:
            extra = sorted(k for k in (r if isinstance(r, dict) else {})
                           if str(k).startswith("ga_"))
            if extra:
                out.append(Mismatch("grad_reach_logging", "no ga_* key (declared OFF)", extra,
                                    f"metrics.jsonl row step={r.get('step')}",
                                    "a key the run did not declare: an OFF arm's schema must "
                                    "equal the pre-instrument trainer's"))
    return out
''', "E7 check_logged_rows")
dp.write_text(d, encoding="utf-8", newline="\n")
print("patched:", tp.name, dp.name)
