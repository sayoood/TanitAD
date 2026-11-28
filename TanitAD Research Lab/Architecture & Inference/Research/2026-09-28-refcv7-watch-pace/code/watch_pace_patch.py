"""Patch build_watch_refcv7.py: the pace is judged against the PI-APPROVED ceiling (SPEC_REFCV7 sec. 25,
10.5 s/step, 2026-09-27), not only the pre-approval 8.0 line that sent the cost to the PI. Every replacement
must match exactly once or the patch refuses (writes nothing)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
raw = p.read_bytes()
crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")

R = [
    # 1. module docstring
    ("* NEW-1's residual prior as the run records it, and the marginal pace against the 8.0 s/step line\n"
     "  (shown, not decided: the PI rule itself is applied at the G-LIVE smoke).",
     "* NEW-1's residual prior as the run records it, and the marginal pace against the PI-APPROVED ceiling\n"
     "  (10.5 s/step, SPEC_REFCV7 sec. 25, 2026-09-27) with the pre-approval 8.0 s/step line kept as a reference."),
    # 2. constant
    ("PACE_PI_LINE_S = 8.0           # 6.4 x 1.25: above it the cost goes to the PI (SPEC 6.2 item 5, 11.1, 12 item 4)\n",
     "PACE_PI_LINE_S = 8.0           # 6.4 x 1.25: above it the cost goes to the PI (SPEC 6.2 item 5, 11.1, 12 item 4)\n"
     "PACE_PI_APPROVED_S = 10.5      # the PI's answer to that item: ~9.9 s/step approved 2026-09-27, ceiling 10.5\n"
     "                               # (SPEC_REFCV7 sec. 25; the gate's pi_cost_approval.json max_s_per_step)\n"),
    # 3. the flag
    ("    pace_above = pace is not None and pace > PACE_PI_LINE_S\n",
     "    pace_above = pace is not None and pace > PACE_PI_LINE_S\n"
     "    pace_over = pace is not None and pace > PACE_PI_APPROVED_S   # the only pace state that is amber\n"),
    # 4. status chip
    ("             + (chip(not pace_above,\n"
     "                     f\"pace {pace:.2f} s/step {pace_lab} ≤ {PACE_PI_LINE_S:.1f} · warm median {fmt(warm_med, 2)}\",\n"
     "                     f\"pace {pace:.2f} s/step {pace_lab} > {PACE_PI_LINE_S:.1f} · warm median {fmt(warm_med, 2)}\",\n",
     "             + (chip(not pace_over,\n"
     "                     f\"pace {pace:.2f} s/step {pace_lab} ≤ PI-approved {PACE_PI_APPROVED_S:.1f} · warm median {fmt(warm_med, 2)}\",\n"
     "                     f\"pace {pace:.2f} s/step {pace_lab} > PI-approved {PACE_PI_APPROVED_S:.1f} · warm median {fmt(warm_med, 2)}\",\n"),
    # 5. pace chart
    ("                      (PACE_PI_LINE_S, f\"PI line {PACE_PI_LINE_S} s/step (+25 %)\")],\n"
     "                note=\"Shown, not decided: more than +25 % over refcv6's 6.4 s/step is the PI's reserved decision \"\n"
     "                     \"(SPEC_REFCV7 6.2 item 5, 12 item 4).\")\n",
     "                      (PACE_PI_LINE_S, f\"pre-approval line {PACE_PI_LINE_S} (+25 %)\"),\n"
     "                      (PACE_PI_APPROVED_S, f\"PI-approved ceiling {PACE_PI_APPROVED_S} s/step\")],\n"
     "                note=\"The PI approved ~9.9 s/step on 2026-09-27 with a 10.5 s/step ceiling (SPEC_REFCV7 25). \"\n"
     "                     \"The 8.0 line (+25 % over refcv6's 6.4) is the rule that sent the cost to the PI.\")\n"),
    # 6. health row
    ("f'without an eval or checkpoint · refcv6 {PACE_REF_S} · line {PACE_PI_LINE_S} (+25 %) · the PI rule itself is '\n"
     "              'applied at the G-LIVE smoke, not by this page</td></tr>'\n",
     "f'without an eval or checkpoint · refcv6 {PACE_REF_S} · pre-approval line {PACE_PI_LINE_S} (+25 %) · '\n"
     "              f'PI-approved ceiling {PACE_PI_APPROVED_S} (SPEC_REFCV7 25, 2026-09-27)</td></tr>'\n"),
    # 7. what it says
    ("                    + (f\"That is ABOVE the {PACE_PI_LINE_S} s/step line (+25 % over refcv6's {PACE_REF_S}), shown and \"\n"
     "                       \"not decided: the PI rule itself is applied at the G-LIVE smoke.\" if pace_above else\n"
     "                       f\"Within the {PACE_PI_LINE_S} s/step line (+25 % over refcv6's {PACE_REF_S}).\") + \"</li>\")\n",
     "                    + (f\"That is ABOVE the PI-approved {PACE_PI_APPROVED_S} s/step ceiling (SPEC_REFCV7 25): \"\n"
     "                       \"a PI item.\" if pace_over else\n"
     "                       f\"Within the PI-approved {PACE_PI_APPROVED_S} s/step ceiling (SPEC_REFCV7 25, 2026-09-27); \"\n"
     "                       f\"above the pre-approval {PACE_PI_LINE_S} line that sent the cost to the PI.\" if pace_above else\n"
     "                       f\"Within the {PACE_PI_LINE_S} s/step line (+25 % over refcv6's {PACE_REF_S}).\") + \"</li>\")\n"),
    # 8. alarm tile
    ("    pace_tile = _tile(\"warn\" if pace_above else (\"info\" if pace is None else \"good\"),\n",
     "    pace_tile = _tile(\"warn\" if pace_over else (\"info\" if pace is None else \"good\"),\n"),
    ("                       f\"line {PACE_PI_LINE_S} s/step = refcv6's {PACE_REF_S} × 1.25\"],\n"
     "                      \"Shown, not decided: amber above the line. The PI rule itself is applied at the G-LIVE smoke, \"\n"
     "                      \"not by the Watch (SPEC_REFCV7 6.2, 12)\")\n",
     "                       f\"PI-approved ceiling {PACE_PI_APPROVED_S} s/step (SPEC_REFCV7 25, 2026-09-27)\",\n"
     "                       f\"pre-approval line {PACE_PI_LINE_S} s/step = refcv6's {PACE_REF_S} × 1.25\"],\n"
     "                      \"Amber only above the PI-approved ceiling. The PI approved ~9.9 s/step with a 10.5 ceiling \"\n"
     "                      \"(SPEC_REFCV7 25, answering the reserved item in 6.2 item 5)\")\n"),
    # 9. top tile
    ("<span>per step, marginal (PI line {PACE_PI_LINE_S})</span>",
     "<span>per step, marginal (PI-approved ≤ {PACE_PI_APPROVED_S})</span>"),
    # 10. summary JSON
    ("        \"pace_above_pi_line\": pace_above, ",
     "        \"pace_above_pi_line\": pace_above, \"pace_over_pi_approval\": pace_over, "),
]
for old, new in R:
    n = s.count(old)
    if n != 1:
        sys.exit(f"REFUSED: {n} matches for {old[:70]!r}")
    s = s.replace(old, new)
out = s.replace("\n", "\r\n") if crlf else s
p.write_bytes(out.encode("utf-8"))
print("PATCHED", len(R), "replacements", "crlf" if crlf else "lf")
