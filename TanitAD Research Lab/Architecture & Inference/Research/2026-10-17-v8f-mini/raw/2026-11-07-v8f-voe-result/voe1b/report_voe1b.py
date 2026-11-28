"""PREREG_V8F_VOE1B step 3: checks controls 1 to 4 FIRST, then writes RESULT_VOE1B.md and fig/*.png from RESULT_VOE1B.json and traces_voe1b.npz.
Judges with the rules of IMPL_CHOICES_VOE1B.md (and VOE1's, which it reuses); computes no new model statistic. A failed control prints INVALID, the failed controls and the cheapest
next lever, and NO label table (there is no --read-invalid mode: the card says an invalid run is not read).
Run: python report_voe1b.py [--dir D]   (D defaults to this directory; matplotlib, numpy only)"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voe1b_core as C  # noqa: E402

DECIDING = ("M2b_s0", "M2b_s1")
LV = ("1", "2", "3")
STEP_S = {"1": 0.2, "2": 0.4, "3": 0.8}
EXPECTED_TESTS = 22                       # VOE1's 13 + the 9 VOE1B tests (test_voe1b_synth.py)
CARD_PREFIX_BYTES = 5785
CODE_FILES = ("voe1b_core.py", "voe1b_real.py", "build_bank_voe1b.py", "counts_voe1b.py", "run_voe1b.py", "test_voe1b_synth.py")
C2_MANIPS = ("FREEZE", "ACTION-SWAP", "ACTION-SWAP-M")
OUT = HERE


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def cell(r, d=3, sgn=False):
    """point [lo, hi] plus S (two-part separated above) / b (bootstrap only) / - ."""
    f = (lambda x: f"{x:+.{d}f}") if sgn else (lambda x: f"{x:.{d}f}")
    flag = "S" if r.get("sep_above") else ("b" if r.get("boot_above") else "-")
    if r.get("sep_below"):
        flag = "S-below"
    return f"{f(r['point'])} [{f(r['lo'])}, {f(r['hi'])}] {flag}"


def ut_tags(R, c):
    return [t for t in R["ut_tags"] if t.startswith(f"UT{c}_")]


def ut_points(R, c, man, l):
    return [R["tags"][t]["manips"][man]["levels"][l]["acc"]["point"] for t in ut_tags(R, c) if R["tags"][t]["manips"][man]["levels"].get(l)]


def compute_labels(R):
    """{(man, l, seed): label dict} for the two trained seeds, and {(man, l): over-both-seeds label}."""
    lab, over = {}, {}
    for man in C.MANIPS:
        for l in LV:
            for s, tag in enumerate(DECIDING):
                rec = R["tags"][tag]["manips"][man]
                if rec["n"] == 0 or l not in rec["levels"]:
                    continue
                r = rec["levels"][l]
                lab[(man, l, s)] = C.label_cell_b(r["acc"], r["diff"], man in C.NULL_COUNTS, ut_points(R, s, man, l))
            if (man, l, 0) in lab and (man, l, 1) in lab:
                over[(man, l)] = C.level_label_b(lab[(man, l, 0)]["label"], lab[(man, l, 1)]["label"])
    return lab, over


def main(D):
    R = json.load(open(os.path.join(D, "RESULT_VOE1B.json"), encoding="utf-8"))
    meta, tags, events = R["meta"], R["tags"], R["events"]
    smoke = bool(meta.get("smoke"))
    md = []
    P = md.append
    ctrl, fails = {}, []
    n_ut = len(ut_tags(R, 0))

    # ------------------------------------------------------------------ integrity
    card = os.path.join(HERE, "..", "PREREG_V8F_VOE1B.md")
    integ = {"card_prefix_ok": hashlib.sha256(open(card, "rb").read()[:CARD_PREFIX_BYTES]).hexdigest() == meta["card_prefix_sha256"],
             "impl_ok": sha_file(os.path.join(HERE, "IMPL_CHOICES_VOE1B.md")) == meta["impl_sha256"] == open(os.path.join(HERE, "IMPL_CHOICES_VOE1B.sha256")).read().split()[0],
             "code_unchanged_since_run": {f: sha_file(os.path.join(HERE, f)) == h for f, h in meta["code_sha256"].items()},
             "bank_sha_equals_impl": bool(meta.get("bank_sha_equals_impl")), "n_untrained_per_chassis": n_ut}
    if not (integ["card_prefix_ok"] and integ["impl_ok"]):
        fails.append("integrity: card prefix or IMPL_CHOICES hash differs from the run's")
    ctrl["integrity"] = integ

    # ------------------------------------------------------------------ control 4 (pytest log)
    log = open(os.path.join(HERE, "PYTEST_VOE1B.log"), encoding="utf-8").read()
    hs = {}
    for line in open(os.path.join(HERE, "PYTEST_VOE1B.hashes")).read().strip().splitlines():
        h, n = line.split()
        hs[n.lstrip("*")] = h
    m = re.search(r"(\d+) passed", log)
    c4 = {"passed": int(m.group(1)) if m else 0, "failed": re.search(r"\d+ failed", log) is not None or re.search(r"\d+ error", log) is not None,
          "test_file_sha_matches_log": hs.get("test_voe1b_synth.py") == sha_file(os.path.join(HERE, "test_voe1b_synth.py")),
          "core_sha_matches_log": hs.get("voe1b_core.py") == sha_file(os.path.join(HERE, "voe1b_core.py"))}
    c4["pass"] = bool(c4["passed"] >= EXPECTED_TESTS and not c4["failed"] and c4["test_file_sha_matches_log"] and c4["core_sha_matches_log"])
    ctrl["c4"] = c4
    if not c4["pass"]:
        fails.append(f"control 4: {c4}")

    # ------------------------------------------------------------------ control 1: calibrated real-vs-real count (M2b, 6 cells) against the 95th percentile of the reference-draw counts
    cells1 = [tags[t]["rvr"]["levels"][l] for t in DECIDING for l in LV]
    c1 = C.control1_verdict(cells1)
    c1["cells"] = {f"{t}/L{l}": {"point": tags[t]["rvr"]["levels"][l]["point"], "lo": tags[t]["rvr"]["levels"][l]["lo"], "hi": tags[t]["rvr"]["levels"][l]["hi"],
                                 "separated_any": tags[t]["rvr"]["levels"][l]["separated_any"], "n": tags[t]["rvr"]["levels"][l]["n"]} for t in DECIDING for l in LV}
    ut_rvr = {t: [bool(tags[t]["rvr"]["levels"][l]["separated_any"]) for l in LV] for t in R["ut_tags"]}
    c1["untrained_beside"] = {"cells": 3 * len(ut_rvr), "separated": int(sum(sum(v) for v in ut_rvr.values())), "tags_with_a_separated_cell": [t for t, v in ut_rvr.items() if any(v)]}
    c1["pass"] = bool(c1["pass"])
    ctrl["c1"] = c1
    if not c1["pass"]:
        fails.append(f"control 1: {c1['real_count']} of 6 M2b real-vs-real cells separated, above the reference 95th percentile {c1['p95_linear']:.2f}")

    # ------------------------------------------------------------------ control 2: untrained leave-one-out NOTICES share (at most 10 %)
    c2_tab, tot, hit = {}, 0, 0
    for c in (0, 1):
        for man in C2_MANIPS:
            for l in LV:
                rows = [tags[t]["manips"][man]["levels"].get(l) for t in ut_tags(R, c)]
                rows = [r["acc"] for r in rows if r]
                if len(rows) < 2:
                    continue
                n_ = int(sum(C.loo_notices(rows)))
                c2_tab[f"UT{c}/{man}/L{l}"] = {"notices": n_, "members": len(rows)}
                tot += len(rows)
                hit += n_
    share = hit / max(tot, 1)
    by_c = {c: (sum(v["notices"] for k, v in c2_tab.items() if k.startswith(f"UT{c}/")), sum(v["members"] for k, v in c2_tab.items() if k.startswith(f"UT{c}/"))) for c in (0, 1)}
    by_m = {m_: (sum(v["notices"] for k, v in c2_tab.items() if k.split("/")[1] == m_), sum(v["members"] for k, v in c2_tab.items() if k.split("/")[1] == m_)) for m_ in C2_MANIPS}
    c2 = {"cells": tot, "notices": hit, "share": share, "limit": 0.10, "expected_max_share": 1 / max(n_ut, 1), "by_chassis": by_c, "by_manipulation": by_m, "table": c2_tab, "pass": bool(share <= 0.10)}
    ctrl["c2"] = c2
    if not c2["pass"]:
        fails.append(f"control 2: {hit} of {tot} untrained leave-one-out cells NOTICES = {share:.3%} (limit 10 %)")

    # ------------------------------------------------------------------ control 3: calibration shares per family
    fam = {k: [0, 0] for k in ("ACC", "NULL-ACC", "DIFF", "RVR", "AP-CHANCE", "AP-NULL")}
    for tag in DECIDING + tuple(R["ut_tags"]):
        for l in LV:
            c = tags[tag]["rvr"]["levels"][l]["cal"]
            fam["RVR"][0] += sum(a or b for a, b in zip(c["sep_above"], c["sep_below"])); fam["RVR"][1] += len(c["sep_above"])
        for man in C.MANIPS:
            for l in LV:
                r = tags[tag]["manips"][man]["levels"].get(l)
                if not r:
                    continue
                for f, k in (("ACC", "acc"), ("NULL-ACC", "null_acc"), ("DIFF", "diff")):
                    if k in r:
                        fam[f][0] += sum(r[k]["cal"]["sep_above"]); fam[f][1] += len(r[k]["cal"]["sep_above"])
    for tag in DECIDING:
        for ev in C.EVENTS:
            for l in LV:
                rows = events[tag][ev]["L" + l]["rows"]
                for f, k in (("AP-CHANCE", "ap_minus_chance"), ("AP-NULL", "ap_minus_null")):
                    fam[f][0] += sum(rows[k]["cal_sep_above"]); fam[f][1] += len(rows[k]["cal_sep_above"])
    c3 = {f: {"separated": a, "rows": b, "share": a / max(b, 1), "pass": a / max(b, 1) <= 0.05} for f, (a, b) in fam.items()}
    ctrl["c3"] = {"families": c3, "pass": all(v["pass"] for v in c3.values())}
    if not ctrl["c3"]["pass"]:
        fails.append("control 3: " + ", ".join(f"{f} {v['share']:.3f}" for f, v in c3.items() if not v["pass"]))

    valid = not fails
    ctrl["valid"] = valid
    ctrl["fails"] = fails
    if smoke and "--smoke-force-tables" in sys.argv:          # plumbing test of the tables and figures on a SMOKE run only; never honoured for the card's run
        valid = True
        ctrl["smoke_forced_tables"] = True
    json.dump(ctrl, open(os.path.join(D, "CONTROLS_VOE1B.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ------------------------------------------------------------------ the markdown: controls first
    P("# RESULT_VOE1B: does each level notice an impossible future beyond what an untrained network does? (violation of expectation, calibrated reference; tier T0)")
    P("Card `../PREREG_V8F_VOE1B.md` (frozen prefix 5,785 bytes, sha256 `aa317fe3...`, unedited). Definitions: `IMPL_CHOICES_VOE1B.md` (sha256 `%s`, deltas) on top of VOE1's `IMPL_CHOICES_VOE1.md` (`848a2c8c`). "
      "Models: M2b s0 and s1, zero training; reference: %d untrained world models per chassis (`torch.manual_seed` 100..%d; every wm.* parameter re-initialised). %s windows of eval-141, onset t0 = window frame 4, "
      "1.6 s = 8 cache steps scored. **Tier T0** (a world-model diagnostic on cached latents, teacher-forced one-step prediction; never a driving claim). **STRATEGIC family: PAUSED (PI ruling, `PI_DECISION_QUEUE.md` item 25); "
      "nothing here is a strategic result.** Every number below is computed by `run_voe1b.py` into `RESULT_VOE1B.json` (run %s to %s, %.0f s)." % (
          meta["impl_sha256"][:8], n_ut, 99 + n_ut, f"{R['window_table']['TELEPORT']['n_windows']:,}", meta["started"], meta["finished"], meta["seconds"]))
    if smoke:
        P("\n> **SMOKE RUN (plumbing test on %s episodes, %d untrained per chassis): not the card's run, not a result.**" % (meta.get("max_episodes") or "all", n_ut))
    P("\n## 0. Controls (checked before any row is read)")
    P(f"**Run status: {'VALID' if valid else 'INVALID'}**" + ("" if valid else " -- " + "; ".join(fails)) + (" (SMOKE: tables forced for a plumbing test; controls failed: " + "; ".join(fails) + ")" if (valid and fails) else ""))
    P("\n| control | rule (card) | result | pass |\n|---|---|---|---|")
    P(f"| 1 real-vs-real, calibrated | count of separated cells among the 6 M2b cells (2 seeds x 3 levels; later copy 0.2 s later) must not exceed the 95th percentile of the same count over the R = 199 reference draws scored as if real | "
      f"real count **{c1['real_count']}** of 6; reference-draw counts: mean {c1['ref_mean']:.2f}, 95th percentile {c1['p95_linear']:.2f} (inverted-cdf {c1['p95_inverted_cdf']:.0f}), histogram {c1['ref_count_hist']} | {'PASS' if c1['pass'] else 'FAIL'} |")
    P(f"| 2 untrained leave-one-out | each untrained seed in turn is 'the model' (rule (a) at that seed plus rule (c) against the other {n_ut - 1} of its chassis); over all (chassis, seed, manipulation, level) cells of FREEZE, ACTION-SWAP, ACTION-SWAP-M at most 10 % NOTICES | "
      f"**{hit} of {tot} = {share:.2%}** NOTICES (expected at most 1/{n_ut} = {1 / max(n_ut, 1):.2%}); by chassis {by_c[0][0]}/{by_c[0][1]} and {by_c[1][0]}/{by_c[1][1]}; by manipulation " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_m.items()) + f" | {'PASS' if c2['pass'] else 'FAIL'} |")
    P("| 3 calibration | 20 extra permutation draws (1001-1020) scored as if real: at most 5 % of rows separated per family | " +
      "; ".join(f"{f} {v['separated']}/{v['rows']} = {v['share']:.3%}" for f, v in c3.items()) + f" | {'PASS' if ctrl['c3']['pass'] else 'FAIL'} |")
    P(f"| 4 synthetic test | `test_voe1b_synth.py`: planted toy NOTICES on FREEZE and TELEPORT (with rule (c)), action-blind toy BLIND on ACTION-SWAP, **a toy family biased in random directions gives INIT-LEVEL not NOTICES (added)**, plus VOE1's instrument tests; {EXPECTED_TESTS} tests | "
      f"{c4['passed']} passed; test file and core hash equal the log's: {c4['test_file_sha_matches_log'] and c4['core_sha_matches_log']} | {'PASS' if c4['pass'] else 'FAIL'} |")
    P("\nControl 1 cells (accuracy of 'the later copy scores higher', two-part rule, either side):\n\n| cell | n | accuracy [interval] | separated either side |\n|---|---|---|---|")
    for k, v in c1["cells"].items():
        P(f"| {k} | {v['n']} | {v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] | {v['separated_any']} |")
    ub = c1["untrained_beside"]
    P(f"\nBeside (gates nothing): the untrained real-vs-real cells, {ub['separated']} of {ub['cells']} separated on either side" + (f" ({', '.join(ub['tags_with_a_separated_cell'])})." if ub["tags_with_a_separated_cell"] else "."))
    P("\nControl 2 detail (NOTICES members of the leave-one-out, per chassis x manipulation x level; each cell has %d members):\n\n| chassis | manipulation | L1 | L2 | L3 |\n|---|---|---|---|---|" % n_ut)
    for c in (0, 1):
        for man in C2_MANIPS:
            P(f"| UT{c} | {man} | " + " | ".join(f"{c2_tab[f'UT{c}/{man}/L{l}']['notices']}" if f"UT{c}/{man}/L{l}" in c2_tab else "-" for l in LV) + " |")
    rp = R["reinit"]["ut"]
    t0_ = next(iter(rp.values()))
    P("\nReproduction (MEASURED): the unmodified path (per-clip frame bank + adapter + wm.z1) vs `latents.pt` `Z1`, 1,976 windows: max abs difference "
      f"{R['repro']['M2b_s0']['max_abs_diff_z1']:.2e} (s0), {R['repro']['M2b_s1']['max_abs_diff_z1']:.2e} (s1) on values up to {R['repro']['M2b_s0']['ref_abs_max']:.1f}; logged actions {R['repro']['actions_max_abs_diff']}. "
      f"Untrained family: {len(rp)} models, each re-initialised {t0_['n_params_reinitialised']:,} of {t0_['n_params_total']:,} wm.* parameters ({t0_['n_tensors']} tensors, none equal to the trained one); chassis (adapter, pos) trained.")
    if not valid:
        P("\n**INVALID: no label table is printed. The failing controls above are the finding; the run is not read (card: the run is INVALID as committed). Raw rows stay in `RESULT_VOE1B.json`.**")
        P("\n### Cheapest next lever (chosen from the control rows only; the M2b manipulation rows were not read to choose it)")
        for f in fails:
            if f.startswith("control 1"):
                P("* Control 1: the real-vs-real count exceeds what null draws produce. Re-run the real-vs-real cells with the later copy 0.4, 0.6, 0.8 and 1.0 s later (control rows only, forward passes only, about 1 minute) to separate a window-overlap or step-phase effect from a model effect; then re-card the control.")
            elif f.startswith("control 2"):
                P("* Control 2: untrained members are NOTICES more often than exchangeability allows, i.e. the 16 are not exchangeable (a shared chassis or architecture moves them together). Cheapest levers: (i) 16 more untrained seeds (116..131, forward passes only, about 3 minutes) and a rank rule among 32 instead of the maximum of 16 (needs a card); (ii) look at which manipulation and chassis carry the NOTICES (table above) before touching anything.")
            elif f.startswith("control 3"):
                P("* Control 3: the two-part rule's own false-positive share exceeds 5 % in a family; the instrument, not the model, is at fault. Lever: widen the reference (R above 199) for that family and re-run control 3 only.")
            elif f.startswith("control 4") or f.startswith("integrity"):
                P("* " + f + ": fix the instrument or restore the frozen files, re-run the tests (`pytest -q test_voe1b_synth.py`) and the report; the run's numbers are not touched.")
        open(os.path.join(D, "RESULT_VOE1B.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
        print("INVALID:", fails)
        return

    # ---------------------------------------------------------------- 1 headline labels
    lab, over = compute_labels(R)
    lab_json = {f"{m_}/L{l}/s{s}": v for (m_, l, s), v in lab.items()}
    lab_json.update({f"{m_}/L{l}/over_both_seeds": v for (m_, l), v in over.items()})
    json.dump(lab_json, open(os.path.join(D, "LABELS_VOE1B.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    wtab = R["window_table"]
    under = wtab["ACTION-SWAP-M"]["n_windows"] < C.SWM_MIN_N
    P("\n## 1. Headline: the label of each (manipulation, level) over both seeds")
    P("NOTICES needs all three of (a) accuracy two-part separated above 0.5, (b) [TELEPORT, REVERSE, SPLICE] separated above the Copy-Last null, (c) accuracy above the MAXIMUM of the %d untrained accuracies of that chassis, manipulation and level, at BOTH seeds. "
      "INIT-LEVEL = (a), (b) hold, (c) fails (an untrained network of the same architecture does this too). NULL-ONLY = (a) holds, (b) fails. BLIND = (a) fails." % n_ut)
    P("\n| manipulation | n windows (episodes) | level 1 (0.2 s) | level 2 (0.4 s) | level 3 (0.8 s) |\n|---|---|---|---|---|")
    for man in C.MANIPS:
        n = tags[DECIDING[0]]["manips"][man]
        nm = f"{man}" + (" **UNDERPOWERED (n < 300)**" if man == "ACTION-SWAP-M" and under else "")
        P(f"| {nm} | {n['n']:,} ({n['n_episodes']}) | " + " | ".join(over.get((man, l), "n/a") for l in LV) + " |")
    P("\nPer-seed labels (s0 / s1): " + "; ".join(f"{man}: " + ", ".join(f"L{l} {lab[(man, l, 0)]['label']}/{lab[(man, l, 1)]['label']}" for l in LV if (man, l, 0) in lab) for man in C.MANIPS))
    P("\nPer-seed conditions (a)(b)(c) of every cell are in the tables of section 2; `LABELS_VOE1B.json` holds them machine-readably.")

    # ---------------------------------------------------------------- 2 tables per manipulation
    P("\n## 2. Pairwise accuracy per manipulation x level x seed, beside the untrained reference")
    P("Accuracy = share of pairs where the manipulated copy's max normalised surprise over the 1.6 s after t0 is above its unmodified copy's (ties 0.5); interval = episode-cluster bootstrap (B 2,000, seed 0); "
      "flag S = two-part separated above (interval above the claim AND point above the 97.5th percentile of the R = 199 whole-episode label-swap reference), b = bootstrap interval only, - = neither. "
      f"The untrained columns are the {n_ut} accuracies (point values) of the untrained models on the SAME chassis, manipulation and level: min / median / max, and how many of them are themselves separated above 0.5. "
      "(a)(b)(c): T = holds, F = fails, . = not applicable. Copy-Last null accuracy and the paired model-minus-null difference (claim 0) are beside.")
    for man in C.MANIPS:
        nn = tags[DECIDING[0]]["manips"][man]
        P(f"\n### {man}" + (f"  (n = {nn['n']:,} windows, {nn['n_episodes']} episodes" + (", UNDERPOWERED" if man == "ACTION-SWAP-M" and under else "") + ")")
          + ("  [the Copy-Last null enters NOTICES]" if man in C.NULL_COUNTS else "  [the null does not enter]"))
        P("| tag | level | n (episodes) | accuracy [interval] | untrained min / median / max (separated above 0.5) | (a)(b)(c) | Copy-Last null acc | model minus null [interval] | label |\n|---|---|---|---|---|---|---|---|---|")
        for s, tag in enumerate(DECIDING):
            rec = tags[tag]["manips"][man]
            for l in LV:
                if (man, l, s) not in lab:
                    continue
                r, lb = rec["levels"][l], lab[(man, l, s)]
                nsep = sum(1 for t in ut_tags(R, s) if tags[t]["manips"][man]["levels"][l]["acc"]["sep_above"])
                abc = "".join("." if v is None else ("T" if v else "F") for v in (lb["a"], lb["b"], lb["c"]))
                P(f"| {tag} | L{l} | {rec['n']:,} ({rec['n_episodes']}) | {cell(r['acc'])} | {lb['ut_min']:.3f} / {lb['ut_median']:.3f} / {lb['ut_max']:.3f} ({nsep} of {lb['n_ut']}) | {abc} | "
                  f"{r['null_acc']['point']:.3f} [{r['null_acc']['lo']:.3f}, {r['null_acc']['hi']:.3f}] | {cell(r['diff'], sgn=True)} | {lb['label']} |")

    # ---------------------------------------------------------------- 3 events
    P("\n## 3. Event alignment (IA-6d, VOE1's definitions, unchanged; reported beside, no bar): do the surprise peaks sit within +-0.4 s of braking, turn and lead-change events?")
    ev_n = R["event_onsets_in_bank"]
    P(f"Unmodified 16-frame windows, every (window, step) point of a level pooled; positive = an event onset within +-0.4 s (+-2 cache steps) of the step. Onsets in the 141 eval clips: brake {ev_n['brake']}, turn {ev_n['turn']}, lead {ev_n['lead']}. "
      "AP = average precision of ranking points by surprise; chance = the event rate; null AP = the same ranking by the Copy-Last distance of the level. S = two-part separated above 0 (bootstrap interval AND point above the 97.5th percentile of R = 199 label-permutation draws). "
      f"The last columns give, for the {n_ut} untrained models of the same chassis, the min / median / max of AP minus chance (no interval).")
    aligns = {}
    for ev in C.EVENTS:
        P(f"\n### {ev}\n\n| level | points (positives) | seed | AP | chance | null AP | AP minus chance [interval] | AP minus null AP [interval] | untrained AP minus chance (min / median / max) | ALIGNS (both seeds) |\n|---|---|---|---|---|---|---|---|---|---|")
        for l in LV:
            ok_both = True
            for t in DECIDING:
                r = events[t][ev]["L" + l]
                ok_both &= bool(r["rows"]["ap_minus_chance"]["sep_above"] and r["rows"]["ap_minus_null"]["sep_above"])
            aligns[(ev, l)] = ok_both
            for i, t in enumerate(DECIDING):
                r = events[t][ev]["L" + l]
                a, b = r["rows"]["ap_minus_chance"], r["rows"]["ap_minus_null"]
                u = np.array([events[x][ev]["L" + l]["ap"] - events[x][ev]["L" + l]["chance"] for x in ut_tags(R, i)])
                P(f"| {'L' + l if i == 0 else ''} | {r['n_points']:,} ({r['n_events_pts']}) | {t[-2:]} | {r['ap']:.4f} | {r['chance']:.4f} | {r['ap_null']:.4f} | {cell(a, 4, True)} | {cell(b, 4, True)} | "
                  f"{u.min():+.4f} / {np.median(u):+.4f} / {u.max():+.4f} | {('**ALIGNS**' if ok_both else 'no') if i == 0 else ''} |")
    P("\n**ALIGNS summary:** " + "; ".join(f"{ev}: " + (", ".join(f"L{l}" for l in LV if aligns[(ev, l)]) or "no level") for ev in C.EVENTS) + ".")
    P("\n**Hierarchy hypothesis (HYPOTHESIS, no bar):** (AP - chance at L3) minus (AP - chance at L1), episode-cluster interval from the shared bootstrap draws:")
    P("| event | seed | L1 AP-chance | L3 AP-chance | L3 minus L1 [interval] |\n|---|---|---|---|---|")
    for ev in C.EVENTS:
        for t in DECIDING:
            e1, e3 = events[t][ev]["L1"], events[t][ev]["L3"]
            d = np.asarray(e3["boot_apc"]) - np.asarray(e1["boot_apc"])
            pt = (e3["ap"] - e3["chance"]) - (e1["ap"] - e1["chance"])
            P(f"| {ev} | {t[-2:]} | {e1['ap'] - e1['chance']:+.4f} | {e3['ap'] - e3['chance']:+.4f} | {pt:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] |")

    # ---------------------------------------------------------------- 4 triggers
    noticing = ("BLIND", "INIT-LEVEL", "BLIND/INIT-LEVEL")
    ev_all = lambda man, ok: all(over.get((man, l)) in ok for l in LV)
    imag1 = os.path.join(HERE, "..", "imag1", "RESULT_IMAG1.md")
    ia = "IMAG1 result file present (`../imag1/RESULT_IMAG1.md`): read it for IA-3 and IA-2" if os.path.exists(imag1) else "IMAG1 has no result file yet: IA-3 / IA-2 status OPEN"
    t_am = (not under) and ev_all("ACTION-SWAP-M", noticing)
    t_f = ev_all("FREEZE", noticing)
    t_ts = ev_all("TELEPORT", ("NULL-ONLY",)) and ev_all("SPLICE", ("NULL-ONLY",))
    notices = [(m_, l) for m_ in ("FREEZE", "ACTION-SWAP-M") for l in LV if over.get((m_, l)) == "NOTICES" and not (m_ == "ACTION-SWAP-M" and under)]
    one_seed = [(m_, l, DECIDING[s]) for m_ in ("FREEZE", "ACTION-SWAP-M") for l in LV for s in (0, 1) if (m_, l, s) in lab and lab[(m_, l, s)]["label"] == "NOTICES" and over.get((m_, l)) != "NOTICES"]
    lv_lab = lambda man: ", ".join(f"L{l} {over.get((man, l), 'n/a')}" for l in LV)
    P("\n## 4. Triggers (the card's committed consequences, evaluated mechanically on the over-both-seeds labels)")
    P("| committed trigger | fires? | consequence if it fires (card) |\n|---|---|---|")
    P(f"| ACTION-SWAP-M BLIND or INIT-LEVEL at every level | **{'not evaluable (UNDERPOWERED)' if under else ('YES' if t_am else 'no')}** ({lv_lab('ACTION-SWAP-M')}) | the error does not use the action beyond what an untrained network does; together with IMAG1's IA-3 ({ia}) the lever is **E7 DIVRS** plus a counterfactual-target term |")
    P(f"| FREEZE BLIND or INIT-LEVEL at every level | **{'YES' if t_f else 'no'}** ({lv_lab('FREEZE')}) | the predictor does not carry motion beyond its architecture: **MRATE** (carded) |")
    P(f"| TELEPORT / SPLICE NULL-ONLY at every level | **{'YES' if t_ts else 'no'}** (TELEPORT {lv_lab('TELEPORT')}; SPLICE {lv_lab('SPLICE')}) | surprise is a frame-change detector; IMAG1's IA-2 decides before any arm |")
    P(f"| NOTICES on FREEZE or ACTION-SWAP-M at any level | **{'YES' if notices else 'no'}**" + (f" ({', '.join(f'{m_} L{l}' for m_, l in notices)})" if notices else "") +
      " | the first positive 'imagination is used' evidence on video; the per-level table and the surprise traces (`fig/`) go into the paper |")
    if one_seed:
        P(f"\nNOTICES at one seed only (not 'NOTICES at the level'): {', '.join(f'{m_} L{l} {t}' for m_, l, t in one_seed)}.")
    P("\nORIGINAL ACTION-SWAP (VOE1's donor rule, beside; feeds no trigger): " + lv_lab("ACTION-SWAP") + ".")
    P("\nEvent alignment (section 3) is VOE1's, reported beside with no bar.")

    # ---------------------------------------------------------------- 5 disclosures
    P("\n## 5. Disclosures")
    P("* Windows: " + ", ".join(f"{k} {v['n_windows']:,} ({v['n_episodes']} ep)" for k, v in wtab.items()) + f". ACTION-SWAP-M candidates per qualifying window: min {R['swapm_candidates']['min']}, median {R['swapm_candidates']['median']:.0f}. "
      "Neighbouring windows overlap by 12 of 16 frames and are 0.8 s apart; the episode-cluster bootstrap and the whole-episode permutation carry that.")
    P("* Estimator discipline: every interval is an episode-cluster bootstrap over the same episodes (it answers 'would another draw of EPISODES say this?'). The untrained reference adds initialisation variance of an untrained architecture (rule (c)); "
      "it is NOT a replicate of the training run (H-ESTIM-SEED-1): s0 and s1 are two trained seeds of one recipe and 'both seeds' is the only guard against training-seed variance here. A NOTICES label is a necessary, not a sufficient, condition for 'training taught the model to use it'.")
    P("* The untrained models of the two chassis share bit-identical wm.* weights per seed (same `torch.manual_seed`); only the tokens they read differ. Each trained seed is compared with the 16 untrained models on its own chassis. Rule (c) is strict (equal to the maximum is not above it) and uses point accuracies.")
    P("* ACTION-SWAP-M: magnitude within +-10 % on mean |lon| and mean |yaw|, opposite sign where both signed means reach 0.5 m/s^2 / 0.05 rad/s (the stricter reading; n under the looser reading 545, magnitude match alone 1,714; IMPL_CHOICES section 1).")
    P("* REVERSE is the 1.6 s segment played backwards (8-step onset jump then reversed motion); FREEZE is a pause; the original ACTION-SWAP donors differ by at least 1.5 m/s^2 or 0.15 rad/s mean action (VOE1 section 2).")
    P("* The models were trained on 16-frame windows with teacher forcing and context 3. Events: brake 47 onsets, turn 100, lead 27; a cell that is not separated says 'not shown', not 'absent'. STRATEGIC family: PAUSED (item 25), not reported as a pass.")
    P("\n## 6. Figures (`fig/`, PNG, no camera frame)")
    P("* `fig/trace_<MANIP>.png` (six): per level (columns) and model seed (rows) the M2b median (over the qualifying windows) of the normalised surprise against time from t0, manipulated and unmodified; the violet band is the min to max of the "
      f"{n_ut} untrained models' manipulated per-step medians on the same chassis (each normalised by its own unmodified median), the violet dashed line their median. (M2b interquartile bands are in the traces file, not drawn.)")

    open(os.path.join(D, "RESULT_VOE1B.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    make_figs(D, R, n_ut)
    print("VALID - wrote RESULT_VOE1B.md, LABELS_VOE1B.json and figures")


# ---------------------------------------------------------------------------------------------------------- figures
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#6250d6"


def make_figs(D, R, n_ut):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": GRID, "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK})
    Z = np.load(os.path.join(D, "traces_voe1b.npz"))
    os.makedirs(os.path.join(D, "fig"), exist_ok=True)
    for man in C.MANIPS:
        fig, axs = plt.subplots(2, 3, figsize=(11, 5.8), sharey="row", squeeze=False)
        for i, tag in enumerate(DECIDING):
            for j, l in enumerate(LV):
                ax = axs[i, j]
                key = f"{tag}/{man}/{l}"
                if key not in Z.files:
                    ax.set_axis_off()
                    continue
                man_a, un_a = Z[key], Z[f"{tag}/{man}/unmod_{l}"]
                fr = C.step_frames(int(l), man_a.shape[1])
                t = (fr - C.T0) * C.DT
                ax.axvspan(0, C.N_POST * C.DT, color=GRID, alpha=0.55, lw=0)
                ax.axvline(0, color=MUTED, lw=1, ls=":")
                ut = np.stack([Z[f"mt/{u}/{man}/{l}"] for u in (f"UT{i}_{sd}" for sd in range(100, 100 + n_ut))])
                ax.fill_between(t, ut.min(0), ut.max(0), color=VIOLET, alpha=0.20, lw=0, label=f"{n_ut} untrained (min to max)")
                ax.plot(t, np.median(ut, 0), color=VIOLET, lw=1.4, ls="--", label="untrained median")
                for arr, col, lb in ((un_a, BLUE, "M2b unmodified (median)"), (man_a, ORANGE, f"M2b {man.lower()} (median)")):
                    ax.plot(t, np.median(arr, axis=0), color=col, lw=2, marker="o", ms=4 if l != "1" else 3, label=lb, markeredgecolor=SURF, markeredgewidth=0.8)
                ax.set_yscale("log")
                ax.set_title(f"{tag}, level {l} ({C.STRIDE[int(l)] * C.DT:.1f} s steps)", loc="left")
                if i == 1:
                    ax.set_xlabel("time from onset t0 (s)")
                if j == 0:
                    ax.set_ylabel("surprise / unmodified median")
                ax.grid(True, color=GRID, lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
        h, lbs = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, lbs, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
        fig.suptitle(f"{man}: M2b median surprise beside {n_ut} untrained world models (shaded = the 1.6 s scored; tier T0)", x=0.01, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(os.path.join(D, "fig", f"trace_{man.replace('-', '_')}.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    d = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else HERE
    main(d)
