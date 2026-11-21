"""PREREG_V8F_VOE1 step 3: checks controls 1 to 4 FIRST, then writes RESULT_VOE1.md and fig/*.png from RESULT_VOE1.json and traces_voe1.npz.
Judges with the rules of IMPL_CHOICES_VOE1.md; computes no new model statistic. A failed control prints INVALID and no label table.
Run: python report_voe1.py   (matplotlib, numpy only)"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voe1_core as C  # noqa: E402

DECIDING, CONTROL, BESIDE = ("M2b_s0", "M2b_s1"), ("RI_s0", "RI_s1"), ("RIs_s0", "RIs_s1")
ALL_TAGS = DECIDING + CONTROL + BESIDE
LV = ("1", "2", "3")
STEP_S = {"1": 0.2, "2": 0.4, "3": 0.8}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def fnum(r, k="point", d=3):
    return f"{r[k]:.{d}f}"


def cell(r, d=3, sgn=False):
    """point [lo, hi] plus S (two-part separated above) / b (bootstrap only) / - ."""
    f = (lambda x: f"{x:+.{d}f}") if sgn else (lambda x: f"{x:.{d}f}")
    flag = "S" if r.get("sep_above") else ("b" if r.get("boot_above") else "-")
    if r.get("sep_below"):
        flag = "S-below"
    return f"{f(r['point'])} [{f(r['lo'])}, {f(r['hi'])}] {flag}"


READ_INVALID = "--read-invalid" in sys.argv


def main():
    R = json.load(open(os.path.join(HERE, "RESULT_VOE1.json"), encoding="utf-8"))
    meta, tags, events = R["meta"], R["tags"], R["events"]
    md = []
    P = md.append
    ctrl, fails = {}, []

    # ------------------------------------------------------------------ integrity
    card = os.path.join(HERE, "..", "PREREG_V8F_VOE1.md")
    integ = {"card_prefix_ok": hashlib.sha256(open(card, "rb").read()[:5822]).hexdigest() == meta["card_prefix_sha256"],
             "impl_ok": sha_file(os.path.join(HERE, "IMPL_CHOICES_VOE1.md")) == meta["impl_sha256"] == open(os.path.join(HERE, "IMPL_CHOICES_VOE1.sha256")).read().split()[0],
             "code_unchanged_since_run": {f: sha_file(os.path.join(HERE, f)) == h for f, h in meta["code_sha256"].items()}}
    if not (integ["card_prefix_ok"] and integ["impl_ok"]):
        fails.append("integrity: card prefix or IMPL_CHOICES hash differs from the run's")
    ctrl["integrity"] = integ

    # ------------------------------------------------------------------ control 4 (pytest log)
    log = open(os.path.join(HERE, "PYTEST_VOE1.log"), encoding="utf-8").read()
    hs = {}
    for line in open(os.path.join(HERE, "PYTEST_VOE1.hashes")).read().strip().splitlines():
        h, n = line.split()
        hs[n.lstrip("*")] = h
    m = re.search(r"(\d+) passed", log)
    c4 = {"passed": int(m.group(1)) if m else 0, "failed": re.search(r"\d+ failed", log) is not None or re.search(r"\d+ error", log) is not None,
          "test_file_sha_matches_log": hs.get("test_voe1_synth.py") == sha_file(os.path.join(HERE, "test_voe1_synth.py")),
          "core_sha_matches_log": hs.get("voe1_core.py") == sha_file(os.path.join(HERE, "voe1_core.py"))}
    c4["pass"] = bool(c4["passed"] >= 13 and not c4["failed"] and c4["test_file_sha_matches_log"] and c4["core_sha_matches_log"])
    ctrl["c4"] = c4
    if not c4["pass"]:
        fails.append(f"control 4: {c4}")

    # ------------------------------------------------------------------ control 1: real-vs-real, none separated either side
    c1 = {}
    for tag in DECIDING + CONTROL:
        for l in LV:
            r = tags[tag]["rvr"]["levels"][l]
            c1[f"{tag}/L{l}"] = {"point": r["point"], "lo": r["lo"], "hi": r["hi"], "separated_any": r["separated_any"], "n": r["n"]}
    c1_fail = [k for k, v in c1.items() if v["separated_any"]]
    ctrl["c1"] = {"rows": c1, "failed": c1_fail, "pass": not c1_fail}
    if c1_fail:
        fails.append(f"control 1: separated from 0.5 at {c1_fail}")

    # ------------------------------------------------------------------ control 2: random-init not NOTICES on FREEZE / ACTION-SWAP
    c2 = {}
    for tag in CONTROL:
        for man in ("FREEZE", "ACTION-SWAP"):
            for l in LV:
                c2[f"{tag}/{man}/L{l}"] = tags[tag]["manips"][man]["levels"][l]["label"]
    c2_fail = [k for k, v in c2.items() if v == "NOTICES"]
    ctrl["c2"] = {"labels": c2, "failed": c2_fail, "pass": not c2_fail}
    if c2_fail:
        fails.append(f"control 2: random-init NOTICES at {c2_fail}")

    # ------------------------------------------------------------------ control 3: calibration shares per family
    fam = {k: [0, 0] for k in ("ACC", "NULL-ACC", "DIFF", "RVR", "AP-CHANCE", "AP-NULL")}
    for tag in ALL_TAGS:
        for l in LV:
            c = tags[tag]["rvr"]["levels"][l]["cal"]
            fam["RVR"][0] += sum(a or b for a, b in zip(c["sep_above"], c["sep_below"])); fam["RVR"][1] += len(c["sep_above"])
        for man in C.MANIPS:
            for l in LV:
                r = tags[tag]["manips"][man]["levels"][l]
                for f, k in (("ACC", "acc"), ("NULL-ACC", "null_acc"), ("DIFF", "diff")):
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
    json.dump(ctrl, open(os.path.join(HERE, "CONTROLS_VOE1.json"), "w", encoding="utf-8", newline="\n"), indent=1)

    # ------------------------------------------------------------------ the markdown
    P("# RESULT_VOE1: does each level notice an impossible future? (violation of expectation on real driving video; tier T0)")
    P("Card `../PREREG_V8F_VOE1.md` (frozen prefix 5,822 bytes, sha256 `555c5a38...`, unedited). Definitions: `IMPL_CHOICES_VOE1.md` (sha256 `%s`). Models: M2b s0 and s1, zero training; controls: random-init world models. "
      "1,976 eval-141 windows (141 episodes), onset t0 = window frame 4, 1.6 s = 8 cache steps scored. **Tier T0** (a world-model diagnostic on cached latents, teacher-forced one-step prediction, never a driving claim). "
      "Every number below is computed by `run_voe1.py` into `RESULT_VOE1.json` (run %s to %s, %.0f s)." % (meta["impl_sha256"][:8], meta["started"], meta["finished"], meta["seconds"]))
    P("\n## 0. Controls (checked before any row is read)")
    P(f"**Run status: {'VALID' if valid else 'INVALID'}**" + ("" if valid else " -- " + "; ".join(fails)))
    P("\n| control | rule | result | pass |\n|---|---|---|---|")
    P(f"| 1 real-vs-real | the second copy from the window 0.2 s later: not two-part separated from 0.5 at any level (M2b s0/s1, RI s0/s1; 12 cells) | "
      f"points {min(v['point'] for v in c1.values()):.3f} to {max(v['point'] for v in c1.values()):.3f}; separated: {', '.join(c1_fail) or 'none'} | {'PASS' if ctrl['c1']['pass'] else 'FAIL'} |")
    P(f"| 2 random-init WM | RI_s0/RI_s1 (every wm.* parameter re-initialised) not NOTICES on FREEZE and ACTION-SWAP at any level (12 cells) | "
      f"NOTICES at: {', '.join(c2_fail) or 'none'} | {'PASS' if ctrl['c2']['pass'] else 'FAIL'} |")
    P("| 3 calibration | 20 extra permutation draws (1001-1020) scored as if real: at most 5 % of rows separated per family | " +
      "; ".join(f"{f} {v['separated']}/{v['rows']} = {v['share']:.3%}" for f, v in c3.items()) + f" | {'PASS' if ctrl['c3']['pass'] else 'FAIL'} |")
    P(f"| 4 synthetic test | `test_voe1_synth.py`: toy constant-velocity model NOTICES on planted FREEZE and TELEPORT, action-blind toy BLIND on ACTION-SWAP; 13 tests | "
      f"{c4['passed']} passed; test file and core hash equal the log's: {c4['test_file_sha_matches_log'] and c4['core_sha_matches_log']} | {'PASS' if c4['pass'] else 'FAIL'} |")
    P("\nControl 1 cells (accuracy of 'the later copy scores higher', two-part rule, either side):\n\n| cell | n | accuracy [interval] | separated either side |\n|---|---|---|---|")
    for k, v in c1.items():
        P(f"| {k} | {v['n']} | {v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] | {v['separated_any']} |")
    rp = R["reinit"]
    P("\nReproduction (MEASURED): the unmodified path (per-clip frame bank + adapter + wm.z1) vs `latents.pt` `Z1`, 1,976 windows: max abs difference "
      f"{R['repro']['M2b_s0']['max_abs_diff_z1']:.2e} (s0), {R['repro']['M2b_s1']['max_abs_diff_z1']:.2e} (s1) on values up to {R['repro']['M2b_s0']['ref_abs_max']:.1f}; logged actions {R['repro']['actions_max_abs_diff']}. "
      f"Random-init: RI re-initialised {rp['RI_s0']['n_params_reinitialised']:,} of {rp['RI_s0']['n_params_total']:,} wm.* parameters ({rp['RI_s0']['n_tensors']} tensors: proj, act_embed, level encoders, macro-encoders, predictors, idm heads); "
      f"RIs (beside) re-initialised {rp['RIs_s0']['n_params_reinitialised']:,} (the stack), trained wm.proj kept; chassis (adapter, pos) trained in both.")
    if not valid and not READ_INVALID:
        P("\n**INVALID: no label table is printed. The failing controls above are the finding; the run is not read.** Raw rows stay in `RESULT_VOE1.json`.")
        open(os.path.join(HERE, "RESULT_VOE1.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
        print("INVALID:", fails)
        return

    out_md = "RESULT_VOE1.md"
    if not valid:
        out_md = "RESULT_VOE1_INVALID_READ.md"
        P("\n> **THE RUN IS INVALID BY THE CARD'S CONTROLS (above). Everything below is printed only because `--read-invalid` was passed: it is NOT admissible as a VOE1 result, "
          "cites no claim, and may enter the register only after the Master Mind amends the controls post hoc (see DIAG_CONTROLS_VOE1.md for the controls' own false-fail rate).**")

    # ---------------------------------------------------------------- 1 headline labels
    lab = {}
    for man in C.MANIPS:
        for l in LV:
            lab[(man, l)] = C.level_label(*(tags[t]["manips"][man]["levels"][l]["label"] for t in DECIDING))
    P("\n## 1. Headline: the label of each (manipulation, level) over both seeds (NOTICES needs both seeds)")
    P("| manipulation | n windows (episodes) | level 1 (0.2 s) | level 2 (0.4 s) | level 3 (0.8 s) |\n|---|---|---|---|---|")
    for man in C.MANIPS:
        n = tags[DECIDING[0]]["manips"][man]
        P(f"| {man} | {n['n']:,} ({n['n_episodes']}) | " + " | ".join(lab[(man, l)] for l in LV) + " |")
    P("\nPer-seed labels (s0 / s1): " + "; ".join(f"{man}: " + ", ".join(f"L{l} {tags['M2b_s0']['manips'][man]['levels'][l]['label']}/{tags['M2b_s1']['manips'][man]['levels'][l]['label']}" for l in LV) for man in C.MANIPS))

    # ---------------------------------------------------------------- 2 tables per manipulation
    P("\n## 2. Pairwise accuracy per manipulation x level x seed")
    P("Accuracy = share of pairs where the manipulated copy's max normalised surprise over the 1.6 s after t0 is above its unmodified copy's (ties 0.5); interval = episode-cluster bootstrap (B 2,000, seed 0); "
      "flag S = two-part separated above (interval above the claim AND point above the 97.5th percentile of the R = 199 whole-episode label-swap reference), b = bootstrap interval only, - = neither. "
      "The Copy-Last null's accuracy is beside it; the difference is the paired model-minus-null accuracy over the same pairs (claim 0).")
    for man in C.MANIPS:
        P(f"\n### {man}" + ("  (the Copy-Last null enters NOTICES)" if man in C.NULL_COUNTS else "  (decisive row: the null does not enter; it reads at most 0.5 by construction)"))
        P("| tag | level | n (episodes) | accuracy [interval] | Copy-Last null accuracy | model minus null [interval] | median score manip / unmod | label |\n|---|---|---|---|---|---|---|---|")
        for tag in DECIDING + CONTROL + BESIDE:
            for l in LV:
                r = tags[tag]["manips"][man]["levels"][l]
                nn = tags[tag]["manips"][man]
                P(f"| {tag}{' (control)' if tag in CONTROL else ' (beside)' if tag in BESIDE else ''} | L{l} | {nn['n']:,} ({nn['n_episodes']}) | {cell(r['acc'])} | {r['null_acc']['point']:.3f} [{r['null_acc']['lo']:.3f}, {r['null_acc']['hi']:.3f}] "
                  f"| {cell(r['diff'], sgn=True)} | {r['median_score_manip']:.2f} / {r['median_score_unmod']:.2f} | {r['label']} |")

    # ---------------------------------------------------------------- 3 events
    P("\n## 3. Event alignment (IA-6d): do the surprise peaks sit within +-0.4 s of braking, turn and lead-change events?")
    ev_n = R["event_onsets_in_bank"]
    P(f"Unmodified 16-frame windows, every (window, step) point of a level pooled; positive = an event onset within +-0.4 s (+-2 cache steps) of the step. Onsets in the 141 eval clips: brake {ev_n['brake']}, turn {ev_n['turn']}, lead {ev_n['lead']} "
      "(definitions in IMPL_CHOICES section 6). AP = average precision of ranking points by surprise; chance = the event rate; null AP = the same ranking by the Copy-Last distance of the level. "
      "S = two-part separated above 0 (bootstrap interval AND point above the 97.5th percentile of R = 199 label-permutation draws).")
    aligns = {}
    for ev in C.EVENTS:
        P(f"\n### {ev}\n\n| level | points (positives) | seed | AP | chance | null AP | AP minus chance [interval] | AP minus null AP [interval] | ALIGNS (both seeds) |\n|---|---|---|---|---|---|---|---|---|")
        for l in LV:
            ok_both = True
            for t in DECIDING:
                r = events[t][ev]["L" + l]
                a, b = r["rows"]["ap_minus_chance"], r["rows"]["ap_minus_null"]
                ok_both &= bool(a["sep_above"] and b["sep_above"])
            aligns[(ev, l)] = ok_both
            for i, t in enumerate(DECIDING):
                r = events[t][ev]["L" + l]
                a, b = r["rows"]["ap_minus_chance"], r["rows"]["ap_minus_null"]
                P(f"| L{l} | {r['n_points']:,} ({r['n_events_pts']}) | {t[-2:]} | {r['ap']:.4f} | {r['chance']:.4f} | {r['ap_null']:.4f} | {cell(a, 4, True)} | {cell(b, 4, True)} | {'**ALIGNS**' if ok_both else 'no'} |" if i == 0 else
                  f"| | | {t[-2:]} | {r['ap']:.4f} | {r['chance']:.4f} | {r['ap_null']:.4f} | {cell(a, 4, True)} | {cell(b, 4, True)} | |")
    P("\nBeside (random-init and stack-only random-init; AP only, no interval): ")
    P("| tag | event | AP L1 / L2 / L3 | null AP L1 / L2 / L3 | chance L1 / L2 / L3 |\n|---|---|---|---|---|")
    for t in CONTROL + BESIDE:
        for ev in C.EVENTS:
            P(f"| {t} | {ev} | " + " / ".join(f"{events[t][ev]["L" + l]['ap']:.4f}" for l in LV) + " | " + " / ".join(f"{events[t][ev]["L" + l]['ap_null']:.4f}" for l in LV) + " | " + " / ".join(f"{events[t][ev]["L" + l]['chance']:.4f}" for l in LV) + " |")
    P("\n**ALIGNS summary:** " + "; ".join(f"{ev}: " + (", ".join(f"L{l}" for l in LV if aligns[(ev, l)]) or "no level") for ev in C.EVENTS) + ".")
    P("\n**Hierarchy hypothesis (HYPOTHESIS, no bar):** L3 aligns at least as well as L1. (AP - chance at L3) minus (AP - chance at L1), episode-cluster interval from the shared bootstrap draws:")
    P("| event | seed | L1 AP-chance | L3 AP-chance | L3 minus L1 [interval] |\n|---|---|---|---|---|")
    for ev in C.EVENTS:
        for t in DECIDING:
            e1, e3 = events[t][ev]["L1"], events[t][ev]["L3"]
            d = np.asarray(e3["boot_apc"]) - np.asarray(e1["boot_apc"])
            pt = (e3["ap"] - e3["chance"]) - (e1["ap"] - e1["chance"])
            P(f"| {ev} | {t[-2:]} | {e1['ap'] - e1['chance']:+.4f} | {e3['ap'] - e3['chance']:+.4f} | {pt:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] |")

    # ---------------------------------------------------------------- 4 triggers
    all_l = lambda man, want: all(lab[(man, l)] == want for l in LV)
    trig = []
    imag1 = os.path.join(HERE, "..", "imag1", "RESULT_IMAG1.md")
    ia = "IMAG1 result file present (`../imag1/RESULT_IMAG1.md`): read it for IA-3 and IA-2" if os.path.exists(imag1) else "IMAG1 has no result file yet: IA-3 / IA-2 status OPEN"
    t_a = all_l("ACTION-SWAP", "BLIND")
    t_f = all_l("FREEZE", "BLIND")
    t_ts = all_l("TELEPORT", "NULL-ONLY") and all_l("SPLICE", "NULL-ONLY")
    notices = [(m, l) for m in ("FREEZE", "ACTION-SWAP") for l in LV if lab[(m, l)] == "NOTICES"]
    one_seed = [(m, l, t) for m in ("FREEZE", "ACTION-SWAP") for l in LV for t in DECIDING if tags[t]["manips"][m]["levels"][l]["label"] == "NOTICES" and lab[(m, l)] != "NOTICES"]
    P("\n## 4. Triggers (the card's committed consequences, evaluated mechanically on the over-both-seeds labels)")
    P("| committed trigger | fires? | consequence if it fires |\n|---|---|---|")
    P(f"| ACTION-SWAP BLIND at every level | **{'YES' if t_a else 'no'}** ({', '.join(f'L{l} {lab[('ACTION-SWAP', l)]}' for l in LV)}) | the world model ignores the action after t0 in its error; joins IMAG1 IA-3 ({ia}); if IA-3 is also INERT or WEAK the lever is E7 DIVRS plus a counterfactual-target term (UWM-JEPA) |")
    P(f"| FREEZE BLIND at every level | **{'YES' if t_f else 'no'}** ({', '.join(f'L{l} {lab[('FREEZE', l)]}' for l in LV)}) | the predictor does not carry motion; joins G1; MRATE is the lever (already carded) |")
    P(f"| TELEPORT and SPLICE NULL-ONLY at every level | **{'YES' if t_ts else 'no'}** (TELEPORT {', '.join(lab[('TELEPORT', l)] for l in LV)}; SPLICE {', '.join(lab[('SPLICE', l)] for l in LV)}) | surprise is a frame-change detector; the next lever is IMAG1's IA-2 before any new arm |")
    P(f"| NOTICES on FREEZE or ACTION-SWAP at any level | **{'YES' if notices else 'no'}**" + (f" ({', '.join(f'{m} L{l}' for m, l in notices)})" if notices else "") +
      " | the first positive 'imagination is used' evidence on video; into the paper with the per-level table and the surprise-trace figure |")
    if one_seed:
        P(f"\nNOTICES at one seed only (not 'NOTICES at the level'): {', '.join(f'{m} L{l} {t}' for m, l, t in one_seed)}.")
    # beside: stack-only random init
    rs = [(t, m, l) for t in BESIDE for m in ("FREEZE", "ACTION-SWAP") for l in LV if tags[t]["manips"][m]["levels"][l]["label"] == "NOTICES"]
    P("\nBeside, not gating: the stack-only random-init tags (trained latents, random predictor) are NOTICES at: " + (", ".join(f"{t} {m} L{l}" for t, m, l in rs) if rs else "no FREEZE / ACTION-SWAP cell") +
      ". " + ("This is a free-lunch signal from the trained latent geometry and is disclosed against any NOTICES above." if rs else ""))

    # ---------------------------------------------------------------- 5 disclosures
    wt = R["window_table"]
    P("\n## 5. Disclosures")
    P("* Windows: " + ", ".join(f"{k} {v['n_windows']:,} ({v['n_episodes']} ep)" for k, v in wt.items()) + ". Neighbouring windows overlap by 12 of 16 frames and are 0.8 s apart; the episode-cluster bootstrap and the whole-episode permutation carry that, no other clustering is applied.")
    P("* Estimator discipline: every interval is an episode-cluster bootstrap over the same 141 episodes (answers 'would another draw of episodes say this?'; it is blind to training-seed variance (H-ESTIM-SEED-1)): s0 and s1 are two trained seeds of one recipe, and 'both seeds' is the only guard against seed variance here. No replicate-arm noise floor exists for this rig.")
    P("* The over-both-seeds label rules, the families of control 3, the event definitions and the tag definitions were fixed in IMPL_CHOICES_VOE1.md before the first statistic (`impl sha256 %s`)." % meta["impl_sha256"][:12])
    P("* REVERSE is read as the 1.6 s segment played backwards (an 8-step onset jump then reversed motion); reversal about t0 was not run. FREEZE is a pause (the picture resumes from the held frame); ACTION-SWAP donors differ by at least 1.5 m/s^2 mean longitudinal action or 0.15 rad/s mean yaw rate.")
    P("* The model was trained on 16-frame windows with teacher forcing and context 3; the whole-clip traces in the clip figure run the same sliding context over 100 frames (a use outside the training length, figure only).")
    P("* Events: brake 47 onsets (39 clips), turn 100 (62), lead 27 (22): the lead and brake cells have wide intervals; a cell that is not separated says 'not shown', not 'absent'.")
    P("\n## 6. Figures (`fig/`, no camera frame)")
    P("* `fig/trace_<MANIP>.png`: per level (columns) and seed (rows) the median and 25-75 % band of the normalised surprise against time from t0, manipulated vs unmodified, qualifying windows only. `fig/clips_events.png`: two unmodified clips, whole-clip one-step surprise per level with the event onsets marked.")

    open(os.path.join(HERE, out_md), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    make_figs(R, "" if valid else "  [INVALID RUN: controls 1, 2 failed; not admissible]")
    print(("VALID" if valid else "INVALID (read-invalid)"), "- wrote", out_md, "and figures")


# ---------------------------------------------------------------------------------------------------------- figures
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"


def make_figs(R, banner=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": GRID, "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK})
    Z = np.load(os.path.join(HERE, "traces_voe1.npz"))
    os.makedirs(os.path.join(HERE, "fig"), exist_ok=True)
    for man in C.MANIPS:
        fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharey="row")
        for i, tag in enumerate(DECIDING):
            for j, l in enumerate(LV):
                ax = axs[i, j]
                man_a, un_a = Z[f"{tag}/{man}/{l}"], Z[f"{tag}/{man}/unmod_{l}"]
                fr = C.step_frames(int(l), man_a.shape[1])
                t = (fr - C.T0) * C.DT
                ax.axvspan(0, C.N_POST * C.DT, color=GRID, alpha=0.55, lw=0)
                ax.axvline(0, color=MUTED, lw=1, ls=":")
                for arr, col, lb in ((un_a, BLUE, "unmodified"), (man_a, ORANGE, man.lower())):
                    q1, q2, q3 = np.percentile(arr, [25, 50, 75], axis=0)
                    ax.fill_between(t, q1, q3, color=col, alpha=0.18, lw=0)
                    ax.plot(t, q2, color=col, lw=2, marker="o", ms=4 if l != "1" else 3, label=lb, markeredgecolor=SURF, markeredgewidth=0.8)
                ax.set_yscale("log")
                ax.set_title(f"{tag}, level {l} ({C.STRIDE[int(l)] * C.DT:.1f} s steps)", loc="left")
                if i == 1:
                    ax.set_xlabel("time from onset t0 (s)")
                if j == 0:
                    ax.set_ylabel("surprise / unmodified median")
                ax.grid(True, color=GRID, lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
        axs[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
        fig.suptitle(f"{man}: surprise around the onset (median and 25-75 % band; shaded = the 1.6 s scored; tier T0)" + banner, x=0.01, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(HERE, "fig", f"trace_{man.replace('-', '_')}.png"), dpi=140)
        plt.close(fig)
    clips = list(R["along_clip"])
    fig, axs = plt.subplots(3, len(clips), figsize=(6.2 * len(clips), 7.2), squeeze=False)
    evcol = {"brake": (AQUA, "v"), "turn": (VIOLET, "s"), "lead": (MAGENTA, "D")}
    for c, clip in enumerate(clips):
        for r, l in enumerate(LV):
            ax = axs[r, c]
            for tag, col, ls in (("M2b_s0", BLUE, "-"), ("M2b_s1", ORANGE, "--")):
                s = Z[f"clip/{clip}/{tag}/L{l}/sur"]
                fr = C.step_frames(int(l), len(s)) * C.DT
                ax.plot(fr, s, color=col, lw=1.4, ls=ls, label=f"{tag} surprise")
            n = Z[f"clip/{clip}/M2b_s0/L{l}/null"]
            ax.plot(C.step_frames(int(l), len(n)) * C.DT, n, color=MUTED, lw=1, alpha=0.8, label="Copy-Last distance (s0)")
            ax.set_yscale("log")
            top = ax.get_ylim()[1]
            for ev, (col, mk) in evcol.items():
                xs = np.array(R["along_clip"][clip]["onsets"][ev]) * C.DT
                ax.plot(xs, np.full(len(xs), top * 0.8), mk, color=col, ms=7, ls="none", markeredgecolor=SURF, markeredgewidth=0.8, label=f"{ev} onset" if (r == 0) else None)
            ax.set_title(f"clip {clip[:8]}, level {l}", loc="left")
            ax.grid(True, color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 2:
                ax.set_xlabel("time in clip (s)")
            if c == 0:
                ax.set_ylabel("one-step surprise / unmodified median")
        if c == 0:
            axs[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Unmodified clips: whole-clip one-step surprise per level with the event onsets (tier T0)" + banner, x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(HERE, "fig", "clips_events.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
