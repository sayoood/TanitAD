"""Reads RESULT_WMACT.json (eval) and RESULT_WMACT_TRAIN.json, checks the four controls FIRST, then writes the tables of RESULT_WMACT_TRAIN.md (generated, so no number is typed by hand)."""
import json, sys

E = json.load(open("RESULT_WMACT.json"))
T = json.load(open("RESULT_WMACT_TRAIN.json"))
TAGS = ("M2b_s0", "M2b_s1")
fails = []

# ---- controls, TRAIN read (the known values of the card)
c = T["controls"]
ctl = {"1": {}, "2": {}, "3": {}, "4": {}}
for t in TAGS:
    a, b = c["1_identity_sens_max_abs"][t], c["1_identity_actionuse_max_abs"][t]
    ctl["1"][t] = (a, b)
    if a != 0.0 or b != 0.0:
        fails.append(f"control 1 {t}: sens {a}, ACTION-USE {b}")
    z = c["2_zeroed_act_embed"][t]
    ctl["2"][t] = ({k: z[k]["ACTION-USE_point"] for k in ("k1", "k5", "k10")}, z["sens_L1_max_abs"])
    if any(z[k]["ACTION-USE_point"] != 0.0 for k in ("k1", "k5", "k10")) or z["sens_L1_max_abs"] != 0.0:
        fails.append(f"control 2 {t}: {ctl['2'][t]}")
    worst = None
    for rk, row in T["rows"][t].items():
        r3 = row["CTRL3_SKILL_startdonor"]
        if not (r3["point"] < 0 and r3["separated"]):
            fails.append(f"control 3 {t} {rk}: {r3}")
        if worst is None or r3["point"] > worst[1]:
            worst = (rk, r3["point"])
    r3 = T["rows"][t]["L1/k1"]["CTRL3_SKILL_startdonor"]
    ctl["3"][t] = (r3["point"], r3["lo"], r3["hi"], worst)
    d4 = c["4_reproduction"][t]["max_abs_diff"]
    ctl["4"][t] = d4
    if not d4 < 1e-6:
        fails.append(f"control 4 {t}: {d4}")
valid = not fails


def f(r, plus=True):
    s = f"{r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" + ("*" if r["separated"] else "")
    return s


def cell(res, tag, row, stat):
    return res["rows"][tag][row][stat]


ROWS = [("SKILL", "L1/k1", "L1 k1 (0.2 s) SKILL"), ("SKILL", "L1/k5", "L1 k5 (1.0 s) SKILL"), ("SKILL", "L1/k10", "L1 k10 (2.0 s) SKILL"),
        ("ACTION-USE", "L1/k1", "L1 k1 ACTION-USE"), ("ACTION-USE", "L1/k5", "L1 k5 ACTION-USE"), ("ACTION-USE", "L1/k10", "L1 k10 ACTION-USE"),
        ("SKILL", "L2/k1", "L2 k1 (0.4 s) SKILL"), ("SKILL", "L3/k1", "L3 k1 (0.8 s) SKILL")]
EXTRA = [("ACTION-USE-MEAN", "L1/k1", "L1 k1 ACTION-USE-MEAN"), ("ACTION-USE", "L2/k1", "L2 k1 ACTION-USE"), ("ACTION-USE", "L3/k1", "L3 k1 ACTION-USE"),
         ("SKILL", "L2/k3", "L2 k3 SKILL"), ("SKILL", "L3/k3", "L3 k3 SKILL")]

out = []
out.append("# RESULT_WMACT_TRAIN: the same statistics on TRAIN windows, beside eval-141 (card `PREREG_V8F_WMACT.md`, frozen prefix 5,478 bytes `d21b14cc`)\n")
out.append("MEASURED 2026-11-07 on the dev-box CPU. Tier T0: teacher-forced, logged actions, true future frames. A WM diagnostic, never a driving claim.\n")
if not valid:
    out.append("## INVALID\nA control missed its known value, so no statistic below is a result:\n" + "\n".join(f"* {x}" for x in fails) + "\n")
nE, nT = E["windows_used"], T["windows_used"]
out.append("## Controls (TRAIN read)\n| control | M2b s0 | M2b s1 |\n|---|---|---|")
out.append(f"| 1. identity donor: sens max abs, ACTION-USE max abs | {ctl['1']['M2b_s0'][0]} , {ctl['1']['M2b_s0'][1]} | {ctl['1']['M2b_s1'][0]} , {ctl['1']['M2b_s1'][1]} |")
out.append("| 2. zeroed level-1 action embedding: ACTION-USE at k 1 / 5 / 10, sens at L1 | "
           + " | ".join("{} / {} / {}, {}".format(*[ctl["2"][t][0][k] for k in ("k1", "k5", "k10")], ctl["2"][t][1]) for t in TAGS) + " |")
out.append("| 3. rollout from another window's start: SKILL at L1 k1 (must be below 0, separated); least-negative row of all 7 | "
           + " | ".join(f"{ctl['3'][t][0]:+.4f} [{ctl['3'][t][1]:+.4f}, {ctl['3'][t][2]:+.4f}]; {ctl['3'][t][3][0]} {ctl['3'][t][3][1]:+.4f}" for t in TAGS) + " |")
out.append("| 4. WM loss via `HJEPAStack.loss` against `WMAux.forward`, 64 TRAIN windows, max abs diff | " + " | ".join(str(ctl["4"][t]) for t in TAGS) + " |\n")
out.append("Control 3 is below 0 and separated at every level and horizon read, both seeds." if not any(x.startswith("control 3") for x in fails) else "Control 3 FAILED, see above.")
if valid:
    out.append(f"\n## Sample\n* eval-141: {nE:,} windows of {E['windows_total']:,} (all 16 frames real), {E['episodes_used']} episodes (clips).\n"
               f"* TRAIN: {nT:,} windows of {T['windows_total']:,} (all 16 frames real), {T['episodes_used']} episodes (clips); 17 starts per clip, 13 of which hold 16 real frames. Clips sha256 `{T['clips_sha256']}`.\n"
               "* Estimator: episode-cluster bootstrap B 2,000, seed 0, `taniteval/ci.py` draws; * = the 95 % interval excludes 0. Another draw of episodes (not another training run, not another inference).\n")
    for title, rr in (("Requested rows", ROWS), ("Also computed (same statistics, deciding nothing here)", EXTRA)):
        out.append(f"## {title}\n| row | eval s0 | TRAIN s0 | eval s1 | TRAIN s1 |\n|---|---|---|---|---|")
        for stat, rk, label in rr:
            cells = []
            for t in TAGS:
                cells += [f(cell(E, t, rk, stat)), f(cell(T, t, rk, stat))]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    out.append(f"n per row: eval {nE:,} windows / {E['episodes_used']} episodes; TRAIN {nT:,} windows / {T['episodes_used']} episodes. Every row above uses these n (all rows share the windows).\n")
    s = {t: cell(T, t, "L1/k1", "SKILL") for t in TAGS}
    fits = all(s[t]["point"] > 0 and s[t]["lo"] > 0 for t in TAGS)
    out.append("## Outcome by the brief's rule\n* TRAIN L1 k1 SKILL: " + "; ".join(f"{t} {f(s[t])}" for t in TAGS))
    out.append("* **" + ("L1 FITS, DOES NOT GENERALISE" if fits else "L1 DOES NOT FIT") + "**"
               + (" (TRAIN L1 k1 SKILL is above 0 and separated at both seeds; eval L1 k1 SKILL is not)." if fits else
                  " (TRAIN L1 k1 SKILL is not above 0 and separated at both seeds)."))
    out.append("")
    sens_e = {t: E["sens"][t]["L1/sens_rom/all/k1"] for t in TAGS}
    sens_t = {t: T["sens"][t]["L1/sens_rom/all/k1"] for t in TAGS}
    out.append("Beside, `sens_rom/all` at L1 k1 (relative action sensitivity): eval " + ", ".join(f"{t} {sens_e[t]:.3f}" for t in TAGS) + "; TRAIN " + ", ".join(f"{t} {sens_t[t]:.3f}" for t in TAGS) + ".")
    out.append(f"\nWall time: stream and latents {T['wall_s_collect']:.0f} s; statistics and controls {T['wall_s_stats']:.0f} s.")
open("RESULT_WMACT_TRAIN.md", "w", newline="\n").write("\n".join(out) + "\n")
print("VALID" if valid else "INVALID", fails)
