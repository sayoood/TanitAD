"""PREREG_V8F_DECINFO report: reads runs/*.json, writes RESULT_DECINFO.md + RESULT_DECINFO.json. Evaluates the card's outcome rules mechanically. Tier T0, no interval.
Usage: python report_decinfo.py   (plain python, no model code needed)"""
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
BAR = {"speed": 0.70, "yaw": 0.50}
FROZEN_LADDER = ("C0_s0", "R35_s0", "R35H_s0", "S35_s0")
TRIGGER = {
    "LADDER-READS": "PLANTEST3 rerun on that checkpoint's z1 and readout.",
    "POOL-LOSS": "next world-model arm replaces the mean pool in z1 with a spatial or attention pool of the adapter tokens, on the ladder rig.",
    "ADAPTER-LOSS": "next arm unfreezes the adapter or adds a speed and yaw-rate auxiliary loss on it.",
    "NOT-READ-FROM-VISION": "the same probe at full n on box 2 (4,572 train clips cached only there; one job, after the ladder), before any conclusion about the encoder.",
}
RUNG_DESC = {"R0": "M2b z1 (128-D)", "R1": "ladder z1 (128-D)", "R2": "pooled adapter token (256-D)", "R3": "adapter grid 4x10, 1 frame (PCA 32)",
             "R4": "adapter grid 4x10, 4 frames (PCA 32)", "R5": "encoder grid, 1 frame (PCA 32)", "R6": "encoder grid, 4 frames (PCA 32)"}


def load():
    units = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        units.append(json.load(open(p, encoding="utf-8")))
    return units


def best(rows, tgt):
    rs = [r for r in rows if r["target"] == tgt]
    if not rs:
        return None
    b = max(rs, key=lambda r: r["r2"])
    return b["r2"], f"{b['reader']} m={b['m']}"


def main():
    units = load()
    out = {"tier": "T0", "card": "PREREG_V8F_DECINFO.md prefix sha256 7e2e8aa66dcb04a275de09df26cd0c5a6f63e2325176e82530e557d2dd7668f7", "units": [f"{u['rung']}_{u['tag']}" for u in units]}
    L = ["# RESULT_DECINFO: where on the vision chain are speed and yaw rate lost?",
         "Tier **T0**, a readout diagnostic on the PLANTEST3 TUNE inner split (53 fit episodes, 17 held-out episodes, 238 held-out windows; salt `plantest2-v1`). No interval, as the card says: differences under 0.05 carry no weight. MEASURED: `decinfo/runs/*.json` (one file per rung and tag). Card: `../PREREG_V8F_DECINFO.md` (prefix sha256 `7e2e8aa6...`). Choices: `IMPL_CHOICES_DECINFO.md`.", ""]
    # ---- controls
    viol = []
    c1 = []; c2 = []; c4 = []; c5 = []
    for u in units:
        for r in u["rows"]:
            ok = r["r2_shuffled"] <= 0.05
            c1.append(ok)
            if not ok:
                viol.append(f"control 1: {u['rung']} {u['tag']} {r['reader']} m={r['m']} {r['target']} shuffled R2 {r['r2_shuffled']:+.3f}")
        for p in u["planted"]:
            c2.append(p["passed"])
            if not p["passed"]:
                viol.append(f"control 2: {u['rung']} {u['tag']} m={p['m']} planted {p['r2_signal']:.3f} zeroed {p['r2_zeroed']:.3f}")
        m = u["meta"]
        if "control4_reproduction" in m:
            for t, d in m["control4_reproduction"].items():
                c4.append(d["passed"])
                if not d["passed"]:
                    viol.append(f"control 4: {u['tag']} {t} diff {d['abs_diff']:.4f}")
        if "control5_chain_identity" in m:
            c5.append(m["control5_chain_identity"]["passed"])
            if not m["control5_chain_identity"]["passed"]:
                viol.append(f"control 5: {u['tag']} not bit for bit")
    valid = not viol and bool(c1) and bool(c4) and bool(c5)
    out["controls"] = dict(c1_n=len(c1), c1_pass=sum(c1), c2_n=len(c2), c2_pass=sum(c2), c4_n=len(c4), c4_pass=sum(c4), c5_n=len(c5), c5_pass=sum(c5), violations=viol, valid=valid)
    L += ["## Controls (the read is INVALID if one fails)",
          f"* 1 shuffled targets (at or below 0.05): {sum(c1)} of {len(c1)} reader rows pass. 2 planted signal (at or above 0.80, zeroed at or below 0.02): {sum(c2)} of {len(c2)} rung/m inputs pass. 3 split: asserted in code and by `test_decinfo_controls.py` (see `TEST_DECINFO.log`). 4 reproduction of the PLANTEST3 gate within 0.01: {sum(c4)} of {len(c4)}. 5 chain identity bit for bit: {sum(c5)} of {len(c5)}.",
          f"* Violations: {('none' if not viol else '; '.join(viol))}", f"* Verdict on validity: **{'VALID' if valid else 'INVALID or incomplete (see violations / missing units)'}**", ""]
    # ---- tables
    by_rung = {}
    for u in units:
        by_rung.setdefault(u["rung"], []).append(u)
    for tgt, name in (("speed", "Speed (m/s)"), ("yaw", "Yaw rate (rad/s)")):
        L += [f"## {name}: held-out R2 per rung and reader (bar {BAR[tgt]:.2f}); shuffled-target control in brackets", "",
              "| rung | input | tag | reader, m | R2 [shuffled] | fit samples | held-out samples | fit / held-out episodes |", "|---|---|---|---|---|---|---|---|"]
        for rung in sorted(by_rung):
            for u in sorted(by_rung[rung], key=lambda u: u["tag"]):
                tag = u["tag"] + (" (added after the freeze)" if u["meta"].get("added_after_freeze") else "")
                for r in [r for r in u["rows"] if r["target"] == tgt]:
                    extra = f" seeds {', '.join(f'{x:+.3f}' for x in r['r2_seeds'])}" if "r2_seeds" in r else ""
                    L.append(f"| {rung} | {RUNG_DESC[rung]} | {tag} | {r['reader']}, m={r['m']} | {r['r2']:+.3f} [{r['r2_shuffled']:+.3f}]{extra} | {r['n_fit']} | {r['n_held']} | {r['n_fit_episodes']} / {r['n_held_episodes']} |")
        L.append("")
    # ---- rung verdicts
    clears = {tgt: {} for tgt in BAR}
    best_tab = {tgt: {} for tgt in BAR}
    for rung, us in by_rung.items():
        for tgt in BAR:
            per = []
            for u in us:
                b = best(u["rows"], tgt)
                if b:
                    per.append((u["tag"], b[0], b[1]))
            best_tab[tgt][rung] = per
            if rung == "R1":
                clears[tgt][rung] = any(v >= BAR[tgt] for _, v, _ in per) if per else None          # an R1 rung = one checkpoint
            else:
                clears[tgt][rung] = (all(v >= BAR[tgt] for _, v, _ in per) if per else None)        # every training seed it has
    out["best_by_rung"] = best_tab; out["clears"] = clears
    L += ["## Deciding quantity: best held-out R2 over the rung's readers (a maximum over a small fixed set, optimistic)", "",
          "| rung | target | best per tag (reader) | bar | clears |", "|---|---|---|---|---|"]
    for tgt in BAR:
        for rung in sorted(best_tab[tgt]):
            per = "; ".join(f"{t}: {v:+.3f} ({rd})" for t, v, rd in best_tab[tgt][rung])
            L.append(f"| {rung} | {tgt} | {per} | {BAR[tgt]:.2f} | {clears[tgt][rung]} |")
    L.append("")
    # ---- outcomes
    L += ["## Committed outcomes, evaluated mechanically (per target)", ""]
    out["outcomes"] = {}
    complete_rungs = {f"R{i}" for i in range(7)}
    for tgt in BAR:
        c = clears[tgt]
        miss = sorted(complete_rungs - set(c))
        get = lambda r: c.get(r)
        hold = []
        if get("R1"):
            hold.append("LADDER-READS")
        if get("R0") is not None and get("R1") is not None and get("R2") is not None and not (get("R0") or get("R1") or get("R2")) and (get("R3") or get("R4")):
            hold.append("POOL-LOSS")
        if get("R3") is not None and get("R4") is not None and not (get("R3") or get("R4")) and (get("R5") or get("R6")):
            hold.append("ADAPTER-LOSS")
        if not miss and not any(c.values()):
            hold.append("NOT-READ-FROM-VISION")
        label = hold if hold else (["UNDETERMINED (rungs missing: " + ", ".join(miss) + ")"] if miss else ["UNLISTED COMBINATION (no committed outcome matches; the rung table above is the result)"])
        trig = [TRIGGER[h] for h in hold]
        out["outcomes"][tgt] = dict(outcome=label, missing_rungs=miss, triggers=trig)
        L.append(f"* **{tgt}: {' + '.join(label)}.** " + (f"Committed trigger: {' | '.join(trig)}" if trig else ""))
    L += ["* **Every outcome:** a card for measured ego state as part of the world-model state is written (speed and yaw rate are measured quantities, D-PI-YAWRATE-1, R-EGO); that arm does not depend on this read.", ""]
    L += ["## Limits",
          "* **n.** 53 fit episodes (744 windows; 11,904 one-frame samples) and 17 held-out episodes (238 windows; 3,808 one-frame samples; 3,094 four-frame samples). R2 has no interval here; one-seed readers differ by seed noise that this card does not measure, so differences under 0.05 carry no weight. Four frames reduce the sample count per window from 16 to 13.",
          "* A rung's deciding value is a maximum over readers (ridge, MLP or conv, one or four frames), an optimistic read. A readout bound at this n does not show absence of the information (`A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY`).",
          "* Grid tokens are read as a 4 x 10 grid with token index = row * 10 + column (assumed from the `pool4x10` cache name). Tier T0: never a driving result. STRATEGIC family: PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL have no metric here for a stated reason (a readout R2 only)."]
    json.dump(out, open(os.path.join(HERE, "RESULT_DECINFO.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(HERE, "RESULT_DECINFO.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
