"""PREREG_V8F_DECN report: reads runs/*.json (+ scratch control files), writes RESULT_DECN.md + RESULT_DECN.json. Evaluates the card's outcome rules mechanically
(IMPL_CHOICES_DECN.md section 7). Tier T0, no interval. Usage: python report_decn.py  (plain python)"""
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SCRATCH = "F:/TanitAD-scratch/decn_scratch/run1"
BAR = {"speed": 0.70, "yaw": 0.50}
RISE_MIN = 0.10
CARD = "PREREG_V8F_DECN.md prefix sha256 cfa5bb1fdda36a77efd6bf203232dbf5cf0d15f08a2189ad908e2436509619e5"
RUNG_DESC = {"R0": "M2b z1 (128-D)", "R1": "ladder z1 (128-D)", "R2": "pooled adapter token (256-D)"}
TRIGGER = {
    "N-LIMITED": "the decoder is refitted on N1053 and PLANTEST3 is rerun on FLAT1 DEC-END (plain and shrink), SCORE windows, both M2b seeds; its NO-GAIN at N53 is reported with this limit stated.",
    "RISES": "the full 4,572-clip probe on box 2 (the DECINFO trigger), as one job after the ladder, before any conclusion about the vision chain.",
    "FLAT": "more data does not lift the read at this scale. The measured-speed world-model arm (`theory/egowm/`) goes first, and the full-n probe is dropped from the box-2 queue unless the PI asks for it.",
}
NORD = {"N53": 0, "N194": 1}


def norder(n):
    if n in NORD:
        return NORD[n]
    return 2 if n == "N1053" else 3


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def f(x):
    return f"{x:+.3f}"


def main():
    units = [jload(p) for p in sorted(glob.glob(os.path.join(RUNS, "*.json")))]
    by = {}
    for u in units:
        by[(u["rung"], u["tag"], u["N"])] = u
    out = {"tier": "T0", "card": CARD, "units": sorted("_".join(k) for k in by)}
    Ns = sorted({k[2] for k in by}, key=norder)
    L = ["# RESULT_DECN: is the weak speed and yaw-rate readout limited by the 53 fit episodes?",
         "Tier **T0**, a readout diagnostic on the PLANTEST3 TUNE inner split: the same 17 held-out episodes (238 windows, 16 frames) at every N; fit sets N53 (53 TUNE fit episodes), N194, N1053 (train clips streamed read-only from box 2). No interval, as the card says: differences under 0.05 carry no weight. "
         f"MEASURED: `decinfo/decn/runs/*.json` (one file per rung, tag and N). Card: `../../PREREG_V8F_DECN.md` ({CARD.split('prefix ')[1][:16]}...). Choices: `IMPL_CHOICES_DECN.md`.", ""]
    # ---------------------------------------------------------------- controls
    viol, c = [], {}
    c1p = f"{SCRATCH}/control1.json"
    c1 = jload(c1p) if os.path.exists(c1p) else None
    c["1"] = bool(c1 and c1["passed"])
    if not c["1"]:
        viol.append("control 1 missing or failed")
    c2 = []
    for (r, t, n), u in by.items():
        for tg, d in u["meta"].get("control2_reproduction", {}).items():
            c2.append(d["passed"])
            if not d["passed"]:
                viol.append(f"control 2: {t} {tg} diff {d['abs_diff']:.4f}")
    c["2"] = bool(c2) and len(c2) == 4 and all(c2)
    if len(c2) != 4:
        viol.append(f"control 2: {len(c2)} of 4 reproduction values present")
    shuf = [(u, r) for u in units for r in u["rows"]]
    bad3 = [(u, r) for u, r in shuf if r["r2_shuffled"] > 0.05]
    for u, r in bad3:
        viol.append(f"control 3: {u['rung']} {u['tag']} {u['N']} {r['reader']} m={r['m']} {r['target']} shuffled R2 {r['r2_shuffled']:+.3f}")
    c["3"] = bool(shuf) and not bad3
    pl = [(u, p) for u in units for p in u["planted"]]
    bad4 = [(u, p) for u, p in pl if not p["passed"]]
    for u, p in bad4:
        viol.append(f"control 4: {u['rung']} {u['tag']} {u['N']} m={p['m']} planted {p['r2_signal']:.3f} zeroed {p['r2_zeroed']:.3f}")
    c["4"] = bool(pl) and not bad4
    digs = {u["meta"]["held_digest"] for u in units}
    c["5"] = len(digs) == 1 and all(u["meta"]["n_held_windows"] == 238 and u["meta"]["n_held_episodes"] == 17 for u in units) and bool(units)
    if not c["5"]:
        viol.append(f"control 5: held-out digests {len(digs)} distinct or held-out size changed")
    stopped = os.path.join(HERE, "STREAM_STOPPED.txt")
    stop_txt = open(stopped).read().strip() if os.path.exists(stopped) else ""
    shards = len(glob.glob(f"{SCRATCH}/lat/*.npz"))
    c["6"] = ("md5 mismatch" not in stop_txt)
    if not c["6"]:
        viol.append("control 6: " + stop_txt)
    complete = shards == 1000 and "N1053" in Ns
    vtr = f"{SCRATCH}/train_v_vs_track.json"
    vmax = None
    if os.path.exists(vtr):
        vals = [v for v in jload(vtr).values() if v is not None]
        vmax = max(vals) if vals else None
    valid = all(c.values()) and not viol
    out["controls"] = dict(c, violations=viol, valid=valid, complete=complete, shards=shards, stream_stopped=stop_txt, control1=c1, train_v_vs_track_max_abs_diff=vmax)
    L += ["## Validity first (the read is INVALID if one control fails)",
          f"* **Verdict on validity: {'VALID' if valid else 'INVALID or incomplete (see violations)'}.** Stream: {shards} of 1,000 train clips with latents" + (f"; stopped: {stop_txt}" if stop_txt else "") + f". N1053 {'complete' if complete else 'NOT complete: outcome lines read UNDETERMINED'}.",
          f"* Control 1, target tie (cache `v` of the eval clips against the eval target on the {c1['n_tune_windows'] if c1 else '?'} TUNE windows): R2 {c1['r2']:.5f}, mean absolute difference {c1['mean_abs_diff']:.5f} m/s over {c1['n_values']} values (bars: R2 at or above 0.99, difference at or below 0.10): {'pass' if c['1'] else 'FAIL'}." if c1 else "* Control 1: not run.",
          f"* Control 2, reproduction of DECINFO's R0 ridge m=1 at N53 (within 0.01, speed and yaw, both seeds): {sum(c2)} of {len(c2)} pass.",
          f"* Control 3, shuffled targets (at or below 0.05): {len(shuf) - len(bad3)} of {len(shuf)} reader rows pass (every N, input, reader, m, target).",
          f"* Control 4, planted signal (at or above 0.80, zeroed at or below 0.02): {len(pl) - len(bad4)} of {len(pl)} unit/m inputs pass (the N1053 units included).",
          f"* Control 5, split: held-out block digest identical across all {len(units)} units: {len(digs) == 1}; 238 held-out windows, 17 episodes; guards asserted in code and in `test_decn_controls.py` (`TEST_DECN.log`).",
          f"* Control 6, clip integrity (md5 against box 2's manifest, mismatch stops the run): {'no mismatch recorded' if c['6'] else 'MISMATCH'}.",
          f"* Violations: {'none' if not viol else '; '.join(viol)}",
          f"* Logged, deciding nothing: train-clip cache `v` against the native-track speed at the same frames, maximum absolute difference over the streamed clips: {vmax if vmax is None else f'{vmax:.6f}'}.", ""]
    # ---------------------------------------------------------------- sample counts
    L += ["## Sample counts (episodes, windows, samples)", "", "| rung | tag | N | fit episodes | fit windows | fit samples m=1 / m=4 (speed) | held-out episodes / windows | held-out samples m=1 / m=4 |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(by, key=lambda k: (k[0], k[1], norder(k[2]))):
        u = by[k]
        g = lambda m, t="speed", rd="ridge": [r for r in u["rows"] if r["reader"] == rd and r["m"] == m and r["target"] == t][0]
        L.append(f"| {k[0]} | {k[1]}{' (added after the freeze)' if u['meta'].get('added_after_freeze') else ''} | {k[2]} | {u['meta']['n_fit_episodes']} | {u['meta']['n_fit_windows']} | {g(1)['n_fit']} / {g(4)['n_fit']} | {u['meta']['n_held_episodes']} / {u['meta']['n_held_windows']} | {g(1)['n_held']} / {g(4)['n_held']} |")
    L.append("")
    # ---------------------------------------------------------------- per target tables
    tags = sorted({(k[0], k[1]) for k in by})
    for tgt, name in (("speed", "Speed (m/s)"), ("yaw", "Yaw rate (rad/s)")):
        L += [f"## {name}: held-out R2 per input, reader and N (bar {BAR[tgt]:.2f}); shuffled-target control in brackets", "",
              "| rung | input | tag | reader, m | " + " | ".join(Ns) + " |", "|---|---|---|---|" + "---|" * len(Ns)]
        for rung, tag in tags:
            for rd, m in (("ridge", 1), ("ridge", 4), ("mlp", 1), ("mlp", 4)):
                cells = []
                for N in Ns:
                    u = by.get((rung, tag, N))
                    if not u:
                        cells.append("-"); continue
                    r = [r for r in u["rows"] if r["reader"] == rd and r["m"] == m and r["target"] == tgt][0]
                    cells.append(f"{f(r['r2'])} [{f(r['r2_shuffled'])}]" + (f" (epochs {r['detail']['epochs']}/{r['detail']['cap']})" if rd == 'mlp' else ""))
                L.append(f"| {rung} | {RUNG_DESC[rung]} | {tag}{' (added after the freeze)' if by[next(k for k in by if k[:2] == (rung, tag))]['meta'].get('added_after_freeze') else ''} | {rd}, m={m} | " + " | ".join(cells) + " |")
        L.append("")
    # ---------------------------------------------------------------- cell values, rises
    def cell(rung, tag, N, tgt):
        u = by.get((rung, tag, N))
        if not u:
            return None
        rs = [r for r in u["rows"] if r["target"] == tgt]
        b = max(rs, key=lambda r: r["r2"])
        return b["r2"], f"{b['reader']} m={b['m']}"

    def same_reader_rise(rung, tag, N, tgt):
        a, b = by.get((rung, tag, "N53")), by.get((rung, tag, N))
        if not a or not b:
            return None
        best = None
        for ra in a["rows"]:
            if ra["target"] != tgt:
                continue
            rb = [r for r in b["rows"] if r["target"] == tgt and r["reader"] == ra["reader"] and r["m"] == ra["m"]][0]
            d = rb["r2"] - ra["r2"]
            if best is None or d > best[0]:
                best = (d, f"{ra['reader']} m={ra['m']}")
        return best

    DEC = [("R0", "M2b_s0"), ("R0", "M2b_s1"), ("R2", "M2b_s0"), ("R2", "M2b_s1")]
    out["cells"] = {}
    out["outcomes"] = {}
    L += ["## Deciding quantity: best held-out R2 over the readers per input, N and target (a maximum over four readers, optimistic)", "",
          "| target | input | tag | " + " | ".join(f"best {N}" for N in Ns) + " | rise N1053 - N53 (cell value) | rise, same reader (max over readers) |", "|---|---|---|" + "---|" * len(Ns) + "---|---|"]
    for tgt in BAR:
        for rung, tag in tags:
            cs = [cell(rung, tag, N, tgt) for N in Ns]
            out["cells"][f"{tgt}|{rung}|{tag}"] = {N: (c_ and c_[0]) for N, c_ in zip(Ns, cs)}
            c53, cbig = cell(rung, tag, "N53", tgt), cell(rung, tag, "N1053", tgt)
            rise = (cbig[0] - c53[0]) if (c53 and cbig) else None
            sr = same_reader_rise(rung, tag, "N1053", tgt)
            L.append(f"| {tgt} | {rung} | {tag} | " + " | ".join(("-" if c_ is None else f"{f(c_[0])} ({c_[1]})") for c_ in cs) + f" | {'-' if rise is None else f(rise)} | {'-' if sr is None else f'{f(sr[0])} ({sr[1]})'} |")
    L.append("")
    # ---------------------------------------------------------------- outcomes
    L += ["## Committed outcomes, evaluated mechanically (per target; rules in `IMPL_CHOICES_DECN.md` section 7)", ""]
    for tgt in BAR:
        miss = [f"{r} {t}" for r, t in DEC if cell(r, t, "N1053", tgt) is None or cell(r, t, "N53", tgt) is None]
        why = []
        if not valid:
            why.append("controls not all valid")
        if miss:
            why.append("N1053 or N53 cells missing for " + ", ".join(miss))
        if not complete:
            why.append("N1053 stream incomplete")
        clears = {r: [cell(r, t, "N1053", tgt)[0] >= BAR[tgt] for rr, t in DEC if rr == r] for r in ("R0", "R2") if not miss}
        rises = {r: [cell(r, t, "N1053", tgt)[0] - cell(r, t, "N53", tgt)[0] for rr, t in DEC if rr == r] for r in ("R0", "R2") if not miss}
        sr_rises = {r: [same_reader_rise(r, t, "N1053", tgt)[0] for rr, t in DEC if rr == r] for r in ("R0", "R2") if not miss}
        if why:
            out["outcomes"][tgt] = dict(outcome="UNDETERMINED", reasons=why)
            L.append(f"* **{tgt}: UNDETERMINED** ({'; '.join(why)}). The tables above are the result. If a committed outcome held on the cells present it would be: " + (_provisional(tgt, cell, DEC, BAR, RISE_MIN) if not miss else "not evaluable."))
            continue
        nl = any(all(v) for v in clears.values())
        ri = any(all(x >= RISE_MIN for x in v) for v in rises.values())
        ri_sr = any(all(x >= RISE_MIN for x in v) for v in sr_rises.values())
        lab = "N-LIMITED" if nl else ("RISES" if ri else "FLAT")
        lab_sr = "N-LIMITED" if nl else ("RISES" if ri_sr else "FLAT")
        out["outcomes"][tgt] = dict(outcome=lab, outcome_same_reader_form=lab_sr, clears=clears, rises=rises, same_reader_rises=sr_rises, trigger=TRIGGER[lab])
        L.append(f"* **{tgt}: {lab}.** Clears the bar {BAR[tgt]:.2f} at N1053 (R0 s0/s1, R2 s0/s1): " + ", ".join(f"{r}: {v}" for r, v in clears.items()) + f". Rise N1053 - N53 per seed (R0, R2): " + ", ".join(f"{r}: " + "/".join(f(x) for x in v) for r, v in rises.items()) +
                 f". Same-reader form gives {lab_sr}" + ("" if lab_sr == lab else " (differs; the cell-value form decides)") + f". Committed trigger: {TRIGGER[lab]}")
    L += ["", "## Limits",
          "* **n.** N53 is 53 fit episodes (744 windows); N194 is 194; N1053 is 1,053. The held-out set is the same 17 episodes (238 windows; 3,808 one-frame, 3,094 four-frame samples) at every N. R2 has no interval here; one-seed readers differ by seed noise that this card does not measure, so differences under 0.05 carry no weight.",
          "* A cell value is a maximum over four readers (ridge and MLP, one and four frames), an optimistic read. MLP epoch cap: 120 at N53, 60 at N194 and N1053; `epochs x/cap` is printed per cell, and a row at its cap may be under-trained. A readout bound at this n and reader family does not show absence of the information.",
          "* Train windows are the 13 of 17 per clip whose 16 frames are all real. Train-clip speed is the cache `v`; its tie to the eval target is control 1. Ladder rows (R1) are read alongside and decide nothing; tags outside the frozen four are marked as added.",
          "* Tier T0: a readout R2 only, never a driving result. LONGITUDINAL, LATERAL, TACTICAL: no metric here, for that reason; STRATEGIC: PAUSED (PI item 25), not a pass."]
    json.dump(out, open(os.path.join(HERE, "RESULT_DECN.json"), "w", encoding="utf-8"), indent=1, default=str)
    open(os.path.join(HERE, "RESULT_DECN.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:14]))


def _provisional(tgt, cell, DEC, BAR, RISE_MIN):
    parts = []
    for r in ("R0", "R2"):
        vals = [(t, cell(r, t, "N1053", tgt), cell(r, t, "N53", tgt)) for rr, t in DEC if rr == r]
        if any(b is None or a is None for _, a, b in vals):
            continue
        parts.append(f"{r}: " + ", ".join(f"{t} {f(a[0])} (rise {f(a[0] - b[0])})" for t, a, b in vals))
    return "; ".join(parts) if parts else "no N1053 cell yet."


if __name__ == "__main__":
    main()
