#!/usr/bin/env python3
"""Measure 5r (eval/PREREG_MEASURE5R.md, blob a8241136) -- gates G-R2, G-R3, G-R4, G-R5 (CPU only).

  gr2   flag OFF == v4: the modified labeller's lines on M5's first 2 chunks (<m5r>/gr2/sets) vs M5's own lines
        (<m5>/sets), set AND side lines, byte for byte with the timing fields (sec, at, labeller) excluded.
  gr34  W3's 200 navtest tokens labelled by the REAL CLI with --repair-last-heading (<v4_labels>/v5rep), read back
        THROUGH train.OnPolicyBank, joined slot by slot to the REPAIRED harness truth:
          originals  proptable/sub200_ep015_repaired/table.npz (like_for_like_016)
          copies     score/refe_sub200_ep015_f075rep_rNN (eval/m5_repaired_extras.py), joined by the source's rank
          STOP       score/refe_sub200_ep015_stopzeros (the repair is the identity on it)
        G-R3: NAVSIM drivable area (navsim_dac.violation == 0 vs harness DAC >= 0.999) and comfort (navsim_comfort >= 0.5
        vs harness C >= 0.999): 100 % on every slot. G-R4 (PREREG_MEASURE5 §2's rule, repaired NC): the 0.75x copies'
        collision agreement >= the originals' and >= 90 %, and neither disagreement direction > 3x the other.
        Also reported: every teacher component vs the harness per group, and the label-ranking ceiling.
  gr5   coverage of the v5 labels (<m5r>/sets): keys, label_version 5 + repair tag on every line, failed /
        selfcheck_failed from the status files.

    python eval/m5r_gates.py gr2 | gr34 | gr5          -> eval/raw/m5r/gate_<name>.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import validate_slow_labels_v4 as V  # noqa: E402

M5 = "D:/Projects/TanitAD/data/refe_m5"
M5R = "D:/Projects/TanitAD/data/refe_m5r"
NAV = "D:/Projects/TanitAD/data/refe_navtest"
TR = f"{NAV}/proptable/sub200_ep015_repaired/table.npz"
REP_CSV = f"{NAV}/score/refe_sub200_ep015_f075rep_r{{:02d}}/refe_sub200_ep015_f075rep_r{{:02d}}.csv"
STOP_CSV = f"{NAV}/score/refe_sub200_ep015_stopzeros/refe_sub200_ep015_stopzeros.csv"
OUTD = os.path.join(HERE, "raw", "m5r")
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")
TIMING = ("sec", "at", "labeller")


def key(r):
    return f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}"


def lines(d, kind):
    pat = "onpolicy_r0_*.jsonl" if kind == "onpolicy_set" else "slowaux_r0_*.jsonl"
    out = []
    for p in sorted(glob.glob(os.path.join(d, pat))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == kind:
                out.append(r)
    return out


def gr2() -> dict:
    res = {}
    for kind in ("onpolicy_set", "onpolicy_slowaux"):
        new = lines(f"{M5R}/gr2/sets", kind)
        ref: dict = {}
        for r in lines(f"{M5}/sets", kind):
            ref.setdefault(key(r), []).append({k: v for k, v in r.items() if k not in TIMING})
        same = diff = missing = 0
        ex = []
        keys_ok, keys_bad = set(), set()
        for r in new:                       # EVERY new line (a re-claimed chunk writes a key twice: both must match)
            s = {k: v for k, v in r.items() if k not in TIMING}
            cands = ref.get(key(r))
            if not cands:
                missing += 1
                keys_bad.add(key(r))
            elif any(json.dumps(s, sort_keys=True) == json.dumps(c, sort_keys=True) for c in cands):
                same += 1
                keys_ok.add(key(r))
            else:
                diff += 1
                keys_bad.add(key(r))
                if len(ex) < 3:
                    c = cands[0]
                    ex.append({"key": key(r), "differing_fields": sorted(k for k in set(s) | set(c) if s.get(k) != c.get(k))})
        res[kind] = {"new_lines": len(new), "identical": same, "different": diff, "no_reference": missing,
                     "distinct_keys_identical": len(keys_ok - keys_bad), "distinct_keys_bad": len(keys_bad),
                     "examples": ex}
    res["pass"] = (res["onpolicy_set"]["distinct_keys_identical"] == 32 and res["onpolicy_set"]["different"] == 0
                   and res["onpolicy_set"]["no_reference"] == 0
                   and res["onpolicy_slowaux"]["different"] == 0 and res["onpolicy_slowaux"]["no_reference"] == 0
                   and res["onpolicy_slowaux"]["distinct_keys_identical"] > 0)
    res["bar"] = "32/32 distinct keys; EVERY set and side line identical to M5's (timing fields excluded)"
    return res


def read_csv(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["token"] != "average":
                out[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in V.SUB))
    return out


def gr34() -> dict:
    import train as TRN
    sets = f"{V.WD}/v5rep"
    T = np.load(TR)
    tok = [str(t) for t in T["token"]]
    ix = {t: i for i, t in enumerate(tok)}
    top = np.load(V.SLOW_RANKS)["top"]
    Hc = {r: read_csv(REP_CSV.format(r, r)) for r in range(8)}
    stop_h = read_csv(STOP_CSV)
    bank = TRN.OnPolicyBank(sets, 64, 20)
    L = lines(sets, "onpolicy_set")
    aux = {r["token"]: r for r in lines(sets, "onpolicy_slowaux")}
    res = {"_label": "G-R3 / G-R4 (PREREG_MEASURE5R §5): v5 labels (C2 + --repair-last-heading) vs the REPAIRED harness, "
                     "W3's 200 navtest tokens, snapshot 015; EP reference = the human future (declared, as in M5)",
           "sets_dir": sets, "truth": {"originals": TR, "copies": REP_CSV, "stop": STOP_CSV},
           "trainer_loader": {"sets_loaded": len(bank.by), "set_lines": bank.n_rows, "incomplete": bank.n_incomplete,
                              "navsim_dac_sets": bank.n_navsim_dac},
           "label_versions": sorted({(r.get("label_version"), r.get("repair")) for r in L}, key=str)}
    res["label_versions"] = [list(x) for x in res["label_versions"]]
    groups = ("orig", "copy_f075", "stop")
    P = {g: {h: [] for h in HEAD + ("PDMS",)} for g in groups}
    nav = {"DAC": [], "C": []}
    mism_loader = rank_mismatch = no_h = 0
    seen = set()
    ceil = []
    for r in L:
        k = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
        if r["token"] in seen:
            continue
        seen.add(r["token"])
        e = bank.by.get(k)
        i = ix[r["token"]]
        sl = r.get("slow") or {}
        slot_of = {s: (sv, fv) for s, sv, fv in zip(sl.get("slots", []), sl.get("src", []), sl.get("factor", []))}
        top_i = [int(x) for x in top[i]]
        srcs = sorted(s for s in sl.get("src", []) if s >= 0)
        if srcs != sorted(top_i[:8]):
            rank_mismatch += 1
        tvals = [V.trainer_targets(t) for t in r["targets"]]
        if e is None or not np.array_equal(np.asarray(tvals, np.float32), e[2]):
            mism_loader += 1
        full = {}                                          # candidate id -> (label vector, harness pdms)
        for slot in range(64):
            t, lab = r["targets"][slot], tvals[slot]
            if slot in slot_of:
                s, f = slot_of[slot]
                if f == 0.0:
                    g, h = "stop", stop_h.get(r["token"])
                    cid = "stop"
                else:
                    g = "copy_f075"
                    rr = top_i.index(s)
                    h = Hc[rr].get(r["token"])
                    cid = f"c{rr}"
                if h is None or not h[0]:
                    no_h += 1
                    continue
                hs, hp = h[2], h[1]
            else:
                g, cid = "orig", slot
                if not T["valid"][i, slot]:
                    no_h += 1
                    continue
                hs, hp = tuple(T["sub"][i, slot]), float(T["pdms"][i, slot])
            full[cid] = (lab, hp)
            for j, hn in enumerate(HEAD):
                P[g][hn].append((lab[j], hs[j]))
            P[g]["PDMS"].append((float(V.v1_agg(lab)), hp))
            nav["DAC"].append((g, float(t["navsim_dac.violation"]) == 0.0, hs[1] >= 0.999))
            nav["C"].append((g, float(t["navsim_comfort"]) >= 0.5, hs[4] >= 0.999))
        ax = aux.get(r["token"])
        if ax is not None:
            for slot, t in zip(ax["slots"], ax["dropped_targets"]):
                if not T["valid"][i, slot]:
                    no_h += 1
                    continue
                lab = V.trainer_targets(t)
                hs, hp = tuple(T["sub"][i, slot]), float(T["pdms"][i, slot])
                full[slot] = (lab, hp)
                for j, hn in enumerate(HEAD):
                    P["orig"][hn].append((lab[j], hs[j]))
                P["orig"]["PDMS"].append((float(V.v1_agg(lab)), hp))
                nav["DAC"].append(("orig", float(t["navsim_dac.violation"]) == 0.0, hs[1] >= 0.999))
                nav["C"].append(("orig", float(t["navsim_comfort"]) >= 0.5, hs[4] >= 0.999))
        # label-ranking ceiling (reported): pairs with >= 1 extra, harness not tied
        ids = list(full)
        agg = np.array([float(V.v1_agg(full[c][0])) for c in ids])
        hp_ = np.array([full[c][1] for c in ids])
        ext = np.array([isinstance(c, str) for c in ids])
        num = den = 0.0
        for a_ in range(len(ids)):
            for b_ in range(a_ + 1, len(ids)):
                if not (ext[a_] or ext[b_]) or abs(hp_[a_] - hp_[b_]) <= 1e-9:
                    continue
                den += 1
                d = (agg[a_] - agg[b_]) * (hp_[a_] - hp_[b_])
                num += 1.0 if d > 0 else (0.5 if agg[a_] == agg[b_] else 0.0)
        if den:
            ceil.append(num / den)
    res["sets_read"] = len(seen)
    res["gates_loader"] = {"loader_equals_line_components_mismatch": mism_loader,
                           "sources_are_the_harness_top8_mismatch": rank_mismatch,
                           "labelled_slots_without_a_valid_harness_row": no_h}
    navres = {}
    for comp, rows in nav.items():
        a_ = np.array([x for _, x, _ in rows]); b_ = np.array([y for _, _, y in rows])
        navres[comp] = {"n": len(rows), "agree": float((a_ == b_).mean()) if rows else None,
                        "disagree": int((a_ != b_).sum()),
                        "by_group": {g: {"n": int(sum(1 for gg, _, _ in rows if gg == g)),
                                         "disagree": int(sum(1 for gg, x, y in rows if gg == g and x != y))}
                                     for g in groups}}
    res["G_R3"] = {"navsim": navres, "bar": "100 % agreement for DAC and comfort on every slot (64 + 8 + STOP)",
                   "pass": (navres["DAC"]["disagree"] == 0 and navres["C"]["disagree"] == 0
                            and navres["DAC"]["n"] == 200 * 73 and res["sets_read"] == 200 and mism_loader == 0
                            and rank_mismatch == 0 and no_h == 0)}
    teach = {g: {h: V._bin_stats(P[g][h]) if h in ("NC", "TTC", "C", "DAC", "DDC") else V._cont_stats(P[g][h])
                 for h in HEAD + ("PDMS",)} for g in groups}
    res["teacher_vs_harness"] = teach
    nc_o, nc_c = teach["orig"]["NC"], teach["copy_f075"]["NC"]
    a1, a2 = nc_c.get("label_fail_harness_pass", 0), nc_c.get("label_pass_harness_fail", 0)
    one_dir = max(a1, a2) > 3 * min(a1, a2)
    res["G_R4"] = {"copies_agree": nc_c.get("agree"), "originals_agree": nc_o.get("agree"),
                   "copies_label_collision_harness_no": a1, "copies_label_no_harness_collision": a2,
                   "one_directional": one_dir,
                   "bar": "copies agreement >= originals' AND >= 0.90 AND neither direction > 3x the other",
                   "pass": (nc_c.get("agree", 0) >= nc_o.get("agree", 1) and nc_c.get("agree", 0) >= 0.90
                            and not one_dir)}
    res["label_ranking_ceiling_extras"] = {"tokens": len(ceil), "mean": float(np.mean(ceil)) if ceil else None}
    return res


def gr5() -> dict:
    Ls = lines(f"{M5R}/sets", "onpolicy_set")
    keys = {key(r) for r in Ls}
    keys_q = set()
    for c in glob.glob(f"{M5R}/queue/props_r0_*.jsonl*"):
        for line in open(c, encoding="utf-8"):
            keys_q.add(key(json.loads(line)))
    lv = {}
    for r in Ls:
        t = (r.get("label_version"), r.get("repair"))
        lv[str(t)] = lv.get(str(t), 0) + 1
    st = [json.load(open(s)) for s in glob.glob(f"{M5R}/queue/status_r0_r*.json")]
    tot = {k: sum(x.get(k, 0) for x in st) for k in ("written", "failed", "selfcheck_failed", "skipped_ndiff")}
    m5keys = set(str(k) for k in np.load(f"{M5}/ft_data_train.npz")["key"]) | \
        set(str(k) for k in np.load(f"{M5}/ft_data_val.npz")["key"])
    res = {"set_lines": len(Ls), "keys_labelled": len(keys), "keys_in_chunks": len(keys_q),
           "keys_unlabelled": len(keys_q - keys), "label_version_repair": lv, **tot,
           "keys_equal_M5_usable_keys": keys == m5keys,
           "bar": "1,900 keys labelled, every line label_version 5 + repair, 0 failed, 0 selfcheck_failed"}
    res["pass"] = (len(keys) == 1900 and not (keys_q - keys) and list(lv) == ["(5, 'last_heading_hold')"]
                   and tot["failed"] == 0 and tot["selfcheck_failed"] == 0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=("gr2", "gr34", "gr5"))
    a = ap.parse_args()
    res = {"gr2": gr2, "gr34": gr34, "gr5": gr5}[a.gate]()
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, f"gate_{a.gate}.json")
    json.dump(res, open(p, "w", encoding="utf-8"), indent=1, default=str)
    if a.gate == "gr34":
        print(json.dumps({"G_R3": {"pass": res["G_R3"]["pass"], "DAC": res["G_R3"]["navsim"]["DAC"]["agree"],
                                   "C": res["G_R3"]["navsim"]["C"]["agree"]}, "G_R4": res["G_R4"],
                          "loader": res["gates_loader"], "sets": res["sets_read"],
                          "ceiling": res["label_ranking_ceiling_extras"]}, indent=1))
        ok = res["G_R3"]["pass"] and res["G_R4"]["pass"]
    else:
        print(json.dumps(res, indent=1, default=str)[:3000])
        ok = res["pass"]
    print(f"ZZM5R_{a.gate.upper()}_{'PASS' if ok else 'FAIL'} -> {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
