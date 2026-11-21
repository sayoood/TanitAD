#!/usr/bin/env python3
"""Record one evaluated REFe snapshot in the package and the steering files -- the steps done by hand for snapshot 014
(2026-09-27), made one command so every later snapshot is recorded the same way.

Reads ONLY banked artifacts: the learning-curve point (data/refe_navtest/points/sub200_epNNN.json), the E-6 readout and
the Amendment 4 read (eval/raw/e6_sub200_epNNN/{readout,amendment4}.json, banked by eval_snapshot.sh). Then:
  1. the like-for-like pair against the previous snapshot under ONE selection rule (snapshot_pair_under_rule.compare)
     -> eval/raw/e6_sub200_epNNN/pair_epPPP_epNNN.json (EXPLORATORY, same 200 tokens)
  2. banks the point JSON into eval/raw/points/ and appends its md5 + the snapshot's md5 to MANIFEST.txt
  3. MODEL_REGISTRY.md, REFe Status row: the PDMS list gains the point; a dated sentence with the result
  4. GOALS_AND_CLAIMS.md: dated updates to D-REFE-SEL-1, D-REFE-OPSWITCH-1 and D-REFE-COMFORT-1
Line endings are preserved byte for byte; a snapshot already in the registry's PDMS list is refused; --dry-run prints
every text and writes nothing.

    python eval/record_snapshot.py --n 015 --prev 014 --snap-log <eval_snap_015.log> [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snapshot_pair_under_rule as SP  # noqa: E402

PTS = "D:/Projects/TanitAD/data/refe_navtest/points"
REG = "D:/Projects/TanitAD/Project Steering/MODEL_REGISTRY.md"
CLM = "D:/Projects/TanitAD/Project Steering/GOALS_AND_CLAIMS.md"
BERLIN = timezone(timedelta(hours=2))
# NAVSIM's own comfort label (label version 3) was first read by the trainer at the epoch-14 bank (2026-09-27 02:36
# Berlin; build_data.py v3_epoch). Snapshot N has trained N epochs, so snapshot 015 is the first trained on it.
V3_FIRST_SNAPSHOT = 15
OUT_NAMES = {"NC": "no-collision", "DAC": "drivable-area", "DDC": "driving-direction", "TTC": "time-to-collision", "C": "comfort"}
# SPEC Amendment 7 (adopted 2026-09-27): the executed plan's last-pose heading is repaired at eval. Its confirmation
# readout is the ONLY admissible size of the repair on its own; a pair that crosses the amendment quotes it.
A7_JSON = os.path.join(HERE, "raw", "a7_confirm", "a7_confirm_ep015.json")
# SPEC Amendment 8 (ADOPTED 2026-09-28): the goal is sanitised when the ego lies > D = 20 m from its route. The census
# (model-free, computed before any confirmation output) names the tokens it fires on; when only ONE side of a pair
# has it, those tokens are left out, and on the rest the two points differ only by the model (gate (e): identical).
A8_CENSUS = os.path.join(HERE, "..", "raw", "2026-09-28-goal-clamp", "route_cover_census_navtest_full.json")
A8_D_M = 20.0


def a8_triggered(tokens) -> list:
    rows = json.load(open(A8_CENSUS, encoding="utf-8"))["rows"]
    d = dict(rows) if isinstance(rows, list) else rows
    return sorted(t for t in tokens if d[t]["ego_to_route_m"] > A8_D_M)


def ci(c, d=2, sign=False):
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return "[" + f.format(c[0]) + ", " + f.format(c[1]) + "]"


def side(a):
    lo, hi = a["ci95"]
    return "above chance" if lo > 0.5 else "backwards" if hi < 0.5 else "at chance"


def point(n):
    """the learning-curve point, read the way report/build_data.py reads it"""
    d = json.load(open(f"{PTS}/sub200_ep{n}.json", encoding="utf-8"))
    assert d.get("verdict") == "OK", d.get("verdict")
    fl = d["floors"]
    arms, pairs = fl["arms"], fl["pairs"]
    x = arms["REFe"].get("interval") or {}
    lo, hi = x.get("lo", x.get("ci_lo")), x.get("hi", x.get("ci_hi"))

    def pr(k):
        p, y = pairs[k], pairs[k].get("interval") or {}
        return {"delta": p["delta_x100"], "wins": p["wins"], "ties": p["ties"], "losses": p["losses"],
                "lo": round(100 * y["lo"], 2), "hi": round(100 * y["hi"], 2), "separated": bool(y.get("separated"))}

    fam = ((d.get("families") or {}).get("families") or {}).get("REFe") or {}
    lon, lat, tac = fam.get("longitudinal", {}), fam.get("lateral", {}), fam.get("tactical", {})
    return {"pdms": d["score"]["summary_x100_4dp"]["PDMS"], "ci": [round(100 * lo, 2), round(100 * hi, 2)],
            "rule": (d.get("seam") or {}).get("rule") or "v2_shape", "vs_stop": pr("REFe__minus__STOP"),
            # the seam's `repair_last_heading` (absent = before Amendment 7 = False), read as report/build_data.py reads it
            "repair": bool((d.get("seam") or {}).get("repair_last_heading", False)),
            # the seam's `sanitize_goal` (absent = before Amendment 8 = False)
            "sanitize": bool((d.get("seam") or {}).get("sanitize_goal", False)),
            "vs_cv": pr("REFe__minus__CV"), "zeroed": arms["REFe"].get("zeroed_by_NC_or_DAC"),
            "sub": {k: d["score"]["summary_x100_4dp"][k] for k in ("NC", "DAC", "EP", "TTC", "C", "DDC")},
            "fam": {"speed_mae": lon.get("speed_mae_mps"), "speed_bias": lon.get("speed_bias_mps"),
                    "progress": (lon.get("ego_progress") or {}).get("progress_ratio_median"),
                    "heading": lat.get("heading_mae_deg"), "curvature": lat.get("curvature_mae_1pm"),
                    "cross": lat.get("cross_mae_m"), "goal": (tac.get("goal_setting") or {}).get("goal_point_error_m")}}


def pair_text(P, v1, v2, rep_a, rep_b, a7=None):
    """the pair sentence. Two points that differ in the Amendment-7 repair are NOT like for like: the step then carries
    the repair itself, so no separation is claimed for it and the repair's own confirmed size is quoted beside it."""
    nums = (f"{v1['b_minus_a']:+.2f} {ci(v1['ci95'], sign=True)} under NAVSIM v1 and {v2['b_minus_a']:+.2f} "
            f"{ci(v2['ci95'], sign=True)} under the v2 shape")
    if rep_a == rep_b:
        return (f"like for like vs after epoch {P} (both re-selected from their E-6 tables with ONE rule, same tokens, "
                f"EXPLORATORY): {v1['b_minus_a']:+.2f} {ci(v1['ci95'], sign=True)} under NAVSIM v1"
                f"{' (separated)' if v1['separated'] else ' (not separated)'} and {v2['b_minus_a']:+.2f} "
                f"{ci(v2['ci95'], sign=True)} under the v2 shape")
    which = "WITHOUT" if not rep_a else "WITH"
    other = "WITH" if not rep_a else "WITHOUT"
    rep = ""
    if a7:
        st = a7["statistic"]
        rep = (f"; the repair ALONE read {st['delta']:+.2f} {ci(st['ci95'], sign=True)} after epoch 15 on "
               f"{a7['tokens']['n']} fresh tokens (`…/eval/raw/a7_confirm/a7_confirm_ep015.json`)")
    return (f"NOT like for like vs after epoch {P}: that point is scored {which} SPEC Amendment 7's last-pose heading "
            f"repair and this one {other} it, so the step on the same tokens ({nums}) carries the repair itself and "
            f"is not a training effect -- no separation is claimed for it{rep}")


def load(p):
    b = open(p, "rb").read()
    return b.decode("utf-8"), b.count(b"\r\n"), b.count(b"\n")


def save(p, s, crlf, lf):
    out = s.encode("utf-8")
    assert out.count(b"\r\n") == crlf and out.count(b"\n") == lf, (p, "line endings changed")
    open(p, "wb").write(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", required=True, help="snapshot number, e.g. 015")
    ap.add_argument("--prev", required=True, help="the previous evaluated snapshot, e.g. 014")
    ap.add_argument("--snap-log", required=True, help="eval_snap_NNN.log (carries the snapshot's md5 as fetched)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--registry", default=REG, help="override for a test copy")
    ap.add_argument("--claims", default=CLM, help="override for a test copy")
    a = ap.parse_args()
    n, p = a.n.zfill(3), a.prev.zfill(3)
    N, P = int(n), int(p)
    raw = os.path.join(HERE, "raw", f"e6_sub200_ep{n}")
    rd = json.load(open(os.path.join(raw, "readout.json"), encoding="utf-8"))
    a4 = json.load(open(os.path.join(raw, "amendment4.json"), encoding="utf-8"))
    pt = point(n)
    done_at = datetime.fromtimestamp(os.path.getmtime(os.path.join(raw, "amendment4.json")), BERLIN).strftime("%Y-%m-%d %H:%M")
    m = re.search(rf"fetched snap_epoch{n}\.pt md5 ([0-9a-f]{{32}})", open(a.snap_log, encoding="utf-8").read())
    assert m, "no snapshot md5 in the eval log"
    snap_md5 = m.group(1)

    # 1. the pair against the previous snapshot (like for like only if both sides share the Amendment-7 repair state)
    san_a, san_b = point(p)["sanitize"], pt["sanitize"]
    excl = a8_triggered(json.load(open(SP.TOK, encoding="utf-8"))["tokens"]) if san_a != san_b else None
    pair = SP.compare(f"sub200_ep{p}", f"sub200_ep{n}", a.boot, exclude=excl)
    pair["sanitize_goal"] = {"a": san_a, "b": san_b, "excluded_tokens": len(excl or [])}
    v1, v2 = pair["rules"]["navsim_v1"], pair["rules"]["v2_shape"]
    rep_a, rep_b = point(p)["repair"], pt["repair"]
    a7 = json.load(open(A7_JSON, encoding="utf-8")) if rep_a != rep_b else None
    pair["repair_last_heading"] = {"a": rep_a, "b": rep_b, "like_for_like": rep_a == rep_b}
    pair_txt = pair_text(P, v1, v2, rep_a, rep_b, a7)
    if excl:
        pair_txt += (f" -- on the {pair['N']} tokens where SPEC Amendment 8's goal sanitisation cannot fire: the "
                     f"{len(excl)} it fires on are left out, since only after epoch {N if san_b else P} is scored with it")

    # the numbers every text below quotes
    A_ = rd["a_pdms"]
    sk = A_["selection_skill"]
    auc = {k: rd["c_components"][k]["within_token_auc"] for k in OUT_NAMES if "within_token_auc" in rd["c_components"].get(k, {})}
    pl = rd["d_path_length"]
    vs = pt["vs_stop"]
    stop_txt = f"{vs['delta']:+.2f} [{vs['lo']:+.2f}, {vs['hi']:+.2f}] vs standing still, " + ("separated" if vs["separated"] else "not separated")
    c_label = "NAVSIM's own comfort label (label version 3)" if N >= V3_FIRST_SNAPSHOT else "the teacher's comfort label"
    ca = auc["C"]

    rep_txt = "; executed plan with the Amendment-7 last-pose heading repair" if pt["repair"] else ""
    if pt["sanitize"]:
        rep_txt += "; goal sanitised off-route (SPEC Amendment 8)"
    reg_sentence = (f" ⭐ **After epoch {N} (evaluated {done_at} Berlin; picked with {pt['rule']}{rep_txt}; scorer trained on {c_label}): "
                    f"PDMS {pt['pdms']:.2f} {ci(pt['ci'])}, {stop_txt}; E-6 best {A_['oracle']['mean']:.1f} / pick "
                    f"{A_['actual']['mean']:.1f} / random {A_['random']['mean']:.1f}, skill {sk['value']:.3f} {ci(sk['ci95'], 3)} -> "
                    f"{rd['verdict']['outcome']}; {pair_txt}; comfort within-scene AUC {ca['mean']:.3f} {ci(ca['ci95'], 3)} "
                    f"({side(ca)}).** (`…/eval/RESULT_E6_sub200_ep{n}.md`, `…/eval/RESULT_A4_sub200_ep{n}.md`, "
                    f"`…/eval/raw/e6_sub200_ep{n}/`, `…/eval/record_snapshot.py`).")
    sel = (f"**Update after epoch {N} ({done_at} Berlin): skill {sk['value']:.3f} {ci(sk['ci95'], 3)} -> {rd['verdict']['outcome']}; "
           f"best {A_['oracle']['mean']:.1f} / pick {A_['actual']['mean']:.1f} / random {A_['random']['mean']:.1f}; within-scene AUC "
           + ", ".join(f"{k} {auc[k]['mean']:.2f} {ci(auc[k]['ci95'])}" for k in ("NC", "DAC", "TTC", "DDC", "C") if k in auc)
           + f"; the aggregate's within-scene Spearman with path length {pl['within_token_spearman_agg_vs_length']['mean']:+.2f} where the "
           f"true PDMS has {pl['within_token_spearman_truePDMS_vs_length']['mean']:+.2f}; the pick is "
           f"{pl['pick_over_mean']['mean']:.2f}x the mean path length.** `eval/RESULT_E6_sub200_ep{n}.md`.")
    pa4 = a4["primary"]
    ops = (f"**{'PRIMARY' if pa4['decides'] else 'Reported, not gating'} (after epoch {N}, rank {a4['rank_among_onpolicy_snapshots']}"
           + ("" if pa4["decides"] else f"; the test stays {pa4['decided_earlier_by']['outcome']}, decided by {pa4['decided_earlier_by']['name']}")
           + f"): {pa4['outcome']}, skill {pa4['skill']:.3f} {ci(pa4['ci95'], 3)} ({a4['eligibility']}); pick {stop_txt}; {pair_txt}.** "
           f"`eval/RESULT_A4_sub200_ep{n}.md` · `eval/raw/e6_sub200_ep{n}/`.")
    cmf = (f"**After epoch {N} (scorer trained on {c_label}): comfort within-scene AUC {ca['mean']:.3f} {ci(ca['ci95'], 3)} -- "
           f"{side(ca)}; pooled AUC {rd['c_components']['C']['pooled_auc']:.3f}, "
           f"{'FAILING' if rd['c_components']['C']['FAILING'] else 'not failing'} the pre-registered bar"
           + (" -- the first read of the label change." if N == V3_FIRST_SNAPSHOT else ".") + f"** `eval/RESULT_E6_sub200_ep{n}.md`.")

    # 3. the registry's Status row
    s_reg, c_reg, l_reg = load(a.registry)
    rows = [ln for ln in s_reg.split("\n") if ln.startswith("| **Status** |") and "A1_sub200_tokens.json" in ln]
    assert len(rows) == 1, len(rows)
    row = rows[0]
    mm = re.search(r"PDMS after epochs ([0-9/]+) = \*\*([0-9. /]+)\*\*", row)
    assert mm, "no PDMS list in the REFe Status row"
    eps = mm.group(1).split("/")
    assert str(N) not in eps, f"after epoch {N} is already in the registry's PDMS list: refusing to record it twice"
    assert eps[-1] == str(P), f"the list ends at epoch {eps[-1]}, not the previous snapshot {P}"
    new_list = f"PDMS after epochs {mm.group(1)}/{N} = **{mm.group(2)} / {pt['pdms']:.2f}**"
    anchor = " The trainer's `traj_L1` is a TRAINING curve"
    assert row.count(anchor) == 1
    new_row = row.replace(mm.group(0), new_list).replace(anchor, reg_sentence + anchor)

    # 4. the claims
    s_clm, c_clm, l_clm = load(a.claims)
    nl = "\r\n" if c_clm == l_clm else "\n"
    L = s_clm.split(nl)
    upd = {}
    for key, text in (("D-REFE-SEL-1", sel), ("D-REFE-OPSWITCH-1", ops), ("D-REFE-COMFORT-1", cmf)):
        i = [k for k, x in enumerate(L) if x.startswith(f"| **{key}** |")]
        assert len(i) == 1, key
        assert f"after epoch {N} (" not in L[i[0]].lower(), f"{key} already records epoch {N}"
        head, sep, tail = L[i[0]].rpartition(" | **MEASURED**")
        assert sep, key
        upd[i[0]] = head + " " + text + sep + tail

    print("== pair:", json.dumps(pair["rules"]))
    print("== registry list:", new_list)
    print("== registry sentence:", reg_sentence)
    for k, t in (("SEL-1", sel), ("OPSWITCH-1", ops), ("COMFORT-1", cmf)):
        print(f"== {k}:", t)
    if a.dry_run:
        print("DRY RUN: nothing written")
        return 0

    json.dump(pair, open(os.path.join(raw, f"pair_ep{p}_ep{n}.json"), "w", encoding="utf-8"), indent=1)
    dst = os.path.join(HERE, "raw", "points", f"sub200_ep{n}.json")
    shutil.copyfile(f"{PTS}/sub200_ep{n}.json", dst)
    man = os.path.join(HERE, "raw", "points", "MANIFEST.txt")
    mb = open(man, "rb").read()
    assert f"sub200_ep{n}.json".encode() not in mb
    md5 = hashlib.md5(open(dst, "rb").read()).hexdigest()
    open(man, "ab").write(f"{md5}  sub200_ep{n}.json  {snap_md5}  snap_epoch{n}.pt\n".encode())
    save(a.registry, s_reg.replace(row, new_row), c_reg, l_reg)
    for k, v in upd.items():
        L[k] = v
    save(a.claims, nl.join(L), c_clm, l_clm)
    print(f"ZZRECORDED sub200_ep{n} point md5 {md5} snapshot md5 {snap_md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
