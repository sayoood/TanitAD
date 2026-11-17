"""AMEND_HWMPLAN_6 controls: AMEND_5's controls on the amend6 folders with C-RA-FIT replaced by C-FIT6, plus K-CONTROL, C-SEL6 and C-CHUNK6 (each with its regression arm).
  python amend6_controls.py run --chassis s0|s1      writes amend6/controls/controls_<k>.json   (needs RA6_FIT_<k>.json, lat6/, controls/kcontrol_<k>.json)
  python amend6_controls.py merge                    amend6/controls/CONTROLS.json; exit 0 only if every control and every regression arm reads as required"""
import amend6_env  # noqa: F401  (first: points amend5_stage at the amend6 folders and RA6)
import argparse, hashlib, json, os, shutil, sys
import numpy as np
from amend6_env import ROOT, ST
import ra_l1 as R
import amend5_controls as K5
import fit_ra6 as F
import select_clips6 as S6


# ---------------------------------------------------------------------------------------------- pure checks
def check_sel(new, c141, excluded, eval_ids, window_eids, n_new=859, checkpoints=(141, 423, 859)):
    """C-SEL6 green: n_new distinct clips, disjoint from the 141, the excluded 60, every eval id and every window eid; the checkpoints fit inside the list."""
    new = list(new)
    bad = {"c141": sorted(set(new) & set(c141)), "excluded": sorted(set(new) & set(excluded)), "eval": sorted(set(new) & set(eval_ids)), "window_eids": sorted(set(new) & set(window_eids))}
    return dict(ok=bool(len(new) == n_new and len(set(new)) == n_new and not any(bad.values()) and all(c <= n_new for c in checkpoints)), n=len(new), overlaps={k: len(v) for k, v in bad.items()})


def check_chunks(J, lat_dir, clips6):
    """C-CHUNK6 green: every chunk file's sha256 equals CHUNKS.json, clips are unique and inside clips6, md5-verified for all of them."""
    bad, seen = [], []
    for ch in J["chunks"]:
        p = f"{lat_dir}/{ch['file']}"
        if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != ch["sha256"]:
            bad.append(ch["file"])
        seen += ch["clips"]
        if ch.get("md5_verified") != len(ch["clips"]):
            bad.append(ch["file"] + ":md5")
    return dict(ok=bool(not bad and len(seen) == len(set(seen)) and set(seen) <= set(clips6)), n_clips=len(seen), duplicates=len(seen) - len(set(seen)), bad=bad)


def verdict(green, red):
    return dict(green=green, regression=red, passed=bool(green["ok"] and not red["ok"]))


# ---------------------------------------------------------------------------------------------- C-FIT6: AMEND_5's C-RA-FIT run on the merged fit set
def install_fit6(k):
    rec = json.load(open(f"{ROOT}/RA6_FIT_{k}.json", encoding="utf-8"))
    orig_rebuild, orig_check = R.rebuild_ra_fit, R.check_ra_fit

    def rebuild6(LT, scale_actions, lam, K=R.KS):
        tag = list(LT["Z1"])[0]
        return orig_rebuild(F.merge_latents([LT] + F.chunk_parts(tag, rec["n_new_clips"])), scale_actions, lam, K)

    def check6(rebuilt, readout_np, score_eids, n_pairs_record=rec["n_pairs"], n_clips_record=rec["n_clips"], tol=1e-8):
        return orig_check(rebuilt, readout_np, score_eids, n_pairs_record, n_clips_record, tol)
    R.rebuild_ra_fit, R.check_ra_fit = rebuild6, check6
    return rec


def run_chassis(k):
    install_fit6(k)
    out = K5.run_chassis(k, bases=("A0",), cstrat=(k == "s0"), root=ROOT)
    ctl = out["controls"]
    ctl["C-FIT6"] = ctl.pop("C-RA-FIT")                       # the same code, now on the merged fit set and RA6's record
    kc = json.load(open(f"{ROOT}/controls/kcontrol_{k}.json", encoding="utf-8"))
    ctl["K-CONTROL"] = dict(green=kc["green"], regression=kc["regression_drop_one_clip"], passed=bool(kc["passed"]))
    c6 = json.load(open(f"{ROOT}/clips6.json", encoding="utf-8"))
    mt = [json.loads(l) for l in open(S6.MAN_TRAIN, encoding="utf-8")]
    me = [json.loads(l) for l in open(S6.MAN_EVAL, encoding="utf-8")]
    c141 = json.load(open(S6.CLIPS141, encoding="utf-8"))
    ex = [m["clip"] for m in sorted(mt, key=lambda m: m["sha12"])[540:600]]
    weids = set()
    for p in (ST.CH[k]["win"], ST.windows_of("train", k)):
        weids |= set(np.load(p, allow_pickle=False)["eid"].tolist())
    ev = {m["clip"] for m in me}
    again = S6.select(mt, me, c141, weids)
    g = check_sel(c6["clips"], c141["clips"], ex, ev, weids)
    g["reselection_equal"] = bool(again["clips"] == c6["clips"] and again["clips_sha256"] == c6["clips_sha256"])
    g["ok"] = bool(g["ok"] and g["reselection_equal"])
    plant = list(c6["clips"]); plant[0] = sorted(ev)[0]
    ctl["C-SEL6"] = verdict(g, check_sel(plant, c141["clips"], ex, ev, weids))
    J = json.load(open(f"{ROOT}/lat6/CHUNKS.json", encoding="utf-8"))
    tmp = f"{ROOT}/controls/tmp_chunk_{k}"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    ch0 = J["chunks"][0]
    raw = bytearray(open(f"{ROOT}/lat6/{ch0['file']}", "rb").read()); raw[len(raw) // 2] ^= 1
    open(f"{tmp}/{ch0['file']}", "wb").write(bytes(raw))
    ctl["C-CHUNK6"] = verdict(check_chunks(J, f"{ROOT}/lat6", c6["clips"]), check_chunks(dict(chunks=[ch0]), tmp, c6["clips"]))
    shutil.rmtree(tmp, ignore_errors=True)
    out["red"] = sorted(n for n, v in ctl.items() if not v["passed"]); out["ok"] = not out["red"]
    json.dump(out, open(f"{ROOT}/controls/controls_{k}.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    return out


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--chassis", required=True); sp.add_parser("merge")
    a = ap.parse_args()
    if a.cmd == "run":
        o = run_chassis(a.chassis); print("RED:", o["red"] or "none", "wall", o["wall_s"]); sys.exit(0)
    m = K5.merge(ROOT); print(json.dumps(m, indent=1)); sys.exit(0 if m["ok"] else 1)


if __name__ == "__main__":
    main()
