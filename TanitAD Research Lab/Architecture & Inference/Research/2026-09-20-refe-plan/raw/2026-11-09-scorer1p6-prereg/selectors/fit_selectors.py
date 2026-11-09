#!/usr/bin/env python3
"""fit_selectors.py -- fit every variant of families R, C, S, P of PREREG_P6_SELECTORS on FIT-A, judge on FIT-B, nominate. CPU only (pod or dev box).

    python fit_selectors.py --pack fit_pack.npz --out OUTDIR [--threads 4] [--only R1,C_platt] [--navtest-tokens FILE ...] [--min-free-gb 6]

Order of work: (1) refuse if any pack token is a navtest token; (2) the log split (seed 20261109) is written WITH its sha256 before the first fit, and the run refuses if
a split already exists in OUTDIR and differs; (3) each variant is fitted on FIT-A rows ONLY (resumable: a variant with a done marker is skipped) and scored on FIT-B;
(4) nomination: per family the best <= 2 variants by mean true PDMS at the pick on FIT-B, only with CI-lo > 0 of the paired log-cluster bootstrap against P6's rule
(10,000 draws, seed 20260927); (5) nominees.json lists each nominee with its fitted-parameter file and sha256. navtest is never read here.
Exit codes: 0 done, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

if "--threads" in sys.argv:                       # before numpy/torch/sklearn load
    os.environ["SEL_THREADS"] = sys.argv[sys.argv.index("--threads") + 1]
os.environ.setdefault("SEL_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = os.environ["SEL_THREADS"]
os.environ["MKL_NUM_THREADS"] = os.environ["SEL_THREADS"]
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selcore as C  # noqa: E402

PREREG_SHA = "d4778ca40be22cbe706efe68e6d3878a9b6f070179dd2b604373b5eef5fc690b"
DEFAULT_NAVTEST = os.path.join(HERE, "navtest_tokens_e3.txt")


class Refused(Exception):
    pass


def load_tokens(path):
    """one token per line (txt), or json {"tokens": [...]}, or npz with a 'token' array"""
    if path.endswith(".npz"):
        return {str(t) for t in np.load(path, allow_pickle=False)["token"]}
    if path.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        return {str(t) for t in (d["tokens"] if isinstance(d, dict) else d)}
    return {ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()}


def check_not_navtest(tokens, files):
    if not files:
        raise Refused("no navtest token list given: cannot prove the pack is free of navtest tokens (fail closed)")
    nav = set()
    for f in files:
        if not os.path.exists(f):
            raise Refused(f"navtest token list {f} does not exist (fail closed)")
        nav |= load_tokens(f)
    if len(nav) < 1000:
        raise Refused(f"navtest token list has only {len(nav)} tokens: expected the 12,146 of E3")
    bad = sorted(set(map(str, tokens)) & nav)
    if bad:
        raise Refused(f"{len(bad)} pack tokens are navtest tokens (e.g. {bad[:3]}): navtest is never used for fitting")
    return len(nav)


def check_prereg(path=None):
    cand = path or os.path.join(HERE, "..", "PREREG_P6_SELECTORS.md")
    if not os.path.exists(cand):
        return {"checked": False, "why": f"{cand} absent on this machine (pod): sha recorded, not verified here"}
    got = C.sha256_file(cand)
    if got != PREREG_SHA:
        raise Refused(f"PREREG_P6_SELECTORS.md sha256 {got} != frozen {PREREG_SHA}")
    sidecar = os.path.join(os.path.dirname(cand), "PREREG_P6_SELECTORS.sha256")
    if os.path.exists(sidecar) and open(sidecar, encoding="utf-8").read().split()[0] != got:
        raise Refused("the .sha256 next to the card does not match the card")
    return {"checked": True, "sha256": got}


def write_split(out, pack):
    """written with its sha256 before the first fit; refuses if a split exists and differs"""
    sp = C.make_split(pack.log)
    sp["pack_sha256"] = pack.sha256
    sp["tokens_sha256"] = C.sha256_bytes("\n".join(sorted(pack.token.tolist())).encode("utf-8"))
    blob = C.canon(sp)
    sha = C.sha256_bytes(blob)
    jp, sp_sha = os.path.join(out, "split.json"), os.path.join(out, "split.sha256")
    if os.path.exists(jp) or os.path.exists(sp_sha):
        old = open(jp, "rb").read() if os.path.exists(jp) else b""
        old_sha = open(sp_sha, encoding="utf-8").read().split()[0] if os.path.exists(sp_sha) else ""
        if C.sha256_bytes(old) != old_sha or old_sha != sha or old != blob:
            raise Refused(f"a split exists in {out} and differs from the one this pack gives (existing sha {old_sha[:12]}, new {sha[:12]}): refusing")
        return sp, sha, False
    C.atomic_write(jp, blob)
    C.atomic_write(sp_sha, f"{sha} *split.json\n".encode("utf-8"))
    return sp, sha, True


def fit_one_set(vids, A, cfg):
    return C.fit_variants(A, vids, cfg)


def evaluate_picks(PB, pk, ref_pick, li, nl, draws):
    """FIT-B mean true PDMS x100 at the pick, and the paired log-cluster bootstrap of (pick - P6 rule's pick): -> (value, delta point, ci_lo, ci_hi)"""
    ix = np.arange(len(PB))
    pt, r = C.boot_stat(100.0 * (PB[ix, pk] - PB[ix, ref_pick]), li, nl, draws)
    lo, hi = C.interval(r, 1)
    return C.mean_pdms_x100(PB, pk), pt, lo, hi


def run(a, log=print):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    cfg = json.loads(a.cfg) if a.cfg else {}
    prereg = check_prereg(a.prereg)
    waited = C.wait_for_ram(a.min_free_gb, log=log) if not a.no_ram_wait else 0.0
    pack = C.Pack(a.pack)
    if a.smoke_standin:
        smoke_ok = os.path.abspath(out).replace("\\", "/").startswith("F:/tanitad_scratch/") or os.environ.get("SEL_SMOKE_ANYWHERE") == "1"
        if not smoke_ok:
            raise Refused("--smoke-standin writes only under F:/tanitad_scratch/")
        navn = None
    else:
        navn = check_not_navtest(pack.token, a.navtest_tokens or [DEFAULT_NAVTEST])
    sp, split_sha, fresh = write_split(out, pack)
    log(f"split {'written' if fresh else 'verified'}: sha256 {split_sha}  A={sp['n_sets_A']} sets/{len(sp['logs_A'])} logs  B={sp['n_sets_B']} sets/{len(sp['logs_B'])} logs")
    iA, iB = C.split_rows(pack.log, sp)
    ctx = C.Ctx(pack.L, pack.X8, pack.v0)
    # FIT-A rows only reach a fit: FitData is built from the A slice and nothing else
    A = C.FitData(ctx.take(iA), pack.P[iA], pack.S[iA], pack.log[iA])
    ctxB, PB, logB = ctx.take(iB), pack.P[iB], pack.log[iB]
    vids = a.only.split(",") if a.only else list(C.VARIANTS)
    bad = [v for v in vids if v not in C.VARIANTS]
    if bad:
        raise Refused(f"unknown variants {bad}")
    if a.smoke_standin:
        C.atomic_write(os.path.join(out, "SMOKE_NOT_A_RESULT"), b"stand-in pack of navtest tokens; no number here is a result\n")
    ref_pick = C.p6_pick(ctxB.L)
    ul, li, draws = C.log_boot(logB)
    ref_val = C.mean_pdms_x100(PB, ref_pick)
    res_dir = os.path.join(out, "variants")
    done = {}
    t_all = time.time()
    for v in vids:
        d = os.path.join(res_dir, v)
        mp = os.path.join(d, "meta.json")
        if os.path.exists(mp) and not a.force:
            m = json.load(open(mp, encoding="utf-8"))
            if m.get("done") and m.get("split_sha256") == split_sha and m.get("fit_cfg") == cfg:
                log(f"{v}: done, skipped"); done[v] = m
                continue
            if m.get("done"):
                raise Refused(f"{v}: a finished fit exists under a different split or config: use --force to redo")
        t0 = time.time()
        got = fit_one_set([v], A, cfg)
        params = got[v]
        if "_R_table" in got:
            C.atomic_write(os.path.join(out, "R_candidates_fit_a.json"), json.dumps(got["_R_table"], indent=1).encode("utf-8"))
        pk = C.pick_of(C.apply_params(params, ctxB))
        val, pt, lo, hi = evaluate_picks(PB, pk, ref_pick, li, len(ul), draws)
        os.makedirs(d, exist_ok=True)
        pbytes = pickle.dumps(params, protocol=4)
        C.atomic_write(os.path.join(d, "params.pkl"), pbytes)
        np.save(os.path.join(d, "pick_fit_b.npy"), pk.astype(np.int16))
        meta = dict(variant=v, family=C.family_of(v), done=True, split_sha256=split_sha, pack_sha256=pack.sha256, fit_cfg=cfg, n_fit_a=int(A.n), n_fit_b=int(len(iB)),
                    fit_b_mean_pdms_x100=val, p6_rule_fit_b_mean_pdms_x100=ref_val, delta_vs_p6_x100=pt, ci_lo=lo, ci_hi=hi, n_logs_b=int(len(ul)),
                    params_sha256=C.sha256_bytes(pbytes), params_file=os.path.join("variants", v, "params.pkl"), v0_source=ctx.v0_source,
                    n_pick_differs_from_p6=int((pk != ref_pick).sum()), fit_secs=round(time.time() - t0, 1), smoke=bool(a.smoke_standin))
        C.atomic_write(mp, json.dumps(meta, indent=1).encode("utf-8"))
        done[v] = meta
        log(f"{v}: FIT-B {val:.3f} vs P6 rule {ref_val:.3f}  delta {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({meta['fit_secs']} s)")
    nom = nominate(out, split_sha, log)
    nom["prereg"] = prereg
    nom["navtest_tokens_checked"] = navn
    nom["waited_for_ram_s"] = waited
    nom["smoke"] = bool(a.smoke_standin)
    nom["pack"] = dict(path=pack.path, sha256=pack.sha256, n_sets_kept=pack.n, n_no_label=pack.n_no_label, n_nonfinite_excluded=pack.n_nonfinite, v0_source=ctx.v0_source)
    nom["total_secs"] = round(time.time() - t_all, 1)
    C.atomic_write(os.path.join(out, "nominees.json"), json.dumps(nom, indent=1).encode("utf-8"))
    C.atomic_write(os.path.join(out, "nominees.sha256"), f"{C.sha256_file(os.path.join(out, 'nominees.json'))} *nominees.json\n".encode("utf-8"))
    C.atomic_write(os.path.join(out, "FIT_REPORT.md"), report_md(nom).encode("utf-8"))
    return nom


def nominate(out, split_sha, log=print):
    """per family: eligible = CI-lo > 0 against P6's rule on FIT-B; nominate the best (at most) two by FIT-B mean PDMS at the pick. Reads only the meta of variants fitted so far."""
    metas = {}
    for v in C.VARIANTS:
        mp = os.path.join(out, "variants", v, "meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp, encoding="utf-8"))
            if m.get("done") and m.get("split_sha256") == split_sha:
                metas[v] = m
    fams, nominees = {}, []
    for f in C.FAMILY_OF:
        vs = [m for v, m in metas.items() if C.family_of(v) == f]
        ok = sorted([m for m in vs if m["ci_lo"] > 0], key=lambda m: -m["fit_b_mean_pdms_x100"])[:2]
        fams[f] = dict(name=C.FAMILY_OF[f], n_variants_fitted=len(vs), nominated=[m["variant"] for m in ok],
                       note=("" if ok else "no variant has a paired log-cluster CI-lo > 0 against P6's rule on FIT-B: the family nominates none (reported, not a failure of the tool)"))
        for rk, m in enumerate(ok, 1):
            nominees.append(dict(family=f, id=m["variant"], family_rank=rk, fit_b_mean_pdms_x100=m["fit_b_mean_pdms_x100"], delta_vs_p6_x100=m["delta_vs_p6_x100"],
                                 ci_lo=m["ci_lo"], ci_hi=m["ci_hi"], params_file=m["params_file"], params_sha256=m["params_sha256"], v0_source=m["v0_source"]))
    for f, d in fams.items():
        log(f"family {f}: nominated {d['nominated'] or 'none'}")
    return dict(card="PREREG_P6_SELECTORS", split_sha256=split_sha, estimator="paired log-cluster bootstrap on FIT-B, 10,000 draws, seed 20260927, 95 % percentile; 'another draw of logs' only",
                variants={v: {k: m[k] for k in ("fit_b_mean_pdms_x100", "p6_rule_fit_b_mean_pdms_x100", "delta_vs_p6_x100", "ci_lo", "ci_hi", "n_pick_differs_from_p6", "fit_secs")} for v, m in metas.items()},
                families=fams, nominees=nominees,
                evidence_class="MEASURED (ours; fit_pack sha256 and split sha256 above). P6 logits are IN-SAMPLE on these logs (PREREG section 1): FIT numbers are not performance claims.")


def report_md(nom):
    L = ["# FIT report -- PREREG_P6_SELECTORS" + ("  (SMOKE STAND-IN: NOT A RESULT)" if nom.get("smoke") else ""), "",
         f"split sha256 `{nom['split_sha256']}`; estimator: {nom['estimator']}", "",
         "| variant | FIT-B PDMS x100 | P6 rule | delta [95 % CI] | picks changed |", "|---|---|---|---|---|"]
    for v, m in nom["variants"].items():
        L.append(f"| {v} | {m['fit_b_mean_pdms_x100']:.3f} | {m['p6_rule_fit_b_mean_pdms_x100']:.3f} | {m['delta_vs_p6_x100']:+.3f} [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}] | {m['n_pick_differs_from_p6']} |")
    L += ["", "## Nominated (FIT-B only, CI-lo > 0)"]
    for f, d in nom["families"].items():
        L.append(f"- {d['name']}: {', '.join(d['nominated']) if d['nominated'] else 'none. ' + d['note']}")
    L += ["", nom["evidence_class"], "Tier: FIT-B is a selection set, not a driving tier."]
    return "\n".join(L) + "\n"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma list of variant ids (default: all)")
    ap.add_argument("--navtest-tokens", action="append", default=None, help="file(s) with the navtest token list; default selectors/navtest_tokens_e3.txt")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--no-ram-wait", action="store_true")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cfg", default=None, help="json override of hyper-parameters (tests only; recorded in every meta)")
    ap.add_argument("--smoke-standin", action="store_true", help="the pack is the navtest stand-in: output stamped SMOKE_NOT_A_RESULT, only under F:/tanitad_scratch/")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        nom = run(a)
    except Refused as e:
        print(f"SELFIT_REFUSED {e}")
        return 2
    print(f"SELFIT_OK nominees={[n['id'] for n in nom['nominees']]} out={os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
