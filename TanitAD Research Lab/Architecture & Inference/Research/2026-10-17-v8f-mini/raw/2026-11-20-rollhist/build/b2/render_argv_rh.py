#!/usr/bin/env python3
"""render_argv_rh.py -- ROLLHIST stage-B argv files for the pair (RH_sk, B0_sk), k = 0, 1, and the pair check. Light (stdlib + the tree's rollhist_pair_check, which imports tanitad.data.v8f_dagger).

  render [--seeds 0 1]          read RH_READOUT (causal_g1x | causal_g1), write  $RH_ARGV/RH_s{k}.tmpl.json  (placeholders @INIT@ @BANK@ @BANKSHA@ kept),  $RH_ARGV/B0_s{k}.argv.json  (concrete),
                                then run the pair check on each (the template's @BANKSHA@ passes the name check), print `ZZ RENDER-RH PASS|FAIL`.
  fill   SEED                   fill RH_s{k}.tmpl.json with the bank (dir = $R/bank/s{k}) -> $R/runs/RH_s{k}/argv.json; B0 -> $R/runs/B0_s{k}/argv.json (init filled, copied); pair check with --bank-sha256.
  pair   RH.json B0.json [--bank-sha256 X]    the pair check with the readout of RH_READOUT.

Recipe of record = /root/dscale10k/argv/U_s{k}.B.tmpl.json (read only). RH = it + `--dagger-bank-readout <token> --dagger-bank-kappa 0.2`, `--init-ckpt` = UE_s{k}.A, `--out` = $R/runs/RH_s{k}/train.
B0 = the same with `--dagger-p 0` and WITHOUT --dagger-bank, --dagger-bank-sha256, --dagger-seed (the DAG-0 rule), `--out` = $R/runs/B0_s{k}/train.
NOTE ON THE CHECKER: tree/stack/scripts/rollhist_pair_check.py HARD-CODES the RH token (RH_READOUT = readout_token("causal_g", 1.0, True) = causal_g1x). For RH_READOUT=causal_g1 the stock checker REJECTS a correct pair. This script
does not edit it: for causal_g1 it (1) runs the stock checker and records that it rejects, (2) re-runs the SAME functions with DECL patched IN PROCESS to the causal_g1 token, and requires that to PASS. The patched
token is cross-checked against the tree's own readout_token(causal_g, 1.0, path_exec)."""
import argparse, json, os, sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
TREE = os.environ.get("RH_TREE", R + "/tree")
ARGV = os.environ.get("RH_ARGV", R + "/argv")
TMPL_DIR = os.environ.get("RH_TMPL_DIR", "/root/dscale10k/argv")
CKPT = {0: os.environ.get("RH_INIT_S0", "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt"), 1: os.environ.get("RH_INIT_S1", "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt")}
for p in (TREE + "/stack", TREE + "/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import rollhist_pair_check as P  # noqa: E402
from tanitad.data.v8f_dagger import readout_token  # noqa: E402

TOKENS = {"causal_g1x": ("causal_g", 1.0, True), "causal_g1": ("causal_g", 1.0, False)}


def readout():
    ro = os.environ.get("RH_READOUT", "")
    if ro not in TOKENS:
        sys.exit(f"ZZ RENDER-RH FAIL ZZ RH_READOUT must be causal_g1x or causal_g1 (got {ro!r})")
    assert readout_token(*TOKENS[ro]) == ro, (ro, readout_token(*TOKENS[ro]))     # the tree's one spelling agrees with this table
    return ro


def setf(t, f, vals):
    t = list(t)
    if f not in t:
        return t + [f] + [str(v) for v in vals]
    i = t.index(f); j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + [str(v) for v in vals] + t[j:]


def delf(t, f):
    t = list(t)
    if f not in t:
        return t
    i = t.index(f); j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i] + t[j:]


def load(p):
    o = json.load(open(p, encoding="utf-8"))
    return [str(x) for x in (o["argv"] if isinstance(o, dict) else o)]


def pair_problems(rh, b0, token, sha=None):
    """stock checker for causal_g1x; for another token: the same functions with DECL patched in process (stock result also returned for the record)."""
    stock = P.pair_problems(rh, b0, sha)
    if token == P.RH_READOUT:
        return stock, stock, False
    old = dict(P.DECL)
    try:
        P.DECL["--dagger-bank-readout"] = token
        return P.pair_problems(rh, b0, sha), stock, True
    finally:
        P.DECL.clear(); P.DECL.update(old)


def build(seed, token):
    t = load(f"{TMPL_DIR}/U_s{seed}.B.tmpl.json")
    for need in ("@INIT@", "@BANK@", "@BANKSHA@"):
        assert need in t, f"template U_s{seed}.B.tmpl.json lacks {need}"
    assert t[t.index("--seed") + 1] == str(seed) and t[t.index("--dagger-seed") + 1] == str(seed)
    rh = [CKPT[seed] if x == "@INIT@" else x for x in t]                            # @BANK@ / @BANKSHA@ stay until the bank exists
    rh = setf(rh, "--out", [f"{R}/runs/RH_s{seed}/train"])
    rh = setf(rh, "--dagger-bank-readout", [token]); rh = setf(rh, "--dagger-bank-kappa", ["0.2"])
    b0 = [CKPT[seed] if x == "@INIT@" else x for x in t]
    b0 = setf(b0, "--dagger-p", ["0"])
    for f in ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"):
        b0 = delf(b0, f)
    b0 = setf(b0, "--out", [f"{R}/runs/B0_s{seed}/train"])
    return rh, b0


def cmd_render(a):
    token = readout(); os.makedirs(ARGV, exist_ok=True); rep = {"readout": token, "seeds": {}}
    ok = True
    for s in a.seeds:
        rh, b0 = build(s, token)
        json.dump({"argv": rh}, open(f"{ARGV}/RH_s{s}.tmpl.json", "w"), indent=1)
        json.dump({"argv": b0}, open(f"{ARGV}/B0_s{s}.argv.json", "w"), indent=1)
        bad, stock, patched = pair_problems(rh, b0, token)
        rep["seeds"][str(s)] = {"problems": bad, "stock_checker_problems": stock, "checker_patched_in_process": patched}
        ok &= not bad
        for m in bad:
            print("PROBLEM", f"s{s}", m)
    json.dump(rep, open(f"{ARGV}/render_report.json", "w"), indent=1)
    print(json.dumps(rep))
    print("ZZ RENDER-RH", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_fill(a):
    token = readout(); s = a.seed
    bank = f"{R}/bank/s{s}"; bj = f"{bank}/bank.json"
    import hashlib
    sha = hashlib.sha256(open(bj, "rb").read()).hexdigest()
    t = load(f"{ARGV}/RH_s{s}.tmpl.json")
    rh = [bank if x == "@BANK@" else sha if x == "@BANKSHA@" else x for x in t]
    assert not any(x.startswith("@") and x.endswith("@") for x in rh), "an unfilled placeholder remains"
    b0 = load(f"{ARGV}/B0_s{s}.argv.json")
    bad, stock, patched = pair_problems(rh, b0, token, sha)
    for m in bad:
        print("PROBLEM", f"s{s}", m)
    for tag, argv in (("RH", rh), ("B0", b0)):
        d = f"{R}/runs/{tag}_s{s}"; os.makedirs(d, exist_ok=True)
        json.dump({"argv": argv}, open(f"{d}/argv.json", "w"), indent=1)
    print("bank_sha256", sha)
    print("ZZ FILL-RH", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


def cmd_pair(a):
    token = readout()
    bad, stock, patched = pair_problems(load(a.rh), load(a.b0), token, a.bank_sha256)
    for m in bad:
        print("PROBLEM", m)
    print("ZZ ROLLHIST-PAIR", "PASS" if not bad else "FAIL", f"(readout {token}; checker {'patched in process' if patched else 'stock'})")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("render"); r.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); r.set_defaults(f=cmd_render)
    f = sp.add_parser("fill"); f.add_argument("seed", type=int); f.set_defaults(f=cmd_fill)
    p = sp.add_parser("pair"); p.add_argument("rh"); p.add_argument("b0"); p.add_argument("--bank-sha256", default=None); p.set_defaults(f=cmd_pair)
    a = ap.parse_args(); sys.exit(a.f(a))


if __name__ == "__main__":
    main()
