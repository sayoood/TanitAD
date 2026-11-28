"""gate_token.py -- turns a PASSED v8f_mini_train.py --selfcheck-only into a token, and verifies a training argv against it (HIERREP_DEFS R16, R17).
  make   --selfcheck selfcheck.json --argv-file gated_argv.txt --out token.json --kind HIER|FLAT|DAG|FLATDAG --commit C --commit-verified true|false [--bank-sha256 H]
  verify --token token.json --argv-file train_argv.txt            (the normalised argv of the training line must hash-equal the gated one)
  verify-run --token token.json --argv-json OUT/argv.json         (the argv the trainer recorded in its run directory)
  An argv file holds one argv token per line.  Normalisation: '--selfcheck-only' removed, the value after '--out' replaced by '<OUT>'.
Exit 0 ok, 1 refused (reason on stdout).  The token is written by the chain only, never by hand."""
import argparse
import hashlib
import json
import sys
import time

EXPECT = {"HIER": ("full", 19164, 0.0), "FLAT": ("cut", 19164, 0.0), "DAG": ("full", 3000, 0.1), "FLATDAG": ("cut", 3000, 0.1)}
V9_MD5 = {"v9_labels_train.npz": "83e30bd3791f0767707eade5f4671759"}


def normalise(argv):
    out, skip = [], False
    for t in argv:
        if skip:
            out.append("<OUT>")
            skip = False
            continue
        if t == "--selfcheck-only":
            continue
        out.append(t)
        if t == "--out":
            skip = True
    return out


def argv_sha(argv):
    return hashlib.sha256("\n".join(normalise(argv)).encode("utf-8")).hexdigest()


def read_argv(p):
    with open(p, encoding="utf-8") as f:
        return [x.rstrip("\r\n") for x in f if x.rstrip("\r\n") != ""]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_make(a):
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    o_if, steps, dag_p = EXPECT[a.kind]
    bad = []
    i, lv, live = sc.get("info", {}), sc.get("levers", {}), sc.get("live", {})
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"declared_vs_built = {sc.get('declared_vs_built')!r}")
    if live.get("pass") is not True:
        bad.append(f"live.pass = {live.get('pass')!r}")
    if live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"frozen-prior control = {live.get('control_prior_grad_sum')!r}, must be exactly 0.0")
    if live.get("unexpected_dead"):
        bad.append(f"unexpected dead params {live.get('unexpected_dead')}")
    if lv.get("o_interface") != o_if:
        bad.append(f"built o_interface {lv.get('o_interface')!r} != {o_if!r}")
    if o_if == "cut" and (live.get("declared_dead") != ["e_code.weight"] or live.get("dead_params") != ["e_code.weight"]):
        bad.append(f"cut: dead set {live.get('dead_params')} / declared {live.get('declared_dead')} != ['e_code.weight']")
    if o_if == "cut" and (sc.get("o_interface_probe") or {}).get("pass") is not True:
        bad.append("o_interface_probe did not pass")
    if sc.get("steps_total") != steps:
        bad.append(f"steps_total {sc.get('steps_total')} != {steps}")
    if (i.get("n_clips"), i.get("n_excluded")) != (4512, 60):
        bad.append(f"clips / excluded {(i.get('n_clips'), i.get('n_excluded'))} != (4512, 60)")
    if not i.get("train_sha256") or not i.get("excluded_sha256"):
        bad.append("train / excluded sha256 missing from the selfcheck")
    if a.kind in ("DAG", "FLATDAG"):
        if lv.get("dagger_p") != dag_p:
            bad.append(f"built dagger_p {lv.get('dagger_p')!r} != {dag_p}")
        dp = sc.get("dagger_probe") or {}
        if not dp or dp.get("dagger") == "off" or (dp.get("live_swapped") or {}).get("pass") is False:
            bad.append(f"dagger_probe absent, off or failed ({str(dp)[:120]})")
        if not a.bank_sha256 or lv.get("dagger_bank_sha256") != a.bank_sha256:
            bad.append(f"built bank sha256 {str(lv.get('dagger_bank_sha256'))[:12]} != declared {str(a.bank_sha256)[:12]}")
        if (lv.get("init_ckpt") or {}).get("applied") is not True:
            bad.append("init_ckpt not applied")
    elif lv.get("dagger_p") not in (0.0, None):
        bad.append(f"built dagger_p {lv.get('dagger_p')!r} on a Stage 1 arm")
    if bad:
        print("REFUSED: " + "; ".join(bad))
        return 1
    argv = read_argv(a.argv_file)
    tok = {"prereg": "PREREG_V8F_PROOF_ADD11", "kind": a.kind, "gate": "PASS", "commit": a.commit, "commit_verified": a.commit_verified == "true",
           "argv_sha256_normalised": argv_sha(argv), "selfcheck_argv_sha": sc.get("argv_sha"), "selfcheck_sha256": sha_file(a.selfcheck),
           "train_sha256": i["train_sha256"], "excluded_sha256": i["excluded_sha256"], "v9_md5": V9_MD5, "bank_sha256": a.bank_sha256 or None,
           "n_params_trainable": lv.get("n_params_trainable"), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(tok, open(a.out, "w", encoding="utf-8"), indent=1)
    print("token written", a.out, tok["argv_sha256_normalised"][:16])
    return 0


def cmd_verify(a):
    tok = json.load(open(a.token, encoding="utf-8"))
    argv = read_argv(a.argv_file) if a.cmd == "verify" else json.load(open(a.argv_json, encoding="utf-8"))["argv"]
    if tok.get("gate") != "PASS":
        print("REFUSED: token is not a PASS")
        return 1
    h = argv_sha(argv)
    if h != tok["argv_sha256_normalised"]:
        print(f"REFUSED: normalised argv sha256 {h[:16]} != token {tok['argv_sha256_normalised'][:16]}")
        return 1
    print("argv equals the gated argv", h[:16])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--selfcheck", required=True)
    m.add_argument("--argv-file", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--kind", required=True, choices=sorted(EXPECT))
    m.add_argument("--commit", required=True)
    m.add_argument("--commit-verified", default="false")
    m.add_argument("--bank-sha256", default="")
    for n in ("verify", "verify-run"):
        v = sub.add_parser(n)
        v.add_argument("--token", required=True)
        v.add_argument("--argv-file")
        v.add_argument("--argv-json")
    a = ap.parse_args(argv)
    return cmd_make(a) if a.cmd == "make" else cmd_verify(a)


if __name__ == "__main__":
    sys.exit(main())
