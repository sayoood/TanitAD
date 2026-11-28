#!/usr/bin/env python3
"""l3_argv.py -- the four LEVER3 arm argvs (D3K_s0, D3K_s1, D9K_s0, D9K_s1) and control b4 (card PREREG_V8F_LEVER3_V2 (sha256 707c5225..) section 3, control b4).
Port of s3_argv.py's idea: the arm argv is the RH_sK template (rollhist/argv/RH_sK.tmpl.json from the records tar, kit/inputs/) with ONLY these substitutions:
  paths: --out --cache-dir --v9-train --lead-train --speed-train --join-train --p10 --tracks --init-ckpt   (pod paths -> dev-box paths)
  the bank: --dagger-bank (the dev-box bank dir) and --dagger-bank-sha256 (the sha256 of its bank.json)     (@BANK@ / @BANKSHA@ of the template)
  --steps 9000 for D9K (D3K keeps the template's 3000); nothing else. --device stays cuda, --batch stays 64.
Control b4 = `verify`: each arm's argv equals its template except exactly those tokens; the --v9-train-md5 stays 9881b170a889f743c5b0894f9ad13f1d; seed and dagger-seed equal K.
`selftest` plants one regression per rule (changed lr, changed batch, extra flag, dropped flag, wrong steps, wrong md5, wrong seed) and requires each to be REFUSED.
usage: l3_argv.py render --bank-sha0 HEX --bank-sha1 HEX [--root C:/Users/Admin/lever3] | verify [--root ...] | selftest | show --arm D9K_s1 --bank-sha HEX"""
import argparse, copy, json, os, sys

KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT_DEFAULT = "C:/Users/Admin/lever3"
P10 = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/p10/p10_prior.pt"
V9_MD5 = "9881b170a889f743c5b0894f9ad13f1d"
ARMS = ("D3K_s0", "D3K_s1", "D9K_s0", "D9K_s1")
PATH_FLAGS = ("--out", "--cache-dir", "--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--init-ckpt")


def tmpl(seed):
    return json.load(open(f"{KIT}/inputs/RH_s{seed}.tmpl.json"))["argv"]


def paths(arm, root):
    seed = arm.split("_s")[1]
    return {"--out": f"{root}/runs/{arm}/train", "--cache-dir": f"{root}/cache_union", "--v9-train": f"{root}/merged/v9_labels_union.npz",
            "--lead-train": f"{root}/merged/lead_union.npz", "--speed-train": f"{root}/merged/speed_union.jsonl", "--join-train": f"{root}/merged/join_union.npz",
            "--p10": P10, "--tracks": f"{root}/merged/native_tracks_union.pt", "--init-ckpt": f"{root}/init/UE_s{seed}/A/ckpt_last.pt"}


def render(arm, bank_sha, root=ROOT_DEFAULT):
    seed = arm.split("_s")[1]
    t = copy.deepcopy(tmpl(seed))
    sub = dict(paths(arm, root))
    sub["--dagger-bank"] = f"{root}/bank/s{seed}"
    sub["--dagger-bank-sha256"] = bank_sha
    if arm.startswith("D9K"):
        sub["--steps"] = "9000"
    for k, v in sub.items():
        t[t.index(k) + 1] = v
    return t


def verify_one(arm, toks, root=ROOT_DEFAULT, bank_sha=None):
    """problems of one arm's argv against its template (control b4); [] = pass"""
    bad = []
    seed = arm.split("_s")[1]
    t = tmpl(seed)
    allowed = set(PATH_FLAGS) | {"--dagger-bank", "--dagger-bank-sha256"} | ({"--steps"} if arm.startswith("D9K") else set())
    if len(toks) != len(t):
        bad.append(f"token count {len(toks)} != template {len(t)}")
        return bad
    for i, (x, y) in enumerate(zip(toks, t)):
        if x != y and not (i > 0 and t[i - 1] in allowed):
            bad.append(f"token {i} '{x}' differs from the template's '{y}' outside the allowed flags")
    g = lambda f: toks[toks.index(f) + 1] if f in toks else None
    for f, want in paths(arm, root).items():
        if g(f) != want:
            bad.append(f"{f} is {g(f)!r}, expected {want!r}")
    if g("--dagger-bank") != f"{root}/bank/s{seed}":
        bad.append(f"--dagger-bank {g('--dagger-bank')!r}")
    sha = g("--dagger-bank-sha256")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)) or (bank_sha and sha != bank_sha):
        bad.append(f"--dagger-bank-sha256 {sha!r} is not the bank's sha256")
    want_steps = "9000" if arm.startswith("D9K") else "3000"
    if g("--steps") != want_steps:
        bad.append(f"--steps {g('--steps')!r}, expected {want_steps}")
    if g("--warmup") != "100":
        bad.append(f"--warmup {g('--warmup')!r}, expected 100 (the cosine runs over the arm's own steps)")
    if g("--seed") != seed or g("--dagger-seed") != seed:
        bad.append(f"--seed {g('--seed')!r} / --dagger-seed {g('--dagger-seed')!r} are not {seed}")
    if g("--v9-train-md5") != V9_MD5:
        bad.append(f"--v9-train-md5 {g('--v9-train-md5')!r} != {V9_MD5}")
    if g("--device") != "cuda" or g("--batch") != "64":
        bad.append("--device / --batch changed (the card keeps cuda and batch 64)")
    for must in ("--fit-join", "--detach-upper"):
        if must not in toks:
            bad.append(f"{must} missing")
    if g("--dagger-bank-readout") != "causal_g1x" or g("--dagger-bank-kappa") != "0.2":
        bad.append("the bank readout / kappa tokens changed")
    return bad


def argv_path(arm, root):
    return f"{root}/runs/{arm}/argv.json"


def cmd_render(a):
    shas = {0: a.bank_sha0, 1: a.bank_sha1}
    for arm in ARMS:
        seed = int(arm.split("_s")[1])
        t = render(arm, shas[seed], a.root)
        bad = verify_one(arm, t, a.root, shas[seed])
        if bad:
            print(f"REFUSED {arm}: {bad}")
            return 1
        p = argv_path(arm, a.root)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"argv": t}, open(p, "w"), indent=1)
        print("wrote", p)
    return 0


def cmd_verify(a):
    n_bad = 0
    for arm in ARMS:
        p = argv_path(arm, a.root)
        if not os.path.isfile(p):
            print(f"MISSING {p}")
            n_bad += 1
            continue
        bad = verify_one(arm, json.load(open(p))["argv"], a.root)
        print(f"{arm}: {'PASS' if not bad else 'FAIL ' + '; '.join(bad)}")
        n_bad += bool(bad)
    print("ZZ L3 ARGV", "PASS" if not n_bad else "FAIL")
    return 1 if n_bad else 0


def cmd_selftest(a):
    sha = "ab" * 32
    res = {}
    for arm in ARMS:
        good = render(arm, sha)
        res[f"{arm}:clean"] = verify_one(arm, good, bank_sha=sha) == []
        seed = arm.split("_s")[1]
        muts = {}
        def setf(f, v):
            t = list(good); t[t.index(f) + 1] = v; return t
        muts["lr"] = setf("--lr", "2e-4")
        muts["batch"] = setf("--batch", "32")
        muts["steps"] = setf("--steps", "9000" if arm.startswith("D3K") else "3000")
        muts["md5"] = setf("--v9-train-md5", "0" * 32)
        muts["seed"] = setf("--seed", "7")
        muts["dagger_p"] = setf("--dagger-p", "0.3")
        muts["warmup"] = setf("--warmup", "300")
        muts["extra_flag"] = list(good) + ["--w-field", "2.0"]
        muts["dropped_flag"] = [x for x in good if x != "--detach-upper"]
        muts["readout"] = setf("--dagger-bank-readout", "causal_g1")
        for k, m in muts.items():
            res[f"{arm}:mut_{k}_refused"] = bool(verify_one(arm, m, bank_sha=sha))
    ok = all(res.values())
    print(json.dumps(res, indent=1))
    print("ZZ L3 ARGV SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "verify", "selftest", "show"])
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--bank-sha0")
    ap.add_argument("--bank-sha1")
    ap.add_argument("--arm")
    ap.add_argument("--bank-sha")
    a = ap.parse_args()
    if a.cmd == "show":
        print(json.dumps(render(a.arm, a.bank_sha or "@BANKSHA@", a.root)))
        return 0
    return {"render": cmd_render, "verify": cmd_verify, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
