"""check_wrapper_j.py -- pod check of the wrapper v8f_t1_eval_reader_rule_rzgfhqetj.py (card ADD94, control j2), after check_shufclose.py.
Part A (no GPU): static facts of the j text. Part B (pod): rollouts of form RLc (label-fed with closing speed; model seed 0; readout K: --readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2;
--rule-every 5; pre-roll of 30 steps (6 s); the first --n-clips clips of the pre-roll grid) with
    jr_old    the generation it was written from (rzgfhqet)
    jr_new    the new generation, no --rule-join
    jr_g0     the new generation with --rule-join G0.npz (the label gap itself)         [only with --g0-join]
 and a refusal (--rule-join with a file that is not there). Checks: J2A jr_new equals jr_old (every numeric array of every npz file: dump files, tick file, rule log and its side files,
 mid-look log and side file, first-plan file; largest difference 0; the run record meta of jr_new has rule_join empty); J2B jr_g0 equals jr_old the same way; J2C the refusal has a non-zero
 return code and the text REFUSED and --rule-join. Writes <workdir>/CHECK_WRAPPER_J_DONE.txt (last line ZZ CHECK-WRAPPER-J PASS or FAIL ...) and CHECK_WRAPPER_J.hashes. No removal, no move, no kill.
Usage: python check_wrapper_j.py --old OLD.py --new NEW.py --tree TREE --ckpt CKPT --tracks T.pt --eval-cache DIR --eval-join J.npz --grid GRID_P30.json --workdir NEWDIR
          [--g0-join G0.npz] [--distrule-dir DIR] [--closest-dir DIR] [--python PY] [--device cuda] [--n-clips 3] [--dry]  |  --pure-only --new NEW.py  |  --selftest"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

BAD = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
SKIP_KEYS = ("tag", "rule_join", "rule_join_sha256")


def maxdiff(a, b, skip=SKIP_KEYS):
    """-> (largest absolute difference over the common numeric arrays, problems). A key only one side has is a problem unless it is in skip."""
    a = {k: v for k, v in a.items() if k not in skip}
    b = {k: v for k, v in b.items() if k not in skip}
    worst, prob = 0.0, []
    if sorted(a) != sorted(b):
        prob.append("keys differ %s" % (sorted(set(a) ^ set(b))[:4],))
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            prob.append("shape of %s" % k)
        elif x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            if not np.array_equal(np.isnan(xf), np.isnan(yf)):
                prob.append("nan pattern of %s" % k)
            elif xf.size and np.isfinite(xf).any():
                worst = max(worst, float(np.nanmax(np.abs(xf - yf))))
        elif x.dtype.kind not in "US" and not np.array_equal(x, y):
            prob.append("content of %s" % k)
    return worst, prob


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def compare_runs(dir_a, name_a, dir_b, name_b):
    """every npz file under dir_a against the file with the run name replaced under dir_b. -> {"files", "largest_diff", "problems"}"""
    files, worst, prob = 0, 0.0, []
    for pa in sorted(glob.glob(os.path.join(dir_a, "**", "*.npz"), recursive=True)):
        rel = os.path.relpath(pa, dir_a)
        pb = os.path.join(dir_b, rel.replace(name_a, name_b))
        if not os.path.isfile(pb):
            prob.append("missing %s" % rel)
            continue
        d, p = maxdiff(load_npz(pa), load_npz(pb))
        files += 1
        worst = max(worst, d)
        prob += ["%s: %s" % (rel, x) for x in p]
    return {"files": files, "largest_diff": worst, "problems": prob}


def static_facts(src):
    f = [("A1 flag --rule-join exists", 'ap.add_argument("--rule-join", default="")' in src),
         ("A2 the rule label source is built from the rule join", 'X.STATE["label"] = build_rule_label(a, mine.rule_join, T.D.load_join)' in src),
         ("A3 the namespace of the pinned tool keeps --eval-join (no assignment to a.eval_join)", "a.eval_join =" not in src and "a.eval_join=" not in src),
         ("A4 install gets no join (the label of the install is not built)", 'W.install(T, "mix", a, None, log, "cap-only")' in src),
         ("A5 the run record has rule_join and rule_join_sha256", '"rule_join": mine.rule_join' in src and '"rule_join_sha256"' in src),
         ("A6 none of the banned words in the text", not [w for w in BAD if w in src])]
    return f


def selftest():
    ok = True
    a = {"x": np.arange(5.0), "rule_join": np.asarray("a")}
    b = {"x": np.arange(5.0), "rule_join": np.asarray("")}
    ok &= maxdiff(a, b) == (0.0, [])
    c = {"x": np.arange(5.0) + 1e-3}
    d, p = maxdiff(a, c)
    ok &= d > 0 or bool(p)
    e = {"x": np.arange(5.0), "y": np.zeros(2)}
    ok &= bool(maxdiff(a, e)[1])
    print("ZZ CHECK-WRAPPER-J-SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    for k in ("old", "new", "tree", "ckpt", "tracks", "eval-cache", "eval-join", "grid", "workdir", "g0-join", "distrule-dir", "closest-dir"):
        ap.add_argument("--" + k)
    ap.add_argument("--pure-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--n-clips", type=int, default=3)
    A = ap.parse_args()
    if A.selftest:
        sys.exit(selftest())
    FAILED = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
        if not cond:
            FAILED.append(name.split(" ")[0])
    src = open(A.new, encoding="utf-8").read()
    sha16 = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    print("wrapper", A.new, sha16, flush=True)
    for nm, c in static_facts(src):
        check(nm, c)
    if A.pure_only:
        print("ZZ CHECK-WRAPPER-J-PURE " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
        sys.exit(0 if not FAILED else 1)
    for k in ("old", "tree", "ckpt", "tracks", "eval_cache", "eval_join", "grid", "workdir"):
        if not getattr(A, k):
            sys.exit("ZZ CHECK-WRAPPER-J STOP missing --" + k.replace("_", "-"))
    W = A.workdir
    if os.path.exists(W):
        sys.exit("ZZ CHECK-WRAPPER-J STOP OUTPUT-EXISTS " + W)
    here = os.path.dirname(os.path.abspath(A.new))
    dist, clos = A.distrule_dir or here, A.closest_dir or here
    tmo = shutil.which("timeout")
    grid_sub = os.path.join(W, "grid_subset.json")
    if not A.dry:
        os.makedirs(W)
        g = json.load(open(A.grid, encoding="utf-8"))
        order = list(g["order"])[:A.n_clips]
        g2 = dict(g, order=order, windows={c: g["windows"][c] for c in order}, n_windows=int(sum(len(g["windows"][c]) for c in order)))
        json.dump(g2, open(grid_sub, "w", encoding="utf-8", newline="\n"))

    def cmd_of(wrapper, name, extra=()):
        out = os.path.join(W, name)
        c = [A.python, wrapper, "--tree", A.tree, "--form", "RLc"] + list(extra) + [
            "--rule-log", os.path.join(out, "ticks", "rule_log_%s.npz" % name), "--rule-every", "5", "--readout", "causal_g", "--readout-log", os.path.join(out, "ticks", "readout_log_%s.json" % name),
            "--acc-gain", "0", "--path-exec", "--kappa-max", "0.2", "--preroll-steps", "30",
            "run", "--ckpt", A.ckpt, "--tag", name + "_m10_causal_heads", "--tracks", A.tracks, "--eval-cache", A.eval_cache, "--eval-grid", grid_sub, "--eval-join", A.eval_join, "--out-dir", out,
            "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze",
            "--device", A.device]
        return out, ([tmo, "1800"] + c if tmo else c)
    env = dict(os.environ, DISTRULE_DIR=dist, CLOSEST_DIR=clos, PYTHONPATH=os.path.join(A.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    runs = [("jr_old", A.old, ()), ("jr_new", A.new, ())] + ([("jr_g0", A.new, ("--rule-join", A.g0_join))] if A.g0_join else [])
    for name, wrapper, extra in runs:
        out, c = cmd_of(wrapper, name, extra)
        if A.dry:
            print("CMD: " + " ".join(c))
            continue
        os.makedirs(os.path.join(out, "ticks"))
        with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
            rc = subprocess.run(c, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print("RUN %s rc=%d" % (name, rc), flush=True)
        if rc != 0:
            print("ZZ CHECK-WRAPPER-J FAIL run-%s-rc-%d" % (name, rc))
            sys.exit(1)
    out, c = cmd_of(A.new, "jr_ref", ("--rule-join", os.path.join(W, "no_such_join.npz")))
    if A.dry:
        print("CMD: " + " ".join(c))
        print("ZZ CHECK-WRAPPER-J dry run: nothing executed")
        sys.exit(0)
    cp = subprocess.run(c, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    txt = cp.stdout.decode("utf-8", "replace")
    open(os.path.join(W, "refusal_missing_join.log"), "w", encoding="utf-8").write(txt)
    d = compare_runs(os.path.join(W, "jr_new"), "jr_new", os.path.join(W, "jr_old"), "jr_old")
    check("J2A jr_new equals jr_old: largest difference %.3g over %d files" % (d["largest_diff"], d["files"]), d["files"] > 0 and d["largest_diff"] == 0.0 and not d["problems"], "; ".join(d["problems"][:3]))
    rl = glob.glob(os.path.join(W, "jr_new", "ticks", "rule_log_jr_new.npz"))
    mj = load_npz(rl[0]).get("rule_join") if rl else None
    check("J2A2 the rule log of jr_new records an empty rule_join", mj is not None and str(np.asarray(mj)) == "")
    if A.g0_join:
        d = compare_runs(os.path.join(W, "jr_g0"), "jr_g0", os.path.join(W, "jr_old"), "jr_old")
        check("J2B jr_g0 equals jr_old: largest difference %.3g over %d files" % (d["largest_diff"], d["files"]), d["files"] > 0 and d["largest_diff"] == 0.0 and not d["problems"], "; ".join(d["problems"][:3]))
    check("J2C a missing --rule-join file is refused", cp.returncode != 0 and "REFUSED" in txt and "--rule-join" in txt)
    open(os.path.join(W, "CHECK_WRAPPER_J.hashes"), "w", encoding="utf-8").write("%s %s\n" % (sha16, os.path.basename(A.new)))
    last = "ZZ CHECK-WRAPPER-J PASS" if not FAILED else "ZZ CHECK-WRAPPER-J FAIL " + "; ".join(FAILED)
    open(os.path.join(W, "CHECK_WRAPPER_J_DONE.txt"), "w", encoding="utf-8").write(last + "\n")
    print(last)
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
