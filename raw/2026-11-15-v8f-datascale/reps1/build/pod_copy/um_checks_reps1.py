"""um_checks.py -- DSCALE10K blockum: the checks the CHAIN makes itself (not prose). usage: um_checks.py pre | post     (pod, CPU; D = $DS_ROOT)
pre  (before stage A; exit 1 = the chain REFUSES): the fresh-reach condition and the data bindings
       1 the stage-A gate token of UM_s1 binds the argv on disk / the tree_um files / the bound extra files (C.token_problems with the arguments the runner passes);
       2 the reach record gate/probe2/UM_s1.A.json is FRESH: measured on tree_um (its `tree`), its own problems_stageA is empty, its sha256 is the one the minted evidence bound (evidence reach_probe_sha256),
         its stage was built from the argv's list (tac_src.list_sha256 = the pinned sha256, lambda 0.0), and it is not older than tree_um's trainer / model files;
       3 the trainer's own pre-launch selfcheck on the real union data (runs/UM_s1/selfcheck_pre/selfcheck.json) carries tac_src_probe (checks 1 to 5) and tac_src, same list sha256;
       4 the list file on disk hashes to the pinned value; the argv's three --tac-src-* flags read the pinned values;
       5 records the sha256 of merged/join_union.npz (--fit-join rewrites it) in runs/UM_s1/join_before.sha256.
post (after stage A; exit 1 = BLOCKED, the chain FAILS): K-TACSRC from runs/UM_s1/A/summary.json (tac_src_w 0.0, tac_src_list_sha256 pinned, weight_share_listed EXACTLY 1.0, rows_other > 0, empty_batches <= 5, rows_listed > 0)
       plus the trainer's own selfcheck.json of the run (tac_src_probe present), steps 61,568, and join_union.npz unchanged (sha256 equals join_before.sha256).
Prints one JSON line; writes runs/UM_s1/um_checks_<pre|post>.json."""
import hashlib
import json
import os
import sys

D = os.environ.get("DS_ROOT", "/root/dscale10k")
TREE = os.environ.get("DS_TREE", D + "/tree_um")
BU = D + "/blockreps1"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study", BU + "/study"):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import dscale10k_check as C  # noqa: E402

if os.path.realpath(C.__file__) != os.path.realpath(BU + "/study/dscale10k_check.py"):
    sys.exit(f"REFUSED: checker imported from {C.__file__}")
COMMIT = "648d4ca3cab588d738f153872473bd3cb6f5e40e"
SHA = C.TAC_SHA
STEPS = 61568


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def L(p):
    return json.load(open(p))


def pre():
    bad = []
    argv_p = f"{D}/argv/UM_s1.A.json"
    toks = L(argv_p)["argv"]
    a = C.parse_argv(toks)
    for f, v in (("--tac-src-w", [C.TAC_W]), ("--tac-src-clips", [C.TAC_LIST]), ("--tac-src-clips-sha256", [SHA]), ("--steps", [str(STEPS)])):
        if a.get(f) != v:
            bad.append(f"argv {f} reads {a.get(f)}, expected {v}")
    if not os.path.isfile(C.TAC_LIST) or sha(C.TAC_LIST) != SHA:
        bad.append("the list file is absent or its sha256 is not the pinned value")
    extras = {"evidence": f"{D}/gate/evidence_UM_s1.json", "dry_record": f"{D}/dry/UM_s1.A.dry.json", "merge_record": f"{D}/merged/merge_record.json", "--union-release": f"{D}/merged/v9_labels_union.npz",
              "tenk_v9": f"{D}/in/labels/v9_labels_datascale10k.npz", "tacsrc_list": C.TAC_LIST, "ue_argv": f"{D}/argv/UE_s1.A.json", "b_dry": f"{D}/dry/B_s0.A.dry.json"}
    bad += ["token: " + x for x in C.token_problems(f"{D}/gate/tokens/UM_s1.A/gate_token.json", "UM_s1", "A", toks, TREE, COMMIT, extras)]
    pr_p = f"{D}/gate/probe2/UM_s1.A.json"
    rec = {}
    if not os.path.isfile(pr_p):
        bad.append("no reach record gate/probe2/UM_s1.A.json")
    else:
        pr = L(pr_p)
        ev = L(extras["evidence"])
        if pr.get("tree") != TREE:
            bad.append(f"fresh-reach: the reach record was measured on {pr.get('tree')}, not {TREE}")
        if pr.get("problems_stageA"):
            bad.append(f"fresh-reach: the reach record's own problems: {pr['problems_stageA'][:2]}")
        if ev.get("reach_probe_sha256") != sha(pr_p):
            bad.append("fresh-reach: the minted evidence bound a different reach record (sha256 differs)")
        ts = pr.get("tac_src") or {}
        if ts.get("list_sha256") != SHA or float(ts.get("tac_src_w", -1)) != 0.0:
            bad.append(f"fresh-reach: the probe's built stage reads {ts}")
        newest = max(os.path.getmtime(f"{TREE}/stack/scripts/v8f_mini_train.py"), os.path.getmtime(f"{TREE}/stack/tanitad/models/v8f_mini.py"))
        if os.path.getmtime(pr_p) < newest:
            bad.append("fresh-reach: the reach record is older than the tree_um trainer / model files")
        rec["reach_sha256"] = sha(pr_p)
    sc_p = f"{D}/runs/UM_s1/selfcheck_pre/selfcheck.json"
    if not os.path.isfile(sc_p):
        bad.append("no pre-launch selfcheck (runs/UM_s1/selfcheck_pre/selfcheck.json): the trainer's own G-LIVE on the real union data has not run")
    else:
        sc = L(sc_p)
        if not sc.get("tac_src_probe") or (sc.get("tac_src") or {}).get("list_sha256") != SHA:
            bad.append("the pre-launch selfcheck carries no tac_src_probe / a different list sha256")
        rec["selfcheck_pre_sha256"] = sha(sc_p)
    os.makedirs(f"{D}/runs/UM_s1", exist_ok=True)
    jp = f"{D}/merged/join_union.npz"
    if os.path.isfile(jp):
        hp = f"{D}/runs/UM_s1/join_before.sha256"
        h = sha(jp)
        if os.path.exists(hp) and open(hp).read().split()[0] != h and not os.path.exists(f"{D}/runs/UM_s1/A/ARM_A_DONE"):
            bad.append("join_union.npz changed since join_before.sha256 was recorded")
        if not os.path.exists(hp):
            open(hp, "w").write(h + "\n")
        rec["join_union_sha256"] = h
    else:
        bad.append("no merged/join_union.npz")
    return bad, rec


def post():
    bad = []
    d = f"{D}/runs/UM_s1/A"
    s = L(f"{d}/summary.json")
    rec = {k: s.get(k) for k in ("tac_src_w", "tac_src_list_sha256", "tac_src_rows_listed", "tac_src_rows_other", "tac_src_weight_share_listed", "tac_src_empty_batches", "steps", "done", "seconds")}
    if s.get("tac_src_w") != 0.0:
        bad.append(f"K-TACSRC tac_src_w {s.get('tac_src_w')} != 0.0")
    if s.get("tac_src_list_sha256") != SHA:
        bad.append("K-TACSRC tac_src_list_sha256 is not the pinned value")
    if s.get("tac_src_weight_share_listed") != 1.0:
        bad.append(f"K-TACSRC tac_src_weight_share_listed {s.get('tac_src_weight_share_listed')} != exactly 1.0")
    if not (s.get("tac_src_rows_other") or 0) > 0 or not (s.get("tac_src_rows_listed") or 0) > 0:
        bad.append(f"K-TACSRC rows listed {s.get('tac_src_rows_listed')} / other {s.get('tac_src_rows_other')} must both be > 0")
    if s.get("tac_src_empty_batches") is None or s["tac_src_empty_batches"] > 5:
        bad.append(f"K-TACSRC tac_src_empty_batches {s.get('tac_src_empty_batches')} > 5")
    if s.get("steps") != STEPS or not s.get("done"):
        bad.append(f"the run reads steps {s.get('steps')} done {s.get('done')}, expected {STEPS} done")
    sc_p = f"{d}/selfcheck.json"
    if not os.path.isfile(sc_p) or not L(sc_p).get("tac_src_probe"):
        bad.append("the run's own selfcheck.json carries no tac_src_probe")
    jp, hp = f"{D}/merged/join_union.npz", f"{D}/runs/UM_s1/join_before.sha256"
    if not os.path.isfile(hp) or sha(jp) != open(hp).read().split()[0]:
        bad.append("join_union.npz differs from the hash recorded before the run (--fit-join rewrote it with different bytes): the arm is BLOCKED")
    return bad, rec


if __name__ == "__main__":
    which = sys.argv[1]
    bad, rec = {"pre": pre, "post": post}[which]()
    out = {"check": which, "problems": bad, "record": rec, "pass": not bad}
    os.makedirs(f"{D}/runs/UM_s1", exist_ok=True)
    json.dump(out, open(f"{D}/runs/UM_s1/um_checks_{which}.json", "w"), indent=1)
    print(json.dumps({"check": which, "pass": not bad, "problems": bad[:4]}))
    sys.exit(0 if not bad else 1)
