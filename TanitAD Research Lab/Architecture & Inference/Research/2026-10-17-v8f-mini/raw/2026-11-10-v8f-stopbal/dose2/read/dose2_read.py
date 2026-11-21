r"""STOPBAL second-dose READ additions (theory/stopbal/AMENDMENT_3_DOSE2.md, frozen): the first dose's read package (theory/stopbal/read/, not edited) with BAL-G2 / BAL-L2 as the lever arms,
the banked BAL-NULL s0 / s1 as the reference of every delta and BAL-NULL_rep as the floor. The rules, outcomes, controls and the estimator are the first dose's: this file only patches the module tables of
`stopbal_read` and `run_dumps_stopbal` (imported by path) and adds a `--levers` selection and the NULL-bank re-assertion. No first-dose file is edited.

  python -I dose2_read.py dumps [TAG ...]            T1-kin dumps (CPU) of the NEW tags only (default: all four); a NULL or first-dose tag is refused (their dumps are the first dose's)
  python -I dose2_read.py refs                        re-assert the banked NULL arms (null_banks.py: ckpt sha256 against the manifest and RUN.done, lever read-back 1.0) and print their checkpoint sha256
  python -I dose2_read.py read --out F [--levers BAL-G2 BAL-L2] [stopbal_read.py options]
                                                      the read: refuses first if a NULL bank is missing or changed; then stopbal_read.main() with the dose-2 tables; adds the NULL checkpoint sha256 to the report
STOPDEC of the four new tags (zero training):  python -I ../../read/stopdec_arm.py --tags BAL-G2_s0 BAL-L2_s0 BAL-G2_s1 BAL-L2_s1 --outdir <dose2/read/out/stopdec> --split-json
(the NULL arms' STOPDEC files stay in the first dose's read/out/stopdec). Dumps go to the same dump folder as the first dose's (one environment): F:/TanitAD-scratch/stopbal_read/t1."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.dirname(HERE)
FIRST_READ = os.path.join(os.path.dirname(D2), "read")
sys.path.insert(0, FIRST_READ)
sys.path.insert(0, D2)
sys.dont_write_bytecode = True
import null_banks as NB  # noqa: E402
import stopbal_read as SR  # noqa: E402
import run_dumps_stopbal as RD  # noqa: E402

NEW_TAGS = ("BAL-G2_s0", "BAL-L2_s0", "BAL-G2_s1", "BAL-L2_s1")
NULL_TAGS = NB.NULL_TAGS
LEVER_ARMS = ("BAL-G2", "BAL-L2")
W_G2, W_L2 = 20.81, 22.08
DOSE2_STOPDEC = os.path.join(HERE, "out", "stopdec")

# ---- the patches (module tables of the first dose's files; the code that reads them is unchanged)
SR.CARD_WEIGHTS["BAL-G2"] = (W_G2, 1.0)           # (goal_pos_weight_stop, lon_class_weight_stop): control 1 expects exactly these read-backs
SR.CARD_WEIGHTS["BAL-L2"] = (1.0, W_L2)
ALL_LEVERS = {"BAL-G2": {"decision": "goal_stop", "score": "goal_score"}, "BAL-L2": {"decision": "lon_stop", "score": "lon_score"}}
SR.LEVERS = dict(ALL_LEVERS)                      # the first dose's BAL-G / BAL-L are not lever arms of this read (their rows stay as measured)
RD.ARMS = tuple(RD.ARMS) + NEW_TAGS
RD.HERE = HERE                                    # DUMPS_<stamp>.json is written beside this file, not into the first dose's read/


def default_map(dumps, sdec, pull):
    """BAL-NULL s0 / s1 / rep from the first dose's banks and STOPDEC files (sdec = the first dose's STOPDEC folder), the four new arms from the dose-2 STOPDEC folder."""
    m = {}
    for tag in NULL_TAGS + NEW_TAGS:
        sd = sdec if tag in NULL_TAGS else DOSE2_STOPDEC
        m[tag] = {"dump": f"{dumps}/dump_{tag}_m10_causal_heads", "stopdec": f"{sd}/stopdec_{tag}.npz", "stopdec_json": f"{sd}/STOPDEC_{tag}.json", "stopdec_tag": tag, "bank": f"{pull}/stopbal_{tag}"}
    return m


SR.default_map = default_map


def select_levers(names):
    """--levers: keep only the named lever arms (default both). An unknown name is refused."""
    names = list(names) if names else list(LEVER_ARMS)
    bad = [n for n in names if n not in ALL_LEVERS]
    if bad:
        raise SystemExit(f"REFUSED: --levers {bad} is not one of {list(LEVER_ARMS)}")
    SR.LEVERS = {k: ALL_LEVERS[k] for k in names}
    return SR.LEVERS


def refs(pulls):
    """-> (problems, {tag: ckpt sha256}) of the banked NULL arms"""
    pr = NB.problems(pulls)
    return pr, ({} if pr else NB.checkpoint_shas(pulls))


def cmd_dumps(tags):
    tags = tags or list(NEW_TAGS)
    bad = [t for t in tags if t not in NEW_TAGS]
    if bad:
        raise SystemExit(f"REFUSED: dumps of {bad}: this package dumps the four new tags only {list(NEW_TAGS)} (the NULL and first-dose dumps are the first dose's)")
    sys.argv = ["run_dumps_stopbal.py"] + tags
    RD.main()


def cmd_read(argv):
    argv = list(argv)
    levers = []
    if "--levers" in argv:
        i = argv.index("--levers")
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            levers.append(argv[j])
            j += 1
        del argv[i:j]
    select_levers(levers)
    pulls = argv[argv.index("--pull") + 1] if "--pull" in argv else RD.PULL
    pr, shas = refs(pulls)
    if pr:
        for x in pr:
            print("REFUSED: banked NULL arm: " + x)
        raise SystemExit(2)
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    sys.argv = ["stopbal_read.py"] + argv
    SR.main()
    if out and os.path.exists(out):
        rep = json.load(open(out, encoding="utf-8"))
        rep["dose2"] = {"amendment": "theory/stopbal/AMENDMENT_3_DOSE2.md", "lever_arms": list(SR.LEVERS), "expected_levers": {k: SR.CARD_WEIGHTS[k] for k in SR.LEVERS},
                        "null_bank_ckpt_sha256": shas, "null_banks_asserted": "ckpt sha256 equals BANK_MANIFEST.json and RUN.done; selfcheck levers read 1.0"}
        json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, default=float)


def main(argv):
    if not argv or argv[0] not in ("dumps", "refs", "read"):
        raise SystemExit(__doc__)
    if argv[0] == "dumps":
        return cmd_dumps(argv[1:])
    if argv[0] == "refs":
        pulls = argv[argv.index("--pulls") + 1] if "--pulls" in argv else RD.PULL
        pr, shas = refs(pulls)
        for x in pr:
            print("REFUSED: " + x)
        if not pr:
            print("NULL banks whole:", json.dumps(shas))
        return 1 if pr else 0
    return cmd_read(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
