"""Unit tests of the STOPBAL second-dose package (pure Python; numpy only via the stand-in; no torch, no GPU, no network, no box call).
Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_dose2_unit.py <scratch dir under F:/TanitAD-scratch/dose2_build/>     -> prints  UNIT2 ok=<n> bad=<m>  (exit 1 on any bad)
Covers: the three stage / job files against the first dose's (only the declared lines differ), the shim's tables, every gate assertion with a planted-bad selfcheck per new tag,
the red arms (wrong weight, wrong argv, changed argv file), the post check on planted checkpoints, and the frozen argv file against the first dose's twin arms."""
import copy
import difflib
import hashlib
import json
import os
import pickle
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
BOX2 = os.path.join(os.path.dirname(PKG), "box2")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(BOX2, "tests"))
import stopbal_check as CK  # noqa: E402   (the dose-2 layer)
import plant_trainer as PT  # noqa: E402   (the first dose's stand-in, imported by path, not edited)

N_OK, N_BAD = 0, 0
TAGS = ["BAL-G2_s0", "BAL-L2_s0", "BAL-G2_s1", "BAL-L2_s1"]
PILOT = "PILOT_BAL-G2_s0"
AMENDMENT = os.path.join(os.path.dirname(PKG), "AMENDMENT_3_DOSE2.md")


def check(name, cond):
    global N_OK, N_BAD
    if cond:
        N_OK += 1
    else:
        N_BAD += 1
        print(f"UNIT2-FAIL {name}")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def put(p, data=b""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lines(p):
    return open(p, encoding="utf-8", newline="").read().split("\n")


def only_changed(orig, new, allowed_orig_lines):
    """True iff every differing block of `new` against `orig` replaces / deletes only original lines (1-based) in the allowed set"""
    sm = difflib.SequenceMatcher(None, orig, new, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            if i1 + 1 not in allowed_orig_lines and i1 not in allowed_orig_lines:
                return False, (tag, i1)
            continue
        if not all(i + 1 in allowed_orig_lines for i in range(i1, i2)):
            return False, (tag, i1 + 1, i2)
    return True, None


def sc_of(tag, tmp):
    toks = CK.tokens(tag)
    gt = CK.gate_argv(tag, toks)
    os.environ["STOPBAL_FAKE_INIT_SHA_FILE"] = os.path.join(tmp, "init_sha.json")
    os.environ["STOPBAL_FAKE_BANK_SHA_FILE"] = os.path.join(tmp, "bank_sha.json")
    return toks, PT.good_selfcheck(gt)


def main(tmp):
    os.makedirs(tmp, exist_ok=True)
    json.dump({"0": CK.INIT_SHA[0], "1": CK.INIT_SHA[1]}, open(os.path.join(tmp, "init_sha.json"), "w"))
    json.dump({"0": CK.BANK_SHA[0], "1": CK.BANK_SHA[1]}, open(os.path.join(tmp, "bank_sha.json"), "w"))
    # ---- the frozen spec and the files that are copies
    am_sha = open(os.path.join(os.path.dirname(PKG), "AMENDMENT_3_DOSE2.sha256"), encoding="utf-8").read().split()[0]
    check("the frozen amendment hashes to its .sha256 file", sha(AMENDMENT) == am_sha)
    for j in ("stopbal_stage_job.cmd", "stopbal_ship_job.cmd"):
        check(f"jobs/{j} is a byte copy of the first dose's", sha(os.path.join(PKG, "jobs", j)) == sha(os.path.join(BOX2, "jobs", j)))
    run_o, run_n = lines(os.path.join(BOX2, "stages", "stopbal_run.sh")), lines(os.path.join(PKG, "stages", "dose2_run.sh"))
    ok, where = only_changed(run_o, run_n, {2, 3, 4, 5, 8, 9, 12, 18, 41, 43})
    check(f"dose2_run.sh differs from stopbal_run.sh only on the declared lines ({where})", ok)
    ship_o, ship_n = lines(os.path.join(BOX2, "stages", "stopbal_ship.sh")), lines(os.path.join(PKG, "stages", "dose2_ship.sh"))
    ok, where = only_changed(ship_o, ship_n, {2, 3, 4, 5, 7, 11, 19})
    check(f"dose2_ship.sh differs from stopbal_ship.sh only on the declared lines ({where})", ok)
    check("dose2_run.sh and dose2_ship.sh keep every first-dose line that is not declared (line counts within 3)", abs(len(run_n) - len(run_o)) <= 3 and abs(len(ship_n) - len(ship_o)) <= 4)
    for f in ("dose2_chain.sh", "stages/dose2_run.sh", "stages/dose2_ship.sh", "stages/stopbal_common.sh"):
        check(f"{f} is LF only", b"\r" not in open(os.path.join(PKG, f), "rb").read())
    # ---- the shim's tables
    check("run order: G2_s0, L2_s0, G2_s1, L2_s1", CK.RUN_ORDER == TAGS)
    check("four runs, no NULL arm and no parity step in the table", len(CK.TABLE) == 4 and CK.PARITY_TAG == "" and not any("NULL" in t for t in CK.TABLE))
    check("the pilot is PILOT_BAL-G2_s0 on BAL-G2_s0, the pilot also gates G2_s0 and L2_s0", CK.PILOT_TAG == PILOT and CK.PILOT_BASE == "BAL-G2_s0" and CK.GATE_AT_PILOT == ["BAL-G2_s0", "BAL-L2_s0"])
    check("the card's weights: goal 20.81, lon 22.08", CK.W_GOAL == 20.81 and CK.W_LON == 22.08 and CK.WEIGHTS == {"null": (1.0, 1.0), "g": (20.81, 1.0), "l": (1.0, 22.08)})
    check("steps 3,000 per run, pilot 200, 12 h limit are the first dose's", CK.RUN_STEPS == 3000 and CK.PILOT_STEPS == 200 and CK.LIMIT_H == 12.0)
    check("the argv file hashes to the pin and is the dose-2 file", CK.sha_file(CK.ARGVS_FILE) == CK.ARGVS_SHA256 == "f00267cbdaffe16ddf8743553f28f4198cba8a5e67f7842dc5ea12a0bd0127f4" and CK.ARGVS_FILE.replace("\\", "/").endswith("frozen/argvs_dose2.json"))
    meta = json.load(open(os.path.join(PKG, "argvs_dose2.meta.json"), encoding="utf-8"))
    check("the argv meta names the code of record 715ef8fc and the pinned hash", CK.CODE_OF_RECORD == "715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf" and CK.ARGVS_SHA256 in json.dumps(meta) and CK.CODE_OF_RECORD in json.dumps(meta))
    # the shim must not have changed the first dose's own tags (a separate load of the base file would read its own tables)
    check("the first dose's checker file on disk still names its own seven tags (not edited)", "BAL-NULL_rep" in open(os.path.join(BOX2, "stopbal_check.py"), encoding="utf-8").read())
    dp = CK.data_paths()
    check("data_paths() works under the dose-2 table (the base read it from a BAL-NULL_s0 token that is not here) and names the six files", set(dp) == {"v9_labels_train", "lead_train", "speed_train", "join_train", "p10_prior", "native_tracks"})
    nul = json.load(open(os.path.join(BOX2, "frozen", "argvs_stopbal.json"), encoding="utf-8"))["BAL-NULL_s0"]["argv"]
    dflags = (("v9_labels_train", "--v9-train"), ("lead_train", "--lead-train"), ("speed_train", "--speed-train"), ("join_train", "--join-train"), ("p10_prior", "--p10"), ("native_tracks", "--tracks"))
    check("those data files are the first dose's NULL arm's (the same six paths)", all(nul[nul.index(f) + 1] == dp[k] for k, f in dflags))
    check("all four arms carry the same data flags", all(CK.flag_value(CK.tokens(t), f) == dp[k] for t in TAGS for k, f in dflags))
    # ---- argv composition: each arm equals the first dose's twin with only the flag value changed
    first = json.load(open(os.path.join(BOX2, "frozen", "argvs_stopbal.json"), encoding="utf-8"))
    for t in TAGS:
        a = CK.tokens(t)
        twin = first[t.replace("G2", "G").replace("L2", "L")]["argv"]
        d = [i for i in range(len(a)) if a[i] != twin[i]] if len(a) == len(twin) else None
        flag = "--goal-pos-weight-stop" if "G2" in t else "--lon-class-weight-stop"
        want = [a.index(flag) + 1, a.index("--out") + 1]
        check(f"{t}: equals the first dose's twin with only {flag} and --out changed", d == sorted(want))
        check(f"{t}: carries exactly its own lever ({'20.81' if 'G2' in t else '22.08'}) and not the other", CK.flag_value(a, flag) == ("20.81" if "G2" in t else "22.08")
              and CK.flag_value(a, "--lon-class-weight-stop" if "G2" in t else "--goal-pos-weight-stop") is None)
        check(f"{t}: --out is D:\\tanitad\\runs\\v8f_stopbal\\<tag>", CK.out_decl(t) == "D:\\tanitad\\runs\\v8f_stopbal\\" + t)
        check(f"{t}: frozen argv accepted", CK.audit_line(t, " ".join(a)) == [])
    check("pilot argv = BAL-G2_s0 argv with --out moved and --steps 200 appended", CK.tokens(PILOT)[-2:] == ["--steps", "200"] and [x for x in CK.tokens(PILOT)[:-2] if x not in CK.tokens("BAL-G2_s0")] == [CK.out_decl(PILOT)])
    check("pilot out dir is under pilot_out and ends BAL-G2_s0", "pilot_out" in CK.out_decl(PILOT) and CK.out_decl(PILOT).endswith("BAL-G2_s0"))
    check("the first dose's tags are not tags here (refused by tokens)", all(_raises(lambda x=x: CK.tokens(x)) for x in ("BAL-G_s0", "BAL-L_s1", "BAL-NULL_s0", "PILOT_BAL-G_s0")))
    check("the CLI refuses an old tag (rc 2)", CK.base.main(["argv", "--tag", "BAL-L_s1"]) == 2 if _quiet() else False)
    # ---- red arms on the argv
    g = CK.tokens("BAL-G2_s0")
    i = g.index("--goal-pos-weight-stop") + 1
    r = CK.audit_line("BAL-G2_s0", " ".join(g[:i] + ["4.5616"] + g[i + 1:]))
    check("RED wrong weight: G2 carrying the first dose's 4.5616 is refused and the first differing token is named", len(r) == 1 and "'4.5616'" in r[0] and "'20.81'" in r[0])
    check("RED wrong weight: L2 carrying 4.6991 is refused", CK.audit_line("BAL-L2_s1", " ".join(CK.tokens("BAL-L2_s1")).replace("22.08", "4.6991")) != [])
    check("RED wrong weight: a half-rounded 20.8 is refused", CK.audit_line("BAL-G2_s1", " ".join(CK.tokens("BAL-G2_s1")).replace("20.81", "20.8")) != [])
    check("RED the lever dropped from G2 is refused", CK.audit_line("BAL-G2_s0", " ".join(g).replace(" --goal-pos-weight-stop 20.81", "")) != [])
    check("RED the lon lever added to G2 is refused (separate arms)", CK.audit_line("BAL-G2_s0", " ".join(g) + " --lon-class-weight-stop 22.08") != [])
    check("RED the G2 argv shipped for the L2 tag is refused", CK.audit_line("BAL-L2_s0", " ".join(g)) != [])
    check("RED a seed-0 argv shipped for an s1 tag is refused", CK.audit_line("BAL-G2_s1", " ".join(g)) != [])
    check("RED a changed seed / steps / detach are refused", CK.audit_line("BAL-G2_s0", " ".join(g).replace("--seed 0", "--seed 3")) != [] and CK.audit_line("BAL-G2_s0", " ".join(g).replace("--steps 3000", "--steps 2000")) != []
          and CK.audit_line("BAL-G2_s0", " ".join(g) + " --detach-upper") != [])
    # RED the frozen argv file changed -> the loader refuses
    keep = CK.base.ARGVS_FILE
    bad_file = os.path.join(tmp, "argvs_changed.json")
    shutil.copyfile(keep, bad_file)
    txt = open(bad_file, encoding="utf-8").read().replace("20.81", "20.82", 1)
    open(bad_file, "w", encoding="utf-8", newline="").write(txt)
    CK.base.ARGVS_FILE = bad_file
    try:
        check("RED a changed frozen argv file is refused by the sha256 pin (SystemExit REFUSED)", _refuses(CK.frozen_argvs))
    finally:
        CK.base.ARGVS_FILE = keep
    check("the genuine argv file loads again", len(CK.frozen_argvs()) == 4)
    # ---- clean selfchecks pass; planted-bad ones are refused, per tag
    runnable = TAGS + [PILOT]
    for t in runnable:
        toks, sc = sc_of(t, tmp)
        check(f"clean selfcheck passes {t}", CK.gate_problems(t, toks, sc) == [])
    for t in runnable:
        toks, sc = sc_of(t, tmp)
        kind = CK.TABLE[CK.base_tag(t)]["kind"]
        own, other_own = ("goal_pos_weight_stop", "lon_class_weight_stop") if kind == "g" else ("lon_class_weight_stop", "goal_pos_weight_stop")
        want_own = CK.WEIGHTS[kind][0 if kind == "g" else 1]
        muts = {
            "own lever reads 1.0 (never built)": lambda s: s["levers"].__setitem__(own, 1.0),
            "own lever reads the first dose's weight": lambda s: s["levers"].__setitem__(own, 4.5616 if kind == "g" else 4.6991),
            "own lever off by one digit": lambda s: s["levers"].__setitem__(own, want_own + 0.01),
            "own lever absent": lambda s: s["levers"].pop(own),
            "other lever built (stacked arm)": lambda s: s["levers"].__setitem__(other_own, 4.0),
            "own lever a string": lambda s: s["levers"].__setitem__(own, str(want_own)),
            "declared_vs_built FAIL": lambda s: s.__setitem__("declared_vs_built", "FAIL"),
            "steps_total": lambda s: s.__setitem__("steps_total", 12345),
            "S frozen": lambda s: s["levers"]["params"].__setitem__("S", 0),
            "T frozen": lambda s: s["levers"]["params"].__setitem__("T", 0),
            "goal head dead": lambda s: s["live"].update(dead_params=["h_goal.0.weight"], unexpected_dead=["h_goal.0.weight"]),
            "goal head dead (dead_params only)": lambda s: s["live"].__setitem__("dead_params", ["h_goal.0.weight"]),
            "lon head dead (dead_params only)": lambda s: s["live"].__setitem__("dead_params", ["h_lon.2.bias"]),
            "live goal term zero": lambda s: s["live"]["terms"].__setitem__("goal", 0.0),
            "live lon term nan": lambda s: s["live"]["terms"].__setitem__("lon", float("nan")),
            "bank sha built": lambda s: s["levers"].__setitem__("dagger_bank_sha256", "0" * 64),
            "init sha": lambda s: s["info"]["init_ckpt"].__setitem__("sha256", "0" * 64),
            "train data sha": lambda s: s["info"].__setitem__("train_sha256", "0" * 64),
            "dagger_p built 0": lambda s: s["levers"].__setitem__("dagger_p", 0.0),
            "detach probe present": lambda s: s.__setitem__("otband_probe", {"otband": "detach-only", "detach_upper": True, "pass": True}),
        }
        for name, f in muts.items():
            s2 = copy.deepcopy(sc)
            f(s2)
            check(f"{t}: planted '{name}' is refused", len(CK.gate_problems(t, toks, s2)) >= 1)
        s3 = copy.deepcopy(sc)
        s3["levers"].update(goal_pos_weight_stop=1.0, lon_class_weight_stop=1.0)
        pr = CK.gate_problems(t, toks, s3)
        check(f"{t}: both weights reading 1.0 is refused and the stop line names the lever and both values", any(own in x and "1.0" in x and repr(want_own) in x for x in pr))
        s4 = copy.deepcopy(sc)
        s4["levers"].update(goal_pos_weight_stop=CK.WEIGHTS["l" if kind == "g" else "g"][0], lon_class_weight_stop=CK.WEIGHTS["l" if kind == "g" else "g"][1])
        check(f"{t}: the twin kind's weights built is refused", CK.gate_problems(t, toks, s4) != [])
    toks, sc = sc_of("BAL-G2_s0", tmp)
    sc["levers"]["goal_pos_weight_stop"] = 4.5616
    check("BAL-G2_s0 reading the first dose's 4.5616: the refusal names the lever, the read value and 20.81",
          any("goal_pos_weight_stop" in x and "4.5616" in x and "20.81" in x for x in CK.gate_problems("BAL-G2_s0", toks, sc)))
    for t, mut in (("BAL-G2_s0", lambda a: [("3" if x == "0" and a[i - 1] == "--seed" else x) for i, x in enumerate(a)]), ("BAL-L2_s1", lambda a: [x for x in a if x not in ("--lon-class-weight-stop", "22.08")]),
                   ("BAL-G2_s1", lambda a: [("20.82" if x == "20.81" else x) for x in a])):
        toks, sc = sc_of(t, tmp)
        check(f"{t}: a clean selfcheck does not rescue a changed pinned argv", len(CK.gate_problems(t, mut(list(toks)), sc)) >= 1)
    # ---- the post check on planted checkpoints
    root = os.path.join(tmp, "box", "D", "tanitad")
    os.environ.update(STOPBAL_ROOT=root, STOPBAL_COMMIT="a" * 40, STOPBAL_CKPT_LOADER="pickle", STOPBAL_FLOOR_BYTES="1000")
    for s in (0, 1):
        put(os.path.join(root, "runs", "v8f_mini_full", f"M2b_s{s}", "ckpt_last.pt"), pickle.dumps({"model": PT.model_state(s)}))
    for t in TAGS:
        spec = CK.TABLE[t]
        toks = CK.tokens(t)
        out = os.path.join(tmp, "post", t)
        os.makedirs(out, exist_ok=True)

        def write(mut=None, cfg_over=None, tag=t, spec=spec, toks=toks, out=out):
            sc = CK.argv_sha16(toks)
            st = PT.model_state(spec["init"])
            for k in st:
                if k.startswith(("read_s", "head_s", "read_t", "trunk_t", "h_goal", "read_o", "head_o")):
                    st[k] = st[k] + 1.0
            wg, wl = CK.WEIGHTS[spec["kind"]]
            cfg = {"detach_upper": False, "w_traj": 1.0, "w_acc": CK.W_ACC, "goal_pos_weight_stop": wg, "lon_class_weight_stop": wl}
            cfg.update(cfg_over or {})
            ck = {"model": st, "cfg": cfg, "step": 3000, "seed": spec["seed"], "arm": "M2b", "argv_sha": sc, "dagger": {"dagger_p": CK.DAGGER_P, "dagger_seed": spec["init"], "dagger_bank_sha256": CK.BANK_SHA[spec["init"]]},
                  "init_ckpt": {"sha256": CK.INIT_SHA[spec["init"]]}}
            if mut:
                mut(ck)
            pickle.dump(ck, open(os.path.join(out, "ckpt_last.pt"), "wb"))
            json.dump({"done": True, "steps": 3000, "arm": "M2b", "seed": spec["seed"], "argv_sha": sc}, open(os.path.join(out, "summary.json"), "w"))
        os.environ["STOPBAL_FAKE_INIT_SHA_FILE"] = os.path.join(tmp, "init_sha.json")
        write()
        # the init checkpoint's sha is the fake one; the base reads the init from its file, so a post problem naming only the init sha is expected here (the fake init differs from the real pin)
        _ev, pp = CK.post_problems(t, toks, out)
        rest = [x for x in pp if "init_ckpt.sha256" not in x]
        check(f"{t}: a correct banked checkpoint raises no post problem (apart from the planted fake init's sha)", rest == [])
        write(cfg_over={"goal_pos_weight_stop": 4.5616 if spec["kind"] == "g" else 1.0, "lon_class_weight_stop": 1.0 if spec["kind"] == "g" else 4.6991})
        check(f"{t}: RED checkpoint carrying the first dose's weight is refused", any("cfg." in x for x in CK.post_problems(t, toks, out)[1]))
        write(cfg_over={"goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0})
        check(f"{t}: RED checkpoint whose config lost the declared weight is refused", any("cfg." in x for x in CK.post_problems(t, toks, out)[1]))
        write(mut=lambda ck: ck.__setitem__("seed", spec["seed"] + 1))
        check(f"{t}: RED checkpoint of another seed is refused", any("seed" in x for x in CK.post_problems(t, toks, out)[1]))
    # ---- text hygiene of the package's own prose
    words = ["vo" + "id", "super" + "seded", "re" + "tracted", "with" + "drawn", "no lon" + "ger", "closed" + " loop", "closed" + "-loop"]
    banned = re.compile("(?<![A-Za-z])(" + "|".join(words) + ")(?![A-Za-z])", re.I)
    for f in ("dose2_chain.sh", "stages/dose2_run.sh", "stages/dose2_ship.sh", "stages/stopbal_common.sh", "stopbal_check.py", "make_argvs_dose2.py", "tests/test_dose2_unit.py", "README.md"):
        if not os.path.exists(os.path.join(PKG, f)):
            continue
        body = open(os.path.join(PKG, f), encoding="utf-8").read()
        check(f"{f} keeps clear of the banned words", banned.search(body) is None)
    print(f"UNIT2 ok={N_OK} bad={N_BAD}")
    return 0 if N_BAD == 0 else 1


def _raises(fn):
    try:
        fn()
    except KeyError:
        return True
    except Exception:
        return False
    return False


def _refuses(fn):
    try:
        fn()
    except SystemExit as e:
        return "REFUSED" in str(e)
    return False


def _quiet():
    return True


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
