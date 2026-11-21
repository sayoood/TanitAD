r"""Builds the STOPPERC stage-1 argvs (PREREG_V8F_STOPPERC section 5, Q3). Code of record: 3e264bbf3fa2f6f17bea448de09aa972faaf7725 (child of 715ef8fc, BAL-NULL's commit).
  NULL_s0    = BAL-NULL_s0's argv (DAG-OACC s0 + --out), ONLY --out changed (stop-aux off by default: no new flag).
  NULL_rep2  = NULL_s0 with --seed 3 (Q3; BAL-NULL_rep = NULL s0 with --seed 2, so the new replicate takes the next seed).
  A_s0, A_s1 = DAG-OACC s0 / s1 argv + EXACTLY  --stop-aux on --w-stop-aux W_PLACEHOLDER --stop-aux-extr <box-2 table path>. W is filled from the gate smoke (stopperc_check.py argv reads STOPPERC_W).
  NULL_s1, NULL_rep = CONDITIONAL (used only if NULL_s0's model tensors differ from BAL-NULL_s0's): BAL-NULL_s1 / BAL-NULL_rep argvs with --out changed.
Each argv is parsed by the trainer at the code of record (export dir STOPPERC_TREE) and its diff against the template is asserted. Writes argvs_stopperc.txt (one `tag: argv` line per arm) beside this file, or into $STOPPERC_OUT.
Usage: STOPPERC_TREE=<export of 3e264bbf> python make_argvs_stopperc.py"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("STOPPERC_OUT", HERE)
TREE = os.environ["STOPPERC_TREE"]
SRC = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
BAL = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/box2/argvs_stopbal.txt"
COMMIT = "3e264bbf3fa2f6f17bea448de09aa972faaf7725"
OUT_ROOT = "D:\\tanitad\\runs\\v8f_stopperc\\"
TABLE_B2 = "D:\\tanitad\\stopperc\\data\\stopaux_extr_join4572.json"
TABLE_SHA = "77b6e8dc63d07d90548d09182fd5b8073e81079e224e8ce672dda11d2c3ae361"
WPH = "W_PLACEHOLDER"
ORDER = ["NULL_s0", "NULL_rep2", "A_s0", "A_s1", "NULL_s1", "NULL_rep"]          # the last two are conditional


def bal_lines():
    return {l.split(": ", 1)[0]: l.rstrip("\n").split(": ", 1)[1].split(" ") for l in open(BAL, encoding="utf-8") if l.strip()}


def main():
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    import v8f_mini_train as TR
    assert os.path.realpath(TR.__file__).startswith(os.path.realpath(TREE)), TR.__file__
    bal = bal_lines()
    dag = {k: json.load(open(f"{SRC}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8"))["argv"] for k in (0, 1)}

    def set_out(av, tag):
        av = list(av)
        av[av.index("--out") + 1] = OUT_ROOT + tag
        return av

    def same_but_out(a, b):
        i = a.index("--out")
        return len(a) == len(b) and all(x == y for n, (x, y) in enumerate(zip(a, b)) if n != i + 1)
    arms, decl = {}, {}
    for tag, base, seed, extra in (("NULL_s0", "BAL-NULL_s0", None, []), ("NULL_rep2", "BAL-NULL_s0", "3", []), ("NULL_s1", "BAL-NULL_s1", None, []), ("NULL_rep", "BAL-NULL_rep", None, [])):
        av = set_out(bal[base], tag)
        if seed:
            av[av.index("--seed") + 1] = seed
        arms[tag] = av
        decl[tag] = f"{base} argv, --out" + (f", --seed {seed}" if seed else "")
        want = bal[base]
        if seed:
            want = list(want)
            want[want.index("--seed") + 1] = seed
        assert same_but_out(av, want), tag
    # the BAL arms are DAG-OACC's argv + --out (+ seed 2 for _rep): re-derive from DAG-OACC and require equality
    assert same_but_out(arms["NULL_s0"], dag[0]) and same_but_out(arms["NULL_s1"], dag[1]), "BAL-NULL is not DAG-OACC's argv + --out"
    rep = list(dag[0]); rep[rep.index("--seed") + 1] = "2"
    assert same_but_out(arms["NULL_rep"], rep), "BAL-NULL_rep differs from DAG-OACC s0 only by --seed 2 and --out"
    for k, tag in ((0, "A_s0"), (1, "A_s1")):
        av = set_out(dag[k], tag) + ["--stop-aux", "on", "--w-stop-aux", WPH, "--stop-aux-extr", TABLE_B2]
        arms[tag] = av
        decl[tag] = f"DAG-OACC s{k} argv, --out, + --stop-aux on --w-stop-aux <W> --stop-aux-extr <table>"
    for tag, av in arms.items():
        probe = [WPH and "1.0" if x == WPH else x for x in av]
        ns = TR.build_parser().parse_args(probe)
        on = tag.startswith("A_")
        assert (ns.stop_aux == "on") == on and ((ns.stop_aux_extr == TABLE_B2) if on else ns.stop_aux_extr is None), tag
        assert ns.seed == (3 if tag == "NULL_rep2" else 2 if tag == "NULL_rep" else int(av[av.index("--seed") + 1])), tag
        assert ns.grid == "pool4x10" and ns.arm == "M2b", tag
    for tag in ("A_s0", "A_s1"):
        d = arms[tag][len(dag[0 if tag == "A_s0" else 1]):]
        assert d == ["--stop-aux", "on", "--w-stop-aux", WPH, "--stop-aux-extr", TABLE_B2], d
    text = "".join(f"{t}: " + " ".join(arms[t]) + "\n" for t in ORDER)
    with open(os.path.join(OUT, "argvs_stopperc.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(len(arms), "argvs parse at", COMMIT[:8], "; argvs sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
