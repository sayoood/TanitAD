"""make_wrapper_p.py -- writes v8f_t1_eval_reader_rule_rzgfhqetjvp.py from ../visrule/v8f_t1_eval_reader_rule_rzgfhqetjv.py (LF text, sha256 prefix 28bc31f21ecb4ccb asserted)
by counted, asserted text edits (each old text exactly once), and WRAPPER_P_GEN_RECORD.json (source hash, variant hash, the edits with a label each, pass flag). Study "preswrap".
Two new arguments, both off by default and both for form RR only:
  --vis-pres-file NPZ    a presence file as written by preshead/fit/pres_dump.py: arrays p (float64, one row per row of the evaluation join, row = off[clip index] + frame; not a number
                         where the head has no output), has_out (bool), clips (text), off, n (int), seed. Where the reader's probability p of a (clip, frame) is turned into presence (the tick
                         look, the later looks, and the earlier frames the closing slope of --vis-closing reads again), p is first replaced by p2 (mode head: the file's value; mode or: the larger
                         of the two); a frame without a finite file value keeps the reader's own p (counted).
  --vis-pres-mode head|or  required with the file, refused without it.
Everything else (ok, the gap, the closing input, --vis-gap-file, --vis-closing, --rule-join) is as in the source, now seeing presence from p2. With the file absent every behaviour and output is
the source's. Usage: python make_wrapper_p.py [--parent PATH] [--out PATH] [--record PATH]. Refuses another parent hash and refuses to overwrite the output or the record. Standard library only."""
import argparse
import ast
import difflib
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "visrule", "v8f_t1_eval_reader_rule_rzgfhqetjv.py"))
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqetjvp.py")
RECORD = os.path.join(HERE, "WRAPPER_P_GEN_RECORD.json")
PARENT_SHA16 = "28bc31f21ecb4ccb"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhqetjv.py -- written by make_wrapper_v.py from'
HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqetjvp.py -- written by make_wrapper_p.py from v8f_t1_eval_reader_rule_rzgfhqetjv.py (study "preswrap").
  New flag --vis-pres-file NPZ (default empty; form RR only): a presence file of preshead/fit/pres_dump.py (arrays p, has_out, clips, off, n, seed; row = off[clip index] + frame).
  New flag --vis-pres-mode head|or (required with the file): the reader's probability p of a (clip, frame) is replaced by p2 before presence is tested: head = the file's value, or = max(p, file value);
    a frame without a finite file value keeps p (counted). The tick look, the later looks and the earlier frames of the closing run all test presence on p2.
  With the file absent every behaviour and output is the source's. Rule log meta gains vis_pres_file, vis_pres_file_sha256, vis_pres_mode, vis_pres_counts (only with the file); the side files
  carry p (the reader's own) and, only with the file, p_used (= p2); the column p_src of the logs is p2. Text of the source (rzgfhqetjv) follows.
'''

PRES_HELPER = r'''PRES_MODES = ("head", "or")
STATE["pres"] = None                                   # set by main() when --vis-pres-file is given
STATE["pres_counts"] = {"rows": 0, "with_file": 0, "no_file": 0, "raised": 0, "lowered": 0}


def load_pres(path, mode, join):
    """-> the state of the presence file (arrays p, has_out, clips, off, n of pres_dump.py) checked against `join` (the evaluation join the run uses: clip, off, n). Refuses a missing or malformed file."""
    if mode not in PRES_MODES:
        raise SystemExit("REFUSED: --vis-pres-mode must be head or or, got %r" % (mode,))
    if not path or not os.path.isfile(path):
        raise SystemExit("REFUSED: --vis-pres-file not found: %s" % path)
    if join is None:
        raise SystemExit("REFUSED: --vis-pres-file needs the evaluation join to check against")
    with np.load(path, allow_pickle=False) as z:
        for k in ("p", "has_out", "clips", "off", "n"):
            if k not in z.files:
                raise SystemExit("REFUSED: --vis-pres-file lacks the array %s" % k)
        p, has = np.asarray(z["p"], np.float64), np.asarray(z["has_out"], bool)
        clips, off, n = np.asarray(z["clips"]), np.asarray(z["off"], np.int64), np.asarray(z["n"], np.int64)
    if not (p.ndim == has.ndim == 1 and len(p) == len(has) and len(p) > 0):
        raise SystemExit("REFUSED: --vis-pres-file arrays p and has_out are not one-dimensional of one length")
    if not (clips.ndim == off.ndim == n.ndim == 1 and len(clips) == len(off) == len(n)):
        raise SystemExit("REFUSED: --vis-pres-file arrays clips, off, n are not one-dimensional of one length")
    jc = [str(c) for c in np.asarray(join["clip"]).ravel()]
    if not np.array_equal(off, np.asarray(join["off"], np.int64)) or not np.array_equal(n, np.asarray(join["n"], np.int64)):
        raise SystemExit("REFUSED: --vis-pres-file off/n differ from those of the evaluation join")
    if [str(c) for c in clips] != jc:
        raise SystemExit("REFUSED: --vis-pres-file clip names differ from those of the evaluation join")
    if len(p) != int((off + n).max()):
        raise SystemExit("REFUSED: --vis-pres-file has %d rows, the join has %d" % (len(p), int((off + n).max())))
    fin = np.isfinite(p)
    if (has & ~fin).any():
        raise SystemExit("REFUSED: --vis-pres-file marks an output where p is not finite")
    if (p[fin] < 0.0).any() or (p[fin] > 1.0).any():
        raise SystemExit("REFUSED: --vis-pres-file has a p outside [0, 1]")
    return {"p": p, "use": has & fin, "off": off, "n": n, "mode": mode, "path": path}


def pres_values(S, ci, fr):
    """-> (pf [B] float64, NaN where there is no usable file value, have [B] bool) at (clip index, frame); a clip index or frame outside the join has none."""
    ci, fr = np.asarray(ci, np.int64), np.asarray(fr, np.int64)
    nc = len(S["off"])
    inside = (ci >= 0) & (ci < nc)
    cc = np.where(inside, ci, 0)
    inside = inside & (fr >= 0) & (fr < S["n"][cc])
    row = np.where(inside, S["off"][cc] + fr, 0)
    have = inside & S["use"][row]
    return np.where(have, S["p"][row], np.nan), have


def pres_replace(S, ci, fr, p, count=None):
    """p2 [B] float64: p where S is None or the file has no value; mode head: the file's value; mode or: the larger of p and the file's value (a p that is not a number takes the file's value)."""
    p = np.asarray(p, np.float64)
    if S is None:
        return p.copy()
    pf, have = pres_values(S, ci, fr)
    p2 = np.where(have, pf if S["mode"] == "head" else np.fmax(p, pf), p)
    if count is not None:
        with np.errstate(invalid="ignore"):
            count["rows"] += int(len(p))
            count["with_file"] += int(have.sum())
            count["no_file"] += int((~have).sum())
            count["raised"] += int(((p < P_MIN) & (p2 >= P_MIN)).sum())
            count["lowered"] += int(((p >= P_MIN) & (p2 < P_MIN)).sum())
    return p2


def pres_look(st, e_off, p, count=True):
    """p2 for frame e = j + e_off of each window of `st` (states with .ci and .j)."""
    ci = np.array([s.ci for s in st], np.int64)
    fr = np.array([int(s.j) + int(e_off) for s in st], np.int64)
    return pres_replace(STATE["pres"], ci, fr, p, STATE["pres_counts"] if count else None)


def pres_tick(p):
    """p2 for the tick-level look (frame j of each window of the batch)."""
    st = STATE["vis_states"]
    if st is None or len(st) != len(p):
        raise RuntimeError("the presence file needs the states of this batch (ReaderPlanner not bound)")
    return pres_look(st, 0, p)


def pres_fields(path, sha, mode):
    """the text the first line of the run log gains when the flag is given."""
    return " vis-pres-file %s vis pres sha256 %s vis-pres-mode %s" % (path, sha, mode)


def pres_meta(path, mode, sha):
    """the run record fields; nothing without the flag."""
    if not path:
        return {}
    return {"vis_pres_file": path, "vis_pres_file_sha256": sha, "vis_pres_mode": mode, "vis_pres_counts": json.dumps(STATE["pres_counts"], sort_keys=True)}


'''


def helper_ns(P_MIN=0.5):
    """namespace in which PRES_HELPER runs on its own (for the selftest and the tests): np, os, json, STATE, P_MIN."""
    import numpy as np
    ns = {"np": np, "os": os, "json": json, "STATE": {}, "P_MIN": P_MIN}
    exec(compile(PRES_HELPER, "<pres_helper>", "exec"), ns)
    return ns


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


def build(parent_text):
    t = parent_text
    assert chr(13) not in t
    parent = t
    anchors, plines = [], {}

    def E(old, new, name):
        nonlocal t
        t = once(t, old, new)
        plines[name] = parent[:parent.index(old)].count("\n") + 1
        anchors.append((name, new))

    E(OLD_FIRST, HEADER + OLD_FIRST.replace('"""v8f_t1_eval_reader_rule_rzgfhqetjv.py -- ', 'v8f_t1_eval_reader_rule_rzgfhqetjv.py -- '), "P1 header docstring")
    E('    ap.add_argument("--vis-closing", action="store_true")\n',
      '    ap.add_argument("--vis-closing", action="store_true")\n    ap.add_argument("--vis-pres-file", default="")\n    ap.add_argument("--vis-pres-mode", choices=("head", "or"), default="")\n', "P2 flags --vis-pres-file and --vis-pres-mode")
    E('        raise SystemExit("REFUSED: --vis-gap-file and --vis-closing belong to form RR only")\n',
      '        raise SystemExit("REFUSED: --vis-gap-file and --vis-closing belong to form RR only")\n'
      '    if (mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RR":\n        raise SystemExit("REFUSED: --vis-pres-file and --vis-pres-mode belong to form RR only")\n'
      '    if mine.vis_pres_file and not mine.vis_pres_mode:\n        raise SystemExit("REFUSED: --vis-pres-file needs --vis-pres-mode head or or")\n'
      '    if mine.vis_pres_mode and not mine.vis_pres_file:\n        raise SystemExit("REFUSED: --vis-pres-mode needs --vis-pres-file")\n', "P3 refusals: form RR only, file and mode together")
    E('        if STATE.get("vis") is not None:\n            STATE["vis_states"] = self.states',
      '        if STATE.get("vis") is not None or STATE.get("pres") is not None:\n            STATE["vis_states"] = self.states', "P4 states for the presence file")
    E('    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])\n    vis_c = None\n',
      '    p_rd = p                                                    # the reader\'s own probability (logged); presence is tested on p2 (study preswrap)\n'
      '    if STATE.get("pres") is not None:\n        p = pres_tick(p)\n'
      '    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])\n    vis_c = None\n', "P5 tick-level p2 before feed")
    E('    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):\n        STATE["cols"][k].append(np.asarray(v))\n',
      '    for k, v in (("p", p_rd), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):\n        STATE["cols"][k].append(np.asarray(v))\n'
      '    if STATE.get("pres") is not None:\n        STATE["cols"].setdefault("p_used", []).append(np.asarray(p))\n', "P6 tick-level log: p own, p_used")
    E('            p, g, okr = _reader_at(st, rd, q)\n            c_read = _reader_closing_at(st, rd, q)',
      '            p, g, okr = _reader_at(st, rd, q)\n            p_rd = p\n            if STATE.get("pres") is not None:            # study preswrap: presence at the later look is tested on p2\n'
      '                p = pres_look(st, q, p)\n            c_read = _reader_closing_at(st, rd, q)', "P7 later-look p2")
    E('            for kk, vv in (("p", p), ("g_read", g), ("g_used", used), ("ok", okr)):\n                M["rd"][kk].append(np.asarray(vv))\n',
      '            for kk, vv in (("p", p_rd), ("g_read", g), ("g_used", used), ("ok", okr)):\n                M["rd"][kk].append(np.asarray(vv))\n'
      '            if STATE.get("pres") is not None:\n                M["rd"].setdefault("p_used", []).append(np.asarray(p))\n', "P8 later-look log: p own, p_used")
    E('            pe, ge, oke = _reader_at(st, rd, int(e_off) - d)\n',
      '            pe, ge, oke = _reader_at(st, rd, int(e_off) - d)\n            if STATE.get("pres") is not None:           # study preswrap: the earlier frames of the run see presence on p2 too (not counted again)\n'
      '                pe = pres_look(st, int(e_off) - d, pe, False)\n', "P9 closing run: earlier frames p2")
    E('        STATE["vis"] = load_vis(mine.vis_gap_file, mine.vis_closing) if (mine.vis_gap_file or mine.vis_closing) else None\n',
      '        STATE["vis"] = load_vis(mine.vis_gap_file, mine.vis_closing) if (mine.vis_gap_file or mine.vis_closing) else None\n'
      '        STATE["pres"] = load_pres(mine.vis_pres_file, mine.vis_pres_mode, T.D.load_join(a.eval_join)) if mine.vis_pres_file else None\n'
      '        if STATE["pres"] is not None:\n'
      '            print("pres run-line fields:" + pres_fields(mine.vis_pres_file, W.sha256_file(mine.vis_pres_file), mine.vis_pres_mode), flush=True)\n', "P10 load and print the run-line fields")
    E('"vis_counts": json.dumps(STATE["vis_counts"], sort_keys=True), "reader": mine.reader,',
      '"vis_counts": json.dumps(STATE["vis_counts"], sort_keys=True),\n'
      '                    **pres_meta(mine.vis_pres_file, mine.vis_pres_mode, W.sha256_file(mine.vis_pres_file) if mine.vis_pres_file and os.path.isfile(mine.vis_pres_file) else ""), "reader": mine.reader,', "P11 run record fields")
    idx = t.index("def build_rule_label(")                 # after the block of the source's own helper, before build_rule_label
    t = t[:idx] + PRES_HELPER + t[idx:]
    anchors.append(("P12 presence helper block", PRES_HELPER))
    plines["P12 presence helper block"] = parent[:parent.index("def build_rule_label(")].count("\n") + 1
    ast.parse(t)
    return t, anchors, plines


def sha_lf(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--parent", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--record", default=RECORD)
    a = ap.parse_args(argv)
    for pth in (a.out, a.record):
        if os.path.exists(pth):
            sys.exit("REFUSED: %s exists" % pth)
    raw = open(a.parent, "rb").read()
    if b"\r" in raw:
        sys.exit("REFUSED: the parent text holds a carriage return (LF text expected)")
    sha = hashlib.sha256(raw).hexdigest()
    if sha[:16] != PARENT_SHA16:
        sys.exit("REFUSED: parent sha256 %s is not %s" % (sha[:16], PARENT_SHA16))
    ptext = raw.decode("utf-8")
    t, anchors, plines = build(ptext)
    compile(t, a.out, "exec")
    out_sha = sha_lf(t)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    rec = {"source": os.path.basename(a.parent), "source_sha256": sha, "variant": os.path.basename(a.out), "variant_sha256": out_sha, "variant_lines": t.count("\n"), "source_lines": ptext.count("\n"),
           "substitutions": [{"id": n.split(" ")[0], "label": n, "source_line": plines[n]} for n, _ in anchors], "pass": True}
    with open(a.record, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    print("wrote", a.out, out_sha[:16], t.count("\n"), "lines (parent %d lines, sha256 %s)" % (ptext.count("\n"), sha[:16]))
    for name, _ in anchors:
        print("   %-52s parent L%d" % (name, plines[name]))
    d = list(difflib.unified_diff(ptext.split("\n"), t.split("\n"), lineterm="", n=0))
    add = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    dele = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    print("diff summary: %d lines added, %d lines removed" % (add, dele))
    print("sha256", out_sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
