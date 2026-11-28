"""make_wrapper_v.py -- writes v8f_t1_eval_reader_rule_rzgfhqetjv.py from ../gaprule/v8f_t1_eval_reader_rule_rzgfhqetj.py (sha256 prefix cbf7063d867c9f4e asserted) by counted, asserted text edits
(each old text exactly once). Card PREREG_V8F_PROOF_ADD102 (VISRULE10K).
Two new arguments, both off by default; with both off the variant behaves as its source (the changed functions return what the source's return; the pod check compares whole rollouts).
  --vis-gap-file NPZ   arrays clip_index (int, row of the clip in the evaluation join), frame (int), gap (float, m; NaN where the head has no output), has_out (bool). In form RR, on a frame
                       where the reader marks a lead (probability >= 0.5 and a full history, the wrapper's own presence rule) and the file has an output for (clip index, frame), the gap the rule
                       receives is the file's gap, as dumped: the reader's own map (min(g, Q(g))) is bypassed on that frame. Where the reader marks a lead and the file has no output (or no row), the
                       rule receives the reader's own gap as in RR (counted). Frames the reader does not mark carry no lead, as in RR.
  --vis-closing        in form RR, the rule's closing input is the slope computed as LabelSource.read computes it (distrule/v8f_t1_eval_rule.py:92-112), on the gap the rule receives: least squares over
                       the trailing run of at most 5 frames (the frame of the look and up to four earlier frames), a frame is in the run when the reader marks a lead on it and the gap the rule
                       receives there is finite and in 0 to 60 m (the test of LabelSource._ok), a step of more than 3.0 m between neighbouring frames ends the run, a run below 3 frames gives 0.
                       Without the switch the closing input is 0 as in RR.
Both arguments belong to form RR only (refused otherwise). The run record meta gains vis_gap_file, vis_gap_file_sha256, vis_closing and vis_counts (JSON: lead frames, gap from the file,
reader's own gap where the file has no output, where the file has no row, rows with a non-zero closing input).
Usage: python make_wrapper_v.py [--parent PATH] [--out PATH]. Refuses a parent with another sha256 and refuses to overwrite the output (VW_OVERWRITE_OWN=1 rewrites it).
Prints per edit the parent line and the new line, and a diff summary. Standard library only."""
import argparse
import ast
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "gaprule", "v8f_t1_eval_reader_rule_rzgfhqetj.py"))
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqetjv.py")
PARENT_SHA16 = "cbf7063d867c9f4e"

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhqetj.py -- written by make_wrapper_j.py'
HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhqetjv.py -- written by make_wrapper_v.py from v8f_t1_eval_reader_rule_rzgfhqetj.py (card ADD102, study "visrule").
  New flag --vis-gap-file NPZ (default empty; form RR only): where the reader marks a lead and the file (the fixed-window head's gap per clip and frame) has an output, the rule receives the file's gap.
  New switch --vis-closing (default off; form RR only): the closing input is the slope of the gap the rule receives, by the run rule of LabelSource (at most 5 frames, steps up to 3.0 m, at least 3 frames).
  With both off every behaviour is the source's. The rule log meta gains vis_gap_file, vis_gap_file_sha256, vis_closing, vis_counts. Text of the source (rzgfhqetj) follows.
'''
VIS_HELPER = r'''VIS_STRIDE = 1 << 20
VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 3        # the constants of LabelSource (distrule/v8f_t1_eval_rule.py:37-39)
STATE["vis"] = None                                    # set by main() when --vis-gap-file or --vis-closing is given
STATE["vis_states"] = None                             # the states of the batch of the tick (ReaderPlanner.__call__)
STATE["vis_counts"] = {"present": 0, "from_head": 0, "no_head_output": 0, "key_missing": 0, "closing_nonzero": 0, "closing_rows": 0}


def load_vis(path, closing):
    """-> the state of the variant: the gap table of --vis-gap-file (arrays clip_index, frame, gap, has_out; one row per (clip, frame), no key twice) and the closing switch. Refuses a missing or malformed file."""
    tab = None
    if path:
        if not os.path.isfile(path):
            raise SystemExit("REFUSED: --vis-gap-file not found: %s" % path)
        with np.load(path, allow_pickle=False) as z:
            for k in ("clip_index", "frame", "gap", "has_out"):
                if k not in z.files:
                    raise SystemExit("REFUSED: --vis-gap-file lacks the array %s" % k)
            ci, fr = np.asarray(z["clip_index"], np.int64), np.asarray(z["frame"], np.int64)
            gap, has = np.asarray(z["gap"], np.float64), np.asarray(z["has_out"], bool)
        if not (ci.ndim == fr.ndim == gap.ndim == has.ndim == 1 and len(ci) == len(fr) == len(gap) == len(has) and len(ci) > 0):
            raise SystemExit("REFUSED: --vis-gap-file arrays are not one-dimensional of one length")
        if (ci < 0).any() or (fr < 0).any() or (fr >= VIS_STRIDE).any():
            raise SystemExit("REFUSED: --vis-gap-file has a negative clip index or a frame outside [0, %d)" % VIS_STRIDE)
        if (has & ~np.isfinite(gap)).any():
            raise SystemExit("REFUSED: --vis-gap-file marks an output with a gap that is not finite")
        key = ci * VIS_STRIDE + fr
        o = np.argsort(key, kind="stable")
        key = key[o]
        if (np.diff(key) == 0).any():
            raise SystemExit("REFUSED: --vis-gap-file has two rows with the same clip index and frame")
        tab = {"key": key, "gap": gap[o], "has": has[o]}
    return {"tab": tab, "close": bool(closing)}


def vis_lookup(tab, ci, fr):
    """-> (found [B] bool, has [B] bool, gap [B] float64, NaN where there is no output) at (clip index, frame)."""
    ci, fr = np.asarray(ci, np.int64), np.asarray(fr, np.int64)
    key = ci * VIS_STRIDE + fr
    pos = np.searchsorted(tab["key"], key)
    pc = np.minimum(pos, len(tab["key"]) - 1)
    found = (pos < len(tab["key"])) & (tab["key"][pc] == key) & (fr >= 0) & (fr < VIS_STRIDE)
    has = found & tab["has"][pc]
    return found, has, np.where(has, tab["gap"][pc], np.nan)


def vis_replace(tab, ci, fr, present, used, count=None):
    """the gap the rule receives: `used` (the reader's own gap after its map and minimum) where the reader marks no lead or the file has no output, else the file's gap as dumped. -> float64 copy."""
    used = np.array(used, np.float64, copy=True)
    if tab is None:
        if count is not None:
            count["present"] += int(np.asarray(present, bool).sum())
        return used
    found, has, g = vis_lookup(tab, ci, fr)
    present = np.asarray(present, bool)
    sel = present & has
    used[sel] = g[sel]
    if count is not None:
        count["present"] += int(present.sum())
        count["from_head"] += int(sel.sum())
        count["no_head_output"] += int((present & found & ~has).sum())
        count["key_missing"] += int((present & ~found).sum())
    return used


def vis_ok(present, gap):
    """a frame is in the run when the reader marks a lead on it and the gap the rule receives is finite and in 0 to 60 m (LabelSource._ok)."""
    gap = np.asarray(gap, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(present, bool) & np.isfinite(gap) & (gap >= 0.0) & (gap <= VIS_GAP_MAX_M)


def vis_run_closing(g, ok, dt=DT_S):
    """g, ok [B, L] (oldest first, the last column is the frame of the look): the closing input by the run rule of LabelSource.read (distrule/v8f_t1_eval_rule.py:92-112): 0 where the frame of the
    look is not in the run; else the least squares slope over the trailing run of at most VIS_RUN_MAX frames, a step of more than VIS_STEP_MAX_M between neighbours ends the run, a run below
    VIS_RUN_MIN frames gives 0. -> float32 [B]."""
    g, ok = np.asarray(g, np.float64), np.asarray(ok, bool)
    B, L = g.shape
    out = np.zeros(B, np.float32)
    for b in range(B):
        if not ok[b, L - 1]:
            continue
        idx = [L - 1]
        f = L - 2
        while f >= 0 and len(idx) < VIS_RUN_MAX and ok[b, f] and abs(g[b, f + 1] - g[b, f]) <= VIS_STEP_MAX_M:
            idx.append(f)
            f -= 1
        if len(idx) < VIS_RUN_MIN:
            continue
        idx = idx[::-1]
        G = g[b, idx]
        t = np.arange(len(idx), dtype=np.float64) * dt
        tb = t.mean()
        out[b] = float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())
    return out


def vis_look(st, rd, e_off, p, g, used, ok):
    """The look on frame e = j + e_off of each window of `st`. p, g, ok, used: what the reader read and fed at that look (used = after feed()). -> (used with the file's gap where the reader marks a lead and
    the file has an output [B] float64, closing [B] float32 or None when --vis-closing is off). The earlier frames of the run are read again from the logged tokens (_reader_at), fed the same way."""
    S = STATE["vis"]
    ci = np.array([s.ci for s in st], np.int64)
    fr = np.array([int(s.j) + int(e_off) for s in st], np.int64)
    with np.errstate(invalid="ignore"):
        present = (np.asarray(p, np.float64) >= P_MIN) & np.asarray(ok, bool)
    used2 = vis_replace(S["tab"], ci, fr, present, used, STATE["vis_counts"])
    clos = None
    if S["close"]:
        cg, co = [], []
        for d in range(VIS_RUN_MAX - 1, 0, -1):         # the four earlier frames, oldest first
            pe, ge, oke = _reader_at(st, rd, int(e_off) - d)
            prs, ue = feed(STATE["form"], pe, ge, oke, STATE["xs"], STATE["ys"])
            ue = vis_replace(S["tab"], ci, fr - d, prs, ue).astype(np.float32).astype(np.float64)
            cg.append(ue)
            co.append(vis_ok(prs, ue))
        uc = used2.astype(np.float32).astype(np.float64)
        cg.append(uc)
        co.append(vis_ok(present, uc))
        clos = vis_run_closing(np.stack(cg, 1), np.stack(co, 1))
        STATE["vis_counts"]["closing_rows"] += int(len(clos))
        STATE["vis_counts"]["closing_nonzero"] += int((clos != 0).sum())
    return used2, clos


def vis_tick(p, g, used, ok):
    """the tick-level look (frame j of each window of the batch)."""
    st = STATE["vis_states"]
    if st is None or len(st) != len(p):
        raise RuntimeError("the variant needs the states of this batch (ReaderPlanner not bound)")
    return vis_look(st, STATE["reader"], 0, p, g, used, ok)


'''


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

    E(OLD_FIRST, HEADER + OLD_FIRST.replace('"""v8f_t1_eval_reader_rule_rzgfhqetj.py -- ', 'v8f_t1_eval_reader_rule_rzgfhqetj.py -- '), "V1 header docstring")
    E('    ap.add_argument("--rule-join", default="")\n',
      '    ap.add_argument("--rule-join", default="")\n    ap.add_argument("--vis-gap-file", default="")\n    ap.add_argument("--vis-closing", action="store_true")\n', "V2 flags --vis-gap-file and --vis-closing")
    E('    if mine.form != "RRe" and mine.closing_table:\n        raise SystemExit("REFUSED: --closing-table belongs to form RRe only")\n',
      '    if mine.form != "RRe" and mine.closing_table:\n        raise SystemExit("REFUSED: --closing-table belongs to form RRe only")\n'
      '    if (mine.vis_gap_file or mine.vis_closing) and mine.form != "RR":\n        raise SystemExit("REFUSED: --vis-gap-file and --vis-closing belong to form RR only")\n', "V3 refusal outside form RR")
    E('    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])\n',
      '    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])\n    vis_c = None\n    if STATE.get("vis") is not None:                            # the variant of card ADD102: the file\'s gap and the closing input of the rule\n'
      '        used, vis_c = vis_tick(p, g, used, ok)\n', "V4 tick-level gap and closing")
    E('    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else (c_estv if c_estv is not None else c_fed)), ok\n',
      '    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else (vis_c if vis_c is not None else (c_estv if c_estv is not None else c_fed))), ok\n', "V5 tick-level closing return")
    E('            _, used = feed(form, p, g, okr, STATE["xs"], STATE["ys"])\n',
      '            _, used = feed(form, p, g, okr, STATE["xs"], STATE["ys"])\n            vis_hc = None\n            if STATE.get("vis") is not None:             # the variant of card ADD102 at the later look\n'
      '                used, vis_hc = vis_look(st, rd, q, p, g, used, okr)\n', "V6 later-look gap and closing")
    E('(est_fed(c_est, est_ok) if form == "RRe" else np.zeros(B, np.float64)))\n',
      '(est_fed(c_est, est_ok) if form == "RRe" else (np.zeros(B, np.float64) if vis_hc is None else vis_hc.astype(np.float64))))\n', "V7 later-look closing")
    E('        plan = X.MixPlanner.__call__(self, batch)\n',
      '        if STATE.get("vis") is not None:\n            STATE["vis_states"] = self.states               # the variant reads the logged tokens of the earlier frames of these windows\n        plan = X.MixPlanner.__call__(self, batch)\n', "V8 states for the variant")
    E('        W.head_values = head_values_reader\n        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head", "RRe": "head"}.get(f, "zero"))\n',
      '        STATE["vis"] = load_vis(mine.vis_gap_file, mine.vis_closing) if (mine.vis_gap_file or mine.vis_closing) else None\n'
      '        W.head_values = head_values_reader\n        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head", "RRe": "head"}.get(f, "zero"))\n'
      '        if mine.vis_closing:                               # form RR only (refused above): the closing input is the one the variant computes, read by the rule from the head slot\n'
      '            mix = (mix[0], mix[1], "head")\n', "V9 load and mix closing")
    E('"rule_join_sha256": W.sha256_file(mine.rule_join) if mine.rule_join and os.path.isfile(mine.rule_join) else "", "reader": mine.reader,',
      '"rule_join_sha256": W.sha256_file(mine.rule_join) if mine.rule_join and os.path.isfile(mine.rule_join) else "", "vis_gap_file": mine.vis_gap_file,\n'
      '                    "vis_gap_file_sha256": W.sha256_file(mine.vis_gap_file) if mine.vis_gap_file and os.path.isfile(mine.vis_gap_file) else "", "vis_closing": int(bool(mine.vis_closing)),\n'
      '                    "vis_counts": json.dumps(STATE["vis_counts"], sort_keys=True), "reader": mine.reader,', "V10 run record fields")
    idx = t.index("def build_rule_label(")                 # the helper is a module-level block placed before build_rule_label
    t = t[:idx] + VIS_HELPER + t[idx:]
    ast.parse(t)
    return t, anchors, plines


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--parent", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.environ.get("VW_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % a.out)
    raw = open(a.parent, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha[:16] != PARENT_SHA16:
        sys.exit("REFUSED: parent sha256 %s is not %s" % (sha[:16], PARENT_SHA16))
    ptext = raw.decode("utf-8")
    t, anchors, plines = build(ptext)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    out_sha = hashlib.sha256(t.encode("utf-8")).hexdigest()
    print("wrote", a.out, out_sha[:16], t.count("\n"), "lines (parent %d lines, sha256 %s)" % (raw.count(b"\n"), sha[:16]))
    for name, new in anchors:
        key = new if len(new) < 400 else new[:200]
        print("   %-44s parent L%-5d new L%d" % (name, plines[name], t[:t.index(key)].count("\n") + 1))
    d = list(difflib.unified_diff(ptext.split("\n"), t.split("\n"), lineterm="", n=0))
    add = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    dele = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    print("diff summary: %d lines added, %d lines removed (parent %d lines, new %d lines)" % (add, dele, ptext.count("\n"), t.count("\n")))
    print("sha256", out_sha)


if __name__ == "__main__":
    main()
