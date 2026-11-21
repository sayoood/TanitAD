#!/usr/bin/env python3
"""Apply the B8 (carve feed) edits to the tip scorer_finetune.py. Each anchor must match EXACTLY ONE line; the (new, old) chunk pairs are written out as the reversal table."""
import json, sys
src_path, out_path, rev_path = sys.argv[1:4]
src = open(src_path, encoding="utf-8", newline="").read()
assert "\r" not in src
lines = src.split("\n")


def find(pred, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    assert len(hits) == 1, f"{what}: {len(hits)} hits"
    return hits[0]


chunks = []         # (new_text, old_text) for the reversal table


def insert_after(pred, what, text):
    i = find(pred, what)
    old = lines[i] + "\n"
    new_lines = text.strip("\n").split("\n")
    lines[i + 1:i + 1] = new_lines
    chunks.append((old + "\n".join(new_lines) + "\n", old))


def insert_before(pred, what, text):
    i = find(pred, what)
    old = lines[i] + "\n"
    new_lines = text.rstrip("\n").split("\n")
    lines[i:i] = new_lines
    chunks.append(("\n".join(new_lines) + "\n" + old, old))


def replace_line(pred, what, newline):
    i = find(pred, what)
    old = lines[i] + "\n"
    lines[i] = newline
    chunks.append((newline + "\n", old))


# ---- E0 import
insert_after(lambda l: l == "import glob", "import glob", "import hashlib")

# ---- E1 constants
insert_after(lambda l: l.startswith("EPO_CON_MUTATE_ENVS = "), "EPO_CON_MUTATE_ENVS", '''
# B8 (raw/2026-10-18-r0e-carvefeed): the CARVE FEED of the SAFEOBJ dual.  --epo-con-feed train (the default) = the dual's accumulator is summed from the TRAINING micro-batches = every arm before B8, bit for bit.
# --epo-con-feed carve = the dual is fed from a held-out CARVE of whole RL-train logs: --epo-carve-micro no-grad micro-batches of the carve per optimiser update, and the carve's logs are REMOVED from the training index.
EPO_CON_FEEDS = ("train", "carve")
EPO_CON_FEED_DEFAULT = "train"      # = every arm before B8, bit for bit; an arm that tests the carve NAMES --epo-con-feed carve (a default that reproduces R0d must not carry the experiment)
EPO_CARVE_DESTS = ("epo_carve_logs", "epo_carve_sha256", "epo_carve_micro")          # every carve lever except the switch itself (named while the feed is train = refused)
EPO_CARVE_MICRO_DEFAULT = 1         # carve micro-batches per optimiser update (the draft prereg R0e-pick 2.3: one)
EPO_CARVE_SEED_OFFSET = 1           # the carve loader owns a torch.Generator seeded --seed + this: it never draws from the training loader's generator
''')

# ---- E2 refusals (sibling called from epo_con_refusals)
insert_before(lambda l: l.startswith("def epo_con_refusals(a, ap):"), "def epo_con_refusals", '''def epo_carve_refusals(a, ap):
    """B8: every reason the carve-feed levers must not start, as messages ([] = fine).  Called from epo_con_refusals (main(), the launch gate).  Reads through getattr so a namespace that predates B8 is the default feed."""
    feed = getattr(a, "epo_con_feed", EPO_CON_FEED_DEFAULT)
    if feed not in EPO_CON_FEEDS:
        return [f"--epo-con-feed {feed!r} is not one of {EPO_CON_FEEDS}"]
    named = [d for d in EPO_CARVE_DESTS if getattr(a, d, ap.get_default(d)) != ap.get_default(d)]
    if feed == "train":
        return [f"--epo-carve-* levers {named} are set but --epo-con-feed is train: the carve would be silently ignored"] if named else []
    bad = []
    if not (getattr(a, "epo", False) and getattr(a, "epo_con", "off") != "off"):
        bad.append("--epo-con-feed carve needs --epo and --epo-con g5: the carve feeds the dual, and with no dual it would be silently ignored")
    if not a.epo_carve_logs:
        bad.append("--epo-con-feed carve needs --epo-carve-logs (the registered carve log list)")
    if not (isinstance(a.epo_carve_sha256, str) and len(a.epo_carve_sha256) == 64 and all(c_ in "0123456789abcdef" for c_ in a.epo_carve_sha256)):
        bad.append("--epo-con-feed carve needs --epo-carve-sha256 (64 lowercase hex: the sha256 of the carve list's bytes, frozen before any data)")
    if not (isinstance(a.epo_carve_micro, int) and a.epo_carve_micro >= 1):
        bad.append("--epo-carve-micro must be an integer >= 1")
    if getattr(a, "aux_drv", False):
        bad.append("--epo-con-feed carve is not wired for --aux-drv (the carve forward would write the head's capture)")
    return bad


''')
insert_after(lambda l: l.startswith("    con_named = [d for d in EPO_CON_DESTS"), "con_named",
             "    bad += epo_carve_refusals(a, ap)                  # B8: the carve feed's own levers (all conditions are inside)")

# ---- E3 parser
insert_before(lambda l: l.startswith('    ap.add_argument("--dump-scores"'), "--dump-scores", '''    # B8 (raw/2026-10-18-r0e-carvefeed): the dual's FEED.  The switch's default keeps the training batches as the feed (every arm before B8); the three carve levers enable nothing by themselves.
    ap.add_argument("--epo-con-feed", default=EPO_CON_FEED_DEFAULT, choices=EPO_CON_FEEDS,
                    help="B8: what the SAFEOBJ dual reads.  train = the violation summed from the training micro-batches (default; every arm before B8), carve = from a held-out carve of whole RL-train logs "
                         "(--epo-carve-logs), cut out of the training index, scored by a no-grad forward of arms A and B once per optimiser update (needs --epo --epo-con g5)")
    ap.add_argument("--epo-carve-logs", default="", help="B8: the registered carve LOG LIST (one RL-train log name per token); disjoint from --heldout-logs (REFUSED otherwise); only with --epo-con-feed carve")
    ap.add_argument("--epo-carve-sha256", default="", help="B8: the sha256 of --epo-carve-logs' bytes, frozen before any data (REFUSED on a mismatch); only with --epo-con-feed carve")
    ap.add_argument("--epo-carve-micro", type=int, default=EPO_CARVE_MICRO_DEFAULT, help="B8: carve micro-batches (--batch sets each) scored per optimiser update; only with --epo-con-feed carve")''')

# ---- E4 main: the carve split, after the dual object
insert_after(lambda l: l.startswith("    con_last: dict = {}"), "con_last", '''    # B8: the carve feed (--epo-con-feed carve; the default feed leaves tr_idx untouched and nothing below runs).  The carve = WHOLE RL-train logs named by a registered, sha256-frozen list, cut out of the
    # training index HERE, so the policy never trains on them; the dual is fed from them by carve_feed_acc (called once per optimiser update).  Every disjointness below REFUSES; none is a warning.
    carve_idx: list = []
    carve_logs: set = set()
    carve_sha, cdl = "", None
    if a.epo_con_feed == "carve":
        carve_raw = open(a.epo_carve_logs, "rb").read()
        carve_sha = hashlib.sha256(carve_raw).hexdigest()
        carve_logs = set(carve_raw.decode("utf-8").split())
        carve_pool = {ds.rows[i].get("log_name") for i in tr_idx}
        carve_bad = []
        if carve_sha != a.epo_carve_sha256:
            carve_bad.append(f"the carve list's sha256 {carve_sha[:16]}.. is not the frozen --epo-carve-sha256 {a.epo_carve_sha256[:16]}..")
        if not carve_logs:
            carve_bad.append("the carve list is empty")
        if carve_logs & ho_logs:
            carve_bad.append(f"{len(carve_logs & ho_logs)} carve log(s) are RL-val logs (--heldout-logs): the carve must be disjoint from RL-val")
        if carve_logs - carve_pool - ho_logs:
            carve_bad.append(f"{len(carve_logs - carve_pool - ho_logs)} carve log(s) are not logs of the RL-train index (a typo, or a log the filters drop)")
        if carve_bad:
            for m_ in carve_bad:
                print("REFUSED: " + m_)
            return 4
        carve_idx = [i for i in tr_idx if ds.rows[i].get("log_name") in carve_logs]
        tr_idx = [i for i in tr_idx if ds.rows[i].get("log_name") not in carve_logs]
        if len(carve_idx) < a.batch or not tr_idx:
            print(f"REFUSED: {len(carve_idx)} carve sets and {len(tr_idx)} training sets survive the carve split (the carve needs >= --batch sets, training needs some)"); return 4
        cdl = DataLoader(make_train_set(ds, carve_idx, a.epo), batch_size=a.batch, shuffle=True, generator=torch.Generator().manual_seed(a.seed + EPO_CARVE_SEED_OFFSET),
                         num_workers=min(a.workers, 1), drop_last=True, pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=(2 if a.workers > 0 else None))
    carve_train_in = sum(1 for i in tr_idx if ds.rows[i].get("log_name") in carve_logs)           # both must be 0 (re-read by G7 from the BUILT indices)
    carve_in_ho = sum(1 for i in carve_idx if ds.rows[i].get("log_name") in ho_logs)''')

# ---- E5 data event
insert_after(lambda l: l.startswith('        info["epo_con"] = dual.built_facts()'), "info epo_con", '''        if cdl is not None:                                  # B8: BUILT facts of the carve feed for G-DVB (absent for the default feed: every pre-B8 data event is byte-identical)
            info["epo_con_carve"] = {"feed": "carve", "logs": len(carve_logs), "sets": len(carve_idx), "sha256": carve_sha, "micro": a.epo_carve_micro,
                                     "train_rows_in_carve_logs": carve_train_in, "carve_rows_in_heldout_logs": carve_in_ho}''')

# ---- E6 the carve stream + the feed closure
insert_before(lambda l: l.startswith("    def aux_backward(lv, div=1.0, retain=False):"), "def aux_backward", '''    def carve_stream():
        while True:
            for b_ in cdl:
                yield b_
    carve_it = carve_stream() if cdl is not None else None

    def carve_feed_acc():
        """B8: the dual's accumulator from the CARVE: --epo-carve-micro micro-batches of whole carve logs, each scored by a no-grad forward of arms A and B (the loop's own fwd) and passed through the
        loop's own epo_con_term, so g is the SAME quantity (same T, eps, rule) as the train feed's.  -> {"n": micro-batches, "g_<c>": the SUM of the detached violation, + cA_ / cB_ / hinge_ / p1gap_}, the
        shape EpoDual.step reads.  NO GRAPH: the carve reaches the scorer only through the VALUE of the duals (Python floats) that train_losses multiplies into the training micro-batches' constraint term."""
        acc = {"n": 0}
        with torch.no_grad():
            for _ in range(a.epo_carve_micro):
                cb_, crow_ = split_rows(next(carve_it), a.epo)
                stray = [int(i_) for i_ in crow_ if ds.rows[int(i_)].get("log_name") not in carve_logs]
                if stray:
                    raise RuntimeError(f"carve feed: {len(stray)} row(s) of a carve micro-batch are not in the carve logs (first row index {stray[0]})")
                _c0, cA_, cB_, ct_ = fwd(cb_)
                cl_: dict = {}
                epo_con_term(cB_, cA_, ct_, sel_score, dual.temp, a.epo_con_eps, dual.lam, cl_)
                for k_, v_ in cl_.items():
                    acc[k_] = acc.get(k_, 0.0) + v_
                acc["n"] += 1
        return acc

''')

# ---- E7 preflight: G7 extension
i7 = find(lambda l: l.strip().startswith('"heldout_sets": len(ho_idx), "train_rows_in_heldout_logs": info["train_rows_in_heldout_logs"]}'), "G7 tail")
assert lines[i7 - 1].lstrip().startswith('gates["G7_heldout_disjoint"] = {"ok"')
old7 = lines[i7] + "\n"
ins7 = '''        if cdl is not None:                                  # B8: the carve's disjointness re-read from the BUILT indices + one REAL no-grad carve forward through the dual's own feed
            cp_ = carve_feed_acc()
            gates["G7_heldout_disjoint"].update({"carve_logs": len(carve_logs), "carve_sets": len(carve_idx), "carve_sha256": carve_sha, "train_rows_in_carve_logs": carve_train_in,
                                                 "carve_rows_in_heldout_logs": carve_in_ho, "carve_probe_n": cp_["n"], "carve_probe_g": {c_: cp_["g_" + c_] for c_ in dual.comps}})
            gates["G7_heldout_disjoint"]["ok"] = bool(gates["G7_heldout_disjoint"]["ok"] and len(carve_idx) >= a.batch and carve_train_in == 0 and carve_in_ho == 0
                                                      and cp_["n"] == a.epo_carve_micro and all(math.isfinite(cp_["g_" + c_]) for c_ in dual.comps))'''.split("\n")
lines[i7 + 1:i7 + 1] = ins7
chunks.append((old7 + "\n".join(ins7) + "\n", old7))

# ---- E8 loop
insert_after(lambda l: l.startswith('    con_log = {"n": 0}') and "since the last logged row" in l, "con_log init",
             '    carve_log = {"n": 0}                                      # B8: the carve feed\'s parts over the updates since the last logged row (--epo-con-feed carve only)')
i8 = find(lambda l: l == "            if micro % a.accum:", "micro % accum")
assert lines[i8 + 1].strip() == "continue"
old8 = lines[i8 + 1] + "\n"
ins8 = ["            carve_acc = carve_feed_acc() if carve_it is not None else None       # B8: the dual's carve accumulator, scored on the SAME weights as this update's training micro-batches (before o.step())"]
lines[i8 + 2:i8 + 2] = ins8
chunks.append((old8 + "\n".join(ins8) + "\n", old8))
replace_line(lambda l: l.strip() == "dual.step(con_acc)", "dual.step(con_acc)",
             "                dual.step(con_acc if carve_acc is None else carve_acc)          # B8: the carve feed when --epo-con-feed carve, else the training micro-batches (as before)")
insert_after(lambda l: l.strip() == 'con_acc = {"n": 0}' and l.startswith("                "), "con_acc reset", '''                if carve_acc is not None:
                    for k_, v_ in carve_acc.items():
                        carve_log[k_] = carve_log.get(k_, 0.0) + v_''')
insert_after(lambda l: l.strip() == 'con_log = {"n": 0}' and l.startswith("                    "), "con_log reset", '''                    if carve_it is not None:                         # B8: the carve feed's window means, beside the in-batch ones above
                        ncv_ = max(carve_log["n"], 1)
                        row.update({"carve_n": carve_log["n"], **{"carve_" + k_: carve_log.get(k_, 0.0) / ncv_ for k_ in ("g_ttc", "g_nc", "cA_ttc", "cB_ttc", "cA_nc", "cB_nc", "p1gap_ttc", "p1gap_nc", "hinge_ttc", "hinge_nc")}})
                        carve_log = {"n": 0}''')

open(out_path, "w", encoding="utf-8", newline="").write("\n".join(lines))
json.dump(chunks, open(rev_path, "w"))
print("edits:", len(chunks), "lines", len(src.split("\n")), "->", len(lines))
