"""apply the P6-HN-S item-4 edits to refe/gate_checks.py in the edit tree (idempotence guarded: every anchor must occur exactly once)"""
import sys
from pathlib import Path
PK = Path("C:/Users/Admin/hn34/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
P = PK / "refe" / "gate_checks.py"
s = P.read_bytes().decode("utf-8")
assert "\r" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- 1. the registry
rep('''    "aux_drv_lr": "built", "aux_drv_labels": "data",
    "out": "run", "eval_every": "run", "workers": "run",''',
    '''    "aux_drv_lr": "built", "aux_drv_labels": "data",
    # P6-HN-S (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md 4.4): both default OFF.  `data` = an input the PASS token must fingerprint (launch_gate.RLS1_DATA lists both flags); the trainer's data event
    # also carries a BUILT block for each (`hn`, `ctx_cache`: SCORER_HN_KEYS / SCORER_CTX_KEYS), which scorer_dvb_rows compares with the argv and with the bytes the flag names.
    "hn_bank": "data", "ctx_cache": "data",
    "out": "run", "eval_every": "run", "workers": "run",''')

# ---- 2. constants + gates table
rep('''SCORER_NAMESPACE_FUNCS = ("main", "epo_refusals", "_g_probe", "aux_s1_refusals")''',
    '''SCORER_HN_KEYS = ("sha256", "rows", "scenes_with_hn", "max_per_sample", "arm", "comps", "weight", "mask")           # P6-HN-S: the keys of the data event's `hn` block (written iff --hn-bank)
SCORER_CTX_KEYS = ("manifest_sha256", "trunk_fingerprint", "n_scenes", "dtype")                                   # ... and of its `ctx_cache` block (written iff --ctx-cache)
SCORER_HN_REGISTERED = {"max_per_sample": 8, "arm": "B", "comps": 6, "weight": "1/(B*64*6) per element", "mask": "hn->fan+self"}          # PREREG_P6HN_SCREEN 4.4: the gate's OWN statement of the built `hn` block
SCORER_CTX_REGISTERED = {"dtype": "bf16"}
G15_MIN_SCENES = 64                                    # PREREG 4.4: G15 compares >= 64 RL-train and >= 64 RL-val scenes (a run with fewer scenes than that side holds compares all of them: see judge_g15)
HN_ARMS = ("lever_ignored", "hn_to_armA", "hn_in_selection", "hn_lane_target", "unmasked_pad", "fan_sees_hn", "stale_bank", "empty_bank_in_launch", "moved_lever", "stale_cache", "shuffled_cache", "epo_with_hn")
SCORER_NAMESPACE_FUNCS = ("main", "epo_refusals", "_g_probe", "aux_s1_refusals")''')
rep('''"G10": "teacher_lane_in_bank", "G11": "epo_live_and_identity", "G12": "aux_drv_live_and_identity", "G13": "epo_con_live_and_identity"}''',
    '''"G10": "teacher_lane_in_bank", "G11": "epo_live_and_identity", "G12": "aux_drv_live_and_identity", "G13": "epo_con_live_and_identity",
                "G14": "hn_live_and_identity", "G15": "ctx_cache_identity"}''')

# ---- 3. judge_g14 / judge_g15 + the evidence arms (inserted before ScorerArgvError)
JUDGES = '''def judge_g14(g14, ns) -> list:
    """G14 (scorer_finetune --preflight, --hn-bank) re-judged from its RAW FIELDS against the argv's namespace -- never from its own `ok`.  [] = proved.  (PREREG_P6HN_SCREEN 4.3 / 4.4)
      table        the hard-negative term ALONE (the run's own assembly minus the stock one on one fresh forward) is finite and non-zero on arm B's score_q_mlp, score_dec and score_head rows 0-5, and
                   EXACTLY 0.0 on the lane row (the control that proves the probe can read zero: it needs a lane row to mean anything), on every arm-A parameter and on the base; the trunk has no gradient
      controls     the bank OFF and the bank EMPTY: the term and every gradient read EXACTLY 0.0; arm A's own BCE gradient in the same probe reads non-zero
      two steps    two optimiser steps with the bank off / empty leave B bit-identical to A; with it on B differs; the probe is shown able to read one ulp
      identities   amendment B2: a bank row that duplicates fan row j adds exactly row j's share of the fan BCE (self-attention neutralised); junk in the pad slots changes nothing (bit for bit); the served
                   mask equals the bank's own
      fan          B's fan logits as the loss uses them equal the one-call path bit for bit; the masked call's fan rows do not see any hn slot (bit for bit); an hn row does not depend on the other hn rows
      D1 / I4      the rows that entered the term equal the rows the bank holds for the served scenes; the held-out items carry no bank"""
    if not isinstance(g14, dict):
        return ["G14 is not a dict"]
    r = []
    mods = set(SCORER_GROUPS)
    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)          # noqa: E731
    if g14.get("ok") is not True:
        r.append("G14 reports ok != true")
    if g14.get("mutate"):
        r.append(f"G14 was produced under regression switch(es) {g14['mutate']}: a regression arm can never count")
    gs = g14.get("grad_abs_sum") or {}
    if not (isinstance(g14.get("n_hn"), int) and not isinstance(g14.get("n_hn"), bool) and g14["n_hn"] > 0):
        r.append(f"G14: the probe batch held {g14.get('n_hn')!r} valid hard-negative rows: the term was not exercised")
    if g14.get("grads_finite") is not True:
        r.append("G14: a gradient of the hard-negative term is not finite")
    if not (fin(g14.get("value_on")) and g14["value_on"] > 0.0):
        r.append(f"G14: the term alone reads {g14.get('value_on')!r} at the declared setting: it is not part of arm B's loss")
    for k in ("score_q_mlp", "score_dec", "score_head_rows_0_5"):
        if not (fin(gs.get(k)) and gs[k] > 0.0):
            r.append(f"G14: the term's gradient on {k} is {gs.get(k)!r}, not finite and non-zero")
    if g14.get("lane_head") is not bool(ns.lane_head):
        r.append(f"G14 probed lane_head {g14.get('lane_head')!r}, the argv has {bool(ns.lane_head)!r}: the lane-row control is not the declared model's")
    if gs.get("score_head_row_6_lane") != 0.0:
        r.append(f"G14: the hn rows trained the LANE output (score_head row 6 reads {gs.get('score_head_row_6_lane')!r}, it must be EXACTLY 0.0)")
    if gs.get("armA_all") != 0.0:
        r.append(f"G14: arm A received a gradient from the hard-negative term ({gs.get('armA_all')!r}); it must read EXACTLY 0.0")
    if gs.get("base_params_with_grad") != 0 or gs.get("frozen_params_with_grad") != 0 or g14.get("trunk_requires_grad") is not False:
        r.append(f"G14: the base ({gs.get('base_params_with_grad')!r} params) or the frozen trunk ({gs.get('frozen_params_with_grad')!r} params, requires_grad {g14.get('trunk_requires_grad')!r}) is on the gradient path")
    a_ctrl = g14.get("A_bce_control") or {}
    if set(a_ctrl) != mods or not all(fin(x) and x > 0.0 for x in a_ctrl.values()):
        r.append(f"G14: arm A's own BCE gradient does not read non-zero in the same probe (a zero on A would be a property of the probe): {a_ctrl}")
    ctl = g14.get("controls") or {}
    if set(ctl) != {"off", "empty"} or any(not (isinstance(v, dict) and v.get("value") == 0.0 and v.get("B_all") == 0.0 and v.get("A_all") == 0.0) for v in ctl.values()):
        r.append(f"G14: the bank-off / bank-empty controls do not read EXACTLY 0.0 (value and every gradient): {ctl}")
    ts = g14.get("two_step") or {}
    for k, want in (("off_equal_A", True), ("empty_equal_A", True), ("on_differs_from_A", True), ("probe_reads_one_ulp", True)):
        if ts.get(k) is not want:
            r.append(f"G14: two_step.{k} is {ts.get(k)!r}, not {want!r} (B == A with the bank off / empty, B != A with it on, and the equality probe reads one ulp)")
    dup = g14.get("dup_identity") or {}
    if not (fin(dup.get("share")) and dup["share"] > 0.0 and fin(dup.get("abs_diff")) and dup["abs_diff"] <= 1e-5 * max(1.0, dup["share"])):
        r.append(f"G14: the analytic identity (a duplicated fan row adds exactly its share of the fan BCE) does not hold: {dup}")
    pad = g14.get("pad") or {}
    if pad.get("served_mask_equals_bank") is not True or not (isinstance(pad.get("pad_slots"), int) and pad["pad_slots"] > 0):
        r.append(f"G14: the served mask is not the bank's own (or the batch had no pad slot to test): {pad}")
    if pad.get("perturbed_loss_equal") is not True:
        r.append("G14: junk written into the pad slots moved the term (padding can reach the loss)")
    if g14.get("fan_call_max_abs_diff") != 0.0:
        r.append(f"G14: B's fan logits as the loss uses them differ from the one-call path ({g14.get('fan_call_max_abs_diff')!r}); they must be bit-identical")
    if g14.get("fan_rows_hn_isolation_max_abs") != 0.0:
        r.append(f"G14: the fan rows of the masked call depend on the hn slots ({g14.get('fan_rows_hn_isolation_max_abs')!r}): the fan can see a hard negative (no mask)")
    if g14.get("hn_row_independence_max_abs") != 0.0:
        r.append(f"G14: a hard-negative row's logits depend on the OTHER hard-negative rows ({g14.get('hn_row_independence_max_abs')!r}); it must be exactly 0.0 (None = the probe batch had no scene with two rows)")
    d1 = g14.get("d1") or {}
    if not (isinstance(d1.get("rows_in_term"), int) and d1.get("rows_in_term") == d1.get("rows_expected") and d1["rows_expected"] > 0):
        r.append(f"G14 (D1): the rows that entered the term ({d1.get('rows_in_term')!r}) are not the rows the bank holds for the served scenes ({d1.get('rows_expected')!r})")
    er = g14.get("eval_reads_bank") or {}
    if not (isinstance(er.get("held_items_checked"), int) and er["held_items_checked"] > 0 and er.get("held_items_with_hn_tail") == 0):
        r.append(f"G14 (I4): held-out items carry a hard-negative tail (evaluation can read the bank): {er}")
    return r


def judge_g15(g15, ns, min_scenes: int = G15_MIN_SCENES) -> list:
    """G15 (scorer_finetune --preflight, --ctx-cache) re-judged from its RAW FIELDS; [] = proved.  On >= `min_scenes` RL-train and >= `min_scenes` RL-val scenes (a side that holds fewer compares all of them) the cached
    context and the base / A / B logits read from the cache equal the LIVE trunk forward's bit for bit, in micro-batches of the argv's --batch; the SAME comparison with the cache's rows rolled across the batch must
    read > 10x the identity's (so the probe can see a wrong cache); the manifest binds the checkpoint's trunk fingerprint and the calib file; the cache covers every scene the run uses; the dtype is bf16."""
    if not isinstance(g15, dict):
        return ["G15 is not a dict"]
    r = []
    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)          # noqa: E731
    if g15.get("ok") is not True:
        r.append("G15 reports ok != true")
    if g15.get("mutate"):
        r.append(f"G15 was produced under regression switch(es) {g15['mutate']}: a regression arm can never count")
    nt, nv = g15.get("n_train"), g15.get("n_val")
    if not (isinstance(nt, int) and isinstance(nv, int) and nt >= min_scenes and nv >= min_scenes):
        r.append(f"G15 compared {nt!r} RL-train and {nv!r} RL-val scenes (PREREG 4.4 needs >= {min_scenes} of each)")
    if g15.get("micro_batch") != ns.batch:
        r.append(f"G15 compared in micro-batches of {g15.get('micro_batch')!r}, the argv's --batch is {ns.batch!r} (a batch-dependent kernel must show here, not in the run)")
    ident = g15.get("logit_max_abs_diff") or {}
    if g15.get("ctx_max_abs_diff") != 0.0:
        r.append(f"G15: the cached context differs from the live trunk's visual_ctx (max |diff| {g15.get('ctx_max_abs_diff')!r}); it must be bit-identical")
    if set(ident) != {"base", "A", "B"} or any(v != 0.0 for v in ident.values()):
        r.append(f"G15: the logits read from the cache differ from the live trunk forward's: {ident} (base / A / B must each read EXACTLY 0.0)")
    sh = g15.get("shuffled") or {}
    sl = sh.get("logit_max_abs_diff") or {}
    if not (fin(sh.get("ctx_max_abs_diff")) and sh["ctx_max_abs_diff"] > 0.0 and set(sl) == {"base", "A", "B"}
            and all(fin(sl[k]) and sl[k] > 0.0 and sl[k] > 10.0 * float(ident.get(k) if fin(ident.get(k)) else 0.0) for k in sl)):
        r.append(f"G15: a row-shuffled cache does not read > 10x the identity's difference (the probe cannot see a wrong cache): {sh}")
    fm, fc = g15.get("trunk_fingerprint_manifest"), g15.get("trunk_fingerprint_checkpoint")
    if not (isinstance(fm, str) and len(fm) == 64 and fm == fc):
        r.append(f"G15: the cache manifest's trunk fingerprint {str(fm)[:16]!r} is not the checkpoint's {str(fc)[:16]!r}")
    if g15.get("covers_run") != {"missing_train": 0, "missing_val": 0}:
        r.append(f"G15: the cache does not cover every scene of the run: {g15.get('covers_run')}")
    if g15.get("dtype") != SCORER_CTX_REGISTERED["dtype"]:
        r.append(f"G15: the cache dtype is {g15.get('dtype')!r}, not {SCORER_CTX_REGISTERED['dtype']!r}")
    cm, cr = g15.get("calib_sha256_manifest"), g15.get("calib_sha256_run")
    if not (isinstance(cr, str) and len(cr) == 64 and cm == cr):
        r.append(f"G15: the cache manifest's calib sha256 {str(cm)[:16]!r} is not the run's calib file's {str(cr)[:16]!r}")
    if not (isinstance(g15.get("manifest_sha256"), str) and len(g15["manifest_sha256"]) == 64):
        r.append("G15: the manifest's sha256 is absent")
    return r


def apply_hn_arm(g14, g15, arm):
    """The deliberate-regression arms of the P6-HN-S launch gate that act on the PREFLIGHT evidence (PREREG_P6HN_SCREEN 4.4): the same preflight, with the raw fields of the defect the arm names.  Each must
    turn judge_g14 / judge_g15 RED (test_launch_gate.py).  Returns (g14, g15) copies; an arm that acts elsewhere (G-DVB: lever_ignored, stale_bank, empty_bank_in_launch; the argv: moved_lever, epo_with_hn)
    returns them unchanged."""
    import copy
    g14, g15 = copy.deepcopy(g14), copy.deepcopy(g15)
    if isinstance(g14, dict):
        gs = g14.setdefault("grad_abs_sum", {})
        if arm_is("hn_to_armA", arm):                                  # the term was applied to arm A's scorer: A receives its gradient
            gs["armA_all"] = 0.37
        if arm_is("hn_in_selection", arm):                             # hard-negative rows reach an evaluation set (the held-out items carry the bank's tail)
            g14["eval_reads_bank"] = {"held_items_checked": 3, "held_items_with_hn_tail": 3}
        if arm_is("hn_lane_target", arm):                              # the hn rows train the lane output
            gs["score_head_row_6_lane"] = 0.021
        if arm_is("unmasked_pad", arm):                                # pad slots are served as valid rows: the term sees more rows than the bank holds, and junk in the pad slots moves it
            d1 = g14.setdefault("d1", {})
            d1["rows_in_term"] = int(d1.get("rows_expected", 0)) + 12
            g14.setdefault("pad", {})["perturbed_loss_equal"] = False
            g14["pad"]["served_mask_equals_bank"] = False
        if arm_is("fan_sees_hn", arm):                                 # no attention mask: the fan rows read the hard negatives
            g14["fan_rows_hn_isolation_max_abs"] = 0.31
            g14["fan_call_max_abs_diff"] = 0.12
    if isinstance(g15, dict):
        if arm_is("stale_cache", arm):                                 # a cache of another trunk
            g15["trunk_fingerprint_manifest"] = "f" * 64
        if arm_is("shuffled_cache", arm):                              # a cache whose rows are shuffled across scenes: it differs from the live forward
            g15["ctx_max_abs_diff"] = 0.953125
            g15["logit_max_abs_diff"] = {k: 0.0923 for k in ("base", "A", "B")}
    return g14, g15


'''
rep('''class ScorerArgvError(Exception):''', JUDGES + '''class ScorerArgvError(Exception):''')

# ---- 4. the required gates
rep('''    if ns.aux_drv:
        req.append("G12")
    return req''', '''    if ns.aux_drv:
        req.append("G12")
    if getattr(ns, "hn_bank", ""):                                    # P6-HN-S: G14 iff --hn-bank, G15 iff --ctx-cache (scorer_finetune.py's own conditions)
        req.append("G14")
    if getattr(ns, "ctx_cache", ""):
        req.append("G15")
    return req''')
rep('''G10 iff --lane-head, G11 iff --epo, G12 iff --aux-drv, G13 iff --epo AND --epo-con != off).""" ''' if False else '''G11 iff --epo, G12 iff --aux-drv, G13 iff --epo AND --epo-con != off)."""''',
    '''G11 iff --epo, G12 iff --aux-drv, G13 iff --epo AND --epo-con != off, G14 iff --hn-bank, G15 iff --ctx-cache)."""''')

# ---- 5. the declared-vs-built rows
rep('''    return rows


def scorer_schedule_rows(ns, data: dict, arm=None) -> list:''', '''    hn_named, cc_named = bool(getattr(ns, "hn_bank", "")), bool(getattr(ns, "ctx_cache", ""))
    if hn_named or "hn" in data:                                   # P6-HN-S: the bank, against the BUILT block of the bank the trainer holds (never an echo of the namespace)
        blk = data.get("hn") if isinstance(data.get("hn"), dict) else {}
        add("hn_bank", hn_named, "hn" in data, "the data event carries an `hn` block iff --hn-bank")
        if hn_named:
            add("hn_bank", sorted(SCORER_HN_KEYS), sorted(blk), "data event: the `hn` block's keys")
            add("hn_bank", _file_sha256(ns.hn_bank) or "<the file --hn-bank names is not readable>", blk.get("sha256"), "data event: hn.sha256 (the sha256 of the bytes the trainer LOADED) vs the sha256 of the file the argv names")
            add("hn_bank", True, bool(isinstance(blk.get("rows"), int) and blk["rows"] > 0 and isinstance(blk.get("scenes_with_hn"), int) and blk["scenes_with_hn"] > 0),
                "data event: hn.rows > 0 and hn.scenes_with_hn > 0 (a launch whose lever has nothing to act on is not an experiment)")
            add("hn_bank", True, bool(isinstance(blk.get("rows"), int) and isinstance(blk.get("scenes_with_hn"), int) and isinstance(data.get("train_sets"), int)
                                      and blk["scenes_with_hn"] <= data["train_sets"] and blk["rows"] <= SCORER_HN_REGISTERED["max_per_sample"] * data["train_sets"]),
                "data event: the bank holds no more scenes / rows than the run trains on")
            for k, want in SCORER_HN_REGISTERED.items():
                add("hn_bank", want, blk.get(k), f"data event: hn.{k} (registered {want!r})")
    if cc_named or "ctx_cache" in data:                           # P6-HN-S: the cache, against the BUILT block
        blk = data.get("ctx_cache") if isinstance(data.get("ctx_cache"), dict) else {}
        add("ctx_cache", cc_named, "ctx_cache" in data, "the data event carries a `ctx_cache` block iff --ctx-cache")
        if cc_named:
            add("ctx_cache", sorted(SCORER_CTX_KEYS), sorted(blk), "data event: the `ctx_cache` block's keys")
            add("ctx_cache", _file_sha256(os.path.join(ns.ctx_cache, "manifest.json")) or "<the manifest in --ctx-cache is not readable>", blk.get("manifest_sha256"),
                "data event: ctx_cache.manifest_sha256 vs the sha256 of the manifest the argv's cache holds")
            tf = blk.get("trunk_fingerprint")
            add("ctx_cache", True, bool(isinstance(tf, str) and len(tf) == 64 and all(c in "0123456789abcdef" for c in tf)), "data event: ctx_cache.trunk_fingerprint is the checkpoint's sha256 fingerprint")
            need = (data.get("train_sets") or 0) + (data.get("heldout_sets") or 0)
            add("ctx_cache", True, bool(isinstance(blk.get("n_scenes"), int) and need > 0 and blk["n_scenes"] >= need), "data event: the cache holds at least every train + held-out scene the run uses")
            add("ctx_cache", SCORER_CTX_REGISTERED["dtype"], blk.get("dtype"), "data event: ctx_cache.dtype")
    return rows


def _file_sha256(path):
    """sha256 of a file's bytes, or None when the path is not a readable file (then the declared side is None and the row cannot be equal to a built hash)"""
    try:
        if not path or not os.path.isfile(path):
            return None
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def scorer_schedule_rows(ns, data: dict, arm=None) -> list:''')

# ---- 6. the G-DVB arms that act on the data event
rep('''        if arm_is("aux_lever_ignored", arm) and isinstance(data.get("aux_drv"), dict):''', '''        if arm_is("lever_ignored", arm) and getattr(ns, "hn_bank", ""):                          # P6-HN-S: the flag was parsed and NEVER added -- the trainer built no `hn` block
            data = {k: v for k, v in data.items() if k != "hn"}
        if arm_is("stale_bank", arm) and isinstance(data.get("hn"), dict):                       # P6-HN-S: the bank's bytes changed after minting -- the trainer holds a bank of another hash
            _h = str(data["hn"].get("sha256", ""))
            data = dict(data, hn=dict(data["hn"], sha256=("0" if _h[:1] != "0" else "1") + _h[1:]))
        if arm_is("empty_bank_in_launch", arm) and isinstance(data.get("hn"), dict):             # P6-HN-S: a bank of 0 hard negatives
            data = dict(data, hn=dict(data["hn"], rows=0, scenes_with_hn=0))
        if arm_is("aux_lever_ignored", arm) and isinstance(data.get("aux_drv"), dict):''')
rep('''        if arm_is("lever_ignored", arm):                                     # the trainer ignored --b-mode: its data event says the other mode''', '''        if arm_is("lever_ignored", arm) and not getattr(ns, "hn_bank", ""):                  # the trainer ignored --b-mode: its data event says the other mode (a P6-HN-S argv gets the `hn` form below instead)''')
P.write_bytes(s.encode("utf-8"))
print("ok", len(s))
