#!/usr/bin/env python3
"""Derive box2_ship/v8f_oaf_chain_checks.py from the DAG2 build's v8f_dag2_chain_checks.py (read-only reference: E:/Projects/TanitAD-artifacts/dag2q/box2_ship/) with EXACT-COUNT edits, then append the new decisions.
The bank judge, the bank-file / bank-verify / bank-plan code and the argv helpers stay byte-identical to the DAG2 file except for the edits listed here (the selftest re-compares them function by function).

  removed : dag_predicate + DAG_READ_SHA256 (neither new PREREG is licensed by a pinned read; the PI's go is the licence) and the `dag-predicate` subcommand
  renamed : ZZDAG2-JUDGE -> ZZOAF-JUDGE, ZZDAG2-BANKFILES -> ZZOAF-BANKFILES; K-B2's control name no longer says DAG (it reads the GENERATOR's checkpoint, which is FLAT for the FLAT-DAG banks)
  added   : wacc-check (C-O3), init-match (the F1 lever-in-tip check, by ast), disk-check (the 1 GiB floor on E:), pred-verdict (the predecessor rule), pin-sha (sha256 of a file)
"""
from pathlib import Path

SRC = Path("E:/Projects/TanitAD-artifacts/dag2q/box2_ship/v8f_dag2_chain_checks.py")
OUT = Path(__file__).resolve().parent / "box2_ship" / "v8f_oaf_chain_checks.py"
NL = chr(10)


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:90], t.count(old))
    return t.replace(old, new)


t = SRC.read_text(encoding="utf-8")
assert "\r" not in t
# ---- the docstring
a = t.index('"""'); b = t.index('"""', a + 3) + 3
doc = '''"""PREREG_V8F_DAG_OACC and PREREG_V8F_FLATDAG on box 2: the DECISIONS the two chains make on files, in one place so a test can run both sides of each. DERIVED from the DAG2 build's v8f_dag2_chain_checks.py by make_helpers.py
(the bank judge and the bank-file code are the DAG2 ones; only the names that said DAG say generator now, and the DAG-read predicate is gone).

    argv-sha LINEFILE / argv-same GATE TRAIN        as DAG2 (launch_gate.argv_sha256 of the training argv; the gate line == the train line with --out and --selfcheck-only changed)
    bank-judge --banked-sha0 S --banked-sha1 S --seed0-dir D0 --seed1-dir D1      PREREG_V8F_DAGGER sections 6-7 on two PULLED banks (K-B0..K-B5, COVERAGE >= 0.5 inclusive); exit 0 PASS | 3 BLOCKED | 2 REFUSED. K-B2 reads the
                                                                                   GENERATOR's checkpoint: for the FLAT bank that is FLAT_s<seed>
    bank-files DIR [--kind ...] / bank-verify DIR [--require-sums] / bank-plan   the published bank files
    wacc-check --box w_acc.json --offline wacc_offline_DAG_s0.json [--offline-sha SHA] [--txt w_acc.txt] [--expect-ckpt-sha SHA] [--rtol 1e-4]
        C-O3: the box-2 value of the w_acc rule equals the offline recomputation on DAG_s0's banked fit plans to within 1e-4 relative (w_acc, L_traj, L_acc, L_ade, L_fde; n_windows exactly), the box record names the rule's
        constants (frac 0.5, w_traj 1.0), its model-route cross-check against the banked fit windows PASSED on the same windows, its checkpoint is DAG_s0's, and w_acc.txt is the bare number. exit 0 = equal | 2 = REFUSED.
    init-match TRAINER.py       the F1 lever-in-tip check: INIT_CKPT_MATCH (read by ast, whatever the formatting) names o_interface. exit 0 | 2
    disk-check --dir D --size N [--floor B] [--free-bytes F]      the disk floor: free(D) - N >= B (default 1 GiB). exit 0 | 2. Prints `free=... size=... after=... floor=...`
    pred-verdict --out LOG --busy FILE           the predecessor rule: prints GO | WAIT | BUSY | HALT and a reason (GO: ZZV8FMINI-DONE with BUSY released, or a ZZV8FMINI-STOP with no ZZGUARD-STOP line and BUSY released; BUSY: a stop whose
                                                 BOX2_BUSY.txt is still there (the caller retries a few times, then HALTs); HALT: a guard stop, or a DONE with BUSY still held; WAIT: no end marker yet)
    pin-sha FILE                 sha256 of a file
"""'''
t = t[:a] + doc + t[b:]
# ---- remove the DAG-read predicate
t = rep(t, 'DAG_READ_SHA256 = "3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d"          # PREREG_V8F_DAGGER2: the read that licenses it (E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json)' + NL, "")
i, j = t.index("def dag_predicate("), t.index("def argv_of(text: str)")
t = t[:i] + t[j:]
t = rep(t, '    o = sub.add_parser("dag-predicate")' + NL + '    o.add_argument("dag_read")' + NL + '    o.add_argument("--expect-sha", default=DAG_READ_SHA256)' + NL, "")
t = rep(t, '    if a.mode == "dag-predicate":' + NL + '        rc, msg = dag_predicate(a.dag_read, a.expect_sha)' + NL + '        print(msg)' + NL + '        return rc' + NL, "")
# ---- renames
t = rep(t, "ZZDAG2-JUDGE", "ZZOAF-JUDGE", t.count("ZZDAG2-JUDGE"))
t = rep(t, "ZZDAG2-BANKFILES", "ZZOAF-BANKFILES", t.count("ZZDAG2-BANKFILES"))
t = rep(t, "K-B2 on-policy: the bank's checkpoint is the banked DAG run's (M2b_s<seed>_dag)", "K-B2 on-policy: the bank's checkpoint is the banked GENERATOR's (DAG_s<seed> / FLAT_s<seed>)")
t = rep(t, "TRAIN list is M2b's (clips, windows; DAG's data flags are M2b's)", "TRAIN list is M2b's (clips, windows; the generator's data flags are M2b's)")
t = rep(t, "import argparse" + NL, "import argparse" + NL + "import ast" + NL + "import shutil" + NL)
# ---- the new decisions, before main()
new = r'''
# ---------------------------------------------------------------------------------------------------------------- C-O3, the F1 lever, the disk floor, the predecessor rule
WACC_REL_TOL = 1e-4


def wacc_check(box_json: str, offline_json: str, offline_sha: str = "", txt: str = "", expect_ckpt_sha: str = "", rtol: float = WACC_REL_TOL) -> tuple:
    """(rc, record). C-O3 (PREREG_V8F_DAG_OACC section 3): the box-2 w_acc record against the offline recomputation on DAG_s0's banked fit plans. Anything that cannot be read or does not match is REFUSED (rc 2)."""
    bad, rec = [], {}
    try:
        if offline_sha and sha256_file(offline_json) != offline_sha.lower():
            bad.append(f"the offline record {offline_json} hashes to {sha256_file(offline_json)[:16]}, not the pinned {offline_sha[:16]}")
        b, o = _j(box_json), _j(offline_json)
    except Exception as e:                                                  # noqa: BLE001
        return 2, {"outcome": "REFUSED", "bad": [f"cannot read the records: {type(e).__name__}: {e}"]}
    for k in ("w_acc", "L_traj", "L_acc", "L_ade", "L_fde"):
        bv, ov = b.get(k), o.get(k)
        if not (_num(bv) and _num(ov) and ov != 0):
            bad.append(f"{k}: box {bv!r} / offline {ov!r} are not comparable numbers")
            continue
        rel = abs(bv - ov) / abs(ov)
        rec[k] = {"box": bv, "offline": ov, "rel": rel}
        if rel > rtol:
            bad.append(f"{k}: box {bv!r} vs offline {ov!r}: relative difference {rel:.3g} > {rtol}")
    if b.get("n_windows") != o.get("n_windows") or b.get("n_windows") != 9582:
        bad.append(f"n_windows box {b.get('n_windows')!r} / offline {o.get('n_windows')!r} (must be equal, 9,582)")
    if (b.get("frac"), b.get("w_traj")) != (0.5, 1.0) or b.get("schema") != "v8f-oacc-wacc-v1":
        bad.append(f"the box record does not carry the rule's constants: frac {b.get('frac')!r} w_traj {b.get('w_traj')!r} schema {b.get('schema')!r}")
    bc = b.get("banked_crosscheck")
    if not isinstance(bc, dict) or bc.get("pass") is not True or bc.get("same_windows") is not True:
        bad.append(f"the model route's cross-check against DAG_s0's banked fit windows did not pass on the same windows: {bc if not isinstance(bc, dict) else {k: bc.get(k) for k in ('pass', 'same_windows')}}")
    elif o.get("source", {}).get("fit_windows_sha256") and bc.get("fit_windows_sha256") != o["source"]["fit_windows_sha256"]:
        bad.append("the cross-check ran against another banked fit-windows file than the offline recomputation's")
    src = b.get("source") or {}
    if expect_ckpt_sha and src.get("ckpt_sha256") != expect_ckpt_sha.lower():
        bad.append(f"the rule was evaluated with checkpoint {str(src.get('ckpt_sha256'))[:16]}, not DAG_s0's {expect_ckpt_sha[:16]}")
    if txt:
        try:
            tv = open(txt, encoding="ascii", newline="").read()
        except OSError as e:
            tv = None
            bad.append(f"w_acc.txt unreadable: {e}")
        if tv is not None and tv != f"{b.get('w_acc')!r}\n":
            bad.append(f"w_acc.txt {tv!r} is not the bare number {b.get('w_acc')!r} + LF (what the queue reads into --w-acc)")
    rec.update({"outcome": "REFUSED" if bad else "PASS", "bad": bad, "rtol": rtol})
    return (2 if bad else 0), rec


def init_match_has(src_text: str, key: str = "o_interface") -> tuple:
    """(ok, why): the module-level assignment INIT_CKPT_MATCH is a tuple of string literals that contains ``key`` (read by ast: formatting and comments do not matter)."""
    try:
        tree = ast.parse(src_text)
    except SyntaxError as e:
        return False, f"the trainer does not parse: {e}"
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INIT_CKPT_MATCH" for t in n.targets):
            try:
                v = ast.literal_eval(n.value)
            except Exception as e:                                          # noqa: BLE001
                return False, f"INIT_CKPT_MATCH is not a literal: {type(e).__name__}"
            return (key in v), (f"INIT_CKPT_MATCH has {len(v)} keys and {'names' if key in v else 'does NOT name'} {key}")
    return False, "no INIT_CKPT_MATCH assignment"


FLOOR_BYTES = 1 << 30                                                       # 1 GiB: a pull must leave at least this much free on the destination drive


def disk_check(d: str, size: int, floor: int = FLOOR_BYTES, free=None) -> tuple:
    f = shutil.disk_usage(d).free if free is None else int(free)
    after = f - int(size)
    return (0 if after >= floor else 2), f"free={f} size={int(size)} after={after} floor={int(floor)}"


def pred_verdict(out_text: str, busy_exists: bool) -> tuple:
    """The predecessor rule (the brief): a predecessor STOP must not idle box 2. The arms are independent, so go when the predecessor ended with DONE, or with a STOP of its own read / judge / run, once its BOX2_BUSY is
    released. Halt on a guard stop (the box may be hot) or a BOX2_BUSY that is not released. Markers are ANCHORED at the start of a line (a 'waiting for ^ZZV8FMINI-STOP' echo starts with a date)."""
    lines = out_text.replace("\r", "").split("\n")
    done = [x for x in lines if x.startswith("ZZV8FMINI-DONE")]
    stop = [x for x in lines if x.startswith("ZZV8FMINI-STOP")]
    guard = [x for x in lines if "ZZGUARD-STOP" in x]
    if done:
        return ("HALT", "the predecessor ended DONE but BOX2_BUSY.txt is still there") if busy_exists else ("GO", done[-1][:200])
    if stop:
        if guard:
            return "HALT", "the predecessor was stopped by the GUARD (" + guard[-1][:160] + "): the box may be hot"
        if busy_exists:
            return "BUSY", "the predecessor stopped but its BOX2_BUSY.txt is not released (yet)"
        return "GO", "the predecessor STOPPED on its own (" + stop[-1][:160] + ") and released the box: the arms are independent"
    return "WAIT", "no end marker yet"

'''
t = rep(t, NL + "def main(argv=None) -> int:", new + NL + "def main(argv=None) -> int:")
t = rep(t, '    b = sub.add_parser("bank-judge")', '''    w = sub.add_parser("wacc-check")
    w.add_argument("--box", required=True)
    w.add_argument("--offline", required=True)
    w.add_argument("--offline-sha", default="")
    w.add_argument("--txt", default="")
    w.add_argument("--expect-ckpt-sha", default="")
    w.add_argument("--rtol", type=float, default=WACC_REL_TOL)
    im = sub.add_parser("init-match")
    im.add_argument("trainer")
    dc = sub.add_parser("disk-check")
    dc.add_argument("--dir", required=True)
    dc.add_argument("--size", type=int, required=True)
    dc.add_argument("--floor", type=int, default=FLOOR_BYTES)
    dc.add_argument("--free-bytes", default="")
    pv = sub.add_parser("pred-verdict")
    pv.add_argument("--out", required=True)
    pv.add_argument("--busy", required=True)
    sub.add_parser("pin-sha").add_argument("file")
    b = sub.add_parser("bank-judge")''')
t = rep(t, "    return bank_judge(a)[0]", '''    if a.mode == "wacc-check":
        rc, rec = wacc_check(a.box, a.offline, a.offline_sha, a.txt, a.expect_ckpt_sha, a.rtol)
        print("ZZDAGOACC-CO3 " + json.dumps(rec, sort_keys=True))
        return rc
    if a.mode == "init-match":
        ok, why = init_match_has(open(a.trainer, encoding="utf-8").read())
        print(why)
        return 0 if ok else 2
    if a.mode == "disk-check":
        rc, msg = disk_check(a.dir, a.size, a.floor, a.free_bytes or None)
        print(msg)
        return rc
    if a.mode == "pred-verdict":
        try:
            txt = open(a.out, encoding="utf-8", errors="replace").read() if os.path.exists(a.out) else ""
        except OSError:
            txt = ""
        v, why = pred_verdict(txt, os.path.exists(a.busy))
        print(v, why)
        return 0
    if a.mode == "pin-sha":
        print(sha256_file(a.file))
        return 0
    return bank_judge(a)[0]''')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(t, encoding="utf-8", newline=NL)
print("wrote", OUT, len(t), "bytes")
