from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/a3_keq_roll.py"); s = p.read_text(encoding="utf-8")
rep = [
("""ap.add_argument("--no-head", action="store_true", help="DIAGNOSTIC (fresh only): the tip tree with the R9 head NOT built (flags off)")""",
 """ap.add_argument("--no-head", action="store_true", help="flags-OFF: the R9 head is NOT built (AMENDMENT1 K-OFF arm on eval139; a diagnostic on fresh)")"""),
("""RC.check_prereg()
PF_CODE""", """RC.check_prereg()
AMEND = FIXPIN = None
if not FRESH and not A.dry_build:          # AMENDMENT1: an eval139 roll runs only on the pinned FIX tree, with the seal and every pin re-hashed
    AMEND, FIXPIN = RC.check_amendment1(), RC.check_fix_pin()
    if LT.resolve() != RC.FIX_TREE.resolve():
        raise SystemExit(f"[keq] eval139 rolls run on the pinned fix tree {RC.FIX_TREE}, not {LT} -- refusing")
PF_CODE"""),
("""if A.no_head and not FRESH:
    raise SystemExit("[keq] --no-head is a fresh-split diagnostic only")
OUTD = RC.SCRATCH / ("keq_dry" if A.dry_build else (((f"keq_fresh_builtunused" if A.built_unused else f"keq_fresh_nohead") if A.no_head else f"keq_fresh_pretest") if FRESH else f"keq_s{A.seed}"))""",
 """if A.built_unused and not FRESH:
    raise SystemExit("[keq] --built-unused is a fresh-split diagnostic only")
OUTD = RC.SCRATCH / ("keq_dry" if A.dry_build else (((f"keq_fresh_builtunused" if A.built_unused else f"keq_fresh_nohead") if A.no_head else f"keq_fresh_pretest") if FRESH
                                                    else f"keq_s{A.seed}_{'off' if A.no_head else 'on'}"))"""),
("""if RC.sha256_file(R9.__file__) != RC.sha256_file(RC.R9_ORIG_FILE):
    raise SystemExit("[keq] the launch tree's refcv9_progress.py differs from the banked one")""",
 """if RC.sha256_file(R9.__file__) != RC.sha256_file(RC.R9_ORIG_FILE):
    raise SystemExit("[keq] the launch tree's refcv9_progress.py differs from the banked one")
from tanitad.refs import refc_v3 as _RCV3  # noqa: E402

RCV3_SHA = RC.sha256_file(_RCV3.__file__)
if FIXPIN is not None and FIXPIN["pins"].get(Path(_RCV3.__file__).resolve().as_posix()) != RCV3_SHA:
    raise SystemExit(f"[keq] the imported refc_v3.py ({_RCV3.__file__}, {RCV3_SHA}) is not the pinned fixed file -- refusing")"""),
("""RR = PF.RR
ST = {""", """RR = PF.RR


def digest_tree(o, pre=""):
    \"\"\"sha256 of every tensor in a (nested) model output, with dtype and shape; non-tensor leaves by repr (flagged if address-bearing).\"\"\"
    res = {}
    if torch.is_tensor(o):
        t = o.detach().cpu().contiguous()
        b = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        res[pre] = f"{t.dtype}|{tuple(t.shape)}|" + hashlib.sha256(b.numpy().tobytes()).hexdigest()
    elif isinstance(o, dict):
        for k, v in o.items():
            res.update(digest_tree(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            res.update(digest_tree(v, f"{pre}[{i}]"))
    else:
        r = repr(o)
        res[pre] = ("OBJ_ADDR|" if " at 0x" in r else "repr|") + hashlib.sha256(r.encode("utf-8")).hexdigest()
    return res


ST = {"digests": [], """),
("""        if nav_cmd is None:
            ST["c"]["navnull"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))""",
 """        if nav_cmd is None:
            ST["c"]["navnull"] += 1
            out0 = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
            ST["digests"].append({"navnull": True, "idx": [int(x) for x in d["ctx"].index], "d": digest_tree(out0)})
            return out0"""),
("""        e_i, t = (int(x) for x in d["ctx"].index)
        f = lambda""", """        e_i, t = (int(x) for x in d["ctx"].index)
        ST["digests"].append({"navnull": False, "idx": [e_i, t], "conds": conds, "d": digest_tree(out)})
        f = lambda"""),
("""PF.RS.main()
recs = ST["records"]""", """PF.RS.main()
recs = ST["records"]
if not FRESH:
    # AMENDMENT1: the roll banks its records + per-call output digests in scratch; a3_keq_compare.py reads the ON / OFF pair -> result/R9PF_KEQ.json
    np.savez(OUTD / "keq_records.npz", e_i=np.array([r["e_i"] for r in recs]), t=np.array([r["t"] for r in recs]), traj=np.stack([r["traj"] for r in recs]),
             pre=np.stack([r["pre"] for r in recs]), s_t=np.stack([r["s_t"] for r in recs]), s_p=np.stack([r["s_p"] for r in recs]),
             x=np.stack([r["x"] for r in recs]), dec=np.array([r["dec"] for r in recs]), z=np.stack([r["z"] for r in recs]))
    RC.jdump(ST["digests"], OUTD / "OUTPUT_DIGESTS.json")
    rec = {"tool": "a3_keq_roll.py", "script_sha256": RC.sha256_file(__file__), "flags": ("OFF (head NOT built)" if A.no_head else "ON (head built, decision applied)"),
           "seed": A.seed, "head": ST["attached"], "pose_clock": POSE, "subset": ST["subset"], "counters": ST["c"], "n_records": len(recs),
           "n_digest_calls": len(ST["digests"]), "tanitad_file": str(TAN), "launch_tree": str(LT), "refc_v3_file": str(_RCV3.__file__), "refc_v3_sha256": RCV3_SHA,
           "amendment1": AMEND, "fix_pin": FIXPIN, "records_sha256": RC.sha256_file(OUTD / "keq_records.npz"),
           "digests_sha256": RC.sha256_file(OUTD / "OUTPUT_DIGESTS.json"), "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
    RC.jdump(rec, OUTD / "ROLL_RECORD.json")
    print(f"[keq] {rec['flags']} seed {A.seed}: {len(recs)} records, {len(ST['digests'])} digested calls  ZZKEQROLLDONEZZ", flush=True)
    sys.exit(0)"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("keq patched", len(s.splitlines()))
