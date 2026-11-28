"""ship_r8_verdict_pf.py -- SHIP-R8 under AMENDMENT 3 (per-frame feed), applied MECHANICALLY.  CPU only, pure json in, json + md out.

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  COPY + EXTEND: clauses (1)-(7) are the battery package's ``ship_r8_verdict.py``, byte-copied
to ``_ship_r8_verdict_base.py`` (sha256 pinned below and re-checked at import), with three changes that AMENDMENT 3 makes:

  * the tree is ``step48852ema_pf`` and the pinned frozen-train sha256 is the REGISTERED PER-FRAME table (read from
    ``REGISTERED_PF_TABLE.sha256``; never the clip-fed b208c9c1..., which is printed beside only -- (ii));
  * NEW clause (1b) FEED PARITY (iii), re-derived from the two ``FEED_PARITY_<tree>_s{0,1}.json`` written by ``feed_parity_pf.py``
    (fed token == v9 per-frame token; ``nav_cmd_clip`` == the per-clip token of a CLIP-FED dump of the same windows; ``r8_nav`` non-zero
    exactly where ``r8_nav_known``) on the FULL window set of the tree's own dumps.  If it fails OR cannot be shown, SHIP-R8 reads
    INCOMPLETE -- whatever the other clauses read (their numbers would be on rolls that are not the amended feed);
  * the registration order (ii): the registered sha256 must equal the frozen file, its provenance, the chain's gate record, and
    FORWARD_TRAINFIT's frozen hash, and the gate record must PRECEDE every pf roll of the tree.  Not shown -> clause (1) INCOMPLETE.
  * (v) a clause that is missing from the list reads INCOMPLETE, never PASS (``overall_pf``).
The per-clip verdict (``SHIP_R8_step48852ema_unc.json``, the deliberate-regression arm) is printed BESIDE and is never binding (iv).

    python ship_r8_verdict_pf.py --tree step48852ema_pf --pf-pkg <hold>/pfpkg --registered C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256 \
        --frozen <hold>/raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json --parity-dir <hold>/raw/feed_parity --gate-json <hold>/raw/REGISTRATION_GATE.json \
        --out-dir <hold>/raw/ship_r8
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_FILE = HERE / "_ship_r8_verdict_base.py"
BASE_SHA256 = "7543c585d9cc099942862c7f644c5fcae797272cd9b4dfec93708450224660f2"     # == battery package ship_r8_verdict.py at copy time (2026-10-20)
CONDITION_TREE = "step48852ema_pf"
CLIPFED_SHA256 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
PERCLIP_VERDICT = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/ship_r8/"
                       "SHIP_R8_step48852ema_unc.json")
REQUIRED_IDS = ("1", "1b", "2", "3", "4", "5", "6", "7")
PARITY_CHECKS = ("T_fed_token_equals_v9_per_frame_token", "C_nav_cmd_clip_equals_clip_fed_reference", "C_reference_constant_within_every_clip",
                 "C_same_window_set_as_clip_fed_reference", "K_known_equals_release_args_valid",
                 "A_r8_nav_nonzero_exactly_where_known_and_equals_independent_scaling", "I_clip_identity_sha12_matches_release",
                 "manifest_mode_is_perframe_full")
AMENDMENT3 = ("AMENDMENT 3: (i) every SHIP-R8 clause and every refcv8 bar is read on rolls fed `perframe_full` (tree step 48,852 EMA, tag `step48852ema_pf`, "
              "inference seeds 0 and 1, paired episode-cluster bootstrap B 2000); (ii) the RETIME table is re-fitted by the unchanged H-R8-RETIME-TRAINFIT-1 "
              "procedure on per-frame-fed train rolls (30k EMA, A6 train139, seeds 0+1 pooled, prior ha0_ext_pose), its sha256 registered BEFORE any 48,852 pf "
              "eval roll; the clip-fed table b208c9c1... is printed beside only; (iii) new clause (1b) FEED PARITY asserted on the roll artifacts -- if it fails "
              "or cannot be shown, SHIP-R8 reads INCOMPLETE; (iv) the per-clip reads (`step48852ema_unc`) are printed beside as the deliberate-regression arm, "
              "never binding; (v) a missing clause means INCOMPLETE, never PASS.  Amendments 1 and 2 still apply.")
CLAUSE_1B = ("(1b) FEED PARITY: on the tree's own roll artifacts, both seeds, every window: the fed token equals the v9 per-frame token, `nav_cmd_clip` "
             "equals the per-clip token, and `r8_nav` is non-zero exactly where `r8_nav_known`; fails or cannot be shown -> SHIP-R8 INCOMPLETE")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_base():
    got = sha256_file(BASE_FILE)
    if got != BASE_SHA256:
        raise SystemExit(f"[ship_pf] {BASE_FILE} hashes to {got}, not the pinned copy {BASE_SHA256}")
    spec = importlib.util.spec_from_file_location("_ship_r8_verdict_base", BASE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load_base()


def _is_hex64(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def read_registered(path) -> str:
    try:
        s = "".join(c for c in Path(path).read_text(encoding="utf-8-sig").split()[0] if c in "0123456789abcdefABCDEF").lower()   # BOM / CRLF / "sha  file" tolerant
    except Exception:      # noqa: BLE001
        return ""
    return s if _is_hex64(s) else ""


# ------------------------------------------------------------------------------------------ the new clause (pure)
def clause1b(parity: dict, tree: str, expected_n_windows: dict, archive: str = ARCHIVE) -> dict:
    """``parity`` = {"0": FEED_PARITY json or None, "1": ...}; ``expected_n_windows`` = {"0": n, "1": n} from the tree's own dump manifests (None if unreadable).
    Re-derives the reading from the per-check booleans and the counts; the file's own status is cross-checked, never trusted alone."""
    ev = {"seeds": {}}
    fails, unk = [], []
    for s in B.SEEDS:
        p = parity.get(s)
        want = expected_n_windows.get(s)
        if p is None:
            unk.append(f"s{s}: FEED_PARITY file absent")
            ev["seeds"][s] = None
            continue
        ch, cn = p.get("checks") or {}, p.get("counts") or {}
        row = {"file_status": p.get("status"), "checks": {k: ch.get(k) for k in PARITY_CHECKS}, "n_windows": cn.get("n_windows"),
               "n_windows_expected_from_tree_manifest": want, "n_windows_only_in_pf": cn.get("n_windows_only_in_pf"),
               "n_windows_only_in_reference": cn.get("n_windows_only_in_reference"), "n_token_differs_from_clip": cn.get("n_token_differs_from_clip"),
               "n_args_known": cn.get("n_args_known"), "subset_ok_smoke_only": p.get("subset_ok_smoke_only"), "pf_dump": p.get("pf_dump"),
               "clip_dump": p.get("clip_dump"), "first_failures": p.get("first_failures")}
        ev["seeds"][s] = row
        exp_dump = f"{archive}/{tree}/dump_s{s}".replace("\\", "/").lower()
        if str(p.get("pf_dump") or "").replace("\\", "/").rstrip("/").lower() != exp_dump:
            unk.append(f"s{s}: parity was run on {p.get('pf_dump')!r}, not on the tree's dump {exp_dump}")
            continue
        if p.get("subset_ok_smoke_only"):
            unk.append(f"s{s}: parity ran in smoke (subset) mode")
            continue
        bad = [k for k in PARITY_CHECKS if ch.get(k) is False]
        miss = [k for k in PARITY_CHECKS if ch.get(k) is None]
        if bad:
            fails.append(f"s{s}: " + ", ".join(bad))
            continue
        if miss:
            unk.append(f"s{s}: checks not stated: {', '.join(miss)}")
            continue
        if not isinstance(want, int) or want <= 0 or cn.get("n_windows") != want:
            unk.append(f"s{s}: parity covered {cn.get('n_windows')} windows, the tree's dump holds {want}")
            continue
        if cn.get("n_windows_only_in_pf") != 0 or cn.get("n_windows_only_in_reference") != 0:
            unk.append(f"s{s}: window sets differ (only pf {cn.get('n_windows_only_in_pf')}, only reference {cn.get('n_windows_only_in_reference')})")
            continue
        if p.get("status") != "PASS":
            unk.append(f"s{s}: every check reads true but the file's own status is {p.get('status')!r}: the two disagree")
    if fails:
        return {"id": "1b", "text": CLAUSE_1B, "status": "FAIL", "reason": "FEED PARITY FAILED (SHIP-R8 therefore INCOMPLETE): " + "; ".join(fails), "evidence": ev}
    if unk:
        return {"id": "1b", "text": CLAUSE_1B, "status": "INCOMPLETE", "reason": "FEED PARITY cannot be shown: " + "; ".join(unk), "evidence": ev}
    return {"id": "1b", "text": CLAUSE_1B, "status": "PASS",
            "reason": "both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known",
            "evidence": ev}


def registration(registered: str, frozen_sha: str | None, prov: dict | None, gate: dict | None, tf: dict | None, roll_mtimes: dict) -> dict:
    """AMENDMENT 3 (ii) as evidence for clause (1): every record of the table's hash agrees, it is not the clip-fed table, and the chain's
    gate record (written when the gate passed, before the GPU stage) predates every pf roll artifact of the tree."""
    ev = {"registered": registered or None, "frozen_file_sha256": frozen_sha, "provenance_sha256": (prov or {}).get("sha256"),
          "gate_registered_sha256": (gate or {}).get("registered_sha256"), "gate_passed_epoch": (gate or {}).get("passed_epoch"),
          "forward_trainfit_frozen_gains_sha256": (tf or {}).get("frozen_gains_sha256"), "roll_artifact_mtimes_epoch": roll_mtimes}
    bad, unk = [], []
    if not _is_hex64(registered):
        unk.append("REGISTERED_PF_TABLE.sha256 is absent or not a 64-hex digest")
    elif registered == CLIPFED_SHA256:
        bad.append("the registered table is the CLIP-FED one (b208c9c1...): Amendment 3 (ii) needs the per-frame re-fit")
    for nm in ("frozen_file_sha256", "provenance_sha256", "gate_registered_sha256", "forward_trainfit_frozen_gains_sha256"):
        v = ev[nm]
        if not _is_hex64(v):
            unk.append(f"{nm} not shown")
        elif _is_hex64(registered) and v != registered:
            bad.append(f"{nm} {v[:12]}... != registered {registered[:12]}...")
    gt = ev["gate_passed_epoch"]
    if not isinstance(gt, (int, float)):
        unk.append("the gate record carries no pass time")
    elif not roll_mtimes or any(not isinstance(t, (int, float)) for t in roll_mtimes.values()):
        unk.append("the tree's roll artifacts carry no time")
    elif any(t <= gt for t in roll_mtimes.values()):
        bad.append(f"a pf roll artifact predates the registration gate ({roll_mtimes} vs gate {gt})")
    return {"ok": not bad and not unk, "bad": bad, "unknown": unk, "evidence": ev}


def clause1_pf(fd8, tf, deploy_sha, reg: dict) -> dict:
    if not _is_hex64(reg["evidence"].get("registered")):           # nothing to compare against: not evaluable, never a FAIL by default and never a PASS
        return {"id": 1, "text": B.CLAUSE_TEXT[1], "status": "INCOMPLETE",
                "reason": "no registered per-frame table (REGISTERED_PF_TABLE.sha256 absent or not 64-hex): clause (1) cannot be evaluated under Amendment 3 (ii)",
                "evidence": {"amendment3_registration": reg}}
    B.FROZEN_SHA256 = reg["evidence"]["registered"]
    c = B.clause1(fd8, tf, deploy_sha)                    # B.FROZEN_SHA256 is the REGISTERED per-frame sha
    c["evidence"]["amendment3_registration"] = reg
    if c["status"] == "FAIL":
        return c
    if reg["bad"]:
        c.update({"status": "FAIL", "reason": c["reason"] + " | REGISTRATION: " + "; ".join(reg["bad"])})
    elif reg["unknown"]:
        c.update({"status": "INCOMPLETE", "reason": c["reason"] + " | REGISTRATION not shown: " + "; ".join(reg["unknown"])})
    else:
        c["reason"] += " | the per-frame table's sha256 was registered before every pf roll of the tree (Amendment 3 (ii))"
    return c


def overall_pf(clauses: list) -> str:
    """(v): every required clause present with a recognised status; (iii): clause (1b) not PASS -> INCOMPLETE, whatever the others read; else the base rule."""
    by = {str(c.get("id")): c.get("status") for c in clauses}
    if any(i not in by or by[i] not in ("PASS", "FAIL", "INCOMPLETE", "N/A") for i in REQUIRED_IDS):
        return "INCOMPLETE"
    if by["1b"] != "PASS":
        return "INCOMPLETE"
    st = [by[i] for i in REQUIRED_IDS]
    if "FAIL" in st:
        return "FAIL"
    if "INCOMPLETE" in st:
        return "INCOMPLETE"
    return "PASS"


def evaluate_pf(tree, fd8, tf, tac, pr, pp, fd_ps, deploy_sha, parity, expected_n, reg, perclip=None, prov=None, archive=ARCHIVE) -> dict:
    clauses = [clause1_pf(fd8, tf, deploy_sha, reg), clause1b(parity, tree, expected_n, archive), B.clause_e(2, "S2", fd8), B.clause_e(3, "S6", fd8),
               B.clause4(fd8), B.clause5(tac), B.clause6(pr), B.clause7(pp)]
    ov = overall_pf(clauses)
    orient = tree != CONDITION_TREE
    bes = B.beside(fd8, tf, tac, fd_ps, None, None)
    bes["clip_fed_table_beside"] = {"sha256": CLIPFED_SHA256, "gains": ((prov or {}).get("beside_clip_fed_table") or {}).get("gains"),
                                    "note": "Amendment 3 (ii): printed beside only"}
    bes["per_clip_verdict_deliberate_regression_arm"] = (
        {"file": str(PERCLIP_VERDICT), "overall": perclip.get("overall"), "per_clause": perclip.get("per_clause"), "tree": perclip.get("tree"),
         "binding": False} if perclip else {"file": str(PERCLIP_VERDICT), "status": "NOT YET COMPUTED (the per-clip chain runs after the pf GPU stage)", "binding": False})
    return {"tool": "ship_r8_verdict_pf.py", "generated": time.strftime("%FT%T%z"), "tree": tree, "condition_tree": CONDITION_TREE, "orientation_only": orient,
            "label": (f"SHIP-R8 (AMENDMENT 3) on {tree}: {ov} -- ORIENTATION ONLY (the condition's tree is {CONDITION_TREE})" if orient else
                      f"SHIP-R8 VERDICT (AMENDMENT 3, per-frame feed) on {tree}: {ov}"),
            "overall": ov, "clauses": clauses, "arm": B.ARM, "registered_pf_table_sha256": reg["evidence"].get("registered"),
            "condition_text": B.CONDITION, "amendment_text": B.AMENDMENT + "  " + AMENDMENT3,
            "per_clause": {str(c["id"]): c["status"] for c in clauses},
            "caveats": ["single training seed", "T1 open loop (not driving performance)", "the stage was found post-hoc", "the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF)",
                        "the bootstrap answers 'another draw of EPISODES' only (not another training or inference run)",
                        "the speed-obedience / vmax-zero proxy arms of the battery carry no r8_nav (unknown row); they enter no SHIP-R8 clause"],
            "beside": bes, "base_verdict_copy": {"file": str(BASE_FILE), "sha256": BASE_SHA256}}


def render_pf(v: dict) -> str:
    L = B.render({**v, "clauses": [c for c in v["clauses"] if c["id"] != "1b"]}).splitlines()
    c1b = next(c for c in v["clauses"] if c["id"] == "1b")
    L[0] = f"# SHIP_R8_{v['tree']} (AMENDMENT 3, per-frame feed)"
    ins = [i for i, x in enumerate(L) if x.startswith("| (1) ")]
    if ins:
        L.insert(ins[0] + 1, f"| {c1b['text']} | **{c1b['status']}** | {c1b['reason']} |")
    L += ["## feed parity (clause 1b), per seed", ""]
    for s, r in (c1b["evidence"].get("seeds") or {}).items():
        L.append(f"* s{s}: " + ("absent" if r is None else
                                f"status {r['file_status']}, windows {r['n_windows']} (tree {r['n_windows_expected_from_tree_manifest']}), token differs from clip on "
                                f"{r['n_token_differs_from_clip']}, args known on {r['n_args_known']}"))
    pc = v["beside"]["per_clip_verdict_deliberate_regression_arm"]
    L += ["", "## per-clip verdict (deliberate-regression arm, NEVER binding)", "",
          f"* {pc.get('tree', 'step48852ema_unc')}: overall {pc.get('overall', pc.get('status'))}; per clause {pc.get('per_clause')}", "",
          f"registered per-frame table sha256: `{v['registered_pf_table_sha256']}`; clip-fed table `{CLIPFED_SHA256}` printed beside only", ""]
    return "\n".join(L)


def _load(p):
    return B._load(p)


def roll_mtimes(tree: str, archive: str = ARCHIVE) -> dict:
    out = {}
    for s in B.SEEDS:
        p = Path(f"{archive}/{tree}/dump_s{s}/manifest.json")
        out[s] = os.path.getmtime(p) if p.exists() else None
    return out


def expected_windows(tree: str, archive: str = ARCHIVE) -> dict:
    out = {}
    for s in B.SEEDS:
        m = _load(f"{archive}/{tree}/dump_s{s}/manifest.json")
        try:
            out[s] = int(m["grid"]["n_windows"])
        except (KeyError, TypeError, ValueError):
            out[s] = None
    return out


def run(tree, pf_pkg, registered_file, frozen, parity_dir, gate_json, out_dir, archive=ARCHIVE, perclip_file=PERCLIP_VERDICT) -> dict:
    reg_sha = read_registered(registered_file)                          # clause1_pf pins B.FROZEN_SHA256 to it (or reads INCOMPLETE without one)
    raw = Path(pf_pkg) / "raw"
    tf_dir, dep = raw / "retime_trainfit", raw / "retime_deploy"
    p = {"tf": tf_dir / f"FORWARD_TRAINFIT_{tree}.json", "fd8": dep / f"FORWARD_DEPLOY8_{tree}.json", "tac": dep / f"TAC_{tree}.json", "pp": dep / "PRIOR_PARITY.json",
         "fd_ps": dep / f"FORWARD_DEPLOY_{tree}.json", "pr": Path(f"{archive}/{tree}/PAIRED_READ_{tree}.json"), "deploy_sha": dep / "FROZEN_GAINS_RETIME_deploy.sha256",
         "frozen": Path(frozen), "provenance": Path(str(frozen).replace(".json", ".provenance.json")), "gate": Path(gate_json), "registered": Path(registered_file),
         "parity_s0": Path(parity_dir) / f"FEED_PARITY_{tree}_s0.json", "parity_s1": Path(parity_dir) / f"FEED_PARITY_{tree}_s1.json", "perclip": Path(perclip_file)}
    dsha = p["deploy_sha"].read_text().strip() if p["deploy_sha"].exists() else None
    tf = _load(p["tf"])
    prov = _load(p["provenance"])
    reg = registration(reg_sha, sha256_file(p["frozen"]) if p["frozen"].exists() else None, prov, _load(p["gate"]), tf, roll_mtimes(tree, archive))
    v = evaluate_pf(tree, _load(p["fd8"]), tf, _load(p["tac"]), _load(p["pr"]), _load(p["pp"]), _load(p["fd_ps"]), dsha,
                    {"0": _load(p["parity_s0"]), "1": _load(p["parity_s1"])}, expected_windows(tree, archive), reg, _load(p["perclip"]), prov, archive)
    v["inputs"] = {k: str(x) for k, x in p.items()}
    v["inputs_sha256"] = {k: (sha256_file(x) if Path(x).exists() else None) for k, x in p.items()}
    v["script_sha256"] = sha256_file(__file__)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    jp = od / f"SHIP_R8_{tree}.json"
    jp.write_text(json.dumps(v, indent=1, default=str) + "\n", encoding="utf-8")
    (od / f"SHIP_R8_{tree}.md").write_text(render_pf(v), encoding="utf-8")
    chk = json.loads(jp.read_text(encoding="utf-8"))                     # assert on the ARTIFACT
    assert chk["overall"] in ("PASS", "FAIL", "INCOMPLETE") and [str(c["id"]) for c in chk["clauses"]] == list(REQUIRED_IDS)
    return v


def main() -> int:
    hold = HERE.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=CONDITION_TREE)
    ap.add_argument("--pf-pkg", default=str(hold / "pfpkg"))
    ap.add_argument("--registered", default="C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256")
    ap.add_argument("--frozen", default=str(hold / "raw" / "retime_trainfit_pf" / "FROZEN_GAINS_RETIME_train_pf.json"))
    ap.add_argument("--parity-dir", default=str(hold / "raw" / "feed_parity"))
    ap.add_argument("--gate-json", default=str(hold / "raw" / "REGISTRATION_GATE.json"))
    ap.add_argument("--out-dir", default=str(hold / "raw" / "ship_r8"))
    ap.add_argument("--archive", default=ARCHIVE)
    a = ap.parse_args()
    v = run(a.tree, a.pf_pkg, a.registered, a.frozen, a.parity_dir, a.gate_json, a.out_dir, a.archive)
    print(v["label"])
    for c in v["clauses"]:
        print(f"  {c['text'][:70]:<70} {c['status']:<10} {c['reason'][:120]}")
    print(f"ZZSHIPPFVERDICT_{v['overall']}ZZ {v['tree']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
