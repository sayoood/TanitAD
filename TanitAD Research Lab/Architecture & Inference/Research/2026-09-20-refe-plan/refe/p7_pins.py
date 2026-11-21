#!/usr/bin/env python3
"""P7-WIRE-INTEGRATE -- READ (never choose) the two P7 launch-gate pins off the artifacts the reserve stage produced.

`refe_p7` REFUSES until three registrations exist (`launch_gate.P7_BANK_TUPLES`, `launch_gate.P7_TARGETS_SHA256`, `launch_gate.P7_AUX_STORE_PINS`) and each is a value that exists only AFTER the reserve filter has run: the bank's tuple count is
92,896 rank-0 rows + however many twins survive the filter (ESTIMATED ~150.8k, never to be registered from an estimate), and the training label store's manifest sha256 / key count are those of the store
AFTER its reserve shards were moved out and it was re-finalised.  The bank's CONTENT digest (`P7_TARGETS_SHA256`: the trainer's own `targets_sha256`) is the third: a tuple count cannot tell two same-size banks apart.  This tool reads all three off the OUTPUTS, with the same definitions the gate consumes, and prints the three registrations.

  python refe/p7_pins.py --reserve refe/fixtures/p7_reserve_sessions.json --bank <.../train_grow_cmd_3c_p7> --store <re-finalised training store> [--calib <filtered calib_table.json>]

It REFUSES (rc 1, `ZZP7PINS_REFUSED <why>`, no registration printed) unless
  * the reserve file loads (`session_reserve.load_reserve`: sha256- and content-pinned),
  * the bank directory holds `targets_rank*.jsonl` and nothing else named `targets_*.jsonl` (the trainer refuses a stray too), and carries the filter's `reserve_filter_report.json`, whose
    `n_out` equals the rows counted here and whose `targets_sha256` equals the digest re-computed here (a bank no filter produced is not a P7 bank),
  * `session_reserve.audit_inputs` finds NO reserve key and no unresolved log in the bank and the store,
  * the bank's scene count (rows grouped by (log_name, token, step), the trainer's own grouping) equals the profile's registered `epoch_len` (`--expect-scenes`, default `launch_gate.P7_SCENES`) and no scene holds a
    rank twice,
  * the store opens with the SAME reader the trainer uses (`aux_drivable_head.AuxLabelSource`, pinned to the manifest's own sha256 and key count), every shard re-hashes to the manifest, and its distinct key
    count equals the manifest's `counts.stored_keys`,
  * (with --calib) every bank row's log has a calibration row (a row without one is DROPPED by the trainer, and then the tuple count it reports is no longer the file's).
Nothing here is a default: a value is printed only when it was measured from the artifact.  The bank pins are keyed by the DIRECTORY NAME of the bank (the gate keys `n_tuples_by_bank` and
`targets_sha256_by_bank` by `--targets`'s basename).
  * the bank digest is cross-checked against the GATE's own definition (`gate_checks.bank_targets_sha256`): two copies of one definition are compared here, so a drift is a refusal now and not a
    G-CLOCK FAIL on the pod.

  python refe/p7_pins.py --check-run <run dir>     AFTER the first start: the trainer's OWN recorded identity (config.json: `identity.targets_sha256`, `identity.n_tuples`, written by train.py before
                                                    step 0 and enforced by the trainer on every --resume) must equal the registered pins of that bank.  rc 0 `ZZP7RUN_OK`, rc 1 `ZZP7RUN_REFUSED <why>`.
                                                    This closes the one window the PASS token leaves (it fingerprints a directory by its LISTING): a same-size edit of the bank between G-CLOCK and the
                                                    trainer's start.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_reserve as SR  # noqa: E402


class Refused(Exception):
    pass


def measure_bank(bank, reserve, calib=None, expect_scenes=None) -> dict:
    """The bank directory as the trainer's TargetBank reads it: the tuple count (rows of `targets_rank*.jsonl`), the scene groups, the filter report's cross-check."""
    bank = Path(bank)
    files = sorted(bank.glob("targets_rank*.jsonl"))
    if not files:
        raise Refused(f"{bank}: no targets_rank*.jsonl")
    stray = sorted(set(bank.glob("targets_*.jsonl")) - set(files))
    if stray:
        raise Refused(f"{[s.name for s in stray]} sit in {bank} but do not match targets_rank*.jsonl: the trainer refuses the directory")
    scenes: dict = {}
    by_rank: dict = {}
    n = 0
    logs = set()
    for p in files:
        for _lineno, _raw, row, why in SR._iter_json_rows(p):
            if row is None:
                raise Refused(f"{p.name}: an unparseable row ({why})")
            n += 1
            rk = int(row.get("rank", 0))
            by_rank[rk] = by_rank.get(rk, 0) + 1
            key = (row.get("log_name", ""), row.get("token", ""), int(row.get("step", 0)))
            scenes.setdefault(key, set())
            if rk in scenes[key]:
                raise Refused(f"scene {key} holds rank {rk} twice (a duplicated row): the trainer refuses the bank")
            scenes[key].add(rk)
            logs.add(row.get("log_name"))
    rep_path = bank / "reserve_filter_report.json"
    if not rep_path.is_file():
        raise Refused(f"{rep_path} is missing: a bank that no `session_reserve filter` run produced is not a P7 bank")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    digest = SR._digest_files(files)
    import gate_checks as GC
    gate_digest = GC.bank_targets_sha256(bank)
    if gate_digest != digest:                                                                         # two copies of one definition (the reserve module's, the gate's): they must agree on THIS bank
        raise Refused(f"the gate's digest of {bank} ({str(gate_digest)[:16]}...) is not the reserve module's ({digest[:16]}...): the two definitions of the trainer's targets_sha256 drifted, and the gate would FAIL this bank on the pod")
    if rep.get("n_out") != n:
        raise Refused(f"the filter report says n_out {rep.get('n_out')}, the files hold {n} rows")
    if rep.get("targets_sha256") != digest:
        raise Refused(f"the filter report's targets_sha256 {str(rep.get('targets_sha256'))[:16]}... is not the files' digest {digest[:16]}...")
    out = {"dir_name": bank.name, "n_tuples": n, "scenes": len(scenes), "rows_by_rank": dict(sorted(by_rank.items())), "files": [p.name for p in files], "targets_sha256": digest,
           "filter_report": {k: rep.get(k) for k in ("n_in", "n_out", "n_excluded", "by_rank_in", "by_rank_excluded", "token_log_checked", "source_targets_sha256")}}
    if expect_scenes is not None and len(scenes) != expect_scenes:
        raise Refused(f"the bank holds {len(scenes)} scenes, the profile registers epoch_len {expect_scenes}: re-register the schedule in the pre-registration, do not pin a bank the schedule is not sized on")
    if calib is not None:
        logs_c = set(SR._calib_logs(calib))
        miss = sorted(lg for lg in logs if lg not in logs_c)
        if miss:
            raise Refused(f"{len(miss)} bank log(s) have no calibration row, e.g. {miss[:2]}: the trainer DROPS those tuples, so its tuple count would differ from the file's")
        out["calib_logs"] = len(logs_c)
    return out


def measure_store(store) -> dict:
    """The training label store as the trainer's `_open_aux_store` opens it: the manifest's `store_sha256` and the distinct (log, token) key count of the shard headers."""
    import aux_drivable_head as AUX
    store = Path(store)
    mp = store / "manifest.json"
    if not mp.is_file():
        raise Refused(f"{mp} is missing: re-finalise the store (aux_label_store.py finalize --out <store>)")
    man = json.loads(mp.read_text(encoding="utf-8"))
    sha, keys = man.get("store_sha256"), (man.get("counts") or {}).get("stored_keys")
    try:
        src = AUX.AuxLabelSource(store, pinned_sha256=sha, pinned_keys=keys, max_logs=None)           # == train._open_aux_store
        ver = src.verify_all()
        all_keys = src.all_keys()
    except (AUX.StoreRefused, OSError, ValueError, KeyError) as e:                                   # OSError: a shard the manifest lists is not on disk (a store that was split but not re-finalised)
        raise Refused(f"{store}: the trainer's own reader refuses this store ({type(e).__name__}): {e}")
    if ver["mismatch"]:
        raise Refused(f"{store}: {ver['mismatch']} shard(s) do not hash to the manifest, e.g. {ver['first_bad']}")
    if len(set(all_keys)) != keys or len(all_keys) != keys:
        raise Refused(f"{store}: {len(set(all_keys))} distinct keys in the shard headers != the manifest's stored_keys {keys}")
    return {"store": str(store), "sha256": sha, "keys": keys, "logs": (man.get("counts") or {}).get("logs"), "shards_verified": ver["files"], "_keys": set(all_keys)}


def measure(reserve_path, bank, store, calib=None, expect_scenes=None) -> dict:
    reserve = SR.load_reserve(reserve_path)
    b = measure_bank(bank, reserve, calib=calib, expect_scenes=expect_scenes)
    s = measure_store(store)
    files = sorted(Path(bank).glob("targets_rank*.jsonl"))
    a = SR.audit_inputs(reserve, bank_files=files, label_store=store, calib_rows=calib)
    if not a["ok"]:
        raise Refused(f"the reserve audit of the bank + store did not pass: leaks {a['counts']['leak_totals']}, unresolved {a['unresolved']}, empty inputs {a['counts']['empty_inputs']}")
    keys = s.pop("_keys")
    rows = [r for p in files for _l, _r, r, _w in SR._iter_json_rows(p)]
    labelled = sum(1 for r in rows if (r.get("log_name", ""), r.get("token", "")) in keys)
    s["coverage_of_bank_rows"] = {"rows": len(rows), "rows_labelled": labelled, "fraction": round(labelled / max(len(rows), 1), 6),
                                  "note": "G-DVB requires >= AUX_MIN_COVERAGE of the trainer's own count; reported, not enforced here"}
    return {"bank": b, "store": s, "reserve_sessions": len(reserve), "audit_counts": {k: a["counts"][k] for k in ("examined", "leak_totals", "inputs_audited")},
            "registration": {"P7_BANK_TUPLES": {b["dir_name"]: b["n_tuples"]}, "P7_TARGETS_SHA256": {b["dir_name"]: b["targets_sha256"]}, "P7_AUX_STORE_PINS": {"sha256": s["sha256"], "keys": s["keys"]}}}


def check_run(run_dir, prof=None) -> dict:
    """AFTER the first start: the identity the TRAINER recorded (config.json, written by train.py before step 0; the same dict ckpt_last.pt carries and `_identity_diff` enforces on every resume) must equal the
    registered pins of the bank it trained on.  -> {"ok", "reasons", "bank", "identity"}.  An unregistered bank, a missing identity key and a mismatch are all refusals."""
    if prof is None:
        import launch_gate as L
        prof = L.PROFILES["refe_p7"]
    cp = Path(run_dir) / "config.json"
    if not cp.is_file():
        return {"ok": False, "reasons": [f"{cp} is missing: the trainer has not started this run (it writes config.json before step 0)"], "bank": None, "identity": {}}
    cfg = json.loads(cp.read_text(encoding="utf-8"))
    ident, tg = cfg.get("identity") or {}, (cfg.get("args") or {}).get("targets")
    name = Path(str(tg).rstrip("/\\")).name if tg else None
    exp = prof.get("expect_schedule") or {}
    want_sha, want_n = (exp.get("targets_sha256_by_bank") or {}).get(name), (exp.get("n_tuples_by_bank") or {}).get(name)
    reasons: list = []
    if not name:
        reasons.append("config.json carries no args.targets: the bank this run trained on is unknown")
    if want_sha is None or want_n is None:
        reasons.append(f"bank {name!r} is not registered (targets_sha256 {want_sha!r}, tuple count {want_n!r}): the run trained on a bank the profile never vouched for")
    if ident.get("targets_sha256") is None:
        reasons.append("the trainer recorded no targets_sha256 (identity.targets_sha256 is null: a --grow run does not hash its still-growing bank)")
    elif want_sha is not None and ident["targets_sha256"] != want_sha:
        reasons.append(f"the trainer's own targets_sha256 {str(ident['targets_sha256'])[:16]}... is not the registered {want_sha[:16]}...: it trained on another bank")
    if want_n is not None and ident.get("n_tuples") != want_n:
        reasons.append(f"the trainer built {ident.get('n_tuples')} tuples, the registration says {want_n}")
    return {"ok": not reasons, "reasons": reasons, "bank": name, "identity": {k: ident.get(k) for k in ("targets_sha256", "n_tuples", "n_scenes")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-run", default=None, metavar="RUN_DIR", help="AFTER the first start: compare the trainer's own recorded identity (RUN_DIR/config.json) with the registered pins; needs no other argument")
    ap.add_argument("--reserve")
    ap.add_argument("--bank", help="the reserve-free bank directory (its name is the key the gate registers the tuple count and the content digest under)")
    ap.add_argument("--store", help="the RE-FINALISED training label store")
    ap.add_argument("--calib", default=None, help="the filtered calib_table.json (optional cross-check)")
    ap.add_argument("--expect-scenes", type=int, default=None, help="default: launch_gate.P7_SCENES")
    a = ap.parse_args(argv)
    if a.check_run:
        try:
            r = check_run(a.check_run)
        except (OSError, ValueError) as e:
            print(f"ZZP7RUN_REFUSED {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if not r["ok"]:
            print("ZZP7RUN_REFUSED " + " | ".join(r["reasons"]), file=sys.stderr)
            return 1
        print(f"ZZP7RUN_OK bank={r['bank']} tuples={r['identity']['n_tuples']} targets_sha256={r['identity']['targets_sha256'][:16]}")
        return 0
    if not (a.reserve and a.bank and a.store):
        ap.error("--reserve, --bank and --store are required (or --check-run <run dir>)")
    try:
        exp = a.expect_scenes
        if exp is None:
            import launch_gate as L
            exp = L.P7_SCENES
        r = measure(a.reserve, a.bank, a.store, calib=a.calib, expect_scenes=exp)
    except (Refused, SR.ReserveError, OSError, ValueError) as e:
        print(f"ZZP7PINS_REFUSED {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(r, indent=1, sort_keys=True))
    reg = r["registration"]
    print("\n# paste into refe/launch_gate.py (P7 section), then regenerate SHIP_MANIFEST; the pod's own reserve stage must reproduce these byte for byte (diff p7_pins.py's output on the pod):")
    print(f"P7_BANK_TUPLES: dict = {json.dumps(reg['P7_BANK_TUPLES'])}")
    print(f"P7_TARGETS_SHA256: dict = {json.dumps(reg['P7_TARGETS_SHA256'])}")
    print(f"P7_AUX_STORE_PINS = {json.dumps(reg['P7_AUX_STORE_PINS'])}")
    print(f"ZZP7PINS_OK tuples={r['bank']['n_tuples']} scenes={r['bank']['scenes']} store_keys={r['store']['keys']} store_sha256={r['store']['sha256'][:16]} targets_sha256={r['bank']['targets_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
