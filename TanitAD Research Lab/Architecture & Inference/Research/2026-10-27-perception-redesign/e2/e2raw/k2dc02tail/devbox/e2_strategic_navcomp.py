"""e2_strategic_navcomp.py -- MM 4a3d05, 2026-11-19. The STRATEGIC family for refcv9 E2 reads (PI 2026-11-16: the strategic pause ended; every eval
reports STRATEGIC in full). refcv9 E2 arms train --no-strategic, so the computable strategic rows are taniteval.nav_compliance's ROUTE FOLLOWING
(the behavioural readouts plan / gstr / anchor vs the TRUE nav command, and the nav_shuffled / nav_zero / nav_flipped interventions).
REPORT-ONLY: it decides nothing about the pre-registered E2 rule (clause (d) is the LONGITUDINAL / LATERAL / TACTICAL rows) and moves no goalpost.

Per arm and inference seed: taniteval.nav_compliance.from_refcv3_dump(<dump_s*>, labels) (needs hold_navcomp_selbank: the sel_bank fix),
with compliance_arm SPIED to recover, per readout, the per-window nav_true compliance vector, the informative mask, tau and the episode ids
-- exactly the arrays the block itself summarises; nothing is re-derived.
Cross-arm, per readout and seed: PAIRED episode-cluster bootstrap (taniteval ci) of compliance_with_TRUE_command, ARM - REF, on the windows
informative for both (same DEV windows: nav_true, eid and N must be identical, else REFUSED). CONTROL: REF - REF must read delta exactly 0.0.
Estimator caveat (CLAUDE.md H-ESTIM-SEED-1): one training run per arm; the interval answers "another draw of EPISODES" only.

usage: python e2_strategic_navcomp.py --labels <s2_labels_v8_eval.jsonl.gz> --out-dir <dir> --ref K0 \
           --arm K0:0=<dir>/dump_s0 --arm K0:1=<dir>/dump_s1 --arm K2DC02:0=<...> --arm K2DC02:1=<...> [--n-boot 2000]
PYTHONPATH must put the FIXED taniteval package first (<tree>/taniteval)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from taniteval import nav_compliance as nc

READOUTS = ("plan", "gstr", "anchor")


def read_arm(dump_dir: str, labels: str, n_boot: int) -> tuple[dict, dict]:
    """(the block, {readout: {"comp": [N], "inf": [N] bool, "tau", "eid": [N], "nav_true": [N]}})."""
    seen_rep: dict = {}
    calls: list = []
    real_rep, real_arm = nc.nav_compliance_report, nc.compliance_arm

    def spy_rep(**kw):
        seen_rep.update(kw)
        return real_rep(**kw)

    def spy_arm(sig, nav_fed, nav_true, informative, eid, *, tau, n_boot=2000, seed=0):
        calls.append((sig, np.asarray(nav_true, dtype=np.int64), np.asarray(informative, dtype=bool), np.asarray(eid), float(tau)))
        return real_arm(sig, nav_fed, nav_true, informative, eid, tau=tau, n_boot=n_boot, seed=seed)

    nc.nav_compliance_report, nc.compliance_arm = spy_rep, spy_arm
    try:
        rep = nc.from_refcv3_dump(dump_dir, labels_path=labels, n_boot=n_boot, seed=0)
    finally:
        nc.nav_compliance_report, nc.compliance_arm = real_rep, real_arm
    if rep.get("status") == "UNAVAILABLE":
        return rep, {}
    if not seen_rep:
        raise SystemExit(f"[strategic] REFUSED: {dump_dir}: nav_compliance_report was not reached")
    by_id = {id(v): k for k, v in seen_rep["readouts"].items()}
    vec = {}
    for sig, nav_true, inf, eid, tau in calls:
        name = by_id.get(id(sig))
        if name not in READOUTS or name in vec:
            continue
        s = np.asarray(sig["nav_true"], dtype=np.float64)
        vec[name] = {"comp": nc.complies(s, nc.commanded_side(nav_true), tau), "inf": inf, "tau": tau, "eid": eid, "nav_true": nav_true}
    return rep, vec


def summary(rep: dict) -> dict:
    out = {}
    for r in READOUTS:
        b = (rep.get("readouts") or {}).get(r)
        if not b:
            out[r] = {"status": "ABSENT"}
            continue
        if b.get("status") == "UNAVAILABLE":
            out[r] = {"status": "UNAVAILABLE", "reason": b.get("reason")}
            continue
        c = b["conditionings"]
        out[r] = {"n_informative_windows": b.get("n_informative_windows"), "n_informative_episodes": b.get("n_informative_episodes"),
                  "tau": b.get("tau"), "powered": b.get("powered"),
                  "compliance_with_TRUE_command": (c.get("nav_true") or {}).get("compliance_with_TRUE_command"),
                  **{k: b[k] for k in b if k.startswith("paired_true_minus_")},
                  **{k: b[k] for k in ("n_excluded_nonfinite_signal", "n_nonfinite_signal_scored_as_miss") if k in b}}
    return out


def paired(va: dict, vb: dict, n_boot: int) -> dict:
    if not (np.array_equal(va["nav_true"], vb["nav_true"]) and np.array_equal(va["eid"], vb["eid"])):
        return {"status": "REFUSED", "reason": "the two dumps are not the same windows (nav_true / eid differ)"}
    w = va["inf"] & vb["inf"]
    if not w.any():
        return {"status": "UNAVAILABLE", "reason": "no window informative for both"}
    r = nc._ci().paired_episode_cluster_bootstrap(va["comp"][w], vb["comp"][w], list(va["eid"][w]), n_boot=n_boot, seed=0)
    return {**r, "n_windows_both_informative": int(w.sum()), "tau_arm": va["tau"], "tau_ref": vb["tau"],
            "row": "compliance_with_TRUE_command, ARM - REF (higher = follows the route more)"}


def _fmt(x, nd=4):
    return "-" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME:SEED=<dump dir>")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms: dict = {}
    for spec in a.arm:
        key, _, d = spec.partition("=")
        name, _, seed = key.partition(":")
        if not (name and seed and d):
            raise SystemExit(f"[strategic] REFUSED: bad --arm {spec!r}")
        arms[(name, seed)] = d
    doc = {"schema": "r9e2-strategic-navcomp-v1", "family": "STRATEGIC (route following vs the TRUE nav command; refcv9 E2 arms are --no-strategic)",
           "decides": "nothing about the pre-registered E2 rule: report-only (PI 2026-11-16: STRATEGIC reported in full)",
           "tier": "T1 self-action open loop", "estimator": "episode-cluster bootstrap; cross-arm = PAIRED on windows informative for both",
           "caveat": "one training run per arm: a separated interval answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
           "labels": a.labels, "ref": a.ref, "arms": {}, "vs_ref": {}, "control_ref_minus_ref": {}}
    vecs = {}
    for (name, seed), d in sorted(arms.items()):
        rep, vec = read_arm(d, a.labels, a.n_boot)
        vecs[(name, seed)] = vec
        (out / f"navcomp_{name}_s{seed}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        doc["arms"][f"{name}:s{seed}"] = {"dump": d, "status": rep.get("status", "OK"), "readouts": summary(rep),
                                         "sel_bank_scatter": rep.get("sel_bank_scatter"), "_ragged_keys": rep.get("_ragged_keys")}
    bad = []
    for (name, seed), v in sorted(vecs.items()):
        ref = vecs.get((a.ref, seed))
        if ref is None:
            continue
        for r in READOUTS:
            if r not in v or r not in ref:
                continue
            if name == a.ref:
                c = paired(v[r], ref[r], a.n_boot)
                doc["control_ref_minus_ref"][f"{r}:s{seed}"] = c
                if c.get("delta") != 0.0:
                    bad.append(f"{r}:s{seed} control delta {c.get('delta')!r} != 0.0")
            else:
                doc["vs_ref"][f"{name}-{a.ref}:{r}:s{seed}"] = paired(v[r], ref[r], a.n_boot)
    doc["control_status"] = "PASS: REF - REF reads exactly 0.0 on every readout and seed" if not bad else "FAIL: " + "; ".join(bad)
    (out / "STRATEGIC_NAVCOMP.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    L = ["# STRATEGIC family (route following) -- refcv9 E2, report-only", "", doc["family"], "", f"Tier: {doc['tier']}. Estimator: {doc['estimator']}.",
         f"Caveat: {doc['caveat']}.", f"Control: {doc['control_status']}.", "",
         "| arm:seed | readout | n inf | compliance TRUE (mean [lo, hi]) | true-shuffled | true-zero | true-flipped |", "|" + "---|" * 7]
    for k, v in doc["arms"].items():
        for r, s in v["readouts"].items():
            if "compliance_with_TRUE_command" not in s:
                L.append(f"| {k} | {r} | - | {s.get('status')}: {str(s.get('reason', ''))[:80]} | | | |")
                continue
            ct = s["compliance_with_TRUE_command"] or {}
            pd = lambda key: _fmt((s.get(key) or {}).get("delta"))   # noqa: E731
            L.append(f"| {k} | {r} | {s.get('n_informative_windows')} | {_fmt(ct.get('mean'))} [{_fmt(ct.get('lo'))}, {_fmt(ct.get('hi'))}] | "
                     f"{pd('paired_true_minus_shuffled')} | {pd('paired_true_minus_zero')} | {pd('paired_true_minus_flipped')} |")
    L += ["", f"## ARM - {a.ref}, paired on windows informative for both", "", "| pair:readout:seed | delta [lo, hi] | separated | n |", "|---|---|---|---|"]
    for k, c in doc["vs_ref"].items():
        L.append(f"| {k} | {_fmt(c.get('delta'))} [{_fmt(c.get('lo'))}, {_fmt(c.get('hi'))}] | {c.get('separated', c.get('status'))} | "
                 f"{c.get('n_windows_both_informative', '-')} |")
    (out / "STRATEGIC_NAVCOMP.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[strategic] wrote {out}/STRATEGIC_NAVCOMP.json/.md; control {doc['control_status'][:60]}")
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
