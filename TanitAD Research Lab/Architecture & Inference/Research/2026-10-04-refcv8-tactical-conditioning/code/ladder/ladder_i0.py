"""SPEC_WPB_LADDER sec. 5 I-0 (+ the MM's two extra assertions) -- the INSTRUMENT IDENTITIES, read from the runs'
own artifacts (CPU, no model). A failure STOPS the ladder; `ladder_score.py` refuses to score without I0.json PASS.

  I0-a  arm-level identity: V-R8's step-1 row (the forward of the FIRST batch, before any update) equals V0's on every
        loss key whose inputs do not depend on the label release or on refcv8-only inputs (SPEC sec. 10.2); the
        label-dependent keys are PRINTED side by side and are not part of the identity. Read on the S0 timing runs
        (`--log-every 1`, the same seed and data order). Tolerance: |a-b| <= 1e-4 * max(1, |b|) (declared here, before
        any arm has run; two processes may pick different cuDNN kernels -- `cudnn_benchmark` is recorded per run).
  I0-b  model-level identity: the warm-start identity test of the refcv8 seams = the landed pytest
        `tests/test_refcv8_init_from.py` + `tests/test_refcv8_trainer_pin.py`, run on the LAUNCH tree by the chain;
        its junit/rc file is read here.
  I0-c  grad_share linearity: every `gs_*_lin_rel_err` reading <= 1e-4, on every arm, every logged reading.
  I0-d  the conflict detector's `cd_*` rows exist on every arm and carry the agent term.
  I0-e  the v9 join census = 100 % on train and eval (config.json: the speed source's n_known == n_windows on both
        splits for every arm; refcv8 arms additionally their v9 join census).
  MM-1  V0's argv carries no live refcv8 flag beyond the speed-only set (ladder_arms.audit, re-run on the ACTUAL argv
        recorded in V0's config.json, not on the intent).
  MM-2  V0's and V-R8's FED speed channel is byte-identical on the same windows and seed: `r8_spd_crc` (CRC32 of the
        fed bytes) equal on EVERY step both arms logged (S0: every step 1..30; the full arms: every 50th).
Usage:  python ladder_i0.py --w <W> [--arms V0,V-R8,...]   -> <W>/I0.json, exit 0 iff PASS
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_arms as LA  # noqa: E402

TOL_REL = 1e-4
LIN_MAX = 1e-4
#: keys whose value depends on the label release or on refcv8-only inputs (SPEC sec. 10.2) -- printed, not compared
LABEL_DEP = re.compile(r"(tac|goal|nav|lat|lon|route|maneuver|manoeuvre|r8|v9|cons|alloc|listwise|sel|compl|rc_)",
                       re.I)
#: per-step bookkeeping that is not a loss of the forward
NOT_LOSS = re.compile(r"(step_s|lr|time|_s$|wall|mem|grad_norm|gs_|cd_|eval_|r8_spd_crc|n_terms|skipped)", re.I)


def rows_of(run: Path):
    f = run / "metrics.jsonl"
    out = []
    if f.is_file():
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("{"):
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def is_loss_key(k):
    return ("loss" in k.lower() or k.startswith("l_")) and not NOT_LOSS.search(k)


def check_identity(r0: dict, r8: dict) -> dict:
    if not r0 or not r8:
        return {"PASS": None, "why": "UNAVAILABLE: an S0 step-1 row is missing"}
    if r0.get("step") != r8.get("step"):
        return {"PASS": False, "why": f"step mismatch {r0.get('step')} vs {r8.get('step')}"}
    shared = sorted(k for k in r0 if k in r8 and is_loss_key(k) and isinstance(r0[k], (int, float))
                    and isinstance(r8[k], (int, float)))
    cmp_k = [k for k in shared if not LABEL_DEP.search(k)]
    side = {k: [r0[k], r8[k]] for k in shared if LABEL_DEP.search(k)}
    bad = {k: [r0[k], r8[k]] for k in cmp_k if abs(float(r8[k]) - float(r0[k])) > TOL_REL * max(1.0, abs(float(r0[k])))}
    return {"PASS": (bool(cmp_k) and not bad), "n_compared": len(cmp_k), "compared": cmp_k, "mismatch": bad,
            "label_dependent_side_by_side": side, "tol": f"|a-b| <= {TOL_REL} * max(1, |b|)",
            "why": None if cmp_k else "no comparable loss key (the identity cannot be vacuous)"}


def check_crc(a_rows, b_rows) -> dict:
    a = {int(r["step"]): r["r8_spd_crc"] for r in a_rows if "r8_spd_crc" in r and "step" in r}
    b = {int(r["step"]): r["r8_spd_crc"] for r in b_rows if "r8_spd_crc" in r and "step" in r}
    both = sorted(set(a) & set(b))
    if not both:
        return {"PASS": None, "why": "UNAVAILABLE: no step logged r8_spd_crc on both arms", "n_a": len(a), "n_b": len(b)}
    diff = [s for s in both if a[s] != b[s]]
    return {"PASS": not diff, "n_steps": len(both), "mismatch_steps": diff[:20], "first": [both[0], a[both[0]]]}


def check_lin(rows) -> dict:
    vals = [(int(r.get("step", -1)), k, float(v)) for r in rows for k, v in r.items()
            if k.startswith("gs_") and k.endswith("_lin_rel_err") and v is not None]
    if not vals:
        return {"PASS": None, "why": "UNAVAILABLE: no gs_*_lin_rel_err reading"}
    worst = max(vals, key=lambda t: t[2])
    return {"PASS": worst[2] <= LIN_MAX, "n": len(vals), "worst": list(worst), "bar": LIN_MAX}


def check_cd(rows) -> dict:
    ks = sorted({k for r in rows for k in r if k.startswith("cd_")})
    return {"PASS": bool(ks) and any("agent" in k for k in ks), "n_keys": len(ks),
            "agent_keys": [k for k in ks if "agent" in k][:10]}


def check_census(cfg: dict) -> dict:
    out, ok = {}, True
    ms = (cfg or {}).get("refcv6_max_speed") or {}
    for sp in ("train", "eval"):
        s = ms.get(sp) or {}
        good = bool(s) and s.get("n_windows") and s.get("n_known") == s.get("n_windows") and s.get("mode") == "n2"
        out[f"speed_{sp}"] = {"n_windows": s.get("n_windows"), "n_known": s.get("n_known"), "mode": s.get("mode"),
                              "PASS": bool(good)}
        ok &= bool(good)
    r8 = (cfg or {}).get("refcv8")
    if r8:
        for sp in ("train", "eval"):
            j = ((r8.get("v9_join") or {}).get(sp)) if isinstance(r8.get("v9_join"), dict) else None
            if j is not None:
                good = j.get("n_joined") == j.get("n_windows") and j.get("n_windows")
                out[f"v9_{sp}"] = {k: j.get(k) for k in ("n_windows", "n_joined")} | {"PASS": bool(good)}
                ok &= bool(good)
    return {"PASS": ok, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--arms", default=",".join(LA.ORDER))
    ap.add_argument("--pytest-rc", default=None, help="file holding the launch tree's I0-b pytest exit code")
    a = ap.parse_args()
    w = Path(a.w)
    arms = [x for x in a.arms.split(",") if (w / "arms" / x / "run" / "config.json").is_file()]
    rep = {"tol_identity": TOL_REL, "lin_bar": LIN_MAX, "arms_read": arms}
    s0 = {x: rows_of(w / "s0" / x / "run") for x in ("V0", "V-R8")}
    step1 = {x: next((r for r in s0[x] if r.get("step") == 1), {}) for x in s0}
    rep["I0-a"] = check_identity(step1["V0"], step1["V-R8"])
    rc = None
    if a.pytest_rc and Path(a.pytest_rc).is_file():
        rc = Path(a.pytest_rc).read_text(encoding="utf-8").strip()
    rep["I0-b"] = {"PASS": None if rc is None else rc == "0", "pytest_rc": rc}
    rep["I0-c"], rep["I0-d"], rep["I0-e"] = {}, {}, {}
    for x in arms:
        rows = rows_of(w / "arms" / x / "run")
        rep["I0-c"][x] = check_lin(rows)
        rep["I0-d"][x] = check_cd(rows)
        cfg = json.loads((w / "arms" / x / "run" / "config.json").read_text(encoding="utf-8"))
        rep["I0-e"][x] = check_census(cfg)
        if x == "V0":
            got = dict(LA.pairs(cfg.get("argv", [])))
            live = [f for f in got if (f.startswith("--r8-") or f.startswith("--w-r8-") or f == "--refcv8")
                    and f not in LA.SPEED_ONLY_SET]
            rep["MM-1"] = {"PASS": not live, "live_refcv8_flags": live, "source": "V0 config.json argv"}
    rep["MM-2"] = {"s0": check_crc(s0["V0"], s0["V-R8"])}
    if all((w / "arms" / x / "run").is_dir() for x in ("V0", "V-R8")):
        rep["MM-2"]["arms"] = check_crc(rows_of(w / "arms" / "V0" / "run"), rows_of(w / "arms" / "V-R8" / "run"))
    rep["MM-2"]["PASS"] = (None if any(v.get("PASS") is None for v in rep["MM-2"].values() if isinstance(v, dict))
                           else all(v["PASS"] for v in rep["MM-2"].values() if isinstance(v, dict)))
    flat = [rep["I0-a"]["PASS"], rep["I0-b"]["PASS"], rep.get("MM-1", {}).get("PASS"), rep["MM-2"]["PASS"]]
    for blk in ("I0-c", "I0-d", "I0-e"):
        flat += [v["PASS"] for v in rep[blk].values()] or [None]
    rep["UNDECIDED"] = [i for i, v in enumerate(flat) if v is None]
    rep["PASS"] = bool(flat) and all(v is True for v in flat)
    (w / "I0.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("I0", "PASS" if rep["PASS"] else ("UNDECIDED" if rep["UNDECIDED"] and not any(v is False for v in flat)
                                            else "FAIL"))
    return 0 if rep["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
