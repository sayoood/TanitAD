"""I-0 AMENDMENT A1 (POST-HOC, written 2026-10-14 AFTER the registered I-0 read FAIL). Re-reads the three failed
identities with readers that match what the trainer actually writes; leaves every registered result in place.
The registered I0.json is never modified: this writes <out>/I0.json for a SHADOW W, with the registered record inside.

  I0-d'  the conflict detector's aux side carries the agent term: every `[v3] conflict detector ON` line of the arm's
         train.log lists `agent` among `aux terms = ...`, and the arm logged `cd_*` rows. (Registered reader: a
         `cd_*` KEY containing 'agent'. The detector sums its aux terms into one gradient, so no such key can exist.)
  I0-e'  100 % speed coverage on train and eval: cfg.refcv6_max_speed.<split> n_windows > 0, n_windows_fed ==
         n_windows, window_n_unknown == 0, n_unknown == 0, source == 'label' (PI Q-B 2026-10-11). refcv8 arms also
         cfg.refcv8.v9.<split>.join n_joined == n_windows > 0. (Registered reader: n_known / mode 'n2', the pre-
         2026-10-11 schema, and cfg.refcv8.v9_join, which this trainer never writes -- that half was inert.)
  MM-2'  V0 vs V-R8 fed-speed CRC: the speed mask's generator restarts from its seed at every RESUME and at every
         500-step boundary. PREDICTED mismatch at logged step s iff last_reset(V0, s) != last_reset(V-R8, s),
         last_reset(a, s) = max{e < s : e in resumes(a) U 500N U {0}}. PASS iff observed == predicted EXACTLY (an
         unpredicted mismatch fails; so does a predicted one that did not happen), on V0/V-R8 AND, out of sample, on
         V0r/V-R8r; S0 (no resumes) must PASS as registered.
usage: ladder_i0_a1.py --w <W> --out <shadow W> | --self-test"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

RESUME = re.compile(r"resumed hier at step (\d+)")
ON = re.compile(r"conflict detector ON .*aux terms = ([^\n]*)")
BLOCK = 500


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


def d_prime(rows, log_text):
    on = ON.findall(log_text)
    cd = sorted({k for r in rows for k in r if k.startswith("cd_")})
    terms = [sorted(t.strip() for t in x.split(",")) for x in on]
    ok = bool(cd) and bool(on) and all("agent" in t for t in terms)
    return {"PASS": ok, "n_cd_keys": len(cd), "n_on_lines": len(on), "aux_terms": terms[:1]}


def e_prime(cfg):
    out, ok = {}, True
    ms = cfg.get("refcv6_max_speed") or {}
    for sp in ("train", "eval"):
        s = ms.get(sp) or {}
        n = s.get("n_windows")
        good = (bool(n) and s.get("n_windows_fed") == n and s.get("window_n_unknown") == 0
                and s.get("n_unknown") == 0 and s.get("source") == "label")
        out[f"speed_{sp}"] = {k: s.get(k) for k in ("n_windows", "n_windows_fed", "window_n_unknown", "n_unknown",
                                                     "source")} | {"PASS": bool(good)}
        ok &= bool(good)
    if cfg.get("refcv8"):
        v9 = cfg["refcv8"].get("v9") or {}
        for sp in ("train", "eval"):
            j = (v9.get(sp) or {}).get("join") or {}
            good = bool(j.get("n_windows")) and j.get("n_joined") == j.get("n_windows")
            out[f"v9_{sp}"] = {k: j.get(k) for k in ("n_windows", "n_joined")} | {"PASS": bool(good)}
            ok &= bool(good)
    return {"PASS": ok, **out}


def last_reset(events, s):
    return max([e for e in events if e < s] + [0])


def mm2_prime(a_rows, b_rows, a_res, b_res, n_steps):
    a = {int(r["step"]): r["r8_spd_crc"] for r in a_rows if "r8_spd_crc" in r}
    b = {int(r["step"]): r["r8_spd_crc"] for r in b_rows if "r8_spd_crc" in r}
    both = sorted(set(a) & set(b))
    grid = set(range(0, n_steps + 1, BLOCK))
    ea, eb = set(a_res) | grid, set(b_res) | grid
    obs = [s for s in both if a[s] != b[s]]
    pred = [s for s in both if last_reset(ea, s) != last_reset(eb, s)]
    return {"PASS": bool(both) and obs == pred, "n_steps": len(both), "observed_mismatch": obs,
            "predicted_mismatch": pred, "resumes": [sorted(a_res), sorted(b_res)]}


def arm(w, x):
    run = w / "arms" / x / "run"
    log = (w / "arms" / x / "train.log").read_text(encoding="utf-8", errors="replace")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    return rows_of(run), log, cfg, sorted({int(m) for m in RESUME.findall(log)})


def main(a):
    w, out = Path(a.w), Path(a.out)
    reg = json.loads((w / "I0.json").read_text(encoding="utf-8"))
    rep = {k: reg[k] for k in ("tol_identity", "lin_bar", "arms_read", "I0-a", "I0-b", "I0-c", "MM-1")}
    rep["AMENDMENT"] = ("A1 POST-HOC 2026-10-14: I0-d, I0-e, MM-2 re-read with ladder_i0_a1.py after the registered "
                        "I-0 read FAIL; the registered record is kept verbatim under 'registered'")
    rep["registered"] = reg
    n = int((w / "N.txt").read_text().strip())
    A = {x: arm(w, x) for x in reg["arms_read"]}
    rep["I0-d"] = {x: d_prime(A[x][0], A[x][1]) for x in A}
    rep["I0-e"] = {x: e_prime(A[x][2]) for x in A}
    mm = {"s0": reg["MM-2"]["s0"],
          "arms": mm2_prime(A["V0"][0], A["V-R8"][0], A["V0"][3], A["V-R8"][3], n),
          "oos_V0r_V-R8r": mm2_prime(A["V0r"][0], A["V-R8r"][0], A["V0r"][3], A["V-R8r"][3], n)}
    mm["PASS"] = all(v.get("PASS") is True for v in mm.values() if isinstance(v, dict))
    rep["MM-2"] = mm
    flat = [rep["I0-a"]["PASS"], rep["I0-b"]["PASS"], rep["MM-1"]["PASS"], mm["PASS"]]
    for blk in ("I0-c", "I0-d", "I0-e"):
        flat += [v["PASS"] for v in rep[blk].values()]
    rep["PASS"] = all(v is True for v in flat)
    out.mkdir(parents=True, exist_ok=True)
    (out / "I0.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("I0-A1", "PASS" if rep["PASS"] else "FAIL",
          "| d", all(v["PASS"] for v in rep["I0-d"].values()),
          "| e", all(v["PASS"] for v in rep["I0-e"].values()),
          "| MM-2", mm["arms"]["PASS"], "oos", mm["oos_V0r_V-R8r"]["PASS"],
          "| obs", mm["arms"]["observed_mismatch"], "pred", mm["arms"]["predicted_mismatch"],
          "| oos-obs", mm["oos_V0r_V-R8r"]["observed_mismatch"], "oos-pred", mm["oos_V0r_V-R8r"]["predicted_mismatch"])
    return 0 if rep["PASS"] else 3


def self_test():
    """Each reader must go RED on the defect it exists to catch (literal fixtures, not the code's own output)."""
    rows = [{"step": s, "r8_spd_crc": 7} for s in range(50, 3001, 50)]
    rows_b = [dict(r, r8_spd_crc=(9 if 250 < r["step"] <= 500 else 7)) for r in rows]
    assert mm2_prime(rows, rows_b, [500], [250, 500], 3000)["PASS"] is True
    bad = copy.deepcopy(rows_b)
    bad[20]["r8_spd_crc"] = 1                                   # step 1050: a predicted MATCH that mismatches
    assert mm2_prime(rows, bad, [500], [250, 500], 3000)["PASS"] is False
    assert mm2_prime(rows, rows, [500], [250, 500], 3000)["PASS"] is False     # a predicted mismatch that is absent
    log = "[v3] conflict detector ON (probe): x; aux terms = agent, box3d, map_hires\n"
    assert d_prime([{"cd_cos": 1}], log)["PASS"] is True
    assert d_prime([{"cd_cos": 1}], log.replace("agent, ", ""))["PASS"] is False
    assert d_prime([], log)["PASS"] is False
    s = {"n_windows": 10, "n_windows_fed": 10, "window_n_unknown": 0, "n_unknown": 0, "source": "label"}
    cfg = {"refcv6_max_speed": {"train": dict(s), "eval": dict(s)}}
    assert e_prime(cfg)["PASS"] is True
    c2 = copy.deepcopy(cfg)
    c2["refcv6_max_speed"]["eval"]["window_n_unknown"] = 1
    assert e_prime(c2)["PASS"] is False
    c3 = copy.deepcopy(cfg)
    c3["refcv8"] = {"v9": {"train": {"join": {"n_windows": 10, "n_joined": 9}},
                           "eval": {"join": {"n_windows": 5, "n_joined": 5}}}}
    assert e_prime(c3)["PASS"] is False
    c4 = copy.deepcopy(cfg)
    c4["refcv8"] = {"enable": True}                            # a refcv8 arm whose census is missing
    assert e_prime(c4)["PASS"] is False
    c5 = copy.deepcopy(cfg)
    c5["refcv6_max_speed"]["train"]["source"] = "n2"
    assert e_prime(c5)["PASS"] is False
    print("SELF-TEST OK (8 mutation arms RED, 3 green)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    sys.exit(main(a))
