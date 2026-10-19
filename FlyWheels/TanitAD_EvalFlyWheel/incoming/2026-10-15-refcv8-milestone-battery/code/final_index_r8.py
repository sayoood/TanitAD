"""final_index_r8.py -- one page that says what the FINAL (step 48,852) chain produced, where it lives, and what each stage ended in.  CPU only, stdlib only.  NEW file (2026-10-19, EvalFlyWheel).

    python -I final_index_r8.py --pkg <PKG> --kit E:/refcv8_eval_kit --arch C:/Users/Admin/refcv8_eval_archive/battery --step 48852 --chain-log <chain_final.log> --out-dir <PKG>/raw/step48852
    -> FINAL_INDEX.json + FINAL_INDEX.md

It READS artifacts and quotes their own status fields; it computes no result.  An artifact that EXISTS but cannot be parsed is reported UNREADABLE (never "absent"); an absent one is ABSENT.
Every stage line is the LAST marker of that stage after the chain's last ZZCFNSTARTZZ, so a resumed chain reports its resumed state.  The weights block carries the md5 chain
(Thor ckpt.pt -> ckpt_full -> derived raw/EMA pair) and the step INSIDE the file as the extractor recorded it -- summary.json is listed as evidence of completion, never as the proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

STAGES = [("PULL", "PULLSEEN|PULLFAIL|PULLSTART"), ("MD5", "MD5OK|MD5BAD|MD5INCONCLUSIVE"), ("SUMMARY.JSON", "SUMJSONOK|SUMJSONBAD"), ("EXTRACT", "EXTRACTOK|EXTRACTHAVE|EXTRACTFAIL|EXTRACTSTART"),
          ("EMA GATE", "EMAGATEOK|EMAGATEFAIL"), ("XCHECK", "XCHECKOK|XCHECKHAVE|XCHECKNA|XCHECKWARN|XCHECKOFF|XCHECKFAIL"), ("EMA BATTERY", "EMADONE|EMAHAVE|EMAFAIL|EMABANKFAIL|EMASTART"),
          ("K-DRAW ROLLS", "KDDONE|KDHAVE|KDSKIP|KDFAIL|KDSTART|KDBUSY"), ("K-DRAW READ", "KDANALYSIS|KDANALYSISFAIL|KDANALYSISSKIP"), ("RAW BATTERY", "RAWDONE|RAWHAVE|RAWINCOMPLETE|RAWSTART"),
          ("RAW G0", "RAWG0OK|RAWG0NOJSON"), ("WRAPPER PROBE", "WRAPFIXOK|WRAPFIXHAVE|WRAPFIXFAIL|RAWWRAPFIXFAILED|WRAPFIXSTART"), ("TRAIN-FIT", "TFOK|TFHAVE|TFSKIP|TFWITHHELD|TFFAIL|TFWAIT"),
          ("COMBO", "COMBODONE|COMBOHAVE|COMBOSKIP|COMBOFAIL"), ("LANE", "LANEEND|LANESKIP|LANETIMEOUT"), ("SUMMARY", "SUMMARYOK|SUMMARYFAIL"), ("CHAIN", "END|ABORT")]


def jl(p: Path):
    if not p.exists():
        return "ABSENT"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"UNREADABLE ({type(e).__name__})"


def md5(p: Path) -> str:
    if not p.exists():
        return "ABSENT"
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def last_markers(log: Path) -> list[str]:
    if not log.exists():
        return []
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max([i for i, l in enumerate(lines) if "ZZCFNSTARTZZ" in l] or [0])
    return lines[start:]


def stage_line(lines: list[str], alts: str) -> str:
    pat = re.compile(r"ZZCFN(" + alts + r")ZZ")
    hit = [l for l in lines if pat.search(l)]
    return hit[-1][:240] if hit else "NO MARKER"


def present(p: Path):
    return "ABSENT" if not p.exists() else ("EMPTY" if p.stat().st_size == 0 else f"present ({p.stat().st_size} B)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--chain-log", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    pkg, kit, arch, s = Path(a.pkg), Path(a.kit), Path(a.arch), a.step
    te, tr = f"step{s}ema_unc", f"step{s}_unc"
    ck = kit / "ckpt"
    ex = jl(ck / f"ckpt_{s}.provenance.extract_ema.json")
    prov = jl(ck / f"ckpt_full_{s}.pt.provenance.json")
    summ = jl(ck / f"summary_thor_{s}.json")
    xc = jl(pkg / "raw" / f"step{s}ema" / f"XCHECK_MODEL_EMA_step{s}.json")
    gate = jl(pkg / "raw" / f"step{s}ema" / f"EMA_CHECK_step{s}.json")
    d = lambda j, *k: (j if isinstance(j, str) else _g(j, k))  # noqa: E731
    weights = {"thor_ckpt_pt_md5_as_pulled": d(prov, "md5"), "ckpt_full_local_md5": md5(ck / f"ckpt_full_{s}.pt"), "step_inside_file_per_extractor": d(ex, "step"), "extract_ok": d(ex, "ok"),
               "extract_src_md5": d(ex, "src_md5"), f"ckpt_{s}.pt_md5": md5(ck / f"ckpt_{s}.pt"), f"ckpt_ema_{s}.pt_md5": md5(ck / f"ckpt_ema_{s}.pt"),
               "trainer_model_ema_md5": md5(ck / f"model_ema_{s}_thor.pt"), "xcheck_ok": d(xc, "ok"), "xcheck_n_tensors": d(xc, "n_tensors"), "ema_gate_ok": d(gate, "ok"),
               "summary_json_copy": {"done": d(summ, "done"), "step": d(summ, "step"), "final_step": d(summ, "final_step"), "note": "evidence of completion, never the proof of the step"}}
    trees = {}
    for t in (te, tr):
        b = arch / t
        bs = jl(b / "battery_summary.json")
        trees[t] = {"battery_summary": ("finished " + str(bs.get("finished"))) if isinstance(bs, dict) else bs, "ckpt_md5": d(bs, "ckpt_md5"), "paired_read": present(b / f"PAIRED_READ_{t}.json"),
                    "clamp_posthoc": present(pkg / "raw/clamp_posthoc/trees" / f"{t}.json"),
                    "compliant_reread": (lambda j: j if isinstance(j, str) else {"controls_ALL_PASS": _g(j, ("controls", "ALL_PASS")), "bars_status": _g(j, ("bars", "status"))})(jl(pkg / "raw/compliant_reread/trees" / f"{t}.json")),
                    "forward_a0k0": _fw(pkg / "raw/a0k0_headroom" / f"FORWARD_{t}.json"), "forward_lon": _fw(pkg / "raw/a0k0_lon" / f"FORWARD_LON_{t}.json"),
                    "forward_retime": _fw(pkg / "raw/a0k0_retime" / f"FORWARD_RETIME_{t}.json"),
                    "forward_trainfit": _fw(pkg / "raw/retime_trainfit" / f"FORWARD_TRAINFIT_{t}.json", also=("const", "ALL_PASS")),
                    "heading_drift": present(pkg / "raw/heading_drift" / (f"HEADING_DRIFT_step{s}ema.json" if t == te else f"HEADING_DRIFT_step{s}raw.json"))}
    kd = jl(pkg / "raw/kdraw" / te / "KDRAW_E1.json")
    cb = jl(pkg / "raw/combo_ens_retime" / f"{te}_train" / f"COMBO_{te}_train.json")
    extra = {"kdraw_rolls": {f"dump_s{k}": ("ok" if _g(jl(kit / "kdraw" / te / f"dump_s{k}" / "KDRAW_VERIFY.json"), ("ok",)) is True else "NOT ok / absent") for k in range(2, 8)},
             "kdraw_control": present(kit / "kdraw" / te / "KDRAW_CONTROL.json"), "kdraw_E1": kd if isinstance(kd, str) else {"overall": kd.get("overall"), "void_controls_failed": kd.get("void_controls_failed"), "status": kd.get("status")},
             "combo": cb if isinstance(cb, str) else {"status": cb.get("status"), "void_controls_failed": cb.get("void_controls_failed")}, "READ_FINAL": present(pkg / "raw" / f"step{s}" / "READ_FINAL.md"),
             "trainfit_gains": present(pkg / "raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json")}
    lines = last_markers(Path(a.chain_log))
    stages = {n: stage_line(lines, alts) for n, alts in STAGES}
    res = {"tool": "final_index_r8.py", "generated": time.strftime("%FT%T"), "step": s, "tier": "T1 open loop on logged eval139 frames (battery); see each artifact for its own tier stamp", "weights": weights, "stages_last_marker": stages, "trees": trees, **extra}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_INDEX.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    L = [f"# FINAL_INDEX -- refcv8 step {s}  (generated {res['generated']})", "", "Reads artifacts and quotes their own status; computes no result. UNREADABLE = exists but does not parse; ABSENT = not there.", "", "## Weights (md5 chain)", ""]
    L += [f"* `{k}`: {v}" for k, v in weights.items()]
    L += ["", "## Stages (last marker after the chain's last START)", ""] + [f"* **{n}**: `{v}`" for n, v in stages.items()]
    L += ["", "## Trees", ""]
    for t, v in trees.items():
        L.append(f"### {t}")
        L += [f"* {k}: {v_}" for k, v_ in v.items()]
        L.append("")
    L += ["## K-draw / combo / train-fit gains", ""] + [f"* {k}: {v}" for k, v in extra.items()]
    (out / "FINAL_INDEX.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"WROTE {out / 'FINAL_INDEX.json'}")
    return 0


def _g(j, keys):
    if isinstance(j, str):
        return j
    for k in (keys if isinstance(keys, (tuple, list)) else (keys,)):
        if isinstance(k, (tuple, list)):
            return _g(j, k)
        if not isinstance(j, dict) or k not in j:
            return "MISSING"
        j = j[k]
    return j


def _fw(p: Path, also=None):
    j = jl(p)
    if isinstance(j, str):
        return j
    r = {"controls_ALL_PASS": _g(j, ("controls", "ALL_PASS"))}
    if also:
        r["/".join(also)] = _g(j, also)
    return r


if __name__ == "__main__":
    raise SystemExit(main())
