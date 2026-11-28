"""SPEC A6 step 4: fit the L2 weights on the TRAIN rolls only, score the battery's EVAL panels with NO
refit, at both inference seeds. CPU only.

usage: python a6_fit_score.py <tag_dir> <a6_dir>
  <tag_dir>  the battery tag (panel_s0 / panel_s1: the eval windows, os_s, ha, ha0_ext, g)
  <a6_dir>   raw/a6/<tag>/ holding dump_s0 / dump_s1 from a6_roll.py
Writes <a6_dir>/a6.json + A6.md. Verdict per SPEC A6: SUCCESS (blend - echo < 0, separated, both
seeds) / FAILURE (not separated at either seed) / NOT PROVEN (one seed). Controls must read their
literals or the panel is VOID (exit 2).
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402
import lever_panel as LP  # noqa: E402


def load_dump(d: str):
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    arrs, eid = {}, []
    for fi, f in enumerate(files):
        z = np.load(f)
        eid += [fi] * z["g"].shape[0]
        for k in ("os", "ha", "ha0_ext", "g"):
            arrs.setdefault(k, []).append(z[k])
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    return man, {k: np.concatenate(v) for k, v in arrs.items()}, np.asarray(eid)


def apply(w, os_, base):
    wf = np.asarray(w, dtype=os_.dtype)[None, :, None]
    return wf * os_ + (1 - wf) * base


def main():
    tag, a6 = Path(sys.argv[1]), Path(sys.argv[2])
    out = {"tool": "a6_fit_score.py", "amendment": "A6", "tag_dir": str(tag), "a6_dir": str(a6),
           "spec_sha256": __import__("hashlib").sha256((HERE.parent / "SPEC.md").read_bytes()).hexdigest(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "estimator": "paired episode-cluster bootstrap, n_boot 2000, seed 0, cluster = clip", "per_seed": {}}
    void = []
    fits = {}
    for s in (0, 1):
        dd = a6 / f"dump_s{s}"
        if not (dd / "manifest.json").exists():
            out["per_seed"][s] = {"status": "ABSENT", "dump": str(dd)}
            continue
        man_t, T, eid_t = load_dump(str(dd))
        dt = float(man_t["grid"]["dt_s"])
        rows = np.arange(len(eid_t))
        w = LP.fit_w(T["os"], T["ha"], T["g"], rows)
        w_sh = LP.fit_w(np.roll(T["os"], len(eid_t) // 2, axis=0), T["ha"], T["g"], rows)
        fits[s] = w
        a_tr = {k: float(LP.ade(v, T["g"], dt).mean()) for k, v in
                (("os", T["os"]), ("ha", T["ha"]), ("ha0_ext", T["ha0_ext"]), ("blend", apply(w, T["os"], T["ha"])))}
        man, A, eid = P.load_panel(str(tag / f"panel_s{s}"))
        G = A["g"]
        dte = float(man["grid"]["dt_s"])
        bl = apply(w, A["os"], A["ha"])
        bl_sh = apply(w_sh, A["os"], A["ha"])
        bl_id = apply([0.0] * G.shape[1], A["os"], A["ha"])
        a_bl, a_sh, a_id = LP.ade(bl, G, dte), LP.ade(bl_sh, G, dte), LP.ade(bl_id, G, dte)
        a_echo, a_ha, a_os = LP.ade(A["ha0_ext"], G, dte), LP.ade(A["ha"], G, dte), LP.ade(A["os"], G, dte)
        sh_c, id_c = LP.paired(a_sh, a_ha, eid), LP.paired(a_id, a_ha, eid)
        ctl = {"shuffled_w_all_zero": all(x == 0.0 for x in w_sh),
               "shuffled_blend_minus_ha_exact_zero": bool(sh_c["delta"] == 0.0 and sh_c["lo"] == 0.0 and sh_c["hi"] == 0.0),
               "identity_blend_equals_ha_bitwise": bool(np.array_equal(bl_id, A["ha"])),
               "identity_blend_minus_ha_exact_zero": bool(id_c["delta"] == 0.0 and id_c["lo"] == 0.0 and id_c["hi"] == 0.0)}
        void += [f"seed {s}: {k}" for k, v in ctl.items() if not v]
        out["per_seed"][s] = {
            "train": {"n_windows": int(len(eid_t)), "n_episodes": int(len(set(eid_t.tolist()))), "ade_mean": a_tr},
            "w_train_fit": w, "w_shuffled_control": w_sh,
            "eval": {"n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
                     "blend_minus_ha0_ext": LP.paired(a_bl, a_echo, eid),
                     "blend_minus_ha": LP.paired(a_bl, a_ha, eid),
                     "blend_minus_os": LP.paired(a_bl, a_os, eid),
                     "shuffled_blend_minus_ha": sh_c, "identity_blend_minus_ha": id_c},
            "controls": ctl}
    lv = tag / "levers" / "levers.json"
    if lv.exists():
        l2 = json.load(open(lv, encoding="utf-8")).get("L2_causal_hold_blend") or {}
        out["A4_eval_crossfit_w"] = {k: v.get("weights") for k, v in l2.items()}
    if 0 in fits and 1 in fits:
        man1, A1, eid1 = P.load_panel(str(tag / "panel_s1"))
        a_x = LP.ade(apply(fits[0], A1["os"], A1["ha"]), A1["g"], float(man1["grid"]["dt_s"]))
        out["cross_seed_w0_on_eval_s1_minus_ha0_ext"] = LP.paired(a_x, LP.ade(A1["ha0_ext"], A1["g"], float(man1["grid"]["dt_s"])), eid1)
    cells = [out["per_seed"][s]["eval"]["blend_minus_ha0_ext"] for s in (0, 1)
             if isinstance(out["per_seed"].get(s), dict) and "eval" in out["per_seed"][s]]
    passes = [bool(c["delta"] < 0 and c["hi"] < 0) for c in cells]
    out["verdict"] = ("VOID (" + "; ".join(void) + ")" if void else
                      "INCOMPLETE (fewer than two seeds)" if len(cells) < 2 else
                      "SUCCESS" if all(passes) else "FAILURE" if not any(passes) else "NOT PROVEN (one seed)")
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(out, open(a6 / "a6.json", "w", encoding="utf-8"), indent=1, default=str)
    md = [f"### SPEC A6: deployable L2 (w fit on TRAIN, eval scored with no refit) -- {tag.name}: **{out['verdict']}**", "",
          "| seed | train n (eps) | w (train fit) | eval blend − echo | blend − ha | blend − os | shuffled w | controls |",
          "|---|---|---|---|---|---|---|---|"]
    for s, v in out["per_seed"].items():
        if "eval" not in v:
            md.append(f"| {s} | {v.get('status')} | | | | | | |")
            continue
        e = v["eval"]
        md.append(f"| {s} | {v['train']['n_windows']} ({v['train']['n_episodes']}) | {v['w_train_fit']} | "
                  f"{LP._c(e['blend_minus_ha0_ext'])} | {LP._c(e['blend_minus_ha'])} | {LP._c(e['blend_minus_os'])} | "
                  f"{v['w_shuffled_control']} | {'PASS' if all(v['controls'].values()) else 'FAIL'} |")
    if "cross_seed_w0_on_eval_s1_minus_ha0_ext" in out:
        md.append(f"\nCross-seed (seed-0 train w on the seed-1 eval panel) blend − echo: "
                  f"{LP._c(out['cross_seed_w0_on_eval_s1_minus_ha0_ext'])}")
    if "A4_eval_crossfit_w" in out:
        md.append(f"\nA4 eval cross-fit w for comparison: {out['A4_eval_crossfit_w']}")
    md.append("\nKnown bias (stated in A6 before any number): the model trained on these clips, so a train-fit w "
              "may over-weight `os`; A6 is conservative.")
    (a6 / "A6.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if void:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
