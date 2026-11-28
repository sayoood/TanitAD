"""SPEC AMENDMENT A4: the zero-training LEVER PANEL (reported, no bar), CPU only.

usage: python lever_panel.py <tag_dir>           e.g. C:/Users/Admin/ev6_battery/raw/step5000

Reads the battery's own panels (`panel_s0`, `panel_s1`: refcv6 `os`, the model-free controls and
`g` on identical windows) and writes `<tag_dir>/levers/levers.json` + `LEVERS.md`:

* L1  `os_avg = (os_s0 + os_s1) / 2`. Its gain over the single seeds is GUARANTEED in sign by the
      triangle inequality; only the magnitude is information (it prices the sampling term).
* L2  `blend_k = w_k os_k + (1 - w_k) ha_k` per instant, w_k on a 0.05 grid fit on one
      episode-parity fold and scored on the other (2-fold cross-fit), per inference seed; with an
      identity control (w = 0 must read exactly 0.0) and a shuffled-plan control (`os` of the
      window N/2 later -> the shrinkage-only gain).
* L2e the same with `ha0_ext` (the echo) as the base: DIAGNOSTIC ONLY (k0 at t0 admissibility is
      unruled at inference) -- "does refcv6 carry information the echo lacks?".

Every cell: paired episode-cluster bootstrap (taniteval.ci), n_boot 2000, seed 0, cluster = clip,
ADE per window from `refav1_arm._components` (the battery's own instrument), n printed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402

GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
N_BOOT, SEED = 2000, 0


def _ci():
    from taniteval import ci
    return ci


def ade(Pp: np.ndarray, G: np.ndarray, dt: float) -> np.ndarray:
    ra = P.RR.ra3().ra
    return np.asarray(ra._components(Pp, G, dt)["ade_m"], dtype=np.float64)


def paired(b_minus_a_b: np.ndarray, a: np.ndarray, eid) -> dict:
    """b − a, the same estimator and argument order as `_paired_families`."""
    return _ci().paired_episode_cluster_bootstrap(b_minus_a_b, a, list(eid), n_boot=N_BOOT, seed=SEED)


def fit_w(os_: np.ndarray, base: np.ndarray, G: np.ndarray, rows: np.ndarray, grid=GRID) -> list:
    """Per instant k: the grid w minimising the mean L2 error on `rows` only."""
    ws = []
    for k in range(G.shape[1]):
        err = [float(np.linalg.norm(w * os_[rows, k] + (1 - w) * base[rows, k] - G[rows, k], axis=-1).mean())
               for w in grid]
        ws.append(float(grid[int(np.argmin(err))]))
    return ws


def cross_fit_blend(os_: np.ndarray, base: np.ndarray, G: np.ndarray, eid: np.ndarray, grid=GRID):
    """2-fold episode-disjoint cross-fit: fold = episode index parity; each fold is scored with the
    OTHER fold's weights, so no window is scored with a weight fit on itself."""
    fold = (np.asarray(eid) % 2).astype(int)
    w = {f: fit_w(os_, base, G, np.where(fold == f)[0], grid) for f in (0, 1)}
    out = np.full_like(os_, np.nan)
    for f in (0, 1):
        rows = np.where(fold == f)[0]
        wf = np.asarray(w[1 - f], dtype=os_.dtype)[None, :, None]      # the OTHER fold's fit
        out[rows] = wf * os_[rows] + (1 - wf) * base[rows]
    return out, {"w_fit_on_fold0_scores_fold1": w[0], "w_fit_on_fold1_scores_fold0": w[1]}


def shuffled(os_: np.ndarray) -> np.ndarray:
    return np.roll(os_, os_.shape[0] // 2, axis=0)


def lever_l2(A: dict, eid: np.ndarray, dt: float, base_key: str) -> dict:
    G, os_, base = A["g"], A["os"], A[base_key]
    blend, wmeta = cross_fit_blend(os_, base, G, eid)
    blend_sh, wmeta_sh = cross_fit_blend(shuffled(os_), base, G, eid)
    a_bl, a_sh = ade(blend, G, dt), ade(blend_sh, G, dt)
    a_base, a_echo = ade(base, G, dt), ade(A["ha0_ext"], G, dt)
    ident_pred, _ = cross_fit_blend(os_, base, G, eid, grid=np.array([0.0]))   # the WHOLE pipeline at w = 0
    ident = ade(ident_pred, G, dt)
    idc = paired(ident, a_base, eid)
    return {"base": base_key, "weights": wmeta, "weights_shuffled_control": wmeta_sh,
            "identity_control_w0_minus_base": idc,
            "identity_control_pass": bool(np.array_equal(ident_pred, base) and idc["delta"] == 0.0
                                          and idc["lo"] == 0.0 and idc["hi"] == 0.0),
            "blend_minus_ha0_ext": paired(a_bl, a_echo, eid),
            "blend_minus_base": paired(a_bl, a_base, eid),
            "blend_minus_blend_shuf": paired(a_bl, a_sh, eid),
            "blend_shuf_minus_base": paired(a_sh, a_base, eid),
            "ade_mean": {"blend": float(a_bl.mean()), "blend_shuf": float(a_sh.mean()),
                         "base": float(a_base.mean()), "ha0_ext": float(a_echo.mean())}}


def main():
    root = Path(sys.argv[1])
    seeds = sorted(int(p.name.split("_s")[-1]) for p in root.glob("panel_s*"))
    panels = {s: P.load_panel(str(root / f"panel_s{s}")) for s in seeds}
    man0, A0, eid0 = panels[seeds[0]]
    dt = float(man0["grid"]["dt_s"])
    for s in seeds[1:]:
        _, As, eids = panels[s]
        if not (np.array_equal(eids, eid0) and np.array_equal(As["g"], A0["g"])
                and all(np.array_equal(As[k], A0[k]) for k in ("ha", "ha0", "ha0_ext"))):
            raise SystemExit(f"[levers] panel_s{s} is not the same windows / controls as panel_s{seeds[0]}")
    out = {"tool": "lever_panel.py", "amendment": "A4", "tag_dir": str(root),
           "spec_sha256": __import__("hashlib").sha256((HERE.parent / "SPEC.md").read_bytes()).hexdigest(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": seeds, "dt_s": dt,
           "n_windows": int(len(eid0)), "n_episodes": int(len(set(eid0.tolist()))),
           "tier": "T1 (self-action open loop); every lever is a DIFFERENT planner from the registered arm",
           "estimator": "paired episode-cluster bootstrap, n_boot 2000, seed 0, cluster = clip"}
    G = A0["g"]
    a_echo = ade(A0["ha0_ext"], G, dt)
    if len(seeds) >= 2:
        o0, o1 = panels[seeds[0]][1]["os"], panels[seeds[1]][1]["os"]
        avg = 0.5 * (o0 + o1)
        a0, a1, aa = ade(o0, G, dt), ade(o1, G, dt), ade(avg, G, dt)
        dpath = np.linalg.norm(o0 - o1, axis=-1)                     # [N, K]
        out["L1_seed_average"] = {
            "os_avg_minus_ha0_ext": paired(aa, a_echo, eid0),
            "os_avg_minus_os_s0": paired(aa, a0, eid0),
            "os_avg_minus_os_s1": paired(aa, a1, eid0),
            "replicate_floor_os_s0_minus_os_s1": paired(a0, a1, eid0),
            "path_diff_s0_s1_m": {"mean_per_instant": dpath.mean(0).round(4).tolist(),
                                  "p95_per_instant": np.percentile(dpath, 95, axis=0).round(4).tolist(),
                                  "instants_s": [0.5, 1.0, 1.5, 2.0]},
            "note": "sign of os_avg - os_s* is guaranteed <= 0 in expectation of per-window ADE by the "
                    "triangle inequality; only the magnitude is information",
            "ade_mean": {"os_avg": float(aa.mean()), "os_s0": float(a0.mean()), "os_s1": float(a1.mean()),
                         "ha0_ext": float(a_echo.mean())}}
    else:
        out["L1_seed_average"] = {"status": "NOT RUN", "reason": "fewer than two inference seeds rolled"}
    out["L2_causal_hold_blend"] = {f"seed{s}": lever_l2(panels[s][1], panels[s][2], dt, "ha") for s in seeds}
    out["L2e_echo_blend_DIAGNOSTIC"] = {f"seed{s}": lever_l2(panels[s][1], panels[s][2], dt, "ha0_ext")
                                        for s in seeds}
    out["identity_controls_all_pass"] = all(
        v["identity_control_pass"] for blk in (out["L2_causal_hold_blend"], out["L2e_echo_blend_DIAGNOSTIC"])
        for v in blk.values())
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    d = root / "levers"
    d.mkdir(exist_ok=True)
    json.dump(out, open(d / "levers.json", "w", encoding="utf-8"), indent=1)
    (d / "LEVERS.md").write_text(render(out), encoding="utf-8")
    print(render(out))
    if not out["identity_controls_all_pass"]:
        raise SystemExit("[levers] an identity control did not read exactly 0 -- the panel is VOID")


def _c(x: dict) -> str:
    if not x or "delta" not in x:
        return "—"
    return f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' **sep**' if x['separated'] else ' ns'}"


def render(o: dict) -> str:
    L = [f"### A4 lever panel — {o['n_windows']} windows / {o['n_episodes']} episodes, S2 ADE 0–2 s, "
         f"{o['tier']}\n", f"Estimator: {o['estimator']}. SPEC sha `{o['spec_sha256'][:12]}…`.\n"]
    l1 = o.get("L1_seed_average") or {}
    if "os_avg_minus_ha0_ext" in l1:
        L += ["**L1 inference-seed average** (sign vs single seeds guaranteed; magnitude only)\n",
              "| cell | ADE m [CI] |", "|---|---|",
              f"| os_avg − ha0_ext | {_c(l1['os_avg_minus_ha0_ext'])} |",
              f"| os_avg − os_s0 | {_c(l1['os_avg_minus_os_s0'])} |",
              f"| os_avg − os_s1 | {_c(l1['os_avg_minus_os_s1'])} |",
              f"| replicate floor os_s0 − os_s1 | {_c(l1['replicate_floor_os_s0_minus_os_s1'])} |",
              f"\n|os_s0 − os_s1| per instant (0.5/1/1.5/2 s): mean {l1['path_diff_s0_s1_m']['mean_per_instant']}, "
              f"p95 {l1['path_diff_s0_s1_m']['p95_per_instant']} m\n"]
    else:
        L += [f"**L1**: {l1.get('status')} ({l1.get('reason')})\n"]
    for key, title in (("L2_causal_hold_blend", "L2 causal-hold blend (base `ha`)"),
                       ("L2e_echo_blend_DIAGNOSTIC", "L2e echo blend (base `ha0_ext`, DIAGNOSTIC ONLY)")):
        L += [f"**{title}**\n", "| seed | blend − ha0_ext | blend − base | blend − blend_shuf | "
              "blend_shuf − base | w (fold0→1 / fold1→0) | identity |", "|---|---|---|---|---|---|---|"]
        for s, v in o[key].items():
            w = v["weights"]
            L.append(f"| {s} | {_c(v['blend_minus_ha0_ext'])} | {_c(v['blend_minus_base'])} | "
                     f"{_c(v['blend_minus_blend_shuf'])} | {_c(v['blend_shuf_minus_base'])} | "
                     f"{w['w_fit_on_fold0_scores_fold1']} / {w['w_fit_on_fold1_scores_fold0']} | "
                     f"{'PASS' if v['identity_control_pass'] else 'FAIL'} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
