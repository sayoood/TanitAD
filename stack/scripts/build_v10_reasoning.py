"""v10 `reasoning`: teacher reasoning text per clip, aligned to the v9 row clock (spec `Paper/HiCAP/design/HICAP_V10_SPEC.md` §3.3, §4).

Sources (both read with `tanitad.hicap.reasoning.parse_coc`):
  * corpus `alpamayo/records.parquet` -- nvidia/Alpamayo2-Super outputs, ONE anchor per clip (t0_us = 5,100,000):
      task meta_action  -> kind `cot` + kind `meta_action`  (source `alpamayo2_super/meta_action`)
      task auto_labeling-> kind `coc` (chain_of_causation)  (source `alpamayo2_super/auto_labeling`)
      task trajectory   -> kind `cot`                       (source `alpamayo2_super/trajectory`; a separate generation, it differs
                                                             from the meta_action CoT on most clips)
    vqa / grounding_via_vqa are not explanation text and are not carried.
  * upstream `reasoning/ood_reasoning.parquet` -- per-event chain-of-causation (`events` JSON); only clips of our corpus (clip index or
    records) are carried (kind `ood_event`, t0_us = event_start_timestamp, source `nvidia_ood_reasoning/<upstream split>`).

Clock: v9 row j of a clip has NOW = `row__now_s[row0 + j]` (V9_SCHEMA.md; sidecar camera timestamps, clip-local seconds); Alpamayo /
OOD timestamps are clip-local microseconds of the same upstream clock (upstream README: the local frame's origin is timestamp 0).
v9_row = argmin_j |now_s[j] - t0_us/1e6| if that distance <= dt/2, else -1 with `v9_reason`. `v9_row` is the WITHIN-CLIP row j
(v9 row j = episode row j); `v9_row_global` indexes the npz arrays.

Also MEASURED here (`meta_action_vs_v9.json`): agreement of the teacher's meta_action Longitudinal / Lane / Lateral text with the v9
tactical label at the aligned row, through the explicit keyword table `reasoning.META_*` (primary class + compatible set), with
confusions, n, chance agreement from the marginals, Cohen's kappa, a clip-bootstrap 95 % CI (one anchor per clip, so the clip IS the
cluster) and a row-offset sweep (diagnostic only; the headline is offset 0, fixed in advance).

    PYTHONPATH=stack python stack/scripts/build_v10_reasoning.py --probe /tmp/claude-0/probe --v9 /tmp/claude-0/hfdata --out DataEng/v10/reasoning
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from collections import Counter
import numpy as np
import pandas as pd

from tanitad.hicap.reasoning import parse_coc, meta_to_v9

LAT9 = ["LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R"]
LON9 = ["HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP"]
V9_KEYS = ["row__now_s", "row__lat_cls_a", "row__lat_cls_b", "row__lat_allowed_a", "row__lat_allowed_b", "row__lon_cls",
           "row__lon_allowed", "row__reversing"]
SWEEP = list(range(-40, 41, 5))            # rows (0.1 s) relative to the aligned row
N_BOOT, SEED = 2000, 0


def sha12(u: str) -> str: return hashlib.sha256(u.encode()).hexdigest()[:12]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


class V9:
    def __init__(self, path: str, split: str):
        z = np.load(path, allow_pickle=True)
        self.split = split
        self.sha = [str(s) for s in z["clip__sha12"]]
        self.idx = {s: i for i, s in enumerate(self.sha)}
        self.row0, self.n, self.dt, self.src = z["clip__row0"], z["clip__n_rows"], z["clip__dt_s"], z["clip__clock_src"]
        self.r = {k[5:]: z[k] for k in V9_KEYS}

    def align(self, s: str, t_s: float):
        i = self.idx[s]; r0, n = int(self.row0[i]), int(self.n[i])
        now = self.r["now_s"][r0:r0 + n]; j = int(np.argmin(np.abs(now - t_s))); d = float(now[j] - t_s)
        if abs(d) > 0.5 * float(self.dt[i]) + 1e-9:
            why = "t_before_first_row" if t_s < now[0] else "t_after_last_row" if t_s > now[-1] else "no_row_within_half_dt"
            return -1, -1, d, why
        return j, r0 + j, d, ("ok" if int(self.src[i]) == 0 else f"ok_clock_src{int(self.src[i])}")


def allowed_set(bits: int, cls: int, names) -> set:
    s = {names[k] for k in range(len(names)) if (int(bits) >> k) & 1}
    return s if s else ({names[cls]} if cls >= 0 else set())


def kappa(y_t, y_v, labels):
    y_t, y_v = np.asarray(y_t), np.asarray(y_v); n = len(y_t)
    if n == 0: return float("nan"), float("nan")
    po = float(np.mean(y_t == y_v))
    pe = float(sum(np.mean(y_t == c) * np.mean(y_v == c) for c in labels))
    return pe, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0: return [float("nan")] * 2
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def build(probe: str, v9dir: str, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    p_rec, p_ood = os.path.join(probe, "records.parquet"), os.path.join(probe, "ood_reasoning.parquet")
    p_ci = {s: os.path.join(probe, f"clip_index_v7.2_{s}.json") for s in ("train", "eval")}
    p_v9 = {"train": os.path.join(v9dir, "v9_labels_train.npz"), "eval139": os.path.join(v9dir, "v9_labels_eval139.npz")}
    ci_split = {}
    for s, p in p_ci.items():
        for u in json.load(open(p))["clips"]: ci_split[u] = s
    v9 = {s: V9(p, s) for s, p in p_v9.items()}
    v9_of = {sh: v for v in v9.values() for sh in v.sha}

    rec = pd.read_parquet(p_rec)
    rec = rec[rec.task.isin(["meta_action", "auto_labeling", "trajectory"])]
    ood = pd.read_parquet(p_ood)
    if "clip_id" not in ood.columns: ood = ood.reset_index()
    corpus = set(ci_split) | set(rec.clip_id)

    def where(u: str, t_us: int):
        sh = sha12(u); v = v9_of.get(sh)
        if v is not None:
            j, g, d, why = v.align(sh, t_us / 1e6); return v.split, sh, j, g, d, why
        split = {"train": "train_not_v9", "eval": "eval_not_v9"}.get(ci_split.get(u), "not_in_clip_index")
        return split, sh, -1, -1, float("nan"), ("clip_not_in_v9" if u in ci_split else "clip_not_in_clip_index")

    rows = []
    for u, t0, task, raw in zip(rec.clip_id, rec.t0_us, rec.task, rec.raw_json):
        split, sh, j, g, d, why = where(u, int(t0))
        for it in parse_coc(raw, task):
            rows.append(dict(split=split, sha12=sh, clip_id=u, t0_us=int(t0), v9_row=j, v9_row_global=g, v9_dt_align_s=d, v9_reason=why,
                             source=f"alpamayo2_super/{task}", kind=it.kind, text=it.text, event_cluster=None, event_start_frame=-1))
    n_ood_dropped = Counter()
    for u, ev, cl, us in zip(ood.clip_id, ood.events, ood.event_cluster, ood.split):
        items = parse_coc(ev)
        if u not in corpus: n_ood_dropped["clip_not_in_corpus"] += len(items); continue
        if not items: n_ood_dropped["null_events"] += 1
        for it in items:
            t = int(it.meta["event_start_timestamp_us"])
            split, sh, j, g, d, why = where(u, t)
            rows.append(dict(split=split, sha12=sh, clip_id=u, t0_us=t, v9_row=j, v9_row_global=g, v9_dt_align_s=d, v9_reason=why,
                             source=f"nvidia_ood_reasoning/{us}", kind="ood_event", text=it.text, event_cluster=str(cl),
                             event_start_frame=int(it.meta["event_start_frame"])))
    df = pd.DataFrame(rows).sort_values(["split", "sha12", "t0_us", "source", "kind"], kind="stable").reset_index(drop=True)
    df = df.astype(dict(v9_row="int32", v9_row_global="int64", t0_us="int64", event_start_frame="int32"))
    p_out = os.path.join(out, "v10_coc.parquet")
    df.to_parquet(p_out, index=False)

    agree = measure(df, v9)
    p_agree = os.path.join(out, "meta_action_vs_v9.json")
    json.dump(agree, open(p_agree, "w"), indent=1)

    try: head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    except Exception: head = None
    man = dict(
        schema="tanitad.v10_reasoning/1",
        columns=dict(split="train | eval139 (v9 membership); train_not_v9 | eval_not_v9 (clip index only); not_in_clip_index",
                     sha12="sha256(clip_id)[:12]", clip_id="upstream UUID", t0_us="anchor / event start, clip-local microseconds",
                     v9_row="within-clip v9 row j aligned to t0 (|now_s - t0| <= dt/2) or -1", v9_row_global="row index into the v9 npz or -1",
                     v9_dt_align_s="now_s[v9_row] - t0 (s)", v9_reason="ok | ok_clock_src<k> (non-sidecar clock) | clip_not_in_v9 | "
                     "clip_not_in_clip_index | t_before_first_row | t_after_last_row | no_row_within_half_dt",
                     source="alpamayo2_super/<task> | nvidia_ood_reasoning/<upstream split>", kind="cot | coc | meta_action | ood_event",
                     text="teacher text (INHERITED quality, not ground truth); special tokens stripped",
                     event_cluster="OOD event cluster (ood rows only)", event_start_frame="OOD event frame (ood rows only, else -1)"),
        n_rows=int(len(df)), n_clips=int(df.clip_id.nunique()),
        counts_by_kind_source={f"{s}|{k}": int(n) for (s, k), n in df.groupby(["source", "kind"]).size().items()},
        counts_by_split={k: int(v) for k, v in df.split.value_counts().items()},
        clips_by_split={k: int(v) for k, v in df.groupby("split").clip_id.nunique().items()},
        v9_reason={k: int(v) for k, v in df.v9_reason.value_counts().items()},
        ood=dict(upstream_clips=int(len(ood)), upstream_null_events=int(ood.events.isna().sum()),
                 clips_carried=int(df[df.kind == "ood_event"].clip_id.nunique()), events_carried=int((df.kind == "ood_event").sum()),
                 events_dropped=dict(n_ood_dropped)),
        records=dict(clips=int(rec.clip_id.nunique()), t0_us=sorted(int(x) for x in rec.t0_us.unique()),
                     clips_in_clip_index=int(len(set(rec.clip_id) & set(ci_split))), model_id=sorted(rec.model_id.unique().tolist()),
                     quantisation=sorted(rec.quantisation.unique().tolist())),
        inputs_sha256={os.path.basename(p): sha256_file(p) for p in [p_rec, p_ood, *p_ci.values(), *p_v9.values()]},
        outputs_sha256={"v10_coc.parquet": sha256_file(p_out), "meta_action_vs_v9.json": sha256_file(p_agree)},
        builder=dict(path="stack/scripts/build_v10_reasoning.py", sha256=sha256_file(__file__),
                     parser_sha256=sha256_file(os.path.join(os.path.dirname(__file__), "..", "tanitad", "hicap", "reasoning.py")),
                     base_commit=head),
        headline=agree["headline"],
    )
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


def measure(df: pd.DataFrame, v9: dict) -> dict:
    """Teacher meta_action text vs the v9 tactical label at the aligned row (and at row offsets, diagnostic)."""
    rng = np.random.default_rng(SEED)
    m = df[(df.kind == "meta_action") & (df.v9_row >= 0)].copy()
    excl = Counter({f"meta_action_rows_unaligned:{k}": int(v) for k, v in df[(df.kind == "meta_action") & (df.v9_row < 0)].v9_reason.value_counts().items()})
    from tanitad.hicap.reasoning import parse_meta_action

    def labels_at(v, g):
        R = v.r
        return dict(lon=int(R["lon_cls"][g]), lon_bits=int(R["lon_allowed"][g]), lat_a=int(R["lat_cls_a"][g]), lat_a_bits=int(R["lat_allowed_a"][g]),
                    lat_b=int(R["lat_cls_b"][g]), lat_b_bits=int(R["lat_allowed_b"][g]), rev=int(R["reversing"][g]))

    res = {}
    for scope in ("all", "train", "eval139"):
        sub = m if scope == "all" else m[m.split == scope]
        out = dict(lon=dict(strict=[], partial=[], compatible=[], t=[], v=[], conf=Counter(), excluded=Counter()))
        for var in ("a", "b"):
            out[f"lat_{var}"] = dict(strict=[], partial=[], t=[], v=[], conf=Counter(), excluded=Counter(), conf_lateral_line=Counter())
        for sh, split, g, text in zip(sub.sha12, sub.split, sub.v9_row_global, sub.text):
            v = v9[split]; L = labels_at(v, int(g))
            f = parse_meta_action(text.replace(". ", ".\n")); mv = meta_to_v9(f)
            # ---- longitudinal
            o = out["lon"]; lon_txt = f.get("Longitudinal", "<missing>"); prim, comp = mv["lon"]
            if L["rev"]: o["excluded"]["v9_reversing"] += 1
            elif L["lon"] < 0: o["excluded"]["v9_lon_ignore"] += 1; o["conf"][(lon_txt, "IGNORE")] += 1
            elif prim is None: o["excluded"][f"text_unmapped:{lon_txt}"] += 1; o["conf"][(lon_txt, LON9[L['lon']])] += 1
            else:
                al = allowed_set(L["lon_bits"], L["lon"], LON9); vc = LON9[L["lon"]]
                o["strict"].append(prim == vc); o["partial"].append(prim in al); o["compatible"].append(bool(comp & al))
                o["t"].append(prim); o["v"].append(vc); o["conf"][(lon_txt, vc)] += 1
            # ---- lateral (variants a = junction DEFAULT, b = heading)
            pl, cl, line = mv["lat"]
            for var in ("a", "b"):
                o = out[f"lat_{var}"]; c, bits = L[f"lat_{var}"], L[f"lat_{var}_bits"]
                lane_txt = f.get("Lane", "<missing>"); lat_line = f.get("Lateral", "<missing>")
                if L["rev"]: o["excluded"]["v9_reversing"] += 1; continue
                if c < 0: o["excluded"]["v9_lat_ignore"] += 1; o["conf"][(lane_txt, "IGNORE")] += 1; continue
                vc = LAT9[c]; o["conf_lateral_line"][(lat_line, vc)] += 1
                if pl is None: o["excluded"][f"text_unmapped:{lane_txt}/{lat_line}"] += 1; continue
                o["conf"][(lane_txt if line == "Lane" else f"[Lateral] {lat_line}", vc)] += 1
                al = allowed_set(bits, c, LAT9)
                o["strict"].append(pl == vc); o["partial"].append(pl in al); o["t"].append(pl); o["v"].append(vc)
        res[scope] = {}
        for ax, o in out.items():
            names = LON9 if ax == "lon" else LAT9
            pe, k = kappa(o["t"], o["v"], names)
            r = dict(n=len(o["strict"]), excluded=dict(o["excluded"]),
                     strict=dict(rate=float(np.mean(o["strict"])) if o["strict"] else float("nan"), ci95_clip_bootstrap=boot_ci(o["strict"], rng)),
                     partial_label=dict(rate=float(np.mean(o["partial"])) if o["partial"] else float("nan"), ci95_clip_bootstrap=boot_ci(o["partial"], rng)),
                     chance_strict_from_marginals=pe, cohen_kappa_strict=k,
                     text_marginal={c: int(n) for c, n in Counter(o["t"]).items()}, v9_marginal={c: int(n) for c, n in Counter(o["v"]).items()},
                     confusion={f"{a} -> {b}": int(n) for (a, b), n in sorted(o["conf"].items())})
            if ax == "lon": r["compatible"] = dict(rate=float(np.mean(o["compatible"])) if o["compatible"] else float("nan"),
                                                   ci95_clip_bootstrap=boot_ci(o["compatible"], rng))
            else: r["confusion_lateral_line_diag"] = {f"{a} -> {b}": int(n) for (a, b), n in sorted(o["conf_lateral_line"].items())}
            res[scope][ax] = r

    # row-offset sweep (diagnostic): strict agreement with the label of row j+off of the same clip
    sweep = {}
    for off in SWEEP:
        lo, la = [], []
        for split, sh, j, text in zip(m.split, m.sha12, m.v9_row, m.text):
            v = v9[split]; i = v.idx[sh]; jj = int(j) + off
            if not (0 <= jj < int(v.n[i])): continue
            L = labels_at(v, int(v.row0[i]) + jj)
            if L["rev"]: continue
            mv = meta_to_v9(parse_meta_action(text.replace(". ", ".\n")))
            if L["lon"] >= 0 and mv["lon"][0] is not None: lo.append(mv["lon"][0] == LON9[L["lon"]])
            if L["lat_a"] >= 0 and mv["lat"][0] is not None: la.append(mv["lat"][0] == LAT9[L["lat_a"]])
        sweep[str(off)] = dict(lon_strict=float(np.mean(lo)) if lo else None, n_lon=len(lo), lat_a_strict=float(np.mean(la)) if la else None, n_lat=len(la))
    a = res["all"]
    head = dict(
        lon_strict=a["lon"]["strict"]["rate"], lon_strict_ci95=a["lon"]["strict"]["ci95_clip_bootstrap"], lon_compatible=a["lon"]["compatible"]["rate"],
        lon_chance=a["lon"]["chance_strict_from_marginals"], lon_kappa=a["lon"]["cohen_kappa_strict"], lon_n=a["lon"]["n"],
        lat_a_strict=a["lat_a"]["strict"]["rate"], lat_a_strict_ci95=a["lat_a"]["strict"]["ci95_clip_bootstrap"],
        lat_a_chance=a["lat_a"]["chance_strict_from_marginals"], lat_a_kappa=a["lat_a"]["cohen_kappa_strict"], lat_a_n=a["lat_a"]["n"])
    return dict(what="Alpamayo2-Super meta_action text vs v9 tactical label at the row aligned to t0 = 5.1 s (MEASURED)",
                keyword_table="tanitad.hicap.reasoning.META_LON_TO_LON9 / META_LANE_TO_LAT9 / META_LATERAL_TO_LAT9 (Lane line primary; "
                              "Lateral line only where Lane is absent)",
                definitions=dict(strict="teacher primary class == v9 class", partial_label="teacher primary class in the v9 partial-label "
                                 "allowed set (lon: FOLLOW-undetermined rows allow {class, FOLLOW})", compatible="(lon only) teacher compatible "
                                 "set intersects the v9 allowed set", chance="sum_c p_text(c) p_v9(c) over the scored rows",
                                 ci="percentile bootstrap over clips (one anchor per clip => clip = cluster), 2000 resamples, seed 0",
                                 band="v9 labels describe [NOW+2, NOW+8] s; the teacher's meta action describes its own predicted future from t0"),
                excluded_global=dict(excl), headline=head, by_scope=res, row_offset_sweep_diag=sweep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="/tmp/claude-0/probe")
    ap.add_argument("--v9", default="/tmp/claude-0/hfdata")
    ap.add_argument("--out", default="DataEng/v10/reasoning")
    a = ap.parse_args()
    man = build(a.probe, a.v9, a.out)
    print(json.dumps({k: man[k] for k in ("n_rows", "n_clips", "counts_by_kind_source", "clips_by_split", "v9_reason", "ood", "headline")}, indent=1))
