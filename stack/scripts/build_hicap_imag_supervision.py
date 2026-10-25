#!/usr/bin/env python3
"""Build the HiCAP v10 imagination supervision (`v10_imag_<split>.npz`) aligned ROW FOR ROW to a v9 label split.

Producer: `tanitad.hicap.supervision` (schema, conventions and the non-reactive caveat are in its docstring).
Inputs (local paths; fetch them from HF `Sayood/tanitad-v7-training-corpus` first, e.g. `huggingface_hub.snapshot_download(...,
allow_patterns=["agents/*", "egomotion/*"])` — 2.4 GB of agent shards + 1.97 GB of egomotion):
  --v9            v9_labels_<split>.npz (clip table: sha12, row0, n_rows, k0, grid_start_s, dt_s)
  --clip-index    one or more index/clip_index_v7.2_<split>.json (UUID keys; sha12 = sha256(uuid)[:12])
  --agents-dir    directory with obstacle_offline_NNN.parquet + index.parquet (agents/obstacle_offline/)
  --ego-tar       egomotion/egomotion_alpamayo.tar (members <clip_id>.parquet; an offset index is cached next to --out)
Output: --out v10_imag_<split>.npz with every row array of length = the v9 split's row count (row r here == v9 row r; clips not built
have exo_w = J_w = 0), the clip table (`clip__sha12`, `clip__row0`, `clip__n_rows`, `clip__built`, `clip__n_tracks`), and --stats JSON
with MEASURED coverage / v9 agreement / mode distribution / timing.
Parity: read-only over the corpus; never re-selects episodes (clips are taken in v9 order, optionally the first --limit with agents).

Example (CPU):
  PYTHONPATH=stack python stack/scripts/build_hicap_imag_supervision.py --v9 v9_labels_eval139.npz \\
     --clip-index clip_index_v7.2_eval.json --agents-dir agents/obstacle_offline --ego-tar egomotion_alpamayo.tar \\
     --out v10_imag_eval139.npz --stats v10_imag_eval139.stats.json --limit 60
Parallel: run N processes with --shard K/N --out part_K.npz, then --merge part_*.npz --out v10_imag_<split>.npz (OMP_NUM_THREADS=1).
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys, tarfile, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tanitad.hicap import supervision as S  # noqa: E402

V9_LEAD_LAG_ROWS = 20          # v9 lead_gap_m is measured at the first band frame, NOW + 2 s = 20 rows later on the ~0.1007 s grid


def sha12(u: str) -> str: return hashlib.sha256(u.encode()).hexdigest()[:12]


def ego_index(tar_path: str, cache: str) -> dict:
    if os.path.exists(cache): return json.load(open(cache))
    with tarfile.open(tar_path) as tf:
        idx = {m.name.split(".")[0]: [m.offset_data, m.size] for m in tf.getmembers() if m.isfile()}
    json.dump(idx, open(cache, "w")); return idx


def load_ego(fh, idx: dict, uuid: str):
    import pandas as pd
    if uuid not in idx: return None
    o, n = idx[uuid]; fh.seek(o)
    return S.EgoTrack.from_columns(pd.read_parquet(io.BytesIO(fh.read(n))))


class AgentReader:
    COLS = ["timestamp_us", "track_id", "center_x", "center_y", "size_x", "size_y", "orientation_x", "orientation_y", "orientation_z",
            "orientation_w", "label_class", "reference_frame"]

    def __init__(self, d: str):
        import pandas as pd
        self.d = d; self.ix = pd.read_parquet(os.path.join(d, "index.parquet")).set_index("clip_id") if os.path.exists(os.path.join(d, "index.parquet")) else None
        self._pf = {}; self._rg = {}

    def has(self, uuid): return self.ix is not None and uuid in self.ix.index

    def _file(self, shard):
        import pyarrow.parquet as pq
        if shard not in self._pf:
            pf = pq.ParquetFile(os.path.join(self.d, shard)); self._pf[shard] = pf
            m = {}
            for g in range(pf.metadata.num_row_groups):
                st = pf.metadata.row_group(g).column(pf.schema_arrow.get_field_index("clip_id")).statistics
                if st is not None and st.has_min_max and st.min == st.max: m[st.min] = g
            if len(m) < sum(pf.metadata.row_group(g).num_rows > 0 for g in range(pf.metadata.num_row_groups)):   # stats missing: read clip_id
                for g in range(pf.metadata.num_row_groups):
                    col = pf.read_row_group(g, columns=["clip_id"]).column(0)
                    if len(col): m[col[0].as_py()] = g
            self._rg[shard] = m
        return self._pf[shard], self._rg[shard]

    def load(self, uuid):
        if not self.has(uuid): return None
        pf, m = self._file(self.ix.loc[uuid, "shard"])
        if uuid not in m:                                              # a source file with ZERO cuboids (6 clips): NO_LABEL, not "empty road"
            return {c: np.zeros(0, object if c in ("track_id", "label_class", "reference_frame") else float) for c in self.COLS}
        t = pf.read_row_group(m[uuid], columns=self.COLS).to_pandas()
        assert (t["reference_frame"] == "rig").all(), f"{uuid}: a box not in the rig frame (schema surprise)"
        return {c: t[c].to_numpy() for c in self.COLS}


ROW_KEYS_FLOAT_NAN = ("lead_gap0", "lead_tg0", "lead_lat0", "exo_cutin_t")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v9", required=True); ap.add_argument("--clip-index", nargs="+", required=True)
    ap.add_argument("--agents-dir", required=True); ap.add_argument("--ego-tar", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N clips (v9 order) that have an agent file; 0 = all")
    ap.add_argument("--vocab", default=None, help="optional .npz with paths/path_cls/profiles/prof_cls/standstill (a TRAIN-split FactoredVocab)")
    ap.add_argument("--n-cand", type=int, default=5)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--stats-only", action="store_true", help="recompute --stats from an existing --out npz")
    ap.add_argument("--shard", default=None, help="K/N: build only clips todo[K::N] (run N processes, then --merge)")
    ap.add_argument("--merge", nargs="+", default=None, help="merge shard npz files (same v9 split) into --out, then write --stats")
    a = ap.parse_args(argv)
    t_start = time.time()
    z = np.load(a.v9, allow_pickle=True); Z = {k: z[k] for k in z.files}
    if a.merge:
        R = None
        for f in a.merge:
            o = np.load(f, allow_pickle=False); P = {k: o[k] for k in o.files}
            if R is None: R = P; continue
            for i in np.flatnonzero(P["clip__built"]):
                assert not R["clip__built"][i], f"clip {i} built in two shards"
                r0, n = int(Z["clip__row0"][i]), int(Z["clip__n_rows"][i])
                for k, v in P.items():
                    if not k.startswith("clip__") and k != "meta": R[k][r0:r0 + n] = v[r0:r0 + n]
                for k in ("clip__built", "clip__n_tracks", "clip__sec"): R[k][i] = P[k][i]
        np.savez_compressed(a.out, **R); a.stats_only = True
    if a.stats_only:
        o = np.load(a.out, allow_pickle=False); R = {k: o[k] for k in o.files}
        st = stats(R, Z, R["clip__built"], list(R["clip__sec"][R["clip__built"] > 0]), json.loads(str(R["meta"])).get("skipped", {}),
                   int(R["clip__built"].sum()))
        st["meta"] = json.loads(str(R["meta"])); json.dump(st, open(a.stats, "w"), indent=1, default=float)
        print(json.dumps({k: st[k] for k in ("n_clips_built", "rows", "lead", "v9_agreement", "mode_dist")}, indent=1, default=float)); return
    shas = [str(s) for s in Z["clip__sha12"]]; n_tot = len(Z["row__k"]) if "row__k" in Z else int(Z["clip__n_rows"].sum())
    uu = {}
    for f in a.clip_index:
        for u in json.load(open(f))["clips"]: uu[sha12(u)] = u
    vocab = None
    if a.vocab:
        from tanitad.hicap.vocab import FactoredVocab
        v = np.load(a.vocab); vocab = FactoredVocab(v["paths"], v["path_cls"], v["profiles"], v["prof_cls"], v["standstill"])
    ar = AgentReader(a.agents_dir)
    eidx = ego_index(a.ego_tar, a.out + ".ego_index.json"); fh = open(a.ego_tar, "rb")
    todo = [i for i, s in enumerate(shas) if s in uu and ar.has(uu[s]) and uu[s] in eidx]
    skipped = {"no_uuid": sum(s not in uu for s in shas), "no_agent_file": sum(s in uu and not ar.has(uu[s]) for s in shas),
               "no_egomotion": sum(s in uu and uu[s] not in eidx for s in shas)}
    if a.limit: todo = todo[:a.limit]
    if a.shard:
        kk, nn = (int(x) for x in a.shard.split("/")); todo = todo[kk::nn]
    out, built, ntr, times, sec = None, np.zeros(len(shas), np.int8), np.zeros(len(shas), np.int32), [], np.zeros(len(shas), np.float32)
    for c_i, i in enumerate(todo):
        t0 = time.time(); u = uu[shas[i]]
        ego = load_ego(fh, eidx, u); obs = ar.load(u)
        r0, n, k0 = int(Z["clip__row0"][i]), int(Z["clip__n_rows"][i]), int(Z["clip__k0"][i])
        R = S.build_rows(ego, obs, float(Z["clip__grid_start_s"][i]), float(Z["clip__dt_s"][i]), k0, n, vocab=vocab, n_cand=a.n_cand,
                         probe=not a.no_probe)
        if out is None:
            out = {}
            for key, v in R.items():
                fill = "" if v.dtype == object else (np.nan if key in ROW_KEYS_FLOAT_NAN else (S.IGNORE if key in ("exo_mode", "exo_mode4") else 0))
                if key.startswith("probe_ttc") or key.startswith("probe_clear"): fill = np.inf
                out[key] = np.full((n_tot,) + v.shape[1:], fill, dtype=v.dtype)
        for key, v in R.items(): out[key][r0:r0 + n] = v
        built[i] = 1; ntr[i] = len(np.unique(obs["track_id"])) if obs is not None else 0
        times.append(time.time() - t0); sec[i] = times[-1]
        if (c_i + 1) % 10 == 0 or c_i + 1 == len(todo):
            print(f"[{c_i + 1}/{len(todo)}] {np.mean(times):.2f} s/clip", flush=True)
    out["lead_tid"] = out["lead_tid"].astype(str)
    clip = dict(clip__sha12=np.array(shas), clip__row0=Z["clip__row0"], clip__n_rows=Z["clip__n_rows"], clip__built=built, clip__n_tracks=ntr, clip__sec=sec)
    meta = dict(schema="tanitad.v10_imag/1", producer="stack/tanitad/hicap/supervision.py", v9=os.path.basename(a.v9),
                constants={k: getattr(S, k) for k in ("H_T", "MODE_ACC", "MODE_NAMES", "MODE4_NAMES", "COST_CHANNELS", "FAMILIES", "CF_MASK",
                                                    "EGO_LEN", "EGO_WID", "EGO_REAR_TO_CENTER", "CORRIDOR_HALF_W", "LEAD_MAX_S", "TTC_THR",
                                                    "CLEAR_THR", "J_WEIGHTS", "MODE_SUSTAIN_S", "LEAD_CLASSES")},
                vocab=a.vocab, n_cand=a.n_cand, skipped=skipped)
    np.savez_compressed(a.out, **out, **clip, meta=np.array(json.dumps(meta, default=list)))
    st = stats(out, Z, built, times, skipped, len(todo))
    st["wall_s_total"] = time.time() - t_start; st["meta"] = meta
    json.dump(st, open(a.stats, "w"), indent=1, default=float)
    print(json.dumps({k: st[k] for k in ("n_clips_built", "rows", "lead", "v9_agreement", "mode_dist")}, indent=1, default=float))


def stats(R, Z, built, times, skipped, n_todo):
    rows = np.zeros(len(R["exo_w"]), bool)
    for i in np.flatnonzero(built): rows[int(Z["clip__row0"][i]):int(Z["clip__row0"][i]) + int(Z["clip__n_rows"][i])] = True
    w = (R["exo_w"] > 0) & rows; hl = (R["exo_has_lead"] > 0) & w
    st = dict(n_clips_built=int(built.sum()), skipped=skipped, sec_per_clip=dict(mean=float(np.mean(times)), median=float(np.median(times)),
              p90=float(np.percentile(times, 90))) if times else None)
    st["rows"] = dict(in_built_clips=int(rows.sum()), exo_w=int(w.sum()), exo_w_frac=float(w.sum() / max(rows.sum(), 1)),
                      J_w=int(((R["J_w"] > 0) & rows).sum()))
    st["lead"] = dict(has_lead_frac_of_labelled=float(hl.sum() / max(w.sum(), 1)), n_has_lead=int(hl.sum()),
                      gap0_median_m=float(np.nanmedian(R["lead_gap0"][hl])) if hl.any() else None,
                      exo_gap_negative_frac=[float((R["exo_gap"][hl, h] < 0).mean()) for h in range(4)] if hl.any() else None,
                      hmask_frac=[float(R["exo_hmask"][hl, h].mean()) for h in range(4)] if hl.any() else None,
                      logsig_median=[float(np.median(R["exo_logsig"][hl, h])) for h in range(4)] if hl.any() else None,
                      cutin_frac_of_labelled=float((R["exo_cutin"][w] > 0).mean()) if w.any() else None,
                      cutout_frac_of_lead=float((R["exo_cutout"][hl] > 0).mean()) if hl.any() else None,
                      lead_class_counts={S.CLASSES[c]: int((R["lead_cls"][hl] == c).sum()) for c in np.unique(R["lead_cls"][hl]) if c >= 0})
    m = R["exo_mode"][hl]; m4 = R["exo_mode4"][hl]
    st["mode_dist"] = dict(exo_mode={S.MODE_NAMES[i]: int((m == i).sum()) for i in range(4)} | {"IGNORE": int((m == S.IGNORE).sum())},
                           exo_mode4={S.MODE4_NAMES[i]: int((m4 == i).sum()) for i in range(4)} | {"IGNORE": int((m4 == S.IGNORE).sum())},
                           onset_median_s_nonhold=float(np.median(R["exo_onset"][hl & (R["exo_mode"] >= 0) & (R["exo_mode"] != 1)])) if (hl & (R["exo_mode"] >= 0) & (R["exo_mode"] != 1)).any() else None)
    # ---- v9 cross-check: v9 row r (gap at NOW + 2 s) vs our row r + 20 (gap at our NOW), same clip, both labelled
    L = V9_LEAD_LAG_ROWS; agree = dict(lag_rows=L)
    vv, vg = Z["row__lead_valid"].astype(int), Z["row__lead_gap_m"].astype(float)
    pr, pg, pw = [], [], []
    for i in np.flatnonzero(built):
        r0, n = int(Z["clip__row0"][i]), int(Z["clip__n_rows"][i])
        a = np.arange(r0, r0 + n - L); b = a + L
        pr.append(a); pg.append(b); pw.append(R["exo_w"][b] > 0)
    if pr:
        a, b, ww = np.concatenate(pr), np.concatenate(pg), np.concatenate(pw)
        a, b = a[ww], b[ww]
        ours, theirs = R["exo_has_lead"][b] > 0, vv[a] == 1
        agree["n_rows_compared"] = int(len(a))
        agree["valid_confusion"] = dict(both=int((ours & theirs).sum()), ours_only=int((ours & ~theirs).sum()), v9_only=int((~ours & theirs).sum()),
                                        neither=int((~ours & ~theirs).sum()))
        agree["valid_agreement"] = float((ours == theirs).mean()) if len(a) else None
        both = ours & theirs & np.isfinite(vg[a]) & np.isfinite(R["lead_gap0"][b])
        d = R["lead_gap0"][b][both] - vg[a][both]
        if both.any():
            agree.update(n_gap_pairs=int(both.sum()), median_abs_delta_m=float(np.median(np.abs(d))), median_signed_delta_m=float(np.median(d)),
                         within_2m=float((np.abs(d) <= 2).mean()), within_2m_after_median_offset=float((np.abs(d - np.median(d)) <= 2).mean()),
                         p90_abs_delta_m=float(np.percentile(np.abs(d), 90)))
            tg = R["lead_tg0"][b][both]; vtg = Z["row__lead_tg_s"][a][both].astype(float); okt = np.isfinite(tg) & np.isfinite(vtg)
            if okt.any():
                agree["tg_median_abs_delta_s"] = float(np.median(np.abs(tg[okt] - vtg[okt])))
                vb = R["v0"][b][both][okt]                    # time gap with our gap moved to v9's reference by the median offset
                agree["tg_median_abs_delta_s_after_offset"] = float(np.median(np.abs((R["lead_gap0"][b][both][okt] - np.median(d)) / np.maximum(vb, 0.5) - vtg[okt])))
    st["v9_agreement"] = agree
    # ---- teacher-cost sanity on the probe set (logged trajectory must almost never collide)
    if "probe_ttc" in R:
        jw = (R["J_w"] > 0) & rows
        names = ("logged", "const_v0", "brake3", "brake6", "acc1.5", "shift+3.5", "shift-3.5", "straight_v0")
        st["probe"] = {nm: dict(collide_frac=float(np.isfinite(R["probe_ttc"][jw, p]).mean()), J_mean=float(R["probe_J"][jw, p].mean()),
                                clear_median_m=float(np.median(np.minimum(R["probe_clear"][jw, p], 99.0))))
                       for p, nm in enumerate(names)} if jw.any() else None
    return st


if __name__ == "__main__":
    main()
