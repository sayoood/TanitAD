"""PREREG_V8F_DSCALE10K study tool (card section 2.2 table, 2.4 and 5; MM freeze entry 2026-11-16 ~19:35Z): NO `stack/` change.

Builds the union per-clip label files U = (corpus 4,572 minus the 60 RA-1 holdout clips) + (10k data-scale clips), and the R_s0
derangement of the 10k label columns.

  v9 release   corpus ``v9_labels_train.npz`` (+ ``.manifest.json``)  +  10k ``v9_labels_datascale10k.npz``  ->  union npz + manifest
  lead         ``lead_train4572.npz`` (keys sha12 / k / lead / gap)    +  ``lead_datascale10k.npz``           ->  union npz
  speed        ``refcv6_speed_max_v8_train.jsonl``                     +  ``speed_sidecar_datascale10k.jsonl`` ->  union jsonl (line bytes verbatim)
  tracks       ``native_tracks.pt``                                    +  ``native_tracks_datascale10k.pt``    ->  union .pt
  manifest     corpus cache ``train/manifest.jsonl``                   +  10k cache manifest                   ->  union ``manifest.jsonl`` (line bytes verbatim)

Rules:
  * The corpus side keeps its original VALUES: every kept corpus clip's rows / lines / track are bit-equal to the input's. Only the 60 holdout clips
    are removed, BY ID.
  * The holdout ids are clips 540..599 of the corpus TRAIN manifest in ``sorted(clips, key=sha12)`` order: exactly the trainer's ``--exclude 540:600``
    (``stack/scripts/v8f_mini_train.py`` lines 3092-3096 at cdbff971: ``excluded = sorted(clips_all, key=D.sha12)[xlo:xhi]``; ``sha12`` =
    ``stack/tanitad/data/v8f_mini_data.py:93``). The digest of the result is the M2b summary's ``excluded_sha256`` (EXCLUDED_SHA256 below).
  * Deterministic: the same inputs give the same bytes. ``np.savez`` stamps the zip with the wall clock, so the npz is written here with a fixed zip
    timestamp; the .pt is serialised to a buffer (``torch.save`` into a file names its archive after the file name).
  * Nothing is overwritten and nothing is deleted. Every output is written to ``<name>.part`` and renamed.

CLI:  merge | derange | verify   (see ``main``). Tests: ``test_dscale10k.py``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

SCHEMA = "tanitad.v9_labels/1"
HOLD_LO, HOLD_HI = 540, 600
#: M2b_s0/summary.json ``excluded_sha256`` (MEASURED, PK/raw/full/runs/M2b_s0): newline-joined holdout clip ids in sha12 order
EXCLUDED_SHA256 = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
#: M2b_s0/summary.json ``train_sha256`` of the 4,512-clip B set (MEASURED)
B_TRAIN_SHA256 = "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1"
CORPUS_V9_MD5 = "83e30bd3791f0767707eade5f4671759"
TENK_V9_MD5 = "395221440e3a3c20f17dd381297af4f0"
R_SEED = 20261117
CLIP_KEYS = ("sid", "row0", "n_rows", "k0")
#: pod-side patch 2026-11-16: the real releases (corpus and 10k alike, MEASURED) carry 10 more per-clip columns; they are carried by sid, only row0 is re-based
EXTRA_CLIP_KEYS = ("sha12", "grid_start_s", "dt_s", "clock_src", "has_record", "has_lead", "has_lc", "n_turns", "n_announced", "suppressed")
CLIP_ALL = CLIP_KEYS + EXTRA_CLIP_KEYS
#: release row columns that stay the CLIP'S OWN in the R arm (card 2.4: the nav INPUT columns, plus the frame index and the label clock)
OWN_EXACT = ("k", "now_s")
OWN_PREFIX = ("nav_",)
MANIFEST_MUST_EQUAL = ("schema", "split", "fields", "vocab", "literals", "row_key", "clock_rule")


class MergeError(RuntimeError):
    """A refused merge / derangement / verification (fail loud: nothing is written)."""


# ------------------------------------------------------------------------------------------------ ids (mirrors of the code of record; a test pins them)
def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


def stable_episode_id(clip_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _commit(tmp, final):
    os.replace(tmp, final)


def _write_bytes(path, data: bytes):
    tmp = str(path) + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    _commit(tmp, path)


# ------------------------------------------------------------------------------------------------ manifests and the holdout
def read_manifest_lines(path) -> list:
    """The cache ``manifest.jsonl`` -> ``[(clip_id, n_frames, raw_line_bytes)]`` in file order (first position, last line wins: as ``read_manifest``)."""
    ents = {}
    with open(path, "rb") as f:
        for raw in f:
            if raw.strip():
                d = json.loads(raw.decode("utf-8"))
                ents.pop(d["clip"], None)                       # keep dict order = file order of the LAST occurrence is NOT what read_manifest does: it keeps the FIRST position
                ents[d["clip"]] = (d["clip"], int(d["n_frames"]), raw if raw.endswith(b"\n") else raw + b"\n")
    return list(ents.values())


def read_manifest_first_position(path) -> list:
    """Exactly ``v8f_mini_data.read_manifest``: ``ents[d["clip"]] = n_frames`` keeps the FIRST position of a clip and the LAST value. -> ``[(clip, n_frames)]``."""
    ents = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                ents[d["clip"]] = int(d["n_frames"])
    return list(ents.items())


def holdout_ids(manifest_path, lo: int = HOLD_LO, hi: int = HOLD_HI) -> list:
    """The RA-1 holdout: clips ``lo..hi`` of the corpus TRAIN manifest in sha12 order (the trainer's ``--exclude lo:hi``)."""
    clips = [c for c, _ in read_manifest_first_position(manifest_path)]
    return sorted(clips, key=sha12)[lo:hi]


def holdout_digest(ids) -> str:
    return hashlib.sha256(chr(10).join(ids).encode("utf-8")).hexdigest()


def train_sha256(clip_ids) -> str:
    """The trainer's ``train_sha256`` of a clip list (``v8f_mini_train.py`` line 3110: newline-joined, sha12 order)."""
    return hashlib.sha256(chr(10).join(sorted(clip_ids, key=sha12)).encode("utf-8")).hexdigest()


def merge_manifests(corpus_path, tenk_path, drop_ids, out_path) -> dict:
    """Union cache manifest: the corpus lines minus ``drop_ids`` (verbatim), then the 10k lines (verbatim). Refuses an overlap."""
    drop = set(drop_ids)
    c = read_manifest_lines(corpus_path)
    t = read_manifest_lines(tenk_path)
    c_ids = {x[0] for x in c}
    miss = sorted(drop - c_ids)
    if miss:
        raise MergeError(f"{len(miss)} holdout ids are not in the corpus manifest (first: {miss[:3]})")
    keep = [x for x in c if x[0] not in drop]
    both = {x[0] for x in keep} & {x[0] for x in t}
    if both:
        raise MergeError(f"{len(both)} clips are in BOTH manifests (first: {sorted(both)[:3]})")
    hit = drop & {x[0] for x in t}
    if hit:
        raise MergeError(f"{len(hit)} holdout clips are in the 10k manifest (first: {sorted(hit)[:3]})")
    _write_bytes(out_path, b"".join(x[2] for x in keep + t))
    return {"n_corpus_kept": len(keep), "n_tenk": len(t), "n_union": len(keep) + len(t)}


# ------------------------------------------------------------------------------------------------ deterministic npz
def write_npz_deterministic(path, arrays: dict):
    """An uncompressed ``.npz`` that ``np.load`` reads like ``np.savez``'s, with a FIXED zip timestamp and entry order (dict order). Same arrays -> same bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            a = np.ascontiguousarray(arr)
            if a.dtype.hasobject:
                raise MergeError(f"array {name!r} has dtype object (pickle): refused")
            b = io.BytesIO()
            np.lib.format.write_array(b, a, allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o600 << 16
            zi.create_system = 3
            zf.writestr(zi, b.getvalue())
    _write_bytes(path, buf.getvalue())


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ v9 release
def _release_parts(Z: dict, name: str):
    rows = {k: v for k, v in Z.items() if k.startswith("row__")}
    clips = {k[6:]: v for k, v in Z.items() if k.startswith("clip__")}
    other = [k for k in Z if not (k.startswith("row__") or k.startswith("clip__"))]
    if other:
        raise MergeError(f"{name}: unexpected keys {other} (a v9 release holds only row__* and clip__*)")
    if set(clips) not in (set(CLIP_KEYS), set(CLIP_ALL)):       # the synthetic fixtures carry the 4 core keys, the real releases all 14 (MEASURED)
        raise MergeError(f"{name}: clip__ keys {sorted(clips)} != {sorted(CLIP_KEYS)} or {sorted(CLIP_ALL)}")
    n = {len(v) for v in rows.values()}
    if len(n) != 1:
        raise MergeError(f"{name}: row arrays differ in length {sorted(n)}")
    return rows, clips, n.pop()


def _clip_table(clips: dict, n_rows: int, name: str):
    """-> list of (sid, row0, nrows, k0) in table order; refuses a table whose row ranges are not contiguous and ascending from 0 (the merge would reorder rows)."""
    t = list(zip(clips["sid"].tolist(), clips["row0"].tolist(), clips["n_rows"].tolist(), clips["k0"].tolist()))
    pos = 0
    for sid, r0, n, k0 in t:
        if r0 != pos or n < 0:
            raise MergeError(f"{name}: clip sid {sid} starts at row {r0}, expected {pos}: the table is not contiguous / ascending")
        pos += n
    if pos != n_rows:
        raise MergeError(f"{name}: the clip table covers {pos} rows, the release holds {n_rows}")
    if len({x[0] for x in t}) != len(t):
        raise MergeError(f"{name}: duplicate sids in the clip table")
    return t


def merge_v9(corpus_npz, tenk_npz, drop_ids, out_npz) -> dict:
    """The union release (npz + ``.manifest.json``). Returns the record. ``drop_ids`` = the 60 holdout CLIP ids (sids derived by ``stable_episode_id``)."""
    Zc, Zt = load_npz(corpus_npz), load_npz(tenk_npz)
    rc, cc, nc = _release_parts(Zc, "corpus")
    rt, ct, nt = _release_parts(Zt, "10k")
    if set(rc) != set(rt):                                        # pod-side patch 2026-11-16: the real files differ in key ORDER only (MEASURED); the union uses the corpus order
        raise MergeError(f"row columns differ: corpus-only {sorted(set(rc) - set(rt))}, 10k-only {sorted(set(rt) - set(rc))} (or a different order)")
    for k in rc:
        if rc[k].dtype != rt[k].dtype or rc[k].shape[1:] != rt[k].shape[1:]:
            raise MergeError(f"row column {k}: dtype / shape differ ({rc[k].dtype}{rc[k].shape[1:]} vs {rt[k].dtype}{rt[k].shape[1:]})")
    tc, tt = _clip_table(cc, nc, "corpus"), _clip_table(ct, nt, "10k")
    drop_sid = {stable_episode_id(c) for c in drop_ids}
    have = {x[0] for x in tc}
    if drop_sid - have:
        raise MergeError(f"{len(drop_sid - have)} holdout clips have no rows in the corpus release: wrong manifest or wrong release")
    if drop_sid & {x[0] for x in tt}:
        raise MergeError("a holdout clip is in the 10k release")
    keep_c = [x for x in tc if x[0] not in drop_sid]
    if {x[0] for x in keep_c} & {x[0] for x in tt}:
        raise MergeError("the corpus and the 10k release share a clip sid")
    idx = np.concatenate([np.arange(r0, r0 + n, dtype=np.int64) for _, r0, n, _ in keep_c]) if keep_c else np.zeros(0, np.int64)
    out = {}
    for k in rc:
        out[k] = np.concatenate([rc[k][idx], rt[k]], axis=0)
    # clip table: corpus kept clips re-based, then the 10k clips shifted by the kept corpus rows
    cset = {x[0] for x in tc}
    new, pos = [], 0
    for sid, _, n, k0 in keep_c:
        new.append((sid, pos, n, k0))
        pos += n
    base = pos
    for sid, r0, n, k0 in tt:
        new.append((sid, base + r0, n, k0))
    present = tuple(k for k in CLIP_ALL if k in cc)
    if set(present) != {k for k in CLIP_ALL if k in ct}:
        raise MergeError("the two releases carry different clip__ key sets")
    cdt = {k: cc[k].dtype for k in present}
    for j, k in enumerate(CLIP_KEYS):
        out["clip__" + k] = np.array([x[j] for x in new], cdt[k])
    for k in present:
        if cdt[k] != ct[k].dtype:
            raise MergeError(f"clip__{k} dtype differs between the releases")
    exs = {}
    for C in (cc, ct):
        for i, sid in enumerate(C["sid"].tolist()):
            exs[sid] = i
    for k in present[len(CLIP_KEYS):]:
        out["clip__" + k] = np.array([(cc if x[0] in cset else ct)[k][exs[x[0]]] for x in new], cdt[k])
    write_npz_deterministic(out_npz, out)
    # manifest
    mc = json.load(open(os.path.splitext(str(corpus_npz))[0] + ".manifest.json", encoding="utf-8"))
    mt = json.load(open(os.path.splitext(str(tenk_npz))[0] + ".manifest.json", encoding="utf-8"))
    for k in MANIFEST_MUST_EQUAL:
        if mc.get(k) != mt.get(k):
            raise MergeError(f"the two release manifests differ in {k!r}: the releases are not one schema")
    md5 = md5_file(out_npz)
    man = dict(mc)
    man.update({"n_clips": len(new), "n_rows": int(len(out["row__k"])), "npz_md5": md5})
    man.pop("wall_s", None)
    man["merge"] = {"tool": "dscale10k_merge.py", "corpus_md5": md5_file(corpus_npz), "tenk_md5": md5_file(tenk_npz), "n_holdout_dropped": len(drop_sid),
                    "holdout_sha256": holdout_digest(list(drop_ids)), "n_corpus_clips_kept": len(keep_c), "n_tenk_clips": len(tt)}
    _write_bytes(os.path.splitext(str(out_npz))[0] + ".manifest.json", (json.dumps(man, sort_keys=True, indent=1) + "\n").encode("utf-8"))
    return {"md5": md5, "n_clips": len(new), "n_rows": int(len(out["row__k"])), "n_corpus_clips_kept": len(keep_c), "n_tenk_clips": len(tt),
            "n_holdout_dropped": len(drop_sid)}


# ------------------------------------------------------------------------------------------------ lead / speed / tracks
def merge_lead(corpus_npz, tenk_npz, drop_ids, out_npz) -> dict:
    """lead npz: every key is a per-row array keyed by ``sha12`` (rows of the holdout clips dropped; corpus rows keep their order and values; the 10k rows follow)."""
    c, t = load_npz_obj(corpus_npz), load_npz_obj(tenk_npz)
    if list(c) != list(t):
        raise MergeError(f"lead keys differ: {list(c)} vs {list(t)}")
    if "sha12" not in c:
        raise MergeError("lead npz has no sha12 key")
    cm, tm = c.pop("meta", None), t.pop("meta", None)                  # pod-side patch 2026-11-16: the real lead files carry a 0-d string `meta` (MEASURED); the trainer does not read it
    if (cm is None) != (tm is None):
        raise MergeError("only one lead file has a meta key")
    n = {k: len(v) for k, v in c.items()}
    if len(set(n.values())) != 1 or len({len(v) for v in t.values()}) != 1:
        raise MergeError("lead arrays differ in length inside one file")
    drop = np.array(sorted({sha12(x) for x in drop_ids}), dtype=str)
    keep = ~np.isin(c["sha12"].astype(str), drop)
    if np.isin(t["sha12"].astype(str), drop).any():
        raise MergeError("a holdout clip has lead rows in the 10k lead file")
    if set(c["sha12"][keep].astype(str).tolist()) & set(t["sha12"].astype(str).tolist()):
        raise MergeError("the corpus and the 10k lead files share a clip (sha12)")
    out = {}
    for k in c:
        if c[k].dtype.kind != t[k].dtype.kind or c[k].shape[1:] != t[k].shape[1:]:
            raise MergeError(f"lead column {k}: dtype kind / shape differ")
        out[k] = np.concatenate([c[k][keep], t[k]], axis=0)
    if cm is not None:
        out["meta"] = _lead_meta(cm, tm)
    write_npz_deterministic(out_npz, out)
    return {"md5": md5_file(out_npz), "n_rows": int(len(out["sha12"])), "n_corpus_rows_kept": int(keep.sum()), "n_tenk_rows": int(len(t["sha12"])),
            "n_corpus_rows_dropped": int((~keep).sum())}


def _lead_meta(cm, tm):
    return np.array(json.dumps({"tool": "dscale10k_merge.py", "merged_from": {"corpus": str(cm), "tenk": str(tm)}}, sort_keys=True))


def load_npz_obj(path) -> dict:
    """The trainer reads the lead file with ``allow_pickle=True`` (``build_label_join``); a file that needs it is refused here (a pickle in a data file is not read by this tool)."""
    return load_npz(path)


def _read_lines(path) -> list:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as f:
        return [ln if ln.endswith(b"\n") else ln + b"\n" for ln in f.read().splitlines(keepends=True) if ln.strip()]


def merge_speed(corpus_path, tenk_path, drop_ids, out_path) -> dict:
    """speed sidecar jsonl: the corpus lines minus the holdout clips (verbatim), then the 10k lines (verbatim). ``clip_id`` is the key (``load_speed_sidecar``)."""
    drop = set(drop_ids)
    lc, lt = _read_lines(corpus_path), _read_lines(tenk_path)
    keep = []
    for ln in lc:
        if json.loads(ln.decode("utf-8"))["clip_id"] not in drop:
            keep.append(ln)
    ids_t = [json.loads(ln.decode("utf-8"))["clip_id"] for ln in lt]
    if drop & set(ids_t):
        raise MergeError("a holdout clip is in the 10k speed sidecar")
    ids_k = {json.loads(ln.decode("utf-8"))["clip_id"] for ln in keep}
    if ids_k & set(ids_t):
        raise MergeError("the corpus and the 10k speed sidecars share a clip")
    if len(ids_t) != len(set(ids_t)) or len(ids_k) != len(keep):
        raise MergeError("a speed sidecar holds a clip twice")
    _write_bytes(out_path, b"".join(keep + lt))
    return {"sha256": sha256_file(out_path), "n_lines": len(keep) + len(lt), "n_corpus_kept": len(keep), "n_tenk": len(lt), "n_corpus_dropped": len(lc) - len(keep)}


def merge_tracks(corpus_pt, tenk_pt, drop_ids, out_pt) -> dict:
    """native_tracks.pt: ``{"tracks": {clip: tensor [n_raw, 4]}, "meta": {...}}``. Holdout tracks dropped; corpus tensors are passed through unchanged; the 10k tracks follow."""
    import torch
    oc = torch.load(corpus_pt, map_location="cpu", weights_only=True)
    ot = torch.load(tenk_pt, map_location="cpu", weights_only=True)
    drop = set(drop_ids)
    tc, tt = oc["tracks"], ot["tracks"]
    if drop & set(tt):
        raise MergeError("a holdout clip is in the 10k tracks")
    keep = {c: v for c, v in tc.items() if c not in drop}
    if set(keep) & set(tt):
        raise MergeError("the corpus and the 10k tracks share a clip")
    for c, v in list(keep.items())[:1] + list(tt.items())[:1]:
        if v.dtype != next(iter(tc.values())).dtype or v.shape[1:] != next(iter(tc.values())).shape[1:]:
            raise MergeError(f"track {c}: dtype / shape differs from the corpus tracks")
    meta = dict(oc.get("meta", {}))
    meta["merge"] = {"tool": "dscale10k_merge.py", "n_corpus_kept": len(keep), "n_tenk": len(tt), "n_holdout_dropped": len(tc) - len(keep)}
    buf = io.BytesIO()
    torch.save({"tracks": {**keep, **tt}, "meta": meta}, buf)
    _write_bytes(out_pt, buf.getvalue())
    return {"sha256": sha256_file(out_pt), "n_tracks": len(keep) + len(tt), "n_corpus_kept": len(keep), "n_tenk": len(tt), "n_corpus_dropped": len(tc) - len(keep)}


# ------------------------------------------------------------------------------------------------ R_s0: the derangement
def is_own_column(col: str) -> bool:
    name = col[5:] if col.startswith("row__") else col
    return name in OWN_EXACT or name.startswith(OWN_PREFIX)


def derange_v9(union_npz, tenk_npz, out_npz, seed: int = R_SEED, max_singletons: int = 0) -> dict:
    """The R release: the union release with the 10k clips' LABEL columns deranged at clip level, within groups of equal ``(n_rows, k0)``.

    Every 10k clip in a group of size g >= 2 receives the rows of ANOTHER clip of its group (a shuffled cycle: no fixed point). A group of size 1 cannot be
    deranged: more than ``max_singletons`` such clips refuse the build (a clip that keeps its own labels would make R weaker than its name). Columns
    ``k``, ``now_s`` and every ``nav_*`` stay the clip's own. The corpus rows are untouched. -> the record (with the donor of every receiver)."""
    U = load_npz(union_npz)
    rows, clips, n = _release_parts(U, "union")
    tab = _clip_table(clips, n, "union")
    tsid = set(load_npz(tenk_npz)["clip__sid"].tolist())
    if not tsid <= {x[0] for x in tab}:
        raise MergeError("some 10k clips are not in the union release")
    groups = {}
    for sid, r0, nr, k0 in tab:
        if sid in tsid:
            groups.setdefault((nr, k0), []).append((sid, r0))
    single = [g for g, m in groups.items() if len(m) < 2]
    n_single = sum(len(groups[g]) for g in single)
    if n_single > max_singletons:
        raise MergeError(f"{n_single} 10k clips sit in groups of one (n_rows, k0) ({sorted(single)[:5]}): they cannot be deranged; allowed {max_singletons}")
    rng = np.random.default_rng(int(seed))
    out = {k: v.copy() for k, v in rows.items()}
    moved = [c for c in rows if not is_own_column(c)]
    donor = {}
    for key in sorted(groups):
        mem = sorted(groups[key])                                 # by sid: the draw does not depend on the table order
        if len(mem) < 2:
            continue
        order = [int(x) for x in rng.permutation(len(mem))]
        for s_, i in enumerate(order):                            # the shuffled cycle: the member at slot s receives the one at slot s+1 -> no fixed point
            sid, r0 = mem[i]
            dsid, dr0 = mem[order[(s_ + 1) % len(mem)]]
            for c in moved:
                out[c][r0:r0 + key[0]] = rows[c][dr0:dr0 + key[0]]
            donor[sid] = dsid
    if any(s == d for s, d in donor.items()):
        raise MergeError("internal: a fixed point")
    arrays = dict(out)
    for k in clips:
        arrays["clip__" + k] = clips[k]
    write_npz_deterministic(out_npz, arrays)
    mu = json.load(open(os.path.splitext(str(union_npz))[0] + ".manifest.json", encoding="utf-8"))
    md5 = md5_file(out_npz)
    mu["npz_md5"] = md5
    mu["derange"] = {"tool": "dscale10k_merge.py", "seed": int(seed), "union_md5": md5_file(union_npz), "columns_own": [c for c in rows if is_own_column(c)],
                     "n_deranged_clips": len(donor), "n_singletons_left": n_single}
    _write_bytes(os.path.splitext(str(out_npz))[0] + ".manifest.json", (json.dumps(mu, sort_keys=True, indent=1) + "\n").encode("utf-8"))
    rec = {"md5": md5, "seed": int(seed), "n_deranged_clips": len(donor), "n_singletons_left": n_single, "n_groups": len(groups),
           "donor": {str(k): int(v) for k, v in sorted(donor.items())}}
    return rec


# ------------------------------------------------------------------------------------------------ C-MERGE (verification, independent of the writers)
def verify_v9(corpus_npz, tenk_npz, union_npz, drop_ids) -> list:
    """C-MERGE for the release: per KEPT corpus clip the rows are bit-equal to the corpus file's; per 10k clip bit-equal to the 10k file's; sids disjoint; the
    holdout is absent; the counts add up. Returns a list of problems ([] = PASS)."""
    bad = []
    Zc, Zt, Zu = load_npz(corpus_npz), load_npz(tenk_npz), load_npz(union_npz)
    rc, cc, nc = _release_parts(Zc, "corpus")
    rt, ct, nt = _release_parts(Zt, "10k")
    ru, cu, nu = _release_parts(Zu, "union")
    if list(ru) != list(rc):
        return ["union columns differ from the corpus columns"]
    tc, tt, tu = _clip_table(cc, nc, "corpus"), _clip_table(ct, nt, "10k"), _clip_table(cu, nu, "union")
    ud = {sid: (r0, n, k0) for sid, r0, n, k0 in tu}
    drop_sid = {stable_episode_id(c) for c in drop_ids}
    if drop_sid & set(ud):
        bad.append(f"{len(drop_sid & set(ud))} holdout clips are in the union release")
    exp = [x for x in tc if x[0] not in drop_sid] + list(tt)
    if {x[0] for x in exp} != set(ud) or len(tu) != len(exp):
        bad.append(f"union clip set differs: {len(ud)} clips, expected {len(exp)}")
        return bad
    if len({x[0] for x in tc} & {x[0] for x in tt}):
        bad.append("corpus and 10k releases share a sid")
    for src, tab, R in (("corpus", [x for x in tc if x[0] not in drop_sid], rc), ("10k", tt, rt)):
        si, di = [], []
        for sid, r0, n, k0 in tab:
            u0, un, uk0 = ud[sid]
            if (un, uk0) != (n, k0):
                bad.append(f"{src} clip {sid}: n_rows/k0 differ")
                continue
            si.append(np.arange(r0, r0 + n, dtype=np.int64))
            di.append(np.arange(u0, u0 + n, dtype=np.int64))
        si = np.concatenate(si) if si else np.zeros(0, np.int64)
        di = np.concatenate(di) if di else np.zeros(0, np.int64)
        for c in R:
            if R[c][si].tobytes() != ru[c][di].tobytes():
                bad.append(f"{src} rows: column {c} is not bit-equal to the {src} release's")
    return bad[:50]


def verify_lead(corpus_npz, tenk_npz, union_npz, drop_ids) -> list:
    c, t, u = load_npz(corpus_npz), load_npz(tenk_npz), load_npz(union_npz)
    drop = np.array(sorted({sha12(x) for x in drop_ids}), dtype=str)
    keep = ~np.isin(c["sha12"].astype(str), drop)
    bad = []
    if list(u) != list(c):
        return ["lead keys differ"]
    if "meta" in c:
        if u["meta"].tobytes() != _lead_meta(c.pop("meta"), t.pop("meta")).tobytes():
            bad.append("lead meta differs from the expected merged meta")
        u.pop("meta")
    for k in c:
        exp = np.concatenate([c[k][keep], t[k]], axis=0)
        if exp.tobytes() != u[k].tobytes() or exp.shape != u[k].shape:
            bad.append(f"lead column {k} differs from corpus-kept + 10k")
    if np.isin(u["sha12"].astype(str), drop).any():
        bad.append("a holdout clip has lead rows in the union")
    return bad


def verify_speed(corpus_path, tenk_path, union_path, drop_ids) -> list:
    drop = set(drop_ids)
    keep = [ln for ln in _read_lines(corpus_path) if json.loads(ln.decode("utf-8"))["clip_id"] not in drop]
    exp = keep + _read_lines(tenk_path)
    got = _read_lines(union_path)
    return [] if exp == got else [f"speed sidecar differs: expected {len(exp)} lines, union has {len(got)} (or a line's bytes differ)"]


def verify_tracks(corpus_pt, tenk_pt, union_pt, drop_ids) -> list:
    import torch
    oc, ot, ou = (torch.load(p, map_location="cpu", weights_only=True) for p in (corpus_pt, tenk_pt, union_pt))
    drop, bad = set(drop_ids), []
    exp = {**{c: v for c, v in oc["tracks"].items() if c not in drop}, **ot["tracks"]}
    got = ou["tracks"]
    if set(exp) != set(got):
        return [f"track clip set differs: expected {len(exp)}, union {len(got)}"]
    for c, v in exp.items():
        if got[c].dtype != v.dtype or got[c].shape != v.shape or got[c].numpy().tobytes() != v.numpy().tobytes():
            bad.append(f"track {c} differs")
            if len(bad) >= 20:
                break
    return bad


def verify_derange(union_npz, r_npz, tenk_npz, rec: dict | None = None) -> list:
    """The R release's properties: corpus rows bit-equal to the union's; own columns (k, now_s, nav_*) bit-equal on EVERY clip; on the 10k clips (outside the
    allowed singletons) every clip's label block differs from its own and equals another 10k clip's block of the same (n_rows, k0); no fixed point."""
    U, Rr, T = load_npz(union_npz), load_npz(r_npz), load_npz(tenk_npz)
    ru, cu, nu = _release_parts(U, "union")
    rr, cr, nr_ = _release_parts(Rr, "R")
    bad = []
    if list(ru) != list(rr) or nu != nr_:
        return ["R and union releases differ in columns or length"]
    for k in cu:
        if cu[k].tobytes() != cr[k].tobytes():
            bad.append(f"clip table {k} changed")
    tab = _clip_table(cu, nu, "union")
    tsid = set(T["clip__sid"].tolist())
    for c in ru:
        if is_own_column(c) and ru[c].tobytes() != rr[c].tobytes():
            bad.append(f"own column {c} changed")
    mv = [c for c in ru if not is_own_column(c)]
    for sid, r0, n, k0 in tab:
        if sid in tsid:
            continue
        for c in mv:
            if ru[c][r0:r0 + n].tobytes() != rr[c][r0:r0 + n].tobytes():
                bad.append(f"corpus clip {sid}: column {c} changed")
                break
    groups = {}
    for sid, r0, n, k0 in tab:
        if sid in tsid:
            groups.setdefault((n, k0), []).append((sid, r0))
    n_fixed = n_same_as_own = 0
    for key, mem in groups.items():
        def dg(R, r0):
            h = hashlib.sha256()
            for c in mv:
                h.update(R[c][r0:r0 + key[0]].tobytes())
            return h.digest()
        blocks = {sid: dg(ru, r0) for sid, r0 in mem}
        allb = set(blocks.values())
        for sid, r0 in mem:
            if len(mem) < 2:
                continue
            got = dg(rr, r0)
            if got == blocks[sid]:
                n_same_as_own += 1
            if got not in allb:
                bad.append(f"10k clip {sid}: its R block is not any clip's block of its group")
    if rec is not None:
        n_fixed = sum(1 for s, d in rec["donor"].items() if int(s) == int(d))
        if n_fixed:
            bad.append(f"{n_fixed} fixed points in the donor map")
    if n_same_as_own:
        bad.append(f"{n_same_as_own} 10k clips kept their own label block (identical blocks counted: a derangement of identical rows would hide here)")
    return bad[:50]


# ------------------------------------------------------------------------------------------------ driver
def _hash_entry(p):
    return {"path": str(p), "size": os.path.getsize(p), "sha256": sha256_file(p), "md5": md5_file(p)}


def merge_all(a) -> dict:
    os.makedirs(a.out_dir, exist_ok=True)
    ids = holdout_ids(a.corpus_manifest, *a.holdout)
    dig = holdout_digest(ids)
    rec = {"tool": "dscale10k_merge.py", "holdout": {"range": list(a.holdout), "n": len(ids), "sha256": dig, "expected_sha256": EXCLUDED_SHA256,
           "matches_M2b_summary": dig == EXCLUDED_SHA256}, "inputs": {}, "outputs": {}}
    if list(a.holdout) == [HOLD_LO, HOLD_HI] and dig != EXCLUDED_SHA256:
        raise MergeError(f"the reproduced holdout digest {dig} != the M2b summary's excluded_sha256 {EXCLUDED_SHA256}: the corpus manifest is not the one the baseline trained on")
    o = lambda n: os.path.join(a.out_dir, n)
    for nm, p in (("corpus_manifest", a.corpus_manifest), ("tenk_manifest", a.tenk_manifest), ("v9_corpus", a.v9[0]), ("v9_tenk", a.v9[1]), ("lead_corpus", a.lead[0]),
                  ("lead_tenk", a.lead[1]), ("speed_corpus", a.speed[0]), ("speed_tenk", a.speed[1]), ("tracks_corpus", a.tracks[0]), ("tracks_tenk", a.tracks[1])):
        rec["inputs"][nm] = _hash_entry(p)
    if rec["inputs"]["v9_corpus"]["md5"] != CORPUS_V9_MD5 or rec["inputs"]["v9_tenk"]["md5"] != TENK_V9_MD5:
        raise MergeError(f"v9 input md5s are {rec['inputs']['v9_corpus']['md5']} / {rec['inputs']['v9_tenk']['md5']}, expected {CORPUS_V9_MD5} / {TENK_V9_MD5}")
    rec["manifest"] = merge_manifests(a.corpus_manifest, a.tenk_manifest, ids, o("manifest.jsonl"))
    rec["v9"] = merge_v9(a.v9[0], a.v9[1], ids, o("v9_labels_union.npz"))
    rec["lead"] = merge_lead(a.lead[0], a.lead[1], ids, o("lead_union.npz"))
    rec["speed"] = merge_speed(a.speed[0], a.speed[1], ids, o("speed_union.jsonl"))
    rec["tracks"] = merge_tracks(a.tracks[0], a.tracks[1], ids, o("native_tracks_union.pt"))
    for nm in ("manifest.jsonl", "v9_labels_union.npz", "v9_labels_union.manifest.json", "lead_union.npz", "speed_union.jsonl", "native_tracks_union.pt"):
        rec["outputs"][nm] = _hash_entry(o(nm))
    prob = (verify_v9(a.v9[0], a.v9[1], o("v9_labels_union.npz"), ids) + verify_lead(a.lead[0], a.lead[1], o("lead_union.npz"), ids)
            + verify_speed(a.speed[0], a.speed[1], o("speed_union.jsonl"), ids) + verify_tracks(a.tracks[0], a.tracks[1], o("native_tracks_union.pt"), ids))
    rec["c_merge_problems"] = prob
    rec["c_merge_pass"] = not prob
    _write_bytes(o("merge_record.json"), (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    if prob:
        raise MergeError("C-MERGE failed right after the merge: " + "; ".join(prob[:5]))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--corpus-manifest", required=True, help="the corpus cache train/manifest.jsonl (4,572 clips): where the holdout ids come from")
    p.add_argument("--tenk-manifest", required=True)
    p.add_argument("--holdout", type=lambda s: tuple(int(x) for x in s.split(":")), default=(HOLD_LO, HOLD_HI))
    p.add_argument("--v9", nargs=2, required=True, metavar=("CORPUS", "TENK"))
    p.add_argument("--lead", nargs=2, required=True, metavar=("CORPUS", "TENK"))
    p.add_argument("--speed", nargs=2, required=True, metavar=("CORPUS", "TENK"))
    p.add_argument("--tracks", nargs=2, required=True, metavar=("CORPUS", "TENK"))
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("derange")
    p.add_argument("--union", required=True); p.add_argument("--tenk-v9", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=R_SEED); p.add_argument("--max-singletons", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "merge":
            r = merge_all(a)
            print(json.dumps({k: r[k] for k in ("holdout", "manifest", "v9", "lead", "speed", "tracks", "c_merge_pass")}, sort_keys=True))
        else:
            r = derange_v9(a.union, a.tenk_v9, a.out, a.seed, a.max_singletons)
            prob = verify_derange(a.union, a.out, a.tenk_v9, r)
            r["verify_problems"] = prob
            _write_bytes(os.path.splitext(a.out)[0] + ".derangement.json", (json.dumps(r, indent=1, sort_keys=True) + "\n").encode("utf-8"))
            print(json.dumps({k: v for k, v in r.items() if k != "donor"}))
            if prob:
                raise MergeError("derangement verification failed: " + "; ".join(prob[:5]))
    except MergeError as e:
        print("REFUSED: " + str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
