#!/usr/bin/env python3
"""count_alpamayo.py -- re-derive EVERY Alpamayo-augmentation count from bytes.

BACKLOG E10 (2026-10-18).  Nothing in here reads a prose count.  Each figure is
computed from a primary artifact (the A2 parquet, its selection manifest, the
v7.2 label release, the aug120 fusion JSONs, the parity clip-id lists and the
committed parity digest oracle), and the script records, for every artifact it
touched, its sha256, byte size and git blob id, so a later reader can tell
WHICH BYTES were counted.  Counts only -- no clip id is written to the output.

Layers (CLAUDE.md "Operating standard" rule 2 -- state the layer, never the bare
phrase "our ingest"):
  PUBLISHED   = the published NVIDIA PhysicalAI-AV corpus
  EPISODE     = our episode build (parity cache / w120 caches / S2 clip index)
  AUGMENTED   = an augmented LABEL RELEASE we produced (A2 augmentation, the
                PH1-fused aug120 layer, the v7.2 label release)

Usage (needs pyarrow; stdlib otherwise):
  PYTHONUTF8=1 python -I count_alpamayo.py --repo <repo root or tip extract> \
      --a2-dir <dir with records.parquet + selection_manifest.json> \
      --dc <data_collection.parquet> --out counts.json

Exit code 0 only if every internal assertion (closure identities, cross-artifact
set equalities, expected single-valued columns) held.

The `paper_claims` section pins paper LINE NUMBERS for paper blob c9c56706 (tip d6a4ce9)
and asserts each quoted form really sits on its line: run against a different paper
blob it FAILS LOUD (exit 1) rather than silently re-pointing -- by design.  Pass
--skip-paper-pins to run the counts alone on any checkout (the 114 counts are byte-identical
whichever paper is present).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------- repo paths --
RL = "TanitAD Research Lab"
AUG = f"{RL}/Benchmarks & Evals/Research/2026-08-06-alpamayo-augmentation"
THOR = f"{RL}/Architecture & Inference/Implementation/incoming/2026-08-17-thor-concurrency-pilot"
S2IDX = f"{RL}/Data Engineering/Implementation/incoming/2026-08-16-s2-v1-labels/review/labels_v2/clip_index.json"
A120 = f"{RL}/Data Engineering/Implementation/incoming/2026-08-15-aug120-fusion/raw"
V72 = f"{RL}/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw"
VAL40 = (f"{RL}/Architecture & Inference/Implementation/incoming/2026-08-18-thor-stranded-rescue/"
         "rescued_beyond_a11/leadwork/val40_lead_index.json")
VAL40_ANON = f"{RL}/Data Engineering/Implementation/incoming/2026-08-04-instrument-durability/raw/val40_lead_index_ANON.json"

REPO_FILES = {
    "road_class": f"{AUG}/aug_road_class.json",
    "alpa_ids": f"{THOR}/alpamayo_clip_ids.txt",
    "excl": f"{THOR}/alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt",
    "parity_ls": f"{THOR}/parity_ls.txt",
    "oracle": "stack/tanitad/data/parity_train_clip_digests.json",
    "s2idx": S2IDX,
    "a120_cov": f"{A120}/aug120_coverage.json",
    "a120_sum": f"{A120}/fused_aug120_summary.json",
    "a120_acct": f"{A120}/fused_aug120_batch_accounting.json",
    "a120_src": f"{A120}/fused_aug120_label_sources.json",
    "v72_train": f"{V72}/s2_labels_v7.2_train.jsonl.gz",
    "v72_eval": f"{V72}/s2_labels_v7.2_eval.jsonl.gz",
    "val40": VAL40,
    "val40_anon": VAL40_ANON,
    "paper": "Paper/TANITAD_PAPER.md",
    "registry": "Project Steering/MODEL_REGISTRY.md",
}

DATE = "2026-10-18"
FAILS: list[str] = []


# ------------------------------------------------------------------ helpers --
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_id(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)
        print("ASSERTION FAILED:", msg, file=sys.stderr)


def lines_of(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def pct(a: int, b: int, nd: int = 4) -> float:
    return round(100.0 * a / b, nd)


def int_forms(n: int) -> list[str]:
    """Every way a count is written in this repo: 4,729 / 4729 / 4 729 / thin-space."""
    c = f"{n:,}"
    return sorted({c, str(n), c.replace(",", " "), c.replace(",", " "),
                   c.replace(",", " "), c.replace(",", " ")}, key=lambda f: (-len(f), f))


def form_regex(form: str) -> re.Pattern:
    return re.compile(r"(?<![\d.,])" + re.escape(form) + r"(?!\d|,\d)")


def hit_lines(text_lines: list[str], forms: list[str], ctx: re.Pattern | None,
              cap: int = 14) -> list[int]:
    rxs = [form_regex(f) for f in forms]
    out = []
    for i, ln in enumerate(text_lines):
        if any(r.search(ln) for r in rxs):
            if ctx is not None:
                win = " ".join(text_lines[max(0, i - 2): i + 3])
                if not ctx.search(win):
                    continue
            out.append(i + 1)
            if len(out) >= cap:
                break
    return out


# ------------------------------------------------------------------- ledger --
class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, cid: str, label: str, value, layer: str, artifacts: list[str],
            how: str, evidence: str = "MEASURED", note: str = "") -> None:
        self.rows.append({"id": cid, "label": label, "value": value, "layer": layer,
                          "artifacts": artifacts, "how": how, "evidence": evidence,
                          "date": DATE, "note": note})

    def v(self, cid: str):
        for r in self.rows:
            if r["id"] == cid:
                return r["value"]
        raise KeyError(cid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repo root (or a tip extract with repo-relative paths)")
    ap.add_argument("--a2-dir", required=True, help="dir holding records.parquet + selection_manifest.json")
    ap.add_argument("--dc", required=True, help="PhysicalAI-AV metadata/data_collection.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-paper-pins", action="store_true",
                    help="do not assert the curated paper line numbers (use on a checkout whose paper blob is "
                         "not c9c56706); the counts themselves do not depend on the paper")
    a = ap.parse_args()
    repo = Path(a.repo)
    P = {k: repo / v for k, v in REPO_FILES.items()}
    P["records"] = Path(a.a2_dir) / "records.parquet"
    P["manifest"] = Path(a.a2_dir) / "selection_manifest.json"
    P["dc"] = Path(a.dc)
    for k, p in P.items():
        if not p.is_file():
            print(f"MISSING {k}: {p}", file=sys.stderr)
            return 2

    # -- artifact table: WHICH BYTES were counted --------------------------------
    art = {}
    for k, p in P.items():
        art[k] = {"path": REPO_FILES.get(k, f"(non-repo) {p.name}"), "bytes": p.stat().st_size,
                  "sha256": sha256_file(p), "git_blob_id": git_blob_id(p)}
        if k in ("records", "manifest", "dc"):
            art[k]["git_blob_id"] = None  # not a repo file

    L = Ledger()
    A_REC, A_MAN = ["records"], ["manifest"]

    # ============================================================ PUBLISHED ====
    dc = pq.read_table(P["dc"], columns=["clip_id"]).column("clip_id").to_pylist()
    dc_set = set(dc)
    L.add("PUB1", "clips in published corpus (data_collection.parquet rows)", len(dc), "PUBLISHED",
          ["dc"], "pq.read_table(dc).num_rows")
    L.add("PUB2", "distinct clip_id in data_collection.parquet", len(dc_set), "PUBLISHED",
          ["dc"], "len(set(clip_id))")

    # ============================================================ A2 parquet ===
    t = pq.read_table(P["records"])
    nrows = t.num_rows
    cid = t.column("clip_id").to_pylist()
    task = t.column("task").to_pylist()
    a2_clips = set(cid)
    L.add("A01", "A2 rows", nrows, "AUGMENTED", A_REC, "pq.read_table(records).num_rows")
    L.add("A02", "A2 distinct clip_id", len(a2_clips), "AUGMENTED", A_REC, "len(set(clip_id))")
    L.add("A03", "A2 columns", t.num_columns, "AUGMENTED", A_REC, "table.num_columns")
    rows_by_task = collections.Counter(task)
    for k in ("trajectory", "meta_action", "auto_labeling", "vqa", "grounding_via_vqa"):
        L.add(f"A04.{k}", f"A2 rows, task={k}", rows_by_task[k], "AUGMENTED", A_REC,
              f"sum(task=='{k}')")
    check(set(rows_by_task) == {"trajectory", "meta_action", "auto_labeling", "vqa", "grounding_via_vqa"},
          "unexpected task set")
    per_clip = collections.Counter(cid)
    n_per = collections.Counter(per_clip.values())
    L.add("A05.5rows", "A2 clips carrying 5 task rows", n_per[5], "AUGMENTED", A_REC,
          "Counter(Counter(clip_id).values())[5]")
    L.add("A05.4rows", "A2 clips carrying 4 task rows", n_per[4], "AUGMENTED", A_REC,
          "Counter(Counter(clip_id).values())[4]")
    short = [c for c, n in per_clip.items() if n < 5]
    have = collections.defaultdict(set)
    for c_, k_ in zip(cid, task):
        have[c_].add(k_)
    miss_task = collections.Counter(
        tk for c_ in short for tk in ({"trajectory", "meta_action", "auto_labeling", "vqa",
                                       "grounding_via_vqa"} - have[c_]))
    L.add("A05.short_task", "task missing on the short clip(s)", dict(miss_task), "AUGMENTED",
          A_REC, "tasks absent for clips with <5 rows")
    err_col = t.column("error")
    L.add("A06", "A2 rows with non-null error", int(nrows - err_col.null_count), "AUGMENTED", A_REC,
          "num_rows - column('error').null_count (column is null-typed)")
    for col, key in (("model_id", "A07.model_id"), ("quantisation", "A07.quantisation"),
                     ("seed", "A07.seed"), ("t0_us", "A07.t0_us")):
        cnt = collections.Counter(t.column(col).to_pylist())
        L.add(key, f"A2 rows per distinct {col}", {str(k): v for k, v in cnt.items()},
              "AUGMENTED", A_REC, f"Counter(column('{col}'))")
        check(len(cnt) == 1, f"{col} not single-valued")

    # manifest ---------------------------------------------------------------
    man = json.loads(P["manifest"].read_text(encoding="utf-8"))
    man_ids = [m["clip_id"] for m in man]
    man_set = set(man_ids)
    L.add("A08.entries", "selection manifest entries", len(man), "AUGMENTED", A_MAN, "len(json list)")
    L.add("A08.unique", "selection manifest distinct clip_id", len(man_set), "AUGMENTED", A_MAN,
          "len(set(clip_id))")
    L.add("A08.t0", "manifest distinct t0_us", sorted({m["t0_us"] for m in man}), "AUGMENTED", A_MAN,
          "sorted(set(t0_us))")
    sel_rows = 5 * len(man_set)
    L.add("A08.rows", "selected clips x 5 tasks", sel_rows, "AUGMENTED", A_MAN, "5 * len(set(clip_id))")
    zero_row = man_set - a2_clips
    off_man = a2_clips - man_set
    L.add("A09.in_manifest", "delivered clips that are in the manifest", len(a2_clips & man_set),
          "AUGMENTED", A_REC + A_MAN, "len(a2 & manifest)")
    L.add("A09.zero_row", "selected clips that produced ZERO rows", len(zero_row), "AUGMENTED",
          A_REC + A_MAN, "len(manifest - a2)")
    L.add("A09.zero_row_rows", "rows those clips would have carried", 5 * len(zero_row), "AUGMENTED",
          A_REC + A_MAN, "5 * len(manifest - a2)")
    L.add("A09.off_manifest", "delivered clips NEVER in the manifest", len(off_man), "AUGMENTED",
          A_REC + A_MAN, "len(a2 - manifest)")
    off_rows = sum(per_clip[c_] for c_ in off_man)
    L.add("A09.off_manifest_rows", "rows carried by the off-manifest clips", off_rows, "AUGMENTED",
          A_REC + A_MAN, "sum(rows per clip in a2 - manifest)")
    short_deficit = sum(5 - per_clip[c_] for c_ in short)
    L.add("A09.short_deficit", "rows missing from partially-delivered clips", short_deficit,
          "AUGMENTED", A_REC, "sum(5 - rows) over clips with <5 rows")
    closure = sel_rows - 5 * len(zero_row) + off_rows - short_deficit
    # NOTE: A10 is a BOOKKEEPING IDENTITY -- it holds by algebra for any manifest, so passing it proves
    # the three terms were counted consistently, not that they are right.  The independent cross-artifact
    # check is D02.union_vs_a2_in_manifest (v7.2 train+eval == A2 delivered-and-in-manifest).
    L.add("A10.closure", "closure: selected rows - zero-row + off-manifest - short = delivered rows",
          closure, "AUGMENTED", A_REC + A_MAN, "24,000 - 405 + 50 - 1")
    check(closure == nrows, f"row closure {closure} != {nrows}")
    check(5 * len(a2_clips) - short_deficit == nrows, "clips*5 - short != rows")
    miss = sel_rows - nrows
    L.add("A11.missing_rows", "rows missing vs selected x 5", miss, "AUGMENTED", A_REC + A_MAN,
          "24,000 - num_rows")
    L.add("A11.missing_pct", "missing rows as % of selected rows", pct(miss, sel_rows), "AUGMENTED",
          A_REC + A_MAN, "100*356/24000")
    L.add("A12.clip_pct", "delivered clips / selected clips (%)", pct(len(a2_clips), len(man_set)),
          "AUGMENTED", A_REC + A_MAN, "100*4729/4800",
          note="numerator includes the off-manifest clips; in-manifest delivered / selected is "
               f"{pct(len(a2_clips & man_set), len(man_set))} %")
    L.add("A12.row_pct", "delivered rows / selected rows (%)", pct(nrows, sel_rows), "AUGMENTED",
          A_REC + A_MAN, "100*23644/24000")
    # card arithmetic: the card values (4,800 clips / 23,999 rows / 'one' missing row) are
    # INHERITED text -- the card itself is not banked in the repo -- so only the arithmetic is ours.
    L.add("A13.card_understatement", "missing rows (measured) / missing rows the card admits ('one')",
          miss / 1, "AUGMENTED", A_REC + A_MAN, "356 / 1", evidence="MEASURED numerator / INHERITED denominator",
          note="card text 'One task row of 24,000 missing (23,999)' is registry §11.1 INHERITED; not re-fetched here")

    # strata -----------------------------------------------------------------
    rc = json.loads(P["road_class"].read_text(encoding="utf-8"))
    classes = rc["classes"]
    L.add("A14.labelled_total", "road-class labelled clips (aug_road_class.json)", len(classes),
          "AUGMENTED", ["road_class"], "len(classes)")
    for scope, ids in (("delivered", a2_clips), ("manifest", man_set)):
        cnt = collections.Counter(classes.get(c_, "unlabelled") for c_ in ids)
        for k in ("urban", "intersection_rich", "highway", "unstructured", "unlabelled"):
            L.add(f"A14.{scope}.{k}", f"{scope} clips, road class {k}", cnt[k], "AUGMENTED",
                  ["road_class"] + (A_REC if scope == "delivered" else A_MAN),
                  f"Counter(classes.get(clip,'unlabelled') for clip in {scope})")
    L.add("A14.zero_row_unlabelled", "zero-row clips that are unlabelled",
          sum(1 for c_ in zero_row if c_ not in classes), "AUGMENTED", ["road_class"] + A_REC + A_MAN,
          "count((manifest - a2) not in classes)")
    L.add("A14.off_manifest_unlabelled", "off-manifest clips that are unlabelled",
          sum(1 for c_ in off_man if c_ not in classes), "AUGMENTED", ["road_class"] + A_REC + A_MAN,
          "count((a2 - manifest) not in classes)")

    lab_deliv = [c_ for c_ in a2_clips if c_ in classes]
    L.add("A14.labelled_complete", "delivered road-class-labelled clips carrying all 5 task rows / delivered labelled clips",
          [sum(1 for c_ in lab_deliv if per_clip[c_] == 5), len(lab_deliv)], "AUGMENTED",
          ["road_class"] + A_REC, "[count(rows==5), len] over delivered clips present in classes")
    check(len(lab_deliv) == len(classes), "labelled delivered != labelled total")

    # cost -------------------------------------------------------------------
    wall = t.column("wall_s").to_pylist()
    tot_s = sum(w for w in wall if w is not None)
    L.add("A15.wall_hours", "sum(wall_s) in hours", round(tot_s / 3600, 2), "AUGMENTED", A_REC,
          "sum(wall_s)/3600")
    L.add("A15.s_per_clip", "sum(wall_s) / distinct clips (s)", round(tot_s / len(a2_clips), 2),
          "AUGMENTED", A_REC, "sum(wall_s)/4729")

    # trajectory variants / meta_action content ------------------------------
    rj = t.column("raw_json").to_pylist()
    var_a = var_b = both = neither = 0
    ade_vs_gt = min_ade = 0
    cots, ans_eq_cot, cot_nonempty = [], 0, 0
    meta_cot: dict[str, str] = {}
    auto_coc: dict[str, object] = {}
    for c_, k_, r_ in zip(cid, task, rj):
        o = json.loads(r_)
        if k_ == "auto_labeling":
            try:
                auto_coc[c_] = json.loads((o.get("cot") or [""])[0]).get("chain_of_causation")
            except (ValueError, AttributeError):
                auto_coc[c_] = None
        if k_ == "trajectory":
            ha = "pred_xyz" in o
            hb = "pred_xyz_shape" in o
            var_a += ha and not hb
            var_b += hb and not ha
            both += ha and hb
            neither += (not ha) and (not hb)
            ade_vs_gt += o.get("ade_vs_gt_m") is not None
            min_ade += o.get("min_ade_m") is not None
        elif k_ == "meta_action":
            c0 = (o.get("cot") or [""])[0]
            cot_nonempty += bool(c0)
            cots.append(c0)
            meta_cot[c_] = c0
            ans_eq_cot += (o.get("answer") or [None])[0] == c0
    L.add("A16.var_a", "trajectory rows with pred_xyz (variant A)", var_a, "AUGMENTED", A_REC,
          "'pred_xyz' in json(raw_json) and 'pred_xyz_shape' not in")
    L.add("A16.var_b", "trajectory rows with pred_xyz_shape (variant B)", var_b, "AUGMENTED", A_REC,
          "'pred_xyz_shape' in json(raw_json) and 'pred_xyz' not in")
    L.add("A16.both_or_neither", "trajectory rows with both / neither variant", [both, neither],
          "AUGMENTED", A_REC, "[both, neither]")
    L.add("A16.ade_vs_gt_m", "trajectory rows with ade_vs_gt_m", ade_vs_gt, "AUGMENTED", A_REC,
          "count(ade_vs_gt_m is not None)")
    L.add("A16.min_ade_m", "trajectory rows with min_ade_m", min_ade, "AUGMENTED", A_REC,
          "count(min_ade_m is not None)")
    L.add("A17.cot_nonempty", "meta_action rows with non-empty cot[0]", cot_nonempty, "AUGMENTED",
          A_REC, "count(json(raw_json)['cot'][0] != '')")
    L.add("A17.cot_distinct", "distinct meta_action cot strings", len(set(cots)), "AUGMENTED", A_REC,
          "len(set(cot[0]))")
    L.add("A17.answer_eq_cot", "meta_action rows where answer[0] == cot[0]", ans_eq_cot, "AUGMENTED",
          A_REC, "count(answer[0]==cot[0])")

    # A2 clips inside the published corpus -----------------------------------
    L.add("PUB3", "A2 delivered clips found in data_collection.parquet", len(a2_clips & dc_set),
          "PUBLISHED", ["dc"] + A_REC, "len(a2 & dc)")
    L.add("PUB4", "A2 clips / published clips (%)", pct(len(a2_clips), len(dc)), "PUBLISHED",
          ["dc"] + A_REC, "100*4729/306152")
    L.add("PUB5", "published clips x 20 s, in hours", round(len(dc) * 20 / 3600, 1), "PUBLISHED", ["dc"],
          "306152*20/3600", evidence="ESTIMATED (20 s/clip assumed, not read from the file)")
    L.add("A18.est_hours", "A2 clips x 20 s, in hours", round(len(a2_clips) * 20 / 3600, 1), "AUGMENTED", A_REC,
          "4729*20/3600", evidence="ESTIMATED (20 s/clip assumed)")

    # ======================================================== EPISODE build ====
    alpa_ids = lines_of(P["alpa_ids"])
    L.add("B01.ids_lines", "alpamayo_clip_ids.txt lines (distinct)", [len(alpa_ids), len(set(alpa_ids))],
          "EPISODE", ["alpa_ids"], "[len(lines), len(set(lines))]")
    L.add("B01.ids_eq_parquet", "alpamayo_clip_ids.txt == parquet clip set", set(alpa_ids) == a2_clips,
          "EPISODE", ["alpa_ids"] + A_REC, "set(lines) == set(clip_id)")
    check(set(alpa_ids) == a2_clips, "alpamayo_clip_ids.txt != parquet clip set")

    pls = [x.split()[-1] for x in lines_of(P["parity_ls"])]
    suffix = ".v2ep.pt"
    ptrain = sorted({n[:-len(suffix)] for n in pls if n.endswith(suffix)})
    L.add("B02.parity_clips", "parity-train cache listing: distinct *.v2ep.pt clip ids", len(ptrain),
          "EPISODE", ["parity_ls"], "len({name[:-8] for name in lines if name.endswith('.v2ep.pt')})",
          note="the committed corpus is 2,376 EPISODES after a 24-clip corrupt skip-set (parity.py:80-81); "
               "this listing and the digest oracle both enumerate 2,400 clip ids")
    oracle = json.loads(P["oracle"].read_text(encoding="utf-8"))
    sdig = hashlib.sha256("\n".join(sorted(ptrain)).encode("utf-8")).hexdigest()
    L.add("B02.listing_reproduces_oracle", "sha256(sorted ids joined by LF) == oracle clip_id_sha256_sorted",
          sdig == oracle["clip_id_sha256_sorted"], "EPISODE", ["parity_ls", "oracle"],
          "sha256('\\n'.join(sorted(ids)))")
    check(sdig == oracle["clip_id_sha256_sorted"], "parity listing does not reproduce the oracle digest")
    L.add("B02.oracle_n", "oracle n_clips / digests", [oracle["n_clips"], len(oracle["clip_id_digests"])],
          "EPISODE", ["oracle"], "[n_clips, len(clip_id_digests)]")

    ov_listing = a2_clips & set(ptrain)
    dig = set(oracle["clip_id_digests"])
    ov_oracle = {c_ for c_ in a2_clips if hashlib.sha256(c_.encode("utf-8")).hexdigest() in dig}
    excl = lines_of(P["excl"])
    L.add("B03.overlap_listing", "A2 clips inside the parity-train clip set (id listing)", len(ov_listing),
          "EPISODE", A_REC + ["parity_ls"], "len(a2 & parity_listing)")
    L.add("B03.overlap_oracle", "same, by the committed sha256 oracle", len(ov_oracle), "EPISODE",
          A_REC + ["oracle"], "count(sha256(clip) in oracle)")
    L.add("B03.excl_list", "banked exclusion list: lines / distinct", [len(excl), len(set(excl))], "EPISODE",
          ["excl"], "[len, len(set)]")
    L.add("B03.excl_eq_overlap", "exclusion list == derived overlap (set equality)",
          set(excl) == ov_listing == ov_oracle, "EPISODE", ["excl", "parity_ls", "oracle"] + A_REC,
          "set(excl) == listing_overlap == oracle_overlap")
    check(set(excl) == ov_listing == ov_oracle and len(ov_listing) == 201, "overlap identity failed")
    L.add("B04.catalogue_pct", "overlap / ALL A2 clips (%) -- the paper's 4.3 %", pct(len(ov_listing), len(a2_clips)),
          "EPISODE", A_REC + ["parity_ls"], "100*201/4729")

    idx = json.loads(P["s2idx"].read_text(encoding="utf-8"))["clips"]
    spl = collections.defaultdict(set)
    for c_, e in idx.items():
        spl[e["label_split"]].add(c_)
    L.add("B05.s2_clips", "S2 clip index: clips / by label_split", {"total": len(idx), **{k: len(v) for k, v in spl.items()}},
          "EPISODE", ["s2idx"], "len(clips), Counter(label_split)")
    built = a2_clips & set(idx)
    L.add("B05.a2_with_w120", "A2 clips that have a w120 cache (A2 ∩ S2 clip index)", len(built), "EPISODE",
          A_REC + ["s2idx"], "len(a2 & set(clip_index.clips))",
          note="'has w120 video' is read from the S2 clip index (which enumerates the built w120 set), "
               "not from the video files, which are pod/Thor-side")
    L.add("B05.a2_w120_by_split", "…of which per label_split",
          {k: len(a2_clips & v) for k, v in spl.items()}, "EPISODE", A_REC + ["s2idx"],
          "len(a2 & split) per split")
    L.add("B05.no_w120", "A2 clips with NO w120 cache", len(a2_clips) - len(built), "EPISODE",
          A_REC + ["s2idx"], "4729 - 257")
    L.add("B05.no_w120_pct", "…as % of A2", pct(len(a2_clips) - len(built), len(a2_clips)), "EPISODE",
          A_REC + ["s2idx"], "100*4472/4729")
    bad_built = {c_ for c_ in built if hashlib.sha256(c_.encode("utf-8")).hexdigest() in dig}
    L.add("B06.buildable_overlap", "buildable A2 clips inside parity train", len(bad_built), "EPISODE",
          A_REC + ["s2idx", "oracle"], "count(sha256(clip) in oracle for clip in a2 & s2idx)")
    L.add("B06.buildable_pct", "buildable contamination (%) -- the registry §12.4 'quote 78.21 %'",
          pct(len(bad_built), len(built)), "EPISODE", A_REC + ["s2idx", "oracle"], "100*201/257")
    check(len(built) == 257 and len(bad_built) == 201, "257/201 identity failed")
    for k, v in spl.items():
        inp = sum(1 for c_ in v if hashlib.sha256(c_.encode("utf-8")).hexdigest() in dig)
        L.add(f"B07.{k}_in_parity", f"S2 split {k}: clips inside parity train / clips", [inp, len(v)],
              "EPISODE", ["s2idx", "oracle"], "count(sha256(clip) in oracle) / len(split)")

    v40 = json.loads(P["val40"].read_text(encoding="utf-8"))
    v40a = json.loads(P["val40_anon"].read_text(encoding="utf-8"))
    v40_ids = {e["clip_id"] for e in v40.values()}
    agree = sum(1 for ep, e in v40.items()
                if "clip_" + hashlib.sha256(e["clip_id"].encode()).hexdigest()[:8] == v40a[ep]["clip_sha8"])
    L.add("B08.val40", "canonical val episodes / inside A2 records / sha8 cross-check agreements / in parity train",
          [len(v40_ids), len(v40_ids & a2_clips), f"{agree}/{len(v40)}",
           sum(1 for c_ in v40_ids if hashlib.sha256(c_.encode()).hexdigest() in dig)],
          "EPISODE", ["val40", "val40_anon"] + A_REC + ["oracle"],
          "[len(val40), len(val40 & a2), sha8 agreements, in-oracle count]")

    # ================================================== AUGMENTED: aug120 PH1 ==
    cov = json.loads(P["a120_cov"].read_text(encoding="utf-8"))
    L.add("C01.cov", "aug120 coverage lists (todo / v2_union / s3_union / no_v2 / no_s3 / no_ego / b184)",
          {k: len(v) for k, v in cov.items()}, "AUGMENTED", ["a120_cov"], "len(list) per key")
    aug_ids = set(cov["todo"])
    L.add("C02.sets_equal", "aug120 todo == v2_union == S2 aug120 split == banked exclusion list",
          aug_ids == set(cov["v2_union_ids"]) == spl["aug120"] == set(excl), "AUGMENTED",
          ["a120_cov", "s2idx", "excl"], "set equality of four id sets")
    check(aug_ids == set(cov["v2_union_ids"]) == spl["aug120"] == set(excl), "aug120 set identity failed")
    src = json.loads(P["a120_src"].read_text(encoding="utf-8"))["sources"]
    sam3_nonnull = sum(1 for e in src.values() if e.get("sam3"))
    L.add("C03.sam3", "label_sources.json: clips / with a SAM3 source / without",
          [len(src), sam3_nonnull, len(src) - sam3_nonnull], "AUGMENTED", ["a120_src"],
          "[len(sources), count(sam3 truthy), count(sam3 null)]")
    acct = json.loads(P["a120_acct"].read_text(encoding="utf-8"))
    sums = {k: sum(b[k] for b in acct.values()) for k in
            ("clips_attributed_here", "with_alpamayo", "sam3_absent", "sam3_attributed_here",
             "corroborated", "conflicts")}
    L.add("C04.acct", f"batch accounting summed over {len(acct)} batches", sums, "AUGMENTED", ["a120_acct"],
          "sum(batch[k]) per key")
    summ = json.loads(P["a120_sum"].read_text(encoding="utf-8"))
    L.add("C05.summary", "fused_aug120_summary.json", summ, "AUGMENTED", ["a120_sum"], "json.load")
    check(sums["clips_attributed_here"] == 201 and sums["sam3_absent"] == len(src) - sam3_nonnull == 115,
          "aug120 accounting does not close")
    check(summ["n_fused"] == sums["clips_attributed_here"] and summ["sam3_missing"] == sums["sam3_absent"]
          and summ["corroborated"] == sums["corroborated"] and summ["conflicts"] == sums["conflicts"],
          "summary json disagrees with batch accounting")
    L.add("C06.sam3_pct", "SAM3 coverage % / gap %", [pct(sam3_nonnull, len(src), 2),
                                                      pct(len(src) - sam3_nonnull, len(src), 2)],
          "AUGMENTED", ["a120_src"], "100*86/201, 100*115/201")
    L.add("C07.gap_ratio", "true SAM3 gap / the '8 clips' first recorded (cov['b184'])",
          round((len(src) - sam3_nonnull) / len(cov["b184"]), 3), "AUGMENTED", ["a120_src", "a120_cov"],
          "115 / len(b184)")
    sub = [(c_, k_) for c_, k_ in zip(cid, task) if c_ in aug_ids]
    L.add("C08.a2_join_rows", "A2 parquet rows for the 201 aug120 clips / per task",
          [len(sub), dict(collections.Counter(k_ for _, k_ in sub))], "AUGMENTED", ["a120_cov"] + A_REC,
          "count(rows with clip in aug120) / Counter(task)")
    L.add("C09.a2_join_clips", "aug120 clips present in the A2 parquet", len(aug_ids & a2_clips),
          "AUGMENTED", ["a120_cov"] + A_REC, "len(aug120 & a2)")

    # ================================================== AUGMENTED: v7.2 release ==
    def load_jsonl_gz(p: Path) -> list[dict]:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    tr = load_jsonl_gz(P["v72_train"])
    ev = load_jsonl_gz(P["v72_eval"])
    tr_ids = [r["clip_id"] for r in tr]
    ev_ids = [r["clip_id"] for r in ev]
    L.add("D01.train", "v7.2 train: records / distinct clip_id", [len(tr), len(set(tr_ids))], "AUGMENTED",
          ["v72_train"], "[len(records), len(set(clip_id))]")
    L.add("D01.eval", "v7.2 eval: records / distinct clip_id", [len(ev), len(set(ev_ids))], "AUGMENTED",
          ["v72_eval"], "[len(records), len(set(clip_id))]")
    for col in ("release", "schema_version", "vocab"):
        c3 = collections.Counter((r[col] for r in tr + ev))
        L.add(f"D01.{col}", f"v7.2 train+eval: distinct {col}", dict(c3), "AUGMENTED",
              ["v72_train", "v72_eval"], f"Counter(record['{col}'])")
    L.add("D02.train_in_a2", "v7.2 train clips present in A2 parquet", len(set(tr_ids) & a2_clips),
          "AUGMENTED", ["v72_train"] + A_REC, "len(train & a2)")
    L.add("D02.eval_in_a2", "v7.2 eval clips present in A2 parquet", len(set(ev_ids) & a2_clips),
          "AUGMENTED", ["v72_eval"] + A_REC, "len(eval & a2)")
    L.add("D02.train_cap_eval", "v7.2 train ∩ eval", len(set(tr_ids) & set(ev_ids)), "AUGMENTED",
          ["v72_train", "v72_eval"], "len(train & eval)")
    union = set(tr_ids) | set(ev_ids)
    L.add("D02.union_vs_a2_in_manifest", "v7.2 (train ∪ eval) / A2 ∩ manifest / equal?",
          [len(union), len(a2_clips & man_set), union == (a2_clips & man_set)], "AUGMENTED",
          ["v72_train", "v72_eval"] + A_REC + A_MAN, "[len(train|eval), len(a2&manifest), equal]")
    L.add("D02.train_in_parity", "v7.2 train clips inside parity train (oracle)",
          sum(1 for c_ in set(tr_ids) if hashlib.sha256(c_.encode()).hexdigest() in dig), "AUGMENTED",
          ["v72_train", "oracle"], "count(sha256(clip) in oracle)")

    ov_tr, ov_ev, ov_off = ov_listing & set(tr_ids), ov_listing & set(ev_ids), ov_listing & off_man
    L.add("D02.parity_overlap_breakdown",
          "the 201 A2 clips inside parity train, split by where they sit: [total, in v7.2 train, in v7.2 eval, "
          "off-manifest, in none of the three]",
          [len(ov_listing), len(ov_tr), len(ov_ev), len(ov_off), len(ov_listing - ov_tr - ov_ev - ov_off)],
          "AUGMENTED", ["v72_train", "v72_eval", "parity_ls"] + A_REC + A_MAN,
          "[|overlap|, |overlap&train|, |overlap&eval|, |overlap&(a2-manifest)|, remainder]")

    def tl_census(recs: list[dict]) -> dict:
        keys = collections.Counter()
        prov = collections.defaultdict(collections.Counter)
        any_tl = col_tl = colourless_only = 0
        states = collections.Counter()
        for r in recs:
            g = r["g_tac"]["goals"]
            tl = [k for k in g if k.startswith("TRAFFIC_LIGHT_REACT")]
            for k in tl:
                keys[k] += 1
                prov[k][g[k].get("provenance")] += 1
                states[(k, g[k].get("state"), g[k].get("object_kind"))] += 1
            any_tl += bool(tl)
            coloured = [k for k in tl if k != "TRAFFIC_LIGHT_REACT"]
            col_tl += bool(coloured)
            colourless_only += bool(tl) and not coloured
        return {"records": len(recs), "goal_counts": dict(keys),
                "provenance": {k: dict(v) for k, v in prov.items()},
                "state_object_kind": {f"{k}|{s}|{o}": n for (k, s, o), n in states.items()},
                "records_with_any_TL_goal": any_tl,
                "records_with_a_coloured_TL_goal": col_tl,
                "records_with_only_the_colourless_TL_goal": colourless_only,
                "sum_of_goal_counts": sum(keys.values())}

    tlt = tl_census(tr)
    L.add("D03.train_TL", "v7.2 train: traffic-light tactical goals", tlt, "AUGMENTED", ["v72_train"],
          "per record: [k for k in record['g_tac']['goals'] if k.startswith('TRAFFIC_LIGHT_REACT')]")
    L.add("D03.eval_TL", "v7.2 eval: traffic-light tactical goals", tl_census(ev), "AUGMENTED",
          ["v72_eval"], "same expression over the eval file")
    check(tlt["records_with_any_TL_goal"] == 779 and tlt["sum_of_goal_counts"] == 779,
          "v7.2 TL census no longer reads 779")

    prov_all = collections.Counter()
    rec_vlm = 0
    for r in tr:
        h = False
        for k, v in r["g_tac"]["goals"].items():
            prov_all[v.get("provenance")] += 1
            h |= v.get("provenance") == "vlm-cot"
        rec_vlm += h
    L.add("D04.goal_provenance", "v7.2 train: g_tac.goals entries by provenance", dict(prov_all),
          "AUGMENTED", ["v72_train"], "Counter(goal['provenance'] for all goals)")
    L.add("D04.records_with_vlm_cot_goal", "v7.2 train: records with >=1 provenance=='vlm-cot' goal",
          rec_vlm, "AUGMENTED", ["v72_train"], "count(any(goal.provenance=='vlm-cot'))")
    for fld in ("cot", "chain_of_causation", "meta_action"):
        L.add(f"D05.cot_source.{fld}", f"v7.2 train: records with non-empty cot_source.{fld}",
              sum(1 for r in tr if r["cot_source"].get(fld)), "AUGMENTED", ["v72_train"],
              f"count(bool(record['cot_source']['{fld}']))")
    for nm, recs, pid in (("train", tr, "v72_train"), ("eval", ev, "v72_eval")):
        L.add(f"D06.cot_eq_a2.{nm}",
              f"v7.2 {nm}: records whose cot_source.cot is byte-equal to the A2 meta_action cot / records",
              [sum(1 for r in recs if r["cot_source"]["cot"] == meta_cot.get(r["clip_id"])), len(recs)],
              "AUGMENTED", [pid] + A_REC, "count(record.cot_source.cot == A2[clip].meta_action.cot[0])")
        L.add(f"D06.coc_eq_a2.{nm}",
              f"v7.2 {nm}: records whose cot_source.chain_of_causation is byte-equal to the A2 auto_labeling "
              "chain_of_causation / records",
              [sum(1 for r in recs if r["cot_source"]["chain_of_causation"] == auto_coc.get(r["clip_id"])),
               len(recs)],
              "AUGMENTED", [pid] + A_REC,
              "count(record.cot_source.chain_of_causation == json(A2[clip].auto_labeling.cot[0])['chain_of_causation'])")
        check(all(r["cot_source"]["cot"] == meta_cot.get(r["clip_id"]) for r in recs),
              f"v7.2 {nm} cot != A2 meta_action cot")
    L.add("D05.scene_provenance", "v7.2 train: scene.provenance values",
          dict(collections.Counter(r["scene"].get("provenance") for r in tr)), "AUGMENTED",
          ["v72_train"], "Counter(record['scene']['provenance'])")

    # ============================================== paper / registry scanning ==
    paper_lines = P["paper"].read_text(encoding="utf-8").splitlines()
    reg_lines = P["registry"].read_text(encoding="utf-8").splitlines()
    ctx = re.compile(r"(?i)alpamayo|augmentation|aug120|sam3|PH1|fused|\bA2\b|manifest|7\.15|11\.[12]|card")
    # (claim id, count id, paper lines (curated by reading), forms that must appear on those lines,
    #  claimed value, derived value, comparison kind)
    claims = []

    def claim(cl, cnt, plines, forms, claimed, derived, kind="eq", note=""):
        claims.append({"claim": cl, "count": cnt, "paper_lines": plines, "forms": forms,
                       "claimed": claimed, "derived": derived, "kind": kind, "note": note})

    claim("PC01 A2 delivered clips", "A02", [198, 2518, 3068, 3886, 3891, 3894, 6488, 6599], ["4,729"],
          4729, L.v("A02"))
    claim("PC02 A2 delivered rows", "A01", [2518, 6488], ["23,644"], 23644, L.v("A01"))
    claim("PC03 rows per task (x4)", "A04.trajectory", [2519], ["4,729"], 4729, L.v("A04.trajectory"))
    claim("PC04 grounding_via_vqa rows", "A04.grounding_via_vqa", [2520], ["4,728"], 4728,
          L.v("A04.grounding_via_vqa"))
    claim("PC05 missing rows", "A11.missing_rows", [2526], ["356"], 356, L.v("A11.missing_rows"))
    claim("PC06 missing rows %", "A11.missing_pct", [2526], ["1.48"], 1.48, L.v("A11.missing_pct"), "round2")
    claim("PC07 zero-row clips", "A09.zero_row", [2526], ["81"], 81, L.v("A09.zero_row"))
    claim("PC08 zero-row rows (-405)", "A09.zero_row_rows", [2526], ["405"], 405, L.v("A09.zero_row_rows"))
    claim("PC09 off-manifest clips", "A09.off_manifest", [2526], ["10"], 10, L.v("A09.off_manifest"))
    claim("PC10 off-manifest rows (+50)", "A09.off_manifest_rows", [2527], ["50"], 50, L.v("A09.off_manifest_rows"))
    claim("PC11 short-clip deficit (-1)", "A09.short_deficit", [2527], ["one"], 1, L.v("A09.short_deficit"))
    claim("PC12 urban", "A14.delivered.urban", [2528], ["1,884"], 1884, L.v("A14.delivered.urban"))
    claim("PC13 intersection_rich", "A14.delivered.intersection_rich", [2528], ["1,241"], 1241,
          L.v("A14.delivered.intersection_rich"))
    claim("PC14 highway", "A14.delivered.highway", [2528], ["384"], 384, L.v("A14.delivered.highway"))
    claim("PC15 unstructured", "A14.delivered.unstructured", [2529], ["83"], 83,
          L.v("A14.delivered.unstructured"))
    claim("PC16 aug120 fused clips", "C04.acct", [2535], ["201/201"], 201, sums["clips_attributed_here"])
    claim("PC17 Alpamayo layer on 201/201", "C04.acct", [2536], ["201/201"], 201, sums["with_alpamayo"])
    claim("PC18 SAM3 covered", "C03.sam3", [2536], ["86/201"], 86, sam3_nonnull)
    claim("PC19 SAM3 coverage %", "C06.sam3_pct", [2536], ["42.8"], 42.8, L.v("C06.sam3_pct")[0], "round1")
    claim("PC20 SAM3 gap clips", "C03.sam3", [2536], ["115"], 115, len(src) - sam3_nonnull)
    claim("PC21 gap first recorded as 8 clips", "C01.cov", [2538], ["8 clips"], 8, len(cov["b184"]))
    claim("PC22 gap ratio 14.4x", "C07.gap_ratio", [2538], ["14.4"], 14.4, L.v("C07.gap_ratio"), "round1")
    claim("PC23 SAM3 gap % (changelog)", "C06.sam3_pct", [6489], ["57.2"], 57.2, L.v("C06.sam3_pct")[1], "round1")
    claim("PC24 A2 clips inside parity train", "B03.overlap_listing", [198, 3068, 3072, 3894, 3923, 6599], ["201"],
          201, L.v("B03.overlap_listing"))
    claim("PC25 '4.3 % contaminated' (any eval split built on the augmentation)", "B06.buildable_pct",
          [199, 3073, 3924, 6600], ["4.3"], 4.3, L.v("B06.buildable_pct"), "CONFLICT_DENOM",
          "4.3 % = 201/4,729 (B04, catalogue denominator) -- reproduces arithmetically, but the registry §12.4 "
          "(C113) and this recount show a split can only contain BUILT clips: 201/257")
    claim("PC26 the card's 4,800 / 23,999 (retracted by the paper)", "A08.entries", [2523, 3887, 3892, 6488],
          ["4,800"], 4800, L.v("A08.entries"), "eq",
          "4,800 is the SELECTION MANIFEST size (measured), not the delivered clip count; "
          "the paper uses it correctly as 'the card's number'")

    cl_out = []
    for c in claims:
        found = {}
        for ln in c["paper_lines"]:
            text = paper_lines[ln - 1] if 0 < ln <= len(paper_lines) else ""
            ok = any(form_regex(f).search(text) or f in text for f in c["forms"])
            found[ln] = ok
        d, cv = c["derived"], c["claimed"]
        if c["kind"] == "eq":
            status = "MATCH" if d == cv else "CONFLICT"
        elif c["kind"] == "round2":
            status = "MATCH" if round(d, 2) == cv else "CONFLICT"
        elif c["kind"] == "round1":
            status = "MATCH" if round(d, 1) == cv else "CONFLICT"
        elif c["kind"] == "CONFLICT_DENOM":
            status = ("ARITHMETIC-MATCH-DENOMINATOR-CONFLICT"
                      if round(L.v("B04.catalogue_pct"), 1) == cv and round(d, 1) != cv else "CHECK")
        else:
            status = "?"
        cl_out.append({**c, "line_has_form": found, "status": status})
        if not a.skip_paper_pins:
            check(all(found.values()), f"{c['claim']}: form not on curated paper lines {found}")

    # full-text locator: where does each headline number appear in paper / registry?
    scan_ints = [4729, 23644, 4728, 4800, 23999, 356, 4472, 3592, 1884, 1241, 306152, 4572, 779, 201, 257]
    loc = {}
    for n in scan_ints:
        forms = int_forms(n)
        strict = n < 1000
        cx = ctx if strict else None
        loc[str(n)] = {"forms": forms,
                       "paper_lines": hit_lines(paper_lines, forms, cx),
                       "registry_lines": hit_lines(reg_lines, forms, cx)}
    for s in ("78.21", "4.3 %", "98.52", "1.48", "42.8", "57.2", "14.4"):
        loc[s] = {"forms": [s], "paper_lines": hit_lines(paper_lines, [s], ctx),
                  "registry_lines": hit_lines(reg_lines, [s], ctx)}

    out = {
        "_what": "BACKLOG E10 -- Alpamayo-augmentation counts re-derived from primary bytes",
        "_date": DATE,
        "_script": "count_alpamayo.py",
        "_python": sys.version.split()[0],
        "_pyarrow": pa.__version__,
        "_assertions_failed": FAILS,
        "paper_base_blob": art["paper"]["git_blob_id"],
        "registry_base_blob": art["registry"]["git_blob_id"],
        "artifacts": art,
        "counts": L.rows,
        "paper_claims": cl_out,
        "number_locator": loc,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {len(L.rows)} counts, {len(cl_out)} paper claims, "
          f"{len(FAILS)} assertion failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
