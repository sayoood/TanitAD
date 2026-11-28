#!/usr/bin/env python3
"""Per-chunk DUMP PLANS for the navtest pre-head FEATURE BANK that `refe_head_confirm.py` needs.  CPU only, no GPU, no outcome.

WHY.  The landed full navtest table (`table_p2_full.npz`, 12,146 tokens / 136 logs) carries P2's stored logits and proposals but NOT the input of `score_head`; its 14 chunk dumps carry none either
(header-listed: token, fingerprint, proposals, logits, pick, ...).  A refit row can only be applied to navtest if `feat [N,64,256]` exists.  The only forward that produces it is
`srttc_s0_dump.py` in its FAN path (no --sets-dir, NOT --lean): `--feat-out` writes `prehead_features` fp16 [n,64,256] captured by a forward hook in the SAME forward as the logits.  It was proven on 4 navtest
tokens against the landed table (DG-DUMP: proposals max abs 0.0 m, logits max abs 0.0, pick equal, negative control FAIL).  This script only prepares its inputs from the landed chunk definitions:

  * reads `chunks.json` + `cNN_tokens.json` of the landed full-table package and VERIFIES each token file's sha256 against `chunks.json`;
  * reads ONLY the `token` and `log` members of the landed table (sha256 pinned) and checks the chunks partition its tokens exactly once and that each chunk's own token_log equals the table's;
  * checks every log's nuPlan DB exists;
  * writes `plan_cNN.json` in the shape `srttc_s0_dump.py --plan` takes ({tokens, token_log, token_root, db}) and `bank_commands.txt`: the exact command per chunk.

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I make_navtest_bank_plan.py --chunks-dir <raw/2026-10-16-p2-full-table/chunks> --out-dir <plan dir> --bank-dir <bank dir>

THE FORWARD IS NOT RUN HERE.  It needs the dev-box GPU (queued behind S0, `gpu_lock.py acquire`): see NAVTEST_BANK_SCOPING.md for hours and disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1_dev as D            # noqa: E402
import refe_head_confirm as C   # noqa: E402

NAVTEST_DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
NAVTEST_FRAMES = "D:/Projects/TanitAD/data/refe_navtest/frames"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
DRIVERL_PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
DUMP_SCRIPT = "srttc_s0_dump.py"
N_TOKENS, N_LOGS, N_CHUNKS = 12146, 136, 14


class PlanRefusal(RuntimeError):
    pass


def load_chunks(guard, chunks_dir: str, expect_chunks: int | None = N_CHUNKS) -> list:
    """-> [(chunk id, tokens list, token_log dict, sha256)] in chunk order, each token file verified against chunks.json."""
    doc, _ = D.load_json(guard, os.path.join(chunks_dir, "chunks.json"))
    ents = doc["chunks"]
    if expect_chunks is not None and len(ents) != expect_chunks:
        raise PlanRefusal(f"chunks.json lists {len(ents)} chunks, expected {expect_chunks}")
    if [e["id"] for e in ents] != list(range(len(ents))):
        raise PlanRefusal("chunk ids are not 0..n-1 in order")
    out = []
    for e in ents:
        raw = D._io_read_bytes(guard, os.path.join(chunks_dir, e["tokens_file"]))
        sha = D.sha256_bytes(raw)
        if sha != e["tokens_sha256"]:                                                          # MUTATION-TARGET-PLAN-SHA
            raise PlanRefusal(f"{e['tokens_file']}: sha256 {sha} != chunks.json {e['tokens_sha256']}")
        j = json.loads(raw.decode("utf-8"))
        toks = [str(t) for t in j["tokens"]]
        if len(toks) != e["n"] or len(set(toks)) != len(toks):
            raise PlanRefusal(f"{e['tokens_file']}: {len(toks)} tokens (distinct {len(set(toks))}), chunks.json says {e['n']}")
        tl = {str(k): str(v) for k, v in j["token_log"].items()}
        if sorted(tl) != sorted(toks):
            raise PlanRefusal(f"{e['tokens_file']}: token_log keys differ from tokens")
        out.append((e["id"], toks, tl, sha))
    return out


def table_token_log(guard, table: str, table_sha256: str | None) -> tuple:
    """Opens ONLY `token` and `log` (never pdms / sub / valid)."""
    if table_sha256 is not None:
        sha = D._io_sha256(guard, table)
        if sha != table_sha256:
            raise PlanRefusal(f"table sha256 {sha} != the pinned {table_sha256}")
    T = D._io_npz(guard, table)
    tok, log = [str(x) for x in T["token"]], [str(x) for x in T["log"]]
    T.close()
    return tok, log


def build(chunks_dir: str, table: str, out_dir: str, bank_dir: str, table_sha256: str | None = C.NAVTEST_TABLE_SHA256, ckpt: str = CKPT, frames_root: str = NAVTEST_FRAMES,
          db_dir: str = NAVTEST_DB, python: str = DRIVERL_PY, expect_tokens: int | None = N_TOKENS, expect_logs: int | None = N_LOGS, expect_chunks: int | None = N_CHUNKS,
          check_db: bool = True) -> dict:
    guard = C.ConfirmGuard(read_dirs=[chunks_dir, db_dir], read_files=[table], write_dirs=[out_dir])
    chunks = load_chunks(guard, chunks_dir, expect_chunks)
    tok, log = table_token_log(guard, table, table_sha256)
    if expect_tokens is not None and len(tok) != expect_tokens:
        raise PlanRefusal(f"table has {len(tok)} tokens, expected {expect_tokens}")
    tlog = dict(zip(tok, log))
    if len(tlog) != len(tok):
        raise PlanRefusal("table has duplicate tokens")
    if expect_logs is not None and len(set(log)) != expect_logs:
        raise PlanRefusal(f"table has {len(set(log))} logs, expected {expect_logs}")
    seen = {}
    for cid, toks, tl, _ in chunks:
        for t in toks:
            if t in seen:                                                                      # MUTATION-TARGET-PLAN-DUP
                raise PlanRefusal(f"token {t} is in chunks {seen[t]} and {cid}")
            seen[t] = cid
            if t not in tlog:
                raise PlanRefusal(f"chunk {cid}: token {t} is not a table token")
            if tl[t] != tlog[t]:                                                               # MUTATION-TARGET-PLAN-LOG
                raise PlanRefusal(f"chunk {cid}: token {t} log {tl[t]} != the table's {tlog[t]}")
    missing = [t for t in tok if t not in seen]
    if missing:                                                                                # MUTATION-TARGET-PLAN-MISSING
        raise PlanRefusal(f"{len(missing)} table tokens are in no chunk (first {missing[:3]})")
    if check_db:
        for lg in sorted(set(log)):
            if not D._io_exists(guard, f"{db_dir}/{lg}.db"):
                raise PlanRefusal(f"nuPlan DB for log {lg} does not exist under {db_dir}")
    D._io_makedirs(guard, out_dir)
    cmds, entries = [], []
    for cid, toks, tl, sha in chunks:
        logs = sorted(set(tl.values()))
        plan = {"tokens": toks, "token_log": {t: tl[t] for t in toks}, "token_root": {t: frames_root for t in toks}, "db": {lg: f"{db_dir}/{lg}.db" for lg in logs}}
        pp = os.path.join(out_dir, f"plan_c{cid:02d}.json")
        psha = D._io_write_bytes(guard, pp, json.dumps(plan, indent=0).encode("utf-8"), exclusive=True)
        cmds.append(f'{python} -I {DUMP_SCRIPT} --ckpt {ckpt} --plan {pp.replace(chr(92), "/")} --out {bank_dir}/c{cid:02d}_dump.npz --feat-out {bank_dir}/c{cid:02d}_feat.npz --device cuda')
        entries.append({"chunk": cid, "n": len(toks), "n_logs": len(logs), "tokens_sha256": sha, "plan": pp.replace("\\", "/"), "plan_sha256": psha})
    D._io_write_bytes(guard, os.path.join(out_dir, "bank_commands.txt"), ("\n".join(cmds) + "\n").encode("utf-8"), exclusive=True)
    man = {"what": "dump plans for the navtest pre-head feature bank (srttc_s0_dump.py fan path, NOT --lean)", "table_sha256": table_sha256, "n_tokens": len(tok), "n_logs": len(set(log)),
           "n_chunks": len(chunks), "ckpt": ckpt, "chunks": entries,
           "after_the_forward": "refe_head_confirm.py manifest --bank-dir <bank>, then refe_head_confirm.py picks|read",
           "note": "NOT --lean: lean drops the features and the fan; run each command from the sel-rank code dir with the driverl-eval env of srttc_s0_dgdump.py, under `gpu_lock.py acquire`"}
    D._io_write_bytes(guard, os.path.join(out_dir, "bank_plan_manifest.json"), D.canon(man), exclusive=True)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--table", default=C.NAVTEST_TABLE)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--no-db-check", action="store_true")
    a = ap.parse_args(argv)
    try:
        man = build(a.chunks_dir, a.table, a.out_dir, a.bank_dir, ckpt=a.ckpt, check_db=not a.no_db_check)
    except (PlanRefusal, D.DevRefusal, D.DevGuardError) as ex:
        print(f"REFUSED: {ex}")
        return 2
    print(f"wrote {man['n_chunks']} plans for {man['n_tokens']} tokens / {man['n_logs']} logs into {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
