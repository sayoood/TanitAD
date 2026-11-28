#!/usr/bin/env python3
"""Pod-side helpers for post_merge_to_p6gate.sh.   stdlib only (python >= 3.8), except `parts`, which imports the P6 tree's own online_labeller (numpy).

Run a READ-ONLY subcommand without writing a file on the pod:      python3 - <subcommand> [args] < pod_helpers.py
Every verdict is ONE line that starts with `ZZV ` followed by `<kind> key=value ...` (values never contain a space).  The orchestrator parses ONLY those lines,
so a traceback, a warning or an echoed command can never be mistaken for a verdict (TRAPS: a filter must not contain the text it searches for).

Subcommands (all READ-ONLY unless marked):
  gpu                                            GPU compute processes (nvidia-smi); an nvidia-smi that cannot answer counts as BUSY
  locks   --data D [--exclude NAME ...]          which `<D>/*/chain.lock` are held by a live chain (flock probe on a READ-WRITE fd, the p6_pod.sh idiom)
  merge   --f F --out OUT --inp IN --prereg-sha H [--quick]
                                                 judge the 3-C merge from ARTIFACTS (not from an rc nobody saved): state ok|yield|refused|merging|crashed|... + facts
  c2      --f F --out OUT                        judge (c'2): outcome pass|slow|notjudged|incomplete|stale|missing
  c3      --f F --ks 1,2,3                       judge (c'3): D7 verify (a) and (b) per rank; (c) REPORTED, never judged
  p6tree  --pkg P6PKG                            facts of the shipped P6 tree: ship commit, p6_pod.sh md5, CBANK knob, R43 line
  parts   --bank OUT --src MC_TWIN --dst MC --p6refe DIR --mode plan|apply [--how auto|link|copy] [--report FILE]
                                                 step (d).  `plan` is READ-ONLY.  `apply` WRITES (hard link or atomic copy of the KEPT twins' parts into MC).
  gate    --lg LG --out OUT --commit C --mc MC --kept N --since-file REF --p6 P6DIR --ranks 1,2,3 --chain-offset N
                                                 judge the P6 dry-run's token and chain log
  busy    --data D [--no-lock-probe]             is the pod FREE for P6?  idle iff no GPU compute process AND no other chain lock AND no running rls1_pod.sh (R0d)
  launchwatch --p6 P6DIR --run RUNDIR --chain-offset N --metrics-before N
                                                 after the REAL launch: refused | trainer_exited | preflight_ok | training (first step line) | ambiguous | gone
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KEEP_F10 = 0.75
YIELD_GUARD_K1 = 0.20
YIELD_GUARD_MIN_N = 30


def emit(kind: str, **kw) -> None:
    parts = []
    for k, v in kw.items():
        s = str(v).replace(" ", "_").replace("\n", "_")
        parts.append(f"{k}={s}")
    print("ZZV " + kind + " " + " ".join(parts), flush=True)


def sha256_file(p, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------------------------------------------------- gpu / locks
def cmd_gpu(a) -> int:
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    except Exception as e:                                                              # noqa: BLE001 -- cannot say == busy (p6_pod.sh gpu_busy)
        emit("gpu", state="unknown", n=-1, why=type(e).__name__)
        return 0
    if p.returncode != 0:
        emit("gpu", state="unknown", n=-1, why=f"nvidia-smi_rc{p.returncode}")
        return 0
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    names = ";".join(os.path.basename(ln.split(",")[1].strip()) if "," in ln else "?" for ln in lines[:4])
    emit("gpu", state="busy" if lines else "idle", n=len(lines), procs=names or "-")
    return 0


def cmd_locks(a) -> int:
    import fcntl
    held, free = [], 0
    root = Path(a.data)
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        lk = d / "chain.lock"
        if d.name in (a.exclude or []) or not lk.is_file():
            continue
        try:
            fd = os.open(str(lk), os.O_RDWR)                     # READ-WRITE: an exclusive flock on a read-only fd is EBADF on the pod's nfs4
        except OSError as e:
            held.append(f"{d.name}(unopenable:{e.errno})")
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free += 1
        except OSError:
            held.append(d.name)
        finally:
            os.close(fd)                                          # closing releases the probe's lock at once
    emit("locks", held=",".join(held) or "-", n_held=len(held), n_free=free)
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- merge
def _proc_alive(needle_a: str, needle_b: str) -> bool:
    """A live process whose argv holds both needles (scans /proc; the helper's own argv holds neither: it is `python3 - merge ...`)."""
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            argv = (d / "cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        if needle_a in argv and needle_b in argv:
            return True
    return False


def _jsonl(p):
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                yield json.loads(ln)


def cmd_merge(a) -> int:
    F, OUT, IN = Path(a.f), Path(a.out), Path(a.inp)
    if (F / "RUN_INCOMPLETE").exists():
        emit("merge", state="run_incomplete", why=(F / "RUN_INCOMPLETE").read_text(errors="replace").strip()[:120])
        return 0
    if not (F / "RUN_DONE").exists():
        emit("merge", state="run_not_done")
        return 0
    mo = F / "merge.out"
    if not mo.exists():
        emit("merge", state="not_started")
        return 0
    txt = mo.read_text(errors="replace")
    if "ZZ3C_YIELD_GUARD_TRIPPED" in txt:
        emit("merge", state="yield", why=re.sub(r"\s+", "_", txt.strip().splitlines()[-1])[:200])
        return 0
    m_ref = re.search(r"^ZZ3C_MERGE_REFUSED.*$", txt, re.M)
    if m_ref:
        emit("merge", state="refused", why=re.sub(r"\s+", "_", m_ref.group(0))[:200])
        return 0
    m_ok = re.search(r"^ZZ3C_MERGE_OK .*$", txt, re.M)
    if not m_ok:
        alive = _proc_alive("reroll_bank_3c.py", "merge")
        emit("merge", state="merging" if alive else "crashed", why=re.sub(r"\s+", "_", (txt.strip().splitlines() or ["empty"])[-1])[:160])
        return 0

    def bad(why: str) -> int:
        emit("merge", state="invalid", why=why)
        return 0

    if (OUT / "STOP_YIELD_GUARD.txt").exists() or (OUT / "targets_rank1.jsonl.REJECTED_YIELD_GUARD").exists():
        return bad("merge.out says OK but the output carries a yield-guard rejection")
    t1, rep_p = OUT / "targets_rank1.jsonl", OUT / "merge_report.json"
    if not (t1.is_file() and rep_p.is_file() and (OUT / "audit_all.jsonl").is_file()):
        return bad("merge.out says OK but targets_rank1.jsonl / merge_report.json / audit_all.jsonl is missing")
    rep = read_json(rep_p)
    if rep.get("prereg_sha256") != a.prereg_sha:
        return bad("merge_report.prereg_sha256_is_not_the_frozen_PREREG_3C")
    yg = rep.get("yield_guard", {})
    if yg.get("status") != "HOLDS":
        return bad(f"yield_guard_status_{yg.get('status')}")
    if "note" in yg:
        return bad("the_two_yield-guard_denominators_disagree_on_tripping:_Master_Mind_decides_(addendum_1_item_4)")
    by = rep["by_rank"]
    kept = {int(k): v["kept"] for k, v in by.items()}
    if sum(kept.values()) != rep["output"]["kept_twins"] or rep["output"]["kept_twins"] <= 0:
        return bad("kept_twins_does_not_equal_sum_of_by_rank_kept_or_is_zero")
    # independent re-reading of the yield guard from the audit file (the merge's own number is not taken on trust)
    ev1 = kp1 = 0
    kept_audit = {1: 0, 2: 0, 3: 0}
    n_audit = 0
    for r in _jsonl(OUT / "audit_all.jsonl"):
        n_audit += 1
        is_kept = bool(r["commit"]) and r["drop_reason"] is None
        if is_kept:
            if not (r["f_at_s"] and r["f_at_s"][-1] >= KEEP_F10):
                return bad("an_audited_kept_twin_has_f10_below_0.75")
            kept_audit[int(r["k"])] = kept_audit.get(int(r["k"]), 0) + 1
        if int(r["k"]) == 1 and r.get("f_at_s") is not None:
            ev1 += 1
            kp1 += int(is_kept)
    if n_audit != rep["input"]["lane_rank_rows"]:
        return bad(f"audit_rows_{n_audit}_!=_input_lane_rank_rows_{rep['input']['lane_rank_rows']}")
    if any(kept_audit.get(k, 0) != kept.get(k, 0) for k in (1, 2, 3)):
        return bad(f"audit_kept_{kept_audit}_!=_report_kept_{kept}")
    if ev1 < YIELD_GUARD_MIN_N or kp1 / ev1 < YIELD_GUARD_K1:
        return bad(f"independent_yield_guard_re-read_K1_{kp1}/{ev1}_is_below_0.20_or_n<30")
    out_lines = sum(1 for _ in open(t1, "rb"))
    if out_lines != rep["output"]["lines"] or out_lines != rep["input"]["lines"] - rep["output"]["dropped_twins"]:
        return bad("line-count_identity_failed")
    out_sha = rep["output"]["sha256"]
    if not a.quick:
        in1 = IN / "targets_rank1.jsonl"
        if sha256_file(in1) != rep["input"]["sha256"]:
            return bad("the_INPUT_bank_changed_after_the_merge_(input_sha256_differs_from_merge_report)")
        if sha256_file(t1) != out_sha:
            return bad("targets_rank1.jsonl_sha256_differs_from_merge_report.output.sha256")
        for extra in sorted(IN.iterdir()):                       # everything else the merge claims to have copied unchanged
            if extra.name != "targets_rank1.jsonl" and extra.is_file() and not extra.name.startswith("."):
                d = OUT / extra.name
                if not d.is_file() or sha256_file(d) != sha256_file(extra):
                    return bad(f"{extra.name}_is_missing_or_differs_in_the_output")
    share = {k: by[str(k)].get("keep_share_evaluated") for k in (1, 2, 3) if str(k) in by}
    emit("merge", state="ok", kept=rep["output"]["kept_twins"], dropped=rep["output"]["dropped_twins"], kept_1=kept.get(1, 0), kept_2=kept.get(2, 0), kept_3=kept.get(3, 0),
         k1_evaluated=ev1, k1_share_eval=round(kp1 / ev1, 4), k1_share_all=yg.get("k1_share_all"), shares_eval=",".join(f"K{k}={v}" for k, v in share.items()),
         lines_in=rep["input"]["lines"], lines_out=out_lines, out_sha=out_sha, verified=("report_only" if a.quick else "rehashed"))
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- c'2 / c'3
def cmd_c2(a) -> int:
    F, OUT = Path(a.f), Path(a.out)
    c2 = F / "c2"
    cp, ss, ms = c2 / "cprime2.json", c2 / "sample.json", c2 / "merged_sha.txt"
    if not (c2 / "C2_FINISHED").exists():
        emit("c2", outcome="missing", why="C2_FINISHED_absent")
        return 0
    if not (cp.is_file() and ss.is_file() and ms.is_file()):
        emit("c2", outcome="missing", why="cprime2.json/sample.json/merged_sha.txt_absent")
        return 0
    cur = sha256_file(OUT / "targets_rank1.jsonl")
    if ms.read_text().strip() != cur:
        emit("c2", outcome="stale", why="c2_was_run_on_another_merged_bank")
        return 0
    res, samp = read_json(cp), read_json(ss)
    kw = {}
    incomplete = []
    for k in ("1", "2", "3"):
        e = res["ranks"].get(k, {})
        want = samp["per_rank"].get(k, {}).get("sampled", 0)
        got = e.get("n_paired", 0)
        if got != want:
            incomplete.append(f"K{k}:{got}/{want}")
        kw[f"r{k}"] = f"n={got}/{want},ratio={e.get('path_ratio_median')},dv={e.get('end_speed_change_median_mps')},judged={e.get('judged')},pass={e.get('pass')}"
    oc = res.get("outcome", "")
    if incomplete:
        emit("c2", outcome="incomplete", why="paired_<_sampled:_" + ",".join(incomplete), **kw)
    elif oc.startswith("3C-SLOW"):
        emit("c2", outcome="slow", why=oc[:120], **kw)
    elif oc == "c'2 PASS":
        emit("c2", outcome="pass", **kw)
    else:
        emit("c2", outcome="notjudged", why=oc[:120], **kw)
    return 0


def cmd_c3(a) -> int:
    F = Path(a.f)
    ok_all = True
    for k in [int(x) for x in a.ks.split(",") if x]:
        lg = F / f"verify_k{k}.log"
        summ = None
        if lg.is_file():
            for ln in lg.read_text(errors="replace").splitlines():
                if ln.startswith("ZZD7V_SUMMARY "):
                    try:
                        summ = json.loads(ln[len("ZZD7V_SUMMARY "):])
                    except ValueError:
                        summ = None
        if summ is None:
            emit("c3", k=k, state="missing", why="no_ZZD7V_SUMMARY_line_in_verify_k%d.log" % k)
            ok_all = False
            continue
        allb = summ.get("all", {})
        a_ok, b_ok, c_ok = bool(summ.get("a_ok")), bool(summ.get("b_ok")), bool(summ.get("c_ok"))
        emit("c3", k=k, state="judged", a=a_ok, b=b_ok, c_reported=c_ok, n_checked=summ.get("n_checked"), n_requested=summ.get("n_requested"), n_error=summ.get("n_error"),
             digest_differs=allb.get("digest_differs"), spec_ok=allb.get("spec_ok"), harness_control=summ.get("harness_reproduces_logged_centerline_frac"),
             c_traj_closer=allb.get("traj_closer_to_twin_div_ge_0_5"), n_div_ge_0_5=allb.get("n_div_ge_0_5"))
        ok_all = ok_all and a_ok and b_ok
    hold = F / "twin_hold_check.out"
    hold_txt = hold.read_text(errors="replace") if hold.is_file() else ""
    m = re.search(r"^ZZTWIN_HOLD_(OK|BAD) (.*)$", hold_txt, re.M)
    emit("c3hold", state=("ok" if (m and m.group(1) == "OK") else "bad" if m else "missing"), detail=(m.group(2) if m else "no_marker"))
    emit("c3all", ok=bool(ok_all and m and m.group(1) == "OK"))
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- the shipped P6 tree
def cmd_p6tree(a) -> int:
    pkg = Path(a.pkg)
    mf = pkg / "SHIP_MANIFEST.json"
    pod = pkg / "refe" / "p6_pod.sh"
    if not (mf.is_file() and pod.is_file()):
        emit("p6tree", state="missing")
        return 0
    m = read_json(mf)
    t = pod.read_text(errors="replace")
    emit("p6tree", state="ok", commit=m.get("commit"), tree_sha256=str(m.get("tree_sha256"))[:16],
         p6_pod_md5=hashlib.md5(pod.read_bytes()).hexdigest()[:12],
         cbank_knob=int("P6_CBANK" in t and 'CBANK="${P6_CBANK:-' in t),
         cbank_commands_guard=int("P6_CBANK is set" in t),
         r43_expandable_segments=int("expandable_segments:True" in t),
         stop_after_launch_gate=int('stop_after launch_gate' in t),
         build_has_pdm_route_rank=int("pdm_route_rank" in (pkg / "refe" / "build_metric_cache.py").read_text(errors="replace")) if (pkg / "refe" / "build_metric_cache.py").is_file() else -1)
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- step (d): the kept twins' parts
def _lane_rows(bank: Path):
    with open(bank / "targets_rank1.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            if '"lane_rank"' not in ln:
                continue
            r = json.loads(ln)
            a = r.get("aug")
            if isinstance(a, dict) and a.get("kind") == "lane_rank":
                yield r


def _count_parts(mc: Path):
    plain = twin = 0
    for d in mc.iterdir():
        if d.is_dir():
            for f in os.scandir(d):
                if f.name.endswith(".pkl.xz"):
                    if "__lane_rank_" in f.name:
                        twin += 1
                    else:
                        plain += 1
    return plain, twin


def cmd_parts(a) -> int:
    sys.path.insert(0, a.p6refe)
    import online_labeller as OL                                                       # the CONSUMER's own path function (the P6 tree), never a re-implementation
    bank, src, dst = Path(a.bank), Path(a.src), Path(a.dst)
    t0 = time.time()
    seen, todo, rows_n = set(), [], 0
    pairs = {}
    miss_src = miss_logged = conflict = identical = same_inode = 0
    first_bad = []
    bytes_new = 0
    for r in _lane_rows(bank):
        rows_n += 1
        s = OL.parts_path(src, r["log_name"], r["token"], r["aug"])
        d = OL.parts_path(dst, r["log_name"], r["token"], r["aug"])
        lg = OL.parts_path(dst, r["log_name"], r["token"], None)
        if d in seen:
            continue
        seen.add(d)
        if not lg.is_file():
            miss_logged += 1
            first_bad.append(f"no_logged:{lg.name}")
        if not s.is_file():
            miss_src += 1
            first_bad.append(f"no_src:{s.name}")
            continue
        pairs[d] = s
        if d.exists():
            try:
                same = os.path.samefile(s, d)
            except OSError:
                same = False
            if same:
                same_inode += 1
            elif d.stat().st_size == s.stat().st_size and sha256_file(d) == sha256_file(s):
                identical += 1
            else:
                conflict += 1
                first_bad.append(f"conflict:{d.name}")
            continue
        todo.append((s, d))
        bytes_new += s.stat().st_size
    base = dict(rows=rows_n, distinct=len(seen), to_place=len(todo), present_same_inode=same_inode, present_identical=identical, missing_src=miss_src, missing_logged=miss_logged,
                conflict=conflict, bytes_to_place=bytes_new, first_bad=";".join(first_bad[:3]) or "-")
    if a.mode == "plan":
        emit("parts", state=("plan_ok" if not (miss_src or miss_logged or conflict) else "plan_bad"), **base)
        return 0

    # ---- apply
    if miss_src or miss_logged or conflict:
        emit("parts", state="refused", **base)                                       # fail BEFORE writing anything
        return 0
    how = a.how
    linked = copied = errors = 0
    err_first = ""
    for s, d in todo:
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            if how in ("auto", "link"):
                try:
                    os.link(s, d)
                    linked += 1
                    continue
                except OSError as e:
                    if how == "link":
                        raise
                    how = "copy"                                                       # the volume refuses hard links: copy from here on (atomic, verified)
                    err_first = err_first or f"link_unsupported_errno{e.errno}"
            tmp = d.with_name("." + d.name + ".p6next.tmp")
            with open(s, "rb") as fi, open(tmp, "wb") as fo:
                h = hashlib.sha256()
                while True:
                    b = fi.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    fo.write(b)
                fo.flush()
                os.fsync(fo.fileno())
            if tmp.stat().st_size != s.stat().st_size or sha256_file(tmp) != h.hexdigest():
                tmp.unlink()
                raise OSError("copy_verification_failed")
            os.replace(tmp, d)                                                         # atomic: a part is whole or absent (the cache's own convention)
            copied += 1
        except OSError as e:
            errors += 1
            err_first = err_first or f"{d.name}:{type(e).__name__}:{getattr(e, 'errno', '')}"
            if errors >= 5:
                break
    # ---- verify the destination from the filesystem, not from the loop's counters
    bad = 0
    for d, s in pairs.items():
        if not d.is_file() or d.stat().st_size != s.stat().st_size:
            bad += 1
    plain, twin = _count_parts(dst)
    st = "ok" if (errors == 0 and bad == 0) else "fail"
    rep = dict(base, state=st, linked=linked, copied=copied, errors=errors, err_first=err_first or "-", dst_missing_after=bad, mode=("link" if (linked and not copied) else "copy" if copied and not linked else "mixed" if copied else "none"),
               logged_parts_in_dst=plain, twin_parts_in_dst=twin, extras_in_dst=twin - len(seen), wall_s=round(time.time() - t0, 1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    emit("parts", **rep)
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- the P6 dry run
def judge_chain_text(blob: str, ranks) -> tuple:
    """PURE.  The chain.log text written AFTER the dry run started (the caller slices by the line offset taken just before the launch) -> (ok, why).
    Fail-closed: a missing PASS line, any REFUSED line, the D7 fallback, a trainer that started, or two contradicting verdicts is NOT ok.
    Matching is on whole log LINES (a timestamp prefix is allowed), never on a substring of a longer sentence that merely quotes a marker."""
    lines = [ln.rstrip("\r").strip() for ln in blob.splitlines()]

    def has(msg: str) -> bool:
        # `<ts> <msg>` with an optional ISO timestamp first; the message must START the rest of the line
        return any(re.fullmatch(r"(?:\d{4}-\d\d-\d\dT[\d:]+Z?\s+)?" + re.escape(msg) + r"(?:\s.*)?", ln) is not None for ln in lines)

    def starts(msg: str) -> bool:
        return any(re.match(r"(?:\d{4}-\d\d-\d\dT[\d:]+Z?\s+)?" + re.escape(msg), ln) is not None for ln in lines)

    if starts("REFUSED:"):
        return False, "the_chain_REFUSED:_" + re.sub(r"\s+", "_", next(ln for ln in lines if "REFUSED:" in ln))[:200]
    if any("ZZP6_REFUSED" in ln for ln in lines):
        return False, "ZZP6_REFUSED_(another_chain_holds_the_lock)"
    if starts("D7 FALLBACK"):
        return False, "the_chain_took_the_D7_FALLBACK_(the_PI_ruled_faithful,_not_the_fallback)"
    # ANY other ZZ...FAIL marker blocks the launch, EXCEPT the one that is expected in every good run: G-EGR's regression arm pushes to a namespace the token cannot write and MUST be refused
    # (`ZZEGRESS_FAIL G-EGR-REGRESSION ... 403 Forbidden` is paired with `ZZEGR_PASS main_ok=True regression_refused=True`).  A failed periodic push tick (ZZEGRESS_FAIL p2r_tick_N) therefore blocks too: fail-closed.
    if any(re.search(r"ZZ[A-Z0-9_]*FAIL", ln) and "ZZEGRESS_FAIL G-EGR-REGRESSION" not in ln for ln in lines):
        return False, "a_ZZ...FAIL_marker_line_(only_the_must-fail_G-EGR-REGRESSION_arm_is_expected):_" + re.sub(r"\s+", "_", next(ln for ln in lines if re.search(r"ZZ[A-Z0-9_]*FAIL", ln) and "ZZEGRESS_FAIL G-EGR-REGRESSION" not in ln))[:160]
    if starts("train.py preflight OK") or starts("attempt ") or has("ZZP6_TRAIN_DONE") or starts("ZZP6_FAILED"):
        return False, "the_trainer_side_of_the_chain_ran_(P6_STOP_AFTER_was_not_honoured_or_this_log_slice_belongs_to_a_real_launch)"
    pass_ct = sum(1 for ln in lines if re.fullmatch(r"(?:\d{4}-\d\d-\d\dT[\d:]+Z?\s+)?launch gate PASS \(token verified\)", ln))
    if pass_ct != 1:
        return False, f"expected_exactly_one_'launch_gate_PASS_(token_verified)'_line,_found_{pass_ct}"
    if not has("lanes ok"):
        return False, "chain.log_after_the_start_lacks_'lanes_ok'"
    if not starts("stopping after launch_gate"):
        return False, "chain.log_after_the_start_lacks_'stopping_after_launch_gate'"
    for k in ranks:
        if not starts(f"D7 rank {k}:"):
            return False, f"chain.log_lacks_'D7_rank_{k}:'"
    # order: the PASS line must come after 'lanes ok' and before the stop line
    idx = {m: next(i for i, ln in enumerate(lines) if m in ln) for m in ("lanes ok", "launch gate PASS (token verified)", "stopping after launch_gate")}
    if not (idx["lanes ok"] < idx["launch gate PASS (token verified)"] < idx["stopping after launch_gate"]):
        return False, "chain.log_markers_are_out_of_order"
    return True, "ok"


def cmd_gate(a) -> int:
    lg, p6, OUT, MC = Path(a.lg), Path(a.p6), Path(a.out), Path(a.mc)
    ranks = [int(x) for x in a.ranks.split(",") if x]
    # every mtime on this volume is the NFS SERVER's clock, so the reference is a file the orchestrator created (on the same server) just before the launch -- not the pod's `date`
    a.since = os.stat(a.since_file).st_mtime

    def fail(why: str) -> int:
        emit("gate", state="fail", why=why)
        return 0

    fails = [p for p in lg.glob("FAIL_*.json") if p.stat().st_mtime >= a.since]
    if fails:
        try:
            rs = read_json(fails[0]).get("reasons", [])
        except Exception:                                                                # noqa: BLE001
            rs = ["unreadable"]
        return fail("a_FAIL_token_was_written:_" + re.sub(r"\s+", "_", "; ".join(map(str, rs)))[:240])
    passes = sorted((p for p in lg.glob("PASS_*.json") if p.stat().st_mtime >= a.since), key=lambda p: p.stat().st_mtime)
    if len(passes) != 1:
        return fail(f"expected_exactly_one_PASS_token_newer_than_the_start,_found_{len(passes)}")
    tok = read_json(passes[0])
    if tok.get("verdict") != "PASS" or tok.get("kind") != "launch-gate-token" or str(tok.get("schema", "")).split("/")[0] != "tanitad.launch_gate" or tok.get("reasons"):
        return fail("token_verdict/kind/schema/reasons_not_a_clean_PASS")
    nonpass = [k for k, v in tok.get("checks", {}).items() if v.get("status") != "PASS"]
    if nonpass or not tok.get("checks"):
        return fail(f"checks_not_all_PASS:{nonpass}")
    b = tok["binding"]
    if b.get("profile") != "refe_p6":
        return fail(f"profile_{b.get('profile')}")
    if b.get("commit") != a.commit:
        return fail(f"token_commit_{str(b.get('commit'))[:12]}_!=_expected_{a.commit[:12]}")
    argv = b["argv"]

    def flag(f):
        return argv[argv.index(f) + 1] if f in argv and argv.index(f) + 1 < len(argv) else None

    if (flag("--targets") or "").rstrip("/") != str(OUT).rstrip("/"):
        return fail(f"token_--targets_is_{flag('--targets')}_not_the_3-C_bank_(the_gate_would_have_vouched_for_the_wrong_bank)")
    if (flag("--metric-cache") or "").rstrip("/") != str(MC).rstrip("/"):
        return fail("token_--metric-cache_path")
    for f, v in (("--yaw-wrap", "off"), ("--nav-mode", "command"), ("--labeller", "online")):
        if flag(f) != v:
            return fail(f"token_{f}_is_{flag(f)}_not_{v}")
    dt, dm = b["data"].get("--targets", {}), b["data"].get("--metric-cache", {})
    out_files = sum(len(fs) for _, _, fs in os.walk(str(OUT)))
    if dt.get("n_files") != out_files:
        return fail(f"token_--targets_n_files_{dt.get('n_files')}_!=_files_in_the_3-C_bank_{out_files}")
    plain, twin = _count_parts(MC)
    if dm.get("n_files") != plain + twin:
        return fail(f"token_--metric-cache_n_files_{dm.get('n_files')}_!=_parts_now_in_MC_{plain + twin}")
    if twin < a.kept:
        return fail(f"MC_holds_{twin}_twin_parts_<_kept_{a.kept}")
    # nothing may have been (re)built by the P6 tree's UNFIXED build_metric_cache: no part in MC is newer than the dry run's start
    newer = 0
    for d in MC.iterdir():
        if d.is_dir():
            for f in os.scandir(d):
                if f.name.endswith(".pkl.xz") and f.stat().st_mtime >= a.since:
                    newer += 1
    if newer:
        return fail(f"{newer}_parts_in_MC_were_written_during_the_dry_run_(P6's_tree_cannot_build_a_twin_on_its_route)")
    # the chain's own words, after the dry run started
    ok, why = judge_chain_text("\n".join((p6 / "chain.log").read_text(errors="replace").splitlines()[a.chain_offset:]), ranks)
    if not ok:
        return fail(why)
    for k in ranks:
        o = p6 / f"lanes_distinct_{k}.out"
        if not (o.is_file() and o.stat().st_mtime >= a.since and "ZZD7_DISTINCT_OK" in o.read_text(errors="replace")):
            return fail(f"P6's_own_D7_acceptance_for_rank_{k}_did_not_run_or_did_not_pass")
    emit("gate", state="ok", token=passes[0].name, commit=b["commit"][:12], argv_sha=str(b.get("argv_sha256"))[:16], tree_sha=str(b.get("tree_sha256"))[:16], targets=flag("--targets"),
         targets_files=dt.get("n_files"), mc_files=dm.get("n_files"), mc_twin_parts=twin, parts_newer_than_start=newer)
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- "is the pod free for P6?"
def busy_decision(gpu_state: str, gpu_n: int, locks_held, rls1_pids, p6_name: str = "refe_p6") -> dict:
    """PURE.  The pod is FREE for P6 only if: nvidia-smi answered and lists NO compute process, AND no other chain holds its chain.lock, AND no `rls1_pod.sh` process is alive.
    An nvidia-smi that cannot answer counts as BUSY.  The P6 chain's OWN lock is reported separately (someone already started P6: not a thing to wait for)."""
    others = [h for h in locks_held if h.split("(")[0] != p6_name]
    why = []
    if gpu_state != "idle":
        why.append(f"gpu_{gpu_state}(n={gpu_n})")
    if others:
        why.append("chain_lock:" + ",".join(others))
    if rls1_pids:
        why.append("rls1_pod.sh_running:pids=" + ",".join(map(str, rls1_pids[:4])))
    return {"state": "idle" if not why else "busy", "why": ";".join(why) or "-", "p6_held": int(any(h.split("(")[0] == p6_name for h in locks_held))}


def _scan_proc(needle: str):
    """PIDs whose argv contains `needle` (never this process).  The needle is NOT in this helper's own argv (it is built in the source, which arrives on stdin)."""
    me, out = os.getpid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            argv = (d / "cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        if needle in argv:
            out.append(int(d.name))
    return out


def cmd_busy(a) -> int:
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_gpu(a)
        if not a.no_lock_probe:                                   # --no-lock-probe: the flock probe takes a lock for a microsecond; --probe on the real pod must write nothing
            cmd_locks(argparse.Namespace(data=a.data, exclude=None))
    g = {}
    for ln in buf.getvalue().splitlines():
        if ln.startswith("ZZV "):
            kind = ln.split()[1]
            g[kind] = dict(t.split("=", 1) for t in ln.split()[2:] if "=" in t)
    held = [h for h in g.get("locks", {}).get("held", "-").split(",") if h and h != "-"]
    pids = _scan_proc("rls1_" + "pod.sh")
    d = busy_decision(g.get("gpu", {}).get("state", "unknown"), int(g.get("gpu", {}).get("n", -1)), held, pids)
    emit("busy", state=d["state"], why=d["why"], p6_held=d["p6_held"], gpu=g.get("gpu", {}).get("state"), gpu_procs=g.get("gpu", {}).get("procs"))
    return 0


# ----------------------------------------------------------------------------------------------------------------------------------- the real P6 launch: did training start?
def _first_step_line(lines):
    """PURE.  The first metrics.jsonl line that is a TRAINING STEP: a JSON object with an integer `step`, no `event` key, and a finite numeric loss-like field (traj_L1 / *loss* / score)."""
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if not isinstance(r, dict) or "event" in r or not isinstance(r.get("step"), int) or isinstance(r.get("step"), bool):
            continue
        for k, v in r.items():
            if re.search(r"loss|L1|score", k, re.I) and isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf"):
                return {"step": r["step"], "key": k, "value": v}
    return None


def judge_launch(chain_after_launch: str, new_metrics_lines, chain_alive: bool) -> dict:
    """PURE.  state: refused | failed | trainer_exited | training | ambiguous | preflight_ok | starting | gone.
    `chain_after_launch` = chain.log text written after the real launch; `new_metrics_lines` = metrics.jsonl lines added since the launch; `chain_alive` = p6 chain.lock held."""
    lines = [ln.rstrip("\r").strip() for ln in chain_after_launch.splitlines()]

    def first(pat):
        for ln in lines:
            if re.match(r"(?:\d{4}-\d\d-\d\dT[\d:]+Z?\s+)?" + pat, ln):
                return ln
        return None

    ref = first(r"REFUSED:") or next((ln for ln in lines if "ZZP6_REFUSED" in ln), None)
    if ref:
        return {"state": "refused", "why": re.sub(r"\s+", "_", ref)[:240]}
    if first(r"ZZP6_FAILED") or first(r"ZZP6_FINAL_FAILED"):
        return {"state": "failed", "why": "chain_reports_ZZP6_FAILED"}
    pre = first(r"train\.py preflight OK")
    step = _first_step_line(new_metrics_lines)
    exited = first(r"attempt \d+ exited rc")
    if step and not pre:
        return {"state": "ambiguous", "why": "a_training_step_line_appeared_but_chain.log_has_no_'train.py_preflight_OK'_after_the_launch", **step}
    if step and pre:
        return {"state": "training", "why": "ok", **step}
    if exited:
        return {"state": "trainer_exited", "why": re.sub(r"\s+", "_", exited)[:160]}
    if pre:
        return {"state": "preflight_ok", "why": "waiting_for_the_first_step"}
    if not chain_alive:
        return {"state": "gone", "why": "the_chain_is_not_running_and_logged_neither_a_preflight_nor_a_refusal"}
    return {"state": "starting", "why": "-"}


def cmd_launchwatch(a) -> int:
    p6, run = Path(a.p6), Path(a.run)
    cl = (p6 / "chain.log").read_text(errors="replace").splitlines()[a.chain_offset:]
    mj = run / "metrics.jsonl"
    new = []
    if mj.is_file():
        new = mj.read_text(errors="replace").splitlines()[a.metrics_before:]
    # liveness by a /proc scan, NOT by a flock probe: a probe that takes the chain's lock for a microsecond while the chain is starting would make `flock -n 200` in p6_pod.sh REFUSE
    alive = bool(_scan_proc("p6_" + "pod.sh"))
    d = judge_launch("\n".join(cl), new, alive)
    emit("launch", chain_alive=int(alive), n_new_metrics=len(new), **d)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("gpu")
    p = sp.add_parser("locks"); p.add_argument("--data", required=True); p.add_argument("--exclude", action="append")
    p = sp.add_parser("merge"); p.add_argument("--f", required=True); p.add_argument("--out", required=True); p.add_argument("--inp", required=True)
    p.add_argument("--prereg-sha", required=True); p.add_argument("--quick", action="store_true")
    p = sp.add_parser("c2"); p.add_argument("--f", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("c3"); p.add_argument("--f", required=True); p.add_argument("--ks", required=True)
    p = sp.add_parser("p6tree"); p.add_argument("--pkg", required=True)
    p = sp.add_parser("parts"); p.add_argument("--bank", required=True); p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--p6refe", required=True); p.add_argument("--mode", choices=("plan", "apply"), required=True); p.add_argument("--how", choices=("auto", "link", "copy"), default="auto")
    p.add_argument("--report", default=None)
    p = sp.add_parser("busy"); p.add_argument("--data", required=True); p.add_argument("--no-lock-probe", action="store_true")
    p = sp.add_parser("launchwatch"); p.add_argument("--p6", required=True); p.add_argument("--run", required=True); p.add_argument("--chain-offset", type=int, required=True)
    p.add_argument("--metrics-before", type=int, required=True)
    p = sp.add_parser("gate"); p.add_argument("--lg", required=True); p.add_argument("--out", required=True); p.add_argument("--commit", required=True); p.add_argument("--mc", required=True)
    p.add_argument("--kept", type=int, required=True); p.add_argument("--since-file", required=True); p.add_argument("--p6", required=True)
    p.add_argument("--ranks", required=True); p.add_argument("--chain-offset", type=int, required=True)
    a = ap.parse_args(argv)
    return {"gpu": cmd_gpu, "locks": cmd_locks, "merge": cmd_merge, "c2": cmd_c2, "c3": cmd_c3, "p6tree": cmd_p6tree, "parts": cmd_parts, "gate": cmd_gate, "busy": cmd_busy, "launchwatch": cmd_launchwatch}[a.cmd](a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:                                                               # noqa: BLE001 -- a crash must read as a FAILED verdict, never as silence
        import traceback
        traceback.print_exc()
        emit("helper_crash", cmd=(sys.argv[1] if len(sys.argv) > 1 else "?"), exc=f"{type(e).__name__}:{str(e)[:120]}")
        sys.exit(0)
