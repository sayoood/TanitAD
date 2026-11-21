"""kdraw_roll_r8.py -- roll ADDITIONAL inference seeds of one refcv8 checkpoint for the K-draw ensemble read (PREREG_KDRAW_E1.md).
NEW file; nothing existing is edited.  GPU for `roll` / `control`; CPU for `freegb` / `status` / `compare`.

    python kdraw_roll_r8.py roll    --seed 2 [--tree step30000ema_unc] [--full]       # one seed, os-only (SPEC A3, as the battery rolls every seed after the first)
    python kdraw_roll_r8.py control [--tree ...] [--min-windows 50]                   # prefix re-rolls vs the BANKED dumps (determinism per seed)
    python kdraw_roll_r8.py status  [--tree ...]                                      # which seeds are rolled AND verified
    python kdraw_roll_r8.py freegb                                                    # free commit in GB (Win32_OperatingSystem.FreeVirtualMemory)

HOW A SEED IS ROLLED (identical to the battery, by construction): the child is `roll_seed_r8.py` (the battery's own child, run UNEDITED) with the
battery's own argv --ckpt --config --seed --dump-dir --windows-json --device --out-json [--os-only], in the battery's environment
(R8_NO_POSE_SYNC=1 = the standalone clock the banked `step30000ema_unc` pass used, `battery_summary.json['r8']['pose_sync_policy']`), and the window list
is the BANKED `windows_s0.json` byte for byte -- so a new dump has the same 4,754 windows in the same order as `dump_s0` / `dump_s1`.
Nothing is trusted from the child's exit code: a roll counts only when `KDRAW_VERIFY.json` (written HERE, from the dump's CONTENT) says ok.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = "E:/refcv8_eval_kit"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
MODEL_FREE = ("ws", "eid", "clip_index", "g", "ha0_ext", "v0")
#: the environment queue_ema_r8.sh gives the battery (standalone pose clock => R8_NO_POSE_SYNC=1; see the module docstring)
CHILD_ENV = {"PYTHONPATH": "C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "6", "R8_NO_POSE_SYNC": "1"}


# ------------------------------------------------------------------------------------------------------------------ paths
def step_of(tree: str) -> int:
    import re
    m = re.match(r"step(\d+)ema", tree)
    if not m:
        raise SystemExit(f"[kdraw_roll] tree {tree!r}: expected 'step<N>ema_unc'")
    return int(m.group(1))


def paths(tree: str, banked_dir: str | None = None, out_root: str | None = None) -> dict:
    step = step_of(tree)
    b = banked_dir or f"{ARCHIVE}/{tree}"
    root = out_root or f"{KIT}/kdraw/{tree}"
    return {"step": step, "ckpt": f"{KIT}/ckpt/ckpt_ema_{step}.pt", "config": f"{KIT}/ckpt/config.json", "banked": b, "root": root,
            "windows": f"{b}/windows_s0.json", "dump0": f"{b}/dump_s0", "tree": tree}


def dump_dir_of(P: dict, seed: int) -> str:
    """Seeds 0 and 1 live in the banked battery tree; every other seed in the kdraw root."""
    return f"{P['banked']}/dump_s{seed}" if seed in (0, 1) else f"{P['root']}/dump_s{seed}"


def md5_of(p: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_of(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ pure helpers (unit-tested)
def select_prefix(manifest: dict, windows: dict, min_windows: int) -> tuple[dict, list]:
    """The WHOLE-clip prefix (file_index order) holding at least `min_windows` windows: {clip_id: [ws,...]} and the clip ids in order."""
    eps = sorted(manifest["episodes"], key=lambda e: int(e["file_index"]))
    sub, order, n = {}, [], 0
    for e in eps:
        cid = str(e["clip_id"])
        if cid not in windows:
            raise ValueError(f"clip {cid} of the manifest is not in the windows file")
        sub[cid] = list(windows[cid])
        order.append(cid)
        n += len(sub[cid])
        if n >= min_windows:
            return sub, order
    raise ValueError(f"only {n} windows in the whole grid (< {min_windows})")


def load_rows(dump_dir: str) -> dict:
    """{(clip_index, ws): {arm: row array}} for every ep*.npz of a dump; arms = every per-window array (first axis = windows)."""
    import glob
    rows = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "ep*.npz"))):
        z = np.load(f)
        ws, ci = z["ws"], int(z["clip_index"][0])
        n = len(ws)
        for i in range(n):
            rec = {}
            for k in z.files:
                if k in ("ws", "eid", "clip_index"):
                    continue
                a = z[k]
                if a.ndim >= 1 and a.shape[0] == n:
                    rec[k] = a[i]
            rows[(ci, int(ws[i]))] = rec
    return rows


def compare_rows(new: dict, ref: dict) -> dict:
    """Per arm: how many of the windows present in BOTH are bit-identical, the max / mean absolute difference.  `new` must be a subset of `ref`."""
    keys = sorted(set(new) & set(ref))
    out = {"n_new": len(new), "n_ref": len(ref), "n_common": len(keys), "n_new_only": len(set(new) - set(ref)), "arms": {}}
    if not keys:
        out["status"] = "NO COMMON WINDOWS"
        return out
    arms = sorted(set(new[keys[0]]) & set(ref[keys[0]]))
    for arm in arms:
        d = np.array([float(np.abs(new[k][arm].astype(np.float64) - ref[k][arm].astype(np.float64)).max()) for k in keys])
        ident = np.array([bool(np.array_equal(new[k][arm], ref[k][arm])) for k in keys])
        out["arms"][arm] = {"n": len(keys), "n_bit_identical": int(ident.sum()), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                            "first_differing_window": (None if ident.all() else list(map(int, keys[int(np.argmin(ident))])))}
    out["status"] = "OK"
    out["os_bit_identical_all"] = bool(out["arms"].get("os", {}).get("n_bit_identical") == len(keys))
    return out


def verify_dump(dump_dir: str, ref_dump0: str, n_expected: int, n_eps_expected: int) -> dict:
    """Judge a rolled seed by the CONTENT of its dump (never by the child's exit code)."""
    import glob
    v = {"dump_dir": dump_dir, "ok": False, "checks": {}}
    mp = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(mp):
        v["checks"]["manifest"] = "ABSENT"
        return v
    man = json.load(open(mp, encoding="utf-8"))
    grid = man.get("grid", {})
    v["checks"]["grid"] = {"n_windows": grid.get("n_windows"), "n_episodes": grid.get("n_episodes"), "skipped": grid.get("n_windows_skipped"),
                           "inference_seed": (man.get("inference_seed") or {}).get("seed")}
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    v["checks"]["n_ep_files"] = len(files)
    ref_files = sorted(glob.glob(os.path.join(ref_dump0, "ep*.npz")))
    ok = (grid.get("n_windows") == n_expected and grid.get("n_episodes") == n_eps_expected and grid.get("n_windows_skipped") == 0
          and len(files) == len(ref_files) == n_eps_expected)
    bad_mf, nonfinite, shas = [], 0, {}
    if ok:
        for f, r in zip(files, ref_files):
            z, zr = np.load(f), np.load(r)
            for k in MODEL_FREE:
                if k in zr.files and not (k in z.files and np.array_equal(z[k], zr[k])):
                    bad_mf.append(f"{os.path.basename(f)}:{k}")
            if "os" not in z.files or not np.isfinite(z["os"]).all():
                nonfinite += 1
            shas[os.path.basename(f)] = sha256_of(f)
    v["checks"]["model_free_arms_bit_identical_to_banked_seed0"] = {"bad": bad_mf[:10], "n_bad": len(bad_mf)}
    v["checks"]["episodes_with_nonfinite_or_missing_os"] = nonfinite
    ex, exr = os.path.join(dump_dir, "refcv6_extras.npz"), os.path.join(ref_dump0, "refcv6_extras.npz")
    if ok and os.path.exists(ex):
        z, zr = np.load(ex), np.load(exr)
        bad = [k for k in ("gt_max_speed_6s", "v_max_ms", "v_max_valid", "ws", "clip_index") if k not in z.files or not np.array_equal(z[k], zr[k])]
        v["checks"]["extras_mask_inputs_equal_banked_seed0"] = {"bad": bad}
        ok = ok and not bad
    else:
        v["checks"]["extras_mask_inputs_equal_banked_seed0"] = "ABSENT or grid mismatch"
        ok = False
    v["ok"] = bool(ok and not bad_mf and nonfinite == 0)
    v["ep_sha256"] = shas
    return v


# ------------------------------------------------------------------------------------------------------------------ the child
def run_child(P: dict, seed: int, dump_dir: str, windows_json: str, out_json: str, log: str, full: bool, device: str = "cuda") -> dict:
    cmd = [sys.executable, str(HERE / "roll_seed_r8.py"), "--ckpt", P["ckpt"], "--config", P["config"], "--seed", str(seed), "--dump-dir", dump_dir,
           "--windows-json", windows_json, "--device", device, "--out-json", out_json] + ([] if full else ["--os-only"])
    env = dict(os.environ)
    env.update(CHILD_ENV)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write(f"[kdraw_roll] {time.strftime('%FT%T%z')} cmd: {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    return {"cmd": cmd, "exit_code_UNTRUSTED": rc, "wall_s": round(time.time() - t0, 1), "log": log,
            "roll_json_exists": os.path.exists(out_json)}


def check_ckpt(P: dict) -> dict:
    banked = json.load(open(f"{P['banked']}/battery_summary.json", encoding="utf-8"))
    want = banked.get("ckpt_md5")
    got = md5_of(P["ckpt"])
    return {"ckpt": P["ckpt"], "md5": got, "banked_md5": want, "equal": bool(want and want == got)}


# ------------------------------------------------------------------------------------------------------------------ commands
def cmd_roll(a) -> int:
    P = paths(a.tree, a.banked_dir, a.out_root)
    seed = int(a.seed)
    if seed in (0, 1) and not a.force:
        print(f"[kdraw_roll] seed {seed} is a BANKED dump; refusing to re-roll (use `control` for the determinism check)")
        return 2
    dd = a.dump_dir or dump_dir_of(P, seed)
    vj = os.path.join(dd, "KDRAW_VERIFY.json")
    if os.path.exists(vj) and not a.force:
        j = json.load(open(vj, encoding="utf-8"))
        if j.get("ok"):
            print(f"[kdraw_roll] seed {seed}: already rolled and verified ({vj}); skipping")
            return 0
    ck = check_ckpt(P)
    if not ck["equal"]:
        print(f"[kdraw_roll] REFUSING: checkpoint md5 {ck['md5']} != banked {ck['banked_md5']}")
        return 3
    Path(P["root"]).mkdir(parents=True, exist_ok=True)
    rj, lg = f"{P['root']}/roll_s{seed}.json", f"{P['root']}/roll_s{seed}.log"
    if os.path.exists(rj):
        os.replace(rj, rj + f".prev{int(time.time())}")
    ch = run_child(P, seed, dd, P["windows"], rj, lg, bool(a.full))
    n_exp = sum(len(v) for v in json.load(open(P["windows"], encoding="utf-8")).values())
    n_ep = len(json.load(open(P["windows"], encoding="utf-8")))
    v = verify_dump(dd, P["dump0"], n_exp, n_ep)
    v.update({"seed": seed, "mode": "full" if a.full else "os-only", "child": ch, "ckpt_check": ck, "finished": time.strftime("%FT%T%z"),
              "windows_file": P["windows"], "r8_env": CHILD_ENV})
    if os.path.exists(rj):
        r = json.load(open(rj, encoding="utf-8"))
        v["roll_record"] = {k: r.get(k) for k in ("seed", "arm_set", "wall_s", "n_windows", "n_episodes", "skipped", "cuda_max_memory_allocated_gib", "device")}
    Path(dd).mkdir(parents=True, exist_ok=True)
    json.dump(v, open(vj, "w", encoding="utf-8"), indent=1, default=str)
    print(f"ZZKDROLL seed={seed} ok={v['ok']} wall_s={ch['wall_s']} ZZ", flush=True)
    return 0 if v["ok"] else 4


def cmd_control(a) -> int:
    """Prefix re-rolls vs the BANKED dumps.  C1 seed 1 os-only (like-for-like for seeds 2..7); C3 seed 0 FULL (the banked seed-0 mode);
    C4 seed 0 os-only (does the arm set change the seed's `os` stream?).  If C1 is not bit-identical a SECOND identical C1 run is added (run-to-run)."""
    P = paths(a.tree, a.banked_dir, a.out_root)
    ck = check_ckpt(P)
    if not ck["equal"]:
        print(f"[kdraw_roll] REFUSING: checkpoint md5 {ck['md5']} != banked {ck['banked_md5']}")
        return 3
    windows = json.load(open(P["windows"], encoding="utf-8"))
    man0 = json.load(open(f"{P['dump0']}/manifest.json", encoding="utf-8"))
    sub, order = select_prefix(man0, windows, int(a.min_windows))
    cdir = Path(P["root"]) / "control"
    cdir.mkdir(parents=True, exist_ok=True)
    wj = str(cdir / "windows_ctrl.json")
    json.dump(sub, open(wj, "w", encoding="utf-8"))
    out = {"tool": "kdraw_roll_r8.py control", "tree": a.tree, "when": time.strftime("%FT%T%z"), "ckpt_check": ck,
           "prefix": {"n_clips": len(order), "n_windows": sum(len(v) for v in sub.values()), "clip_ids": order, "windows_json": wj},
           "meaning": ("bit-identity of a PREFIX re-roll (whole clips, the banked window order) against the banked dump rows, matched on (clip_index, ws). "
                       "Prefix of whole clips: the batches inside every included clip are those of the full roll, so identity is expected iff the "
                       "roller is deterministic per seed and the RNG stream is consumed identically."), "cases": {}}
    cases = [("C1_seed1_osonly_vs_banked_s1", 1, False, f"{P['banked']}/dump_s1"),
             ("C3_seed0_FULL_vs_banked_s0", 0, True, P["dump0"]),
             ("C4_seed0_osonly_vs_banked_s0", 0, False, P["dump0"])]
    ref_cache = {}
    for name, seed, full, refd in cases:
        dd = str(cdir / name / "dump")
        ch = run_child(P, seed, dd, wj, str(cdir / name / "roll.json"), str(cdir / name / "roll.log"), full)
        rec = {"seed": seed, "mode": "full" if full else "os-only", "child": ch, "ref": refd}
        if os.path.exists(os.path.join(dd, "manifest.json")):
            ref_cache.setdefault(refd, load_rows(refd))
            rec["compare"] = compare_rows(load_rows(dd), ref_cache[refd])
        else:
            rec["compare"] = {"status": "NO DUMP (child failed; see roll.log)"}
        out["cases"][name] = rec
        print(f"ZZKDCTRL {name} identical={rec['compare'].get('os_bit_identical_all')} ZZ", flush=True)
        if name.startswith("C1") and rec["compare"].get("os_bit_identical_all") is False:
            dd2 = str(cdir / (name + "_again") / "dump")
            ch2 = run_child(P, seed, dd2, wj, str(cdir / (name + "_again") / "roll.json"), str(cdir / (name + "_again") / "roll.log"), full)
            r2 = {"seed": seed, "mode": "os-only", "child": ch2, "ref": "C1 first run (run-to-run)"}
            if os.path.exists(os.path.join(dd2, "manifest.json")):
                r2["compare"] = compare_rows(load_rows(dd2), load_rows(dd))
            out["cases"][name + "_again"] = r2
    c1 = out["cases"]["C1_seed1_osonly_vs_banked_s1"]["compare"]
    out["verdict"] = {"C1_bit_identical": c1.get("os_bit_identical_all"),
                      "deterministic_per_seed": ("YES (prefix of the banked seed-1 roll reproduced bit-for-bit)" if c1.get("os_bit_identical_all") else
                                                 "NOT BIT-IDENTICAL -- see cases[*].compare.arms.os (max_abs_diff / n_bit_identical); read before trusting per-seed replay")}
    json.dump(out, open(Path(P["root"]) / "KDRAW_CONTROL.json", "w", encoding="utf-8"), indent=1, default=str)
    print("WROTE", Path(P["root"]) / "KDRAW_CONTROL.json", flush=True)
    return 0


def cmd_status(a) -> int:
    P = paths(a.tree, a.banked_dir, a.out_root)
    n_exp = sum(len(v) for v in json.load(open(P["windows"], encoding="utf-8")).values())
    n_ep = len(json.load(open(P["windows"], encoding="utf-8")))
    st = {}
    for s in range(8):
        dd = dump_dir_of(P, s)
        vj = os.path.join(dd, "KDRAW_VERIFY.json")
        if s in (0, 1):
            v = verify_dump(dd, P["dump0"], n_exp, n_ep)
            st[s] = {"banked": True, "ok": v["ok"], "dump": dd}
        elif os.path.exists(vj):
            st[s] = {"banked": False, "ok": bool(json.load(open(vj, encoding="utf-8")).get("ok")), "dump": dd}
        else:
            st[s] = {"banked": False, "ok": False, "dump": dd, "why": "no KDRAW_VERIFY.json"}
    print(json.dumps({"tree": a.tree, "seeds": st, "n_ok": sum(1 for v in st.values() if v["ok"])}, indent=1))
    return 0


def free_commit_gb() -> float:
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                       capture_output=True, text=True, timeout=120).stdout.strip()
    return round(int(r) / 1024 / 1024, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("roll", "control", "status", "freegb"))
    ap.add_argument("--tree", default="step30000ema_unc")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="roll every SPEC 3.2 arm (default: os + model-free only, as the battery does for seeds after the first)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--banked-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--min-windows", type=int, default=50)
    a = ap.parse_args()
    if a.cmd == "freegb":
        print(free_commit_gb())
        return 0
    if a.cmd == "roll":
        if a.seed is None:
            raise SystemExit("--seed required")
        return cmd_roll(a)
    return {"control": cmd_control, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
