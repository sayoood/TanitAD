"""Anchored patch: every edit asserts its anchor occurs exactly once.  python patch_sim.py <in> <out>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


Q3 = '"' * 3

rep("""    python ep_admission_sim.py repro      --workdir D:/epadm1/run                                                                                  # PC5: 20 replicates in a SEPARATE process, bit-identical to run's first 20
""", """    python ep_admission_sim.py repro      --workdir D:/epadm1/run                                                                                  # PC5: 20 replicates in a SEPARATE process, bit-identical to run's first 20
`run` / `repro` take `--rep-offset INT` (default 0, EP-ADMISSION-R2-CONFIRM): replicate `rep` is drawn with `rep + offset` in BOTH generators (`default_rng([OUTER_SALT, rep + offset])`, `default_rng([WILD_SALT, rep + offset])`).
Offset 0 is the banked run bit for bit; a non-zero offset is a FRESH set of replicates of the same conditions, estimator and registered bootstrap draws (`BOOT_SEED` is NOT offset).
""")

rep('''def outer_idx(rep: int) -> np.ndarray:
    return np.random.default_rng([OUTER_SALT, rep]).integers(0, N_LOGS, N_LOGS)


def wild_signs(rep: int) -> np.ndarray:
    return np.random.default_rng([WILD_SALT, rep]).choice(np.array([-1.0, 1.0]), N_LOGS)


def replicate(cond: dict, rep: int, draws: np.ndarray) -> dict:
    ''' + Q3 + '''-> {dm [2, 3] (point, lo, hi) per seed, a2 [2, 3, 3] (head, point/lo/hi) per seed, mp [2], me [2]}.  Generator A resamples the logs; generator B flips the sign of each log's calibrated residual.''' + Q3 + '''
    _, _, F = libs()
    gen, heads = cond["gen"], cond.get("heads", HEADS)
    idx = outer_idx(rep) if gen == "A" else None
    w = wild_signs(rep) if gen == "B" else None
''', '''def outer_idx(rep: int, offset: int = 0) -> np.ndarray:
    return np.random.default_rng([OUTER_SALT, rep + offset]).integers(0, N_LOGS, N_LOGS)


def wild_signs(rep: int, offset: int = 0) -> np.ndarray:
    return np.random.default_rng([WILD_SALT, rep + offset]).choice(np.array([-1.0, 1.0]), N_LOGS)


def replicate(cond: dict, rep: int, draws: np.ndarray, offset: int = 0) -> dict:
    ''' + Q3 + '''-> {dm [2, 3] (point, lo, hi) per seed, a2 [2, 3, 3] (head, point/lo/hi) per seed, mp [2], me [2]}.  Generator A resamples the logs; generator B flips the sign of each log's calibrated residual.
    Replicate `rep` uses the draws of index `rep + offset` (offset 0 = the banked run; the registered bootstrap `draws` are shared by every replicate and are not offset).''' + Q3 + '''
    _, _, F = libs()
    gen, heads = cond["gen"], cond.get("heads", HEADS)
    idx = outer_idx(rep, offset) if gen == "A" else None
    w = wild_signs(rep, offset) if gen == "B" else None
''')

rep('''_CONDS: dict = {}
_DRAWS = None


def _init(conds):
    global _CONDS, _DRAWS
    _CONDS = conds
    _DRAWS = libs()[2].boot_draws(N_LOGS, BOOT_B, BOOT_SEED)


def _chunk(task):
    name, a, b = task
    recs = [replicate(_CONDS[name], r, _DRAWS) for r in range(a, b)]
''', '''_CONDS: dict = {}
_DRAWS = None
_REP_OFFSET = 0


def _init(conds, rep_offset=0):
    global _CONDS, _DRAWS, _REP_OFFSET
    _CONDS = conds
    _REP_OFFSET = int(rep_offset)
    _DRAWS = libs()[2].boot_draws(N_LOGS, BOOT_B, BOOT_SEED)


def _chunk(task):
    name, a, b = task
    recs = [replicate(_CONDS[name], r, _DRAWS, _REP_OFFSET) for r in range(a, b)]
''')

rep('''def run_condition_set(conds: dict, n_reps: int, workers: int, chunk: int = 25, log=print) -> dict:
    ''' + Q3 + '''All replicates of all conditions, in `workers` spawned processes (each loads the module and the registered draws once).''' + Q3 + '''
''', '''def run_condition_set(conds: dict, n_reps: int, workers: int, chunk: int = 25, log=print, rep_offset: int = 0) -> dict:
    ''' + Q3 + '''All replicates of all conditions, in `workers` spawned processes (each loads the module and the registered draws once).  Replicate `r` is the draw of index `r + rep_offset`.''' + Q3 + '''
''')
rep("    with ctx.Pool(workers, initializer=_init, initargs=(conds,)) as pool:", "    with ctx.Pool(workers, initializer=_init, initargs=(conds, int(rep_offset))) as pool:")

rep('''    print(f"run: {len(conds)} conditions x {a.n_reps} replicates (IDENTITY x 50), {a.workers} workers, B = {BOOT_B}, seed {BOOT_SEED}", flush=True)''',
    '''    print(f"run: {len(conds)} conditions x {a.n_reps} replicates (IDENTITY x 50), {a.workers} workers, B = {BOOT_B}, seed {BOOT_SEED}, rep_offset {a.rep_offset}", flush=True)''')
rep('''        recs.update(run_condition_set(part, n_reps[group[0]] if len(group) == 1 else a.n_reps, a.workers, log=lambda m: print(m, flush=True)))''',
    '''        recs.update(run_condition_set(part, n_reps[group[0]] if len(group) == 1 else a.n_reps, a.workers, log=lambda m: print(m, flush=True), rep_offset=a.rep_offset))''')
rep('''"design_sha256": dsha, "n_reps": a.n_reps, "boot":''', '''"design_sha256": dsha, "n_reps": a.n_reps, "rep_offset": int(a.rep_offset), "boot":''')

rep('''def cmd_repro(a) -> int:
    design_guard()
    vec = load_vec(a.workdir)
    conds = make_conditions(vec)
    _, _, F = libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    z = np.load(os.path.join(a.workdir, "replicates.npz"))
    ok = True
    for name in ("G78|A|CAL", "G78|B|CAL", "G78|A|PLANT_EP_0.135"):
        for r in range(20):
            rec = replicate(conds[name], r, draws)
''', '''def cmd_repro(a) -> int:
    ''' + Q3 + '''PC5: the first 20 replicates of 3 conditions, recomputed in this separate process with `--rep-offset`, equal the stored ones bit for bit.
    `--against <replicates.npz>` compares with another run's file instead of the workdir's (with `--rep-offset 0` and the banked EP-ADMISSION-R1 file: the offset-0 identity proof).''' + Q3 + '''
    design_guard()
    vec = load_vec(a.workdir)
    conds = make_conditions(vec)
    _, _, F = libs()
    draws = F.boot_draws(N_LOGS, BOOT_B, BOOT_SEED)
    z = np.load(a.against or os.path.join(a.workdir, "replicates.npz"))
    ok = True
    for name in ("G78|A|CAL", "G78|B|CAL", "G78|A|PLANT_EP_0.135"):
        for r in range(20):
            rec = replicate(conds[name], r, draws, a.rep_offset)
''')
rep('''    out = {"bit_identical_first_20_replicates_of_3_conditions": bool(ok), "pid": os.getpid()}
    json.dump(out, open(os.path.join(a.workdir, "repro.json"), "w"), indent=1)''',
    '''    out = {"bit_identical_first_20_replicates_of_3_conditions": bool(ok), "pid": os.getpid(), "rep_offset": int(a.rep_offset), "against": a.against or "workdir replicates.npz"}
    json.dump(out, open(os.path.join(a.workdir, a.out or "repro.json"), "w"), indent=1)''')

rep('''def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\\n")[0])''', '''def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\\n")[0])''')
rep('''    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("repro")
    q.add_argument("--workdir", required=True)
    q.set_defaults(fn=cmd_repro)
    a = ap.parse_args(argv)''', '''    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--rep-offset", type=int, default=0, help="replicate r uses the draws of index r + offset in both generators; 0 = the banked EP-ADMISSION-R1 draws, 1000000 = the EP-ADMISSION-R2-CONFIRM fresh set")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("repro")
    q.add_argument("--workdir", required=True)
    q.add_argument("--rep-offset", type=int, default=0, help="the offset the compared replicates were run with")
    q.add_argument("--against", default=None, help="replicates.npz to compare with (default: the workdir's own)")
    q.add_argument("--out", default=None, help="file name under --workdir for the result JSON (default repro.json)")
    q.set_defaults(fn=cmd_repro)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)''')
open(dst, "w", encoding="utf-8", newline="").write(s)
print("patched ->", dst)
