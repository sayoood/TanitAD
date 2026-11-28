#!/usr/bin/env python3
"""P7 SESSION RESERVE: a deterministic, seeded draw of ~16 of navtrain's 162 drive sessions (CPU only, no GPU, no network).

D-SESSION-RESERVE (PI, DECIDED 2026-10-22, option (a)): the next trunk (P7) trains WITHOUT ~10 % of navtrain's scenes, held back as a
session-unseen set for (i) fitting selection / calibration / pessimism, (ii) measuring generalisation. This file is the rule; the draw it
records is `raw/reserve_sessions.json`, committed BEFORE any P7 data exists. Nothing here may be re-run with another seed or another
constant after P7's first training input is built: a different draw is a different pre-registration.

A SESSION is the log name minus its last two `_` parts (`2021.06.14.18.42.45_veh-12_01345_01523` -> `2021.06.14.18.42.45_veh-12`).
A TOKEN COUNT is the number of navtrain scenes attributed to the session's logs in the token table (= the P6 3-C bank's rank-0 scenes).

RULE (all constants are module constants, pinned by the tests):
  1. Eligible sessions: not navtest, not holding an SFT-held-out log (those 21 sessions already informed S1 / Stage 0 / C3 readings, so using
     them again for P7 selection would be design-choice reuse), MIN_TOK <= tokens <= MAX_TOK (neither trivial nor dominant), and belonging to
     a vehicle with >= MIN_VEH_SESS sessions in navtrain (a vehicle must not vanish from training: that would confound "unseen session"
     with "unseen vehicle").
  2. Stratify by VEHICLE: the N_SESSIONS seats are shared among vehicles by systematic probability-proportional-to-size sampling over the
     vehicles' eligible token mass (vehicles in name order, seeded random start): every vehicle gets floor or ceil of its quota.
     Each vehicle drives in exactly one city, so this also stratifies by city.
  3. Stratify by TOKEN COUNT: inside a vehicle, its eligible sessions are sorted by (tokens, name) and k are taken by systematic sampling
     (equal rank spacing, seeded random start), so a vehicle's picks span its small and large sessions.
  4. ACCEPT the first attempt (attempt 0, 1, 2, ...; each attempt's randomness is sha256(label|seed|attempt|purpose)) whose reserve meets
     every acceptance bound below. The rejected attempts and their reasons are recorded. No randomness comes from any library.

CLOSURE (what a reserved UNIT is). `--closure session` (default; what D-SESSION-RESERVE decided) reserves whole sessions. MEASURED 2026-10-22 (names
only): with session closure every reserved session keeps a same-vehicle, same-DATE sibling session in training, whereas navtest shares NO
vehicle-day with navtrain (0 of 20) and only 1 of its 9 dates. `--closure vehicle-day` reserves every session of a (date, vehicle) pair together,
which removes that gap; it is an ALTERNATIVE for the MM / PI to choose, drawn by the same rule with the MODES constants below.

Usage:  python draw_reserve.py --src ../raw/src --out ../raw/reserve_sessions.json          # draw and write
        python draw_reserve.py --src ../raw/src --out ../raw/reserve_sessions.json --check  # re-draw, require byte-identity with the file
        python draw_reserve.py --src ../raw/src --closure vehicle-day --out ../raw/reserve_vehicleday.json
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import sys

LABEL = "p7-session-reserve-v1"
SEED = 20261022
N_SESSIONS = 16                      # D-SESSION-RESERVE: "~16 of navtrain's 162 sessions (~10 % of tokens)"
FRAC_BAND = (0.09, 0.11)             # reserve tokens / navtrain scenes (attributed tokens)
MIN_TOK, MAX_TOK = 150, 1500         # a reserved session is neither trivial nor > ~14.6 % of the reserve
MIN_VEH_SESS = 3                     # a vehicle needs >= 3 navtrain sessions to contribute one
MAX_VEH_SHARE = 0.35                 # at most 35 % of any vehicle's navtrain tokens may be reserved
MIN_PER_TERCILE = 3                  # >= 3 reserved sessions in the small tercile and >= 3 in the large (terciles of the eligible pool)
MIN_VEHICLES, MIN_CITIES = 6, 4      # spread over vehicles and over all four nuPlan cities
MAX_ATTEMPTS = 10000
# per-closure constants. "session" repeats the module constants above (pinned by the tests); "vehicle-day" is the alternative, with its own bounds
# (a vehicle-day is ~3.5 sessions and ~2,200 tokens, so ~5 units make ~10 % of the tokens).
MODES = {
    "session": dict(n_seats=N_SESSIONS, min_tok=MIN_TOK, max_tok=MAX_TOK, n_sessions=(N_SESSIONS, N_SESSIONS), min_veh=MIN_VEHICLES,
                    min_cities=MIN_CITIES, min_per_tercile=MIN_PER_TERCILE, veh_share=MAX_VEH_SHARE),
    "vehicle-day": dict(n_seats=5, min_tok=400, max_tok=3000, n_sessions=(12, 24), min_veh=3, min_cities=2, min_per_tercile=0, veh_share=0.5),
}


def session_of(log):
    return log.rsplit("_", 2)[0]


def vehicle_of(session):
    return session.split("_")[1]


def unit_of(session, closure):
    """The reserved unit a session belongs to: itself, or its (date, vehicle) pair."""
    return session if closure == "session" else "%s_%s" % (session.split("_")[0][:10], vehicle_of(session))


def u01(seed, attempt, purpose):
    """A uniform in [0, 1) from sha256 only: identical on every platform and Python version."""
    h = hashlib.sha256(("%s|%d|%d|%s" % (LABEL, seed, attempt, purpose)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- inputs
def load_inputs(src):
    """-> dict(sessions=..., navtest_logs=set, heldout_logs=set, vehicle_city=dict, totals=dict). `sessions[s] = {vehicle, logs{log: tokens}}`."""
    tt = os.path.join(src, "navtrain_token_table.tsv.gz")
    tok, in_navtest, rows, logless = collections.Counter(), 0, 0, 0
    with gzip.open(tt, "rt", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows += 1
            in_navtest += int(r["in_navtest"])
            if r["log"]:
                tok[r["log"]] += 1
            else:
                logless += 1
    yaml_logs = [ln.strip() for ln in open(os.path.join(src, "navtrain_logs.txt"), encoding="utf-8") if ln.strip()]
    unknown = set(tok) - set(yaml_logs)
    if unknown:
        raise SystemExit("token table has logs outside navtrain.yaml: %s" % sorted(unknown)[:3])
    sessions = {}
    for lg in yaml_logs:                               # yaml logs, including a log with no table row (0 tokens)
        s = sessions.setdefault(session_of(lg), {"vehicle": vehicle_of(session_of(lg)), "logs": {}})
        s["logs"][lg] = tok.get(lg, 0)
    navtest = set(json.load(open(os.path.join(src, "navtest_logs.json"), encoding="utf-8"))["logs"])
    heldout = {ln.strip() for ln in open(os.path.join(src, "sft_heldout_logs.txt"), encoding="utf-8") if ln.strip()}
    vcity = json.load(open(os.path.join(src, "vehicle_city.json"), encoding="utf-8"))["vehicle_city"]
    return {
        "sessions": sessions, "navtest_logs": navtest, "heldout_logs": heldout, "vehicle_city": vcity,
        "totals": {"table_rows": rows, "attributed_tokens": sum(tok.values()), "logless_tokens": logless,
                   "table_rows_flagged_navtest": in_navtest, "yaml_logs": len(yaml_logs), "sessions": len(sessions)},
    }


# ---------------------------------------------------------------------------------------------------------------- the rule
def _tokens(s):
    return sum(s["logs"].values())


def units_of(inp, closure):
    """-> {unit: [sessions]} over all navtrain sessions."""
    u = collections.defaultdict(list)
    for name in inp["sessions"]:
        u[unit_of(name, closure)].append(name)
    return u


_UC = {}


def units_of_cache(inp, closure):
    key = (id(inp), closure)
    if key not in _UC:
        _UC[key] = units_of(inp, closure)
    return _UC[key]


def eligible_pool(inp, closure="session"):
    """Rule 1, over UNITS (a unit is a session, or a (date, vehicle) pair). Raises if any navtrain session shares a log or a session with navtest
    (the split is session-disjoint by construction). A unit holding ANY SFT-held-out session is excluded whole."""
    sessions, K = inp["sessions"], MODES[closure]
    nt_sess = {session_of(lg) for lg in inp["navtest_logs"]}
    clash = sorted(set(sessions) & nt_sess) or sorted(lg for s in sessions.values() for lg in s["logs"] if lg in inp["navtest_logs"])
    if clash:
        raise SystemExit("REFUSED: navtrain and navtest share a session or log: %s" % clash[:3])
    ho_sess = {session_of(lg) for lg in inp["heldout_logs"]}
    veh_sessions = collections.Counter(s["vehicle"] for s in sessions.values())
    pool = {}
    for unit, members in units_of_cache(inp, closure).items():
        n = sum(_tokens(sessions[m]) for m in members)
        v = sessions[members[0]]["vehicle"]
        if any(m in ho_sess for m in members) or not (K["min_tok"] <= n <= K["max_tok"]) or veh_sessions[v] < MIN_VEH_SESS:
            continue
        pool[unit] = n
    return pool


def tercile_cuts(pool):
    xs = sorted(pool.values())
    return xs[len(xs) // 3], xs[(2 * len(xs)) // 3]


def stratum(n, cuts):
    return "small" if n < cuts[0] else ("large" if n >= cuts[1] else "mid")


def allocate(veh_mass, n, u0):
    """Rule 2: systematic PPS. Vehicles in name order; seat j sits at (u0 + j) / n of the cumulative mass; a vehicle gets the seats inside its span."""
    total, cum, out = float(sum(veh_mass.values())), 0.0, {}
    for v in sorted(veh_mass):
        lo = cum / total * n
        cum += veh_mass[v]
        hi = cum / total * n
        out[v] = max(0, math.ceil(hi - u0) - math.ceil(lo - u0))
    return out


def pick_within(items, k, u0):
    """Rule 3: equal rank spacing over `items` sorted by (tokens, name), seeded start. Distinct while len(items) >= k."""
    m = len(items)
    return [items[int((j + u0) * m / k)] for j in range(k)]


def attempt_draw(inp, pool, attempt, seed=SEED, closure="session"):
    """-> (sorted chosen UNITS, None) or (None, reason)."""
    tag = "" if closure == "session" else closure + ":"        # session-mode purposes keep their original strings (the recorded draw is pinned)
    by_veh = collections.defaultdict(list)
    for unit, n in pool.items():
        by_veh[inp["sessions"][units_of_cache(inp, closure)[unit][0]]["vehicle"]].append((n, unit))
    for v in by_veh:
        by_veh[v].sort()
    mass = {v: sum(n for n, _ in lst) for v, lst in by_veh.items()}
    seats = allocate(mass, MODES[closure]["n_seats"], u01(seed, attempt, tag + "vehicle-seats"))
    chosen = []
    for v in sorted(seats):
        k = seats[v]
        if k == 0:
            continue
        if k > len(by_veh[v]):
            return None, "vehicle %s was allocated %d seats but has %d eligible units" % (v, k, len(by_veh[v]))
        chosen += [name for _, name in pick_within(by_veh[v], k, u01(seed, attempt, tag + "within:" + v))]
    return sorted(chosen), None


def acceptance_failures(inp, pool, chosen, bank_scenes, closure="session"):
    """`chosen` = units. The bounds come from MODES[closure]; the checks and their order are the session-mode rule's."""
    sessions, K, bad = inp["sessions"], MODES[closure], []
    members = [m for u in chosen for m in units_of_cache(inp, closure)[u]]
    tok = sum(_tokens(sessions[m]) for m in members)
    lo_n, hi_n = K["n_sessions"]
    if not (lo_n <= len(members) <= hi_n):
        bad.append("n_sessions %d outside [%d, %d]" % (len(members), lo_n, hi_n))
    frac = tok / bank_scenes
    if not (FRAC_BAND[0] <= frac <= FRAC_BAND[1]):
        bad.append("token fraction %.4f outside %s" % (frac, FRAC_BAND))
    if K["min_per_tercile"]:
        cuts = tercile_cuts(pool)
        cnt = collections.Counter(stratum(pool[u], cuts) for u in chosen)
        if cnt["small"] < K["min_per_tercile"] or cnt["large"] < K["min_per_tercile"]:
            bad.append("size spread small=%d mid=%d large=%d" % (cnt["small"], cnt["mid"], cnt["large"]))
    vtok = collections.Counter()
    for m in members:
        vtok[sessions[m]["vehicle"]] += _tokens(sessions[m])
    vall = collections.Counter()
    for s in sessions.values():
        vall[s["vehicle"]] += _tokens(s)
    over = sorted(v for v in vtok if vtok[v] > K["veh_share"] * vall[v])
    if over:
        bad.append("vehicle share > %.2f: %s" % (K["veh_share"], over))
    if len(vtok) < K["min_veh"]:
        bad.append("vehicles %d < %d" % (len(vtok), K["min_veh"]))
    cities = {inp["vehicle_city"][v] for v in vtok}
    if len(cities) < K["min_cities"]:
        bad.append("cities %d < %d" % (len(cities), K["min_cities"]))
    return bad


def digest(lines):
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def draw(src, logless_probe=None, closure="session"):
    inp = load_inputs(src)
    pool = eligible_pool(inp, closure)
    bank_scenes = inp["totals"]["attributed_tokens"]
    rejected = []
    for attempt in range(MAX_ATTEMPTS):
        chosen_units, why = attempt_draw(inp, pool, attempt, closure=closure)
        bad = [why] if why else acceptance_failures(inp, pool, chosen_units, bank_scenes, closure)
        if not bad:
            break
        rejected.append({"attempt": attempt, "reasons": bad})
    else:
        raise SystemExit("no attempt met the acceptance bounds in %d tries" % MAX_ATTEMPTS)

    sessions = inp["sessions"]
    chosen = sorted(m for u in chosen_units for m in units_of_cache(inp, closure)[u])
    cuts = tercile_cuts(pool)
    ho_sess = {session_of(lg) for lg in inp["heldout_logs"]}
    out_sessions = []
    for s in chosen:
        out_sessions.append({
            "session": s, "unit": unit_of(s, closure), "vehicle": sessions[s]["vehicle"], "city": inp["vehicle_city"][sessions[s]["vehicle"]],
            "tokens": _tokens(sessions[s]), "size_stratum": stratum(pool[unit_of(s, closure)], cuts),
            "logs": [{"log": lg, "tokens": n} for lg, n in sorted(sessions[s]["logs"].items())],
        })
    res_tok = sum(x["tokens"] for x in out_sessions)
    res_logs = sorted(lg["log"] for x in out_sessions for lg in x["logs"])
    veh_rows = {}
    for v in sorted({s["vehicle"] for s in sessions.values()}):
        allv = [n for n, s in sessions.items() if s["vehicle"] == v]
        rv = [x for x in out_sessions if x["vehicle"] == v]
        veh_rows[v] = {"city": inp["vehicle_city"][v], "sessions_total": len(allv), "tokens_total": sum(_tokens(sessions[n]) for n in allv),
                       "sessions_reserved": len(rv), "tokens_reserved": sum(x["tokens"] for x in rv)}
    # the closure measurement: reserved sessions that keep a same-vehicle, same-date sibling session in TRAINING (navtest: 0 of 20 vehicle-days)
    res_set = set(chosen)
    vd = collections.defaultdict(list)
    for n in sessions:
        vd[unit_of(n, "vehicle-day")].append(n)
    straddle = sorted(n for n in chosen if any(m not in res_set for m in vd[unit_of(n, "vehicle-day")]))
    K = MODES[closure]
    result = {
        "schema": LABEL, "closure": closure, "seed": SEED, "attempt_accepted": attempt, "rejected_attempts": rejected,
        "rule": {"n_seats": K["n_seats"], "n_sessions_band": list(K["n_sessions"]), "fraction_band": list(FRAC_BAND), "min_tokens": K["min_tok"],
                 "max_tokens": K["max_tok"], "min_vehicle_sessions": MIN_VEH_SESS, "max_vehicle_share": K["veh_share"],
                 "min_per_tercile": K["min_per_tercile"], "min_vehicles": K["min_veh"], "min_cities": K["min_cities"],
                 "session_definition": "log name minus its last two '_' parts",
                 "token_count": "navtrain scenes attributed to the session's logs in navtrain_token_table.tsv.gz",
                 "unit": "session" if closure == "session" else "all sessions of one (date, vehicle) pair"},
        "pool": {"eligible_units": len(pool), "eligible_tokens": sum(pool.values()),
                 "excluded_heldout_sessions": len(ho_sess & set(sessions)), "tercile_cuts_tokens": list(cuts)},
        "inputs": {**inp["totals"], "sha256": {f: sha256_file(os.path.join(src, f)) for f in
                  ("navtrain_token_table.tsv.gz", "navtrain_logs.txt", "navtest_logs.json", "navtest_tokens.txt.gz", "sft_heldout_logs.txt",
                   "vehicle_city.json")}},
        "reserve": {
            "n_sessions": len(out_sessions), "n_units": len(chosen_units), "n_logs": len(res_logs), "tokens": res_tok,
            "fraction_of_bank_scenes": round(res_tok / inp["totals"]["attributed_tokens"], 6),
            "fraction_of_navtrain_yaml_tokens": round(res_tok / inp["totals"]["table_rows"], 6),
            "bank_scenes_left_for_training": inp["totals"]["attributed_tokens"] - res_tok,
            "sessions_sha256": digest([x["session"] for x in out_sessions]),
            "logs_sha256": digest(res_logs),
            "size_strata": dict(collections.Counter(x["size_stratum"] for x in out_sessions)),
            "cities": dict(collections.Counter(x["city"] for x in out_sessions)),
            "sessions_with_same_vehicle_same_date_sibling_in_training": len(straddle),
            "navtest_overlap": {"sessions": 0, "logs": 0, "table_rows_flagged_in_navtest": inp["totals"]["table_rows_flagged_navtest"]},
        },
        "sessions": out_sessions,
        "by_vehicle": veh_rows,
    }
    if logless_probe:
        probe = json.load(open(logless_probe, encoding="utf-8"))
        chosen_set = set(chosen)
        res_found = {t: v[0] for t, v in probe["found"].items() if session_of(v[0]) in chosen_set}
        result["logless_tokens"] = {
            "total": inp["totals"]["logless_tokens"], "resolved_to_a_log_by_probe": len(probe["found"]),
            "unresolved": inp["totals"]["logless_tokens"] - len(probe["found"]), "resolved_into_reserve_sessions": len(res_found),
            "resolved_into_reserve_tokens": sorted(res_found),
            "note": "251 table rows carry no log and were never served to the bank (served_train_sets = 0); the bank has 103,037 scenes = the attributed "
                    "rows. 103 resolve to a nuPlan DB (MEASURED, logless251_probe.json); 148 stay unattributed (UNVERIFIED which session). They are in "
                    "no bank row, but the aux label store holds 103,139 frames > 103,037, so a DB-resolved token can reach a label store: exclude by SESSION.",
            "probe_sha256": sha256_file(logless_probe),
        }
    return result


def dumps(res):
    return json.dumps(res, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--closure", choices=sorted(MODES), default="session")
    ap.add_argument("--check", action="store_true", help="re-draw and require byte-identity with --out; write nothing")
    a = ap.parse_args(argv)
    probe = os.path.join(a.src, "logless251_probe.json")
    text = dumps(draw(a.src, probe if os.path.exists(probe) else None, a.closure))
    if a.check:
        have = open(a.out, encoding="utf-8", newline="").read()
        if have != text:
            print("CHECK FAILED: the recorded draw differs from a fresh draw", file=sys.stderr)
            return 1
        print("CHECK OK: byte-identical to", a.out)
        return 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    r = json.loads(text)["reserve"]
    print("reserve: %d sessions, %d logs, %d tokens = %.2f %% of %d bank scenes; attempt %d; sessions_sha256 %s"
          % (r["n_sessions"], r["n_logs"], r["tokens"], 100 * r["fraction_of_bank_scenes"], json.loads(text)["inputs"]["attributed_tokens"],
             json.loads(text)["attempt_accepted"], r["sessions_sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
