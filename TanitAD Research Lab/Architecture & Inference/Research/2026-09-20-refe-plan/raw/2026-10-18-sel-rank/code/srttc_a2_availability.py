"""SR-TTC Amendment 2 (DRAFT): dev-box staging check for the nested log subsample.

FILE EXISTENCE AND FILE SIZE ONLY - no file is opened. Availability is REPORTED, never used to
select logs (the selection is srttc_a2_select.py's sha256 rule, fixed before this ran).
Frame completeness INSIDE a camera zip is NOT checked (the held-out table found zips present
with needed frames missing), so "4 zips present" is an upper bound on frame availability.

Usage: python -I srttc_a2_availability.py <ranked_fit_logs.tsv> <db_dir> <pixel_dir> <out.json>
"""
import csv, json, os, statistics as st, sys

CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
FRACTIONS = ["0.10", "0.25", "0.50", "1.00"]


def main(ranked, dbdir, px, out):
    rows = list(csv.DictReader(open(ranked, encoding="utf-8"), delimiter="\t"))
    dbs = {f[:-3] for f in os.listdir(dbdir) if f.endswith(".db")}
    zips = set(os.listdir(px))
    z4 = lambda lg: all(f"{lg}_{c}.zip" in zips for c in CAMS)
    db_bytes = [os.path.getsize(f"{dbdir}/{r['log']}.db") for r in rows if r["log"] in dbs]
    res = {"what": "dev-box availability of nuPlan DB + 4 camera zips for the SR-TTC fit logs (existence only)",
           "db_dir_n_db": len(dbs), "pixel_dir_n_zip": len(zips),
           "fit_log_db_MB": {"n": len(db_bytes), "median": round(st.median(db_bytes) / 1e6, 1),
                             "mean": round(st.mean(db_bytes) / 1e6, 1)},
           "fractions": {}}
    mean_db = st.mean(db_bytes)
    for fr in FRACTIONS:
        sel = [r for r in rows if float(r["first_fraction"]) <= float(fr)]
        tok = lambda L: sum(int(r["n_fit_tokens"]) for r in L)
        has_db = [r for r in sel if r["log"] in dbs]
        has_z = [r for r in sel if z4(r["log"])]
        both = [r for r in sel if r["log"] in dbs and z4(r["log"])]
        miss_db = len(sel) - len(has_db)
        res["fractions"][f"{int(float(fr) * 100)}%"] = {
            "logs": len(sel), "tokens": tok(sel),
            "logs_with_db": len(has_db), "logs_with_4_zips": len(has_z), "logs_with_both": len(both),
            "tokens_in_logs_with_both": tok(both),
            "logs_missing_db": miss_db,
            "missing_db_GB_at_mean_size_ESTIMATED": round(miss_db * mean_db / 1e9, 1)}
    with open(out, "w", encoding="utf-8", newline="\n") as w:
        json.dump(res, w, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
