"""make_dry_report.py -- reads the pulled dry-mint results (dry_results/RK_s{0,1}/ : selfcheck.json, checks.json, RH_GATE_RECORD.json, logs) and prints the census per bank and tick plus the checks.json summary as Markdown.
usage: python -I make_dry_report.py <dry_results dir> > DRY_MINT_RK_tables.md   (stdlib only)"""
import json
import os
import sys

D = sys.argv[1]


def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from find(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from find(v, key)


out = []
for s in (0, 1):
    d = f"{D}/RK_s{s}"
    p = f"{d}/selfcheck.json"
    if not os.path.isfile(p):
        out.append(f"## RK_s{s}: no selfcheck.json (see the logs in {d})\n")
        continue
    sc = json.load(open(p))
    dp = sc.get("dagger_probe") or {}
    counts = dp.get("kin_status_counts") or {}
    n_ok = sum(counts.values())
    kin = counts.get("kin_emitted", 0)
    out.append(f"## RK_s{s} (bank /root/rollhist/bank/s{s}, full bank)\n")
    out.append(f"* `declared_vs_built` {sc.get('declared_vs_built')}; `dagger_probe.pass` {dp.get('pass')}; `n_label_kin` {dp.get('n_label_kin')}; `rl_label_rows_changed` {dp.get('rl_label_rows_changed')}; `grad_diff_all` {dp.get('grad_diff_all')}; `control_p0_grad_diff` {dp.get('control_p0_grad_diff')}")
    out.append(f"* **relabelled share of ok records (bar 0.5): {kin / n_ok:.4f}** ({kin:,} of {n_ok:,}); `kin_status_counts` = {json.dumps(counts)}" if n_ok else "* no kin_status_counts")
    sl = [v for v in find(sc.get("info", {}), "speed_label") if isinstance(v, dict)]
    if sl:
        x = sl[0]
        out.append(f"* labeller wall (CPU, nice 19, shared pod): `wall_s` {x.get('wall_s')} s; n_eligible {x.get('n_eligible')}, n_eligible_after {x.get('n_eligible_after')}, n_labelled {x.get('n_labelled')}, n_emitted {x.get('n_emitted')}")
        cen = x.get("census") or {}
        pt = cen.get("per_tick") or {}
        if pt:
            out.append("\n| tick | ok | relabelled | identity | relabelled share | refused by reason |\n|---|---|---|---|---|---|")
            for t, r in pt.items():
                sh = r["relabelled"] / r["ok"] if r.get("ok") else float("nan")
                out.append(f"| {t} | {r['ok']:,} | {r['relabelled']:,} | {r['identity']:,} | {sh:.4f} | {json.dumps(r.get('refused', {}))} |")
            out.append(f"\n* census `relabelled_share_of_ok` (labeller's own): {cen.get('relabelled_share_of_ok')}")
    gr = f"{d}/RH_GATE_RECORD.json"
    gf = f"{d}/RH_GATE_REFUSED.json"
    out.append(f"* gate record: {'MINTED (pass ' + str(json.load(open(gr)).get('pass')) + ', census ' + str(json.load(open(gr)).get('census', {}).get('relabelled_share_of_ok')) + ')' if os.path.isfile(gr) else 'REFUSED: ' + str(json.load(open(gf)).get('problems')) if os.path.isfile(gf) else 'none'}")
    ck = f"{d}/checks.json"
    if os.path.isfile(ck):
        c = json.load(open(ck))
        cs = c.get("checks", c)
        out.append("\n**checks.json (study-side dry mint), all checks:**\n\n| check | pass | note |\n|---|---|---|")
        for k, v in cs.items():
            out.append(f"| {k} | {v.get('pass')} | {str(v.get('problems') or v.get('note') or '')[:160]} |")
    out.append("")
print("\n".join(out))
