"""PRE-WRITTEN cause rule for the 25 unrepaired REPAIR-D tokens (written and hashed BEFORE the classified table was produced; see ANATOMY_25.md).
Input: the N4D trace of one token (n4_repair_trace.py primary): iterations i = 0..5, each with metrics.dac, failed (clauses of the FULL certificate),
ext_max_m (worst outside extent, m, over the 41 steps and 4 corners), rv (None on the last candidate of a capped run, or when no corner is outside).
Precedence, first match wins:
  (b) CERT-REFUSED : some iteration has metrics.dac == 1.0 (a DAC = 1 candidate existed) but the certificate refused it. Cause = the failed clauses
                     among NC / DDC / TTC / C (the loop then stops: no outside corner remains, repair_target returns None, n4_repair.py:125-126).
  (d) NO-RV        : no DAC = 1 candidate, and an iteration had dac == 0 while return_vector() returned None (no outside corner over the 41 steps, so no
                     shift is defined; DAC fails for a reason the corner-vs-drivable-union geometry does not see).
  (c) NO-FOLLOW    : the cap was hit, and the median over consecutive iterations of realised = (ext_i - ext_{i+1}) / (ext_i + MARGIN) is < 0.25
                     (the label pipeline, i.e. the tracker inversion, does not carry the requested shift to the executed path).
  (a) CAP-SHRINKING: the cap was hit, DAC still 0, ext_5 < ext_4 and ext_5 < ext_0 (the loop was converging; more iterations or a larger margin could finish).
  (e) OTHER        : everything else (cap hit, ext not shrinking at the end: oscillation or growth); described by its ext sequence.
"""
MARGIN = 0.05
NOFOLLOW_BAR = 0.25


def classify(iters: list, margin: float = MARGIN) -> dict:
    ext = [float(i["ext_max_m"]) for i in iters]
    dac1 = [i for i in iters if i["metrics"]["dac"] == 1.0]
    if dac1:
        f = [c for c in dac1[0]["failed"] if c != "dac"]
        return {"cause": "b", "name": "CERT-REFUSED", "clauses": f, "it": dac1[0]["it"]}
    nrv = [i for i in iters if i["metrics"]["dac"] == 0.0 and i["rv"] is None and i["ext_max_m"] <= 0]
    if nrv:
        return {"cause": "d", "name": "NO-RV", "it": nrv[0]["it"]}
    real = [(ext[k] - ext[k + 1]) / (ext[k] + margin) for k in range(len(ext) - 1)]
    med = sorted(real)[len(real) // 2] if len(real) % 2 else 0.5 * (sorted(real)[len(real) // 2 - 1] + sorted(real)[len(real) // 2])
    if med < NOFOLLOW_BAR:
        return {"cause": "c", "name": "NO-FOLLOW", "median_realised": med}
    if ext[-1] < ext[-2] and ext[-1] < ext[0]:
        return {"cause": "a", "name": "CAP-SHRINKING", "median_realised": med}
    return {"cause": "e", "name": "OTHER", "ext": ext, "median_realised": med}
