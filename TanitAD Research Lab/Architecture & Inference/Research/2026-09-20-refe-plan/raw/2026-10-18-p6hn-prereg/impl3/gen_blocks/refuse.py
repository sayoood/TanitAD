    bad_hn = hn_refusals(a, ap)                                         # P6-HN-S: --hn-bank / --ctx-cache levers named inconsistently never start
    if bad_hn:
        for msg in bad_hn:
            print("REFUSED: " + msg)
        return 4
