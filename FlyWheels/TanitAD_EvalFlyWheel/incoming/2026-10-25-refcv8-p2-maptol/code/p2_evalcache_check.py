"""Prints '<n_banked> <n_equal>' : this run's eval-cache md5 list vs perc8's banked dev-box list (must read '139 139')."""
def load(p):
    d = {}
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if l:
            h, n = l.split(None, 1)
            d[n.lstrip("*")] = h
    return d


a = load("C:/Users/Admin/refcv8_pf/perc8tol/raw/md5_p2_evalcache.txt")
b = load("C:/Users/Admin/refcv8_pf/perc8/raw/md5_devbox_evalcache.txt")
print(len(b), sum(1 for k in b if a.get(k) == b[k]))
