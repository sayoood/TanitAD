import multiprocessing as mp, os, sys, time

def init(x):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_calls.log"), "a") as fh:
        fh.write(f"{os.getpid()}\n")
    raise SystemExit("patch not active")

def job(i):
    return i

if __name__ == "__main__":
    t0 = time.time()
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_calls.log")
    if os.path.exists(log):
        os.remove(log)
    with mp.get_context("spawn").Pool(2, initializer=init, initargs=(1,)) as pool:
        r = pool.imap_unordered(job, range(4))
        try:
            print("got", next(r, None))   # blocks forever if the pool keeps respawning
        except Exception as e:
            print("raised", type(e).__name__)
    print("finished", round(time.time() - t0, 1))
