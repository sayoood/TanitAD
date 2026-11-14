R="C:/Users/Admin/v8proof_c/motion/checks/"
def edit(f, pairs):
    s=open(R+f,encoding="utf-8",newline="").read()
    for a,b in pairs:
        assert s.count(a)==1,(f,a); s=s.replace(a,b)
    open(R+f,"w",encoding="utf-8",newline="").write(s)
edit("m1_identity.py",[("/tmp/m1/","/tmp/m1f/")])
edit("m_worker.py",[('base = ["--synthetic",','base = (["--no-resume"] if os.environ.get("MW_NORESUME") else []) + ["--synthetic",')])
edit("m7_cost.py",[('PYTHONDONTWRITEBYTECODE="1",\n','PYTHONDONTWRITEBYTECODE="1", MW_NORESUME="1",\n'),
  ('    rows = [json.loads(l) for l in open(f"{out}/run/train_log.jsonl", encoding="utf-8")]\n','    rows = [json.loads(l) for l in open(f"{out}/run/train_log.jsonl", encoding="utf-8")][-STEPS:]      # the log is opened in append mode: this repetition = the last STEPS rows\n')])
