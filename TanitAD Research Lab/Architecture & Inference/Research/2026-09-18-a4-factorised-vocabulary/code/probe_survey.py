import json, hashlib, pathlib

base = pathlib.Path('TanitAD Research Lab/Architecture & Inference/Implementation/'
                    'incoming/2026-07-29-deep-research-sota')
out = {"probe": "A4 survey record - what the 0-3 vote actually contains",
       "date": "2026-09-18", "evidence_class": "MEASURED (ours)", "files": {}}
for name in ("raw_report1_autonomous_driving.json", "raw_report2_robotics_wm_jepa.json"):
    p = base / name
    d = json.loads(p.read_text(encoding='utf-8'))
    wp = d.get('workflowProgress', [])
    previews = [e['resultPreview'] for e in wp if 'resultPreview' in e]
    out['files'][name] = {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "n_workflowProgress": len(wp),
        "n_resultPreview": len(previews),
        "distinct_resultPreview_lengths": sorted(set(len(x) for x in previews)),
    }
d = json.loads((base / 'raw_report1_autonomous_driving.json').read_text(encoding='utf-8'))
r = d['result']['refuted'][12]
out['refuted_entry_12'] = {"keys": list(r.keys())}
out['refuted_entry_12'].update(r)
out['logs_23'] = d['logs'][23]
out['verifier_previews'] = {}
for i in (83, 84, 85):
    e = d['workflowProgress'][i]
    out['verifier_previews']["workflowProgress[%d]" % i] = {
        "label": e.get('label'), "lastToolSummary": e.get('lastToolSummary'),
        "resultPreview": e.get('resultPreview'),
        "resultPreview_len": len(e.get('resultPreview', '')),
    }
out['open_question_2'] = d['result']['openQuestions'][2]
lib = pathlib.Path('TanitAD Research Lab/Library/library.json')
out['library_banking'] = {
    "library_json": lib.as_posix(),
    "occurrences_of_29163": lib.read_text(encoding='utf-8').count('29163'),
    "reading": ("0 => arXiv 2603.29163 is NOT banked; any claim about its tables "
                "is PUBLISHED-SECONDARY and inadmissible."),
}
dst = pathlib.Path('TanitAD Research Lab/Architecture & Inference/Research/'
                   '2026-09-18-a4-factorised-vocabulary/raw/survey_record_probe.json')
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(out['files'], indent=1))
print('refuted[12] keys:', out['refuted_entry_12']['keys'])
print('library 29163 count:', out['library_banking']['occurrences_of_29163'])
print('wrote', dst)
