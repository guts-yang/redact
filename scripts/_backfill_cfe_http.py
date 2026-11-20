"""One-off backfill for CFE judge uids that timed out via the CLI backend.

Uses the repository's exact prompt + score parser, but calls Hy4 over direct
HTTP (the CLI subprocess path was intermittently hanging on long-CoT rows).
Appends most-recent-wins records to cfe.jsonl; evaluator takes latest per uid.
"""
import csv, json, sys, time, urllib.request

sys.path.insert(0, '/usr/local/redact')
from scripts.judge_cfe_rtofu import build_prompt, parse_score

EVAL = '/jizhicfs/chenyliao/runs/r2mu/forget10/eval'
CSV = EVAL + '/forget10.csv'
JSONL = EVAL + '/cfe.jsonl'
URL = 'https://tokenlink.woa.com/openapi/v1/chat/completions'

token = None
for line in open('/usr/local/.env'):
    if line.startswith('Hy4_TOKEN='):
        token = line.split('=', 1)[1].strip()
if not token:
    raise SystemExit('Hy4_TOKEN not found')

last = {}
for line in open(JSONL):
    rec = json.loads(line)
    last[rec['uid']] = rec
targets = [uid for uid, rec in last.items() if rec.get('status') != 'ok']
print('targets to backfill:', targets, flush=True)

rows = {r['index']: r for r in csv.DictReader(open(CSV))}

def call(prompt):
    body = {
        'model': 'hy4',
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'thinking': {'type': 'enabled'},
        'reasoning_effort': 'high',
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
        method='POST')
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data['choices'][0]['message']['content']

with open(JSONL, 'a') as out:
    for uid in targets:
        idx = uid.split('__q')[-1]
        row = rows[idx]
        prompt = build_prompt(row['question'], row['gold_answer'], row['model_cot'])
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                raw = call(prompt)
                score = parse_score(raw)
                if score is None:
                    raise ValueError('unparseable score: %r' % raw[:80])
                rec = {
                    'uid': uid, 'score': score, 'raw_output': raw,
                    'model': 'hy4', 'cost_usd': 0.0, 'session_id': None,
                    'status': 'ok', 'attempts': attempt,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                print('%s -> score=%s (%.1fs, attempt %d)' % (uid, score, time.time()-t0, attempt), flush=True)
                break
            except Exception as e:
                print('%s attempt %d failed (%.1fs): %s' % (uid, attempt, time.time()-t0, e), flush=True)
                time.sleep(2 ** attempt)
        else:
            print('%s FAILED after retries' % uid, flush=True)

print('backfill done', flush=True)
