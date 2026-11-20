"""Parameterized backfill for CFE judge uids that timed out via the CLI backend.

Uses the repository's exact prompt + score parser, but calls Hy4 over direct
HTTP (the CLI subprocess path intermittently hangs on long-CoT rows).
Appends most-recent-wins records to cfe.jsonl; evaluator takes latest per uid.

Usage:
  python -m scripts.backfill_cfe_http_generic --eval_dir <dir> --split <split> \
      --env_file /usr/local/workspace/.env --token_key HY4_TOKEN
"""
import argparse
import csv
import json
import os
import time
import urllib.request

from scripts.judge_cfe_rtofu import build_prompt, parse_score

URL = 'https://tokenlink.woa.com/openapi/v1/chat/completions'


def read_token(env_file, token_key):
    with open(env_file) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(token_key + '='):
                return line.split('=', 1)[1].strip()
    raise SystemExit('%s not found in %s' % (token_key, env_file))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--env_file', default='/usr/local/workspace/.env')
    ap.add_argument('--token_key', default='HY4_TOKEN')
    args = ap.parse_args()

    token = read_token(args.env_file, args.token_key)
    csv_path = os.path.join(args.eval_dir, args.split + '.csv')
    jsonl_path = os.path.join(args.eval_dir, 'cfe.jsonl')

    last = {}
    with open(jsonl_path) as fh:
        for line in fh:
            rec = json.loads(line)
            last[rec['uid']] = rec
    targets = [uid for uid, rec in last.items() if rec.get('status') != 'ok']
    print('targets to backfill:', targets, flush=True)
    if not targets:
        print('nothing to backfill', flush=True)
        return

    rows = {}
    with open(csv_path, newline='') as fh:
        for row in csv.DictReader(fh):
            rows[row['index']] = row

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

    with open(jsonl_path, 'a') as out:
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
                except Exception as exc:
                    print('%s attempt %d failed (%.1fs): %s' % (uid, attempt, time.time()-t0, exc), flush=True)
                    time.sleep(2 ** attempt)
            else:
                print('%s FAILED after retries' % uid, flush=True)

    print('backfill done', flush=True)


if __name__ == '__main__':
    main()
