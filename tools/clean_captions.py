"""清洗中文 caption：按长度过滤（配合 recaption_zh 的 --keep-truncated）"""
import json, re, argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--inp', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--max-chars', type=int, default=40)
a = ap.parse_args()

seen, rows = set(), []
for f in a.inp:
    p = Path(f)
    if not p.exists():
        print('[skip] %s not found' % p)
        continue
    for line in p.open(encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        zh = (r.get('zh') or '').strip()
        tg = (r.get('tags') or '').strip()
        if not zh or not tg:
            continue
        if len(zh) > a.max_chars:      # ⛔ 超长= 很可能被 max_new 截断成残句
            continue
        key = tg[:120]
        if key in seen:               # 去重
            continue
        seen.add(key)
        rows.append({'i': len(rows), 'tags': tg, 'zh': zh})

outp = Path(a.out)
outp.parent.mkdir(parents=True, exist_ok=True)
with outp.open('w', encoding='utf-8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
ls = [len(r['zh']) for r in rows]
print('[OK] %s  %d rows  平均 %.0f 字  最长 %d'
      % (outp, len(rows), (sum(ls) / len(ls)) if ls else 0, max(ls) if ls else 0))
if rows:
    for r in rows[:3]:
        print('   ', r['zh'])
