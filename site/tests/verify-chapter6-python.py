"""Verify chapter six content, causal behavior, and rejection of unfinished work."""
import ast
import json
from pathlib import Path
import runpy
import contextlib
import io

ROOT=Path(__file__).resolve().parents[1]
DOWNLOADS=ROOT/'public'/'downloads'
data=json.loads((ROOT/'lib'/'ml-chapter-six.json').read_text(encoding='utf-8'))
assert len(data['program'])==12 and len(data['courses'])==len(data['lessons'])==12
assert sum(len(c['quiz']) for c in data['courses'].values())==60
for course in data['courses'].values():
    assert all(len(q['options'])==4 and 0<=q['answer']<4 for q in course['quiz'])
    assert 'npimport' not in course['text'] and 'pdimport' not in course['text']
    for block in __import__('re').findall(r'~~~python\n(.*?)~~~',course['text'],flags=__import__('re').S):
        ast.parse(block)
for script in DOWNLOADS.glob('ch06_cours*.py'):
    ast.parse(script.read_text(encoding='utf-8'))
for number in range(1,13):
    for kind in ['lab','starter','solution']:
        filename=f'ch06_cours{number:02d}_{kind}.py'
        local=ROOT.parent/'03_Machine_learning'/'CHAPITRE_06'/filename
        # The hosted source checkout contains only the site; compare when local chapter sources exist.
        if local.exists():assert local.read_bytes()==(DOWNLOADS/filename).read_bytes(),filename
count=0
for number in range(1,13):
    exercise=f'ch06_cours{number:02d}'
    with contextlib.redirect_stdout(io.StringIO()):
        namespace=runpy.run_path(str(DOWNLOADS/(exercise+'_solution.py')),run_name='verification')
        result=runpy.run_path(str(ROOT/'public'/'python'/'checks.py'),init_globals={'namespace':namespace,'exercise':exercise})['results']
    assert result and all(row['passed'] for row in result),(exercise,result)
    count+=len(result)
    if number>=3:
        starter=runpy.run_path(str(DOWNLOADS/(exercise+'_starter.py')),run_name='verification')
        failed=runpy.run_path(str(ROOT/'public'/'python'/'checks.py'),init_globals={'namespace':starter,'exercise':exercise})['results']
        assert any(not row['passed'] for row in failed),exercise
    print('PASS',exercise)
print(f'PASS {count} behavioral checks, 60 quiz questions, 12 courses and 12 program entries')
