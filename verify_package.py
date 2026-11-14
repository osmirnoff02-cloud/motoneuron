"""Codex 2026-11-13: standard-library-only integrity and portability checks."""
from pathlib import Path
import ast, hashlib, json, re, sys

ROOT=Path(__file__).resolve().parent
manifest=ROOT/'provenance/package_manifest.json'
if not manifest.is_file():raise SystemExit('Package manifest is missing; finish the export before distribution.')
rows=json.loads(manifest.read_text(encoding='utf-8'))['files']
errors=[]
for row in rows:
    p=ROOT/row['path']
    if not p.is_file():errors.append('Missing '+row['path']);continue
    raw=p.read_bytes()
    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:errors.append('Changed '+row['path'])
    if p.suffix=='.py':
        try:ast.parse(raw.decode('utf-8-sig'),filename=row['path'])
        except SyntaxError as e:errors.append(str(e))
    if p.suffix=='.json':
        try:json.loads(raw)
        except ValueError as e:errors.append(str(e))
    if len(raw)>100*1024**2:errors.append('Too large for ordinary Git: '+row['path'])
    if any(x in p.relative_to(ROOT).parts for x in ('pacient','letters','.git','node_modules')):errors.append('Excluded category '+row['path'])
    if p.suffix in ('.md','.py','.json','.txt','.csv','.cff','.yml'):
        s=raw.decode('utf-8-sig',errors='replace')
        if re.search(r'(?:[A-Za-z]:[\\/]+Users[\\/]+|(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',s):
            errors.append('Review sensitive-looking text: '+row['path'])
    if p.suffix=='.md':
        text=raw.decode('utf-8-sig')
        for link in re.findall(r'\]\(([^)]+)\)',text):
            link=link.strip('<>').split('#')[0]
            if not link or '://' in link or link.startswith('mailto:'):continue
            if not (p.parent/link).exists():errors.append('Broken documentation link: '+row['path']+' -> '+link)
print(json.dumps({'files_checked':len(rows),'bytes_checked':sum(x['bytes'] for x in rows),'errors':errors,'scope':'Hash/schema/syntax/link checks and selected sensitive-pattern screening; not exhaustive rights or privacy clearance.'},indent=2))
if errors:raise SystemExit(1)
