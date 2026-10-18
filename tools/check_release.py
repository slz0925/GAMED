"""Check public inventory, Python syntax and local Markdown file links.

No research imports, network calls, bytecode generation or Git changes.
This checks source packaging; it does not validate model execution.
"""
import ast
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_EXTENSIONS = {
    '.pth', '.pt', '.pkl', '.bin', '.safetensors', '.ckpt', '.zip', '.npz',
    '.pickle', '.csv', '.tsv', '.xlsx', '.out', '.log', '.pyc', '.pem', '.key',
}


def check():
    result = {'inventory_files':0, 'python_files_compiled':0,
              'local_markdown_links_checked':0, 'local_imports_checked':0,
              'secret_patterns_checked':0, 'errors':[]}
    manifest = ROOT/'PUBLISH_FILES.txt'
    if not manifest.is_file():
        result['errors'].append('PUBLISH_FILES.txt is missing')
        return result
    names = manifest.read_text(encoding='utf-8').splitlines()
    result['inventory_files'] = len(names)
    if names != sorted(set(names)):
        result['errors'].append('Inventory must be sorted with unique entries')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    # Unlisted files should not be included in a distributable package.
    extras = actual-set(names)
    for name in sorted(extras):
        if name == 'config/local.json' or name.startswith('outputs/') or '__pycache__' in name.split('/'):
            continue
        result['errors'].append('Unlisted file in source tree: '+name)
    local_roots = {p.stem for p in ROOT.glob('*.py')} | {
        p.name for p in ROOT.iterdir() if p.is_dir() and any(p.rglob('*.py'))}
    secret_patterns = (
        r'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'\bghp_[A-Za-z0-9]{30,}\b',
        r'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
        r'\bAKIA[A-Z0-9]{16}\b',
        r'\bAIza[A-Za-z0-9_-]{30,}\b',
        r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b',
        r'(?i)(?:password|api_key|access_token)\s*[:=]\s*[\"\'][^\"\'\s]{12,}[\"\']',
    )
    for name in names:
        relative = PurePosixPath(name)
        path = ROOT/name
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(ROOT):
            result['errors'].append('Unsafe inventory path: '+name)
            continue
        if not path.is_file():
            result['errors'].append('Missing inventory file: '+name)
            continue
        if set(relative.parts) & {'.git', '.idea', '.vscode', '__pycache__', '.ipynb_checkpoints'}:
            result['errors'].append('Local editor/cache file in inventory: '+name)
        if (name == 'config/local.json' or name.startswith(('dataset/', 'ernie/', 'outputs/'))
                and name != 'outputs/.gitkeep'
                or path.suffix.lower() in FORBIDDEN_EXTENSIONS
                or path.name.startswith('.env') and path.name != '.env.example'
                or path.name.endswith('.local.json')
                or path.stat().st_size > 10*1024*1024):
            result['errors'].append('Private/resource/generated file in inventory: '+name)
        if path.suffix == '.py':
            try:
                # compile, rather than parse only, also checks compiler restrictions.
                compile(path.read_bytes(), name, 'exec', dont_inherit=True)
                result['python_files_compiled'] += 1
                tree=ast.parse(path.read_text(encoding='utf-8-sig'))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import):
                        modules=[alias.name for alias in node.names]
                    elif isinstance(node,ast.ImportFrom):
                        modules=[node.module or '']
                    else:
                        continue
                    for module in modules:
                        if isinstance(node,ast.ImportFrom) and node.level:
                            base=path.parent
                            for _ in range(node.level-1):base=base.parent
                            target=base.joinpath(*module.split('.')) if module else base
                        elif module.split('.')[0] in local_roots:
                            target=ROOT.joinpath(*module.split('.'))
                        elif module.startswith('test_'):
                            target=path.parent/module
                        else:
                            continue
                        result['local_imports_checked']+=1
                        if not (target.is_dir() or target.with_suffix('.py').is_file()):
                            result['errors'].append(name+': unresolved local import '+module)
            except (SyntaxError, ValueError) as error:
                result['errors'].append(name+': '+str(error))
        if path.suffix == '.md':
            text = path.read_text(encoding='utf-8-sig')
            text = re.sub(r'```.*?```', '', text, flags=re.S)
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                target = target.strip().strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                linked = (path.parent/unquote(parsed.path)).resolve()
                result['local_markdown_links_checked'] += 1
                if not linked.is_relative_to(ROOT) or not linked.exists():
                    result['errors'].append(name+': broken local link '+target)
                elif linked.is_file() and linked.relative_to(ROOT).as_posix() not in names:
                    result['errors'].append(name+': linked file absent from public inventory '+target)
        try:
            text=path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            result['errors'].append('Non-UTF-8 source file: '+name)
            continue
        for pattern in secret_patterns:
            result['secret_patterns_checked']+=1
            if re.search(pattern,text):
                # Report only the filename, never the credential value.
                result['errors'].append('Potential credential in '+name)
                break
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if line.startswith(('-r ','--requirement ')):
            target=(ROOT/line.split(None,1)[1]).resolve()
            if not target.is_file() or not target.is_relative_to(ROOT):
                result['errors'].append('Missing/unsafe requirements include')
    return result


if __name__ == '__main__':
    output = check()
    print(json.dumps(output, ensure_ascii=True, indent=2))
    raise SystemExit(bool(output['errors']))
