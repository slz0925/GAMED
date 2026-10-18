"""Optional filesystem routing. No model, dataset, or metric policy lives here.

With no configuration or overrides, every input is returned unchanged.
Read routing never extracts, copies, downloads, or writes an original resource.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_ROOT_PREFIXES = (
    '/root/autodl-tmp/ssd/yyh/tmp/fnd-bootstrap-master',
    '/ssd/yyh/tmp/fnd-bootstrap-master',
    '/data/yyh/tmp/fnd-bootstrap-master',
)
_ENV_KEYS = {
    'resource_root': 'FND_RESOURCE_ROOT', 'groupshare_root': 'FND_GROUPSHARE_ROOT',
    'bert_root': 'FND_BERT_ROOT', 'cache_root': 'FND_CACHE_ROOT',
    'output_root': 'FND_OUTPUT_ROOT',
}
_GAMED_ENV_KEYS = {key: name.replace('FND_', 'GAMED_', 1) for key, name in _ENV_KEYS.items()}


@lru_cache(maxsize=8)
def _load_config(filename):
    path = Path(filename)
    if not path.exists():
        explicit_config = os.environ.get('GAMED_CONFIG') or os.environ.get('FND_CONFIG')
        if explicit_config:
            raise FileNotFoundError('Configured GAMED path file does not exist: ' + str(path))
        return {}
    with path.open(encoding='utf-8-sig') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError('GAMED configuration must be a JSON object.')
    if not isinstance(value.get('prefix_map', {}), dict):
        raise ValueError('prefix_map must be an object of path prefixes.')
    return value


def settings():
    """Return config values with explicit environment overrides applied."""
    filename = os.environ.get('GAMED_CONFIG') or os.environ.get('FND_CONFIG', str(PROJECT_ROOT/'config/local.json'))
    value = dict(_load_config(str(Path(filename).resolve())))
    for key, env in _ENV_KEYS.items():
        if env in os.environ:
            value[key] = os.environ[env] or None
    for key, env in _GAMED_ENV_KEYS.items():
        if env in os.environ:
            value[key] = os.environ[env] or None
    return value


def _root(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else PROJECT_ROOT/path


def _suffix(path, prefix):
    prefix = prefix.replace('\\', '/').rstrip('/')
    if not prefix:
        raise ValueError('Path prefixes must not be empty.')
    # Windows paths are case-insensitive; Linux legacy paths stay case-sensitive.
    is_windows = len(prefix)>1 and prefix[1]==':'
    lhs, rhs = (path.casefold(), prefix.casefold()) if is_windows else (path, prefix)
    if lhs == rhs:
        return ''
    if lhs.startswith(rhs+'/'):
        return path[len(prefix)+1:]
    return None


def _mapped(path, prefix, destination):
    suffix = _suffix(path, prefix)
    if suffix is None:
        return None
    # Normalize separators without treating common-prefix siblings as children.
    return str(_root(destination).joinpath(*[p for p in suffix.split('/') if p not in ('', '.')]))


def resolve_read_path(value):
    """Map a configured resource location; preserve unconfigured inputs exactly."""
    if not isinstance(value, (str, os.PathLike)):
        return value
    cfg = settings()
    if not cfg:
        return value
    raw = os.fspath(value)
    if not isinstance(raw, str):
        return value
    path = raw.replace('\\', '/')
    for prefix, destination in sorted(cfg.get('prefix_map', {}).items(), key=lambda x: len(x[0]), reverse=True):
        if destination:
            mapped = _mapped(path, prefix, destination)
            if mapped is not None:
                return mapped
    if cfg.get('resource_root'):
        for prefix in _ROOT_PREFIXES:
            mapped = _mapped(path, prefix, cfg['resource_root'])
            if mapped is not None:
                return mapped
        local = _suffix(path, PROJECT_ROOT.as_posix())
        relative = local if local is not None else path
        while relative.startswith('./'):
            relative = relative[2:]
        if relative.split('/')[0] in ('dataset', 'checkpoint', 'ernie'):
            # An existing local checkpoint takes precedence over external resources.
            candidate=PROJECT_ROOT.joinpath(*[p for p in relative.split('/') if p not in ('', '.')])
            if candidate.is_file():
                return value
            return str(_root(cfg['resource_root']).joinpath(*[p for p in relative.split('/') if p not in ('', '.')]))
    for key, prefixes in (
        ('groupshare_root', ('/home/groupshare', '/groupshare')),
        ('bert_root', ('/ssd/yyh/model',)),
    ):
        if cfg.get(key):
            for prefix in prefixes:
                mapped = _mapped(path, prefix, cfg[key])
                if mapped is not None:
                    return mapped
    filename=path[2:] if path.startswith('./') else path
    if cfg.get('cache_root') and '/' not in filename:
        if filename.endswith(('.pickle', '.npz')):
            return str(_root(cfg['cache_root'])/filename)
    # Leave Hugging Face IDs, file-like objects and unrelated paths unchanged.
    return value


def resolve_write_path(value):
    """Route supported absolute result locations; leave other writes unchanged."""
    if not isinstance(value, (str, os.PathLike)):
        return value
    cfg=settings()
    if not cfg.get('output_root'):
        return value
    raw=os.fspath(value)
    if not isinstance(raw,str):
        return value
    path=raw.replace('\\','/')
    for prefix, subdir in (
        ('/home/groupshare/mae-main/example', 'examples'),
        ('/home/groupshare/CIKM_ying_output', 'checkpoints'),
        ('/groupshare/CIKM_ying_output', 'checkpoints'),
    ):
        mapped=_mapped(path,prefix,_root(cfg['output_root'])/subdir)
        if mapped is not None:
            return mapped
    return value

