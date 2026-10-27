"""Optional filesystem routing for datasets, checkpoints and encoder resources."""
import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_GAMED_ENV_KEYS = {'resource_root': 'GAMED_RESOURCE_ROOT'}
_ENV_KEYS = {'resource_root': 'FND_RESOURCE_ROOT'}
ENGLISH_ENCODER_PATH = 'ernie/models--nghuyong--ernie-2.0-base-en'


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
    # Windows prefixes are case-insensitive; Unix prefixes stay case-sensitive.
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
        local = _suffix(path, PROJECT_ROOT.as_posix())
        relative = local if local is not None else path
        while relative.startswith('./'):
            relative = relative[2:]
        if relative.split('/')[0] in ('dataset', 'checkpoint', 'ernie'):
            # Existing local resource files take precedence over the external root.
            candidate=PROJECT_ROOT.joinpath(*[p for p in relative.split('/') if p not in ('', '.')])
            if candidate.is_file():
                return value
            return str(_root(cfg['resource_root']).joinpath(*[p for p in relative.split('/') if p not in ('', '.')]))
    # Leave Hugging Face IDs, file-like objects and unrelated paths unchanged.
    return value


def resolve_english_encoder_path():
    """Locate the model and tokenizer through the shared resource configuration."""
    return resolve_read_path(ENGLISH_ENCODER_PATH)
