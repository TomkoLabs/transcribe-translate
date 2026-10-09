"""Repeatable, recoverable installation at the final canonical runtime path."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from transcribe_translate.config import Config, load_config
from transcribe_translate.installation import editable_root, installation_lock, source_version
from setup_runtime import create_defaults, preflight
from transcribe_translate.models import cache_operation, locations
from transcribe_translate.managed import runtime_stamp


def run(command, *, capture=False, env=None):
    return subprocess.run(command, check=True, text=True, capture_output=capture,
                          env=env or clean_environment(), cwd=ROOT)


def clean_environment():
    # Preserve credentials/model caches, and intentional index/cache controls.
    # Drop interpreter injection and pip location/options that could install into
    # a foreign environment or add requirements outside our pinned files.
    pip_controls = {'PIP_NO_INDEX', 'PIP_FIND_LINKS', 'PIP_INDEX_URL',
                    'PIP_EXTRA_INDEX_URL', 'PIP_CACHE_DIR', 'PIP_CERT',
                    'PIP_CLIENT_CERT', 'PIP_TRUSTED_HOST', 'PIP_TIMEOUT',
                    'PIP_DEFAULT_TIMEOUT', 'PIP_RETRIES', 'PIP_PROXY'}
    env = {k:v for k,v in os.environ.items()
           if not k.startswith('PYTHON') and k != 'VIRTUAL_ENV'
           and (not k.startswith('PIP_') or k in pip_controls)}
    env['PIP_CONFIG_FILE'] = os.devnull
    env.setdefault('PIP_CACHE_DIR', str(ROOT/'Work/Installation/pip'))
    env.setdefault('HF_XET_CACHE', str(ROOT/'Work/Installation/huggingface/xet'))
    env.setdefault('HF_ASSETS_CACHE', str(ROOT/'Work/Installation/huggingface/assets'))
    env['TT_ROOT'] = str(ROOT)
    return env


def old_checkout(runtime: Path) -> Path | None:
    # Read local metadata rather than importing a potentially foreign installation.
    if runtime.is_symlink():
        return None
    candidates = list(runtime.glob('lib/python*/site-packages/transcribe_translate-*.dist-info/direct_url.json'))
    roots = set()
    for path in candidates:
        try:
            root = editable_root(path)
            if root is not None:
                roots.add(root)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return next(iter(roots)) if len(roots) == 1 else None


def unsafe_links(runtime: Path) -> bool:
    for parent, dirs, files in os.walk(runtime, followlinks=False):
        for name in dirs + files:
            p = Path(parent) / name
            if p.is_symlink():
                if p.parent == runtime / 'bin' and name in {'python','python3','python3.11','python3.12','python3.13'}:
                    continue  # A normal venv links to its base interpreter.
                try:
                    if not p.resolve().is_relative_to(runtime):
                        return True
                except RuntimeError:
                    return True
    return False


def runtime_health(runtime: Path) -> tuple[bool, str, str | None]:
    if runtime.is_symlink():
        return False, 'environment directory is a symlink (external target will remain untouched)', None
    if not runtime.is_dir() or not (runtime / 'pyvenv.cfg').is_file():
        return False, 'missing/incomplete virtual environment', None
    if unsafe_links(runtime):
        return False, 'environment contains external links', None
    marker = runtime / 'installation.json'
    if marker.exists():
        try:
            if json.loads(marker.read_text()) != {'schema':1,'root':str(ROOT)}:
                return False, 'runtime was moved/copied or has an invalid identity marker', None
        except (OSError, ValueError):
            return False, 'invalid runtime identity marker', None
    origin = old_checkout(runtime)
    metadata = list(runtime.glob('lib/python*/site-packages/transcribe_translate-*.dist-info'))
    legacy = list(runtime.glob('lib/python*/site-packages/*transcribe*.egg-link'))
    if legacy or len(metadata) > 1 or (metadata and origin is None):
        return False, 'ambiguous/legacy/broken application installation metadata', None
    if origin is not None and origin != ROOT:
        return False, 'editable metadata points to another checkout', None
    python = runtime / 'bin/python'
    try:
        result = run([str(python), '-I', '-B', '-c',
                      'import sys,pip,json; print(json.dumps({"prefix":sys.prefix,"base":sys._base_executable,"pip":pip.__file__,"version":list(sys.version_info[:2])}))'], capture=True)
        info = json.loads(result.stdout)
        if (Path(info['prefix']).resolve() != runtime or not Path(info['pip']).resolve().is_relative_to(runtime)
                or not (3,11) <= tuple(info['version']) < (3,14) or not Path(info['base']).is_file()):
            return False, 'interpreter/pip is not a supported project-local runtime', None
        for name in ('pip','pip3','transcribe-translate'):
            script = runtime / 'bin' / name
            if script.exists() and str(runtime / 'bin/python') not in script.read_text(errors='replace')[:500]:
                return False, 'entry-point shebang refers to another runtime', None
        # A broken editable import is repaired by rebuilding, not by uninstalling
        # metadata that might refer to another source tree.
        if origin is not None:
            probe = run([str(python), '-I', '-B', '-c',
                         'import transcribe_translate,json;print(json.dumps(transcribe_translate.__file__))'], capture=True)
            if Path(json.loads(probe.stdout)).resolve() != ROOT / 'src/transcribe_translate/__init__.py':
                return False, 'package import points to another checkout', None
        return True, 'healthy project runtime; dependencies may be reused', info['base']
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return False, 'broken interpreter/pip/editable import', None


def active_runtime(runtime: Path):
    """Bridge older launchers that did not hold the installation directory lock."""
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().split(b'\0')
            for value in arguments[:2]:
                path = Path(os.fsdecode(value))
                if not path.is_absolute():
                    path = (process / 'cwd').resolve() / path
                if path.parent.resolve() == runtime / 'bin':
                    raise RuntimeError(f'Project runtime is in use by PID {process.name}; wait for it to finish')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue


def output_locks(stack: ExitStack, output: Path):
    if not output.exists():
        return
    for path in (output, output / '.transcribe-translate.lock'):
        if not path.exists() and not path.is_symlink():
            continue
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if path == output:
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        stack.callback(os.close, fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Processing is using the configured output; wait before setup') from exc


def backup(runtime: Path, label: str) -> Path:
    # Only rename this literal child of the canonical root. Never rmtree a link,
    # follow its target or accept a config-derived cleanup path.
    if runtime != ROOT / '.venv' or runtime.parent != ROOT:
        raise ValueError('Refusing non-project runtime cleanup')
    target = ROOT / f'.venv.{label}-{uuid.uuid4().hex}'
    if runtime.is_symlink():
        from transcribe_translate.storage import load, save
        with cache_operation(Config(root=ROOT,source=ROOT/'Video Source',output=ROOT/'Transcribed')):
            journal = ROOT/'Work/Installation/runtime-links.json'
            links = load(journal) if journal.exists() else {}
            links[target.name] = os.readlink(runtime)
            save(journal,links)
    else:
        runtime_stamp(runtime)
    runtime.rename(target)
    return target


def python_call(python: Path, code: str, *arguments):
    return run([str(python), '-I', '-B', '-c', code, str(ROOT), *arguments])


def model_preparation(python: Path, mode: str):
    # Weight preparation does not run configured CUDA or change its configuration.
    python_call(python, '''import sys;from pathlib import Path
from transcribe_translate.config import load_config,MODELS
from transcribe_translate.asr import prepare_model
c=load_config(overrides={'translation_backend':'local','target_language':'en'})
if c.asr_model not in MODELS and not (Path(c.asr_model)/'model.bin').is_file():raise ValueError('Selected converted model directory has no model.bin')
print('ASR weights ready: '+str(prepare_model(c)))''')
    if mode == 'cuda':
        env = clean_environment()
        libraries = sorted((ROOT / '.venv/lib').glob('python*/site-packages/nvidia/*/lib'))
        env['LD_LIBRARY_PATH'] = ':'.join(map(str,libraries)) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        run([str(python), '-I', '-B', '-c', '''from transcribe_translate.config import load_config
from transcribe_translate.asr import runtime_check,cuda_smoke_check
c=load_config(overrides={'translation_backend':'local','target_language':'en','asr_device':'cuda'})
r=cuda_smoke_check(runtime_check(c))
print('CUDA smoke passed: '+str(r)+'; runtime execution only, not speech fidelity')'''], env=env)


def main():
    parser = argparse.ArgumentParser(description='Install/refresh current source; preserve configuration, data and model caches')
    parser.add_argument('--root', type=Path, required=True, help=argparse.SUPPRESS)
    parser.add_argument('--python', help='Explicit supported base interpreter')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--skip-model-downloads', action='store_true')
    parser.add_argument('--reinstall', action='store_true', help='Rebuild .venv at its final path; keep old runtime for recovery')
    args = parser.parse_args()
    previous = None
    rebuild = False
    stage = 'preflight'
    stack = ExitStack()
    runtime = ROOT / '.venv'
    try:
        if args.root.resolve() != ROOT:
            raise ValueError('Installer root does not match its source file')
        os.environ['TT_ROOT'] = str(ROOT)
        expected = source_version(ROOT)
        print(f'Canonical project: {ROOT}\nCurrent source version: {expected}', flush=True)
        stack.enter_context(installation_lock(ROOT, exclusive=True))
        active_runtime(runtime)
        cfg = load_config(overrides={'translation_backend':'local','target_language':'en'})
        output_locks(stack, cfg.output)
        mode = preflight(ROOT, cpu_only=args.cpu, require_cuda=args.cuda)
        prior_root = None
        if not runtime.is_symlink():
            try:
                prior_root = Path(json.loads((runtime / 'installation.json').read_text())['root'])
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if prior_root is None:
            prior_root = old_checkout(runtime)
        healthy, reason, existing_base = runtime_health(runtime)
        print(f'Runtime: {reason}', flush=True)
        rebuild = (args.reinstall or not healthy
                   or bool(args.python and existing_base and Path(existing_base).resolve() != Path(sys.executable).resolve()))
        base = sys.executable if args.python or not existing_base else existing_base
        # The shell has resolved --python to a supported base already.
        if rebuild:
            if runtime.exists() or runtime.is_symlink():
                previous = backup(runtime, 'backup')
                print(f'Previous runtime retained: {previous}', flush=True)
            stage = 'runtime creation'
            run([str(base), '-I', '-m', 'venv', str(runtime)])
        python = runtime / 'bin/python'
        with cache_operation(cfg):
            stage = 'pinned dependency installation'
            run([str(python), '-I', '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(ROOT/'requirements.lock')])
            if mode == 'cuda':
                run([str(python), '-I', '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(ROOT/'requirements-cuda.lock')])
            stage = 'current source refresh'
            run([str(python), '-I', '-m', 'pip', 'install', '--disable-pip-version-check', '--force-reinstall', '--no-deps', '--no-build-isolation', '-e', str(ROOT)])
            run([str(python), '-I', '-m', 'pip', 'check'])
        # Checked-hash bytecode prevents same-version/same-second source edits
        # from being hidden by copied or stale timestamp-based bytecode.
        python_call(python, '''import compileall,py_compile,sys
if not compileall.compile_dir(sys.argv[1]+'/src',force=True,quiet=1,invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH):raise SystemExit(2)''')
        # Route the installed entry point through the isolated, locked launcher.
        # A Python console-script shebang alone honors foreign PYTHONHOME/PATH.
        console = runtime / 'bin/transcribe-translate'
        console.write_text('#!/usr/bin/env bash\n# Project Python: '+str(python)+'\nset -euo pipefail\nSCRIPT="$(readlink -f -- "${BASH_SOURCE[0]}")"\nROOT="$(cd -P -- "$(dirname -- "$SCRIPT")/../.." && pwd)"\nexec "$ROOT/transcribe-translate" "$@"\n')
        console.chmod(0o755)
        stage = 'identity verification'
        python_call(python, '''from pathlib import Path;import sys,json
from transcribe_translate.installation import verify
print('Installation verified: '+json.dumps(verify(Path(sys.argv[1])),ensure_ascii=False))''')
        create_defaults(ROOT, cpu_only=args.cpu)
        if not args.skip_model_downloads:
            stage = 'weight preparation / CUDA smoke'
            model_preparation(python, mode)
        elif mode == 'cuda':
            print('CUDA kernel smoke deferred; use --cuda-smoke-check with cached weights')
        cfg = load_config(overrides={'translation_backend':'local','target_language':'en'})
        if prior_root is not None and prior_root != ROOT:
            stage = 'verified Work relocation'
            python_call(python, '''import sys;from pathlib import Path
from transcribe_translate.config import load_config
from transcribe_translate.relocation import relocate
relocate(load_config(overrides={'translation_backend':'local','target_language':'en'}),Path(sys.argv[2]))''', str(prior_root))
        (runtime / 'installation.json').write_text(json.dumps({'schema':1,'root':str(ROOT)})+'\n')
        runtime_stamp(runtime)
        for name,value in locations(cfg).items():
            print(f'{name}: {value}',flush=True)
        print(f'Dependencies installed: {mode}; configured inference: {cfg.asr_device}/{cfg.asr_compute_type}, model {cfg.asr_model}', flush=True)
        print('Validated: local installation identity/dependencies' + (' and bounded CUDA kernels' if mode=='cuda' and not args.skip_model_downloads else '; CUDA execution not validated'), flush=True)
        print(f'Setup complete. Place videos in "{cfg.source}" and run:\n{ROOT}/transcribe-translate\nConfiguration: {ROOT}/config.local.toml (existing choices preserved)\nInstalled CLI: {runtime}/bin/transcribe-translate', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # Roll back at the exact original path, where venv shebangs belong. Failed
        # environments and backups are retained; no recursive deletion occurs.
        if rebuild and previous is not None:
            if runtime.exists() or runtime.is_symlink():
                failed = backup(runtime, 'failed')
                print(f'Failed runtime retained: {failed}', file=sys.stderr)
            previous.rename(runtime)
            print('Previous runtime restored at .venv; a moved/broken runtime may still need repair.', file=sys.stderr)
        print(f'Setup failed during {stage}: {exc}\nConfiguration, media, Work and model caches were not removed. Fix the error, then rerun ./setup.sh --reinstall.', file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        stack.close()


if __name__ == '__main__':
    raise SystemExit(main())
