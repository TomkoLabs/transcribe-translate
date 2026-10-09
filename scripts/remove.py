"""Uninstall using base Python and verified local ownership, never a broken venv."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from transcribe_translate.config import load_config
from transcribe_translate.installation import installation_lock
from transcribe_translate.managed import MARKER, no_links, remove, owner
from transcribe_translate.storage import load, StateError, file_hash
from install import active_runtime, output_locks


def main():
    parser = argparse.ArgumentParser(description='Remove owned runtime/install caches; preserve configuration, recordings, transcripts and recovery')
    parser.add_argument('--root',type=Path,required=True,help=argparse.SUPPRESS)
    parser.add_argument('--python',help='Supported base interpreter')
    parser.add_argument('--remove-models',action='store_true',help='Also remove verified owned model files inside this installation; external storage retained')
    args = parser.parse_args()
    failures=[]
    try:
        if args.root.resolve()!=ROOT:
            raise StateError('Uninstall root does not match its source')
        os.environ['TT_ROOT']=str(ROOT)
        with ExitStack() as stack:
            stack.enter_context(installation_lock(ROOT,exclusive=True))
            active_runtime(ROOT/'.venv')
            cfg=load_config(overrides={'translation_backend':'local','target_language':'en'})
            output_locks(stack,cfg.output)
            print(f'Canonical uninstall project: {ROOT}')
            def clean(path,kind):
                try:
                    if remove(path,ROOT,kind):print(f'Removed: {path}')
                except (OSError,RuntimeError) as exc:
                    failures.append(str(exc));print(f'Retained: {path}: {exc}',file=sys.stderr)
            clean(ROOT/'.venv','runtime-v1')
            for pattern in ['.venv.backup-*','.venv.failed-*']:
                for path in sorted(ROOT.glob(pattern)):
                    if path.is_symlink():continue # Only journaled installer-created links are removed below.
                    clean(path,'runtime-v1')
            cache=cfg.installation_cache
            journal=cache/'runtime-links.json'
            if journal.exists():
                no_links(journal,ROOT)
                cached_owner=owner(cache,'installation-cache-v1')
                if cached_owner['files'].get('runtime-links.json')!={'sha256':file_hash(journal)}:
                    raise StateError('Edited runtime-link journal retained')
                for name,target in load(journal).items():
                    if not re.fullmatch(r'\.venv\.(backup|failed)-[0-9a-f]{32}',name):
                        raise StateError('Unsafe runtime-link ownership')
                    path=ROOT/name
                    if path.is_symlink() and os.readlink(path)==target:
                        path.unlink();print(f'Removed owned local link: {path}; target retained')
                    elif path.exists() or path.is_symlink():
                        failures.append(f'Changed link retained: {path}')
            clean(cache,'installation-cache-v1')
            if args.remove_models:
                for folder in dict.fromkeys([ROOT/'Models',cfg.models_root]):
                    if folder.is_symlink() or not folder.is_relative_to(ROOT):
                        print(f'Retained linked/external/shared model location: {folder}')
                        continue
                    if folder!=ROOT:no_links(folder,ROOT)
                    if folder.exists():
                        for path in sorted(folder.iterdir()):
                            if path.is_dir() and not path.is_symlink() and (path/MARKER).is_file():
                                kind=load(path/MARKER).get('layout')
                                if kind in {'pinned-model-v1','named-model-cache-v1'}:clean(path,kind)
            for name in ['Models','config.local.toml','config.api.local.toml','Video Source','Transcribed','Work']:
                print(f'Retained data/configuration location: {ROOT/name}')
            for path in [cfg.source,cfg.output,cfg.work_root,cfg.models_root]:
                print(f'Retained configured data/model parent: {path}')
            if cfg.shared_hub:print(f'Retained explicit shared Hub: {cfg.shared_hub}')
            print('Source/launchers and other credential files retained; rerun ./setup.sh to reinstall.')
        return 2 if failures else 0
    except (OSError,RuntimeError,ValueError) as exc:
        print(f'Uninstall refused: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
