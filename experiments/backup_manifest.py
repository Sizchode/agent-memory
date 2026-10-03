"""Inventory backup contents without following symlinks or changing sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(base, names):
    base = Path(base).resolve(strict=True)
    if not names or len(names) != len(set(names)):
        raise ValueError('Nonempty, distinct backup roots are required')
    entries = {}

    def visit(path):
        key = path.relative_to(base).as_posix()
        before = path.lstat()
        if path.is_symlink():
            entries[key] = dict(type='symlink', target=os.readlink(path))
        elif path.is_dir():
            entries[key] = dict(type='directory')
            for child in sorted(path.iterdir()):
                visit(child)
        elif path.is_file():
            entries[key] = dict(type='file', bytes=before.st_size, sha256=file_hash(path))
        else:
            raise ValueError(f'Unsupported filesystem entry: {path}')
        after = path.lstat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError(f'Source changed during inventory: {path}')

    for name in sorted(names):
        if not name or name in ('.', '..') or Path(name).name != name:
            raise ValueError('Each backup root must be an immediate child name')
        path = base / name
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f'Backup root must be a real directory: {path}')
        visit(path)
    return dict(roots=sorted(names), entries=entries,
                regular_file_bytes=sum(e['bytes'] for e in entries.values() if e['type'] == 'file'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--include', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.base, args.include)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('backup_manifest_complete', len(result['entries']), result['regular_file_bytes'], file_hash(args.output))
