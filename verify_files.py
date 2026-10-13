"""Verify the shipped file manifest using Python's standard library only."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath


def verify(root):
    root = Path(root).resolve()
    manifest = root / 'CHECKSUMS.sha256'
    checked = 0
    seen = set()
    for line in manifest.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        wanted, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or name in seen or len(wanted) != 64
                or any(c not in '0123456789abcdef' for c in wanted)):
            raise ValueError('Invalid manifest entry: ' + name)
        seen.add(name)
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe file: ' + name)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != wanted:
            raise ValueError('Checksum mismatch: ' + name)
        checked += 1
    if not checked:
        raise ValueError('Empty checksum manifest')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print('Verified', verify(args.root), 'shipped files. Extra local files are not validated.')


if __name__ == '__main__':
    main()
