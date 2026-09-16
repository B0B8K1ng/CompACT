"""Download the authenticated ModelScope Go2 snapshot and verify/extract it on NAS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import dataset_snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--use-proxy', action='store_true', help='Opt into environment proxies (direct by default)')
    args = parser.parse_args()
    if not args.use_proxy:
        for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy'):
            os.environ.pop(key, None)
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'
    args.root.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    files = api.get_dataset_files('LittleBoss/unitree-go2-data', recursive=True)
    (args.root / 'modelscope_manifest.json').write_text(json.dumps(files, indent=2))
    raw = args.root / 'raw'
    dataset_snapshot_download('LittleBoss/unitree-go2-data', local_dir=str(raw),
                              cache_dir=str(args.root / 'download_cache'), max_workers=4)
    for record in files:
        if record['Type'] != 'blob':
            continue
        path = raw / record['Path']
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if path.stat().st_size != record['Size'] or digest != record['Sha256']:
            raise ValueError(f'Snapshot changed or checksum mismatch: {path}')
        print(f'VERIFIED {path.name} bytes={path.stat().st_size} sha256={digest}', flush=True)
        if path.suffix != '.zip':
            continue
        dest = args.root / 'extracted'
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = (dest / info.filename).resolve()
                if not target.is_relative_to(dest.resolve()) or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f'Unsafe archive member: {info.filename}')
            print(f'EXTRACT {len(archive.infolist())} members, '
                  f'{sum(x.file_size for x in archive.infolist())} bytes', flush=True)
            archive.extractall(dest)
    print('DOWNLOAD_AND_EXTRACTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
