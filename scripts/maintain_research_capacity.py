"""One-shot operating headroom maintenance using registered cache retirement."""
import argparse
import json
import os
from pathlib import Path
import uuid

from ima.research_cache_capacity import trim_for_capacity
from ima.research_cache_policy import cache_holds
from ima.research_storage import (StorageAuthority, StorageBlocked, admitted_output,
                                  reserved_output_file)


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def persist(receipts, record):
    payload = (json.dumps(record, sort_keys=True) + '\n').encode()
    target = receipts / (uuid.uuid4().hex + '.json')
    temporary = target.with_suffix('.tmp')
    with admitted_output(receipts, estimated_bytes=len(payload) + 4096,
                         stage='capacity_retirement_receipt', inodes=8):
        receipts.mkdir(mode=0o700, parents=True, exist_ok=True)
        with reserved_output_file(temporary, 'xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
        # Link publishes atomically without replacing an existing immutable receipt.
        os.link(temporary, target)
        temporary.unlink()
        fd = os.open(receipts, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return str(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True, type=Path)
    parser.add_argument('--required-bytes', required=True, type=int)
    parser.add_argument('--receipts', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.required_bytes < 0:
        parser.error('--required-bytes must be nonnegative')
    result = {'event': 'completion', 'status': 'blocked', 'evicted': []}
    before = {}
    try:
        if not all(os.environ.get(key) for key in
                   ('IMA_CACHE_HOLD_POLICY', 'IMA_CACHE_HOLD_POLICY_SHA256')):
            raise ValueError('Pinned cache hold policy and SHA256 are required')
        cache_holds()
        authority_path = args.authority.absolute()
        if (authority_path.resolve() != authority_path or not authority_path.is_dir()
                or not (authority_path / 'state.json').is_file()):
            raise ValueError('Existing canonical authority with state.json required')
        os.environ['IMA_STORAGE_ROOT'] = str(authority_path)
        authority = StorageAuthority(authority_path)
        with authority._locked() as state:
            before = {path: row['allocated_bytes'] for path, row in state['caches'].items()}
        emit({'event': 'intent', 'authority': str(authority_path),
              'required_bytes': args.required_bytes, 'cache_growth_bytes': 0,
              'registered_caches': len(before)})
        try:
            outcome = trim_for_capacity(authority, required_bytes=args.required_bytes,
                                        cache_growth_bytes=0)
            result.update(status='ok', evicted=outcome['evicted'])
        except StorageBlocked as error:
            result.update(reason=str(error), diagnostic=error.diagnostic,
                          evicted=error.diagnostic.get('evicted', []))
        except Exception as error:
            result.update(status='audit_incomplete', error_type=type(error).__name__,
                          retirement_outcome='unknown',
                          reason='Cache API failed; retirement audit unavailable')
        result['retired'] = [{'path': path, 'allocated_bytes': before.get(path)}
                             for path in result['evicted']]
        if result['evicted']:
            try:
                result['receipt'] = persist(args.receipts.absolute(), result)
            except Exception as error:
                result.update(status='audit_incomplete', receipt_error=type(error).__name__,
                              reason='Retirements occurred; durable receipt publication failed')
    except Exception as error:
        # Avoid echoing arbitrary API exception text that could contain secrets.
        result.update(status='blocked', reason='Preflight or cache API failed',
                      error_type=type(error).__name__)
    emit(result)
    return 0 if result['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
