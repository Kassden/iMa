"""Hash-pinned cache retention holds, independent of cache quota accounting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .research_storage import StorageBlocked


def _unique_object(pairs):
    result={}
    for name,value in pairs:
        if name in result:raise ValueError('Duplicate cache-hold policy field')
        result[name]=value
    return result


def cache_holds():
    """Read configured holds without creating state or granting deletion authority.

    A qualified deployment pins both variables in its release environment.
    Legacy/unconfigured callers retain the existing cooperative lock policy.
    Missing, changed or malformed configured policy fails closed.
    """
    configured=os.environ.get('IMA_CACHE_HOLD_POLICY')
    expected=os.environ.get('IMA_CACHE_HOLD_POLICY_SHA256')
    if configured is None and expected is None:return frozenset()
    try:
        if not configured or not expected or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Both cache-hold policy path and SHA256 are required')
        source=Path(configured).absolute()
        if source.resolve()!=source:
            raise ValueError('Noncanonical cache-hold policy path')
        descriptor=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            before=os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1
                    or before.st_uid!=os.getuid() or before.st_mode&0o022
                    or before.st_size>512*1024):
                raise ValueError('Unsafe cache-hold policy file')
            with os.fdopen(descriptor,'rb') as stream:
                descriptor=-1
                encoded=stream.read(512*1024+1)
                after=os.fstat(stream.fileno())
            fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
            if any(getattr(before,key)!=getattr(after,key) for key in fields):
                raise ValueError('Cache-hold policy changed during read')
        finally:
            if descriptor>=0:os.close(descriptor)
        if hashlib.sha256(encoded).hexdigest()!=expected:
            raise ValueError('Cache-hold policy digest mismatch')
        payload=json.loads(encoded,object_pairs_hook=_unique_object)
        if (not isinstance(payload,dict) or set(payload)!={'schema_version','protected_cache_paths'}
                or type(payload['schema_version']) is not int or payload['schema_version']!=1):
            raise ValueError('Unsupported cache-hold policy schema')
        paths=payload['protected_cache_paths']
        if not isinstance(paths,list) or any(not isinstance(value,str) for value in paths):
            raise ValueError('Explicit protected cache paths required')
        held=set()
        for value in paths:
            path=Path(value)
            if (not path.is_absolute() or path.resolve()!=path or '..' in path.parts
                    or path.parent.name!='fold-preparation-cache'
                    or not re.fullmatch('[0-9a-f]{64}',path.name)):
                raise ValueError('Invalid held cache identity')
            root=path.parent.stat()
            if (not stat.S_ISDIR(root.st_mode) or root.st_uid!=os.getuid()
                    or root.st_mode&0o022 or root.st_dev!=before.st_dev):
                raise ValueError('Held cache root must be owned on the policy filesystem')
            if str(path) in held:raise ValueError('Duplicate held cache identity')
            held.add(str(path))
        return frozenset(held)
    except (OSError,ValueError,TypeError,KeyError) as error:
        raise StorageBlocked('Cache retention policy cannot be verified',
            {'stage':'cache_retention_policy','reason':str(error)}) from error


def cache_is_held(path):
    return str(Path(path).absolute()) in cache_holds()
