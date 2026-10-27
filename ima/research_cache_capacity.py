"""Physical-headroom admission layered on the existing shared cache policy."""
from pathlib import Path

from .research_storage import StorageBlocked, _atomic_json, _owned_path


def trim_for_capacity(authority, *, required_bytes=0, required_inodes=8,
                      cache_growth_bytes=None):
    """Trim quota first, then inactive LRU payloads until physical admission fits.

    This does not reserve the resulting capacity. The caller must still obtain
    its normal atomic reservation; another producer may win that capacity.
    Only registered regenerable payloads are eligible, through the cache's
    existing nonblocking key/pin locks. Models and raw inputs are not candidates.
    """
    if cache_growth_bytes is None:
        cache_growth_bytes = required_bytes
    for value in (required_bytes, required_inodes, cache_growth_bytes):
        if type(value) is not int or value < 0:
            raise ValueError('Nonnegative integer capacity demand required')
    quota = authority.trim_caches(cache_growth_bytes)
    from .research_preparation import PreparationCache

    with authority._locked() as state:
        authority._reconcile(state)

        def physical():
            snapshot = authority._snapshot(state)
            return snapshot, {
                'bytes': max(0, snapshot['headroom_bytes'] + snapshot['reserved_bytes']
                             + required_bytes - snapshot['available_bytes']),
                'inodes': max(0, snapshot['headroom_inodes'] + snapshot['reserved_inodes']
                              + required_inodes - snapshot['available_inodes']),
            }

        before, deficit = physical()
        evicted = list(quota['evicted'])
        cache = state['caches']
        for name, record in sorted(list(cache.items()), key=lambda item: item[1]['last_access']):
            if not any(deficit.values()):
                break
            if not Path(record['root']).is_dir():
                raise StorageBlocked('blocked_storage: registered cache root unavailable')
            path = _owned_path(name, record['root'], directory=True)
            if PreparationCache(record['root']).evict(record['artifact_id']):
                del cache[str(path)]
                evicted.append(str(path))
                _, deficit = physical()
        _atomic_json(authority.root / 'state.json', state)
        after, deficit = physical()
        if any(deficit.values()):
            raise StorageBlocked('blocked_storage: no eligible cache can restore physical headroom',
                                 {'stage': 'cache_physical_headroom', 'deficit': deficit,
                                  'snapshot': after, 'evicted': evicted})
        return {'evicted': evicted, 'remaining_bytes': after['cache_bytes'],
                'physical_before': before, 'physical_after': after,
                'required_bytes': required_bytes, 'required_inodes': required_inodes}
