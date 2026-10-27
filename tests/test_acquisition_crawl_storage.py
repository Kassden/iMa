"""Exercise the actual native reactor, scheduler and collector without network I/O."""
import json
import os
import subprocess
import sys

import pytest

CHILD = r'''"""One isolated real-reactor crawl; fixture downloader guarantees no network I/O."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
from scrapy.http import HtmlResponse
from scrapy.crawler import CrawlerProcess
from scripts import collect_official_corpus as collector
from scrapper import research_storage_io as io
from ima import research_storage as storage


class FixtureDownloader:
    def process_request(self, request, spider):
        return HtmlResponse(request.url, body=b'<html>Private storage fixture</html>',
                            encoding='utf-8', request=request,
                            headers={'Content-Type': 'text/html'})


def process(settings):
    settings = dict(settings)
    settings.update(ROBOTSTXT_OBEY=False, DOWNLOAD_DELAY=0, RANDOMIZE_DOWNLOAD_DELAY=False,
                    AUTOTHROTTLE_ENABLED=False, CLOSESPIDER_TIMEOUT=3,
                    DOWNLOADER_MIDDLEWARES={'__main__.FixtureDownloader': 1})
    return CrawlerProcess(settings)


collector.CrawlerProcess = process
deny = '--deny' in sys.argv
if deny:
    def blocked(self, request):
        raise storage.StorageBlocked('injected native scheduler admission denial')
    io.AdmittedScheduler._dqpush = blocked
if '--close-deny' in sys.argv:
    def blocked_close(self, reason):
        raise storage.StorageBlocked('injected native scheduler close denial')
    io.Scheduler.close = blocked_close
if '--state-deny' in sys.argv or '--receipt-deny' in sys.argv:
    native_publish = io.atomic_native_bytes
    selected = 'spider.state' if '--state-deny' in sys.argv else 'closed.json'
    def blocked_publish(path, body):
        if Path(path).name == selected:
            raise storage.StorageBlocked('injected native shutdown publication denial')
        return native_publish(path, body)
    io.atomic_native_bytes = blocked_publish
    if selected == 'closed.json':
        import scrapper.official_corpus as corpus
        corpus.atomic_native_bytes = blocked_publish
with tempfile.TemporaryDirectory(prefix='ima-native-crawl-probe-') as folder:
    root = Path(folder).resolve()
    original = storage.StorageAuthority
    storage.StorageAuthority = lambda path: original(path, headroom_bytes=0, headroom_inodes=0)
    os.environ['IMA_STORAGE_ROOT'] = str(root / 'authority')
    seed = root / 'seeds.json'
    urls = ['https://racing.hkjc.com/en-us/local/information/horse?HorseId=HK_2020_D001']
    seed.write_text(json.dumps(urls))
    args = SimpleNamespace(output=root / 'corpus', seed_file=seed, additional_seed_file=[],
                           horse_seed_file=None, limit=1, seconds=3, retry_scope_errors=False)
    code = 0
    try:
        collector._collect_segment(args, urls, argparse.ArgumentParser())
    except SystemExit as stopped:
        code = stopped.code
    closed = args.output / 'closed.json'
    report = {'denied': deny, 'exit': code,
              'closed': json.loads(closed.read_text()) if closed.exists() else None,
              'leases': storage.StorageAuthority(os.environ['IMA_STORAGE_ROOT']).status()['reserved_bytes']}
    print('PROBE_RESULT=' + json.dumps(report, sort_keys=True))
'''


@pytest.mark.parametrize('fault', [None, '--deny', '--close-deny', '--state-deny', '--receipt-deny'])
def test_actual_collector_reactor_and_storage_hold(fault):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('IMA_STORAGE_', 'MLFLOW_', 'IMA_MLFLOW_'))}
    result = subprocess.run([sys.executable, '-B', '-c', CHILD, *([fault] if fault else [])],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith('PROBE_RESULT='))
    report = json.loads(line.split('=', 1)[1])
    assert report['leases'] == 0
    if fault:
        assert report['exit'] == 2
        if fault == '--deny':
            assert report['closed']['reason'] == 'blocked_storage'
            assert report['closed']['storage_error']['failure_kind'] == 'infrastructure_storage'
    else:
        assert report['exit'] == 0
        assert report['closed']['pages'] == 1
        assert report['closed']['reason'] == 'closespider_pagecount'
