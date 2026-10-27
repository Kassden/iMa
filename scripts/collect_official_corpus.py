"""Collect a separate, public HKJC-only raw corpus; never modify training data."""
import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
from filelock import FileLock

from scrapy.crawler import CrawlerProcess
import scrapy
from scrapper.official_corpus import CorpusSpider, HORSE_ID, ROOT, MOVEMENT_PDF, canonical_url
from urllib.parse import parse_qsl, urlencode, urlsplit
from ima.research_storage import admitted_output,reserved_output_file,as_storage_blocked
from scrapper.research_storage_io import (
    AdmittedPickleFifoDiskQueue,
    AdmittedRFPDupeFilter,
    AdmittedScheduler,
    AdmittedSpiderState,
    BoundedRotatingFileHandler,
    admitted_native_growth,
    cleanup_owned_temporary,
    prepare_jobdir,
    validate_owned_temporary,
)


def run_segment(command):
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return child.wait()
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT)
        child.wait()
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horse-seed-file", type=Path, help="Official recovered horse URLs; expand each identity to form/workout/vet/movement routes")
    parser.add_argument("--limit", type=int, default=30, help="Response ceiling per invocation; resume with same output")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--continuous", action="store_true", help="Resume bounded segments until frontier is exhausted or access is denied")
    parser.add_argument("--retry-scope-errors", action="store_true", help="Retry prior failures only when their URLs are now inside the verified allowlist")
    parser.add_argument("--additional-seed-file", type=Path, action="append", default=[], help="Additional verified public URLs; repeat for independently discovered frontiers")
    args = parser.parse_args()
    if args.limit < 1 or args.seconds < 1:
        parser.error("limit and seconds must be positive")
    seeds = json.loads(args.seed_file.read_text())
    if not isinstance(seeds, list):
        parser.error("Seed file must be a JSON list")
    for additional_path in args.additional_seed_file:
        additional = json.loads(additional_path.read_text())
        if not isinstance(additional, list):
            parser.error("Additional seed file must be a JSON list")
        seeds.extend(additional)
    if not isinstance(seeds,list) or not seeds or any(not isinstance(u,str) or canonical_url(u) is None for u in seeds):
        parser.error("Seed file must be a nonempty JSON list of allowlisted public HKJC URLs")
    if args.horse_seed_file:
        horses = json.loads(args.horse_seed_file.read_text())
        if not isinstance(horses, list):
            parser.error("Horse seed file must be a JSON list")
        for url in reversed(horses):
            if not isinstance(url, str) or canonical_url(url) is None:
                parser.error("Horse seed outside official allowlist")
            horse = dict(parse_qsl(urlsplit(url).query)).get("horseid", "")
            if not HORSE_ID.fullmatch(horse):
                parser.error("Horse seed requires a full official identity")
            for route in ("otherhorse", "trackworkresult", "ovehorse", "movementrecords"):
                seeds.append(ROOT + route + "?" + urlencode({"horseid": horse}))
    if args.continuous:
        command = [sys.executable, "-m", "scripts.collect_official_corpus", "--seed-file", str(args.seed_file),
                   "--output", str(args.output), "--limit", str(args.limit), "--seconds", str(args.seconds)]
        if args.horse_seed_file:
            command.extend(["--horse-seed-file", str(args.horse_seed_file)])
        for additional_path in args.additional_seed_file:
            command.extend(["--additional-seed-file", str(additional_path)])
        if args.retry_scope_errors:
            command.append("--retry-scope-errors")
        while True:
            result = run_segment(command)
            if result:
                raise SystemExit(result)
            report = json.loads((args.output / "closed.json").read_text())
            if report.get("reason") not in {"closespider_pagecount", "closespider_timeout"}:
                print(json.dumps({"continuous_stopped": report}), flush=True)
                return
            time.sleep(60)
    return _collect_segment(args,seeds,parser)


def _collect_segment(args,seeds,parser):
    try:
        with admitted_output(args.output,estimated_bytes=64*1024**2,stage='official_segment',inodes=64):
            args.output.mkdir(parents=True,exist_ok=True)
            with FileLock(args.output/'collector.lock',timeout=0):
                return _collect_locked(args,seeds,parser)
    except Exception as error:
        blocked=as_storage_blocked(error,stage='official_segment',output=args.output)
        if blocked is None:raise
        print(json.dumps({'status':'blocked_storage','diagnostic':blocked.diagnostic}),flush=True)
        raise SystemExit(2) from blocked


def _publish_release_file(target,body):
    if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
        raise ValueError('Linked collector release input is not immutable')
    if target.exists():
        if target.read_bytes()!=body:raise ValueError('Immutable collector release input changed')
        return
    temporary=target.with_name('.collector-'+uuid4().hex+'.tmp')
    identity=None
    try:
        with admitted_native_growth(target,bytes_required=64*1024):
            with reserved_output_file(temporary,'xb') as stream:
                opened=os.fstat(stream.fileno())
                identity=(opened.st_dev,opened.st_ino)
                stream.write(body);stream.flush();os.fsync(stream.fileno())
            validate_owned_temporary(temporary,identity)
            try:os.link(temporary,target)
            except FileExistsError:
                if target.read_bytes()!=body:raise ValueError('Immutable collector release input changed')
            directory=os.open(target.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
    except BaseException as primary:
        cleanup_owned_temporary(temporary,identity,primary)
        raise
    else:
        cleanup_owned_temporary(temporary,identity)


def _collect_locked(args,seeds,parser):
    closed=args.output/'closed.json'
    if closed.exists():
        closed.rename(args.output/('closed-'+uuid4().hex+'.previous.json'))
    prepare_jobdir(args.output/'job',args.output)
    settings = {"ROBOTSTXT_OBEY":True,"CONCURRENT_REQUESTS":2,
        "CONCURRENT_REQUESTS_PER_DOMAIN":2,"DOWNLOAD_DELAY":2,"RANDOMIZE_DOWNLOAD_DELAY":True,
        "AUTOTHROTTLE_ENABLED":True,"AUTOTHROTTLE_START_DELAY":3,
        "AUTOTHROTTLE_MAX_DELAY":60,"AUTOTHROTTLE_TARGET_CONCURRENCY":0.5,
        "DOWNLOAD_TIMEOUT":30,"RETRY_TIMES":1,"RETRY_HTTP_CODES":[500,502,503,504,408],
        "HTTPERROR_ALLOW_ALL":True,"COOKIES_ENABLED":False,"TELNETCONSOLE_ENABLED":False,
        "REMOTE_CONTROL_ENABLED":False,
        "SCHEDULER":"scrapper.research_storage_io.AdmittedScheduler",
        "DUPEFILTER_CLASS":"scrapper.research_storage_io.AdmittedRFPDupeFilter",
        "SCHEDULER_DISK_QUEUE":"scrapper.research_storage_io.AdmittedPickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE":"scrapy.squeues.FifoMemoryQueue","DEPTH_PRIORITY":1,
        "USER_AGENT":"iMa-official-racing-research/1.0",
        "JOBDIR":str(args.output / "job"),"CLOSESPIDER_PAGECOUNT":args.limit,
        "CLOSESPIDER_TIMEOUT":args.seconds,"DEPTH_LIMIT":0,
        "LOG_FILE":None,"LOG_ENABLED":False,"LOG_LEVEL":"INFO",
        "EXTENSIONS":{"scrapy.extensions.spiderstate.SpiderState":None,
            "scrapper.research_storage_io.AdmittedSpiderState":0},
        "DOWNLOADER_MIDDLEWARES":{"scrapper.official_corpus.ScopeMiddleware":40}}
    root = Path(__file__).resolve().parents[1]
    code = {name: (root / name).read_bytes() for name in (
        "scripts/collect_official_corpus.py", "scrapper/official_corpus.py",
        "scrapper/historical/results.py", "scrapper/horse_pages.py",
        "scrapper/research_storage_io.py", "ima/research_storage.py",
        "ima/research_sqlite.py")}
    input_paths = [args.seed_file, *args.additional_seed_file]
    if args.horse_seed_file:
        input_paths.append(args.horse_seed_file)
    if len({p.name for p in input_paths}) != len(input_paths):
        parser.error("Seed input filenames must be distinct for immutable provenance")
    inputs = {p.name: p.read_bytes() for p in input_paths}
    identity = {"python": sys.version, "scrapy": scrapy.__version__,
                "code_hashes": {name: hashlib.sha256(body).hexdigest() for name, body in code.items()},
                "seed_hashes": {name: hashlib.sha256(body).hexdigest() for name, body in inputs.items()},
        "limits": {"pages": args.limit, "seconds": args.seconds}, "retry_scope_errors": args.retry_scope_errors}
    if any(urlsplit(url).path == MOVEMENT_PDF for url in seeds):
        identity["pypdf"] = version("pypdf")
    encoded = json.dumps(identity, sort_keys=True).encode()
    revision = hashlib.sha256(encoded).hexdigest()
    release = args.output / "collector-releases" / revision
    release.mkdir(parents=True, exist_ok=True)
    for name, body in code.items():
        target = release / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _publish_release_file(target,body)
    for name, body in inputs.items():
        target = release / "seeds" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _publish_release_file(target,body)
    _publish_release_file(release/'manifest.json',encoded)
    process = CrawlerProcess(settings)
    crawler=process.create_crawler(CorpusSpider)
    scrapy_logger=logging.getLogger("scrapy")
    previous_propagate=scrapy_logger.propagate
    def hold_log_writer(error):
        spider=getattr(crawler,'spider',None)
        if spider is not None and not getattr(spider,'storage_error',None):
            spider.storage_error={**error.diagnostic,'error':str(error)}
            from twisted.internet import reactor
            reactor.callLater(0,spider._storage_hold,error)
    log_handler=BoundedRotatingFileHandler(args.output/'crawl.log',on_storage_block=hold_log_writer)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    scrapy_logger.addHandler(log_handler)
    scrapy_logger.propagate=False
    try:
        process.crawl(crawler,seeds=seeds,output=str(args.output.resolve()),
                      retry_scope_errors=args.retry_scope_errors,collector_revision=revision)
        process.start()
    finally:
        scrapy_logger.removeHandler(log_handler)
        scrapy_logger.propagate=previous_propagate
        log_handler.close()
    blocked=getattr(crawler.spider,'storage_error',None)
    if blocked:
        print(json.dumps({'status':'blocked_storage','diagnostic':blocked}),flush=True)
        raise SystemExit(2)
    closed = args.output / "closed.json"
    if not closed.exists():
        raise SystemExit("No corpus readback; inspect crawl.log")
    report = json.loads(closed.read_text())
    print(json.dumps(report, indent=2), flush=True)
    if report.get("reason") in {"access_or_rate_denied", "shutdown",'blocked_storage'}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
