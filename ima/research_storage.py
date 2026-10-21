"""Own-user filesystem admission and identity-bound regenerable-cache reclaim.

This authority controls cooperating iMa writers, not other filesystem users.
Stable lock inodes are never replaced or removed.
"""
from __future__ import annotations

from contextlib import ExitStack, closing, contextmanager
from contextvars import ContextVar
import errno
import fcntl
import hashlib
import functools
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid


_OUTPUT_BUDGET=ContextVar('ima_output_budget',default=None)


class StorageBlocked(RuntimeError):
    """Retryable admission failure; not a scientific trial failure."""
    def __init__(self,message,diagnostic=None):
        super().__init__(message)
        self.diagnostic={**(diagnostic or {}),'failure_kind':'infrastructure_storage','retryable':True}

    def __reduce__(self):
        return type(self),(str(self),self.diagnostic)


def as_storage_blocked(error,*,stage,output):
    if isinstance(error,StorageBlocked):return error
    if not os.environ.get('IMA_STORAGE_ROOT'):return None
    pressure=isinstance(error,OSError) and error.errno in {errno.ENOSPC,errno.EDQUOT}
    full_sqlite=isinstance(error,sqlite3.Error) and (getattr(error,'sqlite_errorcode',0)&255)==sqlite3.SQLITE_FULL
    if pressure or full_sqlite:
        return StorageBlocked(str(error),{'stage':stage,'output':str(output),
            'errno':getattr(error,'errno',None),'sqlite_errorcode':getattr(error,'sqlite_errorcode',None)})
    return None


def _owned_path(path, root, *, directory=False):
    path, root = Path(path).absolute(), Path(root).absolute()
    if '..' in path.parts or root.resolve()!=root or path.resolve()!=path:
        raise ValueError('Noncanonical path or symlink escape rejected')
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Path is not strictly beneath the allowed root')
    current = root
    for part in path.relative_to(root).parts:
        if current.is_symlink():
            raise ValueError('Symlink ancestor rejected')
        current = current/part
        if current.exists() or current.is_symlink():
            s = current.lstat()
            if stat.S_ISLNK(s.st_mode) or s.st_uid != os.getuid():
                raise ValueError('Foreign owner or symlink rejected')
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError('Allowed root must be owned and not a symlink')
    if path.exists() and directory and not path.is_dir():
        raise ValueError('Expected directory')
    return path


def _process_identity(pid):
    try:
        if Path('/proc').exists():
            info = Path(f'/proc/{pid}/stat').read_text()
            start = info[info.rfind(')')+2:].split()[19]
            return Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+start
        os.kill(pid, 0)
        return 'alive:'+str(pid)
    except ProcessLookupError:
        return None
    except FileNotFoundError:
        return None
    except PermissionError:
        return 'unknown'


def _atomic_json(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(data,stream,sort_keys=True,allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name,path)
        directory = os.open(path.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)


class StorageAuthority:
    """One shared control root per filesystem, across campaigns and workers."""
    def __init__(self, root, *, headroom_bytes=None, headroom_inodes=1024,
                 cache_limit_bytes=50*1024**3, cache_soft_bytes=None,
                 cache_target_bytes=None, statvfs=None):
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in
               ((headroom_bytes,) if headroom_bytes is not None else ())+(headroom_inodes,cache_limit_bytes)):
            raise ValueError('Storage policy values must be nonnegative integers')
        self.root=Path(root).absolute()
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.resolve()!=self.root or self.root.stat().st_uid!=os.getuid() or self.root.stat().st_mode & 0o022:
            raise ValueError('Authority must be an own-user, non-shared real directory')
        self._statvfs=statvfs or os.statvfs
        s=self._statvfs(self.root)
        self.headroom_bytes=(max(20_000_000_000,int(s.f_blocks*s.f_frsize*.05))
                             if headroom_bytes is None else headroom_bytes)
        self.headroom_inodes=headroom_inodes
        self.cache_limit_bytes=cache_limit_bytes
        self.cache_soft_bytes=(cache_limit_bytes*9//10 if cache_soft_bytes is None else cache_soft_bytes)
        self.cache_target_bytes=(cache_limit_bytes*8//10 if cache_target_bytes is None else cache_target_bytes)
        if any(isinstance(x,bool) or not isinstance(x,int) for x in
               (self.cache_soft_bytes,self.cache_target_bytes)) or not (
               0<=self.cache_target_bytes<=self.cache_soft_bytes<=cache_limit_bytes):
            raise ValueError('Cache policy requires target <= soft <= hard limit')

    @contextmanager
    def _locked(self):
        lock=self.root/'authority.lock'
        fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a+b') as stream:
            if os.fstat(stream.fileno()).st_uid != os.getuid(): raise ValueError('Foreign lock')
            fcntl.flock(stream,fcntl.LOCK_EX)
            try:
                path=self.root/'state.json'
                if path.is_symlink(): raise ValueError('Symlink control state')
                state=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'reservations':{},'caches':{}}
                yield state
            finally: fcntl.flock(stream,fcntl.LOCK_UN)

    def _snapshot(self,state):
        s=self._statvfs(self.root)
        return {'available_bytes':s.f_bavail*s.f_frsize,'available_inodes':s.f_favail,
            'reserved_bytes':sum(r['bytes'] for r in state['reservations'].values()),
            'reserved_inodes':sum(r['inodes'] for r in state['reservations'].values()),
            'headroom_bytes':self.headroom_bytes,'headroom_inodes':self.headroom_inodes,
            'cache_bytes':sum(r['allocated_bytes'] for r in state['caches'].values()),
            'cache_limit_bytes':self.cache_limit_bytes,'cache_soft_bytes':self.cache_soft_bytes,
            'cache_target_bytes':self.cache_target_bytes,'reservations':state['reservations']}

    def status(self):
        with self._locked() as state: return self._snapshot(state)

    def reserve(self, *, bytes_required, inodes_required=1, stage, output, cache_bytes=0):
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (bytes_required,inodes_required,cache_bytes)):
            raise ValueError('Reservation size must be nonnegative integer')
        output=Path(output).absolute()
        parent=output
        while not parent.exists(): parent=parent.parent
        if output.resolve()!=output or parent.stat().st_uid!=os.getuid():
            raise ValueError('Output must have an own-user canonical parent')
        if parent.stat().st_dev!=self.root.stat().st_dev:
            raise ValueError('Reservation and output must share a filesystem')
        with self._locked() as state:
            self._reconcile(state)
            snap=self._snapshot(state)
            pending_cache=sum(r.get('cache_growth_bytes',0) for r in state['reservations'].values())
            if snap['cache_bytes']+pending_cache+cache_bytes>self.cache_limit_bytes:
                raise StorageBlocked('blocked_storage: pending cache growth exceeds shared limit')
            if (bytes_required+snap['reserved_bytes']+self.headroom_bytes>snap['available_bytes'] or
                inodes_required+snap['reserved_inodes']+self.headroom_inodes>snap['available_inodes']):
                raise StorageBlocked('blocked_storage: '+json.dumps(snap,sort_keys=True))
            token=uuid.uuid4().hex
            state['reservations'][token]={'bytes':bytes_required,'inodes':inodes_required,
                'pid':os.getpid(),'process_identity':_process_identity(os.getpid()),
                'created_at':time.time(),'stage':stage,'output':str(output),'cache_growth_bytes':cache_bytes}
            _atomic_json(self.root/'state.json',state)
            return token

    def extend(self,token,*,extra_bytes,extra_inodes=0,extra_cache_bytes=0):
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (extra_bytes,extra_inodes,extra_cache_bytes)):
            raise ValueError('Extension must be nonnegative integer')
        with self._locked() as state:
            record=state['reservations'][token]
            if record['pid']!=os.getpid() or record['process_identity']!=_process_identity(os.getpid()):
                raise ValueError('Reservation owner changed')
            snap=self._snapshot(state)
            pending_cache=sum(r.get('cache_growth_bytes',0) for r in state['reservations'].values())
            if snap['cache_bytes']+pending_cache+extra_cache_bytes>self.cache_limit_bytes:
                raise StorageBlocked('blocked_storage: cache growth extension denied')
            if extra_bytes+snap['reserved_bytes']+self.headroom_bytes>snap['available_bytes'] or extra_inodes+snap['reserved_inodes']+self.headroom_inodes>snap['available_inodes']:
                raise StorageBlocked('blocked_storage: growth extension denied')
            record['bytes']+=extra_bytes; record['inodes']+=extra_inodes
            record['cache_growth_bytes']=record.get('cache_growth_bytes',0)+extra_cache_bytes
            _atomic_json(self.root/'state.json',state)

    def release(self,token):
        with self._locked() as state:
            record=state['reservations'].get(token)
            if record and (record['pid']!=os.getpid() or record['process_identity']!=_process_identity(os.getpid())):
                raise ValueError('Cannot release another live worker reservation')
            state['reservations'].pop(token,None)
            _atomic_json(self.root/'state.json',state)

    def consume(self,token,*,written_bytes,created_inodes=0):
        """Already-written bytes are now accounted by statvfs, not future leases."""
        if any(type(value) is not int or value<0 for value in (written_bytes,created_inodes)):
            raise ValueError('Consumed capacity must be nonnegative integers')
        with self._locked() as state:
            record=state['reservations'][token]
            if record['pid']!=os.getpid() or record['process_identity']!=_process_identity(os.getpid()):
                raise ValueError('Reservation owner changed')
            if written_bytes>record['bytes'] or created_inodes>record['inodes']:
                raise ValueError('Write exceeded its admitted future byte capacity')
            record['bytes']-=written_bytes
            record['inodes']-=created_inodes
            _atomic_json(self.root/'state.json',state)

    @staticmethod
    def _reconcile(state):
        for token,r in list(state['reservations'].items()):
            current=_process_identity(r['pid'])
            if current is None or (current!='unknown' and current!=r['process_identity']):
                del state['reservations'][token]

    def reconcile(self):
        with self._locked() as state:
            self._reconcile(state); _atomic_json(self.root/'state.json',state)
            return self._snapshot(state)

    @contextmanager
    def admitted(self,**kwargs):
        token=self.reserve(**kwargs)
        try: yield token
        finally: self.release(token)

    def record_cache(self,root,artifact_id,allocated_bytes,*,enforce_limit=True):
        if not re.fullmatch('[0-9a-f]{64}',artifact_id): raise ValueError('Invalid cache identity')
        path=Path(root).absolute()/artifact_id
        _owned_path(path,path.parent,directory=True)
        with self._locked() as state:
            state['caches'][str(path)]={'root':str(path.parent),'artifact_id':artifact_id,
                'allocated_bytes':int(allocated_bytes),'last_access':time.time()}
            _atomic_json(self.root/'state.json',state)
            if enforce_limit and sum(r['allocated_bytes'] for r in state['caches'].values())>self.cache_limit_bytes:
                raise StorageBlocked('blocked_storage: discovered cache exceeds shared limit')

    def commit_cache(self,token,root,artifact_id,allocated_bytes):
        """Convert producer growth into committed bytes under one authority lock."""
        if not re.fullmatch('[0-9a-f]{64}',artifact_id): raise ValueError('Invalid cache identity')
        path=_owned_path(Path(root).absolute()/artifact_id,Path(root).absolute(),directory=True)
        with self._locked() as state:
            reservation=state['reservations'][token]
            if reservation['pid']!=os.getpid() or reservation['process_identity']!=_process_identity(os.getpid()):
                raise ValueError('Cache publisher reservation owner changed')
            state['caches'][str(path)]={'root':str(path.parent),'artifact_id':artifact_id,
                'allocated_bytes':int(allocated_bytes),'last_access':time.time()}
            reservation['cache_growth_bytes']=0
            _atomic_json(self.root/'state.json',state)
            if sum(r['allocated_bytes'] for r in state['caches'].values())>self.cache_limit_bytes:
                raise StorageBlocked('blocked_storage: published cache exceeds shared limit')

    def trim_caches(self,required_bytes=0):
        """Use existing per-key exclusive eviction; active shared pins win."""
        from .research_preparation import PreparationCache
        with self._locked() as state:
            self._reconcile(state)
            cache=state['caches']; total=sum(r['allocated_bytes'] for r in cache.values())
            pending=sum(r.get('cache_growth_bytes',0) for r in state['reservations'].values())
            target=(self.cache_target_bytes if total+pending+required_bytes>self.cache_soft_bytes
                    else self.cache_limit_bytes)
            evicted=[]
            for path,r in sorted(list(cache.items()),key=lambda item:item[1]['last_access']):
                if total+pending+required_bytes<=target: break
                _owned_path(path,r['root'],directory=True)
                if PreparationCache(r['root']).evict(r['artifact_id']):
                    total-=r['allocated_bytes']; del cache[path]; evicted.append(path)
            _atomic_json(self.root/'state.json',state)
            if total+pending+required_bytes>self.cache_limit_bytes:
                raise StorageBlocked('blocked_storage: pinned cache budget exhausted')
            return {'evicted':evicted,'remaining_bytes':total}


@contextmanager
def admitted_output(output, *, estimated_bytes, stage, inodes=8):
    """Legacy behavior unless a qualified release enables the shared authority."""
    control=os.environ.get('IMA_STORAGE_ROOT')
    if not control:
        yield None
        return
    authority=StorageAuthority(control)
    with authority.admitted(bytes_required=int(estimated_bytes),inodes_required=inodes,
                            stage=stage,output=output) as token:
        budget={'authority':authority,'token':token,'limit':int(estimated_bytes),
                'written':0,'consume':True,'root':Path(output).absolute(),
                'inode_remaining':inodes,'inode_unconsumed':0}
        context=_OUTPUT_BUDGET.set(budget)
        try:yield authority,token
        finally:_OUTPUT_BUDGET.reset(context)


def storage_writer(*,stage,output,estimate):
    """Admit an existing writer entrypoint before it creates any artifacts.

    Output/estimate functions consume bound arguments, including named defaults.
    A release must qualify its estimates and its individual publication paths;
    the decorator alone is not a hard filesystem quota.
    """
    def decorate(function):
        if (inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function)
                or inspect.iscoroutinefunction(function)):
            raise TypeError('Storage admission requires a synchronous writer, not deferred execution')
        signature=inspect.signature(function)
        @functools.wraps(function)
        def wrapped(*args,**kwargs):
            if not os.environ.get('IMA_STORAGE_ROOT'):
                return function(*args,**kwargs)
            bound=signature.bind(*args,**kwargs); bound.apply_defaults()
            arguments=bound.arguments
            try:
                with admitted_output(output(arguments),estimated_bytes=int(estimate(arguments)),stage=stage):
                    return function(*args,**kwargs)
            except Exception as exc:
                converted=as_storage_blocked(exc,stage=stage,output=output(arguments))
                if converted is None or converted is exc:raise
                raise converted from exc
        return wrapped
    return decorate


class ReservedFile:
    """Extend the shared lease BEFORE buffered Joblib bytes reach the disk."""
    def __init__(self,stream,budget):
        self.stream=stream; self.budget=budget
        self.unconsumed=0

    def write(self,data):
        size=memoryview(data).nbytes
        required=self.budget['written']+size+(64*1024 if self.budget.get('cache') else 0)
        if required>self.budget['limit']:
            extra=max(1024**2,required-self.budget['limit'])
            self.budget['authority'].extend(self.budget['token'],extra_bytes=extra,
                extra_cache_bytes=extra if self.budget.get('cache') else 0)
            self.budget['limit']+=extra
        result=self.stream.write(data)
        self.budget['written']+=result
        if self.budget.get('consume'):
            self.unconsumed+=result
            if self.unconsumed>=16*1024**2:self.flush()
        return result

    def flush(self):
        self.stream.flush()
        pending=self.unconsumed
        inodes=self.budget.get('inode_unconsumed',0)
        if self.budget.get('consume') and (pending or inodes):
            self.budget['authority'].consume(self.budget['token'],written_bytes=pending,created_inodes=inodes)
            self.unconsumed=0
            self.budget['inode_unconsumed']=0

    def __getattr__(self,name):
        return getattr(self.stream,name)


@contextmanager
def reserved_output_file(path,mode='wb',*,encoding='utf-8'):
    """A bounded stream for cooperating exports inside an admitted worker."""
    budget=_OUTPUT_BUDGET.get()
    if budget is None:
        with open(path,mode,**({} if 'b' in mode else {'encoding':encoding})) as stream:
            yield stream
        return
    path=Path(path).absolute()
    if path.resolve()!=path or not path.is_relative_to(budget['root']):
        raise ValueError('Export escaped its admitted output root')
    if budget['root'].is_dir():
        _owned_path(path,budget['root'])
    elif path!=budget['root'] or path.parent.stat().st_uid!=os.getuid():
        raise ValueError('Export has no admitted own-user parent')
    if mode not in {'w','wb','x','xb','a','ab'}:
        raise ValueError('Reserved exports require a write-only file mode')
    flags=os.O_WRONLY|os.O_NOFOLLOW
    if mode.startswith('a'):flags|=os.O_APPEND
    fd=None
    if not mode.startswith('x'):
        try:fd=os.open(path,flags)
        except FileNotFoundError:pass
    if fd is None:
        if budget['inode_remaining']==0:
            budget['authority'].extend(budget['token'],extra_bytes=0,extra_inodes=1)
            budget['inode_remaining']=1
        try:fd=os.open(path,flags|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:
            if mode.startswith('x'):raise
            fd=os.open(path,flags)
        else:
            budget['inode_remaining']-=1
            budget['inode_unconsumed']+=1
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_uid!=os.getuid()
                or info.st_dev!=budget['authority'].root.stat().st_dev):
            raise ValueError('Unsafe or cross-filesystem export target')
        if mode.startswith('w'):os.ftruncate(fd,0)
        stream=os.fdopen(fd,'ab' if mode.startswith('a') else 'wb')
    except BaseException:
        os.close(fd)
        raise
    with stream:
        reserved=ReservedFile(stream,budget)
        if 'b' in mode:
            try:yield reserved
            finally:reserved.flush()
        else:
            text=io.TextIOWrapper(reserved,encoding=encoding)
            try:
                yield text
                text.flush()
            finally:
                text.detach()


def payload_identity(path, allowed_root):
    """Bind every removable file's owner/inode/content; reject unusual objects."""
    path=_owned_path(path,allowed_root,directory=True)
    if path.parent.name!='fold-preparation-cache' or not re.fullmatch('[0-9a-f]{64}',path.name):
        raise ValueError('Only exact fold-cache payload directories are eligible')
    files=[]
    for p in sorted(path.iterdir()):
        s=p.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid():
            raise ValueError('Unexpected payload entry or hardlink')
        if p.name not in {'arrays.joblib','fitted.joblib','manifest.json'}:
            raise ValueError('Payload includes unknown protected material')
        digest=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''): digest.update(chunk)
        files.append({'name':p.name,'dev':s.st_dev,'ino':s.st_ino,'uid':s.st_uid,
            'size':s.st_size,'mtime_ns':s.st_mtime_ns,'allocated_bytes':s.st_blocks*512,
            'sha256':digest.hexdigest()})
    s=path.stat()
    return {'path':str(path),'root':str(Path(allowed_root).absolute()),'dev':s.st_dev,'ino':s.st_ino,
        'uid':s.st_uid,'files':files,'allocated_bytes':sum(f['allocated_bytes'] for f in files)+s.st_blocks*512}


def reclaim_payload(record, *, authority, canary_receipt):
    """Require independently qualified scope, stable key lock and exact identity."""
    if not authority or not canary_receipt.get('qualified') or not canary_receipt.get('package_id'):
        raise ValueError('Explicit authority and qualified real-package receipt required')
    path=Path(record['path']); root=Path(record['root'])
    _owned_path(path,root,directory=True)
    lock=path.parent/(path.name+'.lock')
    # Existing inode only: never manufacture an absent lock during emergency cleanup.
    fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(fd,'r+b') as stream:
        if os.fstat(stream.fileno()).st_uid!=os.getuid(): raise ValueError('Foreign lock')
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ExitStack() as pins:
            companion=path.parent/(path.name+'.pin.lock')
            if companion.exists() or companion.is_symlink():
                pin=pins.enter_context(os.fdopen(os.open(companion,os.O_RDWR|os.O_NOFOLLOW),'r+b'))
                if os.fstat(pin.fileno()).st_uid!=os.getuid(): raise ValueError('Foreign companion pin')
                fcntl.flock(pin,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                return _reclaim_locked_payload(path,root,record,authority,stream,lock)
            finally: fcntl.flock(stream,fcntl.LOCK_UN)


def _reclaim_locked_payload(path,root,record,authority,stream,lock):
    """Both existing stable key and any companion pin are held by the caller."""
    actual=payload_identity(path,root)
    if actual!=record: raise ValueError('Candidate changed since approval')
    assert_no_own_open_files(path)
    before=os.statvfs(root)
    # The invoking controller must prove quiescence and rebuild prerequisites.
    for item in actual['files']:
        (path/item['name']).unlink()
    path.rmdir()
    after=os.statvfs(root)
    return {'path':str(path),'authority':authority,'selected_allocated_bytes':actual['allocated_bytes'],
        'observed_available_byte_delta':(after.f_bavail-before.f_bavail)*after.f_frsize,
        'observed_free_byte_delta':(after.f_bfree-before.f_bfree)*after.f_frsize,
        'lock_inode_retained':os.fstat(stream.fileno()).st_ino==lock.stat().st_ino}


def assert_no_own_open_files(path):
    """Fail closed on unreadable own-process evidence or an fd/mmap under path."""
    if not Path('/proc').exists():
        raise ValueError('Live reclaim requires Linux own-process fd/mapping evidence')
    prefix=str(Path(path).absolute())+'/'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try:
            if process.stat().st_uid!=os.getuid(): continue
            maps=(process/'maps').read_text()
            if prefix in maps: raise ValueError('Cache payload is mapped by own process')
            for fd in (process/'fd').iterdir():
                try: target=os.readlink(fd)
                except FileNotFoundError: continue
                if target.startswith(prefix): raise ValueError('Cache payload is open by own process')
        except (FileNotFoundError,ProcessLookupError):
            continue
        except PermissionError as exc:
            raise ValueError('Own-process mapping evidence unavailable') from exc


def capture_sqlite(source,destination,*,allow_quiescent_capture=False,timeout=120):
    """Publish a verified healthy copy; failed staging evidence remains separate."""
    destination=Path(destination).absolute()
    if destination.resolve()!=destination or destination.exists() or destination.is_symlink():
        raise ValueError('New canonical healthy destination required')
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    _owned_path(destination,destination.parent)
    if any(Path(str(destination)+suffix).exists() or Path(str(destination)+suffix).is_symlink()
           for suffix in ('-wal','-shm','-journal')):
        raise ValueError('Healthy destination sidecars must not already exist')
    staging=Path(tempfile.mkdtemp(prefix='.database-capture-',dir=destination.parent))
    temporary=staging/destination.name
    # Keep failed stages for diagnosis; never leave a misleading final backup.
    receipt=_capture_sqlite_into(source,temporary,allow_quiescent_capture=allow_quiescent_capture,timeout=timeout)
    os.link(temporary,destination)
    temporary.unlink()
    triad=staging/(destination.name+'.original-triad')
    receipt.update(destination=str(destination),preserved_triad_path=str(triad) if triad.exists() else None)
    if not triad.exists(): staging.rmdir()
    return receipt


def _capture_sqlite_into(source,destination,*,allow_quiescent_capture=False,timeout=120):
    """Consistent online backup; quiescent fallback preserves matching DB/WAL/SHM.

    Failed originals are never repaired. Any SHM reconstruction happens only in
    the private healthy copy, after the original triad has been preserved.
    """
    source=Path(source).absolute(); destination=Path(destination).absolute()
    if (source.resolve()!=source or source.stat().st_uid!=os.getuid() or
        destination.resolve()!=destination or destination.exists() or destination.is_symlink()):
        raise ValueError('Own real source and new healthy destination required')
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    _owned_path(destination,destination.parent)
    if any(Path(str(destination)+suffix).exists() or Path(str(destination)+suffix).is_symlink()
           for suffix in ('-wal','-shm','-journal')):
        raise ValueError('Healthy destination sidecars must not already exist')
    descriptor=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    os.close(descriptor)
    started=time.monotonic(); method='sqlite_online_backup'; captured=[]
    def progress(status,remaining,total):
        if time.monotonic()-started>timeout: raise TimeoutError('Database capture deadline')
    try:
        shm_source=Path(str(source)+'-shm')
        # Do not let SQLite reconstruct damaged SOURCE evidence even during a
        # nominally read-only connection; reconstruct only the healthy copy.
        if shm_source.exists() and shm_source.stat().st_size<32768:
            raise sqlite3.OperationalError('Source SHM is truncated; preserve triad before SQLite access')
        with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=2)) as src:
            with closing(sqlite3.connect(destination)) as dst:
                src.backup(dst,pages=256,progress=progress,sleep=.02)
    except (sqlite3.Error,TimeoutError):
        if not allow_quiescent_capture: raise
        preserved=destination.parent/(destination.name+'.original-triad')
        preserved.mkdir(mode=0o700)
        # SQLite uses POSIX byte-range locks. Hold whole-file exclusive locks on
        # the matching DB/WAL/SHM while capturing a stopped writer's triad.
        # This does not depend on reading unrelated systemd/SSH process maps.
        with ExitStack() as locks:
            originals=[]
            for suffix in ('','-wal','-shm'):
                p=Path(str(source)+suffix)
                if not p.exists(): continue
                if p.is_symlink() or p.stat().st_uid!=os.getuid(): raise ValueError('Foreign database sidecar')
                stream=locks.enter_context(p.open('r+b',buffering=0))
                fcntl.lockf(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                originals.append((p,suffix))
            for p,suffix in originals:
                before=p.stat(); target=preserved/(source.name+suffix)
                shutil.copyfile(p,target)
                after=p.stat()
                if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                    raise ValueError('Source changed during quiescent capture')
                captured.append({'name':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        shutil.copyfile(preserved/source.name,destination)
        wal=preserved/(source.name+'-wal')
        if wal.exists() and wal.stat().st_size:
            shutil.copyfile(wal,Path(str(destination)+'-wal'))
        # An empty WAL has no missing committed transaction. Preserve any broken
        # SHM in original-triad, but permit SQLite to reconstruct COPY-side SHM.
        shm=Path(str(destination)+'-shm')
        if shm.exists(): shm.unlink()
        method='verified_quiescent_triad_capture_copy_side_shm_reconstruction'
    with closing(sqlite3.connect(destination)) as c:
        integrity=[r[0] for r in c.execute('PRAGMA integrity_check')]
        if integrity!=['ok']: raise ValueError('Captured database integrity check failed: '+repr(integrity))
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts={name:c.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in
                ('attempts','decisions','runs','model_versions','experiments') if name in tables}
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    return {'method':method,'source':str(source),'destination':str(destination),
        'integrity':integrity,'application_counts':counts,'preserved_triad':captured,
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started,'source_repair_performed':False,
        'source_shm_policy':'Damaged SHM captured before source SQLite access; healthy live readers may update normal SHM bookkeeping'}
