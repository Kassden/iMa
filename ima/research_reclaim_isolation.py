"""One-shot IPC between a root read-only observer and an own-user reclaimer.

No account, sudo, service or mount policy is changed. This module does not
approve selections. The caller must supply current, independently qualified
manifest, producer-quiescence and real-package/rebuild evidence.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import pwd
import secrets
import socket
import struct
import subprocess
import sys
import time

MAX_FRAME = 2*1024**2


class ReclaimRefused(RuntimeError):
    def __init__(self,message,diagnostic):
        super().__init__(message)
        self.diagnostic=diagnostic


def _send(channel, body):
    data=json.dumps(body,sort_keys=True,allow_nan=False).encode()
    if len(data)>MAX_FRAME:raise ValueError('Reclaim protocol frame exceeds bound')
    channel.sendall(struct.pack('!I',len(data))+data)


def _exact(channel, length):
    result=bytearray()
    while len(result)<length:
        part=channel.recv(length-len(result))
        if not part:raise EOFError('Reclaim peer closed before a complete frame')
        result.extend(part)
    return bytes(result)


def _receive(channel):
    length=struct.unpack('!I',_exact(channel,4))[0]
    if length>MAX_FRAME:raise ValueError('Reclaim protocol frame exceeds bound')
    body=json.loads(_exact(channel,length))
    if not isinstance(body,dict):raise ValueError('Reclaim protocol expects an object')
    return body


def _remaining_payload(record):
    """Read back partial failure without following a replaced payload directory."""
    try:
        fd=os.open(record['path'],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    except FileNotFoundError:return {'directory_exists':False,'files':{}}
    except OSError as exc:return {'state':'unreadable','error_type':type(exc).__name__}
    try:
        info=os.fstat(fd)
        if (info.st_dev,info.st_ino,info.st_uid)!=(record['dev'],record['ino'],record['uid']):
            return {'state':'directory_identity_changed'}
        files={}
        for name in ('arrays.joblib','fitted.joblib','manifest.json'):
            try:
                item=os.stat(name,dir_fd=fd,follow_symlinks=False)
                files[name]={'uid':item.st_uid,'dev':item.st_dev,'ino':item.st_ino,'size':item.st_size}
            except FileNotFoundError:files[name]={'exists':False}
        return {'directory_exists':True,'files':files}
    finally:os.close(fd)


def _status(pid):
    rows=Path(f'/proc/{pid}/status').read_text().splitlines()
    return {key:value.strip() for key,value in (row.split(':',1) for row in rows if ':' in row)}


def _unprivileged_status(pid, uid=1001, gid=1001):
    status=_status(pid)
    if [int(x) for x in status['Uid'].split()] != [uid]*4:
        raise PermissionError('Child retains an unexpected real/effective/saved/filesystem UID')
    if [int(x) for x in status['Gid'].split()] != [gid]*4:
        raise PermissionError('Child retains an unexpected real/effective/saved/filesystem GID')
    if status.get('Groups','').split():raise PermissionError('Child retains supplementary groups')
    if status.get('NoNewPrivs')!='1':raise PermissionError('Child can acquire new privileges')
    for key in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb'):
        if int(status[key],16):raise PermissionError('Child retains capabilities: '+key)
    return {key:status[key] for key in ('Uid','Gid','Groups','NoNewPrivs','CapInh','CapPrm','CapEff','CapBnd','CapAmb')}


def _validate_account():
    if not sys.platform.startswith('linux') or os.getresuid()!=(0,0,0):
        raise PermissionError('Root-started Linux observer is required')
    account=pwd.getpwuid(1001)
    if (account.pw_name,account.pw_dir,account.pw_gid)!=('imaopt','/home/imaopt',1001):
        raise ValueError('imaopt identity changed')
    return account


def _trusted_root_peer(channel):
    if struct.unpack('3i',channel.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))[1]!=0:
        raise PermissionError('Inspection peer is not root')


def _child(channel, *, native_guards=False):
    """Called only after setpriv has dropped all credentials and capabilities."""
    try:
        _trusted_root_peer(channel)
        credentials=_unprivileged_status(os.getpid())
        guard_proof=None
        if native_guards:
            import ctypes
            import resource
            signal=ctypes.c_int()
            libc=ctypes.CDLL(None,use_errno=True)
            if libc.prctl(2,ctypes.byref(signal),0,0,0)!=0 or signal.value!=9:
                raise PermissionError('Missing observer-parent death signal')
            guard_proof={'parent_death_signal':signal.value,
                'file_size':resource.getrlimit(resource.RLIMIT_FSIZE),
                'core_size':resource.getrlimit(resource.RLIMIT_CORE),
                'address_space':resource.getrlimit(resource.RLIMIT_AS),
                'cpu_seconds':resource.getrlimit(resource.RLIMIT_CPU)}
            if guard_proof!={'parent_death_signal':9,'file_size':(0,0),'core_size':(0,0),
                             'address_space':(536870912,536870912),'cpu_seconds':(120,120)}:
                raise PermissionError('Child resource guard changed')
        initial=_receive(channel)
        if initial.get('operation')!='begin':raise ValueError('Missing begin frame')
        record=initial['record']
        import types
        module=types.ModuleType('_ima_sealed_reclaim_storage')
        sys.modules[module.__name__]=module
        exec(compile(initial['storage_source'],'<sealed-research-storage>','exec'),module.__dict__)
        def inspect(path):
            if str(path)!=record['path']:raise ValueError('Inspection target changed')
            nonce=secrets.token_hex(24)
            _send(channel,{'operation':'inspect','nonce':nonce,'pid':os.getpid(),'record':record})
            reply=_receive(channel)
            if reply.get('operation')!='inspection' or reply.get('nonce')!=nonce:
                raise ValueError('Inspection response identity mismatch')
            report=reply['report']
            if (report.get('process_snapshot_complete') is not True or
                    report.get('reference_count')!=0 or report.get('error_count')!=0):
                raise ValueError('Privileged inspection did not clear the selected payload')
        if initial.get('inspect_only'):
            # Hold exactly the same stable locks and checks without mutation.
            original=module._reclaim_locked_payload
            def readback(path,root,expected,authority,stream,lock,reference_check=None,check_locks=None):
                actual=module.payload_identity(path,root)
                if actual!=expected:raise ValueError('Candidate changed since selection')
                inspect(path)
                if module.payload_identity(path,root)!=expected:
                    raise ValueError('Candidate changed during inspection')
                if check_locks:check_locks()
                return {'mode':'inspect_only','path':str(path),'unchanged':True}
            module._reclaim_locked_payload=readback
        result=module._reclaim_payload(record,authority=initial['authority'],
            canary_receipt=initial['canary_receipt'],reference_check=inspect)
        _send(channel,{'operation':'result','ok':True,'result':result,'credentials':credentials,'native_guards':guard_proof})
    except BaseException as exc:
        try:_send(channel,{'operation':'result','ok':False,'error_type':type(exc).__name__,'error':str(exc)[:2048]})
        except BaseException:pass
        return 1
    return 0


def inspect_or_reclaim_one(record, *, storage_source, worker_source, canary_receipt, authority,
                           inspect_only=True, timeout_seconds=600, observer=None):
    """Root only observes; setpriv owns every potentially destructive syscall.

    `observer` is the trusted operator's read-only inspection function. It is
    supplied by the sealed launcher; never accept executable code from a
    selection file or expose this helper through an application endpoint.
    """
    _validate_account()
    if not isinstance(worker_source,str) or not worker_source:
        raise ValueError('Hash-sealed worker source must be supplied by the operator')
    if (not isinstance(authority,str) or not authority or canary_receipt.get('qualified') is not True
            or not isinstance(canary_receipt.get('package_id'),str) or not canary_receipt['package_id']):
        raise ValueError('Current selected authority and qualified real-package receipt required')
    if type(inspect_only) is not bool:raise ValueError('Explicit boolean inspection mode required')
    if observer is None:
        from .research_reclaim_inspection import inspect_payload_references
        observer=inspect_payload_references
    if (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float))
            or not math.isfinite(timeout_seconds) or not 0<timeout_seconds<=600):
        raise ValueError('One-shot deadline must be bounded by 600 seconds')
    deadline=time.monotonic()+timeout_seconds
    # Validate the exact production target before starting any mutation child.
    from .research_reclaim_inspection import validate_candidate
    validate_candidate(record)
    setpriv=Path('/usr/bin/setpriv')
    executable=setpriv.resolve(strict=True)
    if executable.stat().st_uid!=0 or executable.stat().st_mode & 0o022:
        raise PermissionError('Untrusted setpriv executable')
    parent,child=socket.socketpair()
    parent.settimeout(timeout_seconds)
    # This source is already trusted and hash-sealed by the operator launcher.
    # Isolated Python imports only the standard library before receiving core.
    bootstrap=('import socket,sys,types,resource,os,ctypes\n'
        'libc=ctypes.CDLL(None,use_errno=True)\n'
        'assert libc.prctl(1,9,0,0,0)==0,"Cannot set parent death signal"\n'
        'assert os.getppid()=='+str(os.getpid())+',"Observer parent disappeared"\n'
        'resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n'
        'resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))\n'
        'resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))\n'
        'resource.setrlimit(resource.RLIMIT_CPU,(120,120))\n'
        'm=types.ModuleType("_ima_reclaim_ipc");sys.modules[m.__name__]=m\nexec(compile('
        +repr(worker_source)+',"<sealed-reclaim-ipc>","exec"),m.__dict__)\nraise SystemExit(m._child(socket.socket(fileno='
        +str(child.fileno())+'),native_guards=True))')
    process=None
    report=None
    try:
        process=subprocess.Popen([str(executable),'--reuid=1001','--regid=1001','--clear-groups',
            '--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs',
            '/usr/bin/python3','-I','-S','-c',bootstrap],pass_fds=(child.fileno(),),close_fds=True,
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
            cwd='/',env={'PATH':'/usr/bin:/bin','LC_ALL':'C'},start_new_session=True)
        child.close()
        _send(parent,{'operation':'begin','record':record,'storage_source':storage_source,
            'authority':authority,'canary_receipt':canary_receipt,'inspect_only':inspect_only})
        parent.settimeout(max(.001,deadline-time.monotonic()))
        message=_receive(parent)
        if message.get('operation')=='inspect':
            if message.get('pid')!=process.pid or message.get('record')!=record:
                raise ValueError('Child inspection request differs from the selected record')
            credentials=_unprivileged_status(process.pid)
            report=observer(record)
            if time.monotonic()>=deadline:raise TimeoutError('Reclaim inspection deadline exceeded')
            _send(parent,{'operation':'inspection','nonce':message['nonce'],'report':report})
            parent.settimeout(max(.001,deadline-time.monotonic()))
            message=_receive(parent)
            if message.get('ok') is True and message.get('credentials')!=credentials:
                raise PermissionError('Child credential proof changed')
        if message.get('operation')!='result' or not isinstance(message.get('ok'),bool):
            raise ValueError('Invalid child terminal result')
        _,errors=process.communicate(timeout=max(.001,min(10,deadline-time.monotonic())))
        if process.returncode!=0 or not message['ok']:
            raise ReclaimRefused('Reclaim child refused: '+json.dumps(message,sort_keys=True),
                                 {'inspection':report,'child_result':message,'remaining_payload':_remaining_payload(record)})
        if report is None:raise RuntimeError('Child did not request privileged inspection')
        return {'child_pid':process.pid,'inspection':report,**message}
    finally:
        parent.close();child.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
