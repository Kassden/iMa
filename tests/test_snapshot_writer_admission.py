import errno
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import ima.research_storage as storage
from scripts import refresh_official_snapshot as refresh_module
from scripts.audit_official_coverage import publish_report as publish_coverage_report
from scripts.build_official_dataset import build
from scripts.verify_official_snapshot import publish_report as publish_verify_report, verify
from tests.test_dataset_registry_adversarial import source_rows
from scrapper.official_corpus import PARSER_VERSION
from scrapper import research_storage_io as native_io


def base_fixture(root):
    import gzip
    import hashlib
    base=root/'base';base.mkdir()
    raw=base/'raw';raw.mkdir()
    frame=source_rows()
    lineage=[]
    for race,group in frame.groupby('race_id'):
        header='<tr>'+''.join('<th>'+x+'</th>' for x in ['Pla.','Horse No.','Horse']+['x']*8)+'</tr>'
        rows=''.join('<tr><td>'+str(row.result)+'</td><td>'+str(row.horse_no)+
            '</td><td><a href="?HorseId='+row.horse_id+'">HORSE</a></td>'+'<td>x</td>'*8+'</tr>'
            for row in group.itertuples())
        body=('<html><h1>'+race+'</h1><table>'+header+rows+'</table></html>').encode()
        digest=hashlib.sha256(body).hexdigest()
        path=raw/(digest+'.html.gz')
        with gzip.open(path,'wb') as stream:stream.write(body)
        frame.loc[group.index,'source_body_hash']=digest
        lineage.append({'body_hash':digest,'raw_path':str(path),'source_url':group.source_url.iloc[0]})
    frame.to_parquet(base/'runners.parquet',index=False)
    (base/'lineage.json').write_text(json.dumps(lineage))
    (base/'exclusions.json').write_text('[]')
    (base/'manifest.json').write_text(json.dumps({'parser_version':PARSER_VERSION,
        'source_policy':'HKJC-only; raw replay; no third-party values',
        'fixture_note':'Synthetic raw rosters for storage tests, not production race data',
        'files':{name:hashlib.sha256((base/name).read_bytes()).hexdigest()
                 for name in ('runners.parquet','lineage.json','exclusions.json')}}))
    archive,profiles=root/'archive',root/'profiles'
    archive.mkdir();profiles.mkdir()
    return base,archive,profiles,frame


def _authority(tmp_path, monkeypatch):
    authority=storage.StorageAuthority(tmp_path/'control',headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    return authority


def test_tiny_legacy_snapshot_build_and_verify(tmp_path,monkeypatch):
    base,archive,profiles,frame=base_fixture(tmp_path)
    _authority(tmp_path,monkeypatch)
    output=tmp_path/'snapshots'/'legacy'
    build(SimpleNamespace(archive=archive,profiles=profiles,corpus=[],base_snapshot=base,
        output=output,storage_format='legacy',export_csv=False))

    report=verify(output,base/'runners.parquet',base/'manifest.json')

    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest.get('storage_format') is None
    assert report['baseline_runner_keys_retained']==len(frame)
    assert report['races']==frame.race_id.nunique()
    assert (output/'runners.parquet').is_file()
    assert (output/'features.parquet').is_file()
    assert (output/'runners.csv').is_file()
    assert not output.with_name(output.name+'.building').exists()


def test_build_and_report_admission_denial_precedes_output_directories(tmp_path,monkeypatch):
    authority=_authority(tmp_path,monkeypatch)
    monkeypatch.setattr(authority,'reserve',lambda **kwargs:(_ for _ in ()).throw(
        storage.StorageBlocked('blocked_storage: fixture denial')))
    output=tmp_path/'not-created'/'snapshot'
    args=SimpleNamespace(archive=tmp_path/'archive',profiles=tmp_path/'profiles',corpus=[],
        base_snapshot=None,output=output,storage_format='legacy',export_csv=False)

    with pytest.raises(storage.StorageBlocked):
        build(args)
    refresh_output=tmp_path/'not-created-refresh'
    with pytest.raises(storage.StorageBlocked):
        refresh_module.refresh(tmp_path/'archive',tmp_path/'profiles',[],refresh_output)
    with pytest.raises(storage.StorageBlocked):
        publish_coverage_report(tmp_path/'reports'/'coverage.json',{'gaps':[]})
    with pytest.raises(storage.StorageBlocked):
        publish_verify_report(tmp_path/'reports'/'verify.json',{'verdict':'passed'})

    assert not output.parent.exists()
    assert not refresh_output.exists()
    assert not (tmp_path/'reports').exists()


@pytest.mark.parametrize('refresh', [False, True])
def test_deep_parent_inode_denial_precedes_any_directory_creation(tmp_path,monkeypatch,refresh):
    authority=_authority(tmp_path,monkeypatch)
    monkeypatch.setattr(authority,'_statvfs',lambda _:SimpleNamespace(
        f_frsize=4096,f_blocks=10**6,f_bavail=10**6,f_favail=9))
    output=tmp_path.joinpath('deep',*['level'+str(i) for i in range(12)],'snapshot')
    with pytest.raises(storage.StorageBlocked):
        if refresh:
            refresh_module.refresh(tmp_path/'archive',tmp_path/'profiles',[],output)
        else:
            build(SimpleNamespace(archive=tmp_path/'archive',profiles=tmp_path/'profiles',corpus=[],
                base_snapshot=None,output=output,storage_format='legacy',export_csv=False))
    assert not (tmp_path/'deep').exists()


def test_enospc_during_receipt_write_preserves_previous_receipt(tmp_path,monkeypatch):
    _authority(tmp_path,monkeypatch)
    output=tmp_path/'snapshots'
    output.mkdir()
    receipt_path=output/'latest_verified.json'
    previous={'snapshot':str(output/'previous'),'parser_version':'old','input_digest':'old'}
    receipt_path.write_text(json.dumps(previous))
    monkeypatch.setattr(refresh_module,'input_digest',lambda *args:'new-input')
    monkeypatch.setattr(refresh_module,'build',lambda args:None)
    monkeypatch.setattr(refresh_module,'verify',lambda *args:{'races':1,'runners':2})
    original_writer=native_io.reserved_output_file

    class PartialWriter:
        def __init__(self,stream):self.stream=stream
        def write(self,body):
            self.stream.write(body[:12])
            raise OSError(errno.ENOSPC,'no space left while publishing receipt')
        def __getattr__(self,name):return getattr(self.stream,name)

    @contextmanager
    def fail_receipt(path,mode='wb',**kwargs):
        with original_writer(path,mode,**kwargs) as stream:
            if Path(path).name.startswith('.latest_verified.json.'):
                yield PartialWriter(stream)
            else:
                yield stream

    monkeypatch.setattr(native_io,'reserved_output_file',fail_receipt)
    with pytest.raises(storage.StorageBlocked) as error:
        refresh_module.refresh(tmp_path/'archive',tmp_path/'profiles',[],output)

    assert error.value.__cause__.errno==errno.ENOSPC
    assert json.loads(receipt_path.read_text())==previous
    assert not list(output.glob('.latest_verified.json.*.tmp'))


def test_refresh_readback_is_json_and_latest_receipt_has_no_stray_temp(tmp_path,monkeypatch):
    output=tmp_path/'snapshots'
    monkeypatch.setattr(refresh_module,'input_digest',lambda *args:'input-1')

    def create_snapshot(args):
        args.output.mkdir()

    report={'races':3,'runners':24,'verdict':'passed'}
    monkeypatch.setattr(refresh_module,'build',create_snapshot)
    monkeypatch.setattr(refresh_module,'verify',lambda *args:report)

    result=refresh_module.refresh(tmp_path/'archive',tmp_path/'profiles',[],output)

    snapshot=output/Path(result['snapshot']).name
    receipt=json.loads((output/'latest_verified.json').read_text())
    assert json.loads(Path(receipt['readback']).read_text())==report
    assert receipt['snapshot']==str(snapshot.resolve())
    assert result['action']=='built_and_verified'
    assert not list(output.glob('*.tmp'))
    assert not list(snapshot.parent.glob(snapshot.name+'.*.tmp'))
