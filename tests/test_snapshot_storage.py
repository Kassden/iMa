"""Exercise actual snapshot publication and verifier under shared admission."""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ima.research_storage as storage
from ima.research_objects import read_snapshot_table
from scrapper.official_corpus import PARSER_VERSION
from scripts.build_official_dataset import build
from scripts.verify_official_snapshot import verify
from tests.test_dataset_registry_adversarial import source_rows


def base_fixture(root):
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


@pytest.mark.parametrize('representation',['legacy','objects-v2'])
def test_admitted_snapshot_preserves_raw_rosters_and_readback(tmp_path,monkeypatch,representation):
    base,archive,profiles,frame=base_fixture(tmp_path)
    authority=storage.StorageAuthority(tmp_path/'control',headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    output=tmp_path/'snapshots'/'successor'
    build(SimpleNamespace(archive=archive,profiles=profiles,corpus=[],base_snapshot=base,
        output=output,storage_format=representation,export_csv=False))
    report=verify(output,base/'runners.parquet',base/'manifest.json')
    assert report['baseline_runner_keys_retained']==len(frame)
    assert report['races']==frame.race_id.nunique()
    assert len(read_snapshot_table(output,'runners.parquet'))==len(frame)
    assert (output/'runners.csv').exists()==(representation=='legacy')
    assert authority.status()['reserved_bytes']==0
    assert not output.with_name(output.name+'.building').exists()
    manifest=json.loads((output/'manifest.json').read_text())
    for source in ('ima/research_objects.py','ima/research_dataset_capacity.py'):
        expected=hashlib.sha256(Path(source).read_bytes()).hexdigest()
        assert manifest['code_hashes'][source]==expected
        assert hashlib.sha256((output/'code'/Path(source).name).read_bytes()).hexdigest()==expected


def test_missing_object_store_read_never_creates_source_directories(tmp_path):
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    (snapshot/'manifest.json').write_text(json.dumps({'storage_format':'ima.snapshot.objects.v2',
        'tables':{'runners.parquet':{'object_root':str(tmp_path/'objects')}}}))
    with pytest.raises(FileNotFoundError,match='Missing immutable'):
        read_snapshot_table(snapshot,'runners.parquet')
    assert not (tmp_path/'objects').exists()
