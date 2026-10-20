"""Pairwise-vs-no-pairwise paired quality summaries; retain all depths/budgets."""
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import audit_msong_quality as audit

OUT=Path(__file__).resolve().parent
inputs=['msong-equal512-geometry-audit.json','msong-pairwise-layer-audit.json']
summary={'status':'complete','no_speed_claims':True,'historical_replication':False,'sources':[], 'groups':[]}
for filename in inputs:
    raw=OUT/filename;data=json.loads(raw.read_text());assert data['status']=='complete'
    expected=108 if 'equal512' in filename else 117
    assert len(data['rows'])==expected
    summary['sources'].append({'file':filename,'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'rows':expected,'limitations':data.get('limits',data.get('limitations'))})
    groups={}
    for row in data['rows']:
        key=(row.get('centers',128),row['mode'],row['dimension'],row['rounds'],row['bits'])
        assert row['seed'] not in groups.setdefault(key,{})
        groups[key][row['seed']]=row
    for key,by_seed in sorted(groups.items()):
        centers,mode,dimension,rounds,bits=key
        seeds=sorted(by_seed)
        a=np.array([by_seed[s]['recall_by_query'] for s in seeds])
        row={'source':filename,'centers':centers,'mode':mode,'dimension':dimension,'rounds':rounds,'bits':bits,
            'quantizer':by_seed[seeds[0]].get('quantizer','native_per_residual'),
            'raw_payload_bytes':(dimension*bits+7)//8,'seeds':seeds,'recall100':float(a.mean()),
            'recall100_by_seed':[by_seed[s]['recall100'] for s in seeds],
            'unbiased_nmse':float(np.mean([by_seed[s]['unbiased_nmse'] for s in seeds])),
            'unbiased_vnmse':float(np.mean([by_seed[s]['unbiased_vnmse'] for s in seeds])),
            'best_scalar_vnmse':float(np.mean([by_seed[s]['best_scalar_vnmse'] for s in seeds]))}
        if mode.endswith('_pairwise') and not mode.endswith('_no_pairwise'):
            reference=groups[(centers,mode.replace('_pairwise','_no_pairwise'),dimension,rounds,bits)]
            b=np.array([reference[s]['recall_by_query'] for s in seeds])
            row['paired_pairwise_minus_no_pairwise']=audit.helpers.paired_bootstrap(a,b,1000,9876)
            row['paired_delta_by_seed']=(a-b).mean(axis=1).tolist()
        if rounds!=1:
            reference=groups[(centers,mode,dimension,1,bits)]
            b=np.array([reference[s]['recall_by_query'] for s in seeds])
            row['paired_depth_minus_1']=audit.helpers.paired_bootstrap(a,b,1000,9876)
            row['depth_delta_by_seed']=(a-b).mean(axis=1).tolist()
        summary['groups'].append(row)
path=OUT/'msong-geometry-controls-summary.json';path.write_text(json.dumps(summary,indent=2))
bundle=OUT/'msong-geometry-controls-evidence.tar.xz'
with tarfile.open(bundle,'w:xz',preset=9) as archive:
    for file in [*inputs,path.name,'audit_pairwise_layer.py','audit_equal512_geometry.py',Path(__file__).name,
                 'audit_msong_quality.py','build-frozen.json','pilot-data/msong/manifest.json']:
        archive.add(OUT/file,arcname=file)
compressed=bundle.read_bytes()
(OUT/'msong-geometry-controls-evidence.tar.xz.base64').write_text(base64.b64encode(compressed).decode('ascii'))
transfer={'status':'complete','rows':225,'no_speed_claims':True,'compressed_sha256':hashlib.sha256(compressed).hexdigest(),
    'compressed_bytes':len(compressed),'raw_files':summary['sources']}
(OUT/'msong-geometry-controls-transfer.json').write_text(json.dumps(transfer,indent=2))
print(json.dumps(transfer,indent=2),flush=True)
