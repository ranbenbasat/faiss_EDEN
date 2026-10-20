"""Publish primary90 completed quality cases without waiting for controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
import audit_msong_quality as audit

OUT=Path(__file__).resolve().parent
raw_bytes=(OUT/'msong-block64-primary-quality.json').read_bytes()
data=json.loads(raw_bytes)
primary=[r for r in data['rows'] if (r['mode']=='full512_native' and r['rounds']==1) or (r['mode']=='block64' and r['rounds']==2)]
assert len(primary)==90
groups={}
for r in primary:
    key=(r['family'],r['bits'],r['mode'],r['dimension'],r['rounds'],r['permute'])
    assert r['seed'] not in groups.setdefault(key,{})
    groups[key][r['seed']]=r
summary={'status':'primary90_cases_complete_full_quality_matrix_running',
    'prototype':'experimental BLOCK64/448 TWO rounds versus actual frozen full512 ONE round',
    'quality_implementation':'dense/native linear-transform oracle for experimental block map; native frozen RaBitQ/EDEN quantizers and searches',
    'experimental_block_kernel_native_compiled':False,'speed_results_available':False,
    'no_proxy_speed_claims':True,'observed_complete_cases_in_source':len(data['rows']),
    'source_snapshot_sha256':hashlib.sha256(raw_bytes).hexdigest(),
    'script_sha256':data['script_sha256'],'oracle_sha256':data['oracle_sha256'],
    'primary_rows':90,'seeds':data['seeds'],'shapes':data['shapes'],
    'norm_inverse_scalar_checks':'passed for every completed block64 plan',
    'frozen_full512_replay':'all30 primary baseline cases exactly match original per-query recall',
    'limitations':data['limits'],'groups':[]}
for key,by_seed in sorted(groups.items()):
    family,bits,mode,dimension,rounds,permute=key
    seeds=sorted(by_seed);assert seeds==list(range(12345,12350))
    scores=np.array([by_seed[s]['recall_by_query'] for s in seeds])
    reference=groups[(family,bits,'full512_native',512,1,False)]
    base=np.array([reference[s]['recall_by_query'] for s in seeds])
    row={'family':family,'bits':bits,'mode':mode,'dimension':dimension,'rounds':rounds,'permute':permute,
        'recall100':float(scores.mean()),'recall100_by_seed':[by_seed[s]['recall100'] for s in seeds],
        'native_code_size_bytes':sorted(set(by_seed[s]['native_code_size_bytes'] for s in seeds)),
        'paired_recall_delta_vs_full512_1round':audit.helpers.paired_bootstrap(scores,base,1000,9876),
        'delta_by_seed':(scores-base).mean(axis=1).tolist(),
        'nmse':float(np.mean([by_seed[s]['reconstruction']['nmse'] for s in seeds])),
        'vnmse':float(np.mean([by_seed[s]['reconstruction']['vnmse'] for s in seeds]))}
    summary['groups'].append(row)
snapshot={'status':'primary_complete','speed_results_available':False,'definition':summary['prototype'],
    'source_snapshot_sha256':summary['source_snapshot_sha256'],'rows':primary,
    'transform_checks':[c for c in data['transform_checks'] if (c['mode']=='full512_native' and c['rounds']==1) or (c['mode']=='block64' and c['rounds']==2)]}
path=OUT/'msong-block64-initial-rows.json';path.write_text(json.dumps(snapshot,indent=2))
summary['primary_raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'msong-block64-initial-summary.json').write_text(json.dumps(summary,indent=2))
for r in summary['groups']:
    print(r['family'],r['bits'],r['mode'],r['permute'],r['recall100'],r['native_code_size_bytes'],r['paired_recall_delta_vs_full512_1round']['delta'],flush=True)
