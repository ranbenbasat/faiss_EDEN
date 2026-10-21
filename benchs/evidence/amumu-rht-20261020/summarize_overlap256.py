"""Paired recall summary; no speed claim until native speed phase qualifies."""
import hashlib
import json
from pathlib import Path
import numpy as np
import audit_msong_quality as audit

OUT=Path(__file__).resolve().parent
path=OUT/'overlap256-recall.json'
raw=json.loads(path.read_text());assert raw['status']=='complete'
summary={'status':'recall_complete_speed_pending','raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'scope':raw['scope'],'estimators':raw['estimators'],'dll_sha256':raw['dll_sha256'],
    'operation_count':{'fullH512':512*9,'twoH256':2*256*8,'reduction_fraction':1-(2*256*8)/(512*9),
        'limitation':'Arithmetic count is NOT measured speed; excludes copy, signs, normalization, allocation and quantized search.'},
    'groups':[]}
for family in ('RaBitQ','EDEN'):
    for bits in (2,3,4):
        groups={m:sorted([r for r in raw['rows'] if (r['family'],r['bits'],r['mode'])==(family,bits,m)],key=lambda r:r['seed']) for m in ('full512','overlap420')}
        assert all(len(v)==5 for v in groups.values())
        row={'family':family,'bits':bits}
        for mode,values in groups.items():
            row[mode]={'code_bytes':values[0]['code_bytes'],
                **{f'recall{k}':float(np.mean([r['recall'][str(k)]['mean'] for r in values])) for k in (10,100)},
                'reconstruction_pooled_nmse20k':float(np.mean([r['reconstruction_full20k']['nmse'] for r in values]))}
        row['overlap_minus_full']={str(k):audit.helpers.paired_bootstrap(
            np.array([r['recall'][str(k)]['by_query'] for r in groups['overlap420']]),
            np.array([r['recall'][str(k)]['by_query'] for r in groups['full512']]),2000,20261021) for k in (10,100)}
        summary['groups'].append(row)
(OUT/'overlap256-recall-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
