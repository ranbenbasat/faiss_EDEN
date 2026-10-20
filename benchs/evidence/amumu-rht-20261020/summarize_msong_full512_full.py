"""Snapshot completed full-MSsong quality rows, never infer unfinished cases."""
import hashlib
import json
from pathlib import Path
import numpy as np
import audit_msong_quality as audit

OUT=Path(__file__).resolve().parent
RAW=OUT/'msong-full512-full-quality.json'

def main():
    content=RAW.read_bytes();raw=json.loads(content)
    rows=raw['rows']
    assert len({(r['bits'],r['seed'],r['rounds'],r['family']) for r in rows})==len(rows)
    summary={'status':raw['status'],'phase':raw['phase'],'pid':raw['pid'],
        'raw_snapshot_sha256':hashlib.sha256(content).hexdigest(),'completed_cases':len(rows),'planned_cases':90,
        'shapes':raw['shapes'],'seeds':raw['seeds'],'quality_only':True,
        'sign_protocol':raw['sign_protocol'],'centering':raw['centering'],'storage_comparison':raw['storage_comparison'],
        'primary_estimator':raw['primary_estimator'],'native_estimator_secondary':raw['native_estimator_secondary'],
        'exact_pilot_replay_cases':len(raw['pilot_replay']),'groups':[],'paired_depth_minus_one':[]}
    for family in ('RaBitQ','EDEN'):
        for bits in (2,3,4):
            for rounds in (1,2,3):
                group=sorted([r for r in rows if (r['family'],r['bits'],r['rounds'])==(family,bits,rounds)],key=lambda r:r['seed'])
                if not group:continue
                assert len({r['code_bytes'] for r in group})==1
                summary['groups'].append(dict(family=family,bits=bits,rounds=rounds,seeds=[r['seed'] for r in group],
                    code_bytes=group[0]['code_bytes'],complete_five_seeds=len(group)==5,
                    **{f'{estimator}_recall{k}':float(np.mean([r[estimator][str(k)]['mean'] for r in group])) for estimator in ('primary','native') for k in (10,100)},
                    **{f'{scope}_{key}':float(np.mean([r['reconstruction_'+scope][key] for r in group])) for scope in ('full','10k') for key in ('pooled_nmse','vnmse')}))
                if rounds==1:continue
                baseline={r['seed']:r for r in rows if (r['family'],r['bits'],r['rounds'])==(family,bits,1)}
                matched=[r for r in group if r['seed'] in baseline]
                if not matched:continue
                seeds=[r['seed'] for r in matched]
                comparison={'family':family,'bits':bits,'rounds':rounds,'paired_seeds':seeds,'complete_five_seeds':len(seeds)==5}
                for estimator in ('primary','native'):
                    for k in (10,100):
                        one=np.array([baseline[r['seed']][estimator][str(k)]['by_query'] for r in matched])
                        two=np.array([r[estimator][str(k)]['by_query'] for r in matched])
                        comparison[f'{estimator}_recall{k}']=audit.helpers.paired_bootstrap(two,one,2000,20261021)
                for key in ('pooled_nmse','vnmse'):
                    values=np.array([r['reconstruction_full'][key]-baseline[r['seed']]['reconstruction_full'][key] for r in matched])
                    comparison[f'full_{key}_delta']={'mean':float(values.mean()),'by_seed':values.tolist()}
                    if len(values)>1:
                        rng=np.random.default_rng(20261021)
                        comparison[f'full_{key}_delta']['seed_bootstrap95_conditional']=np.quantile(values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1),[.025,.975]).tolist()
                summary['paired_depth_minus_one'].append(comparison)
    target=OUT/'msong-full512-full-summary.json';target.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
