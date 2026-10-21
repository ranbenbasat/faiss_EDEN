"""Paired recall and qualified native throughput; preserve all failed/noisy rows."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import audit_msong_quality as audit

OUT=Path(__file__).resolve().parent
def main():
    path=OUT/'overlap2scale-quality.json';raw=json.loads(path.read_text());assert raw['status']=='complete'
    speedpath=OUT/'overlap2scale-speed.json';speed=json.loads(speedpath.read_text()) if speedpath.exists() else None
    summary={'status':('complete' if speed['status']=='complete_qualified' else 'recall_complete_speed_provisional') if speed and speed['status'].startswith('complete') else 'recall_complete_speed_incomplete',
        'scope':raw['scope'],'quality_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'estimators':raw['estimators'],'attribution':raw['attribution'],'RaBitQ_caveat':raw['RaBitQ_caveat'],
        'speed_status':speed['status'] if speed else 'not_started','recall_groups':[],'speed_groups':[],
        'speed_qualification':'Lowtargetcoreinterference, notwholemachineidle. Only paired rows satisfying entire overlapping telemetry windows are used; no fastest-sample selection. Fixedbits notmatchedrecall.',
        'speed_notes':'NativeFAISSbaseline includes original frozenHR512+nativeleaf; compactfull512 attribution includes commonoptimized full512/customSIMDscanner/compactmetadata. Do not attribute all nativebaseline speed ratio to transform/groupnormalization.'}
    rows=raw['rows']
    for family in ('EDEN','RaBitQ'):
        for bits in (2,3,4):
            bymode={mode:sorted([r for r in rows if (r['family'],r['bits'],r['mode'])==(family,bits,mode)],key=lambda r:r['seed']) for mode in raw['modes']}
            assert all(len(v)==5 for v in bymode.values())
            g={'family':family,'bits':bits,'modes':{},'two_scale_minus':{}}
            for mode,values in bymode.items():
                g['modes'][mode]={'code_bytes':values[0]['code_bytes'],
                    **{f'recall{k}':float(np.mean([r['recall'][str(k)]['mean'] for r in values])) for k in (10,100)}}
                if 'reconstruction_full20k' in values[0]:
                    g['modes'][mode]['nmse20k']=float(np.mean([r['reconstruction_full20k']['nmse'] for r in values]))
                    g['modes'][mode]['vnmse20k']=float(np.mean([r['reconstruction_full20k']['vnmse'] for r in values]))
            for baseline in ('native_full512','compact_full512','overlap_one_scale'):
                g['two_scale_minus'][baseline]={str(k):audit.helpers.paired_bootstrap(
                    np.array([r['recall'][str(k)]['by_query'] for r in bymode['overlap_two_scale']]),
                    np.array([r['recall'][str(k)]['by_query'] for r in bymode[baseline]]),2000,20261021) for k in (10,100)}
            summary['recall_groups'].append(g)
            if not speed:continue
            sr=[r for r in speed['rows'] if (r['family'],r['bits'])==(family,bits)]
            for baseline in ('native_full512','compact_full512','overlap_one_scale'):
                ratios=[];seedlogs={}
                diagnostic_seedlogs={}
                for seed in raw['seeds']:
                    perseed=[]
                    diagnostic=[]
                    for pass_id in (0,1):
                        candidates=[r for r in sr if (r['seed'],r['pass_id'],r['mode'])==(seed,pass_id,'overlap_two_scale')]
                        controls=[r for r in sr if (r['seed'],r['pass_id'],r['mode'])==(seed,pass_id,baseline)]
                        if not candidates or not controls:continue
                        new,old=candidates[0],controls[0]
                        diagnostic.append(math.log(new['qps']/old['qps']))
                        if not(new.get('qualified_low_target_core_interference') and old.get('qualified_low_target_core_interference')):continue
                        ratio=new['qps']/old['qps'];ratios.append({'seed':seed,'pass_id':pass_id,'ratio':ratio,'new_qps':new['qps'],'baseline_qps':old['qps']});perseed.append(math.log(ratio))
                    if perseed:seedlogs[seed]=float(np.mean(perseed))
                    if diagnostic:diagnostic_seedlogs[seed]=float(np.mean(diagnostic))
                result={'family':family,'bits':bits,'two_scale_versus':baseline,'qualified_pairs':len(ratios),
                    'qualified_seeds':list(seedlogs),'all5seeds_bothpasses_qualified':len(ratios)==10,
                    'pairs':ratios,'geomean_paired_ratio':math.exp(np.mean(list(seedlogs.values()))) if seedlogs else None}
                result['diagnostic_all_recorded_samples']={'geomean_paired_ratio':math.exp(np.mean(list(diagnostic_seedlogs.values()))) if diagnostic_seedlogs else None,
                    'seed_count':len(diagnostic_seedlogs),'qualification':'PROVISIONAL ONLY: includes workload-rejected rows, not verified uncontended five-seed speed. All samples retained, no fastest selection.'}
                if len(seedlogs)>1:
                    values=np.array(list(seedlogs.values()));rng=np.random.default_rng(20261021)
                    result['conditional_seed_bootstrap95_ratio']=np.exp(np.quantile(values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1),[.025,.975])).tolist()
                summary['speed_groups'].append(result)
    if speed:
        summary['speed_sha256']=hashlib.sha256(speedpath.read_bytes()).hexdigest()
        summary['speed_rows']=len(speed['rows']);summary['speed_qualified_rows']=sum(r.get('qualified_low_target_core_interference',False) for r in speed['rows'])
    (OUT/'overlap2scale-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
