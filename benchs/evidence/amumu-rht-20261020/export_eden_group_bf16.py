"""Budget-qualified grouped BF16 quality report; no timing claims."""
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import audit_msong_quality as helpers

OUT=Path(__file__).resolve().parent
raw=OUT/'msong-eden-group-bf16-quality.json';data=json.loads(raw.read_text())
assert data['status']=='complete' and len(data['rows'])==60
groups={}
for row in data['rows']:
    assert row['seed'] not in groups.setdefault((row['mode'],row['bits']),{})
    groups[(row['mode'],row['bits'])][row['seed']]=row
summary={'status':'complete','scope':'EDEN grouped BF16 first QUALITY pilot, speed unavailable',
    'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'codec_sha256':data['codec_sha256'],'script_sha256':data['script_sha256'],
    'seeds':data['seeds'],'shapes':data['shapes'],'speed_results_available':False,'layout':data['layout'],
    'norm_convention':data['norm_convention'],'zero_rule':data['zero_rule'],'centering':data['centering'],
    'score_rule':data['scores'],'limits':data['limits'],'groups':[]}
for (mode,bits),by_seed in sorted(groups.items()):
    seeds=sorted(by_seed);assert seeds==list(range(12345,12350))
    baseline=groups[('full512_global',bits)]
    result={'mode':mode,'bits':bits,'code_bytes_per_vector':sorted(set(r['code_bytes_per_vector'] for r in by_seed.values())),
        'budget_cap_bytes':{2:136,3:200,4:264}[bits],'within_cap':all(r['within_cap'] for r in by_seed.values()),
        'k':{},'reconstruction':{}}
    if 'group_bf16' in mode:
        result.update({'tiny_group_count_by_seed':[by_seed[s]['tiny_group_count'] for s in seeds],
            'whole_zero_count_by_seed':[by_seed[s]['whole_zero_count'] for s in seeds],
            'reserved_padding_bytes':sorted(set(r['reserved_padding_bytes'] for r in by_seed.values()))})
    for k in ('10','100'):
        a=np.array([by_seed[s]['k_metrics'][k]['recall_by_query'] for s in seeds])
        b=np.array([baseline[s]['k_metrics'][k]['recall_by_query'] for s in seeds])
        result['k'][k]={'recall':float(a.mean()),'by_seed':a.mean(axis=1).tolist(),
            'paired_delta_vs_frozen_full512':helpers.helpers.paired_bootstrap(a,b,1000,9876)}
    for population in ('reconstruction_10k','reconstruction_full20k'):
        a=[by_seed[s][population] for s in seeds];b=[baseline[s][population] for s in seeds]
        def aggregate(metrics):
            return {'pooled_nmse':sum(r['squared_error_sum'] for r in metrics)/sum(r['residual_squared_norm_sum'] for r in metrics),
                    'mean_vector_vnmse':float(np.mean([r['vnmse'] for r in metrics])),
                    'vectors_per_seed':metrics[0]['vector_count']}
        value=aggregate(a);reference=aggregate(b)
        draws=[];rng=np.random.default_rng(9876)
        for _ in range(1000):
            indices=rng.integers(0,5,5)
            av=aggregate([a[i] for i in indices]);bv=aggregate([b[i] for i in indices])
            draws.append([av['pooled_nmse']-bv['pooled_nmse'],av['mean_vector_vnmse']-bv['mean_vector_vnmse']])
        value['paired_seed_delta_vs_frozen_full512']={
            'nmse':value['pooled_nmse']-reference['pooled_nmse'],
            'vnmse':value['mean_vector_vnmse']-reference['mean_vector_vnmse'],
            'ci95_by_metric':np.quantile(np.array(draws),[.025,.975],axis=0).T.tolist(),
            'method':'paired seed bootstrap; conditional on fixed diagnostic vectors, NOT a base-vector bootstrap'}
        result['reconstruction'][population]=value
    summary['groups'].append(result)
for bits in (2,3,4):
    compact=groups[('block448_group_bf16_compact',bits)];padded=groups[('block448_group_bf16_samestride',bits)]
    for seed in data['seeds']:
        assert compact[seed]['k_metrics']==padded[seed]['k_metrics']
        assert compact[seed]['reconstruction_10k']==padded[seed]['reconstruction_10k']
summary['strict_same_stride_accuracy_identical_to_compact']=True
summary['group_fp32_reference_budget_if_run']={str(b):{'bytes':448*b//8+32,'cap':{2:136,3:200,4:264}[b],
    'within_cap':448*b//8+32<={2:136,3:200,4:264}[b]} for b in (2,3,4)}
path=OUT/'msong-eden-group-bf16-summary.json';path.write_text(json.dumps(summary,indent=2))
bundle=OUT/'msong-eden-group-bf16-evidence.tar.xz'
with tarfile.open(bundle,'w:xz',preset=9) as archive:
    for file in [raw.name,path.name,'eden_group_bf16.py','test_eden_group_bf16.py','audit_eden_group_bf16.py',Path(__file__).name,
                 'group-bf16-tests-before.log','group-bf16-tests-after.log','build-frozen.json',
                 'experimental-kernel/eden_group.cpp','experimental-kernel/block_transform.cpp',
                 'audit-eden-group-bf16-final.log','audit-eden-group-bf16.err','audit-eden-group-bf16-resumed.err']:
        archive.add(OUT/file,arcname=file)
compressed=bundle.read_bytes();(OUT/'msong-eden-group-bf16-evidence.tar.xz.base64').write_text(base64.b64encode(compressed).decode('ascii'))
transfer={'status':'complete','raw_sha256':summary['raw_sha256'],'compressed_sha256':hashlib.sha256(compressed).hexdigest(),
    'compressed_bytes':len(compressed),'raw_rows':60,'speed_results_available':False}
(OUT/'msong-eden-group-bf16-transfer.json').write_text(json.dumps(transfer,indent=2))
for row in summary['groups']:
    print(row['mode'],row['bits'],row['code_bytes_per_vector'],row['k']['100']['recall'],
          row['reconstruction']['reconstruction_10k']['pooled_nmse'],row['reconstruction']['reconstruction_10k']['mean_vector_vnmse'],flush=True)
print(json.dumps(transfer,indent=2),flush=True)
