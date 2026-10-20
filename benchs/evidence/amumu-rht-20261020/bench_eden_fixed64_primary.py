"""Primary k100/b256 comparison first; C++ clocks, fused full512 baseline.

V1 other batches/k retained separately. Native quality/cache reuse requires
bitwise validation on every xt/xb/xq pilot row for all five seeds.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import faiss
from threadpoolctl import threadpool_limits,threadpool_info
import audit_msong_quality as helpers
from native_block64_fixed64 import NativeTransform
from native_eden_search_fixed64 import EdenCodeSearch
OUT=Path(__file__).resolve().parent
TARGET=OUT/'native-eden-fixed64-primary-benchmark.json'
def main():
    prepared=json.loads((OUT/'native-eden-fixed64-flat-quality.json').read_text());assert prepared['status']=='complete'
    validation=json.loads((OUT/'native-fixed64-validation.json').read_text());assert validation['status']=='complete' and prepared['corrected_no_permutation']
    dll=hashlib.sha256((OUT/'experimental-kernel/eden_group_fixed64.dll').read_bytes()).hexdigest();assert dll==validation['dll_sha256']
    strategies=json.loads((OUT/'native-eden-fixed64-tuning.json').read_text())['chosen_strategies']
    transforms=json.loads((OUT/'native-block64-fixed64-tuning.json').read_text())['chosen_strategies']
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xb','xq')]
    flat=faiss.IndexFlatL2(420);flat.add(xb);_,truth=flat.search(xq,100)
    report={'status':'running','stage':'primary','version':'CORRECTED FIXED64: no permutation/cross-group mixing','corrected_no_permutation':True,
        'dll_sha256':dll,'native_quality_dll_sha256':prepared['dll_sha256'],'validation':validation,
        'clock':'C++ steady_clock, no Python/dense proxy work inside timed loop','k':100,'batch':256,
        'threadpools':threadpool_info(),'affinity_mask':os.getenv('AMUMU_EFFECTIVE_AFFINITY_MASK'),
        'policy':'all gate failures remain diagnostic_nonquiet, not validated speedups; paired seed groups contiguous, models randomized within groups',
        'strategies':strategies,'transform_strategies':transforms,'rows':[],'completed_stages':[]}
    if TARGET.exists():
        previous=json.loads(TARGET.read_text());assert previous['dll_sha256']==dll
        if previous['status']=='complete':return
        report=previous
    def save():
        temporary=TARGET.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(TARGET)
    def gate(stage,repeat):
        path=OUT/f'eden-fixed64-gate-{os.getpid()}-{stage}-{repeat}.json'
        (OUT/'eden-fixed64-gate-request.json').write_text(json.dumps({'worker_pid':os.getpid(),'stage':stage,'repeat':repeat,'gate':str(path)}))
        while not path.exists():time.sleep(1)
        return json.loads(path.read_text())
    for stage in ('primary','attribution'):
        if stage in report['completed_stages']:continue
        report['stage']=stage;save()
        for repeat in range(2):
            quiet=gate(stage,repeat)
            pairs=[(seed,bits) for seed in range(12345,12350) for bits in (2,3,4)]
            np.random.default_rng(20261120+repeat).shuffle(pairs)
            for seed,bits in pairs:
                modes=('frozen_full512','group_bf16_compact','group_bf16_samestride') if stage=='primary' else ('global512_uniform','global448','group_fp32_reference')
                rows=[r for r in prepared['rows'] if r['seed']==seed and r['bits']==bits and r['mode'] in modes and r['within_cap']]
                np.random.default_rng(20261122+seed+bits+repeat).shuffle(rows)
                for row in rows:
                    mode=row['mode'];kind=row['kind']
                    if any((r['mode'],r['seed'],r['bits'],r['repeat'])==(mode,seed,bits,repeat) for r in report['rows']):continue
                    path=Path(row['cache_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['cache_sha256']
                    with np.load(path) as cache:codes=cache['codes'];center=cache['center']
                    tm='full512' if kind==2 else 'block64';t=NativeTransform(tm,seed,1 if kind==2 else 2,False,transforms[tm])
                    if mode in ('frozen_full512','global448'):
                        leaf=faiss.read_index(row['leaf_path']);t.search(leaf,xq,100,256,0);_,pred,samples=t.search(leaf,xq,100,256,5)
                    else:
                        strategy=strategies['global512_uniform' if mode=='global512_uniform' else 'group_bf16'][str(bits)]
                        index=EdenCodeSearch(codes,bits,kind,center,strategy);index.time_pipeline(t,xq,100,256,1);_,pred,samples=index.time_pipeline(t,xq,100,256,5)
                    recall=helpers.helpers.per_query_recall(truth,pred);np.testing.assert_array_equal(recall,row['k_metrics']['100']['recall_by_query'])
                    report['rows'].append({'stage':stage,'repeat':repeat,'mode':mode,'seed':seed,'bits':bits,
                        'code_bytes_per_vector':row['code_bytes_per_vector'],'cap':row['cap'],'within_cap':True,
                        'seconds_samples':samples.tolist(),'seconds_median':float(np.median(samples)),
                        'qps':len(xq)/float(np.median(samples)),'recall':float(recall.mean()),'recall_by_query':recall.tolist(),
                        'quiet_gate':quiet,'qualification':'gate_passed_pending_telemetry_review' if quiet['available'] else 'diagnostic_nonquiet'})
                    save();print('FIXED64_NATIVE_CASE',stage,repeat,mode,bits,seed,flush=True)
        report['completed_stages'].append(stage);save();print('FIXED64_STAGE_COMPLETE',stage,flush=True)
    assert len(report['rows'])==170
    report.update(status='complete',stage=None);save()
if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):main()

