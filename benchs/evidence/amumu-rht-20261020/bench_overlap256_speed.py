"""Native speed worker. ONLY run under run_overlap256_speed.py monitor.

Predeclared low-target-core-interference protocol, not whole-machine idle.
No speed run while the multi-dataset quality queue is consuming CPUs.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import ctypes
import json
import math
import time
from pathlib import Path
import numpy as np
import faiss
import psutil
from native_overlap256 import Overlap,TT,fp

OUT=Path(__file__).resolve().parent
TARGET=OUT/'overlap256-native-speed.json'

def cpu(p):
    t=p.cpu_times();return t.user+t.system
def used(t):return t.user+t.system+t.interrupt+t.dpc
def transform_time(t,x,mode,repeats):
    output=np.empty((len(x),420 if mode==1 else 512),dtype='float32')
    result=TT(fp(x),fp(output),len(x),mode,fp(t.signs),t.strategy,int(t.frozen.this),repeats)
    if result<0:raise RuntimeError(result)
    return result/len(x),output

def main():
    assert os.environ.get('OVERLAP_MONITORED')=='1'
    assert not TARGET.exists(),'Never overwrite measurements'
    own=psutil.Process();own.cpu_affinity([0]);parent=psutil.Process(os.getppid())
    raw=json.loads((OUT/'overlap256-recall.json').read_text());assert raw['status']=='complete'
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:xq=np.ascontiguousarray(data['xq'],dtype='float32')
    center=np.load(OUT/'overlap256-native-cache/shared.npz')['center']
    models={}
    for r in raw['rows']:models[(r['seed'],r['family'],r['bits'],r['mode'])]=faiss.read_index(r['index_path'])
    report={'status':'preloaded','pid':os.getpid(),'affinity':own.cpu_affinity(),
        'protocol':'LOW TARGET-CORE INTERFERENCE under persistent maintenance background, not fully idle whole machine',
        'gate':'3consecutive5s windows CPU0raw<=5% AND externalglobal excluding worker+monitor<=0.6core; max3attempts x12windows',
        'runtime':'NativeC++ transform+centering+batchalloc+actualIndex.search, k100,batch256, five native samples, two randomized paired passes; no Python/dense proxy timing',
        'seeds':raw['seeds'],'qb':'RaBitQ4 production default and EDENoriginal-unbiased; recall uses same caches/configuration',
        'label':'Fixedbits with actual420vs512 native compact storage; not equalbyte comparison',
        'modes':{'0':'actual frozenFAISSHR512+actualnativeleaf','1':'native overlap420+actualnativeleaf','2':'equallyoptimized full512 transform+same nativeleaf attribution control'},
        'gates':[],'tuning':[],'transform_only':[],'rows':[]}
    def save():
        tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)
    save()
    clean=False
    for attempt in range(3):
        consecutive=0
        for window in range(12):
            a=psutil.cpu_times(percpu=True);pa=cpu(own)+cpu(parent);start=time.perf_counter()
            time.sleep(5)
            b=psutil.cpu_times(percpu=True);pb=cpu(own)+cpu(parent);elapsed=time.perf_counter()-start
            global_external=max(0,sum(used(y)-used(x) for x,y in zip(a,b))-(pb-pa))/elapsed
            core0=(used(b[0])-used(a[0]))/elapsed
            ok=global_external<=.6 and core0<=.05
            consecutive=consecutive+1 if ok else 0
            report['gates'].append(dict(attempt=attempt,window=window,external_cpu_cores=global_external,cpu0_fraction=core0,passed=ok));save()
            if consecutive==3:clean=True;break
        if clean:break
    if not clean:
        report['status']='quiet_gate_failed_no_measurements';save();return
    report['status']='tuning';save()
    # Candidate optimization on Gaussian inputs unrelated to evaluation seeds.
    gaussian=np.random.default_rng(73021).standard_normal((20000,420)).astype('float32')
    scores={s:[] for s in (0,1,2)};rng=np.random.default_rng(73022)
    for trial in range(9):
        order=list((s,n) for s in (0,1,2) for n in (32,256,20000));rng.shuffle(order)
        for strategy,n in order:
            t=Overlap(77889,strategy);seconds,out=transform_time(t,gaussian[:n],1,max(1,20000//n))
            np.testing.assert_allclose(out,Overlap(77889,2).apply(gaussian[:n]),rtol=1e-5,atol=1e-5)
            report['tuning'].append(dict(trial=trial,strategy=strategy,n=n,seconds_per_vector=seconds))
    for strategy in (0,1,2):
        scores[strategy]=math.exp(np.mean([math.log(np.median([r['seconds_per_vector'] for r in report['tuning'] if r['strategy']==strategy and r['n']==n])) for n in (32,256,20000)]))
    chosen=min(scores,key=scores.get);report['chosen_overlap_strategy']=chosen;save()
    report['status']='measuring';save()
    cases=list((seed,family,bits) for seed in raw['seeds'] for family in ('RaBitQ','EDEN') for bits in (2,3,4))
    for pass_id in (0,1):
        rng.shuffle(cases)
        for seed,family,bits in cases:
            t=Overlap(seed,chosen)
            full=models[(seed,family,bits,'full512')];overlap=models[(seed,family,bits,'overlap420')]
            fullleaf=faiss.downcast_index(full.index)
            order=[0,1,2];rng.shuffle(order)
            for mode in order:
                index={0:full,1:overlap,2:fullleaf}[mode]
                t.pipeline(xq[:4],center,index,mode)
                start=time.perf_counter();d,ids,samples=t.pipeline(xq,center,index,mode,repeats=5);end=time.perf_counter()
                reference=next(r for r in raw['rows'] if (r['seed'],r['family'],r['bits'],r['mode'])==(seed,family,bits,'overlap420' if mode==1 else 'full512'))
                truth=np.load(OUT/'overlap256-native-cache/shared.npz')['truth']
                import audit_msong_quality as audit
                np.testing.assert_array_equal(audit.helpers.per_query_recall(truth,ids),reference['recall']['100']['by_query'])
                report['rows'].append(dict(pass_id=pass_id,seed=seed,family=family,bits=bits,mode=mode,start=start,end=end,
                    samples_seconds=samples.tolist(),qps=len(xq)/float(np.median(samples)),code_bytes=reference['code_bytes']))
                save();print('NATIVE_SPEED_ROW',pass_id,seed,family,bits,mode,flush=True)
    # Transform-only measurements use native clocks, same pinned process.
    for seed in raw['seeds']:
        t=Overlap(seed,chosen)
        for n in (1,32,200):
            order=[0,1,2];rng.shuffle(order)
            for mode in order:
                start=time.perf_counter();samples=[]
                for trial in range(5):samples.append(transform_time(t,xq[:n],mode,max(1,20000//n))[0])
                end=time.perf_counter()
                report['transform_only'].append(dict(seed=seed,n=n,mode=mode,start=start,end=end,seconds_per_vector=samples))
    report['status']='measurements_complete_pending_telemetry';save()
    # Let monitor capture a valid full interval covering final measurements.
    time.sleep(7)

if __name__=='__main__':faiss.omp_set_num_threads(1);main()
