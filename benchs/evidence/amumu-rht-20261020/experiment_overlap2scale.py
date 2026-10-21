"""Two-scale overlap recall/native speed pilot with explicit scanner attribution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import faiss
import psutil
from threadpoolctl import threadpool_limits
import audit_msong_quality as audit
from native_overlap256 import Overlap
from native_overlap2scale import encode,decode,Search

OUT=Path(__file__).resolve().parent
CACHE=OUT/'overlap2scale-cache'
QUALITY=OUT/'overlap2scale-quality.json'
SPEED=OUT/'overlap2scale-speed.json'
SEEDS=list(range(12345,12350))
MODES=('native_full512','compact_full512','overlap_one_scale','overlap_two_scale')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,report):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
def data():
    with np.load(OUT/'pilot-data/msong/vectors.npz') as arrays:return [np.ascontiguousarray(arrays[k],dtype='float32') for k in ('xt','xb','xq')]
def native_index(seed,family,bits):
    path=OUT/'overlap256-native-cache'/f'full512-{family}-b{bits}-s{seed}.index'
    index=faiss.read_index(str(path));leaf=faiss.downcast_index(index.index)
    if family=='RaBitQ':leaf.qb=0
    return index,leaf,path

def quality():
    assert not QUALITY.exists()
    validation=json.loads((OUT/'overlap2scale-validation.json').read_text());assert validation['status']=='complete'
    CACHE.mkdir(exist_ok=True)
    xt,xb,xq=data();center=xt.mean(axis=0);rb=np.ascontiguousarray(xb-center);rq=np.ascontiguousarray(xq-center)
    exact=faiss.IndexFlatL2(420);exact.add(xb);_,truth=exact.search(xq,100)
    np.savez(CACHE/'shared.npz',center=center,truth=truth)
    report={'status':'running','pid':os.getpid(),'scope':'MSong20k base/200queries/fivepairedseeds, signednormalizedH256[0:256] THEN H256[164:420]; final164+256 quantization groups, twoFP32scales/globalFP32 original norm',
        'seeds':SEEDS,'bits':[2,3,4],'families':['EDEN','RaBitQ'],'modes':MODES,
        'script_sha256':sha(Path(__file__)),'dll_sha256':validation['dll_sha256'],
        'estimators':'FP32 unquantized queries throughout; native full512 FAISS HR+leaf primary comparator (RaBitQqb0/EDENoriginal), compact common SIMD scorer uses stored original norm and one final clamp',
        'attribution':'compact_full512 isolates custom format/SIMD scanner changes from transform/group scales; overlap_one_scale versus overlap_two_scale isolates group normalization at fixed420 geometry. DO NOT attribute full native baseline ratio wholly to overlap.',
        'RaBitQ_caveat':'Native pergroup164/256 RaBitQ encoder; compacted full multibit codes/twoFP32scales/globalnorm used by custom FP32-query scorer. This is NOT nativeqb4 production layout; speed uses qb0 on actual FAISS baseline.',
        'EDEN_group_policy':'Actual native EDEN164 and EDEN256 normalize independently by sqrt164/normA and sqrt256/normB, native original unbiased scale; not merely split global-scale codes.',
        'row_layout':'Compact scalar bitstream420bits*b, twoFP32scales(8B), one original whole transformed FP32norm2(4B); no pergroupnorms stored. Unused tailbitszero.',
        'centering':'Shared original-spaceFP32 mean of4096training vectors; subtract before both transforms; native leaf centers zero.',
        'speed_status':'Not yet measured','rows':[]}
    save(QUALITY,report)
    for seed in SEEDS:
        t=Overlap(seed,2);projected={'full512':t.frozen.apply(rb),'overlap':t.apply(rb)}
        queries={'full512':t.frozen.apply(rq),'overlap':t.apply(rq)}
        for family in ('EDEN','RaBitQ'):
            for bits in (2,3,4):
                index,leaf,indexpath=native_index(seed,family,bits)
                _,ids,_=t.pipeline(xq,center,index,0)
                baseline={'seed':seed,'family':family,'bits':bits,'mode':'native_full512','code_bytes':int(leaf.code_size),
                    'index_path':str(indexpath),'index_sha256':sha(indexpath),'qb_override':0 if family=='RaBitQ' else None,
                    'recall':{str(k):{'mean':float(audit.helpers.per_query_recall(truth[:,:k],ids[:,:k]).mean()),'by_query':audit.helpers.per_query_recall(truth[:,:k],ids[:,:k]).tolist()} for k in (10,100)}}
                report['rows'].append(baseline);save(QUALITY,report)
                for mode,d,groups,key in (('compact_full512',512,1,'full512'),('overlap_one_scale',420,1,'overlap'),('overlap_two_scale',420,2,'overlap')):
                    values=projected[key];q=queries[key]
                    codes,table,native_rec=encode(values,family,bits,groups)
                    rec=decode(codes,d,bits,groups,table);np.testing.assert_array_equal(rec,native_rec)
                    search=Search(codes,table,d,bits,groups);dist,ids=search.search(q)
                    # Selected scores independently reproduce actual stored scales and norm.
                    payload=(d*bits+7)//8
                    norms=np.ascontiguousarray(codes[:,payload+4*groups:]).view('<f4').ravel()
                    sample=np.maximum(np.sum(q[:3].astype('float64')**2,axis=1)[:,None]+norms[:37]-2*q[:3].astype('float64')@rec[:37].astype('float64').T,0)
                    observed=Search(codes[:37],table,d,bits,groups).scores(q[:3])
                    np.testing.assert_allclose(observed,sample,rtol=5e-6,atol=1e-2)
                    path=CACHE/f'{mode}-{family}-b{bits}-s{seed}.npz';np.savez(path,codes=codes,table=table)
                    row={'seed':seed,'family':family,'bits':bits,'mode':mode,'dimension':d,'groups':groups,
                        'code_bytes':int(codes.shape[1]),'cache_path':str(path),'cache_sha256':sha(path),
                        'codes_sha256':audit.digest(codes),
                        'recall':{str(k):{'mean':float(audit.helpers.per_query_recall(truth[:,:k],ids[:,:k]).mean()),'by_query':audit.helpers.per_query_recall(truth[:,:k],ids[:,:k]).tolist()} for k in (10,100)},
                        'reconstruction_full20k':audit.helpers.reconstruction_metrics(values,rec,np.zeros(d,dtype='float32')),
                        'reconstruction_10k':audit.helpers.reconstruction_metrics(values[:10000],rec[:10000],np.zeros(d,dtype='float32'))}
                    report['rows'].append(row);save(QUALITY,report)
                    print('TWO_SCALE_RECALL',seed,family,bits,mode,'R100',row['recall']['100']['mean'],'bytes',row['code_bytes'],flush=True)
    assert len(report['rows'])==120
    report['status']='complete';save(QUALITY,report)

def cpu(p):
    t=p.cpu_times();return t.user+t.system
def used(t):return t.user+t.system+t.interrupt+t.dpc

def speed():
    assert os.environ.get('OVERLAP2_MONITORED')=='1'
    assert not SPEED.exists()
    raw=json.loads(QUALITY.read_text());assert raw['status']=='complete'
    _,_,xq=data()
    with np.load(CACHE/'shared.npz') as shared:center=shared['center'];truth=shared['truth']
    models={};refs={}
    for r in raw['rows']:
        key=(r['seed'],r['family'],r['bits'],r['mode']);refs[key]=r
        if r['mode']=='native_full512':models[key]=native_index(r['seed'],r['family'],r['bits'])[0]
        else:
            assert sha(Path(r['cache_path']))==r['cache_sha256']
            with np.load(r['cache_path']) as f:models[key]=Search(f['codes'],f['table'],r['dimension'],r['bits'],r['groups'])
    own=psutil.Process();own.cpu_affinity([0]);parent=psutil.Process(os.getppid())
    report={'status':'waiting_for_gate','pid':os.getpid(),'affinity':own.cpu_affinity(),
        'protocol':'Low target-core interference under persistent maintenance background, NOT fully idle whole machine',
        'gate':'3consecutive5sec windows rawCPU0<=5%, globalexternal excluding worker+monitor<=0.6core; max3x12windows',
        'native_pipeline':'C++ centering+transform+alloc+actualsearch or native SIMD compactsearch; k100,batch256,3native samples,2pairedpasses; noPython/dense proxy timings',
        'attribution':raw['attribution'],'recall_source_sha256':sha(QUALITY),'gates':[],'tuning':[],'rows':[]}
    save(SPEED,report);clean=False
    for attempt in range(3):
        count=0
        for w in range(12):
            a=psutil.cpu_times(percpu=True);ca=cpu(own)+cpu(parent);start=time.perf_counter();time.sleep(5)
            b=psutil.cpu_times(percpu=True);cb=cpu(own)+cpu(parent);elapsed=time.perf_counter()-start
            ext=max(0,sum(used(y)-used(x) for x,y in zip(a,b))-(cb-ca))/elapsed
            core0=(used(b[0])-used(a[0]))/elapsed;ok=ext<=.6 and core0<=.05;count=count+1 if ok else 0
            report['gates'].append(dict(attempt=attempt,window=w,external_cpu_cores=ext,cpu0_fraction=core0,passed=ok));save(SPEED,report)
            if count==3:clean=True;break
        if clean:break
    if not clean:report['status']='quiet_gate_failed_no_speed_claim';save(SPEED,report);return
    # Tune overlap transform on unrelated Gaussian values, not evaluation seeds.
    from native_overlap256 import TT,fp
    gaussian=np.random.default_rng(73021).standard_normal((20000,420)).astype('float32')
    rng=np.random.default_rng(73022)
    for trial in range(7):
        order=list((s,n) for s in (0,1,2) for n in (32,256,20000));rng.shuffle(order)
        for strategy,n in order:
            t=Overlap(77889,strategy);out=np.empty((n,420),dtype='float32')
            seconds=TT(fp(gaussian[:n]),fp(out),n,1,fp(t.signs),strategy,int(t.frozen.this),max(1,20000//n))
            np.testing.assert_allclose(out,Overlap(77889,2).apply(gaussian[:n]),rtol=1e-5,atol=1e-5)
            report['tuning'].append(dict(strategy=strategy,n=n,trial=trial,seconds_per_vector=seconds/n))
    objectives={s:float(np.exp(np.mean([np.log(np.median([r['seconds_per_vector'] for r in report['tuning'] if r['strategy']==s and r['n']==n])) for n in (32,256,20000)]))) for s in (0,1,2)}
    chosen=min(objectives,key=objectives.get);report['overlap_strategy']=chosen;report['status']='measuring';save(SPEED,report)
    cases=list((seed,family,bits) for family in ('EDEN','RaBitQ') for bits in (2,3,4) for seed in SEEDS)
    for pass_id in (0,1):
        # Randomize within family, keeping EDEN results first (primary Gaussian question).
        ordered=[]
        for family in ('EDEN','RaBitQ'):
            subset=[c for c in cases if c[1]==family];rng.shuffle(subset);ordered+=subset
        for seed,family,bits in ordered:
            t=Overlap(seed,chosen);modes=list(MODES);rng.shuffle(modes)
            for mode in modes:
                key=(seed,family,bits,mode);model=models[key]
                if mode=='native_full512':t.pipeline(xq[:4],center,model,0)
                else:model.time_pipeline(t,xq[:4],center,repeats=1)
                start=time.perf_counter()
                if mode=='native_full512':dist,ids,samples=t.pipeline(xq,center,model,0,repeats=3)
                else:dist,ids,samples=model.time_pipeline(t,xq,center,repeats=3)
                end=time.perf_counter()
                recall=audit.helpers.per_query_recall(truth,ids)
                np.testing.assert_array_equal(recall,refs[key]['recall']['100']['by_query'])
                report['rows'].append(dict(pass_id=pass_id,seed=seed,family=family,bits=bits,mode=mode,start=start,end=end,
                    samples_seconds=samples.tolist(),qps=len(xq)/float(np.median(samples)),code_bytes=refs[key]['code_bytes']))
                save(SPEED,report);print('TWO_SCALE_SPEED',pass_id,seed,family,bits,mode,flush=True)
    assert len(report['rows'])==240
    report['status']='measurements_complete_pending_telemetry';save(SPEED,report);time.sleep(7)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('quality','speed'),required=True);args=parser.parse_args()
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):quality() if args.phase=='quality' else speed()
