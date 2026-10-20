"""Full MSong quality only: preserve genuine full512 NumPy-sign pilot protocol.

Five paired seeds, 1/2/3 full HD rounds, frozen RaBitQ and EDEN codecs.
Common reconstruction estimator is primary; actual native FAISS search separate.
No measured latency/QPS. Dense FP64 transforms/scores here are quality oracles.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='8'
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_info,threadpool_limits
import audit_equal512_geometry as geometry
import audit_msong_quality as audit

OUT=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\Ran\Documents\Codex\TQ-AMUMU\data\msong\msong')
TARGET=OUT/'msong-full512-full-quality.json'
DETAIL=OUT/'msong-full512-full-detail'
SEEDS=list(range(202605170,202605175))
CHUNK=20000

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(8<<20):h.update(data)
    return h.hexdigest()

def fvecs(path):
    raw=np.memmap(path,dtype='<i4',mode='r');d=int(raw[0]);assert raw.size%(d+1)==0
    rows=raw.reshape(-1,d+1);assert np.all(rows[:,0]==d)
    return rows[:,1:].view('<f4')

def save(report):
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)

def merged_topk(best_d,best_i,scores,start,k=100):
    count=min(k,scores.shape[1]);ids=np.argpartition(scores,count-1,axis=1)[:,:count]
    values=np.take_along_axis(scores,ids,axis=1)
    cutoff=values.max(axis=1)
    tied=np.count_nonzero(scores==cutoff[:,None],axis=1)>1
    for row in np.flatnonzero(tied):
        lower=ids[row][values[row]<cutoff[row]]
        equal=np.flatnonzero(scores[row]==cutoff[row])[:count-len(lower)]
        ids[row]=np.r_[lower,equal]
        values[row]=scores[row,ids[row]]
    all_d=np.concatenate((best_d,values),axis=1)
    all_i=np.concatenate((best_i,ids+start),axis=1)
    # Stable lexicographic distance/ID sort resolves ties among retained candidates.
    order=np.lexsort((all_i,all_d),axis=1)[:,:k]
    return np.take_along_axis(all_d,order,axis=1),np.take_along_axis(all_i,order,axis=1)

def replay_pilot(report,center):
    old=json.loads((OUT/'msong-equal512-geometry-audit.json').read_text())
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:xb,xq=[data[k] for k in ('xb','xq')]
    exact=faiss.IndexFlatL2(420);exact.add(xb);_,truth=exact.search(xq,100)
    for seed in SEEDS[:2]:
        for rounds in (1,2,3):
            matrix,check=geometry.matrix('full512_HD',rounds,seed)
            residual=xb.astype('float64')-center
            z=residual@matrix[:420]
            leaf=faiss.index_factory(512,'RaBitQ2');leaf.train(np.ascontiguousarray(z,dtype='float32'))
            faiss.copy_array_to_vector(np.zeros(512,dtype='float32'),leaf.center)
            leaf.add(np.ascontiguousarray(z,dtype='float32'))
            rec=audit.helpers.reconstruct_leaf(faiss,leaf,0,len(xb)).astype('float64')
            norm=np.sum(residual**2,axis=1)
            qr=xq.astype('float64')-center;qz=qr@matrix[:420]
            scores=np.sum(qr**2,axis=1)[:,None]+norm[None,:]-2*qz@rec.T
            ids=np.argsort(scores,axis=1,kind='stable')[:,:100]
            recall=audit.helpers.per_query_recall(truth,ids)
            reference=next(r for r in old['rows'] if (r['centers'],r['seed'],r['rounds'],r['bits'],r['mode'])==(1,seed,rounds,2,'full512_HD'))
            np.testing.assert_array_equal(recall,reference['recall_by_query'])
            assert audit.digest(faiss.vector_to_array(leaf.codes))==reference['native_codes_sha256']
            report['pilot_replay'].append(dict(seed=seed,rounds=rounds,bits=2,recall100=float(recall.mean()),
                exact_codes_and_perquery_recall=True,**check))
            save(report);print('EXACT_PILOT_REPLAY',seed,rounds,float(recall.mean()),flush=True)

def run():
    assert platform.node().lower()=='amumu'
    assert not TARGET.exists(),'Do not overwrite results; explicit resume required'
    DETAIL.mkdir(exist_ok=True)
    base=fvecs(SOURCE/'msong_base.fvecs');query=fvecs(SOURCE/'msong_query.fvecs')
    assert base.shape==(994185,420) and query.shape==(1000,420)
    xt=np.ascontiguousarray(base[:4096]);center=xt.mean(axis=0).astype('float64')
    q=np.ascontiguousarray(query);qr=q.astype('float64')-center;qnorm=np.sum(qr**2,axis=1)
    report={'status':'running','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'phase':'provenance_and_pilot_replay','source_sha256':sha(Path(__file__)),'seeds':SEEDS,'bits':[2,3,4],
        'rounds':[1,2,3],'families':['RaBitQ','EDEN'],'shapes':{'xt':list(xt.shape),'xb':list(base.shape),'xq':list(q.shape)},
        'dimension':512,'quality_only':True,'no_timing_results':True,'threads_quality_only':8,
        'threadpools':threadpool_info(),'centering':'Single original-space centroid = NumPyfloat32 mean of first4096base rows, same as archived2seed pilot; subtract inFP64 BEFORE rotation; native leaf center forcedzero.',
        'training_overlap':'Full994185base includes4096training rows deliberately; no row-disjoint training claim. All1000 query-source rows used.',
        'sign_protocol':'NumPy default_rng(seed).choice([-1.,1.],512) eachround, matches archived full512_HD exactly; same first signs across depths. NOT native SplitMix RNG.',
        'primary_estimator':'FP64 ||q-c||² + original unquantized FP64 ||x-c||² - 2<rotated(q-c),native-decoded full multibit residual>. No query quantization, no clamping, full database candidates; matches archived diagnostic. Native codecs retain their different per-residual scales.',
        'native_estimator_secondary':'Actual frozen Index.search on identical codes; RaBitQqb=0 versus EDEN ORIGINAL-unbiased scale/default final clamp. Uses native stored FP32 factors/norms and FP32 transformed queries; not conflated with primary estimator.',
        'scale_contracts':{'RaBitQ':'Frozen multibit per-residual threshold optimization and full-code scale decoded from extended IPfactor; NOT sign-only sa_decode.',
            'EDEN':'Frozen Gaussian Lloyd-Max assignments and ORIGINAL-unbiased residual scale; oneFP32 scale and original FP32 norm.'},
        'storage_comparison':'Record actual code_size; fixed within each family/bit across1/2/3rounds. Cross-family bytes may differ; no equal-byte cross-family winner claim.',
        'centroid_sha256':audit.digest(center),'sources':{},'pilot_replay':[],'transform_checks':[],'rows':[]}
    for name in ('msong_base.fvecs','msong_query.fvecs'):
        p=SOURCE/name;report['sources'][name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    report['frozen_binaries']={name:sha(OUT/name) for name in ('runtime/faiss/_swigfaiss_avx2.pyd','../build-amumu/faiss/libfaiss_avx2.dll')}
    save(report);replay_pilot(report,center)
    report['phase']='exact_full_database_ground_truth';save(report)
    exact=faiss.IndexFlatL2(420)
    for start in range(0,len(base),CHUNK):exact.add(np.ascontiguousarray(base[start:start+CHUNK]))
    truth=np.empty((len(q),100),dtype='int64')
    for start in range(0,len(q),100):
        _,truth[start:start+100]=exact.search(q[start:start+100],100)
        print('EXACT_FULL_TRUTH',start+100,len(q),flush=True)
    del exact
    np.savez_compressed(DETAIL/'shared.npz',truth=truth,center=center)
    report['phase']='full_quality_cases';save(report)
    # Prioritize2bit allseeds/depths/families before3/4bit. No selection on results.
    for bits in (2,3,4):
        for seed in SEEDS:
            for rounds in (1,2,3):
                matrix,check=geometry.matrix('full512_HD',rounds,seed)
                report['transform_checks'].append(dict(bits=bits,seed=seed,rounds=rounds,**check))
                qz=qr@matrix[:420]
                for family in ('RaBitQ','EDEN'):
                    report['active_case']=dict(bits=bits,seed=seed,rounds=rounds,family=family,stage='encoding');save(report)
                    print('FULL_CASE_START',bits,seed,rounds,family,flush=True)
                    leaf=faiss.index_factory(512,f'{family}{bits}')
                    leaf.train(np.ascontiguousarray((xt.astype('float64')-center)@matrix[:420],dtype='float32'))
                    faiss.copy_array_to_vector(np.zeros(512,dtype='float32'),leaf.center)
                    if family=='RaBitQ':leaf.qb=0
                    # Encoding and primary scoring chunked; same frozen native codes.
                    best_d=np.empty((len(q),0));best_i=np.empty((len(q),0),dtype='int64')
                    errors=np.empty(len(base));norms=np.empty(len(base));vnmse=np.empty(len(base))
                    codedigest=hashlib.sha256()
                    for start in range(0,len(base),CHUNK):
                        end=min(start+CHUNK,len(base));residual=base[start:end].astype('float64')-center
                        z=residual@matrix[:420];leaf.add(np.ascontiguousarray(z,dtype='float32'))
                        rec=audit.helpers.reconstruct_leaf(faiss,leaf,start,end-start).astype('float64')
                        errors[start:end]=np.sum((z-rec)**2,axis=1);norms[start:end]=np.sum(residual**2,axis=1)
                        vnmse[start:end]=np.divide(errors[start:end],norms[start:end],out=np.zeros(end-start),where=norms[start:end]>0)
                        # Primary oracle nativecode decode, not a speed implementation.
                        scores=qnorm[:,None]+norms[None,start:end]-2*qz@rec.T
                        best_d,best_i=merged_topk(best_d,best_i,scores,start)
                        if (start//CHUNK)%5==0:
                            report['active_case']['base_rows_done']=end;save(report)
                            print('FULL_CASE_PROGRESS',bits,seed,rounds,family,end,len(base),flush=True)
                    allcodes=faiss.rev_swig_ptr(leaf.codes.data(),leaf.codes.size())
                    codedigest.update(memoryview(allcodes))
                    primary={str(k):audit.helpers.per_query_recall(truth[:,:k],best_i[:,:k]) for k in (10,100)}
                    report['active_case']['stage']='actual_native_search';save(report)
                    native_i=np.empty((len(q),100),dtype='int64')
                    for start in range(0,len(q),100):
                        _,native_i[start:start+100]=leaf.search(np.ascontiguousarray(qz[start:start+100],dtype='float32'),100)
                    native={str(k):audit.helpers.per_query_recall(truth[:,:k],native_i[:,:k]) for k in (10,100)}
                    filename=f'{family}-b{bits}-s{seed}-r{rounds}.npz'
                    np.savez_compressed(DETAIL/filename,primary_ids=best_i,native_ids=native_i,
                        squared_error=errors,residual_norm2=norms,vnmse=vnmse)
                    row=dict(bits=bits,seed=seed,rounds=rounds,family=family,code_bytes=int(leaf.code_size),
                        packed_code_sha256=codedigest.hexdigest(),detail_file=filename,detail_sha256=sha(DETAIL/filename),
                        primary={k:{'mean':float(v.mean()),'by_query':v.tolist()} for k,v in primary.items()},
                        native={k:{'mean':float(v.mean()),'by_query':v.tolist()} for k,v in native.items()},
                        reconstruction_full={'pooled_nmse':float(errors.sum()/norms.sum()),'vnmse':float(vnmse[norms>0].mean()),
                            'squared_error_sum':float(errors.sum()),'original_norm2_sum':float(norms.sum()),'count':len(base)},
                        reconstruction_10k={'pooled_nmse':float(errors[:10000].sum()/norms[:10000].sum()),'vnmse':float(vnmse[:10000][norms[:10000]>0].mean())})
                    report['rows'].append(row);save(report)
                    print('FULL_CASE_DONE',bits,seed,rounds,family,'primaryR100',primary['100'].mean(),
                        'nativeR100',native['100'].mean(),'NMSE',row['reconstruction_full']['pooled_nmse'],flush=True)
                    del leaf
    assert len(report['rows'])==90
    report['status']='complete';report['phase']='complete';save(report)

if __name__=='__main__':
    faiss.omp_set_num_threads(8)
    with threadpool_limits(limits=8):run()
