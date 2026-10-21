"""Full-data quality extension of MSong protocol. No latency or QPS claims.

Native codec/actual native search plus separate FP64 reconstruction estimator.
Finite resumable90-case matrix per dataset; never overwrite completed cases.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='8'
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_info,threadpool_limits
import audit_msong_quality as audit
from quality_msong_full512_full import fvecs,sha,merged_topk

OUT=Path(__file__).resolve().parent
DATA=Path(r'C:\Users\Ran\Documents\Codex\TQ-AMUMU\data')
SEEDS=list(range(202605170,202605175))
CHUNK=20000
QBATCH=200

def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def paths(dataset):
    root=OUT/'source-data/sift' if dataset=='sift' else DATA/dataset/dataset
    return {'base':root/f'{dataset}_base.fvecs','query':root/f'{dataset}_query.fvecs',
            'train':root/'sift_learn.fvecs' if dataset=='sift' else root/f'{dataset}_base.fvecs'}

def matrix(d,seed,rounds):
    rng=np.random.default_rng(seed);mat=np.eye(d,dtype='float64');signhash=[]
    for _ in range(rounds):
        signs=rng.choice(np.array([-1.,1.]),d)
        signhash.append(audit.digest(signs));mat=audit.fwht(mat*signs)
    error=float(np.max(np.abs(mat@mat.T-np.eye(d))))
    assert error<1e-12
    return mat,{'dimension':d,'seed':seed,'rounds':rounds,'sign_sha256_by_round':signhash,
                'matrix_sha256':audit.digest(mat),'orthogonality_max_abs_error':error,
                'full_transform_every_round':True,'permutation':None}

def run(args):
    assert platform.node().split('.')[0].lower()=='amumu'
    files=paths(args.dataset)
    for path in files.values():assert path.is_file(),path
    base=fvecs(files['base']);queries=fvecs(files['query']);train=fvecs(files['train'])
    expected={'word2vec':(1000000,300,1000),'gist':(1000000,960,1000),'sift':(1000000,128,10000)}[args.dataset]
    assert base.shape==(expected[0],expected[1]) and queries.shape==(expected[2],expected[1])
    input_d=base.shape[1];d=1<<(input_d-1).bit_length()
    xt=np.ascontiguousarray(train[:4096]);center=xt.mean(axis=0).astype('float64')
    q=np.ascontiguousarray(queries);qr=q.astype('float64')-center;qnorm=np.sum(qr**2,axis=1)
    target=OUT/f'{args.dataset}-full-rht-quality.json'
    detail=OUT/f'{args.dataset}-full-rht-detail';detail.mkdir(exist_ok=True)
    def save(report):
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(target)
    report={'status':'running','pid':os.getpid(),'started_utc':stamp(),'phase':'provenance',
        'dataset':args.dataset,'script_sha256':sha(Path(__file__)),'seeds':SEEDS,'bits':[2,3,4],
        'rounds':[1,2,3],'families':['RaBitQ','EDEN'],'planned_cases':90,
        'shapes':{'xt':list(xt.shape),'xb':list(base.shape),'xq':list(q.shape)},
        'input_dimension':input_d,'padded_dimension':d,'quality_only':True,'no_timing_results':True,
        'threads_quality_only':8,'threadpools':threadpool_info(),
        'centering':'Shared original-space NumPyfloat32 mean of4096training rows, residual subtraction inFP64 before full RHT; native codec center forcedzero.',
        'training_source':('Separate SIFT learn file first4096rows' if args.dataset=='sift' else 'First4096base rows; full database includes training rows deliberately, same as completed MSong protocol'),
        'sign_protocol':'Same NumPy default_rng seeds as completed MSong; fresh d signs per full normalized H_d round. First signs shared across depths. NOT native SplitMix RNG.',
        'primary_estimator':'FP64 original ||q-c||²+||x-c||²-2<full-RHT(q-c),native-full-decoded residual>. No query quantization/clamp, exhaustive database; quality oracle, not runtime proxy.',
        'native_estimator_secondary':'Actual frozen native Index.search on identical codes and FP32 transformed queries; RaBitQqb0 and native EDEN default unbiased scale/final clamp. Separate from FP64 reconstruction estimator.',
        'scale_contracts':{'RaBitQ':'Frozen multibit native per-residual optimized threshold/full extended IPfactor reconstruction, not sign-only decoder',
            'EDEN':'Frozen Gaussian Lloyd-Max and ORIGINAL-unbiased per-vector FP32 scale with original FP32 residual norm'},
        'storage':'Native code_size fixed across rounds within each family/bit. No equal-byte cross-family claim; implicit sequential flat IDs. Native 128/512/1024 quantizer dimensions match padded full RHT.',
        'center_sha256':audit.digest(center),'sources':{},'transform_checks':[],'rows':[],
        'limitations':['One source-defined query population per dataset, five paired sign seeds.',
            'Separate conditional seed/query uncertainty; no claim of dataset-general population significance.',
            'Quality-only8threads. Runtime requires separately optimized native preprocessing/scanner and predeclared measurement protocol.']}
    if target.exists():
        prior=json.loads(target.read_text())
        assert prior['script_sha256']==report['script_sha256'],'Source changed; do not mix checkpoints'
        assert prior['center_sha256']==report['center_sha256']
        report=prior
        for name,path in files.items():
            assert path.stat().st_size==report['sources'][name]['bytes']
            assert path.stat().st_mtime_ns==report['sources'][name]['mtime_ns'],'Source file changed'
        report.update(status='running',pid=os.getpid(),resumed_utc=stamp(),requested_bits_this_invocation=args.bits)
    else:
        hashcache={}
        for name,path in files.items():
            key=str(path)
            if key not in hashcache:hashcache[key]=sha(path)
            report['sources'][name]={'path':key,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'sha256':hashcache[key]}
        report['frozen_binaries']={name:sha(OUT/name) for name in ('runtime/faiss/_swigfaiss_avx2.pyd','../build-amumu/faiss/libfaiss_avx2.dll')}
        report['dependency_hashes']={name:sha(OUT/name) for name in ('audit_msong_quality.py','quality_msong_full512_full.py')}
    save(report)
    shared=detail/'shared.npz'
    if shared.exists():
        with np.load(shared) as cached:
            np.testing.assert_array_equal(center,cached['center']);truth=cached['truth']
    else:
        report['phase']='exact_full_ground_truth';save(report)
        exact=faiss.IndexFlatL2(input_d)
        for start in range(0,len(base),CHUNK):exact.add(np.ascontiguousarray(base[start:start+CHUNK]))
        truth=np.empty((len(q),100),dtype='int64')
        for start in range(0,len(q),100):
            end=min(start+100,len(q));_,truth[start:end]=exact.search(q[start:end],100)
            report['truth_queries_done']=end;save(report)
            print('EXACT_TRUTH',args.dataset,end,len(q),flush=True)
        del exact
        np.savez_compressed(shared,truth=truth,center=center)
    assert truth.shape==(len(q),100) and np.all((truth>=0)&(truth<len(base)))
    report['shared_truth_sha256']=sha(shared)
    done={(r['bits'],r['seed'],r['rounds'],r['family']) for r in report['rows']}
    report['phase']='quality_cases';save(report)
    for bits in args.bits:
        for seed in SEEDS:
            for rounds in (1,2,3):
                if all((bits,seed,rounds,fam) in done for fam in ('RaBitQ','EDEN')):continue
                mat,check=matrix(d,seed,rounds)
                key=(seed,rounds)
                if not any((c['seed'],c['rounds'])==key for c in report['transform_checks']):report['transform_checks'].append(check)
                qz=qr@mat[:input_d]
                for family in ('RaBitQ','EDEN'):
                    if (bits,seed,rounds,family) in done:continue
                    report['active_case']={'bits':bits,'seed':seed,'rounds':rounds,'family':family,'stage':'encoding_and_primary_scores','base_rows_done':0}
                    save(report);print('CASE_START',args.dataset,bits,seed,rounds,family,flush=True)
                    leaf=faiss.index_factory(d,f'{family}{bits}')
                    leaf.train(np.ascontiguousarray((xt.astype('float64')-center)@mat[:input_d],dtype='float32'))
                    faiss.copy_array_to_vector(np.zeros(d,dtype='float32'),leaf.center)
                    if family=='RaBitQ':leaf.qb=0
                    best_d=np.empty((len(q),0));best_i=np.empty((len(q),0),dtype='int64')
                    errors=np.empty(len(base));norms=np.empty(len(base));vnmse=np.empty(len(base))
                    for start in range(0,len(base),CHUNK):
                        end=min(start+CHUNK,len(base));res=base[start:end].astype('float64')-center
                        z=res@mat[:input_d];leaf.add(np.ascontiguousarray(z,dtype='float32'))
                        rec=audit.helpers.reconstruct_leaf(faiss,leaf,start,end-start).astype('float64')
                        errors[start:end]=np.sum((z-rec)**2,axis=1);norms[start:end]=np.sum(res**2,axis=1)
                        vnmse[start:end]=np.divide(errors[start:end],norms[start:end],out=np.zeros(end-start),where=norms[start:end]>0)
                        next_d=np.empty((len(q),min(100,best_d.shape[1]+len(res))))
                        next_i=np.empty(next_d.shape,dtype='int64')
                        for qs in range(0,len(q),QBATCH):
                            qe=min(qs+QBATCH,len(q))
                            scores=qnorm[qs:qe,None]+norms[None,start:end]-2*qz[qs:qe]@rec.T
                            next_d[qs:qe],next_i[qs:qe]=merged_topk(best_d[qs:qe],best_i[qs:qe],scores,start)
                        best_d,best_i=next_d,next_i
                        report['active_case']['base_rows_done']=end;save(report)
                        print('CASE_BASE_PROGRESS',args.dataset,bits,seed,rounds,family,end,len(base),flush=True)
                    primary={str(k):audit.helpers.per_query_recall(truth[:,:k],best_i[:,:k]) for k in (10,100)}
                    # Save primary completion BEFORE potentially expensive secondary search.
                    prefix=f'{family}-b{bits}-s{seed}-r{rounds}'
                    primaryfile=detail/(prefix+'-primary.npz')
                    np.savez_compressed(primaryfile,primary_ids=best_i,squared_error=errors,residual_norm2=norms,vnmse=vnmse)
                    codes=faiss.rev_swig_ptr(leaf.codes.data(),leaf.codes.size())
                    codehash=hashlib.sha256(memoryview(codes)).hexdigest()
                    row={'bits':bits,'seed':seed,'rounds':rounds,'family':family,'code_bytes':int(leaf.code_size),
                        'packed_code_sha256':codehash,'primary':{k:{'mean':float(v.mean()),'by_query':v.tolist()} for k,v in primary.items()},
                        'reconstruction_full':{'pooled_nmse':float(errors.sum()/norms.sum()),'vnmse':float(vnmse[norms>0].mean()),
                            'squared_error_sum':float(errors.sum()),'original_norm2_sum':float(norms.sum()),'count':len(base)},
                        'reconstruction_10k':{'pooled_nmse':float(errors[:10000].sum()/norms[:10000].sum()),'vnmse':float(vnmse[:10000][norms[:10000]>0].mean())},
                        'primary_detail_file':primaryfile.name,'primary_detail_sha256':sha(primaryfile)}
                    report['primary_checkpoint']=row
                    report['active_case'].update(stage='actual_native_search',native_queries_done=0);save(report)
                    print('PRIMARY_DONE',args.dataset,bits,seed,rounds,family,float(primary['100'].mean()),flush=True)
                    native_ids=np.empty((len(q),100),dtype='int64')
                    for qs in range(0,len(q),100):
                        qe=min(qs+100,len(q))
                        _,native_ids[qs:qe]=leaf.search(np.ascontiguousarray(qz[qs:qe],dtype='float32'),100)
                        report['active_case']['native_queries_done']=qe;save(report)
                        print('NATIVE_QUERY_PROGRESS',args.dataset,bits,seed,rounds,family,qe,len(q),flush=True)
                    native={str(k):audit.helpers.per_query_recall(truth[:,:k],native_ids[:,:k]) for k in (10,100)}
                    nativefile=detail/(prefix+'-native.npz');np.savez_compressed(nativefile,native_ids=native_ids)
                    row.update(native={k:{'mean':float(v.mean()),'by_query':v.tolist()} for k,v in native.items()},
                        native_detail_file=nativefile.name,native_detail_sha256=sha(nativefile))
                    report['rows'].append(row);done.add((bits,seed,rounds,family));report.pop('primary_checkpoint',None)
                    save(report);print('CASE_DONE',args.dataset,bits,seed,rounds,family,'primaryR100',primary['100'].mean(),
                        'nativeR100',native['100'].mean(),'NMSE',row['reconstruction_full']['pooled_nmse'],flush=True)
                    del leaf
    report.update(status='complete' if len(report['rows'])==90 else 'partial_complete',phase='invocation_finished',finished_invocation_utc=stamp())
    report.pop('active_case',None);save(report)
    print('INVOCATION_DONE',args.dataset,args.bits,len(report['rows']),90,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('word2vec','gist','sift'),required=True)
    parser.add_argument('--bits',type=int,nargs='+',choices=(2,3,4),default=[2,3,4]);args=parser.parse_args()
    faiss.omp_set_num_threads(8)
    with threadpool_limits(limits=8):run(args)
