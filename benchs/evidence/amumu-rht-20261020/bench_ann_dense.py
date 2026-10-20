"""Native-only dense-nprobe follow-up. No kernel/quantizer tuning.

Predeclared targets90/93/95%; select operating points on calibration queries,
then report held-out recall without selecting again on the held-out queries.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import faiss
from threadpoolctl import threadpool_limits,threadpool_info
import bench_ann_v2 as frozen

OUT=Path(__file__).resolve().parent
PROBES=[1,2,4,*range(8,41),64]


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare_queries(label,xt,xb,xq):
    source=Path(r'C:\Users\Ran\Documents\Codex\TQ-AMUMU\data\msong\msong\msong_query.fvecs')
    candidates=frozen.helpers.read_fvecs(source)
    forbidden={np.where(row==0,0,row).astype('float32').tobytes() for values in (xt,xb,xq) for row in values}
    chosen=[]
    for index in np.random.default_rng(20261022).permutation(len(candidates)):
        row=np.where(candidates[index]==0,0,candidates[index]).astype('float32')
        key=row.tobytes()
        if key not in forbidden:
            chosen.append(int(index));forbidden.add(key)
        if len(chosen)==200:
            break
    assert len(chosen)==200
    held=np.ascontiguousarray(candidates[chosen],dtype='float32')
    return np.concatenate((xq,held)),{'seed':20261022,'calibration_queries':len(xq),'heldout_queries':len(held),
        'query_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'heldout_source_indices':chosen,'heldout_sha256':sha(held),'calibration_sha256':sha(xq),
        'heldout_value_disjoint_from_training_base_calibration':True,
        'selection':'first200 accepted from query-source permutation, excluding all xt/xb/calibration values and repeated heldout values'}


def run(args):
    directory=OUT/'pilot-data'/args.dataset
    source=directory/'vectors.npz'
    with np.load(source) as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    queries,query_plan=prepare_queries(args.dataset,xt,xb,xq)
    exact=faiss.IndexFlatL2(xb.shape[1]);exact.add(xb)
    _,truth=exact.search(queries,100)
    case_order=list(itertools.product(('RaBitQ','RaBitQFastScan','EDEN'),args.bits,('rht1','rht2','rht3'),range(12345,12350)))
    np.random.default_rng(20261023).shuffle(case_order)
    report={'status':'running','driver_version':'native-dense-v1','metadata':frozen.helpers.metadata(faiss,args),
        'dataset':args.dataset,'source_npz_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'query_plan':query_plan,'targets':[.90,.93,.95],'target_selection':'calibration only; never reselect on heldout',
        'nprobe_grid':PROBES,'nlist':64,'bits':args.bits,'depths':[1,2,3],'k':[10,100],
        'case_order':[list(c) for c in case_order],'completed_cases':[], 'rows':[],
        'timing_qualification':'provisional; inspect parent telemetry and per-pass quiet gates. No interpolation of recall/QPS.',
        'training':'same frozen v2 factory: cp.seed97531 niter25 nredo1, trained once and serialized per case',
        'threadpools':threadpool_info(),'groundtruth':'exact original-space IndexFlatL2 for actual xb',
        'shapes':{'xt':list(xt.shape),'xb':list(xb.shape),'queries':list(queries.shape)}}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        assert old['source_npz_sha256']==report['source_npz_sha256'] and old['query_plan']==query_plan
        assert old['bits']==args.bits and old['driver_version']==report['driver_version']
        report=old
    def save():
        frozen.helpers.write_results(args.output,report)
    cache=OUT/'dense-index-cache'/args.dataset/report['source_npz_sha256'][:16]
    cache.mkdir(parents=True,exist_ok=True)
    save()
    for number,(family,bits,mode,seed) in enumerate(case_order,1):
        case=[family,bits,mode,seed]
        if case in report['completed_cases']:
            continue
        cache_path=cache/f'{family}{bits}-{mode}-{seed}.index'
        if not cache_path.exists():
            index,leaf,_=frozen.make_index(420,family,bits,mode,seed,False,64)
            index.train(xt);index.add(xb)
            faiss.write_index(index,str(cache_path))
        # All passes and nprobe settings reuse these exact trained codes/centroids.
        index=faiss.read_index(str(cache_path));leaf=faiss.downcast_index(index.index)
        cache_hash=hashlib.sha256(cache_path.read_bytes()).hexdigest()
        for repeat_pass in range(2):
            gate=args.output.with_suffix(f'.gate-{os.getpid()}-{number}-{repeat_pass}.json')
            # Parent performs quiet gating on CPU2; this worker waits without doing measurements.
            request={'case_number':number,'repeat_pass':repeat_pass,'gate':str(gate)}
            args.output.with_suffix('.gate-request.json').write_text(json.dumps(request))
            while not gate.exists():
                time.sleep(1)
            quiet=json.loads(gate.read_text())
            conditions=list(itertools.product(PROBES,(10,100)))
            np.random.default_rng(20261024+repeat_pass).shuffle(conditions)
            for nprobe,k in conditions:
                key=(family,bits,mode,seed,repeat_pass,nprobe,k)
                if any(tuple(r[q] for q in ('family','bits','mode','seed','repeat_pass','nprobe','k'))==key for r in report['rows']):
                    continue
                leaf.nprobe=nprobe
                predicted=frozen.helpers.search_batches(index,queries,k,256)
                samples=[]
                if quiet['available']:
                    for _ in range(3):
                        begin=time.perf_counter();predicted=frozen.helpers.search_batches(index,queries,k,256)
                        samples.append(time.perf_counter()-begin)
                recall=frozen.helpers.per_query_recall(truth[:,:k],predicted)
                report['rows'].append({'family':family,'bits':bits,'mode':mode,'seed':seed,'repeat_pass':repeat_pass,
                    'nprobe':nprobe,'k':k,'nlist':64,'qb':4 if family!='EDEN' else None,'d_out':512,
                    'index_cache_sha256':cache_hash,'quiet_gate':quiet,'search_seconds_samples':samples,
                    'search_seconds_median':float(np.median(samples)) if samples else None,
                    'qps':len(queries)/float(np.median(samples)) if samples else None,
                    'calibration_recall':float(recall[:len(xq)].mean()),'heldout_recall':float(recall[len(xq):].mean()),
                    'calibration_recall_by_query':recall[:len(xq)].tolist(),'heldout_recall_by_query':recall[len(xq):].tolist()})
            save()
        report['completed_cases'].append(case);save()
        print(f'[{number}/{len(case_order)}] {family}{bits}/{mode} seed={seed} saved with dense probes, two frozen-index passes',flush=True)
    report['status']='complete';save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['msong','msong_random','msong_random_value_disjoint'],required=True)
    parser.add_argument('--bits',type=int,nargs='+',choices=[1,2],default=[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1,choices=[1])
    args=parser.parse_args()
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)
