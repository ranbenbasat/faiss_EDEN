"""Bounded QUALITY-only factor audit, not historical Figure 2 replication.

Historical algorithm/configuration supplied by coordinator, 2026-10-20.
Exact historical source/RNG stream and full-data centroids are still required.
No latency/QPS result from this diagnostic is evidence about speed.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import heapq
import ast
import math
import importlib.util
import json
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'amumu-evidence'
spec = importlib.util.spec_from_file_location('helpers', ROOT/'benchs/bench_rht.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fwht(x):
    n = x.shape[1]
    y = x.copy()
    width = 1
    while width < n:
        z = y.reshape(len(y), -1, 2*width)
        left, right = z[:, :, :width].copy(), z[:, :, width:].copy()
        z[:, :, :width], z[:, :, width:] = left+right, left-right
        width *= 2
    return y / np.sqrt(n)


def matrix(mode, rounds, seed):
    """Explicit diagnostic oracle: same NumPy sign RNG for both transform types."""
    d = 448 if mode == 'legacy448_description' else 512
    rng = np.random.default_rng(seed)
    y = np.eye(d, dtype=np.float64)
    for r in range(rounds):
        y *= rng.choice(np.array([-1., 1.]), d)
        if d == 512:
            y = fwht(y)
        else:
            start = 0 if r % 2 == 0 else 192
            y[:, start:start+256] = fwht(y[:, start:start+256])
            first, second = y[:, :224].copy(), y[:, 224:].copy()
            y[:, :224] = (first+second)/np.sqrt(2)
            y[:, 224:] = (first-second)/np.sqrt(2)
    error = float(np.max(np.abs(y @ y.T - np.eye(d))))
    assert error < 1e-12, (mode, rounds, error)
    return y, {'mode': mode, 'rounds': rounds, 'seed': seed,
               'orthogonality_max_abs_error': error,
               'basis_e300_nonzeros': int(np.count_nonzero(np.abs(y[300]) > 1e-12)),
               'basis_e300_norm2': float(y[300] @ y[300]),
               'matrix_sha256_float64': digest(y)}


def recall(prediction, truth):
    return [len(set(a) & set(b))/len(b) for a, b in zip(prediction, truth)]


def corrected_threshold(values, ex_bits):
    """Merged events, capped initial counts, and initial feasible candidate.

    Preserve the historical bounded search interval; this is not a claim of
    global optimality outside that interval. Evaluate every feasible event.
    """
    limit = 2**ex_bits-1
    end = (limit+10)/values.max()
    start = end*[0.,.15,.20,.52,.59,.71,.75,.77,.81][ex_bits]
    cur = np.minimum(np.floor(start*values+1e-5),limit).astype('int64')
    numerator = float(np.dot(cur+.5,values))
    denominator = float(np.sum((cur+.5)**2))
    best_score, best_t = numerator/np.sqrt(denominator), start
    heap = [(float((cur[i]+1)/v),i) for i,v in enumerate(values) if v>0 and cur[i]<limit]
    heapq.heapify(heap)
    while heap:
        t,i = heapq.heappop(heap)
        if t >= end:
            break
        cur[i] += 1
        numerator += float(values[i])
        denominator += 2*int(cur[i])
        score = numerator/np.sqrt(denominator)
        if score > best_score:
            best_score,best_t = score,t
        if cur[i]<limit:
            heapq.heappush(heap,(float((cur[i]+1)/values[i]),i))
    return float(best_t)


def centered_scores(xq, centers, labels, rotation, reconstruction, norms2):
    """Historical centroid-relative estimator; no original-space norm shortcut.

    q-c must be used: unbiased reconstruction need not preserve <c,r>.
    """
    scores = np.empty((len(xq),len(labels)), dtype='float64')
    for cid,c in enumerate(centers):
        ids = np.flatnonzero(labels==cid)
        if not len(ids):
            continue
        qr = xq.astype('float64')-c
        qrot = qr @ rotation[:420]
        scores[:,ids] = np.sum(qr**2,axis=1)[:,None]+norms2[ids][None,:]-2*(qrot @ reconstruction[ids].T)
    return scores


def save(report):
    target = OUT/'msong-quality-audit.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(target)


def main():
    faiss.omp_set_num_threads(1)
    source = OUT/'pilot-data/msong/vectors.npz'
    with np.load(source) as data:
        xt, xb, xq = [np.ascontiguousarray(data[k], dtype='float32') for k in ('xt','xb','xq')]
    truth_index = faiss.IndexFlatL2(420)
    truth_index.add(xb)
    _, truth = truth_index.search(xq, 100)
    report = {'status':'running', 'purpose':'quality-only bounded factor diagnostic; NOT Figure2 replication',
              'source_npz_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'shapes':{k:list(v.shape) for k,v in zip(('xt','xb','xq'),(xt,xb,xq))},
              'historical_mismatch_remaining':['prefix 20k rather than full 994185 base',
                '200 rather than 1000 queries', '128 centers trained on pilot xt, not full base',
                'description-derived legacy transform; exact historical source/RNG not yet obtained',
                'fixed448 t used unchanged at512 to isolate transform without recalibration'],
              'seeds':[202605170,202605171], 'groundtruth':'exact original-space flat L2 top100',
              'fixed_t':{'2':21.21314389348,'3':27.90001520854,'4':59.20726108563},
              'transform_checks':[], 'fixed_t_rows':[], 'native_query_rows':[]}
    manifest = json.loads((OUT/'paper-source/source-manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        assert hashlib.sha256((OUT/'paper-source'/relative).read_bytes()).hexdigest()==expected
    report['historical_source_manifest'] = manifest
    report['historical_helper_caveat'] = 'Original ZIP omitted imported rht_glove_ann_search.py; surviving helper is NOT historically authenticated.'
    historical_tree = ast.parse((OUT/'paper-source/frozen/rht_rabitq_multibit_distance_probe.py').read_text())
    historical_function = next(n for n in historical_tree.body if isinstance(n,ast.FunctionDef) and n.name=='best_rescale_factor')
    historical_namespace = {'np':np,'math':math,'List':list,'Tuple':tuple}
    exec(compile(ast.Module(body=[historical_function],type_ignores=[]),'frozen-best-rescale-factor','exec'),historical_namespace)
    historical_threshold = historical_namespace['best_rescale_factor']
    gaussian = np.random.default_rng(42).standard_normal((100,448))
    gaussian = np.abs(gaussian/np.linalg.norm(gaussian,axis=1,keepdims=True))
    replayed_t = {str(bits):float(np.mean([historical_threshold(v,bits-1) for v in gaussian])) for bits in (2,3,4)}
    assert all(np.isclose(replayed_t[k],v,rtol=1e-11,atol=1e-11) for k,v in report['fixed_t'].items())
    report['replayed_historical_calibration'] = replayed_t
    counterexample = np.array([1.,.3,.2,.1]); counterexample /= np.linalg.norm(counterexample)
    report['calibration_counterexample'] = {}
    for name,fn in (('historical',historical_threshold),('corrected',corrected_threshold)):
        value = fn(counterexample,3)
        code = np.minimum(np.floor(value*counterexample+1e-5),7)+.5
        report['calibration_counterexample'][name] = {'t':value,'cosine':float(np.dot(counterexample,code)/np.linalg.norm(code))}
    corrected_t = {str(bits):float(np.mean([corrected_threshold(v,bits-1) for v in gaussian])) for bits in (2,3,4)}
    report['corrected_fixed_t_448'] = corrected_t
    report['corrected_calibration_definition'] = 'Gaussian seed42 100x448; historical bounded interval; capped initialization, initial objective evaluated, merged threshold heap. Same448 constants used at512 to isolate geometry.'
    for centers_count in (1,128):
        if centers_count == 1:
            centers = xt.mean(axis=0, keepdims=True)
        else:
            km = faiss.Kmeans(420,128,niter=20,nredo=1,seed=20260517,verbose=False)
            km.train(xt)
            centers = km.centroids
        coarse = faiss.IndexFlatL2(420)
        coarse.add(centers)
        _, labels = coarse.search(xb,1)
        base_centers = centers[labels[:,0]].astype('float64')
        residual = xb.astype('float64')-base_centers
        norms2 = np.sum(residual**2,axis=1)
        for seed in report['seeds']:
            for rounds in (1,2,3):
                for mode in ('legacy448_description','full512_oracle'):
                    rotation, check = matrix(mode,rounds,seed)
                    report['transform_checks'].append(check)
                    z = residual @ rotation[:420]
                    # Independently check the centroid-relative scoring algebra with exact residuals.
                    sample_scores = centered_scores(xq[:8],centers,labels[:32,0],rotation,z[:32],norms2[:32])
                    exact_sample = np.sum((xq[:8,None,:].astype('float64')-xb[None,:32,:])**2,axis=2)
                    np.testing.assert_allclose(sample_scores,exact_sample,rtol=1e-10,atol=1e-6)
                    unit = np.divide(z,np.sqrt(norms2)[:,None],out=np.zeros_like(z),where=norms2[:,None]>0)
                    for bits in (2,3,4):
                        for quantizer in ('historical_fixed_t','corrected_fixed_t','native_per_residual'):
                            t = None
                            if quantizer == 'native_per_residual':
                                # Train to initialize internals, then enforce residual center zero.
                                native = faiss.index_factory(z.shape[1],f'RaBitQ{bits}')
                                native.train(np.ascontiguousarray(z,dtype='float32'))
                                faiss.copy_array_to_vector(np.zeros(z.shape[1],dtype='float32'),native.center)
                                native.add(np.ascontiguousarray(z,dtype='float32'))
                                reconstruction = helpers.reconstruct_leaf(faiss,native,0,len(z)).astype('float64')
                                del native
                            else:
                                t = (report['fixed_t'] if quantizer=='historical_fixed_t' else corrected_t)[str(bits)]
                                code = np.where(unit>=0,1.,-1.)*(np.minimum(np.floor(t*np.abs(unit)+1e-5),2**(bits-1)-1)+.5)
                                dot = np.sum(z*code,axis=1)
                                scale = np.divide(norms2,dot,out=np.zeros_like(dot),where=dot!=0)
                                reconstruction = scale[:,None]*code
                            rec_norm2 = np.sum(reconstruction**2,axis=1)
                            best = np.divide(np.sum(z*reconstruction,axis=1),rec_norm2,out=np.zeros_like(rec_norm2),where=rec_norm2>0)
                            unbiased_error = np.sum((z-reconstruction)**2,axis=1)
                            best_error = np.sum((z-best[:,None]*reconstruction)**2,axis=1)
                            score = centered_scores(xq,centers,labels[:,0],rotation,reconstruction,norms2)
                            prediction = np.argpartition(score,99,axis=1)[:,:100]
                            rq = recall(prediction,truth)
                            row = {'centers':centers_count,'centers_sha256':digest(centers),'bits':bits,
                                'mode':mode,'rounds':rounds,'seed':seed,'recall100':float(np.mean(rq)),
                                'recall_by_query':rq,'unbiased_nmse':float(unbiased_error.sum()/norms2.sum()),
                                'unbiased_vnmse':float(np.mean(unbiased_error[norms2>0]/norms2[norms2>0])),
                                'best_scalar_vnmse':float(np.mean(best_error[norms2>0]/norms2[norms2>0])),
                                'fixed_t':t,'query_bits':0,'quantizer':quantizer}
                            report['fixed_t_rows'].append(row)
                            save(report)
                            print('FACTOR',centers_count,mode,rounds,bits,seed,quantizer,row['recall100'],flush=True)
    # Reuse exact frozen native factory; query quantization is the ONLY changed factor.
    for seed in range(12345,12350):
        for rounds in (1,2,3):
            for bits in (2,3,4):
                index, leaf, _ = helpers.make_index(faiss,420,'RaBitQ',bits,f'rht{rounds}',seed,0,False)
                index.train(xt)
                index.add(xb)
                codes_before = digest(faiss.vector_to_array(leaf.codes))
                for qb in (0,4):
                    leaf.qb = qb
                    _, pred = index.search(xq,100)
                    rq = recall(pred,truth)
                    assert digest(faiss.vector_to_array(leaf.codes)) == codes_before
                    report['native_query_rows'].append({'bits':bits,'rounds':rounds,'seed':seed,'qb':qb,
                        'codes_sha256':codes_before,'recall100':float(np.mean(rq)),'recall_by_query':rq})
                    save(report)
                print('NATIVE',seed,rounds,bits,flush=True)
    assert len(report['fixed_t_rows']) == 216
    assert len(report['native_query_rows']) == 90
    report['status'] = 'complete'
    save(report)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
