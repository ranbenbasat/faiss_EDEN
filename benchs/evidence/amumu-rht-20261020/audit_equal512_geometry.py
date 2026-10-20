"""Equal512-budget geometry/Kac mechanism control; quality only.

Same NumPy512 sign stream, frozen native per-residual encoder, float queries,
centroids and raw payload. Partial256 mechanism is NOT historical448 geometry.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
import audit_msong_quality as helpers

OUT=Path(__file__).resolve().parent
TARGET=OUT/'msong-equal512-geometry-audit.json'


def matrix(mode,rounds,seed):
    rng=np.random.default_rng(seed)
    y=np.eye(512,dtype='float64');sign_hashes=[]
    for r in range(rounds):
        signs=rng.choice(np.array([-1.,1.]),512)
        sign_hashes.append(helpers.digest(signs));y *= signs
        if mode=='full512_HD':
            y=helpers.fwht(y)
        else:
            start=0 if r%2==0 else 256
            y[:,start:start+256]=helpers.fwht(y[:,start:start+256])
            if mode=='partial256_pairwise':
                a,b=y[:,:256].copy(),y[:,256:].copy()
                y[:,:256],y[:,256:]=(a+b)/np.sqrt(2),(a-b)/np.sqrt(2)
    error=float(np.max(np.abs(y@y.T-np.eye(512))))
    assert error<1e-12
    return y,{'sign_sha256_by_round':sign_hashes,'orthogonality_max_abs_error':error,
        'basis_e300_nonzeros':int(np.count_nonzero(np.abs(y[300])>1e-12)),
        'basis_e300_norm2':float(y[300]@y[300]),'matrix_sha256':helpers.digest(y)}


def main():
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    prior=json.loads((OUT/'msong-quality-audit.json').read_text())
    exact=faiss.IndexFlatL2(420);exact.add(xb)
    _,truth=exact.search(xq,100)
    report={'status':'running','purpose':'equal512-budget geometry and pairwise-layer mechanism control; quality-only',
        'source_sha256':prior['source_npz_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seeds':[202605170,202605171],'centers':[1,128],'bits':[2,3,4],'rounds':[1,2,3],
        'modes':['full512_HD','partial256_no_pairwise','partial256_pairwise'],
        'dimension':512,'quantizer':'frozen native RaBitQ per-residual threshold encoder; trained residual center forced zero',
        'query':'float centroid-relative full-code estimator; no IVF traversal or candidate filtering',
        'limits':['prefix20k base/200 queries; two NumPy sign seeds','512 partial256 construction is mechanism control, NOT historical448 replay',
                  'Native RNG full512 controls are separately reported in msong-pairwise-layer-audit.json'],
        'rows':[],'checks':[]}
    if TARGET.exists():
        previous=json.loads(TARGET.read_text())
        if previous['status']=='complete':
            return
        assert previous['script_sha256']==report['script_sha256']
        report['rows']=previous['rows']
    existing={(r['centers'],r['seed'],r['rounds'],r['mode'],r['bits']) for r in report['rows']}
    def save():
        tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)
    for count in report['centers']:
        if count==1:
            centers=xt.mean(axis=0,keepdims=True)
        else:
            km=faiss.Kmeans(420,128,niter=20,nredo=1,seed=20260517,verbose=False);km.train(xt);centers=km.centroids
        assert helpers.digest(centers)==next(r['centers_sha256'] for r in prior['fixed_t_rows'] if r['centers']==count)
        coarse=faiss.IndexFlatL2(420);coarse.add(centers)
        _,labels=coarse.search(xb,1);labels=labels[:,0]
        residual=xb.astype('float64')-centers[labels].astype('float64')
        norms2=np.sum(residual**2,axis=1);valid=norms2>0
        for seed in report['seeds']:
            for rounds in report['rounds']:
                for mode in report['modes']:
                    mat,check=matrix(mode,rounds,seed)
                    report['checks'].append({'centers':count,'seed':seed,'rounds':rounds,'mode':mode,**check})
                    z=residual@mat[:420]
                    np.testing.assert_allclose(helpers.centered_scores(xq[:8],centers,labels[:32],mat,z[:32],norms2[:32]),
                        np.sum((xq[:8,None,:].astype('float64')-xb[None,:32,:])**2,axis=2),rtol=1e-10,atol=1e-6)
                    for bits in report['bits']:
                        if (count,seed,rounds,mode,bits) in existing:
                            continue
                        native=faiss.index_factory(512,f'RaBitQ{bits}')
                        native.train(np.ascontiguousarray(z,dtype='float32'))
                        faiss.copy_array_to_vector(np.zeros(512,dtype='float32'),native.center)
                        native.add(np.ascontiguousarray(z,dtype='float32'))
                        reconstruction=helpers.helpers.reconstruct_leaf(faiss,native,0,len(z)).astype('float64')
                        code_sha=helpers.digest(faiss.vector_to_array(native.codes));code_bytes=int(native.code_size)
                        del native
                        error=np.sum((z-reconstruction)**2,axis=1)
                        cn=np.sum(reconstruction**2,axis=1)
                        best=np.divide(np.sum(z*reconstruction,axis=1),cn,out=np.zeros_like(cn),where=cn>0)
                        best_error=np.sum((z-best[:,None]*reconstruction)**2,axis=1)
                        scores=helpers.centered_scores(xq,centers,labels,mat,reconstruction,norms2)
                        prediction=np.argpartition(scores,99,axis=1)[:,:100]
                        recalls=helpers.recall(prediction,truth)
                        report['rows'].append({'centers':count,'centers_sha256':helpers.digest(centers),'mode':mode,
                            'seed':seed,'rounds':rounds,'bits':bits,'dimension':512,'raw_payload_bytes':bits*64,
                            'native_code_size_bytes':code_bytes,'native_codes_sha256':code_sha,
                            'recall100':float(np.mean(recalls)),'recall_by_query':recalls,
                            'unbiased_nmse':float(error.sum()/norms2.sum()),
                            'unbiased_vnmse':float(np.mean(error[valid]/norms2[valid])),
                            'best_scalar_vnmse':float(np.mean(best_error[valid]/norms2[valid]))})
                        save();print('EQUAL512',count,seed,rounds,mode,bits,float(np.mean(recalls)),flush=True)
    assert len(report['rows'])==108
    for count in report['centers']:
        for seed in report['seeds']:
            for rounds in report['rounds']:
                hashes=[r['sign_sha256_by_round'] for r in report['checks'] if r['centers']==count and r['seed']==seed and r['rounds']==rounds]
                assert len(hashes)==3 and all(h==hashes[0] for h in hashes)
    report['status']='complete';save()


if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
