"""Quality-only pairwise-layer ablation, not a historical Figure2 replay.

Equal448 geometry/signs/block schedule/centroids/quantizer/bit budget isolate
the pairwise layer. Quantized codewords must be regenerated for each changed
transform; holding literal codewords fixed would test an inconsistent decoder.
Native-RNG full512 results are a separate control, not shared sign realizations.
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
TARGET=OUT/'msong-pairwise-layer-audit.json'


def rotation(rounds,seed,pairwise,pad512):
    rng=np.random.default_rng(seed)
    y=np.eye(448,dtype='float64')
    sign_hashes=[]
    for r in range(rounds):
        signs=rng.choice(np.array([-1.,1.]),448)
        sign_hashes.append(helpers.digest(signs))
        y *= signs
        start=0 if r%2==0 else 192
        y[:,start:start+256]=helpers.fwht(y[:,start:start+256])
        if pairwise:
            a,b=y[:,:224].copy(),y[:,224:].copy()
            y[:,:224],y[:,224:]=(a+b)/np.sqrt(2),(a-b)/np.sqrt(2)
    error=float(np.max(np.abs(y@y.T-np.eye(448))))
    assert error<1e-12
    if pad512:
        y=np.pad(y,((0,0),(0,64)))
    return y,{'sign_sha256_by_round':sign_hashes,'orthogonality_max_abs_error':error,
              'basis_e300_nonzeros':int(np.count_nonzero(np.abs(y[300])>1e-12)),
              'basis_e300_norm2':float(y[300]@y[300]),'matrix_sha256':helpers.digest(y)}


def main():
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    prior=json.loads((OUT/'msong-quality-audit.json').read_text())
    km=faiss.Kmeans(420,128,niter=20,nredo=1,seed=20260517,verbose=False)
    km.train(xt);centers=km.centroids
    assert helpers.digest(centers)==next(r['centers_sha256'] for r in prior['fixed_t_rows'] if r['centers']==128)
    coarse=faiss.IndexFlatL2(420);coarse.add(centers)
    _,labels=coarse.search(xb,1);labels=labels[:,0]
    residual=xb.astype('float64')-centers[labels].astype('float64')
    norms2=np.sum(residual**2,axis=1)
    exact=faiss.IndexFlatL2(420);exact.add(xb)
    _,truth=exact.search(xq,100)
    report={'status':'running','purpose':'quality-only equal448 pairwise-layer ablation; no speed claims',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'centers':128,'centers_sha256':helpers.digest(centers),'shapes':{'xt':list(xt.shape),'xb':list(xb.shape),'xq':list(xq.shape)},
        'fixed_t':prior['fixed_t'],'equal448_seeds':[202605170,202605171],'native_RNG_seeds':list(range(12345,12350)),
        'code_budget':'raw database payload ceil(bits*dimension/8) bytes; excludes common scale factors, IDs, centroids and headers',
        'limitations':['prefix20k base and200 queries, not full dataset','two NumPy sign seeds for isolated448 comparison',
                       'missing frozen historical sign/FWHT helper remains unauthenticated',
                       'native full512 uses its own actual RNG and separate five-seed control; not same signs as NumPy448'],
        'rows':[],'checks':[]}
    if TARGET.exists():
        previous=json.loads(TARGET.read_text())
        if previous['status']=='complete':
            return
        assert previous['script_sha256']==report['script_sha256']
        report['rows']=previous['rows']
    existing={(r['mode'],r['seed'],r['rounds'],r['dimension'],r['bits']) for r in report['rows']}
    def save():
        tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)
    configurations=[]
    for seed in report['equal448_seeds']:
        for rounds in (1,2,3):
            for pairwise in (False,True):
                for pad512 in (False,True):
                    mat,check=rotation(rounds,seed,pairwise,pad512)
                    configurations.append(('block448_pairwise' if pairwise else 'block448_no_pairwise',seed,rounds,mat,check))
    for seed in report['native_RNG_seeds']:
        for rounds in (1,2,3):
            native=faiss.HadamardRotation(420,seed,rounds)
            mat=native.apply(np.eye(420,dtype='float32')).astype('float64')
            error=float(np.max(np.abs(mat@mat.T-np.eye(420))))
            assert error<2e-6
            configurations.append(('native_full512_RNG',seed,rounds,mat,{'orthogonality_max_abs_error':error,
                'matrix_sha256':helpers.digest(mat),'basis_e300_nonzeros':int(np.count_nonzero(np.abs(mat[300])>1e-12)),
                'basis_e300_norm2':float(mat[300]@mat[300])}))
    for mode,seed,rounds,mat,check in configurations:
        dimension=mat.shape[1]
        report['checks'].append({'mode':mode,'seed':seed,'rounds':rounds,'dimension':dimension,**check})
        z=residual@mat[:420]
        unit=np.divide(z,np.sqrt(norms2)[:,None],out=np.zeros_like(z),where=norms2[:,None]>0)
        zero=norms2==0
        unit[zero]=0
        # Validate the centroid-relative algebra before replacing residuals by codes.
        np.testing.assert_allclose(helpers.centered_scores(xq[:8],centers,labels[:32],mat,z[:32],norms2[:32]),
            np.sum((xq[:8,None,:].astype('float64')-xb[None,:32,:])**2,axis=2),rtol=3e-6,atol=1e-5)
        for bits in (2,3,4):
            if (mode,seed,rounds,dimension,bits) in existing:
                continue
            t=report['fixed_t'][str(bits)]
            code=np.where(unit>=0,1.,-1.)*(np.minimum(np.floor(t*np.abs(unit)+1e-5),2**(bits-1)-1)+.5)
            dot=np.sum(z*code,axis=1)
            scale=np.divide(norms2,dot,out=np.zeros_like(dot),where=dot!=0)
            reconstruction=scale[:,None]*code
            quantizer='historical_fixed_t'
            native_code_size=None
            native_codes_sha=None
            if mode=='native_full512_RNG':
                quantizer='native_per_residual'
                native=faiss.index_factory(512,f'RaBitQ{bits}')
                native.train(np.ascontiguousarray(z,dtype='float32'))
                faiss.copy_array_to_vector(np.zeros(512,dtype='float32'),native.center)
                native.add(np.ascontiguousarray(z,dtype='float32'))
                reconstruction=helpers.helpers.reconstruct_leaf(faiss,native,0,len(z)).astype('float64')
                code=reconstruction
                dot=np.sum(z*code,axis=1)
                native_code_size=int(native.code_size)
                native_codes_sha=helpers.digest(faiss.vector_to_array(native.codes))
                del native
            error=np.sum((z-reconstruction)**2,axis=1)
            cn=np.sum(code**2,axis=1)
            best=dot/cn
            best_error=np.sum((z-best[:,None]*code)**2,axis=1)
            scores=helpers.centered_scores(xq,centers,labels,mat,reconstruction,norms2)
            pred=np.argpartition(scores,99,axis=1)[:,:100]
            recalls=helpers.recall(pred,truth)
            row={'mode':mode,'seed':seed,'rounds':rounds,'dimension':dimension,'bits':bits,
                'raw_payload_bytes':(bits*dimension+7)//8,'fixed_t':t if quantizer=='historical_fixed_t' else None,'codewords_sha256':helpers.digest(code),
                'quantizer':quantizer,'native_code_size_bytes':native_code_size,'native_codes_sha256':native_codes_sha,
                'recall100':float(np.mean(recalls)),'recall_by_query':recalls,
                'unbiased_nmse':float(error.sum()/norms2.sum()),
                'unbiased_vnmse':float(np.mean(error[~zero]/norms2[~zero])),
                'best_scalar_vnmse':float(np.mean(best_error[~zero]/norms2[~zero]))}
            report['rows'].append(row);save()
            print(mode,seed,rounds,dimension,bits,row['recall100'],flush=True)
    assert len(report['rows'])==117
    # Padding unused zero coordinates must not change fixed-t estimated IPs/recall.
    for r in report['rows']:
        if r['dimension']==448:
            padded=next(p for p in report['rows'] if all(p[k]==r[k] for k in ('mode','seed','rounds','bits')) and p['dimension']==512)
            assert r['recall_by_query']==padded['recall_by_query']
    report['status']='complete';save()


if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
