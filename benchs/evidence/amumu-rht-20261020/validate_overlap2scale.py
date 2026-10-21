"""Independent packed-layout/scoring checks, native per-group codec replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
import audit_msong_quality as audit
from native_overlap2scale import encode,decode,Search

OUT=Path(__file__).resolve().parent
def main():
    rng=np.random.default_rng(73191);rows=[]
    for family in ('EDEN','RaBitQ'):
        for bits in (2,3,4):
            for d,groups in ((420,2),(420,1),(512,1)):
                x=rng.standard_normal((37,d)).astype('float32');q=rng.standard_normal((7,d)).astype('float32')
                codes,table,native_rec=encode(x,family,bits,groups);rec=decode(codes,d,bits,groups,table)
                np.testing.assert_array_equal(rec,native_rec)
                widths=[164,256] if groups==2 else [d];start=0
                for width in widths:
                    part=np.ascontiguousarray(x[:,start:start+width]);leaf=faiss.index_factory(width,f'{family}{bits}');leaf.train(part)
                    faiss.copy_array_to_vector(np.zeros(width,dtype='float32'),leaf.center);leaf.add(part)
                    expected=audit.helpers.reconstruct_leaf(faiss,leaf,0,len(x))
                    np.testing.assert_array_equal(rec[:,start:start+width],expected);start+=width
                payload=(d*bits+7)//8
                norm=np.ascontiguousarray(codes[:,payload+4*groups:]).view('<f4').ravel()
                np.testing.assert_array_equal(norm,np.add.accumulate(x*x,axis=1,dtype='float32')[:,-1])
                reference=np.maximum(np.sum(q.astype('float64')**2,axis=1)[:,None]+norm-2*q.astype('float64')@rec.astype('float64').T,0)
                search=Search(codes,table,d,bits,groups);scores=search.scores(q)
                np.testing.assert_allclose(scores,reference,rtol=2e-6,atol=2e-4)
                dist,ids=search.search(q,10);expected_ids=np.argsort(reference,axis=1,kind='stable')[:,:10]
                np.testing.assert_array_equal(ids,expected_ids)
                np.testing.assert_array_equal(dist,np.take_along_axis(scores,ids,axis=1))
                rows.append({'family':family,'bits':bits,'dimension':d,'groups':groups,'bytes':codes.shape[1],
                    'native_group_reconstruction_exact':True,'independent_FP64_score_max_abs_error':float(np.max(np.abs(scores-reference))),
                    'independent_rankings_exact':True})
    eps=np.finfo('float32').eps
    special=np.zeros((2,420),dtype='float32');special[1]=np.sqrt(eps/300)
    for family in ('EDEN','RaBitQ'):
        codes,table,_=encode(special,family,2,2);payload=105
        scales=np.ascontiguousarray(codes[:,payload:payload+8]).view('<f4')
        norm=np.ascontiguousarray(codes[:,payload+8:payload+12]).view('<f4').ravel()
        assert norm[0]==0 and norm[1]>eps
        if family=='EDEN':
            np.testing.assert_array_equal(scales,0)
            np.testing.assert_array_equal(decode(codes,420,2,2,table),0)
        else:
            # Native RaBitQ retains tiny nonzero groups; do not impose EDEN's EPS rule.
            assert np.isfinite(scales).all()
    report={'status':'complete','rows':rows,'whole_original_norm_preserved_when_both_EDEN_groups_tiny':True,
        'codebook_policy':'Native EDEN164 and256 each normalize with their OWN dimension/norm and original unbiased scales; RaBitQ native164/256 per-residual optimization. FP32 query score, one final clamp.',
        'dll_sha256':hashlib.sha256((OUT/'experimental-kernel/overlap2scale.dll').read_bytes()).hexdigest()}
    (OUT/'overlap2scale-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':faiss.omp_set_num_threads(1);main()
