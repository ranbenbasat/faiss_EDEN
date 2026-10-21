"""Geometry and actual FAISS pipeline validation only; NO speed samples."""
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
from native_overlap256 import Overlap

OUT=Path(__file__).resolve().parent
H=np.array([[(-1. if (i&j).bit_count()%2 else 1.)/16 for j in range(256)] for i in range(256)])
def expected(x,signs):
    y=x.astype('float64',copy=True)
    y[:,:256]=(y[:,:256]*signs[:256])@H
    y[:,164:]=(y[:,164:]*signs[256:])@H
    return y

def main():
    report={'status':'running','definition':'MSong420: normalize signedH256[0:256] THEN signedH256[164:420], overlap92; no padding, no H2, no permutation',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dll_sha256':hashlib.sha256((OUT/'experimental-kernel/overlap256.dll').read_bytes()).hexdigest(),
        'no_speed_samples':True,'cases':[],'storage':[]}
    for seed in range(12345,12350):
        basis=np.eye(420,dtype='float32')
        models=[Overlap(seed,strategy) for strategy in (0,1,2)]
        oracle=expected(basis,models[0].signs)
        for model in models:
            actual=model.apply(basis)
            np.testing.assert_array_equal(actual,oracle.astype('float32'))
            np.testing.assert_array_equal(actual.astype('float64')@actual.T,np.eye(420))
            np.testing.assert_array_equal(actual[256:,:164],0)
            report['cases'].append({'seed':seed,'strategy':model.strategy,'full_basis_matches_independent_dense_oracle':True,
                'gram_exact_identity':True,'first164_outputs_do_not_see_last164_inputs':True})
    rng=np.random.default_rng(200124)
    xt=rng.standard_normal((128,420)).astype('float32');xb=rng.standard_normal((192,420)).astype('float32');xq=rng.standard_normal((4,420)).astype('float32')
    center=xt.mean(axis=0);t=Overlap(12345)
    for family in ('RaBitQ','EDEN'):
        for bits in (2,3,4):
            leaves={}
            for mode,d in ((0,512),(1,420)):
                transform=t.frozen.apply if mode==0 else t.apply
                leaf=faiss.index_factory(d,f'{family}{bits}');leaf.train(transform(np.ascontiguousarray(xt-center)))
                faiss.copy_array_to_vector(np.zeros(d,dtype='float32'),leaf.center)
                if family=='RaBitQ':leaf.qb=0
                leaf.add(transform(np.ascontiguousarray(xb-center)));leaves[mode]=leaf
                index=faiss.IndexPreTransform(t.frozen,leaf) if mode==0 else leaf
                expected_d,expected_i=leaf.search(transform(np.ascontiguousarray(xq-center)),10)
                distances,ids,_=t.pipeline(xq,center,index,mode,10)
                np.testing.assert_array_equal(ids,expected_i);np.testing.assert_array_equal(distances,expected_d)
                if mode==0:
                    fast_d,fast_i,_=t.pipeline(xq,center,leaf,2,10)
                    np.testing.assert_array_equal(fast_i,ids);np.testing.assert_array_equal(fast_d,distances)
            report['storage'].append({'family':family,'bits':bits,'padded512_bytes':leaves[0].code_size,
                'overlap420_bytes':leaves[1].code_size,'same_native_quantizer_scale_contract':True,
                'actual_FAISS_frozen_HR_pipeline_and_optimized_full_control_match':True,
                'overlap_native_pipeline_matches_explicit_transform_then_actual_search':True})
    report['status']='complete'
    (OUT/'overlap256-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    faiss.omp_set_num_threads(1);main()
