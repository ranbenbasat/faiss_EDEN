"""Explicit native timed routing: kind2→full512 ONE, others→fixed64 TWO/null permutation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
from native_block64_fixed64 import NativeTransform
from native_eden_group import GroupEDEN
from native_eden_search_fixed64 import EdenCodeSearch
import eden_group_bf16 as oracle

OUT=Path(__file__).resolve().parent
def main():
    rng=np.random.default_rng(60130);raw=rng.standard_normal((128,420),dtype='float32');q=rng.standard_normal((7,420),dtype='float32')
    rows=[]
    for bits in (2,3,4):
        for mode,kind in (('full512',2),('block64',3),('block64',0),('block64',1)):
            rounds=1 if kind==2 else 2;t=NativeTransform(mode,12345,rounds,False,2 if kind==2 else 1)
            assert t.permutation is None
            xb=t.apply(raw);xq=t.apply(q);center=np.zeros(t.dimension,dtype='float32')
            for padded in ((False,True) if kind==0 else (False,)):
                if kind in (2,3):
                    leaf=faiss.index_factory(t.dimension,f'EDEN{bits}');leaf.train(np.zeros((1,t.dimension),dtype='float32'));leaf.add(xb)
                    codes=faiss.vector_to_array(leaf.codes).reshape(len(xb),leaf.code_size)
                else:
                    stride={2:136,3:200,4:264}[bits] if padded else None
                    codes=GroupEDEN(bits,center,stride,'bf16' if kind==0 else 'fp32').encode(xb)
                index=EdenCodeSearch(codes,bits,kind,center,1)
                expected_d,expected_i=index.search(xq,100)
                for batch in (1,3,32,256):
                    actual_d,actual_i,samples=index.time_pipeline(t,q,100,batch,1)
                    np.testing.assert_array_equal(actual_i,expected_i);np.testing.assert_array_equal(actual_d,expected_d);assert samples[0]>0
                row={'bits':bits,'kind':kind,'dimension':t.dimension,'expected_rounds':rounds,'permutation_pointer_null':True,
                    'actual_code_stride':codes.shape[1],'strict_same_stride':padded,
                    'native_timed_pipeline_matches_explicit_projected_search':True,'tested_batches':[1,3,32,256]}
                if padded:assert codes.shape[1]=={2:136,3:200,4:264}[bits]
                rows.append(row)
        # The primary comparator is the ACTUAL frozen Faiss virtual search.
        tf=NativeTransform('full512',12345,1,False,2);leaf=faiss.index_factory(512,f'EDEN{bits}');leaf.train(tf.apply(raw));leaf.add(tf.apply(raw))
        fd,fi=leaf.search(faiss.HadamardRotation(420,12345,1).apply(q),100)
        for batch in (1,32,256):
            nd,ni,_=tf.search(leaf,q,100,batch,1);np.testing.assert_array_equal(ni,fi);np.testing.assert_array_equal(nd,fd)
        rows.append({'bits':bits,'actual_frozen_FAISS_Index_search_primary_comparator':True,'full512_ONE_round':True})
    # Finite-precision differential near clamp/self scores: report error, not exact
    # unbiasedness/self-distance-zero. Equal-score heap policy is tested separately.
    x=np.full((3,448),np.float32(1.510417608499078),dtype='float32')
    near=[]
    for bits in (2,3,4):
        codes=GroupEDEN(bits).encode(x)
        queries=np.concatenate([x[:1]*np.float32(v) for v in (.999999,1.,1.000001)])
        expected=oracle.scores(queries,codes,bits)
        actual=EdenCodeSearch(codes,bits,0,np.zeros(448,dtype='float32')).scores(queries)
        scale=float(np.max(np.sum(queries.astype('float64')**2,axis=1)))
        tolerance=float(64*np.finfo('float32').eps*max(1.,scale*4))
        assert np.max(np.abs(actual-expected))<=tolerance
        near.append({'bits':bits,'native_scores':actual[:,0].tolist(),'oracle_scores':expected[:,0].tolist(),
            'max_abs_difference':float(np.max(np.abs(actual-expected))),'conservative_fp32_absolute_error_bound':tolerance,
            'exact_unbiasedness_or_zero_self_distance_claimed':False})
    result={'status':'complete','corrected_no_permutation':True,'routing':'codekind2=>modefull512,rounds1; kinds0/1/3=>modefixed64,rounds2; permutation pointer null',
        'cases':rows,'case_count':len(rows),'near_clamp_differential':near,'not_performance_evidence':True,
        'dll_sha256':hashlib.sha256((OUT/'experimental-kernel/eden_group_fixed64.dll').read_bytes()).hexdigest()}
    (OUT/'native-fixed64-pipeline-mapping-validation.json').write_text(json.dumps(result,indent=2));print('FIXED64_PIPELINE_MAPPING_VALIDATED',len(rows),flush=True)
if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):main()
