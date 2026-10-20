"""Validate packed native BF16 bytes, zero policies and scalar-reference scores."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import eden_group_bf16 as oracle
from native_eden_group import GroupEDEN,to_bf16

OUT=Path(__file__).resolve().parent
def main():
    words=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x00008000,0x00018000,0,0x80000000,
        0x7f800000,0xff800000,0x7fc00001,0xffc00001],dtype='uint32')
    np.testing.assert_array_equal(to_bf16(words.view('float32')),oracle.to_bf16(words.view('float32')))
    rng=np.random.default_rng(20261027);x=rng.standard_normal((17,448),dtype='float32');x[0]=0;x[1,64:]=0
    eps=np.finfo('float32').eps;x[2]=np.sqrt(eps*.4/64);x[3]=np.sqrt(eps*.1/448)
    center=rng.standard_normal(448,dtype='float32')
    rows=[]
    for bits in (2,3,4):
        for shifted in (False,True):
            c=center if shifted else None;values=x if c is None else x+c
            for stride in (448*bits//8+18,{2:136,3:200,4:264}[bits]):
                codec=GroupEDEN(bits,c,stride)
                actual=codec.encode(values);expected=oracle.encode(values,bits,c,stride)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(codec.decode(actual),oracle.decode(expected,bits,c))
                q=rng.standard_normal((5,448),dtype='float32');q=q if c is None else q+c
                expected_scores=oracle.scores(q,expected,bits,c)
                actual_scores=codec.scores_reference(q,actual)
                np.testing.assert_allclose(actual_scores,expected_scores,rtol=3e-6,atol=3e-4)
                backing=np.empty(actual.size+1,dtype='uint8');unaligned=backing[1:].reshape(actual.shape);unaligned[:]=actual
                np.testing.assert_array_equal(codec.decode(unaligned),codec.decode(actual))
                with np.testing.assert_raises(ValueError):codec.encode(np.full((1,448),np.nan,dtype='float32'))
                with np.testing.assert_raises(ValueError):codec.scores_reference(np.full((1,448),np.inf,dtype='float32'),actual)
                rows.append({'bits':bits,'shifted_center':shifted,'stride':stride,'bytes_exactly_equal_oracle':True,
                    'decode_exactly_equal_oracle':True,'score_max_abs_error':float(np.max(np.abs(actual_scores-expected_scores))),
                    'unaligned_decode_valid':True,'nonfinite_inputs_rejected':True})
    report={'status':'complete','scope':'native codec and scalar-reference score validation ONLY; not optimized speed evidence',
        'dll_sha256':hashlib.sha256((OUT/'experimental-kernel/eden_group_native.dll').read_bytes()).hexdigest(),
        'cpp_sha256':hashlib.sha256((OUT/'experimental-kernel/eden_group.cpp').read_bytes()).hexdigest(),
        'oracle_sha256':hashlib.sha256((OUT/'eden_group_bf16.py').read_bytes()).hexdigest(),
        'bf16_special_and_ties_case_count':len(words),'test_case_count':len(rows),'rows':rows}
    (OUT/'native-eden-group-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
