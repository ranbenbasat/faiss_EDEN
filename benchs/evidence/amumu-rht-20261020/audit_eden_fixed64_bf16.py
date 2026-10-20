"""EDEN-only grouped BF16 first quality pilot, no proxy performance claims."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
import eden_group_bf16 as codec
import audit_msong_quality as helpers
from native_block64 import NativeTransform

OUT=Path(__file__).resolve().parent
TARGET=OUT/'msong-eden-fixed64-bf16-quality.json'
def main():
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    exact=faiss.IndexFlatL2(420);exact.add(xb)
    _,truth=exact.search(xq,100)
    frozen=json.loads((OUT/'results/msong_native_k100_pilot.json').read_text())
    baseline={(r['bits'],r['seed']):r for r in frozen['rows'] if r['index']=='EDEN' and r['mode']=='rht1'}
    tuning=json.loads((OUT/'native-block64-tuning.json').read_text())
    report={'status':'running','priority':'EDEN only:CORRECTED:448 fixed H64x7 TWO independent-sign rounds, NO permutation/cross-group mixing, group BF16 rounded EDEN scales',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'codec_sha256':hashlib.sha256((OUT/'eden_group_bf16.py').read_bytes()).hexdigest(),
        'native_transform_dll_sha256':hashlib.sha256((OUT/'experimental-kernel/block_transform.dll').read_bytes()).hexdigest(),
        'shapes':{'xt':list(xt.shape),'xb':list(xb.shape),'xq':list(xq.shape)},'seeds':list(range(12345,12350)),
        'bits':[2,3,4],'k':[10,100],'speed_results_available':False,'corrected_primary_scope_no_permutation':True,'native_permutation_pointer_is_null':True,
        'quality_oracle':'native optimized transform and frozen native64 EDEN encoders; BF16 packing/score aggregation oracle is NOT a timed search implementation',
        'layout':'448*b/8 scalar bytes, seven BF16 scales RNE(14B), one original whole-residual FP32 norm2(4B); strict-same-stride suffix reserved zero',
        'scale_formula':'ORIGINAL unbiased EDEN group norm2 / dot(group residual, codeword); per-group dimension64, no biased-scale substitution or extra clipping',
        'norm_convention':'whole UNQUANTIZED transformed residual norm2, sequential FP32 summation across448 coordinates; orthogonal original-space equivalent, not reconstructed norm',
        'zero_rule':'whole norm2<=floatEPS gives zero vector/norm metadata; when whole passes, each group uses original EDEN64 epsilon policy, without erasing the original whole norm',
        'scores':'one max(total L2 estimate,0) clamp after summing all groups; never per-group clamps; stable distance/ID ordering',
        'centering':'group BF16 and global448 controls share frozen native EDEN448 trained centroid exactly; full512 baseline retains its frozen native centroid',
        'limits':['prefix20k/200 on one fixed dataset','conditional paired seed/query intervals','no native grouped speed claims until native codec/search is implemented and validated'],
        'rows':[]}
    if TARGET.exists():
        previous=json.loads(TARGET.read_text())
        if previous['codec_sha256']!=report['codec_sha256'] or previous['script_sha256']!=report['script_sha256']:
            assert all(r['mode'] in ('full512_global','fixed64_global') for r in previous['rows'])
            previous['resume_provenance']={'old_script_sha256':previous['script_sha256'],'old_codec_sha256':previous['codec_sha256'],
                'retained_control_rows':len(previous['rows']),
                'reason':'Checkpoint call fix; compare equal-shape compact/padded scores bitwise, cross-GEMM-size scores with tight roundoff tolerance; explicit invalid-input tests added before first grouped row was saved'}
            previous['script_sha256']=report['script_sha256'];previous['codec_sha256']=report['codec_sha256']
        if previous['status']=='complete':return
        report=previous
    existing={(r['seed'],r['bits'],r['mode']) for r in report['rows']}
    def save():
        temporary=TARGET.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(TARGET)
    for seed in report['seeds']:
        transforms={'full512':NativeTransform('full512',seed,1,False,tuning['chosen_strategies']['full512']),
                    'block64':NativeTransform('block64',seed,2,False,tuning['chosen_strategies']['block64'])}
        projected={mode:[t.apply(v) for v in (xt,xb,xq)] for mode,t in transforms.items()}
        for bits in (2,3,4):
            cap={2:136,3:200,4:264}[bits]
            global_controls={}
            for label,projection in (('full512_global',projected['full512']),('fixed64_global',projected['block64'])):
                zt,zb,zq=projection
                leaf=faiss.index_factory(zb.shape[1],f'EDEN{bits}');leaf.train(zt);leaf.add(zb)
                center=faiss.vector_to_array(leaf.center)
                _,pred=leaf.search(zq,100)
                rec=leaf.reconstruct_n(0,len(zb))
                recalls={k:helpers.helpers.per_query_recall(truth[:,:k],pred[:,:k]) for k in (10,100)}
                if label=='full512_global':np.testing.assert_array_equal(recalls[100],baseline[(bits,seed)]['recall_by_query'])
                global_controls[label]=(center,zb,zq)
                row={'seed':seed,'bits':bits,'mode':label,'dimension':zb.shape[1],'code_bytes_per_vector':int(leaf.code_size),
                    'hard_budget_cap_bytes':cap,'within_cap':int(leaf.code_size)<=cap,'centroid_sha256':helpers.digest(center),
                    'k_metrics':{str(k):{'recall':float(v.mean()),'recall_by_query':v.tolist()} for k,v in recalls.items()},
                    'reconstruction_10k':helpers.helpers.reconstruction_metrics(zb[:10000],rec[:10000],center),
                    'reconstruction_full20k':helpers.helpers.reconstruction_metrics(zb,rec,center)}
                if (seed,bits,label) not in existing:report['rows'].append(row);existing.add((seed,bits,label));save()
            center,zb,zq=global_controls['fixed64_global']
            compact=codec.encode(zb,bits,center=center)
            padded=codec.encode(zb,bits,center=center,stride=cap)
            np.testing.assert_array_equal(compact,padded[:,:compact.shape[1]])
            reconstruction=codec.decode(compact,bits,center=center)
            scores=codec.scores(zq,compact,bits,center=center)
            # Sort top100 once so its first10 are the true grouped top10.
            prediction=codec.topk(scores,100)
            sample_scores=codec.scores(zq[:3],padded[:19],bits,center=center)
            np.testing.assert_array_equal(sample_scores,codec.scores(zq[:3],compact[:19],bits,center=center))
            np.testing.assert_allclose(sample_scores,scores[:3,:19],rtol=1e-12,atol=1e-8)
            residual=np.ascontiguousarray(zb-center,dtype='float32')
            group_norms=np.add.accumulate((residual.reshape(-1,7,64)**2),axis=2,dtype='float32')[:,:,-1]
            whole_norms=np.add.accumulate(residual**2,axis=1,dtype='float32')[:,-1]
            metrics={str(k):{'recall':float(helpers.helpers.per_query_recall(truth[:,:k],prediction[:,:k]).mean()),
                'recall_by_query':helpers.helpers.per_query_recall(truth[:,:k],prediction[:,:k]).tolist()} for k in (10,100)}
            for label,codes in (('fixed64_group_bf16_compact',compact),('fixed64_group_bf16_samestride',padded)):
                row={'seed':seed,'bits':bits,'mode':label,'dimension':448,'code_bytes_per_vector':int(codes.shape[1]),
                    'scalar_payload_bytes':448*bits//8,'bf16_scale_bytes':14,'original_norm_bytes':4,
                    'reserved_padding_bytes':int(codes.shape[1]-compact.shape[1]),'hard_budget_cap_bytes':cap,'within_cap':True,
                    'padding_has_no_accuracy_credit':True,'centroid_sha256':helpers.digest(center),
                    'codes_sha256':helpers.digest(codes),'k_metrics':metrics,
                    'tiny_group_count':int(np.count_nonzero(group_norms<=np.finfo('float32').eps)),
                    'total_groups':int(len(zb)*7),'whole_zero_count':int(np.count_nonzero(whole_norms<=np.finfo('float32').eps)),
                    'all_groups_tiny_whole_passes_count':int(np.count_nonzero(np.all(group_norms<=np.finfo('float32').eps,axis=1)&(whole_norms>np.finfo('float32').eps))),
                    'reconstruction_10k':helpers.helpers.reconstruction_metrics(zb[:10000],reconstruction[:10000],center),
                    'reconstruction_full20k':helpers.helpers.reconstruction_metrics(zb,reconstruction,center)}
                if (seed,bits,label) not in existing:report['rows'].append(row);existing.add((seed,bits,label));save()
            print('GROUP_BF16_CASE',seed,bits,'recall100',metrics['100']['recall'],'compact',compact.shape[1],'cap',cap,flush=True)
    assert len(report['rows'])==60
    report['status']='complete';save()
if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):main()

