"""Predeclared BLOCK64/448 primary quality pilot; no proxy timing/QPS.

All train/base/query rows share one transform. Global native RaBitQ/EDEN
training and scales remain unchanged. Native HadamardRotation is unmodified.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
import audit_msong_quality as audit
import experimental_block32 as block
import bench_ann_v2 as ann

OUT=Path(__file__).resolve().parent
TARGET=OUT/'msong-block64-primary-quality.json'
SEEDS=list(range(12345,12350))
CONFIGS=[('full512_native',512,1,False),('block64',448,2,True),('block64',448,2,False),
         ('block64',448,1,True),('block64',448,3,True),
         ('full512_native',512,2,False),('full512_native',512,3,False),
         *[(mode,d,0,False) for d in (420,448,512) for mode in ('none','urr')]]


def transform(mode,d,rounds,permute,seed):
    if mode=='block64':
        p=block.plan(d,seed,64)
        checks=block.validate(p,rounds,permute)
        matrix=block.forward(np.eye(420),p,rounds,permute).astype('float32')
        checks.update(widths=p['widths'],block_size=64,plan_state_bytes_float32_signs_int32_permutation=4*(3*d+d))
        vt=faiss.LinearTransform(420,d)
        faiss.copy_array_to_vector(matrix.T.copy().ravel(),vt.A)
        vt.is_trained=vt.is_orthonormal=True
    elif mode=='full512_native':
        vt=faiss.HadamardRotation(420,seed,rounds)
        matrix=vt.apply(np.eye(420,dtype='float32'))
        checks={'butterfly_addsub_count':rounds*512*9,'block_size':512,
                'plan_state_bytes_float32_signs_int32_permutation':3*512*4}
        # Authenticate our portable RNG against all native full-RHT signs.
        signs=block.plan(512,seed,64)['signs']
        for i in range(3):
            np.testing.assert_array_equal(signs[i],faiss.vector_to_array(getattr(vt,f'signs{i+1}')))
    else:
        matrix=np.eye(d,420,dtype='float32').T if mode=='none' else audit.helpers.haar_matrix(d,seed)[:,:420].T.copy()
        vt=faiss.LinearTransform(420,d)
        faiss.copy_array_to_vector(matrix.T.copy().ravel(),vt.A)
        vt.is_trained=vt.is_orthonormal=True
        checks={'butterfly_addsub_count':0 if mode=='none' else None}
    np.testing.assert_allclose(matrix.astype('float64')@matrix.T,np.eye(420),rtol=2e-6,atol=2e-6)
    checks['matrix_sha256_float32']=audit.digest(matrix)
    checks['embedding_gram_max_abs_error']=float(np.max(np.abs(matrix.astype('float64')@matrix.T-np.eye(420))))
    return vt,checks


def save(report):
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)


def main():
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    exact=faiss.IndexFlatL2(420);exact.add(xb)
    _,truth=exact.search(xq,100)
    pilot=json.loads((OUT/'results/msong_native_k100_pilot.json').read_text())
    references={(r['index'],r['bits'],r['mode'],r['seed']):r for r in pilot['rows'] if r['mode'].startswith('rht')}
    report={'status':'running','purpose':'experimental BLOCK64/448 primary QUALITY pilot, not public HR semantics',
        'speed_claims':False,'dense_matrix_transforms':'QUALITY ORACLE ONLY; no timing data/QPS reported',
        'seeds':SEEDS,'configs':[list(c) for c in CONFIGS],'bits':[2,3,4],'families':['RaBitQ','EDEN'],
        'shapes':{'xt':list(xt.shape),'xb':list(xb.shape),'xq':list(xq.shape)},
        'primary_comparison':'block64 dimension448 rounds2 permute versus actual nativefull512 rounds1',
        'transform_definition':'SplitMix64 three independent sign arrays; separate seedxorD1B54A32D192ED03 Fisher-Yates permutation, identical at each between-round gap; normalized orthogonal H64 blocks',
        'code_scale':'one global native trained center and quantizer scale, NEVER per-block scales',
        'limits':['row-disjoint prefix split, not value-disjoint','single fixed dataset; conditional paired seed/query CIs',
                  'no full-RHT concentration theory assumed for this block construction','native optimized experimental kernel required before any speed benchmark'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'oracle_sha256':hashlib.sha256((OUT/'experimental_block32.py').read_bytes()).hexdigest(),
        'rows':[],'transform_checks':[],'fastscan_storage_checks':[]}
    if TARGET.exists():
        previous=json.loads(TARGET.read_text())
        if previous['status']=='complete':
            return
        assert previous['script_sha256']==report['script_sha256'] and previous['oracle_sha256']==report['oracle_sha256']
        report=previous
    existing={(r['mode'],r['dimension'],r['rounds'],r['permute'],r['seed'],r['family'],r['bits']) for r in report['rows']}
    # Primary90 cases precede secondary depth and none/URR controls.
    for config in CONFIGS:
        mode,d,rounds,permute=config
        for seed in SEEDS:
            if all((mode,d,rounds,permute,seed,family,bits) in existing for family in ('RaBitQ','EDEN') for bits in (2,3,4)):
                continue
            vt,checks=transform(mode,d,rounds,permute,seed)
            check={'mode':mode,'dimension':d,'rounds':rounds,'permute':permute,'seed':seed,**checks}
            if not any(all(c[k]==check[k] for k in ('mode','dimension','rounds','permute','seed')) for c in report['transform_checks']):
                report['transform_checks'].append(check)
            zt,zb,zq=[np.ascontiguousarray(vt.apply(v),dtype='float32') for v in (xt,xb,xq)]
            for family in ('RaBitQ','EDEN'):
                for bits in (2,3,4):
                    key=(mode,d,rounds,permute,seed,family,bits)
                    if key in existing:
                        continue
                    leaf=faiss.index_factory(d,f'{family}{bits}')
                    if family=='RaBitQ':
                        leaf.qb=4
                    leaf.train(zt);leaf.add(zb)
                    _,prediction=leaf.search(zq,100)
                    recalls=audit.helpers.per_query_recall(truth,prediction)
                    reconstruction=audit.helpers.reconstruct_leaf(faiss,leaf,0,min(10000,len(zb)))
                    metrics=audit.helpers.reconstruction_metrics(zb[:len(reconstruction)],reconstruction,faiss.vector_to_array(leaf.center))
                    row={'mode':mode,'dimension':d,'rounds':rounds,'permute':permute,'seed':seed,'family':family,'bits':bits,
                         'recall100':float(recalls.mean()),'recall_by_query':recalls.tolist(),
                         'native_code_size_bytes':int(leaf.code_size),'raw_payload_bytes':(d*bits+7)//8,
                         'code_storage_bytes':int(leaf.code_size)*len(xb),'reconstruction':metrics,
                         'global_center_sha256':audit.digest(faiss.vector_to_array(leaf.center)),
                         'codes_sha256':audit.digest(faiss.vector_to_array(leaf.codes))}
                    if mode=='full512_native':
                        reference=references[(family,bits,f'rht{rounds}',seed)]
                        diff=float(np.max(np.abs(recalls-np.array(reference['recall_by_query']))))
                        assert diff==0,('Frozen full512 replay mismatch',row,diff)
                        row['max_per_query_recall_difference_vs_frozen_pilot']=diff
                    report['rows'].append(row);existing.add(key);save(report)
                    print('QUALITY',mode,d,rounds,permute,seed,family,bits,row['recall100'],leaf.code_size,flush=True)
    # Storage-only FastScan check, actual packed blocks and actual list sizes.
    # It is not a FastScan quality/speed result for the proposed transform.
    for d in (420,448,512):
        padded=np.pad(xb,((0,0),(0,d-420)))
        trained=np.pad(xt,((0,0),(0,d-420)))
        for bits in (2,3,4):
            if any(r['dimension']==d and r['bits']==bits for r in report['fastscan_storage_checks']):
                continue
            leaf=faiss.index_factory(d,f'IVF64,RaBitQfs{bits}')
            leaf.cp.seed=97531;leaf.cp.niter=25;leaf.cp.nredo=1;leaf.qb=4
            leaf.train(trained);leaf.add(padded)
            memory=ann.storage(leaf,leaf)
            report['fastscan_storage_checks'].append({'dimension':d,'bits':bits,**memory,
                'physical_bytes_per_vector_including_block_padding':memory['physical_code_bytes_including_block_padding']/len(xb)})
            save(report);print('FASTSCAN_STORAGE',d,bits,memory['physical_code_bytes_including_block_padding'],flush=True)
    assert len(report['rows'])==390
    assert len(report['fastscan_storage_checks'])==9
    report['status']='complete';save(report)


if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
