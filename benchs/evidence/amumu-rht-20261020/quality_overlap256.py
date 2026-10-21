"""Bounded MSong recall/storage pilot, NO timing while full-data queue is busy."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
import audit_msong_quality as audit
from native_overlap256 import Overlap

OUT=Path(__file__).resolve().parent
TARGET=OUT/'overlap256-recall.json'
CACHE=OUT/'overlap256-native-cache'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not TARGET.exists(),'Preserve existing pilot; do not overwrite'
    validation=json.loads((OUT/'overlap256-validation.json').read_text());assert validation['status']=='complete'
    CACHE.mkdir(exist_ok=True)
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    center=xt.mean(axis=0)
    rt,rb,rq=[np.ascontiguousarray(v-center) for v in (xt,xb,xq)]
    exact=faiss.IndexFlatL2(420);exact.add(xb);_,truth=exact.search(xq,100)
    np.savez(CACHE/'shared.npz',center=center,truth=truth)
    report={'status':'running','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scope':'MSong prefix20k/200queries, five paired seeds12345..49; overlapH256[0:256] then[164:420] versus actual frozen FAISS fullH512 ONE round',
        'seeds':list(range(12345,12350)),'bits':[2,3,4],'families':['RaBitQ','EDEN'],
        'mode_labels':{'full512':'actual frozen FAISS IndexPreTransform/HR512+actual native leaf search',
            'overlap420':'native overlap420 transform+actual native leaf search'},
        'source_sha256':sha(Path(__file__)),'dll_sha256':validation['dll_sha256'],
        'centering':'Shared original xt FP32 mean, subtract original-space center inFP32 before both transforms; leaf centers zero.',
        'estimators':'Actual native Index.search; RaBitQqb4 default production query quantization, EDENoriginal-unbiased/defaultclamp. Same within each comparison. RaBitQqb0 recall recorded separately, not used to label qb4 speed.',
        'memory':'Actual native compact codes, different420vs512coordinate counts. Fixedbits not equalbytes. Implicit sequential IDs.',
        'speed_status':'Not measured: other full-data quality worker active. Native isolated speed phase requires quiet host / current queue pause.',
        'validation':'Full420basis independent dense parityH256 and exact gram identity; actual native query pipeline matches explicit transform/search; actual fullHR and equallyoptimizedfull512 control match.',
        'rows':[]}
    def save():
        tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)
    save()
    for seed in report['seeds']:
        t=Overlap(seed,2)
        transformed={'full512':[t.frozen.apply(v) for v in (rt,rb,rq)],
                     'overlap420':[t.apply(v) for v in (rt,rb,rq)]}
        for family in ('RaBitQ','EDEN'):
            for bits in (2,3,4):
                for mode in ('full512','overlap420'):
                    zt,zb,zq=transformed[mode];leaf=faiss.index_factory(zb.shape[1],f'{family}{bits}');leaf.train(zt)
                    faiss.copy_array_to_vector(np.zeros(zb.shape[1],dtype='float32'),leaf.center)
                    if family=='RaBitQ':leaf.qb=4
                    leaf.add(zb)
                    index=faiss.IndexPreTransform(t.frozen,leaf) if mode=='full512' else leaf
                    _,ids,_=t.pipeline(xq,center,index,0 if mode=='full512' else 1)
                    explicit_d,explicit_i=leaf.search(zq,100);np.testing.assert_array_equal(ids,explicit_i)
                    if mode=='full512':
                        _,fast_i,_=t.pipeline(xq,center,leaf,2);np.testing.assert_array_equal(fast_i,ids)
                    reconstruction=audit.helpers.reconstruct_leaf(faiss,leaf,0,len(xb))
                    rec=audit.helpers.reconstruction_metrics(zb,reconstruction,np.zeros(zb.shape[1],dtype='float32'))
                    prefix=f'{mode}-{family}-b{bits}-s{seed}'
                    path=CACHE/(prefix+'.index');faiss.write_index(index,str(path))
                    codes=faiss.vector_to_array(leaf.codes)
                    row={'seed':seed,'family':family,'bits':bits,'mode':mode,'dimension':zb.shape[1],
                        'code_bytes':int(leaf.code_size),'code_storage_bytes':int(len(codes)),
                        'signs_sha256':audit.digest(t.signs),'codes_sha256':audit.digest(codes),
                        'index_path':str(path),'index_sha256':sha(path),
                        'recall':{str(k):{'mean':float(audit.helpers.per_query_recall(truth[:,:k],ids[:,:k]).mean()),
                            'by_query':audit.helpers.per_query_recall(truth[:,:k],ids[:,:k]).tolist()} for k in (10,100)},
                        'reconstruction_full20k':rec}
                    if family=='RaBitQ':
                        leaf.qb=0;_,q0=leaf.search(zq,100);leaf.qb=4
                        row['qb0_secondary_recall100']=float(audit.helpers.per_query_recall(truth,q0).mean())
                    report['rows'].append(row);save()
                    print('OVERLAP_RECALL',seed,family,bits,mode,'R100',row['recall']['100']['mean'],'bytes',leaf.code_size,flush=True)
    assert len(report['rows'])==60
    report['status']='complete';save()

if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):main()
