"""Resume only missing query-bit audit rows after host interruption."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
import audit_msong_quality as audit

out=Path(__file__).resolve().parent
report=json.loads((out/'msong-quality-audit.json').read_text())
assert len(report['fixed_t_rows'])==216
source=out/'pilot-data/msong/vectors.npz'
assert hashlib.sha256(source.read_bytes()).hexdigest()==report['source_npz_sha256']
report['resume_provenance']={'driver':Path(__file__).name,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'preserved_factor_cases':216,'preserved_native_query_rows':len(report['native_query_rows']),
    'reason':'host interruption; existing codes/recall checked when rebuilding a partial query pair'}
existing={(r['bits'],r['rounds'],r['seed'],r['qb']):r for r in report['native_query_rows']}
assert len(existing)==len(report['native_query_rows'])
with threadpool_limits(limits=1):
    faiss.omp_set_num_threads(1)
    with np.load(source) as data:
        xt,xb,xq=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb','xq')]
    truth_index=faiss.IndexFlatL2(420);truth_index.add(xb)
    _,truth=truth_index.search(xq,100)
    for seed in range(12345,12350):
        for rounds in (1,2,3):
            for bits in (2,3,4):
                if all((bits,rounds,seed,qb) in existing for qb in (0,4)):
                    continue
                index,leaf,_=audit.helpers.make_index(faiss,420,'RaBitQ',bits,f'rht{rounds}',seed,0,False)
                index.train(xt);index.add(xb)
                codes=audit.digest(faiss.vector_to_array(leaf.codes))
                for qb in (0,4):
                    key=(bits,rounds,seed,qb)
                    if key in existing:
                        assert existing[key]['codes_sha256']==codes
                        continue
                    leaf.qb=qb
                    _,prediction=index.search(xq,100)
                    recalls=audit.recall(prediction,truth)
                    row={'bits':bits,'rounds':rounds,'seed':seed,'qb':qb,'codes_sha256':codes,
                         'recall100':float(np.mean(recalls)),'recall_by_query':recalls}
                    existing[key]=row;report['native_query_rows'].append(row)
                    audit.save(report)
                    print('RESUMED',seed,rounds,bits,qb,flush=True)
assert len(report['native_query_rows'])==90
report['status']='complete';audit.save(report)
subprocess.run([sys.executable,'-u',str(out/'export_quality_audit.py')],check=True)
