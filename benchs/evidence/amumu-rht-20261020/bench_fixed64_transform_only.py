"""Corrected NO-PERM transform-only hardware timings, native C++ clock."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import itertools
import json
from pathlib import Path
import time
import numpy as np
from native_block64_fixed64 import NativeTransform
OUT=Path(__file__).resolve().parent
TARGET=OUT/'native-fixed64-transform-benchmark.json'
def main():
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:x=np.ascontiguousarray(data['xb'],dtype='float32')
    tuning=json.loads((OUT/'native-block64-fixed64-tuning.json').read_text())
    report={'status':'running','corrected_no_permutation':True,'clock':'native C++ steady_clock',
        'scope':'fixed64 TWO independent sign rounds vs equally optimized bitwise-frozen full512 ONE round',
        'allocation':'allocation-free transform kernel timing; end-to-end query allocations separately included in pipeline benchmark',
        'dll_sha256':tuning['dll_sha256'],'strategies':tuning['chosen_strategies'],'rows':[]}
    for repeat in range(2):
        gate=OUT/f'fixed64-transform-gate-{os.getpid()}-{repeat}.json'
        (OUT/'fixed64-transform-gate-request.json').write_text(json.dumps({'worker_pid':os.getpid(),'gate':str(gate),'repeat':repeat}))
        while not gate.exists():time.sleep(1)
        quiet=json.loads(gate.read_text())
        cases=list(itertools.product(('full512','block64'),range(12345,12350),(1,32,256,20000)))
        np.random.default_rng(20261124+repeat).shuffle(cases)
        for mode,seed,n in cases:
            t=NativeTransform(mode,seed,1 if mode=='full512' else 2,False,tuning['chosen_strategies'][mode]);assert t.permutation is None
            t.time_transform(x[:n],3)
            samples=[t.time_transform(x[:n],max(10,100000//n))[0] for _ in range(5)]
            report['rows'].append({'mode':mode,'seed':seed,'n':n,'repeat':repeat,'quiet_gate':quiet,
                'seconds_per_apply_samples':samples,'ns_per_vector_median':float(np.median(samples)/n*1e9),
                'qualification':'gate_passed_pending_telemetry_review' if quiet['available'] else 'diagnostic_nonquiet'})
        (TARGET).write_text(json.dumps(report,indent=2));print('FIXED64_TRANSFORM_PASS',repeat,flush=True)
    assert len(report['rows'])==80
    report['status']='complete';TARGET.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
