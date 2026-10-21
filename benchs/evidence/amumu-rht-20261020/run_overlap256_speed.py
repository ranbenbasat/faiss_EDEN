"""Finite speed controller, CPU0 interference telemetry. No system changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

OUT=Path(__file__).resolve().parent
TARGET=OUT/'overlap256-speed-workload.json'
assert not TARGET.exists(),'Inspect prior attempt before running again'
own=psutil.Process();own.cpu_affinity([2])
def cpu(p):
    t=p.cpu_times();return t.user+t.system
def used(t):return t.user+t.system+t.interrupt+t.dpc
env=os.environ.copy();env.update(AMUMU_PIN_PCORE='1',OVERLAP_MONITORED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
windows=[]
with (OUT/'overlap256-speed.log').open('w') as log,(OUT/'overlap256-speed.err').open('w') as error:
    worker=subprocess.Popen([sys.executable,'-u',str(OUT/'bench_overlap256_speed.py')],cwd=OUT.parent,env=env,stdout=log,stderr=error)
    proc=psutil.Process(worker.pid);a=psutil.cpu_times(percpu=True);wa=cpu(proc);ma=cpu(own);start=time.perf_counter()
    print('SPEED_WORKER_LAUNCHED',worker.pid,flush=True)
    while worker.poll() is None:
        try:worker.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        b=psutil.cpu_times(percpu=True);end=time.perf_counter();elapsed=end-start
        try:wb=cpu(proc);valid=True
        except psutil.Error:wb=wa;valid=False
        mb=cpu(own)
        if elapsed>0:
            windows.append(dict(start=start,end=end,valid=valid,
                external_cpu_cores=max(0,sum(used(y)-used(x) for x,y in zip(a,b))-(wb-wa)-(mb-ma))/elapsed,
                cpu0_external_fraction=max(0,used(b[0])-used(a[0])-(wb-wa))/elapsed,
                note=None if valid else 'INVALID terminal process counter unavailable; prior cumulative counter retained, no reset-tozero delta'))
        TARGET.write_text(json.dumps({'worker_pid':worker.pid,'windows':windows},indent=2))
        a,wa,ma,start=b,wb,mb,end
rawfile=OUT/'overlap256-native-speed.json'
if rawfile.exists():
    raw=json.loads(rawfile.read_text())
    def qualify(row):
        overlap=[w for w in windows if w['valid'] and w['end']>row['start'] and w['start']<row['end']]
        return bool(overlap) and overlap[0]['start']<=row['start'] and overlap[-1]['end']>=row['end'] and all(w['external_cpu_cores']<=.6 and w['cpu0_external_fraction']<=.05 for w in overlap)
    for row in raw.get('rows',[])+raw.get('transform_only',[]):row['qualified_low_target_core_interference']=qualify(row)
    if raw['status']=='measurements_complete_pending_telemetry':
        raw['status']='complete_qualified' if all(r['qualified_low_target_core_interference'] for r in raw['rows']) else 'complete_with_unqualified_samples'
    raw['worker_exit_code']=worker.returncode
    rawfile.write_text(json.dumps(raw,indent=2))
print('SPEED_ATTEMPT_FINISHED',worker.returncode,flush=True)
