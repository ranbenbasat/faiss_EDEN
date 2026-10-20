"""Attach to quality preparation, then native runtime with quiet gates/telemetry."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

OUT=Path(__file__).resolve().parent
psutil.Process().cpu_affinity([2])
env=os.environ.copy();env.update(AMUMU_PIN_PCORE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',FAISS_OPT_LEVEL='AVX2')
state={'status':'waiting_for_native_quality','supervisor_pid':os.getpid(),'quality_worker_pid':6672,
    'priority':'CORRECTED NO-PERMUTATION fixed64 grouped BF16 native flat timing, strict same-stride and optimized uniform-global controls',
    'other_jobs':'native dense checkpoint four cases retained; no other measurements run concurrently'}
def save():(OUT/'eden-fixed64-runtime-status.json').write_text(json.dumps(state,indent=2))
def sample(child):
    cpu=psutil.cpu_times();own=psutil.Process().cpu_times()
    try:worker=child.cpu_times();worker_time=worker.user+worker.system
    except psutil.Error:worker_time=0
    return (time.perf_counter(),cpu.user+cpu.system,own.user+own.system,worker_time)
def busy(a,b):return max(0.,b[1]-a[1]-(b[2]-a[2])-(b[3]-a[3]))/(b[0]-a[0])
def quiet(child):
    windows=[];streak=0;begin=time.perf_counter()
    for _ in range(6):
        a=sample(child);time.sleep(5);b=sample(child);value=busy(a,b)
        windows.append({'external_cpu_cores':value});streak=streak+1 if value<.35 else 0
        if streak>=3:break
    return {'available':streak>=3,'threshold_cpu_cores':.35,'required_consecutive_windows':3,
        'elapsed_wall_seconds':time.perf_counter()-begin,'time':dt.datetime.now().astimezone().isoformat(),
        'windows':windows,'fallback':'diagnostic_nonquiet, not validated speed evidence' if streak<3 else None}
try:
    save()
    while True:
        path=OUT/'native-eden-fixed64-flat-quality.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':break
        if not psutil.pid_exists(6672):raise RuntimeError('Native quality worker exited before completion; inspect log')
        time.sleep(2)
    with (OUT/'bench-eden-fixed64-primary.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'bench_eden_fixed64_primary.py')],cwd=OUT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',current={'pid':child.pid,'script':'bench_eden_fixed64_primary.py'});save()
        process=psutil.Process(child.pid);before=sample(process)
        with (OUT/'eden-fixed64-runtime.workload.jsonl').open('a') as telemetry:
            while child.poll() is None:
                request=OUT/'eden-fixed64-gate-request.json'
                if request.exists():
                    data=json.loads(request.read_text());gate=Path(data['gate'])
                    if data['worker_pid']==child.pid and not gate.exists():
                        state.update(status='quiet_gate',gate_request=data);save()
                        result=quiet(process);gate.write_text(json.dumps(result,indent=2))
                        state.update(status='running');save();before=sample(process)
                time.sleep(1);after=sample(process)
                telemetry.write(json.dumps({'time':dt.datetime.now().astimezone().isoformat(),
                    'external_cpu_cores':busy(before,after),'above_idle_threshold':busy(before,after)>.35,
                    'monitor':'system user+system CPU minus supervisor and worker CPU; no per-process enumeration overhead'})+'\n');telemetry.flush();before=after
    if child.returncode:raise RuntimeError(f'Native grouped runtime exited{child.returncode}; inspect log')
    state.update(status='fixed64_primary_complete_pending_export',current=None);save()
except BaseException as exc:
    state.update(status='failed',error=repr(exc));save();raise

