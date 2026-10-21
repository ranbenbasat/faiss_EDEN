"""Finite requested recall+speed experiment; resume original queue in finally."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

OUT=Path(__file__).resolve().parent
TARGET=OUT/'overlap2scale-run-status.json'
assert not TARGET.exists()
own=psutil.Process();own.cpu_affinity([2])
state={'status':'waiting_for_saved_quality_case','pid':os.getpid(),'started':time.time(),'current':None}
def save():
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(TARGET)
def cpu(p):
    t=p.cpu_times();return t.user+t.system
def used(t):return t.user+t.system+t.interrupt+t.dpc
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AMUMU_PIN_PCORE='0',FAISS_OPT_LEVEL='AVX2')
paused=False
try:
    save()
    for _ in range(1800):
        pause=json.loads((OUT/'overlap2-paused-quality.json').read_text())
        if pause['status']=='paused_at_saved_case':paused=True;break
        time.sleep(2)
    if not paused:raise RuntimeError('No saved case pause within1hour; other experiment left intact')
    for p in psutil.process_iter(['pid','name','cmdline']):
        command=' '.join(p.info.get('cmdline') or [])
        assert 'quality_other_datasets_full.py' not in command,'Full-data worker still active; do not measure speed'
    state.update(status='quality');save()
    with (OUT/'overlap2scale-quality.log').open('w') as log,(OUT/'overlap2scale-quality.err').open('w') as error:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'experiment_overlap2scale.py'),'--phase','quality'],cwd=OUT.parent,env=env,stdout=log,stderr=error)
        state['current']={'phase':'quality','worker_pid':child.pid,'log':str(OUT/'overlap2scale-quality.log')};save()
        if child.wait():raise RuntimeError('Two-scale quality/validation failed; see preserved error log')
    state['status']='speed';save();speedenv=env.copy();speedenv.update(AMUMU_PIN_PCORE='1',OVERLAP2_MONITORED='1')
    windows=[]
    with (OUT/'overlap2scale-speed.log').open('w') as log,(OUT/'overlap2scale-speed.err').open('w') as error:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'experiment_overlap2scale.py'),'--phase','speed'],cwd=OUT.parent,env=speedenv,stdout=log,stderr=error)
        proc=psutil.Process(child.pid);state['current']={'phase':'speed','worker_pid':child.pid,'log':str(OUT/'overlap2scale-speed.log')};save()
        a=psutil.cpu_times(percpu=True);wa=cpu(proc);ma=cpu(own);start=time.perf_counter()
        while child.poll() is None:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            b=psutil.cpu_times(percpu=True);end=time.perf_counter();elapsed=end-start;mb=cpu(own)
            try:wb=cpu(proc);valid=True
            except psutil.Error:wb=wa;valid=False
            if elapsed>0:
                windows.append({'start':start,'end':end,'valid':valid,
                    'global_external_cpu_cores':max(0,sum(used(y)-used(x) for x,y in zip(a,b))-(wb-wa)-(mb-ma))/elapsed,
                    'cpu0_external_fraction':max(0,used(b[0])-used(a[0])-(wb-wa))/elapsed,
                    'terminal_counter_note':None if valid else 'Invalidterminalwindow: retainpreviouscumulativeCPU, NEVER reset tozero'})
            (OUT/'overlap2scale-workload.json').write_text(json.dumps({'worker_pid':child.pid,'windows':windows},indent=2))
            a,wa,ma,start=b,wb,mb,end
    speedpath=OUT/'overlap2scale-speed.json'
    if speedpath.exists():
        speed=json.loads(speedpath.read_text())
        for row in speed['rows']:
            covering=[w for w in windows if w['valid'] and w['end']>row['start'] and w['start']<row['end']]
            row['qualified_low_target_core_interference']=bool(covering) and covering[0]['start']<=row['start'] and covering[-1]['end']>=row['end'] and all(w['global_external_cpu_cores']<=.6 and w['cpu0_external_fraction']<=.05 for w in covering)
        if speed['status']=='measurements_complete_pending_telemetry':speed['status']='complete_qualified' if all(r['qualified_low_target_core_interference'] for r in speed['rows']) else 'complete_some_samples_unqualified'
        speed['exit_code']=child.returncode;speedpath.write_text(json.dumps(speed,indent=2))
    if child.returncode:raise RuntimeError('Native speed worker failed; preserve rows/errors')
    state.update(status='experiment_finished',current=None);save()
except Exception as error:
    state.update(status='failed',error=str(error));save()
finally:
    if paused:
        # The previous requested experiment resumes even if this pilot failed/gate unavailable.
        with (OUT/'quality-resume-after-overlap2.log').open('w') as log,(OUT/'quality-resume-after-overlap2.err').open('w') as error:
            child=subprocess.Popen([sys.executable,'-u',str(OUT/'resume_quality_after_overlap2.py')],cwd=OUT.parent,env={**env,'AMUMU_PIN_PCORE':'0'},stdout=log,stderr=error)
        state['quality_resume_controller_pid']=child.pid;save()
        print('ORIGINAL_QUALITY_RESUMED',child.pid,flush=True)
