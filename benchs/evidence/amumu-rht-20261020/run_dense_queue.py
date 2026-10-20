"""Serial native dense curves, external telemetry and independent quiet gates."""
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
env=os.environ.copy()
env.update(AMUMU_PIN_PCORE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',FAISS_OPT_LEVEL='AVX2')
state={'status':'starting','supervisor_pid':os.getpid(),'completed':[],
       'plan':'dense8..40 every1 plus1/2/4/64 anchors; k10/100; seeds12345..49; full512 RHT1/2/3; families RaBitQ/FastScan/EDEN; 2bit primary, 1bit secondary',
       'targets':[.9,.93,.95],'quiet_threshold_cpu_cores':.35,'quiet_consecutive_5s_windows':3,
       'note':'All speed interpretation provisional until per-pass workload is reviewed. No target choice on heldout queries.'}
def save():
    (OUT/'dense-queue-status.json').write_text(json.dumps(state,indent=2))
def sample():
    result={}
    for p in psutil.process_iter(['name','cpu_times']):
        try:
            t=p.info['cpu_times'];result[p.pid]=(p.info['name'],t.user+t.system)
        except (psutil.Error,TypeError):
            pass
    return result
def activity(before,after,elapsed,exclude):
    active=[{'pid':pid,'name':name,'cpu_cores':max(0,cpu-before[pid][1])/elapsed}
            for pid,(name,cpu) in after.items() if pid in before and pid not in exclude and pid not in (0,4)]
    return sum(r['cpu_cores'] for r in active),sorted(active,key=lambda r:-r['cpu_cores'])[:10]
def quiet_gate(child):
    quiet=0;windows=[]
    for _ in range(24):
        before=sample();begin=time.perf_counter();time.sleep(5)
        busy,active=activity(before,sample(),time.perf_counter()-begin,{child.pid,os.getpid()})
        windows.append({'external_cpu_cores':busy,'active':active})
        quiet=quiet+1 if busy<.35 else 0
        if quiet>=3:
            return {'available':True,'time':dt.datetime.now().astimezone().isoformat(),'windows':windows,'threshold_cpu_cores':.35}
    return {'available':False,'time':dt.datetime.now().astimezone().isoformat(),'windows':windows,'threshold_cpu_cores':.35,
            'reason':'No three-consecutive-window quiet interval in120sec; quality measured without speed samples'}
try:
    save()
    for bits in (2,1):
        for dataset in ('msong','msong_random','msong_random_value_disjoint'):
            label=f'{dataset}_ann_dense_b{bits}'
            target=OUT/'results'/(label+'.json')
            if target.exists() and json.loads(target.read_text()).get('status')=='complete':
                continue
            with (OUT/(label+'.log')).open('a') as log,(OUT/(label+'.workload.jsonl')).open('a') as telemetry:
                child=subprocess.Popen([sys.executable,'-u',str(OUT/'bench_ann_dense.py'),'--dataset',dataset,
                    '--bits',str(bits),'--output',str(target)],cwd=OUT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(status='running',current={'label':label,'pid':child.pid,'output':str(target)})
                save();print('STARTED',label,child.pid,flush=True)
                request_path=target.with_suffix('.gate-request.json')
                before=sample();begin=time.perf_counter()
                while child.poll() is None:
                    if request_path.exists():
                        request=json.loads(request_path.read_text());gate=Path(request['gate'])
                        if not gate.exists() and f'.gate-{child.pid}-' in gate.name:
                            state['status']='quiet_gate';save()
                            result=quiet_gate(child)
                            gate.write_text(json.dumps(result,indent=2))
                            state['status']='running';save()
                    time.sleep(2)
                    now=time.perf_counter();after=sample()
                    busy,active=activity(before,after,now-begin,{child.pid,os.getpid()})
                    telemetry.write(json.dumps({'time':dt.datetime.now().astimezone().isoformat(),
                        'external_cpu_cores':busy,'above_idle_threshold':busy>.35,'active':active})+'\n');telemetry.flush()
                    before,begin=after,now
            if child.returncode:
                raise RuntimeError(f'{label} exited{child.returncode}; inspect log')
            state['completed'].append(label);save()
    state.update(status='complete',current=None);save()
    # Existing approved flat/random/other-data queue remains sequential afterward.
    env['AMUMU_PIN_PCORE']='0'
    with (OUT/'queue-after-dense.log').open('a') as log:
        subprocess.Popen([sys.executable,'-u',str(OUT/'run_queue.py')],cwd=OUT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
except BaseException as exc:
    state.update(status='failed',error=repr(exc));save();raise
