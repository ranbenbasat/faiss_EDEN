"""Finite sequential quality queue:2bits across all datasets, then3/4bits.

This is a finite requested experiment, not a recurring scheduler/monitor.
One worker at a time; never stop unrelated jobs/services or change OS settings.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
TARGET=OUT/'other-datasets-full-quality-queue.json'
assert not TARGET.exists(),'Inspect previous queue before starting duplicates'
state={'status':'running','pid':os.getpid(),'started':datetime.datetime.now().astimezone().isoformat(),
       'quality_only':True,'planned_cases':270,'datasets':['word2vec','gist','sift'],
       'bits_order':[2,3,4],'seeds':list(range(202605170,202605175)),'rounds':[1,2,3],
       'families':['RaBitQ','EDEN'],'completed_invocations':[],'current':None}
def save():
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(TARGET)
env=os.environ.copy();env.update(AMUMU_PIN_PCORE='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',FAISS_OPT_LEVEL='AVX2',PYTHONUNBUFFERED='1')
save()
for bits in (2,3,4):
    for dataset in ('word2vec','gist','sift'):
        label=f'{dataset}-full-rht-b{bits}'
        command=[sys.executable,'-u',str(OUT/'quality_other_datasets_full.py'),'--dataset',dataset,'--bits',str(bits)]
        with (OUT/(label+'.log')).open('w') as log,(OUT/(label+'.err')).open('w') as error:
            worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=error)
            state['current']={'dataset':dataset,'bits':bits,'worker_pid':worker.pid,
                'log':str(OUT/(label+'.log')),'error':str(OUT/(label+'.err')),
                'started':datetime.datetime.now().astimezone().isoformat()}
            save();print('LAUNCHED',dataset,bits,worker.pid,str(OUT/(label+'.log')),flush=True)
            code=worker.wait()
        state['completed_invocations'].append({**state['current'],'exit_code':code,
            'finished':datetime.datetime.now().astimezone().isoformat()})
        if code:
            state['status']='failed';save();raise RuntimeError(f'{label} failed; inspect durable logs, preserve checkpoints')
        save()
state['status']='complete';state['current']=None;save();print('ALL_OTHER_DATASETS_COMPLETE',flush=True)
