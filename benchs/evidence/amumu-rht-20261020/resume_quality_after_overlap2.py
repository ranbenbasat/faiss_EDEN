"""Resume original finite quality matrix, skipping completed case checkpoints."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

OUT=Path(__file__).resolve().parent
psutil.Process().cpu_affinity(list(range(psutil.cpu_count())))
TARGET=OUT/'other-datasets-full-quality-resumed.json'
assert not TARGET.exists()
state={'status':'running','pid':os.getpid(),'reason':'Resume after bounded requested overlap two-scale recall/speed experiment','completed_invocations':[]}
def save():
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(TARGET)
env=os.environ.copy();env.update(AMUMU_PIN_PCORE='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
for bits in (2,3,4):
    for dataset in ('word2vec','gist','sift'):
        result=OUT/f'{dataset}-full-rht-quality.json'
        if result.exists():
            rows=json.loads(result.read_text())['rows']
            if sum(r['bits']==bits for r in rows)==30:continue
        label=f'{dataset}-full-rht-b{bits}-resumed'
        with (OUT/(label+'.log')).open('w') as log,(OUT/(label+'.err')).open('w') as error:
            child=subprocess.Popen([sys.executable,'-u',str(OUT/'quality_other_datasets_full.py'),'--dataset',dataset,'--bits',str(bits)],
                cwd=OUT.parent,env=env,stdout=log,stderr=error)
            state.update(current={'dataset':dataset,'bits':bits,'worker_pid':child.pid,'log':str(OUT/(label+'.log')),
                'started':datetime.datetime.now().astimezone().isoformat()});save()
            code=child.wait()
        state['completed_invocations'].append({**state['current'],'exit_code':code});save()
        if code:state['status']='failed';save();raise RuntimeError(label)
state.update(status='complete',current=None);save()
