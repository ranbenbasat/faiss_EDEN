"""Dense worker was stopped only at a closed quiet gate; preserve checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

OUT=Path(__file__).resolve().parent
psutil.Process().cpu_affinity([2])
env=os.environ.copy();env.update(AMUMU_PIN_PCORE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',FAISS_OPT_LEVEL='AVX2')
state={'status':'running','supervisor_pid':os.getpid(),'completed':[],
       'dense_checkpoint':'case1 completed, paused at case2/pass0 closed gate; caches and rows retained'}
def save():
    (OUT/'geometry-control-status.json').write_text(json.dumps(state,indent=2))
try:
    save()
    for script in ('audit_equal512_geometry.py','audit_pairwise_layer.py','export_geometry_controls.py'):
        with (OUT/(Path(script).stem+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(OUT/script)],cwd=OUT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(current={'script':script,'pid':child.pid});save()
            if child.wait():
                raise RuntimeError(f'{script} failed; dense measurement remains stopped')
        state['completed'].append(script);save()
    env['AMUMU_PIN_PCORE']='0'
    with (OUT/'dense-queue.log').open('a') as log,(OUT/'dense-queue.err').open('a') as err:
        parent=subprocess.Popen([sys.executable,'-u',str(OUT/'run_dense_queue.py')],cwd=OUT.parent,env=env,stdout=log,stderr=err)
    state.update(status='complete_dense_resumed',current={'supervisor_pid':parent.pid});save()
    print('DENSE_RESUMED',parent.pid,flush=True)
except BaseException as exc:
    state.update(status='failed',error=repr(exc));save();raise
