"""Compact verified reproducibility capsule, no vectors/binaries/large caches."""
import base64
import hashlib
import json
import tarfile
from pathlib import Path

OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
raw=json.loads((OUT/'overlap2scale-quality.json').read_text());assert raw['status']=='complete'
speed=json.loads((OUT/'overlap2scale-speed.json').read_text())
assert speed['status']!='measuring','Do not export mutable timing evidence'
metadata={'quality_sha256':sha(OUT/'overlap2scale-quality.json'),'speed_sha256':sha(OUT/'overlap2scale-speed.json'),
    'dataset_npz_sha256':sha(OUT/'pilot-data/msong/vectors.npz'),
    'repository_helper_bench_rht_sha256':sha(OUT.parent/'benchs/bench_rht.py'),
    'native_dll_sha256':sha(OUT/'experimental-kernel/overlap2scale.dll'),
    'compiler_flags':'GCC16.2UCRT -O3 -DNDEBUG -march=native -ffp-contract=off -std=c++17 -shared; nofastmath; includes unchanged experimental full512 and overlap256 controls; links frozen libfaiss_avx2.dll',
    'frozen_binaries':{p:sha(OUT/p) for p in ('runtime/faiss/_swigfaiss_avx2.pyd','../build-amumu/faiss/libfaiss_avx2.dll')},
    'large_caches':'Prototype code arrays and native indexes retained locally, perfilehashes/paths in qualityraw; not embedded. Compact format not integrated production FAISS serialization.',
    'storage':'Two-scale candidate117/170/222B (scalar420bits*b +2FP32scales+1originalFP32norm). EDENactualnativebaseline136/200/264B. RaBitQactualnativebaseline148/212/276B, qb0. Compactfull512 control136/200/264B.',
    'attribution':raw['attribution'],'RaBitQ_caveat':raw['RaBitQ_caveat']}
assert metadata['frozen_binaries']['runtime/faiss/_swigfaiss_avx2.pyd']=='10ed67de00233a013af049273950f2eccfdc9dd41dadd41b3947bcec1439d9f6'
assert metadata['frozen_binaries']['../build-amumu/faiss/libfaiss_avx2.dll']=='8ce27795db58f84d7d25685263b5010c1c35d3466028411886162259a860aace'
(OUT/'overlap2scale-provenance.json').write_text(json.dumps(metadata,indent=2))
names=['experimental-kernel/overlap2scale.cpp','experimental-kernel/overlap256.cpp','experimental-kernel/block_transform_fixed64.cpp',
       'native_overlap2scale.py','native_overlap256.py','experiment_overlap2scale.py','validate_overlap2scale.py',
       'run_overlap2scale_experiment.py','pause_quality_for_overlap2.py','resume_quality_after_overlap2.py',
       'summarize_overlap2scale.py','export_overlap2scale.py','audit_msong_quality.py',
       'overlap2scale-validation.json','overlap256-validation.json','overlap2scale-quality.json','overlap2scale-speed.json',
       'overlap2scale-summary.json','overlap2scale-workload.json','overlap2scale-provenance.json',
       'overlap2scale-quality.log','overlap2scale-quality.err','overlap2scale-speed.log','overlap2scale-speed.err',
       'pilot-data/msong/manifest.json','overlap2scale-cache/shared.npz']
manifest={n:{'bytes':(OUT/n).stat().st_size,'sha256':sha(OUT/n)} for n in names}
archivepath=OUT/'overlap2scale-evidence.tar.xz'
with tarfile.open(archivepath,'w:xz') as tar:
    for n in names:tar.add(OUT/n,arcname=n)
with tarfile.open(archivepath,'r:xz') as tar:
    for n in names:assert hashlib.sha256(tar.extractfile(n).read()).hexdigest()==manifest[n]['sha256']
encoded=OUT/(archivepath.name+'.base64');encoded.write_text(base64.b64encode(archivepath.read_bytes()).decode())
transfer={'status':'complete','capsule_sha256':sha(archivepath),'capsule_bytes':archivepath.stat().st_size,
          'base64_characters':encoded.stat().st_size,'files':manifest}
(OUT/'overlap2scale-transfer.json').write_text(json.dumps(transfer,indent=2))
print(json.dumps({k:v for k,v in transfer.items() if k!='files'},indent=2))
