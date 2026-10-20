"""Archive all positive/neutral/negative bounded quality outcomes, no speed claims."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
spec = importlib.util.spec_from_file_location('helpers',ROOT/'benchs/bench_rht.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
raw = OUT/'msong-quality-audit.json'
data = json.loads(raw.read_text())
assert data['status']=='complete'
assert len(data['fixed_t_rows'])==216 and len(data['native_query_rows'])==90
groups = {}
for row in data['fixed_t_rows']:
    key = tuple(row[k] for k in ('centers','quantizer','bits','mode','rounds'))
    assert row['seed'] not in groups.setdefault(key,{})
    groups[key][row['seed']] = row
assert len(groups)==108
rows = []
for key, by_seed in sorted(groups.items()):
    assert sorted(by_seed)==[202605170,202605171]
    center,quantizer,bits,mode,rounds=key
    reference = groups[(center,quantizer,bits,mode,1)]
    scores = np.array([by_seed[s]['recall_by_query'] for s in sorted(by_seed)])
    refs = np.array([reference[s]['recall_by_query'] for s in sorted(reference)])
    rows.append({'centers':center,'quantizer':quantizer,'bits':bits,'mode':mode,'rounds':rounds,
        'recall100':float(scores.mean()),'recall100_by_seed':[by_seed[s]['recall100'] for s in sorted(by_seed)],
        'delta_vs_1rht':helpers.paired_bootstrap(scores,refs,1000,9876),
        'delta_by_seed':(scores-refs).mean(axis=1).tolist(),
        'unbiased_nmse':float(np.mean([r['unbiased_nmse'] for r in by_seed.values()])),
        'unbiased_vnmse':float(np.mean([r['unbiased_vnmse'] for r in by_seed.values()])),
        'best_scalar_vnmse':float(np.mean([r['best_scalar_vnmse'] for r in by_seed.values()]))})
native = {}
for row in data['native_query_rows']:
    key = (row['bits'],row['rounds'],row['qb'])
    native.setdefault(key,{})[row['seed']] = row
assert len(native)==18
pilot=json.loads((OUT/'results/msong_native_k100_pilot.json').read_text())
pilot_rows={(r['bits'],int(r['mode'][-1]),r['seed']):r for r in pilot['rows']
            if r['index']=='RaBitQ' and r['mode'].startswith('rht') and r['qb']==4}
assert len(pilot_rows)==45
pilot_max_difference=max(float(np.max(np.abs(np.array(r['recall_by_query'])-
    np.array(pilot_rows[(r['bits'],r['rounds'],r['seed'])]['recall_by_query']))))
    for r in data['native_query_rows'] if r['qb']==4)
assert pilot_max_difference==0, ('qb4 replay differed from frozen pilot',pilot_max_difference)
query_rows=[]
for bits in (2,3,4):
    for rounds in (1,2,3):
        zero,four=native[(bits,rounds,0)],native[(bits,rounds,4)]
        assert sorted(zero)==sorted(four)==list(range(12345,12350))
        assert all(zero[s]['codes_sha256']==four[s]['codes_sha256'] for s in zero)
        a=np.array([zero[s]['recall_by_query'] for s in sorted(zero)])
        b=np.array([four[s]['recall_by_query'] for s in sorted(four)])
        query_rows.append({'bits':bits,'rounds':rounds,'qb0_recall100':float(a.mean()),'qb4_recall100':float(b.mean()),
            'paired_qb0_minus_qb4':helpers.paired_bootstrap(a,b,1000,9876),
            'delta_by_seed':(a-b).mean(axis=1).tolist(),'identical_stored_codes':True})
summary={'status':'complete','raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
    'purpose':data['purpose'],'no_speed_claims':True,'shapes':data['shapes'],
    'limitations':data['historical_mismatch_remaining']+[data['historical_helper_caveat'],
        '128 pilot centers use frozen native Faiss Lloyd training, not historical Torch Lloyd.',
        'NumPy diagnostic signs are not historical Torch/GPU RNG replay.',
        'Two transform seeds only in geometry/calibration diagnostic; intervals are exploratory.',
        'Native per-residual factor includes its code/float32 conventions; do not attribute differences solely to t.'],
    'qb4_replay_validation':{'cases':45,'max_per_query_recall_difference_vs_original_pilot':pilot_max_difference},
    'historical_source_manifest':data['historical_source_manifest'],
    'replayed_historical_calibration':data['replayed_historical_calibration'],
    'corrected_fixed_t_448':data['corrected_fixed_t_448'],
    'calibration_counterexample':data['calibration_counterexample'],
    'transform_checks':data['transform_checks'],'factor_summaries':rows,'native_query_summaries':query_rows}
summary_path=OUT/'msong-quality-audit-summary.json'
summary_path.write_text(json.dumps(summary,indent=2))
bundle=OUT/'msong-quality-audit-evidence.tar.xz'
paths=[raw,summary_path,OUT/'audit_msong_quality.py',Path(__file__),OUT/'msong-audit-plan.json',
       OUT/'build-frozen.json',OUT/'pilot-data/msong/manifest.json']
if (OUT/'resume_query_audit.py').exists():
    paths.append(OUT/'resume_query_audit.py')
paths += sorted(p for p in (OUT/'paper-source').rglob('*') if p.is_file())
with tarfile.open(bundle,'w:xz',preset=9) as archive:
    for path in paths:
        name = str(path.relative_to(OUT)) if path.is_relative_to(OUT) else path.name
        archive.add(path,arcname=name)
compressed=bundle.read_bytes()
(OUT/'msong-quality-audit-evidence.tar.xz.base64').write_text(base64.b64encode(compressed).decode('ascii'))
transfer={'status':'complete','raw_sha256':summary['raw_sha256'],'compressed_sha256':hashlib.sha256(compressed).hexdigest(),
    'compressed_bytes':len(compressed),'factor_rows':216,'native_query_rows':90,'factor_summaries':108,
    'no_speed_claims':True,'historical_replication':False}
(OUT/'msong-quality-audit-transfer.json').write_text(json.dumps(transfer,indent=2))
print(json.dumps(transfer,indent=2),flush=True)
