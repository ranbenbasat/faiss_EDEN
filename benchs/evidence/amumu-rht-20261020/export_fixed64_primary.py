"""Publish corrected exact-stride primary speed as soon as15 paired settings finish."""
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import audit_msong_quality as helpers

OUT=Path(__file__).resolve().parent
raw=OUT/'native-eden-fixed64-primary-benchmark.json'
all_bytes=raw.read_bytes();data=json.loads(all_bytes)
assert 'primary' in data['completed_stages'] and data['corrected_no_permutation']
rows=[r for r in data['rows'] if r['stage']=='primary']
assert len(rows)==90
quality=json.loads((OUT/'native-eden-fixed64-flat-quality.json').read_text());assert quality['status']=='complete'
grouped={}
for r in rows:
    key=(r['mode'],r['bits']);pair=(r['seed'],r['repeat'])
    assert pair not in grouped.setdefault(key,{})
    grouped[key][pair]=r
snapshot={'status':'primary_complete','corrected_no_permutation':True,'source_snapshot_sha256':hashlib.sha256(all_bytes).hexdigest(),
    'source_overall_status':data['status'],'source_overall_stage':data['stage'],'rows':rows,
    'dll_sha256':data['dll_sha256'],'clock':data['clock'],'strategies':data['strategies'],'transform_strategies':data['transform_strategies']}
raw_primary=OUT/'native-eden-fixed64-primary-results.json';raw_primary.write_text(json.dumps(snapshot,indent=2))
telemetry=[json.loads(line) for line in (OUT/'eden-fixed64-runtime.workload.jsonl').read_text().splitlines()]
busy=sum(r['above_idle_threshold'] for r in telemetry)
all_gates=all(r['quiet_gate']['available'] for r in rows)
diagnostic=not all_gates or busy>0
report={'status':'bounded_primary_complete','corrected_no_permutation':True,'scope':'420pad448; seven FIXED64 groups; two independent-sign normalizedH64 rounds WITHIN groups; NO cross-group mixing/permutation',
    'primary_baseline':'ACTUAL frozen FAISS EDEN Index::search plus independently optimized full512 ONE-RHT preprocessing, bitwise frozen-HR equivalent',
    'separate_control':'custom global512_uniform scorer is NOT substituted for the actual-FAISS primary comparator',
    'k':100,'batch':256,'shapes':{'xt':[4096,420],'xb':[20000,420],'xq':[200,420]},
    'seed_count':5,'passes':2,'native_repeats_per_setting_pass':5,'paired_settings':15,
    'raw_primary_sha256':hashlib.sha256(raw_primary.read_bytes()).hexdigest(),'dll_sha256':data['dll_sha256'],
    'native_timing_measured':True,'timing_diagnostic_nonquiet':diagnostic,'validated_uncontended_speedup_claims_allowed':not diagnostic,
    'workload':{'samples_at_export':len(telemetry),'busy_samples':busy,'all_primary_gates_passed':all_gates,'threshold_cpu_cores':.35},
    'primary_six_rows':[],'compact_secondary':[],
    'limits':['prefix20k and200 queries on one dataset','recall intervals paired seed/query, conditional on fixed dataset',
        'timing intervals paired seed/pass; do NOT remove load-related bias','all noise flags retained; no uncontended speed claim on gate failure',
        'NMSE uses original whole residual norms; not a mean of equally weighted group relative errors','first10k reconstruction population paired with prior runner'],
    'routing_validation':json.loads((OUT/'native-fixed64-pipeline-mapping-validation.json').read_text())}
for (mode,bits),records in sorted(grouped.items()):
    assert len(records)==10
    baseline=grouped[('frozen_full512',bits)];seeds=list(range(12345,12350))
    a=np.array([records[(s,0)]['recall_by_query'] for s in seeds]);b=np.array([baseline[(s,0)]['recall_by_query'] for s in seeds])
    ratios=np.array([[baseline[(s,p)]['seconds_median']/records[(s,p)]['seconds_median'] for p in (0,1)] for s in seeds])
    rng=np.random.default_rng(9876);draws=[]
    for _ in range(1000):
        indices=rng.integers(0,5,5);passes=rng.integers(0,2,(5,2))
        draws.append(float(np.exp(np.mean([np.log(ratios[s,passes[i,j]]) for i,s in enumerate(indices) for j in range(2)]))))
    qr=[r for r in quality['rows'] if r['mode']==mode and r['bits']==bits];assert len(qr)==5
    err=[r['reconstruction_10k'] for r in qr]
    if mode=='group_bf16_samestride':
        for r in qr:
            with np.load(r['cache_path']) as cache:
                assert cache['codes'].shape==(20000,{2:136,3:200,4:264}[bits])
                assert cache['codes'].nbytes==20000*{2:136,3:200,4:264}[bits]
    result={'mode':mode,'bits':bits,'actual_code_bytes_per_vector':records[(12345,0)]['code_bytes_per_vector'],
        'hard_cap_bytes':records[(12345,0)]['cap'],'within_cap':True,
        'recall100':float(a.mean()),'recall_delta_vs_actual_FAISS':helpers.helpers.paired_bootstrap(a,b,1000,9876),
        'pooled_nmse_10k':sum(m['squared_error_sum'] for m in err)/sum(m['residual_squared_norm_sum'] for m in err),
        'mean_vector_vnmse_10k':float(np.mean([m['vnmse'] for m in err])),
        'qps_median':float(np.median([r['qps'] for r in records.values()])),
        'paired_measured_speedup_vs_actual_FAISS':float(np.exp(np.mean(np.log(ratios)))),
        'paired_speedup_seed_pass_ci95':np.quantile(draws,[.025,.975]).tolist(),
        'paired_speedup_by_seed':np.exp(np.mean(np.log(ratios),axis=1)).tolist(),'diagnostic_nonquiet':diagnostic}
    if mode=='group_bf16_compact':report['compact_secondary'].append(result)
    else:report['primary_six_rows'].append(result)
freeze=json.loads((OUT/'build-frozen.json').read_text())
native_files=[OUT/'runtime/faiss/_swigfaiss_avx2.pyd',OUT.parent/'build-amumu/faiss/libfaiss_avx2.dll']
report['frozen_FAISS_binary_checks']=[]
for file in native_files:
    actual=hashlib.sha256(file.read_bytes()).hexdigest();expected=freeze['hashes'][str(file)]
    assert actual==expected
    report['frozen_FAISS_binary_checks'].append({'file':str(file),'sha256':actual,'unchanged_from_frozen_build':True})
report['build_provenance']={'flags':['-O3','-DNDEBUG','-march=native','-ffp-contract=off','-std=c++17','-shared','-I workspace-root'],
    'linked_import_library':'build-amumu/faiss/libfaiss_avx2.dll.a',
    'source_hashes':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
        ['experimental-kernel/block_transform_fixed64.cpp','experimental-kernel/eden_group.cpp','experimental-kernel/eden_group_search.cpp']}}
report['all_native_flat_rows_at_export']=len(data['rows'])
report['overall_native_flat_run_complete']=data['status']=='complete'
report['attribution_controls']=[]
for mode in ('global512_uniform','global448','group_fp32_reference'):
    for bits in (2,3,4):
        controls=[r for r in data['rows'] if r['stage']=='attribution' and r['mode']==mode and r['bits']==bits]
        if len(controls)!=10:continue
        records={(r['seed'],r['repeat']):r for r in controls};base=grouped[('frozen_full512',bits)]
        ratios=[base[p]['seconds_median']/records[p]['seconds_median'] for p in sorted(records)]
        report['attribution_controls'].append({'mode':mode,'bits':bits,'code_bytes':controls[0]['code_bytes_per_vector'],
            'within_cap':all(r['within_cap'] for r in controls),'recall100':float(np.mean([r['recall'] for r in controls])),
            'qps_median':float(np.median([r['qps'] for r in controls])),
            'paired_measured_ratio_vs_actual_FAISS':float(np.exp(np.mean(np.log(ratios)))),'diagnostic_nonquiet':diagnostic})
report['group_vs_uniform_global_control']=[]
for bits in (2,3,4):
    control=[r for r in data['rows'] if r['stage']=='attribution' and r['mode']=='global512_uniform' and r['bits']==bits]
    if len(control)!=10:continue
    reference={(r['seed'],r['repeat']):r for r in control};group=grouped[('group_bf16_samestride',bits)]
    ratios=[reference[p]['seconds_median']/group[p]['seconds_median'] for p in sorted(group)]
    report['group_vs_uniform_global_control'].append({'bits':bits,'paired_measured_ratio':float(np.exp(np.mean(np.log(ratios)))),
        'scope':'same optimized scan kernel; still different geometry and group scales, diagnostic nonquiet',
        'not_the_primary_actual_FAISS_comparator':True})
report['attribution_warning']='Do not attribute the entire3-bit ratio to BF16/group scales: frozen FAISS3-bit scalar decoding differs from the separately measured common optimized SIMD scanner.'
transform_path=OUT/'native-fixed64-transform-benchmark.json'
report['transform_only']=[]
if transform_path.exists():
    td=json.loads(transform_path.read_text())
    if td['status']=='complete':
        assert len(td['rows'])==80 and td['corrected_no_permutation']
        for n in (1,32,256,20000):
            full={(r['seed'],r['repeat']):r for r in td['rows'] if r['mode']=='full512' and r['n']==n}
            small={(r['seed'],r['repeat']):r for r in td['rows'] if r['mode']=='block64' and r['n']==n}
            assert len(full)==len(small)==10
            ratios=[full[p]['ns_per_vector_median']/small[p]['ns_per_vector_median'] for p in sorted(full)]
            report['transform_only'].append({'n':n,'full512_ns_per_vector_median':float(np.median([r['ns_per_vector_median'] for r in full.values()])),
                'fixed64_two_ns_per_vector_median':float(np.median([r['ns_per_vector_median'] for r in small.values()])),
                'paired_ratio_full_over_fixed':float(np.exp(np.mean(np.log(ratios)))),
                'allocation_free_native_kernel':True,'all_gates_passed':all(r['quiet_gate']['available'] for r in td['rows']),
                'diagnostic_nonquiet':any(not r['quiet_gate']['available'] for r in td['rows'])})
report['storage_scope']={'flat_ID_bytes':0,'flat_ID_rule':'implicit row IDs',
    'actual_grouped_code_buffer_bytes_by_bits':{str(b):20000*{2:136,3:200,4:264}[b] for b in (2,3,4)},
    'reserved_grouped_padding_bytes_by_bits':{'2':6,'3':14,'4':22},
    'transform_sign_buffer_bytes':{'fixed448':5376,'full512':6144},'permutation_buffer_bytes':0,
    'serialization':'Prototype NPZ code+centroid caches plus manifest, not a production integrated FAISS serialization; frozen baseline .index files retained locally. No IVF IDs/list padding in this flat experiment.'}
path=OUT/'native-eden-fixed64-primary-summary.json';path.write_text(json.dumps(report,indent=2))
bundle=OUT/'native-eden-fixed64-primary-evidence.tar.xz'
files=[raw_primary.name,path.name,'native-eden-fixed64-flat-quality.json','native-fixed64-validation.json',
    'native-fixed64-pipeline-mapping-validation.json','native-block64-fixed64-tuning.json','native-eden-fixed64-tuning.json',
    'native-eden-group-validation.json','native-eden-group-optimized-validation.json','test_fixed64_transform.py','fixed64-transform-tests.log',
    'build-frozen.json','EXPERIMENT_SCOPE.json','bench_eden_fixed64_primary.py','run_eden_fixed64_runtime.py',Path(__file__).name,
    'prepare_eden_fixed64_native_flat.py','native_block64_fixed64.py','native_eden_search_fixed64.py','native_eden_group.py',
    'validate_fixed64_native.py','validate_fixed64_pipeline_mapping.py','eden_group_bf16.py','test_eden_group_bf16.py','experimental_block32.py',
    'experimental-kernel/block_transform_fixed64.cpp','experimental-kernel/eden_group.cpp','experimental-kernel/eden_group_search.cpp',
    'eden-fixed64-runtime.workload.jsonl','bench-eden-fixed64-primary.log']
if data['status']=='complete':files.append(raw.name)
if report['transform_only']:files.extend(['native-fixed64-transform-benchmark.json','fixed64-transform.workload.jsonl',
    'bench_fixed64_transform_only.py','run_fixed64_transform_finish.py','bench-fixed64-transform.log'])
with tarfile.open(bundle,'w:xz',preset=9) as archive:
    for name in files:archive.add(OUT/name,arcname=name)
compressed=bundle.read_bytes();(OUT/'native-eden-fixed64-primary-evidence.tar.xz.base64').write_text(base64.b64encode(compressed).decode('ascii'))
transfer={'status':'bounded_primary_complete','raw_primary_sha256':report['raw_primary_sha256'],
    'compressed_sha256':hashlib.sha256(compressed).hexdigest(),'compressed_bytes':len(compressed),
    'primary_timing_rows':90,'corrected_no_permutation':True,'diagnostic_nonquiet':diagnostic}
(OUT/'native-eden-fixed64-primary-transfer.json').write_text(json.dumps(transfer,indent=2))
for r in report['primary_six_rows']:print('PRIMARY',r['mode'],r['bits'],r['actual_code_bytes_per_vector'],r['recall100'],r['pooled_nmse_10k'],r['qps_median'],r['paired_measured_speedup_vs_actual_FAISS'],flush=True)
print(json.dumps(transfer,indent=2),flush=True)
