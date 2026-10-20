"""Summarize and hash independent canaries; preserve raw and frozen files."""
import base64
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
RAW=OUT/'fixed64-independent-quantizer-canaries.json'
raw=json.loads(RAW.read_text());assert raw['status']=='complete'
assert len(raw['gaussian'])==120 and len(raw['basis'])==18 and len(raw['msong_group0'])==30
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary={'status':'complete','raw_sha256':digest(RAW),'source_sha256':raw['source_sha256'],
    'purpose':raw['purpose'],'independence':raw['independence'],'objectives':raw['objectives'],
    'tables':raw['tables'],'gaussian_paired_summary':raw['gaussian_paired_summary'],
    'basis_summary':[],'msong_group0_summary':[],
    'no_broad_matrix_or_speed_or_OS_changes':True,
    'limitations':['Bounded canaries do not prove absence of every implementation issue.',
        'Gaussian-invariance control does not prove paper and fixed64 results are identical experiments.',
        'MSong check covers only dominant first64 group, all20k base vectors, five seeds.',
        'Independent dense centering uses FP64 training mean while archived native uses FP32 centering; comparison explicitly reports finite precision differences.']}
for bits in (2,3,4):
    for fmt in ('fp32','bf16'):
        for rounds in (1,2):
            basis=[r[fmt] for r in raw['basis'] if (r['bits'],r['rounds'])==(bits,rounds)]
            summary['basis_summary'].append(dict(bits=bits,format=fmt,rounds=rounds,
                native_nmse_mean=float(np.mean([r['native_nmse'] for r in basis])),
                independent_exact_nmse_mean=float(np.mean([r['independent_exact_nmse'] for r in basis])),
                native_nmse_by_seed=[r['native_nmse'] for r in basis]))
            msong=[r[fmt] for r in raw['msong_group0'] if (r['bits'],r['rounds'])==(bits,rounds)]
            summary['msong_group0_summary'].append(dict(bits=bits,format=fmt,rounds=rounds,
                **{f'{name}_{n}':float(np.mean([r[f'{name}_{n}'] for r in msong])) for n in (10000,20000) for name in ('native_residual_pooled_nmse','independent_dense_pooled_nmse')},
                max_abs_independent_vs_native_pooled_delta=max(abs(r['native_residual_pooled_nmse_20000']-r['independent_dense_pooled_nmse_20000']) for r in msong),
                assignment_mismatches=sum(r['assignment_mismatches'] for r in msong),
                assignment_total=sum(r['assignment_total'] for r in msong),
                far_threshold_mismatches=sum(r['mismatches_farther_than5e-5_from_threshold'] for r in msong)))
allrows=[r[fmt] for group in ('gaussian','basis','msong_group0') for r in raw[group] for fmt in ('fp32','bf16')]
summary['total_assignment_checks']=sum(r['assignment_total'] for r in allrows)
summary['assignment_mismatches']=sum(r['assignment_mismatches'] for r in allrows)
summary['mismatches_farther_than5e-5_from_threshold']=sum(r['mismatches_farther_than5e-5_from_threshold'] for r in allrows)
assert summary['mismatches_farther_than5e-5_from_threshold']==0
summary['all_global_norms_match_independent_fp32_accumulation']=all(r['whole_norm_matches_independent_fp32_accumulation'] for r in allrows)
summary['max_fp32_same_assignment_relative_scale_error']=max(v for group in ('gaussian','basis','msong_group0') for r in raw[group] if (v:=r['fp32']['max_same_assignment_scale_relative_error']) is not None)
summary['tie_policy_note']='Brute force argmin chooses lower-index centroid for exact equal-distance ties; native upper_bound chooses upper interval. Basis two-round outputs include exact zeros, hence assignment mismatches at threshold0 without error/scale disagreement. All mismatches farther than5e-5 from an independent threshold are rejected.'
expected={'experimental-kernel/eden_group_fixed64.dll':'875e2e701f56b049b828017e1d2092e77fb15b5b093694b17d504bb5b7209ab0',
          'runtime/faiss/_swigfaiss_avx2.pyd':'10ed67de00233a013af049273950f2eccfdc9dd41dadd41b3947bcec1439d9f6'}
summary['frozen_binary_checks']={name:{'sha256':digest(OUT/name),'matches_expected':digest(OUT/name)==value} for name,value in expected.items()}
assert all(r['matches_expected'] for r in summary['frozen_binary_checks'].values())
target=OUT/'fixed64-independent-quantizer-summary.json';target.write_text(json.dumps(summary,indent=2))
names=['canary_fixed64_quantizer_independent.py','export_fixed64_independent_canaries.py',RAW.name,target.name,
       'independent-quantizer-canaries.log','independent-quantizer-canaries.err']
manifest={n:{'sha256':digest(OUT/n),'bytes':(OUT/n).stat().st_size} for n in names}
capsule=OUT/'fixed64-independent-quantizer-evidence.tar.xz'
with tarfile.open(capsule,'w:xz') as archive:
    for n in names:archive.add(OUT/n,arcname=n)
encoded=OUT/(capsule.name+'.base64');encoded.write_text(base64.b64encode(capsule.read_bytes()).decode())
transfer={'status':'complete','raw_sha256':digest(RAW),'capsule_sha256':digest(capsule),
          'capsule_bytes':capsule.stat().st_size,'base64_characters':encoded.stat().st_size,'files':manifest}
(OUT/'fixed64-independent-quantizer-transfer.json').write_text(json.dumps(transfer,indent=2))
with tarfile.open(capsule,'r:xz') as archive:
    for n in names:assert hashlib.sha256(archive.extractfile(n).read()).hexdigest()==manifest[n]['sha256']
print(json.dumps(summary,indent=2));print('TRANSFER',json.dumps(transfer))
