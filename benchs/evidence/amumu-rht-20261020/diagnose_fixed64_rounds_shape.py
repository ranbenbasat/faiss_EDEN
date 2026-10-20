"""Post-hoc descriptive shape/error associations; no causal or speed claim."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json
from pathlib import Path
import numpy as np
import faiss
from native_block64_fixed64 import NativeTransform

OUT=Path(__file__).resolve().parent
DETAIL=OUT/'fixed64-rounds-archivedcenter-detail'

def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def shape(residual):
    square=residual.reshape(-1,7,64)**2
    energy=square.sum(axis=2)
    kurtosis=64*(square**2).sum(axis=2)/energy**2
    concentration=square.max(axis=2)/energy
    return energy,kurtosis,concentration

def weighted(v,w):return float((v*w).sum()/w.sum())

def main():
    raw=json.loads((OUT/'fixed64-rounds-archivedcenter-quality.json').read_text())
    assert raw['status']=='complete'
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb')]
    leaf=faiss.index_factory(420,'EDEN2');leaf.train(xt)
    center=faiss.vector_to_array(leaf.center)
    energy0,k0,c0=shape(np.pad(xb.astype('float64')-center,((0,0),(0,28))))
    report={'status':'complete','purpose':'POST-HOC DESCRIPTIVE ONLY: normalized kurtosis and dominant-coordinate energy vs paired error changes',
        'definition':'kurtosis=64*sum(x^4)/sum(x^2)^2; uniform absolute magnitudes=1, Gaussian population near3. Energy weighting uses original group residual energy. Concentration=max(x^2)/sum(x^2).',
        'inference_limits':['Association is not causation; changing kurtosis is not proof of a particular mechanism.',
            'No independence/p-values claimed for correlated vectors/groups/seeds.',
            'Only fixed MSong prefix20k, five transform seeds; no additional dataset/transform settings.'],
        'input_energy_weighted_kurtosis':weighted(k0,energy0),
        'input_energy_weighted_dominant_coordinate_share':weighted(c0,energy0),'seeds':[],'groups':[]}
    for seed in range(12345,12350):
        shapes={}
        for rounds in (1,2):
            transform=NativeTransform('block64',seed,rounds,False,1)
            zt,zb=[transform.apply(v) for v in (xt,xb)]
            leaf=faiss.index_factory(448,'EDEN2');leaf.train(zt)
            zcenter=faiss.vector_to_array(leaf.center)
            shapes[rounds]=shape(zb.astype('float64')-zcenter)
        e1,k1,c1=shapes[1];e2,k2,c2=shapes[2]
        seedrow={'seed':seed,'one_energy_weighted_kurtosis':weighted(k1,energy0),
            'two_energy_weighted_kurtosis':weighted(k2,energy0),
            'one_energy_weighted_concentration':weighted(c1,energy0),
            'two_energy_weighted_concentration':weighted(c2,energy0),'associations':[]}
        for bits in (2,3,4):
            for fmt in ('bf16','fp32'):
                with np.load(DETAIL/f'b{bits}-s{seed}-r1-{fmt}.npz') as a, np.load(DETAIL/f'b{bits}-s{seed}-r2-{fmt}.npz') as b:
                    er1=a['group_squared_error'];er2=b['group_squared_error']
                    delta=er2/e2-er1/e1
                    seedrow['associations'].append({'bits':bits,'format':fmt,
                        'group_vnmse_delta_vs_kurtosis_delta_pearson':corr(delta,k2-k1),
                        'group_vnmse_delta_vs_input_concentration_pearson':corr(delta,c0),
                        'vector_vnmse_delta_vs_energyweighted_input_concentration_pearson':corr(
                            er2.sum(axis=1)/e2.sum(axis=1)-er1.sum(axis=1)/e1.sum(axis=1),(c0*energy0).sum(axis=1)/energy0.sum(axis=1))})
                    for g in range(7):
                        report['groups'].append({'seed':seed,'bits':bits,'format':fmt,'group':g,
                            'energy_fraction':float(energy0[:,g].sum()/energy0.sum()),
                            'input_kurtosis':weighted(k0[:,g],energy0[:,g]),
                            'one_kurtosis':weighted(k1[:,g],energy0[:,g]),
                            'two_kurtosis':weighted(k2[:,g],energy0[:,g]),
                            'input_concentration':weighted(c0[:,g],energy0[:,g]),
                            'error_sum_delta':float(er2[:,g].sum()-er1[:,g].sum())})
        report['seeds'].append(seedrow)
    target=OUT/'fixed64-rounds-shape-diagnostic.json';target.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))

if __name__=='__main__':
    faiss.omp_set_num_threads(1);main()
