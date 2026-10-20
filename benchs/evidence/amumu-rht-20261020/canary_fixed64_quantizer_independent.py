"""Bounded independent quality canaries; expected values never use native tables.

Expected quantizer: Gaussian integrals and Lloyd fixed point from quantile init.
Expected rotation: explicit parity H64 and a standalone specified SplitMix64.
Expected scale: independent dot products for unbiased and MSE-optimal objectives.
Native functions are comparison targets only. No timing or source/binary edits.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
import numpy as np
import faiss
from threadpoolctl import threadpool_limits
from native_block64_fixed64 import NativeTransform
from native_eden_group import GroupEDEN

OUT=Path(__file__).resolve().parent
TARGET=OUT/'fixed64-independent-quantizer-canaries.json'
H=np.array([[(-1. if (i&j).bit_count()%2 else 1.)/8 for j in range(64)] for i in range(64)])

def pdf(x):return math.exp(-x*x/2)/math.sqrt(2*math.pi) if math.isfinite(x) else 0.
def cdf(x):return .5*math.erfc(-x/math.sqrt(2))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def lloyd(bits):
    k=1<<bits
    levels=np.array([statistics.NormalDist().inv_cdf((i+.5)/k) for i in range(k)])
    for iteration in range(20000):
        boundary=(levels[:-1]+levels[1:])/2
        edges=np.r_[-np.inf,boundary,np.inf]
        proposed=np.array([(pdf(a)-pdf(b))/(cdf(b)-cdf(a)) for a,b in zip(edges[:-1],edges[1:])])
        proposed=(proposed-proposed[::-1])/2
        step=float(np.max(np.abs(proposed-levels)));levels=proposed
        if step<2e-14:break
    assert step<2e-14
    boundary=(levels[:-1]+levels[1:])/2
    edges=np.r_[-np.inf,boundary,np.inf]
    mse=0.
    for a,b,c in zip(edges[:-1],edges[1:],levels):
        p=cdf(b)-cdf(a);m1=pdf(a)-pdf(b)
        m2=p+(a*pdf(a) if math.isfinite(a) else 0)-(b*pdf(b) if math.isfinite(b) else 0)
        mse+=m2-2*c*m1+c*c*p
    return levels,boundary,{'iterations':iteration+1,'last_update':step,'gaussian_scalar_mse_integral':mse}

def signs(seed):
    # Independent implementation of the documented 3x448 SplitMix64 sign stream.
    state=int(seed);mask=(1<<64)-1;output=[]
    for _ in range(3*448):
        state=(state+0x9e3779b97f4a7c15)&mask
        value=state;value=((value^(value>>30))*0xbf58476d1ce4e5b9)&mask
        value=((value^(value>>27))*0x94d049bb133111eb)&mask
        value^=value>>31;output.append(1. if value&1 else -1.)
    return np.array(output).reshape(3,448)

def rotate64(x,seed,rounds):
    s=signs(seed);y=x.astype('float64',copy=True)
    for r in range(rounds):y=(y*s[r,:64])@H
    return y

def bfround(scale):
    u=np.asarray(scale,dtype='float32').view('uint32')
    high=((u.astype('uint64')+0x7fff+((u>>16)&1))>>16).astype('uint32')
    return (high<<16).view('float32')

def independent_quantize(x,levels,boundary):
    norm=np.sum(x*x,axis=1)
    unit=x*(8/np.sqrt(norm))[:,None]
    # Brute-force nearest centroid, not the code-under-test's threshold search.
    assignment=np.argmin((unit[:,:,None]-levels[None,None,:])**2,axis=2)
    q=levels[assignment]
    dot=np.sum(x*q,axis=1);qnorm=np.sum(q*q,axis=1)
    unbiased=norm/dot
    mse_optimal=dot/qnorm
    errors={}
    for name,scale in (('unbiased_exact',unbiased),('unbiased_fp32',unbiased.astype('float32')),
                        ('unbiased_bf16',bfround(unbiased)),('mse_optimal_exact',mse_optimal)):
        errors[name]=np.sum((x-q*scale[:,None])**2,axis=1)/norm
    assert np.all(errors['mse_optimal_exact']<=errors['unbiased_exact']+1e-14)
    return assignment,unbiased,errors,unit

def native_quantize(x,bits,fmt):
    z=np.zeros((len(x),448),dtype='float32');z[:,:64]=x
    codec=GroupEDEN(bits,scale_format=fmt);codes=codec.encode(z)
    payload=448*bits//8;groupsize=64*bits//8
    packed=np.unpackbits(codes[:,:groupsize],axis=1,bitorder='little').reshape(len(x),64,bits)
    assignments=np.sum(packed*(1<<np.arange(bits)),axis=2,dtype='int64')
    count=2 if fmt=='bf16' else 4
    stored=codes[:,payload:payload+count].copy()
    scale=((stored.view('<u2').astype('uint32')<<16).view('float32') if fmt=='bf16' else stored.view('<f4')).ravel()
    norm_offset=payload+(14 if fmt=='bf16' else 28)
    stored_norm=np.ascontiguousarray(codes[:,norm_offset:norm_offset+4]).view('<f4').ravel()
    reconstruction=codec.decode(codes)[:,:64]
    norm=np.sum(x.astype('float64')**2,axis=1)
    error=np.sum((x.astype('float64')-reconstruction)**2,axis=1)/norm
    return assignments,scale,stored_norm,error,codes

def compare(x,bits,levels,boundary):
    expected,scale,error,unit=independent_quantize(x.astype('float64'),levels,boundary)
    rows={};native={}
    for fmt in ('fp32','bf16'):
        assignment,stored,norm,observed,codes=native_quantize(x,bits,fmt)
        native[fmt]=(assignment,stored,observed,codes)
        mismatch=assignment!=expected
        margins=np.min(np.abs(unit[:,:,None]-boundary[None,None,:]),axis=2)
        # A finite input can land right on a threshold: record those separately.
        far=int(np.count_nonzero(mismatch & (margins>5e-5)))
        assert far==0,(bits,fmt,far)
        norm_f32=np.add.accumulate(np.asarray(x,dtype='float32')**2,axis=1,dtype='float32')[:,-1]
        np.testing.assert_array_equal(norm,norm_f32)
        same=np.all(~mismatch,axis=1)
        if fmt=='fp32' and np.any(same):np.testing.assert_allclose(stored[same],scale[same],rtol=2e-6,atol=1e-7)
        rows[fmt]={'native_nmse':float(np.mean(observed)),
            'independent_nmse':float(np.mean(error['unbiased_'+fmt])),
            'independent_exact_nmse':float(np.mean(error['unbiased_exact'])),
            'independent_mse_optimal_nmse':float(np.mean(error['mse_optimal_exact'])),
            'assignment_mismatches':int(mismatch.sum()),'assignment_total':int(mismatch.size),
            'mismatches_farther_than5e-5_from_threshold':far,
            'max_same_assignment_scale_relative_error':float(np.max(np.abs(stored[same]/scale[same]-1))) if same.any() else None,
            'whole_norm_matches_independent_fp32_accumulation':True}
    np.testing.assert_array_equal(native['fp32'][0],native['bf16'][0])
    np.testing.assert_array_equal(bfround(native['fp32'][1]),native['bf16'][1])
    return rows,native,error

def save(report):
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(TARGET)

def main():
    assert not TARGET.exists()
    report={'status':'running','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_sha256':sha(Path(__file__)),'native_dll_sha256':sha(OUT/'experimental-kernel/eden_group_fixed64.dll'),
        'purpose':'Independent quality-only canaries, no timing or frozen source/binary modifications',
        'independence':'Expected centroids/boundaries from Gaussian pdf/cdf integrals and Lloyd iteration, expected nearest assignments brute force; expected transform explicit parity matrix and independently generated signs; native tables/transform/codec used only as comparison targets.',
        'objectives':'Lloyd-Max minimizes scalar MSE for N(0,1), not worst-case input reconstruction. EDEN uses sqrt64-normalized residuals and an ORIGINAL-unbiased vector scale ||x||²/<x,q>. The vector MSE-optimal scale is <x,q>/||q||²; these are different objectives.',
        'tables':[],'gaussian':[],'basis':[],'msong_group0':[]}
    models={}
    for bits in (2,3,4):
        levels,boundary,info=lloyd(bits);models[bits]=(levels,boundary)
        native=faiss.vector_to_array(faiss.index_factory(64,f'EDEN{bits}').sq.trained)
        k=1<<bits
        centroid_delta=float(np.max(np.abs(native[:k]-levels)))
        threshold_delta=float(np.max(np.abs(native[k:]-boundary)))
        assert centroid_delta<1e-6 and threshold_delta<1e-6
        grid=np.linspace(-8,8,100001)
        nearest=np.argmin((grid[:,None]-levels)**2,axis=1)
        threshold=np.searchsorted(boundary,grid,side='right')
        assert np.array_equal(nearest[grid!=0],threshold[grid!=0])
        report['tables'].append(dict(bits=bits,levels=levels.tolist(),thresholds=boundary.tolist(),
            native_centroid_max_abs_error=centroid_delta,native_threshold_max_abs_error=threshold_delta,
            native_threshold_vs_native_centroid_midpoint_max_abs=float(np.max(np.abs(native[k:]-(native[:k-1]+native[1:k])/2))),
            independent_grid_nearest_vs_threshold=True,**info))
    save(report)
    for seed in range(50000,50020):
        x=np.random.default_rng(seed+20261020).standard_normal((4096,64)).astype('float32')
        for rounds in (1,2):
            expected=rotate64(x,seed,rounds)
            t=NativeTransform('block64',seed,rounds,False,1)
            np.testing.assert_array_equal(t.signs,signs(seed).astype('float32'))
            actual=t.apply(x)[:,:64]
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
            for bits,(levels,boundary) in models.items():
                result,_,_=compare(actual,bits,levels,boundary)
                report['gaussian'].append(dict(seed=seed,rounds=rounds,bits=bits,count=len(x),
                    transform_max_abs_error=float(np.max(np.abs(actual-expected))),**result))
        save(report);print('GAUSSIAN_CANARY',seed,flush=True)
    for seed in (12345,12349,50000):
        basis=np.eye(64,dtype='float32')
        for rounds in (1,2):
            x=rotate64(basis,seed,rounds)
            for bits,(levels,boundary) in models.items():
                result,_,errors=compare(x.astype('float32'),bits,levels,boundary)
                if rounds==1:
                    assert float(np.max(errors['unbiased_exact']))<1e-28
                    assert result['fp32']['native_nmse']<1e-12
                report['basis'].append(dict(seed=seed,rounds=rounds,bits=bits,**result))
        print('BASIS_CANARY',seed,flush=True)
    with np.load(OUT/'pilot-data/msong/vectors.npz') as data:
        xt,xb=[np.ascontiguousarray(data[k],dtype='float32') for k in ('xt','xb')]
    for seed in range(12345,12350):
        for rounds in (1,2):
            t=NativeTransform('block64',seed,rounds,False,1)
            zt,zb=t.apply(xt),t.apply(xb)
            leaf=faiss.index_factory(448,'EDEN2');leaf.train(zt)
            center=faiss.vector_to_array(leaf.center)
            residual=np.ascontiguousarray(zb[:,:64]-center[:64],dtype='float32')
            # Fully independent dense rotation and FP64 training mean.
            dense=rotate64(xb[:,:64].astype('float64')-xt[:,:64].astype('float64').mean(axis=0),seed,rounds)
            for bits,(levels,boundary) in models.items():
                result,native,_=compare(residual,bits,levels,boundary)
                _,_,independent_errors,_=independent_quantize(dense,levels,boundary)
                for fmt in ('fp32','bf16'):
                    detail=OUT/'fixed64-rounds-archivedcenter-detail'/f'b{bits}-s{seed}-r{rounds}-{fmt}.npz'
                    with np.load(detail) as d:
                        archived_scales=d['actual_stored_scales'][:,0]
                        np.testing.assert_array_equal(native[fmt][1],archived_scales)
                        # Archived reconstruction adds center inFP32 before comparison;
                        # this independent residual-domain error avoids that extra rounding.
                        archived_group_error=d['group_squared_error'][:,0]
                        archived_group_norm=d['group_residual_squared_norm'][:,0]
                        err=native[fmt][2]*np.sum(residual.astype('float64')**2,axis=1)
                        result[fmt]['archived_group0_pooled_nmse_10k']=float(archived_group_error[:10000].sum()/archived_group_norm[:10000].sum())
                        result[fmt]['archived_group0_pooled_nmse_20k']=float(archived_group_error.sum()/archived_group_norm.sum())
                        for n in (10000,20000):
                            norm=np.sum(residual[:n].astype('float64')**2,axis=1)
                            dn=np.sum(dense[:n]**2,axis=1)
                            result[fmt][f'native_residual_pooled_nmse_{n}']=float(err[:n].sum()/norm.sum())
                            result[fmt][f'independent_dense_pooled_nmse_{n}']=float((independent_errors['unbiased_'+fmt][:n]*dn).sum()/dn.sum())
                        result[fmt]['stored_group_scale_matches_archived_exactly']=True
                report['msong_group0'].append(dict(seed=seed,rounds=rounds,bits=bits,
                    dense_vs_native_centered_max_abs=float(np.max(np.abs(dense-residual))),**result))
            save(report);print('MSONG_GROUP0_CANARY',seed,rounds,flush=True)
    report['gaussian_paired_summary']=[]
    for bits in (2,3,4):
        for fmt in ('fp32','bf16'):
            differences=[]
            for seed in range(50000,50020):
                values=[next(r for r in report['gaussian'] if (r['seed'],r['rounds'],r['bits'])==(seed,rounds,bits))[fmt]['native_nmse'] for rounds in (1,2)]
                differences.append(values[1]-values[0])
            mean=float(np.mean(differences));sem=float(np.std(differences,ddof=1)/math.sqrt(20))
            report['gaussian_paired_summary'].append(dict(bits=bits,format=fmt,paired_seeds=20,vectors_per_round=81920,
                two_minus_one_nmse_mean=mean,paired_seed_sem=sem,approx_t19_95_interval=[mean-2.093*sem,mean+2.093*sem],
                deltas_by_seed=differences))
    report['status']='complete';save(report)
    print('CANARIES_COMPLETE',sha(TARGET),flush=True)

if __name__=='__main__':
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):main()
