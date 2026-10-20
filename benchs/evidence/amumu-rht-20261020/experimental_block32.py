"""EXPERIMENTAL quality oracle, not the public HadamardRotation feature.

Independent SplitMix64 global signs each round. One fixed permutation between
rounds, shared by every vector. No per-block quantizer scales are introduced.
NumPy/block/dense implementations here MUST NOT be used for speed claims.
"""
import numpy as np
import audit_msong_quality as audit

MASK=(1<<64)-1
class SplitMix64:
    def __init__(self,seed):
        self.state=seed&MASK
    def next(self):
        self.state=(self.state+0x9e3779b97f4a7c15)&MASK
        z=self.state
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK
        z=((z^(z>>27))*0x94d049bb133111eb)&MASK
        return z^(z>>31)


def plan(d,seed,block_size=32):
    assert d in (424,448,512)
    rng=SplitMix64(seed)
    signs=np.array([[1. if rng.next()%2 else -1. for _ in range(d)] for _ in range(3)],dtype='float64')
    rng=SplitMix64(seed^0xd1b54a32d192ed03)
    permutation=np.arange(d,dtype='int64')
    for i in range(d-1):
        j=i+rng.next()%(d-i)
        permutation[i],permutation[j]=permutation[j],permutation[i]
    widths=[block_size]*(d//block_size)+([d%block_size] if d%block_size else [])
    assert all(w>0 and w&(w-1)==0 for w in widths)
    return {'dimension':d,'seed':seed,'block_size':block_size,'signs':signs,'permutation':permutation,'widths':widths}


def block_forward(x,widths):
    y=x.copy();start=0
    for width in widths:
        y[:,start:start+width]=audit.fwht(y[:,start:start+width]);start+=width
    return y


def forward(values,p,rounds,permute=True):
    d=p['dimension'];y=np.zeros((len(values),d),dtype='float64')
    y[:,:values.shape[1]]=values
    for r in range(rounds):
        if r and permute:
            y=y[:,p['permutation']]
        y=block_forward(y*p['signs'][r],p['widths'])
    return y


def inverse(values,p,rounds,permute=True):
    y=values.astype('float64',copy=True)
    for r in reversed(range(rounds)):
        y=block_forward(y,p['widths'])*p['signs'][r]
        if r and permute:
            y=y[:,np.argsort(p['permutation'])]
    return y


def scalar_reference(vector,p,rounds,permute):
    # Independent dense block formula using parity, not the butterfly routine.
    y=np.pad(vector.astype('float64'),(0,p['dimension']-len(vector)))
    for r in range(rounds):
        if r and permute:
            y=y[p['permutation']]
        y=y*p['signs'][r];out=np.empty_like(y);start=0
        for width in p['widths']:
            for i in range(width):
                out[start+i]=sum((-1. if (i&j).bit_count()%2 else 1.)*y[start+j] for j in range(width))/np.sqrt(width)
            start+=width
        y=out
    return y


def validate(p,rounds,permute):
    d=p['dimension'];rng=np.random.default_rng(78901)
    x=rng.standard_normal((8,d));z=forward(x,p,rounds,permute)
    reference=np.array([scalar_reference(row,p,rounds,permute) for row in x])
    np.testing.assert_allclose(z,reference,rtol=1e-12,atol=1e-12)
    restored=inverse(z,p,rounds,permute)
    np.testing.assert_allclose(restored,x,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(z@z.T,x@x.T,rtol=1e-12,atol=1e-10)
    block_change=0.;start=0
    for width in p['widths']:
        before=np.sum(x[:,start:start+width]**2,axis=1)
        after=np.sum(z[:,start:start+width]**2,axis=1)
        block_change=max(block_change,float(np.max(np.abs(before-after))));start+=width
    if not permute or rounds==1:
        assert block_change<1e-10
    elif rounds>1:
        assert block_change>1e-4
    return {'scalar_reference_max_abs_error':float(np.max(np.abs(z-reference))),
        'inverse_max_abs_error':float(np.max(np.abs(restored-x))),
        'gram_max_abs_error':float(np.max(np.abs(z@z.T-x@x.T))),
        'block_energy_change_max_abs':block_change,
        'sign_sha256':audit.digest(p['signs']),'permutation_sha256':audit.digest(p['permutation']),
        'butterfly_addsub_count':int(rounds*sum(w*int(np.log2(w)) for w in p['widths'])),
        'sign_permutation_loop_scale_overheads_included_in_operation_count':False}
