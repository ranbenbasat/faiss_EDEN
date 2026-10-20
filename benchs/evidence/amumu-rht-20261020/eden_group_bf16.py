"""Requested experimental grouped EDEN QUALITY oracle, not a timed codec.

Per64 groups reuse frozen FAISS ORIGINAL-unbiased EDEN64 encoding (sqrt64
normalization, same zero epsilon and scalar table). Only seven decoded BF16
scales plus the original whole-vector FP32 residual norm are stored afterward.
"""
import functools
import numpy as np
import faiss

def to_bf16(values):
    v=np.ascontiguousarray(values,dtype='<f4');words=v.view('<u4')
    rounded=((words.astype('uint64')+0x7fff+((words>>16)&1))>>16).astype('<u2')
    # Preserve NaNs as quiet NaNs instead of rounding a tiny payload to infinity.
    nan=((words&0x7f800000)==0x7f800000)&((words&0x007fffff)!=0)
    return np.where(nan,((words>>16)|0x40).astype('<u2'),rounded).astype('<u2')
def from_bf16(values):
    return (np.ascontiguousarray(values,dtype='<u2').astype('<u4')<<16).view('<f4')
@functools.lru_cache(None)
def group_quantizer(bits):
    if bits not in (2,3,4):raise ValueError('Only approved2/3/4 bits')
    index=faiss.index_factory(64,f'EDEN{bits}')
    index.train(np.zeros((1,64),dtype='float32'))
    return index
def encode(values,bits,center=None,stride=None,original_norm2=None):
    x=np.ascontiguousarray(values,dtype='float32')
    if x.ndim!=2 or x.shape[1]!=448:raise ValueError('Expected N x448')
    residual=np.ascontiguousarray(x-(0 if center is None else np.asarray(center,dtype='float32')),dtype='float32')
    if not np.isfinite(residual).all():raise ValueError('Nonfinite input residual')
    payload=448*bits//8;compact=payload+18;stride=compact if stride is None else stride
    if stride<compact:raise ValueError('Stride smaller than compact budget')
    native=group_quantizer(bits).sa_encode(residual.reshape(-1,64))
    group_payload=64*bits//8
    factors=native[:,group_payload:].copy().view('<f4').reshape(len(x),7,2)
    scales=to_bf16(factors[:,:,1])
    if not np.isfinite(from_bf16(scales)).all():raise FloatingPointError('Unrepresentable BF16 scale; no clipping permitted')
    norm=(np.add.accumulate(residual*residual,axis=1,dtype='float32')[:,-1] if original_norm2 is None
          else np.asarray(original_norm2,dtype='float32'))
    if norm.shape!=(len(x),):raise ValueError('One original whole residual norm per vector required')
    if not np.isfinite(norm).all() or np.any(norm<0):raise ValueError('Invalid original residual norm')
    result=np.zeros((len(x),stride),dtype='uint8')
    result[:,:payload]=native[:,:group_payload].reshape(len(x),payload)
    result[:,payload:payload+14]=scales.copy().view('uint8').reshape(len(x),14)
    result[:,payload+14:payload+18]=norm.astype('<f4').copy().view('uint8').reshape(len(x),4)
    # Native whole-vector shortcut, distinct from seven per-group shortcuts.
    whole_zero=norm<=np.finfo('float32').eps
    result[whole_zero]=0
    return result
def decode(codes,bits,center=None):
    codes=np.ascontiguousarray(codes,dtype='uint8');payload=448*bits//8
    packed=np.unpackbits(codes[:,:payload],axis=1,bitorder='little').reshape(len(codes),448,bits)
    indices=np.sum(packed*(1<<np.arange(bits)),axis=2,dtype='int64')
    table=faiss.vector_to_array(group_quantizer(bits).sq.trained)[:1<<bits]
    scales=from_bf16(codes[:,payload:payload+14].copy().view('<u2'))
    if not np.isfinite(scales).all():raise ValueError('Invalid stored BF16 scale')
    result=table[indices]*np.repeat(scales,64,axis=1)
    return result if center is None else result+np.asarray(center,dtype='float32')
def scores(queries,codes,bits,center=None):
    q=np.asarray(queries,dtype='float32')-(0 if center is None else np.asarray(center,dtype='float32'))
    if not np.isfinite(q).all():raise ValueError('Nonfinite query residual')
    rec=decode(codes,bits)
    payload=448*bits//8
    norm=np.ascontiguousarray(codes[:,payload+14:payload+18]).view('<f4')[:,0]
    if not np.isfinite(norm).all() or np.any(norm<0):raise ValueError('Invalid stored original residual norm')
    return np.maximum(np.sum(q.astype('float64')**2,axis=1)[:,None]+norm[None,:]-2*q.astype('float64')@rec.astype('float64').T,0)
def topk(distances,k):
    # Lexicographic(distance,ID), matching FAISS CMax tie policy. Quality-only.
    return np.argsort(distances,axis=1,kind='stable')[:,:k].astype('int64')
