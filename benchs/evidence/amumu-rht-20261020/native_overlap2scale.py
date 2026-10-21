"""Native group encoders; compact two-scales/globalnorm and common native scan.
RaBitQ compact layout is a diagnostic format for FP32 queries, NOT nativeqb4.
"""
import ctypes
from pathlib import Path
import numpy as np
import faiss

OUT=Path(__file__).resolve().parent
DLL=ctypes.CDLL(str(OUT/'experimental-kernel/overlap2scale.dll'))
F=ctypes.POINTER(ctypes.c_float);U8=ctypes.POINTER(ctypes.c_uint8);I64=ctypes.POINTER(ctypes.c_int64)
S=DLL.overlap2_scores;S.argtypes=[F,ctypes.c_size_t,U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F];S.restype=ctypes.c_int
K=DLL.overlap2_search;K.argtypes=S.argtypes[:-1]+[ctypes.c_int64,F,I64];K.restype=ctypes.c_int
T=DLL.overlap2_time_pipeline;T.argtypes=[F,F,ctypes.c_size_t,U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F,ctypes.c_int,ctypes.c_int64,ctypes.c_int,F,I64,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];T.restype=ctypes.c_int
def fp(x):return x.ctypes.data_as(F)
def up(x):return x.ctypes.data_as(U8)

def unpack_payload(codes,d,bits):
    return np.sum(np.unpackbits(codes,axis=1,bitorder='little')[:,:d*bits].reshape(len(codes),d,bits)*(1<<np.arange(bits)),axis=2,dtype='int64')

def encode(values,family,bits,groups):
    x=np.ascontiguousarray(values,dtype='float32');d=x.shape[1]
    widths=[d] if groups==1 else [164,256]
    indices=[];scales=[];native_recs=[];start=0;payload=(d*bits+7)//8
    for width in widths:
        part=np.ascontiguousarray(x[:,start:start+width]);leaf=faiss.index_factory(width,f'{family}{bits}');leaf.train(part[:1])
        faiss.copy_array_to_vector(np.zeros(width,dtype='float32'),leaf.center)
        native=leaf.sa_encode(part)
        if family=='EDEN':
            size=(width*bits+7)//8
            index=unpack_payload(native[:,:size],width,bits)
            scale=np.ascontiguousarray(native[:,size+4:size+8]).view('<f4').ravel()
            table=faiss.vector_to_array(leaf.sq.trained)[:1<<bits]
        else:
            binary=(width+7)//8;extra_bits=bits-1;extra_bytes=(width*extra_bits+7)//8
            sign=np.unpackbits(native[:,:binary],axis=1,bitorder='little')[:,:width]
            extra=unpack_payload(native[:,binary+12:binary+12+extra_bytes],width,extra_bits)
            index=(sign.astype('int64')<<extra_bits)+extra
            factors=np.ascontiguousarray(native[:,binary+12+extra_bytes:binary+12+extra_bytes+8]).view('<f4')
            scale=-.5*factors[:,1]
            table=np.arange(1<<bits,dtype='float32')-((1<<bits)-1)/2
        indices.append(index);scales.append(scale)
        native_recs.append(scale[:,None]*table[index]);start+=width
    indices=np.concatenate(indices,axis=1);scales=np.column_stack(scales).astype('float32')
    rawbits=((indices[:,:,None]>>np.arange(bits))&1).astype('uint8').reshape(len(x),-1)
    codes=np.zeros((len(x),payload+4*groups+4),dtype='uint8');codes[:,:payload]=np.packbits(rawbits,axis=1,bitorder='little')
    codes[:,payload:payload+4*groups]=scales.copy().view('uint8')
    norm=np.add.accumulate(x*x,axis=1,dtype='float32')[:,-1]
    codes[:,payload+4*groups:]=norm.copy().view('uint8').reshape(-1,4)
    zero=norm<=np.finfo('float32').eps;codes[zero]=0
    return codes,np.ascontiguousarray(table,dtype='float32'),np.concatenate(native_recs,axis=1)

def decode(codes,d,bits,groups,table):
    payload=(d*bits+7)//8;ids=unpack_payload(codes[:,:payload],d,bits)
    scales=np.ascontiguousarray(codes[:,payload:payload+groups*4]).view('<f4')
    repeated=np.repeat(scales,[d] if groups==1 else [164,256],axis=1)
    return repeated*table[ids]

class Search:
    def __init__(self,codes,table,d,bits,groups):
        self.codes=np.ascontiguousarray(codes,dtype='uint8');self.table=np.ascontiguousarray(table,dtype='float32')
        self.d=d;self.bits=bits;self.groups=groups
    def args(self,q):return [fp(q),len(q),up(self.codes),len(self.codes),self.d,self.bits,self.groups,self.codes.shape[1],fp(self.table)]
    def scores(self,q):
        q=np.ascontiguousarray(q,dtype='float32');out=np.empty((len(q),len(self.codes)),dtype='float32')
        code=S(*self.args(q),fp(out))
        if code:raise RuntimeError(code)
        return out
    def search(self,q,k=100):
        q=np.ascontiguousarray(q,dtype='float32');dist=np.empty((len(q),k),dtype='float32');ids=np.empty((len(q),k),dtype='int64')
        code=K(*self.args(q),k,fp(dist),ids.ctypes.data_as(I64))
        if code:raise RuntimeError(code)
        return dist,ids
    def time_pipeline(self,t,values,center,k=100,batch=256,repeats=3):
        values=np.ascontiguousarray(values,dtype='float32');center=np.ascontiguousarray(center,dtype='float32')
        dist=np.empty((len(values),k),dtype='float32');ids=np.empty((len(values),k),dtype='int64');samples=np.empty(repeats,dtype='float64')
        code=T(fp(values),fp(center),len(values),up(self.codes),len(self.codes),self.d,self.bits,self.groups,self.codes.shape[1],fp(self.table),fp(t.signs),t.strategy,k,batch,fp(dist),ids.ctypes.data_as(I64),repeats,samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if code:raise RuntimeError(code)
        return dist,ids,samples
