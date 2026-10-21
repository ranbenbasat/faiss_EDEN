"""Native overlapping H256 comparator. No production FAISS source changes."""
import ctypes
from pathlib import Path
import numpy as np
import faiss

OUT=Path(__file__).resolve().parent
DLL=ctypes.CDLL(str(OUT/'experimental-kernel/overlap256.dll'))
F=ctypes.POINTER(ctypes.c_float);I64=ctypes.POINTER(ctypes.c_int64)
T=DLL.overlap256_transform;T.argtypes=[F,F,ctypes.c_size_t,F,ctypes.c_int];T.restype=ctypes.c_int
TT=DLL.overlap256_time_transform;TT.argtypes=[F,F,ctypes.c_size_t,ctypes.c_int,F,ctypes.c_int,ctypes.c_size_t,ctypes.c_int];TT.restype=ctypes.c_double
P=DLL.overlap256_pipeline;P.argtypes=[F,F,ctypes.c_size_t,ctypes.c_int,F,ctypes.c_int,ctypes.c_size_t,ctypes.c_int64,ctypes.c_int,F,I64];P.restype=ctypes.c_int
TP=DLL.overlap256_time_pipeline;TP.argtypes=P.argtypes+[ctypes.c_int,ctypes.POINTER(ctypes.c_double)];TP.restype=ctypes.c_int
def fp(a):return a.ctypes.data_as(F)

class Overlap:
    def __init__(self,seed,strategy=2):
        self.seed=seed;self.strategy=strategy
        self.frozen=faiss.HadamardRotation(420,seed,1)
        self.signs=np.ascontiguousarray(faiss.vector_to_array(self.frozen.signs1),dtype='float32')
        assert self.signs.shape==(512,)
    def apply(self,values):
        values=np.ascontiguousarray(values,dtype='float32');assert values.shape[1]==420
        out=np.empty((len(values),420),dtype='float32')
        code=T(fp(values),fp(out),len(values),fp(self.signs),self.strategy)
        if code:raise RuntimeError(code)
        return out
    def pipeline(self,values,center,index,mode,k=100,batch=256,repeats=0):
        values=np.ascontiguousarray(values,dtype='float32');center=np.ascontiguousarray(center,dtype='float32')
        distances=np.empty((len(values),k),dtype='float32');labels=np.empty((len(values),k),dtype='int64')
        args=[fp(values),fp(center),len(values),mode,fp(self.signs),self.strategy,int(index.this),k,batch,fp(distances),labels.ctypes.data_as(I64)]
        samples=None
        if repeats:
            samples=np.empty(repeats,dtype='float64')
            code=TP(*args,repeats,samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        else:code=P(*args)
        if code:raise RuntimeError(code)
        return distances,labels,samples
