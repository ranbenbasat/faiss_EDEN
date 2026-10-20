"""Native grouped search plus uniform-kernel global EDEN control."""
import ctypes
from pathlib import Path
import numpy as np
import faiss
from native_eden_group import F,U8,fp,up

OUT=Path(__file__).resolve().parent
DLL=ctypes.CDLL(str(OUT/'experimental-kernel/eden_group_fixed64.dll'))
I64=ctypes.POINTER(ctypes.c_int64);I32=ctypes.POINTER(ctypes.c_int32)
SCORES=DLL.eden_codes_scores
SCORES.argtypes=[F,ctypes.c_size_t,U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F,F,ctypes.c_int];SCORES.restype=ctypes.c_int
SEARCH=DLL.eden_codes_search
SEARCH.argtypes=[F,ctypes.c_size_t,U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F,ctypes.c_int64,F,I64,ctypes.c_int];SEARCH.restype=ctypes.c_int
TIME=DLL.eden_codes_time_pipeline
TIME.argtypes=[F,ctypes.c_size_t,U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F,F,I32,ctypes.c_int,
    ctypes.c_int64,ctypes.c_int,F,I64,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];TIME.restype=ctypes.c_int
class EdenCodeSearch:
    def __init__(self,codes,bits,kind,center,strategy=1):
        self.codes=np.ascontiguousarray(codes,dtype='uint8');self.bits=bits;self.kind=kind
        self.center=np.ascontiguousarray(center,dtype='float32');self.d=512 if kind==2 else 448
        assert self.center.shape==(self.d,)
        self.table=faiss.vector_to_array(faiss.index_factory(64,f'EDEN{bits}').sq.trained)
        self.strategy=strategy
    def _args(self,queries):
        return [fp(queries),len(queries),up(self.codes),len(self.codes),self.bits,self.codes.shape[1],self.kind,fp(self.table),fp(self.center)]
    def scores(self,queries):
        queries=np.ascontiguousarray(queries,dtype='float32');assert queries.shape[1]==self.d
        result=np.empty((len(queries),len(self.codes)),dtype='float32')
        code=SCORES(*self._args(queries),fp(result),self.strategy)
        if code:raise ValueError(f'Native optimized scores returned{code}')
        return result
    def search(self,queries,k):
        queries=np.ascontiguousarray(queries,dtype='float32');assert queries.shape[1]==self.d
        distances=np.empty((len(queries),k),dtype='float32');labels=np.empty((len(queries),k),dtype='int64')
        code=SEARCH(*self._args(queries),k,fp(distances),labels.ctypes.data_as(I64),self.strategy)
        if code:raise ValueError(f'Native optimized search returned{code}')
        return distances,labels
    def time_pipeline(self,transform,values,k,batch=256,repeats=5):
        values=np.ascontiguousarray(values,dtype='float32');assert values.shape[1]==420
        if self.kind!=2:assert transform.permutation is None, 'No permutation allowed in corrected experiment'
        distances=np.empty((len(values),k),dtype='float32');labels=np.empty((len(values),k),dtype='int64')
        samples=np.empty(repeats,dtype='float64')
        permutation=None if transform.permutation is None else transform.permutation.ctypes.data_as(I32)
        code=TIME(fp(values),len(values),up(self.codes),len(self.codes),self.bits,self.codes.shape[1],self.kind,fp(self.table),fp(self.center),
            fp(transform.signs),permutation,transform.strategy,k,batch,fp(distances),labels.ctypes.data_as(I64),self.strategy,repeats,
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if code:raise ValueError(f'Native optimized pipeline returned{code}')
        return distances,labels,samples

