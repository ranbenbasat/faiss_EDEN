"""Bindings for standalone experimental native kernels, not a FAISS feature."""
import ctypes
from pathlib import Path
import numpy as np
import experimental_block32 as oracle

ROOT=Path(__file__).resolve().parent
DLL=ctypes.CDLL(str(ROOT/'experimental-kernel/eden_group_fixed64.dll'))
FLOAT=ctypes.POINTER(ctypes.c_float)
INT=ctypes.POINTER(ctypes.c_int32)
CALL=DLL.experimental_transform
CALL.argtypes=[FLOAT,FLOAT,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,FLOAT,INT,ctypes.c_int]
CALL.restype=ctypes.c_int
TIME=DLL.experimental_time_transform
TIME.argtypes=CALL.argtypes+[ctypes.c_int]
TIME.restype=ctypes.c_double
SEARCH=DLL.experimental_pipeline
SEARCH.argtypes=[FLOAT,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,FLOAT,INT,ctypes.c_int,
    ctypes.c_size_t,ctypes.c_int64,ctypes.c_int,FLOAT,ctypes.POINTER(ctypes.c_int64)]
SEARCH.restype=ctypes.c_int
SEARCH_TIME=DLL.experimental_time_pipeline
SEARCH_TIME.argtypes=SEARCH.argtypes+[ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
SEARCH_TIME.restype=ctypes.c_int

class NativeTransform:
    def __init__(self,mode,seed,rounds,permute=False,strategy=2):
        assert mode in ('block64','full512')
        if mode=='block64' and permute:raise ValueError('Corrected fixed64 experiment FORBIDS permutation')
        self.mode=mode;self.seed=seed;self.rounds=rounds;self.permute=permute;self.strategy=strategy
        self.dimension=448 if mode=='block64' else 512
        plan=oracle.plan(self.dimension,seed,64)
        self.signs=np.ascontiguousarray(plan['signs'],dtype='float32')
        self.permutation=np.ascontiguousarray(plan['permutation'],dtype='int32') if mode=='block64' and permute else None
    def apply_into(self,values,output):
        assert values.dtype==np.float32 and values.flags.c_contiguous and 1<=values.shape[1]<=448
        assert output.dtype==np.float32 and output.flags.c_contiguous and output.shape==(len(values),self.dimension)
        permutation=None if self.permutation is None else self.permutation.ctypes.data_as(INT)
        result=CALL(values.ctypes.data_as(FLOAT),output.ctypes.data_as(FLOAT),len(values),values.shape[1],
                    1 if self.mode=='block64' else 0,self.rounds,self.signs.ctypes.data_as(FLOAT),permutation,self.strategy)
        if result:
            raise RuntimeError(f'Native transform returned{result}')
        return output
    def apply(self,values):
        values=np.ascontiguousarray(values,dtype='float32')
        return self.apply_into(values,np.empty((len(values),self.dimension),dtype='float32'))
    def time_transform(self,values,repeats=100):
        values=np.ascontiguousarray(values,dtype='float32')
        output=np.empty((len(values),self.dimension),dtype='float32')
        permutation=None if self.permutation is None else self.permutation.ctypes.data_as(INT)
        seconds=TIME(values.ctypes.data_as(FLOAT),output.ctypes.data_as(FLOAT),len(values),values.shape[1],
            1 if self.mode=='block64' else 0,self.rounds,self.signs.ctypes.data_as(FLOAT),permutation,self.strategy,repeats)
        if seconds<0:
            raise RuntimeError(f'Native transform timing returned{seconds}')
        return seconds/repeats,output
    def search(self,index,values,k,batch=256,repeats=0):
        values=np.ascontiguousarray(values,dtype='float32')
        distances=np.empty((len(values),k),dtype='float32');labels=np.empty((len(values),k),dtype='int64')
        permutation=None if self.permutation is None else self.permutation.ctypes.data_as(INT)
        args=[values.ctypes.data_as(FLOAT),len(values),values.shape[1],1 if self.mode=='block64' else 0,self.rounds,
            self.signs.ctypes.data_as(FLOAT),permutation,self.strategy,int(index.this),k,batch,
            distances.ctypes.data_as(FLOAT),labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))]
        if repeats:
            samples=np.empty(repeats,dtype='float64')
            result=SEARCH_TIME(*args,repeats,samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        else:
            result=SEARCH(*args);samples=None
        if result:
            raise RuntimeError(f'Native pipeline returned{result}')
        return distances,labels,samples

