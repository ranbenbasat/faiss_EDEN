"""Experimental native grouped codec. Reference scorer is NOT a speed result."""
import ctypes
from pathlib import Path
import numpy as np
import faiss

OUT=Path(__file__).resolve().parent
DLL=ctypes.CDLL(str(OUT/'experimental-kernel/eden_group_native.dll'))
F=ctypes.POINTER(ctypes.c_float);U8=ctypes.POINTER(ctypes.c_uint8);U16=ctypes.POINTER(ctypes.c_uint16)
ENCODE=DLL.eden_group_encode;ENCODE.argtypes=[F,F,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,U8];ENCODE.restype=ctypes.c_int
DECODE=DLL.eden_group_decode;DECODE.argtypes=[U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F,F];DECODE.restype=ctypes.c_int
SCORE=DLL.eden_group_scores_reference;SCORE.argtypes=[F,ctypes.c_size_t,U8,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,F,F,F];SCORE.restype=ctypes.c_int
BF16=DLL.eden_bf16_convert;BF16.argtypes=[F,U16,ctypes.c_size_t];BF16.restype=None
def fp(a):return a.ctypes.data_as(F)
def up(a):return a.ctypes.data_as(U8)
def to_bf16(values):
    values=np.ascontiguousarray(values,dtype='float32');output=np.empty(values.shape,dtype='uint16')
    BF16(fp(values),output.ctypes.data_as(U16),values.size);return output
class GroupEDEN:
    def __init__(self,bits,center=None,stride=None,scale_format='bf16'):
        self.bits=bits;self.format=0 if scale_format=='bf16' else 1
        self.compact=448*bits//8+(18 if self.format==0 else 32)
        self.stride=self.compact if stride is None else stride
        self.center=np.zeros(448,dtype='float32') if center is None else np.ascontiguousarray(center,dtype='float32')
        self.table=faiss.vector_to_array(faiss.index_factory(64,f'EDEN{bits}').sq.trained)
    def encode(self,values):
        values=np.ascontiguousarray(values,dtype='float32');assert values.ndim==2 and values.shape[1]==448
        output=np.empty((len(values),self.stride),dtype='uint8')
        result=ENCODE(fp(values),fp(self.center),len(values),self.bits,self.stride,self.format,fp(self.table),up(output))
        if result:raise ValueError(f'Native group encode returned{result}')
        return output
    def decode(self,codes):
        codes=np.ascontiguousarray(codes,dtype='uint8');assert codes.shape[1]==self.stride
        output=np.empty((len(codes),448),dtype='float32')
        result=DECODE(up(codes),len(codes),self.bits,self.stride,self.format,fp(self.table),fp(self.center),fp(output))
        if result:raise ValueError(f'Native group decode returned{result}')
        return output
    def scores_reference(self,queries,codes):
        queries=np.ascontiguousarray(queries,dtype='float32');codes=np.ascontiguousarray(codes,dtype='uint8')
        output=np.empty((len(queries),len(codes)),dtype='float32')
        result=SCORE(fp(queries),len(queries),up(codes),len(codes),self.bits,self.stride,self.format,fp(self.table),fp(self.center),fp(output))
        if result:raise ValueError(f'Native reference scores returned{result}')
        return output
