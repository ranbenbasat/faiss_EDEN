"""Regression: exactly two independent-sign H64 rounds, NEVER cross groups."""
import unittest
import numpy as np
import experimental_block32 as oracle
from native_block64 import NativeTransform

class TestFixed64(unittest.TestCase):
    def test_no_support_crosses_group_boundary(self):
        for seed in (12345,12349):
            t=NativeTransform('block64',seed,2,False,1)
            self.assertIsNone(t.permutation)
            actual=t.apply(np.eye(448,dtype='float32'))
            for group in range(7):
                outside=np.ones(448,dtype=bool);outside[group*64:(group+1)*64]=False
                np.testing.assert_array_equal(actual[group*64:(group+1)*64,outside],0)
            np.testing.assert_allclose(actual.astype('float64')@actual.T,np.eye(448),rtol=0,atol=0)
    def test_exact_independent_rounds_vs_explicit_oracle(self):
        for seed in range(12345,12350):
            p=oracle.plan(448,seed,64)
            self.assertFalse(np.array_equal(p['signs'][0],p['signs'][1]))
            t=NativeTransform('block64',seed,2,False,1)
            for coordinate in (0,63,64,255,383,419,447):
                basis=np.zeros(448,dtype='float32');basis[coordinate]=1
                expected=oracle.scalar_reference(basis,p,2,False).astype('float32')
                np.testing.assert_array_equal(t.apply(basis[None,:])[0],expected)
    def test_wrong_scope_permutation_is_detectable(self):
        x=np.zeros((1,448),dtype='float32');x[0,7]=1
        fixed=NativeTransform('block64',12345,2,False,1).apply(x)
        wrong=NativeTransform('block64',12345,2,True,1).apply(x)
        np.testing.assert_array_equal(fixed[:,64:],0)
        self.assertGreater(np.count_nonzero(wrong[:,64:]),0)

if __name__=='__main__':unittest.main(verbosity=2)
