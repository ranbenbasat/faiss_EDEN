import unittest
import numpy as np
import faiss
import eden_group_bf16 as codec

class TestGroupedEDEN(unittest.TestCase):
    def test_bf16_known_round_nearest_even(self):
        words=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x00008000,0x00018000,0,0x80000000],dtype='uint32')
        expected=np.array([0x3f80,0x3f82,0xbf80,0xbf82,0,2,0,0x8000],dtype='uint16')
        np.testing.assert_array_equal(codec.to_bf16(words.view('float32')),expected)
        np.testing.assert_array_equal(codec.to_bf16(codec.from_bf16(expected)),expected)
    def test_special_conversion_invalid_input_and_unaligned_metadata(self):
        words=np.array([0x7f800000,0xff800000,0x7fc00001,0xffc00001],dtype='uint32')
        converted=codec.from_bf16(codec.to_bf16(words.view('float32')))
        self.assertTrue(np.isposinf(converted[0]));self.assertTrue(np.isneginf(converted[1]))
        self.assertTrue(np.isnan(converted[2:]).all())
        with self.assertRaises(ValueError):codec.encode(np.full((1,448),np.nan,dtype='float32'),2)
        packed=codec.encode(np.ones((2,448),dtype='float32'),2)
        with self.assertRaises(ValueError):codec.scores(np.full((1,448),np.inf,dtype='float32'),packed,2)
        backing=np.empty(packed.size+1,dtype='uint8');unaligned=backing[1:].reshape(packed.shape);unaligned[:]=packed
        np.testing.assert_array_equal(codec.decode(unaligned,2),codec.decode(packed,2))
    def test_exact_sizes_and_18_metadata_bytes(self):
        x=np.ones((3,448),dtype='float32')
        for bits,compact,cap in ((2,130,136),(3,186,200),(4,242,264)):
            a=codec.encode(x,bits);b=codec.encode(x,bits,stride=cap)
            self.assertEqual(a.shape,(3,compact));self.assertEqual(compact-448*bits//8,18)
            np.testing.assert_array_equal(a,b[:,:compact]);self.assertTrue(np.all(b[:,compact:]==0))
            np.testing.assert_array_equal(codec.decode(a,bits),codec.decode(b,bits))
    def test_zero_groups_zero_vector(self):
        x=np.zeros((2,448),dtype='float32');x[1,:64]=np.linspace(-1,1,64)
        codes=codec.encode(x,3);decoded=codec.decode(codes,3)
        self.assertTrue(np.isfinite(decoded).all());np.testing.assert_array_equal(decoded[0],0)
        np.testing.assert_array_equal(decoded[1,64:],0)
        payload=448*3//8
        scales=codes[:,payload:payload+14].copy().view('<u2')
        self.assertTrue(np.all(scales[0]==0));self.assertTrue(np.all(scales[1,1:]==0))
    def test_frozen_native_group_codes_and_unbiased_scale(self):
        rng=np.random.default_rng(91);x=rng.standard_normal((2,448),dtype='float32')
        for bits in (2,3,4):
            leaf=faiss.index_factory(64,f'EDEN{bits}');leaf.train(np.zeros((1,64),dtype='float32'))
            native=leaf.sa_encode(x.reshape(-1,64));p=64*bits//8
            codes=codec.encode(x,bits)
            np.testing.assert_array_equal(codes[:,:7*p],native[:,:p].reshape(2,7*p))
            expected_scales=codec.to_bf16(native[:,p:].copy().view('<f4').reshape(2,7,2)[:,:,1])
            np.testing.assert_array_equal(codes[:,7*p:7*p+14].copy().view('<u2'),expected_scales)
    def test_decode_uses_rounded_scales(self):
        rng=np.random.default_rng(19);x=rng.standard_normal((3,448),dtype='float32')
        codes=codec.encode(x,2);p=112
        altered=codes.copy();altered[:,p:p+14]=0
        np.testing.assert_array_equal(codec.decode(altered,2),0)
        self.assertGreater(float(np.linalg.norm(codec.decode(codes,2))),0)
    def test_independent_scalar_encode_original_unbiased(self):
        rng=np.random.default_rng(119);x=rng.standard_normal((1,448),dtype='float32')
        for bits in (2,3,4):
            packed=codec.encode(x,bits);payload=448*bits//8
            leaf=faiss.index_factory(64,f'EDEN{bits}');trained=faiss.vector_to_array(leaf.sq.trained)
            table,thresholds=trained[:1<<bits],trained[1<<bits:]
            expected=np.zeros(payload,dtype='uint8');scales=[]
            for group in range(7):
                values=x[0,group*64:(group+1)*64]
                norm=np.float32(0)
                for value in values:norm=np.float32(norm+np.float32(value*value))
                inv=np.float32(1)/np.sqrt(norm,dtype='float32')
                indices=[];dot=0.
                for j,value in enumerate(values):
                    z=np.float32(np.float32(value*np.float32(8))*inv)
                    index=int(np.searchsorted(thresholds,z,side='right'));indices.append(index)
                    dot+=float(table[index])*float(value)
                    offset=(group*64+j)*bits;encoded=index<<(offset%8)
                    expected[offset//8]|=encoded&255
                    if offset%8+bits>8:expected[offset//8+1]|=encoded>>8
                scales.append(np.float32(float(norm)/dot))
            np.testing.assert_array_equal(packed[0,:payload],expected)
            np.testing.assert_array_equal(packed[:,payload:payload+14].copy().view('<u2')[0],codec.to_bf16(scales))
    def test_original_whole_norm_term_not_reconstructed_norm(self):
        x=np.arange(448,dtype='float32')[None,:]/448
        original=np.array([1234.5],dtype='float32');codes=codec.encode(x,2,original_norm2=original)
        zero=np.zeros((1,448),dtype='float32')
        self.assertEqual(float(codec.scores(zero,codes,2)[0,0]),1234.5)
        stored=codes[:,126:130].copy().view('<f4')[:,0]
        np.testing.assert_array_equal(stored,original)
    def test_independent_scalar_decoding_and_scores_centering(self):
        rng=np.random.default_rng(4);center=rng.standard_normal(448).astype('float32')
        x=rng.standard_normal((3,448),dtype='float32');q=rng.standard_normal((2,448),dtype='float32')
        for bits in (2,3,4):
            codes=codec.encode(x+center,bits,center=center)
            leaf=faiss.index_factory(64,f'EDEN{bits}')
            table=faiss.vector_to_array(leaf.sq.trained)[:1<<bits]
            p=448*bits//8;scales=codec.from_bf16(codes[:,p:p+14].copy().view('<u2'))
            rec=np.empty((3,448),dtype='float32')
            for n in range(3):
                for j in range(448):
                    offset=j*bits;value=(int(codes[n,offset//8])>>(offset%8))
                    if offset%8+bits>8:value |= int(codes[n,offset//8+1])<<(8-offset%8)
                    rec[n,j]=scales[n,j//64]*table[value&((1<<bits)-1)]
            np.testing.assert_allclose(codec.decode(codes,bits,center=center),rec+center,rtol=0,atol=0)
            norm=codes[:,p+14:p+18].copy().view('<f4')[:,0]
            qr=(q+center).astype('float32')-center
            expected=np.sum(qr.astype('float64')**2,axis=1)[:,None]+norm[None,:]-2*qr.astype('float64')@rec.astype('float64').T
            np.testing.assert_allclose(codec.scores(q+center,codes,bits,center=center),expected,rtol=1e-12,atol=1e-9)
    def test_translation_consistency(self):
        x=np.ones((2,448),dtype='float32')*2;q=np.ones((1,448),dtype='float32')*3;c=np.ones(448,dtype='float32')*4
        a=codec.encode(x,4);b=codec.encode(x+c,4,center=c)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(codec.scores(q,a,4),codec.scores(q+c,b,4,center=c))
    def test_epsilon_whole_and_per_group_policy(self):
        eps=np.finfo('float32').eps
        x=np.full((2,448),np.sqrt(eps*.4/64),dtype='float32')
        x[0]*=np.float32(.1)
        packed=codec.encode(x,2);payload=112
        norms=packed[:,payload+14:payload+18].copy().view('<f4')[:,0]
        self.assertEqual(norms[0],0)
        self.assertGreater(norms[1],eps)
        self.assertTrue(np.all(packed[:,:payload+14]==0))
        result=codec.scores(np.zeros((1,448),dtype='float32'),packed,2)
        self.assertEqual(result[0,0],0);self.assertEqual(result[0,1],float(norms[1]))
    def test_final_l2_clamp_and_tie_order(self):
        x=np.ones((3,448),dtype='float32');packed=codec.encode(x,2)
        rec=codec.decode(packed,2);payload=112
        packed[:,payload+14:payload+18]=0
        score=codec.scores(rec[:1],packed,2)
        np.testing.assert_array_equal(score,0)
        np.testing.assert_array_equal(codec.topk(score,2),np.array([[0,1]],dtype='int64'))
        # Large positive group2 contribution plus negative group1 contribution:
        # clamping any group separately would disagree with the total formula.
        packed=codec.encode(x[:1],2);q=np.zeros((1,448),dtype='float32')
        q[:,:64]=rec[:1,:64];q[:,64:128]=10
        raw=np.sum(q.astype('float64')**2)+float(packed[:,126:130].copy().view('<f4')[0,0])-2*(q.astype('float64')@codec.decode(packed,2).astype('float64').T).item()
        self.assertAlmostEqual(codec.scores(q,packed,2)[0,0],max(raw,0),places=8)

if __name__=='__main__':unittest.main(verbosity=2)
