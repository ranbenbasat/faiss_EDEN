// Experimental standalone native transform; does NOT modify FAISS HR semantics.
// Full512 and Block64 are compiled together with identical optimization flags.
#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <faiss/Index.h>

static inline __m256 h8(__m256 x) {
    const __m256 s1 = _mm256_castsi256_ps(_mm256_setr_epi32(0,INT32_MIN,0,INT32_MIN,0,INT32_MIN,0,INT32_MIN));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_setr_epi32(0,0,INT32_MIN,INT32_MIN,0,0,INT32_MIN,INT32_MIN));
    const __m256 s4 = _mm256_castsi256_ps(_mm256_setr_epi32(0,0,0,0,INT32_MIN,INT32_MIN,INT32_MIN,INT32_MIN));
    x = _mm256_add_ps(_mm256_xor_ps(x,s1),_mm256_permute_ps(x,0xb1));
    x = _mm256_add_ps(_mm256_xor_ps(x,s2),_mm256_permute_ps(x,0x4e));
    x = _mm256_add_ps(_mm256_xor_ps(x,s4),_mm256_permute2f128_ps(x,x,1));
    return x;
}
static inline void butterfly(__m256& a,__m256& b) {
    __m256 x=a;a=_mm256_add_ps(x,b);b=_mm256_sub_ps(x,b);
}
static inline __m256 sign_load(const float* x,const float* signs) {
    return _mm256_xor_ps(_mm256_loadu_ps(x),_mm256_and_ps(_mm256_loadu_ps(signs),_mm256_set1_ps(-0.0f)));
}
template<bool Normalize=true>
static inline void block64(float* y,const float* signs) {
    __m256 a=h8(sign_load(y,signs)),b=h8(sign_load(y+8,signs+8));
    __m256 c=h8(sign_load(y+16,signs+16)),d=h8(sign_load(y+24,signs+24));
    __m256 e=h8(sign_load(y+32,signs+32)),f=h8(sign_load(y+40,signs+40));
    __m256 g=h8(sign_load(y+48,signs+48)),h=h8(sign_load(y+56,signs+56));
    butterfly(a,b);butterfly(c,d);butterfly(e,f);butterfly(g,h);
    butterfly(a,c);butterfly(b,d);butterfly(e,g);butterfly(f,h);
    butterfly(a,e);butterfly(b,f);butterfly(c,g);butterfly(d,h);
    const __m256 scale=_mm256_set1_ps(0.125f);
    _mm256_storeu_ps(y,(Normalize?_mm256_mul_ps(a,scale):a));_mm256_storeu_ps(y+8,(Normalize?_mm256_mul_ps(b,scale):b));
    _mm256_storeu_ps(y+16,(Normalize?_mm256_mul_ps(c,scale):c));_mm256_storeu_ps(y+24,(Normalize?_mm256_mul_ps(d,scale):d));
    _mm256_storeu_ps(y+32,(Normalize?_mm256_mul_ps(e,scale):e));_mm256_storeu_ps(y+40,(Normalize?_mm256_mul_ps(f,scale):f));
    _mm256_storeu_ps(y+48,(Normalize?_mm256_mul_ps(g,scale):g));_mm256_storeu_ps(y+56,(Normalize?_mm256_mul_ps(h,scale):h));
}
static inline void scalar_fwht(float* x,int n) {
    for(int width=1;width<n;width*=2)
        for(int start=0;start<n;start+=2*width)
            for(int j=0;j<width;j++) {
                float a=x[start+j],b=x[start+width+j];
                x[start+j]=a+b;x[start+width+j]=a-b;
            }
}

static inline void finish64(__m256& a,__m256& b,__m256& c,__m256& d,__m256& e,__m256& f,__m256& g,__m256& h) {
    butterfly(a,b);butterfly(c,d);butterfly(e,f);butterfly(g,h);
    butterfly(a,c);butterfly(b,d);butterfly(e,g);butterfly(f,h);
    butterfly(a,e);butterfly(b,f);butterfly(c,g);butterfly(d,h);
}
static inline __m256 second_h8(__m256 value,const float* signs) {
    const __m256 normalized=_mm256_mul_ps(value,_mm256_set1_ps(.125f));
    const __m256 signed_value=_mm256_xor_ps(normalized,_mm256_and_ps(_mm256_loadu_ps(signs),_mm256_set1_ps(-0.f)));
    return h8(signed_value);
}
static inline void fixed64_two(float* y,const float* s1,const float* s2) {
    // Both rounds remain inside this SAME64-coordinate group, in registers.
    __m256 a=h8(sign_load(y,s1)),b=h8(sign_load(y+8,s1+8));
    __m256 c=h8(sign_load(y+16,s1+16)),d=h8(sign_load(y+24,s1+24));
    __m256 e=h8(sign_load(y+32,s1+32)),f=h8(sign_load(y+40,s1+40));
    __m256 g=h8(sign_load(y+48,s1+48)),h=h8(sign_load(y+56,s1+56));
    finish64(a,b,c,d,e,f,g,h);
    a=second_h8(a,s2);b=second_h8(b,s2+8);c=second_h8(c,s2+16);d=second_h8(d,s2+24);
    e=second_h8(e,s2+32);f=second_h8(f,s2+40);g=second_h8(g,s2+48);h=second_h8(h,s2+56);
    finish64(a,b,c,d,e,f,g,h);
    const __m256 scale=_mm256_set1_ps(.125f);
    _mm256_storeu_ps(y,_mm256_mul_ps(a,scale));_mm256_storeu_ps(y+8,_mm256_mul_ps(b,scale));
    _mm256_storeu_ps(y+16,_mm256_mul_ps(c,scale));_mm256_storeu_ps(y+24,_mm256_mul_ps(d,scale));
    _mm256_storeu_ps(y+32,_mm256_mul_ps(e,scale));_mm256_storeu_ps(y+40,_mm256_mul_ps(f,scale));
    _mm256_storeu_ps(y+48,_mm256_mul_ps(g,scale));_mm256_storeu_ps(y+56,_mm256_mul_ps(h,scale));
}

static inline void full512(float* y,const float* signs,int strategy) {
    if(strategy!=2)for(int j=0;j<512;j+=8)
        _mm256_storeu_ps(y+j,sign_load(y+j,signs+j));
    if(strategy==0) {
        scalar_fwht(y,512);
    } else if(strategy==1) {
        for(int j=0;j<512;j+=8)_mm256_storeu_ps(y+j,h8(_mm256_loadu_ps(y+j)));
        for(int width=8;width<512;width*=2)
            for(int start=0;start<512;start+=2*width)
                for(int j=0;j<width;j+=8) {
                    __m256 a=_mm256_loadu_ps(y+start+j),b=_mm256_loadu_ps(y+start+width+j);
                    _mm256_storeu_ps(y+start+j,_mm256_add_ps(a,b));
                    _mm256_storeu_ps(y+start+width+j,_mm256_sub_ps(a,b));
                }
    } else {
        // V2: fuse signs into unnormalized register H64, then cross blocks.
        // No identity normalization/undo passes; final full512 scale unchanged.
        for(int j=0;j<512;j+=64)block64<false>(y+j,signs+j);
        for(int width=64;width<512;width*=2)
            for(int start=0;start<512;start+=2*width)
                for(int j=0;j<width;j+=8) {
                    __m256 a=_mm256_loadu_ps(y+start+j),b=_mm256_loadu_ps(y+start+width+j);
                    _mm256_storeu_ps(y+start+j,_mm256_add_ps(a,b));
                    _mm256_storeu_ps(y+start+width+j,_mm256_sub_ps(a,b));
                }
    }
}

extern "C" __declspec(dllexport) __attribute__((noinline)) int experimental_transform(
    const float* input,float* output,std::size_t n,int input_d,int mode,int rounds,
    const float* signs,const int32_t* permutation,int strategy) {
    if(!input||!output||!signs||input_d<1||input_d>448||rounds<1||rounds>3||mode<0||mode>1)return -1;
    const int d=mode==0?512:448;
    if(mode==1&&permutation)for(int j=0;j<d;j++)if(permutation[j]<0||permutation[j]>=d)return -2;
    for(std::size_t row=0;row<n;row++) {
        float* y=output+row*d;
        std::memcpy(y,input+row*input_d,input_d*sizeof(float));
        std::memset(y+input_d,0,(d-input_d)*sizeof(float));
        alignas(64) float scratch[512];
        if(mode==1&&!permutation&&rounds==2&&strategy==2) {
            for(int j=0;j<448;j+=64)fixed64_two(y+j,signs+j,signs+448+j);
            continue;
        }
        for(int r=0;r<rounds;r++) {
            if(mode==1) {
                if(r&&permutation) {
                    if(strategy==1) {
                        for(int j=0;j<d;j+=8) {
                            __m256i ids=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(permutation+j));
                            _mm256_store_ps(scratch+j,_mm256_i32gather_ps(y,ids,4));
                        }
                    } else {
                        for(int j=0;j<d;j++)scratch[j]=y[permutation[j]];
                    }
                    std::memcpy(y,scratch,d*sizeof(float));
                }
                for(int j=0;j<d;j+=64)block64(y+j,signs+r*d+j);
            } else {
                full512(y,signs+r*d,strategy);
            }
        }
        if(mode==0) {
            float scale=rounds==1?1.0f/std::sqrt(512.0f):rounds==2?1.0f/512.0f:1.0f/(512.0f*std::sqrt(512.0f));
            for(int j=0;j<512;j+=8)_mm256_storeu_ps(y+j,_mm256_mul_ps(_mm256_loadu_ps(y+j),_mm256_set1_ps(scale)));
        }
    }
    return 0;
}

// Native-only clocks. No Python loops, dense-matrix transforms or proxy QPS.
extern "C" __declspec(dllexport) double experimental_time_transform(
    const float* input,float* output,std::size_t n,int input_d,int mode,int rounds,
    const float* signs,const int32_t* permutation,int strategy,int repeats) {
    if(repeats<1)return -1;
    auto begin=std::chrono::steady_clock::now();
    for(int r=0;r<repeats;r++)
        if(experimental_transform(input,output,n,input_d,mode,rounds,signs,permutation,strategy))return -2;
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
}

extern "C" __declspec(dllexport) __attribute__((noinline)) int experimental_pipeline(
    const float* input,std::size_t n,int input_d,int mode,int rounds,
    const float* signs,const int32_t* permutation,int strategy,std::uintptr_t index_address,
    int64_t k,int batch,float* distances,int64_t* labels) {
    if(!index_address||k<1||batch<1)return -3;
    const faiss::Index* index=reinterpret_cast<const faiss::Index*>(index_address);
    const int d=mode==0?512:448;
    if(index->d!=d)return -4;
    try {
        for(std::size_t start=0;start<n;start+=batch) {
            std::size_t count=std::min<std::size_t>(batch,n-start);
            // Query-preprocessing scratch allocation is inside the timed loop.
            std::vector<float> transformed(count*d);
            int result=experimental_transform(input+start*input_d,transformed.data(),count,input_d,mode,
                rounds,signs,permutation,strategy);
            if(result)return result;
            index->search(count,transformed.data(),k,distances+start*k,labels+start*k);
        }
    } catch(...) {return -5;}
    return 0;
}

extern "C" __declspec(dllexport) int experimental_time_pipeline(
    const float* input,std::size_t n,int input_d,int mode,int rounds,
    const float* signs,const int32_t* permutation,int strategy,std::uintptr_t index_address,
    int64_t k,int batch,float* distances,int64_t* labels,int repeats,double* samples) {
    if(repeats<1||!samples)return -6;
    for(int r=0;r<repeats;r++) {
        auto begin=std::chrono::steady_clock::now();
        int result=experimental_pipeline(input,n,input_d,mode,rounds,signs,permutation,strategy,index_address,
            k,batch,distances,labels);
        if(result)return result;
        samples[r]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    }
    return 0;
}


