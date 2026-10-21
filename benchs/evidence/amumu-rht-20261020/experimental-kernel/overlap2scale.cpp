// Isolated compact 164+256 codec scorer. Original FAISS binaries stay frozen.
#include "overlap256.cpp"
#include <faiss/utils/distances.h>
#include <faiss/utils/Heap.h>
namespace overlap2 {
static float readf(const uint8_t* p){float v;std::memcpy(&v,p,4);return v;}
static int index(const uint8_t* c,int j,int b){int p=j*b,shift=p&7;unsigned v=c[p>>3];if(shift+b>8)v|=unsigned(c[(p>>3)+1])<<8;return(v>>shift)&((1<<b)-1);}
struct Scorer {
    int d,bits,groups,stride,payload;const float* table;std::vector<float> lut;float qnorm;
    Scorer(int dimension,int b,int g,int s,const float* t):d(dimension),bits(b),groups(g),stride(s),payload((d*b+7)/8),table(t),lut(b==3?d*8:payload*32){}
    void query(const float* q){
        qnorm=faiss::fvec_norm_L2sqr(q,d);
        if(bits==3){for(int j=0;j<d;j++)for(int k=0;k<8;k++)lut[j*8+k]=q[j]*table[k];}
        else for(int byte=0;byte<payload;byte++)for(int value=0;value<16;value++){
            int j=byte*(8/bits);
            lut[byte*32+value]=bits==2?q[j]*table[value&3]+q[j+1]*table[(value>>2)&3]:q[j]*table[value];
            lut[byte*32+16+value]=bits==2?q[j+2]*table[value&3]+q[j+3]*table[(value>>2)&3]:q[j+1]*table[value];
        }
    }
    void batch8(const uint8_t* base,float* output,int count){
        const uint8_t* c[8];for(int i=0;i<8;i++)c[i]=base+(i<count?i:0)*stride;
        __m256 cross=_mm256_setzero_ps();
        for(int group=0;group<groups;group++){
            int begin=groups==1?0:group==0?0:164;
            int end=groups==1?d:group==0?164:420;
            __m256 dot=_mm256_setzero_ps();
            if(bits==3){for(int j=begin;j<end;j++){
                __m256i ids=_mm256_setr_epi32(index(c[0],j,3),index(c[1],j,3),index(c[2],j,3),index(c[3],j,3),index(c[4],j,3),index(c[5],j,3),index(c[6],j,3),index(c[7],j,3));
                dot=_mm256_add_ps(dot,_mm256_i32gather_ps(lut.data()+j*8,ids,4));
            }}else for(int byte=begin*bits/8;byte<end*bits/8;byte++){
                __m256i v=_mm256_setr_epi32(c[0][byte],c[1][byte],c[2][byte],c[3][byte],c[4][byte],c[5][byte],c[6][byte],c[7][byte]);
                const float* t=lut.data()+byte*32;
                dot=_mm256_add_ps(dot,_mm256_i32gather_ps(t,_mm256_and_si256(v,_mm256_set1_epi32(15)),4));
                dot=_mm256_add_ps(dot,_mm256_i32gather_ps(t,_mm256_add_epi32(_mm256_srli_epi32(v,4),_mm256_set1_epi32(16)),4));
            }
            __m256 scale=_mm256_setr_ps(readf(c[0]+payload+group*4),readf(c[1]+payload+group*4),readf(c[2]+payload+group*4),readf(c[3]+payload+group*4),readf(c[4]+payload+group*4),readf(c[5]+payload+group*4),readf(c[6]+payload+group*4),readf(c[7]+payload+group*4));
            cross=_mm256_add_ps(cross,_mm256_mul_ps(scale,dot));
        }
        int offset=payload+groups*4;
        __m256 norms=_mm256_setr_ps(readf(c[0]+offset),readf(c[1]+offset),readf(c[2]+offset),readf(c[3]+offset),readf(c[4]+offset),readf(c[5]+offset),readf(c[6]+offset),readf(c[7]+offset));
        __m256 result=_mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps(qnorm),norms),_mm256_mul_ps(_mm256_set1_ps(2.f),cross));
        _mm256_storeu_ps(output,_mm256_max_ps(result,_mm256_setzero_ps()));
    }
};
static bool valid(int d,int b,int g,int stride){return(d==420||d==512)&&b>=2&&b<=4&&(g==1||(g==2&&d==420))&&stride>=(d*b+7)/8+g*4+4;}
}
extern "C" __declspec(dllexport) int overlap2_scores(const float* q,std::size_t nq,const uint8_t* codes,std::size_t nb,int d,int bits,int groups,int stride,const float* table,float* output){
    if(!overlap2::valid(d,bits,groups,stride))return -1;
    overlap2::Scorer scorer(d,bits,groups,stride,table);alignas(32)float values[8];
    for(std::size_t row=0;row<nq;row++){
        scorer.query(q+row*d);
        for(std::size_t i=0;i<nb;i+=8){int n=int(std::min<std::size_t>(8,nb-i));scorer.batch8(codes+i*stride,values,n);std::memcpy(output+row*nb+i,values,n*4);}
    }return 0;
}
extern "C" __declspec(dllexport) int overlap2_search(const float* q,std::size_t nq,const uint8_t* codes,std::size_t nb,int d,int bits,int groups,int stride,const float* table,int64_t k,float* distances,int64_t* labels){
    if(!overlap2::valid(d,bits,groups,stride)||k<1)return -1;
    overlap2::Scorer scorer(d,bits,groups,stride,table);alignas(32)float values[8];
    for(std::size_t row=0;row<nq;row++){
        scorer.query(q+row*d);float* heap=distances+row*k;int64_t* ids=labels+row*k;faiss::maxheap_heapify(k,heap,ids);
        for(std::size_t i=0;i<nb;i+=8){int n=int(std::min<std::size_t>(8,nb-i));scorer.batch8(codes+i*stride,values,n);
            for(int j=0;j<n;j++)if(faiss::CMax<float,int64_t>::cmp2(heap[0],values[j],ids[0],i+j))faiss::maxheap_replace_top(k,heap,ids,values[j],i+j);
        }faiss::maxheap_reorder(k,heap,ids);
    }return 0;
}
extern "C" __declspec(dllexport) int overlap2_time_pipeline(const float* input,const float* center,std::size_t nq,const uint8_t* codes,std::size_t nb,int d,int bits,int groups,int stride,const float* table,const float* signs,int strategy,int64_t k,int batch,float* distances,int64_t* labels,int repeats,double* samples){
    if(repeats<1||batch<1||!overlap2::valid(d,bits,groups,stride))return -1;
    for(int r=0;r<repeats;r++){
        auto start=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<nq;i+=batch){
            std::size_t n=std::min<std::size_t>(batch,nq-i);std::vector<float> res(n*420),rot(n*d);
            for(std::size_t row=0;row<n;row++)for(int j=0;j<420;j++)res[row*420+j]=input[(i+row)*420+j]-center[j];
            int error=d==420?overlap256_transform(res.data(),rot.data(),n,signs,strategy):experimental_transform(res.data(),rot.data(),n,420,0,1,signs,nullptr,2);
            if(error)return error;
            error=overlap2_search(rot.data(),n,codes,nb,d,bits,groups,stride,table,k,distances+i*k,labels+i*k);
            if(error)return error;
        }samples[r]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }return 0;
}
