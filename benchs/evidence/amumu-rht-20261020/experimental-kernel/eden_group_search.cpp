// Native grouped EDEN search and a SAME-kernel global-scale control.
// Layout0 BF16 groups448;1 FP32 groups448;2 frozen global512;3 frozen global448.
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <limits>
#include <faiss/utils/distances.h>
#include <faiss/utils/Heap.h>

extern "C" int experimental_transform(const float*,float*,std::size_t,int,int,int,const float*,const int32_t*,int);
namespace {
float rf(const uint8_t* p){float v;std::memcpy(&v,p,4);return v;}
float rb(const uint8_t* p){uint16_t b;std::memcpy(&b,p,2);uint32_t u=uint32_t(b)<<16;float v;std::memcpy(&v,&u,4);return v;}
int index_at(const uint8_t* code,int j,int bits){int shift=(j*bits)&7,byte=(j*bits)>>3;uint32_t v=code[byte];if(shift+bits>8)v|=uint32_t(code[byte+1])<<8;return(v>>shift)&((1<<bits)-1);}
struct Layout {
    int bits,stride,kind,d,groups,width,payload,norm_offset;
    Layout(int b,int s,int k):bits(b),stride(s),kind(k),d(k==2?512:448),groups(k<=1?7:1),width(d/groups),
        payload(d*b/8),norm_offset(payload+(k==0?14:k==1?28:0)){}
    float scale(const uint8_t* code,int g)const{return kind==0?rb(code+payload+2*g):kind==1?rf(code+payload+4*g):rf(code+payload+4);}
    bool valid(const uint8_t* codes,std::size_t n)const {
        if(bits<2||bits>4||kind<0||kind>3||stride<norm_offset+4+(kind>=2?4:0))return false;
        for(std::size_t i=0;i<n;i++){
            auto c=codes+i*stride;float norm=rf(c+norm_offset);
            if(!std::isfinite(norm)||norm<0)return false;
            for(int g=0;g<groups;g++)if(!std::isfinite(scale(c,g)))return false;
        }return true;
    }
};
struct Query {
    const Layout& l;const float* table;std::vector<float> q,lut;float norm;
    Query(const Layout& layout,const float* values):l(layout),table(values),q(l.d),lut(l.bits==3?l.d*8:l.payload*32){}
    bool set(const float* input,const float* center){
        norm=faiss::fvec_L2sqr(input,center,l.d);
        if(!std::isfinite(norm))return false;
        for(int j=0;j<l.d;j++){q[j]=input[j]-center[j];if(!std::isfinite(q[j]))return false;}
        if(l.bits==3){for(int j=0;j<l.d;j++)for(int k=0;k<8;k++)lut[j*8+k]=q[j]*table[k];}
        else {
            int per=8/l.bits;
            for(int byte=0;byte<l.payload;byte++)for(int value=0;value<16;value++){
                int j=byte*per;float low,high;
                if(l.bits==2){low=q[j]*table[value&3]+q[j+1]*table[(value>>2)&3];high=q[j+2]*table[value&3]+q[j+3]*table[(value>>2)&3];}
                else {low=q[j]*table[value];high=q[j+1]*table[value];}
                lut[byte*32+value]=low;lut[byte*32+16+value]=high;
            }
        }return true;
    }
    float scalar(const uint8_t* code)const {
        float cross=0;
        for(int g=0;g<l.groups;g++){
            float dot=0;for(int j=0;j<l.width;j++){int coord=g*l.width+j;dot+=q[coord]*table[index_at(code,coord,l.bits)];}
            cross+=l.scale(code,g)*dot;
        }
        float value=norm+rf(code+l.norm_offset)-2.f*cross;return value<0?0:value;
    }
    void batch8(const uint8_t* base,float* output)const {
        const uint8_t* c[8];for(int i=0;i<8;i++)c[i]=base+i*l.stride;
        __m256 cross=_mm256_setzero_ps();
        const __m256i mask=_mm256_set1_epi32(15),offset=_mm256_set1_epi32(16);
        for(int g=0;g<l.groups;g++){
            __m256 dot=_mm256_setzero_ps();
            if(l.bits==3){
                for(int j=0;j<l.width;j++){
                    int coord=g*l.width+j;
                    __m256i ids=_mm256_setr_epi32(index_at(c[0],coord,3),index_at(c[1],coord,3),index_at(c[2],coord,3),index_at(c[3],coord,3),
                        index_at(c[4],coord,3),index_at(c[5],coord,3),index_at(c[6],coord,3),index_at(c[7],coord,3));
                    dot=_mm256_add_ps(dot,_mm256_i32gather_ps(lut.data()+coord*8,ids,4));
                }
            }else {
                int bytes=l.width*l.bits/8,start=g*bytes;
                for(int j=0;j<bytes;j++){
                    int b=start+j;
                    __m256i values=_mm256_setr_epi32(c[0][b],c[1][b],c[2][b],c[3][b],c[4][b],c[5][b],c[6][b],c[7][b]);
                    const float* table=lut.data()+b*32;
                    dot=_mm256_add_ps(dot,_mm256_i32gather_ps(table,_mm256_and_si256(values,mask),4));
                    dot=_mm256_add_ps(dot,_mm256_i32gather_ps(table,_mm256_add_epi32(_mm256_srli_epi32(values,4),offset),4));
                }
            }
            __m256 scales=_mm256_setr_ps(l.scale(c[0],g),l.scale(c[1],g),l.scale(c[2],g),l.scale(c[3],g),
                l.scale(c[4],g),l.scale(c[5],g),l.scale(c[6],g),l.scale(c[7],g));
            cross=_mm256_add_ps(cross,_mm256_mul_ps(scales,dot));
        }
        __m256 norms=_mm256_setr_ps(rf(c[0]+l.norm_offset),rf(c[1]+l.norm_offset),rf(c[2]+l.norm_offset),rf(c[3]+l.norm_offset),
            rf(c[4]+l.norm_offset),rf(c[5]+l.norm_offset),rf(c[6]+l.norm_offset),rf(c[7]+l.norm_offset));
        __m256 result=_mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps(norm),norms),_mm256_mul_ps(_mm256_set1_ps(2.f),cross));
        _mm256_storeu_ps(output,_mm256_max_ps(result,_mm256_setzero_ps()));
    }
};
int scores(const float* queries,std::size_t nq,const uint8_t* codes,std::size_t nb,const Layout& l,const float* table,const float* center,
    float* output,int strategy){
    Query query(l,table);
    for(std::size_t q=0;q<nq;q++){
        if(!query.set(queries+q*l.d,center))return -2;
        std::size_t i=0;
        if(strategy)for(;i+8<=nb;i+=8)query.batch8(codes+i*l.stride,output+q*nb+i);
        for(;i<nb;i++)output[q*nb+i]=query.scalar(codes+i*l.stride);
    }return 0;
}
int search(const float* queries,std::size_t nq,const uint8_t* codes,std::size_t nb,const Layout& l,const float* table,const float* center,
    int64_t k,float* distances,int64_t* labels,int strategy){
    Query query(l,table);alignas(32) float values[8];
    for(std::size_t q=0;q<nq;q++){
        if(!query.set(queries+q*l.d,center))return -2;
        float* heap=distances+q*k;int64_t* ids=labels+q*k;
        faiss::maxheap_heapify(k,heap,ids);
        std::size_t i=0;
        if(strategy)for(;i+8<=nb;i+=8){
            query.batch8(codes+i*l.stride,values);
            for(int j=0;j<8;j++)if(faiss::CMax<float,int64_t>::cmp2(heap[0],values[j],ids[0],i+j))
                faiss::maxheap_replace_top(k,heap,ids,values[j],i+j);
        }
        for(;i<nb;i++){
            float value=query.scalar(codes+i*l.stride);
            if(faiss::CMax<float,int64_t>::cmp2(heap[0],value,ids[0],i))faiss::maxheap_replace_top(k,heap,ids,value,i);
        }
        faiss::maxheap_reorder(k,heap,ids);
    }return 0;
}
}
extern "C" __declspec(dllexport) int eden_codes_scores(
    const float* q,std::size_t nq,const uint8_t* codes,std::size_t nb,int bits,int stride,int kind,
    const float* table,const float* center,float* output,int strategy){
    Layout l(bits,stride,kind);if(!l.valid(codes,nb))return -1;
    return scores(q,nq,codes,nb,l,table,center,output,strategy);
}
extern "C" __declspec(dllexport) int eden_codes_search(
    const float* q,std::size_t nq,const uint8_t* codes,std::size_t nb,int bits,int stride,int kind,
    const float* table,const float* center,int64_t k,float* distances,int64_t* labels,int strategy){
    Layout l(bits,stride,kind);if(!l.valid(codes,nb)||k<1)return -1;
    return search(q,nq,codes,nb,l,table,center,k,distances,labels,strategy);
}
extern "C" __declspec(dllexport) int eden_codes_time_pipeline(
    const float* input,std::size_t nq,const uint8_t* codes,std::size_t nb,int bits,int stride,int kind,
    const float* table,const float* center,const float* signs,const int32_t* permutation,int transform_strategy,
    int64_t k,int batch,float* distances,int64_t* labels,int codec_strategy,int repeats,double* samples){
    Layout l(bits,stride,kind);if(!l.valid(codes,nb)||k<1||batch<1||repeats<1)return -1;
    const int mode=kind==2?0:1,rounds=kind==2?1:2;
    for(int r=0;r<repeats;r++){
        auto begin=std::chrono::steady_clock::now();
        for(std::size_t start=0;start<nq;start+=batch){
            std::size_t n=std::min<std::size_t>(batch,nq-start);std::vector<float> transformed(n*l.d);
            int result=experimental_transform(input+start*420,transformed.data(),n,420,mode,rounds,signs,permutation,transform_strategy);
            if(result)return result;
            result=search(transformed.data(),n,codes,nb,l,table,center,k,distances+start*k,labels+start*k,codec_strategy);
            if(result)return result;
        }
        samples[r]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    }return 0;
}
