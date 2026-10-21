// Isolated experimental DLL. Frozen FAISS sources/binaries are not modified.
// Reuse the equally compiled optimized full512 attribution control.
#include "block_transform_fixed64.cpp"
#include <faiss/VectorTransform.h>

static inline void overlap_h256(float* y,const float* signs,int strategy) {
    if(strategy==2) {
        for(int j=0;j<256;j+=64)block64<false>(y+j,signs+j);
        for(int width=64;width<256;width*=2)
            for(int start=0;start<256;start+=2*width)
                for(int j=0;j<width;j+=8) {
                    __m256 a=_mm256_loadu_ps(y+start+j),b=_mm256_loadu_ps(y+start+width+j);
                    _mm256_storeu_ps(y+start+j,_mm256_add_ps(a,b));
                    _mm256_storeu_ps(y+start+width+j,_mm256_sub_ps(a,b));
                }
    } else {
        for(int j=0;j<256;j+=8)_mm256_storeu_ps(y+j,sign_load(y+j,signs+j));
        if(strategy==0)scalar_fwht(y,256);
        else {
            for(int j=0;j<256;j+=8)_mm256_storeu_ps(y+j,h8(_mm256_loadu_ps(y+j)));
            for(int width=8;width<256;width*=2)
                for(int start=0;start<256;start+=2*width)
                    for(int j=0;j<width;j+=8) {
                        __m256 a=_mm256_loadu_ps(y+start+j),b=_mm256_loadu_ps(y+start+width+j);
                        _mm256_storeu_ps(y+start+j,_mm256_add_ps(a,b));
                        _mm256_storeu_ps(y+start+width+j,_mm256_sub_ps(a,b));
                    }
        }
    }
    // Each embedded orthogonal H256 MUST normalize before the overlapping one.
    for(int j=0;j<256;j+=8)_mm256_storeu_ps(y+j,_mm256_mul_ps(_mm256_loadu_ps(y+j),_mm256_set1_ps(.0625f)));
}
extern "C" __declspec(dllexport) int overlap256_transform(
    const float* input,float* output,std::size_t n,const float* signs,int strategy) {
    if(!input||!output||!signs||strategy<0||strategy>2)return -1;
    for(std::size_t row=0;row<n;row++) {
        float* y=output+row*420;
        std::memcpy(y,input+row*420,420*sizeof(float));
        overlap_h256(y,signs,strategy);       // coordinates0..255
        overlap_h256(y+164,signs+256,strategy); // coordinates164..419, overlap92
    }
    return 0;
}
static int overlap_comparison_transform(
    const float* input,float* output,std::size_t n,int mode,const float* signs,int strategy,
    std::uintptr_t frozen_transform) {
    if(mode==0) {
        if(!frozen_transform)return -2;
        reinterpret_cast<const faiss::VectorTransform*>(frozen_transform)->apply_noalloc(n,input,output);
        return 0;
    }
    if(mode==1)return overlap256_transform(input,output,n,signs,strategy);
    if(mode==2)return experimental_transform(input,output,n,420,0,1,signs,nullptr,2);
    return -3;
}
extern "C" __declspec(dllexport) double overlap256_time_transform(
    const float* input,float* output,std::size_t n,int mode,const float* signs,int strategy,
    std::uintptr_t frozen_transform,int repeats) {
    if(repeats<1)return -1;
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<repeats;i++)
        if(overlap_comparison_transform(input,output,n,mode,signs,strategy,frozen_transform))return -2;
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
}
extern "C" __declspec(dllexport) int overlap256_pipeline(
    const float* input,const float* original_center,std::size_t n,int mode,const float* signs,int strategy,
    std::uintptr_t index_pointer,int64_t k,int batch,float* distances,int64_t* labels) {
    if(!input||!original_center||!index_pointer||k<1||batch<1||mode<0||mode>2)return -1;
    const auto* index=reinterpret_cast<const faiss::Index*>(index_pointer);
    const int output_d=mode==1?420:512;
    if(index->d!=(mode==2?512:420))return -2;
    try {
        for(std::size_t start=0;start<n;start+=batch) {
            const std::size_t count=std::min<std::size_t>(batch,n-start);
            std::vector<float> residual(count*420),rotated;
            for(std::size_t i=0;i<count;i++)for(int j=0;j<420;j++)
                residual[i*420+j]=input[(start+i)*420+j]-original_center[j];
            if(mode==0) {
                // Actual frozen FAISS IndexPreTransform: HR512 + actual leaf search.
                index->search(count,residual.data(),k,distances+start*k,labels+start*k);
            } else {
                rotated.resize(count*output_d);
                int code=overlap_comparison_transform(residual.data(),rotated.data(),count,mode,signs,strategy,0);
                if(code)return code;
                index->search(count,rotated.data(),k,distances+start*k,labels+start*k);
            }
        }
    } catch(...) {return -3;}
    return 0;
}
extern "C" __declspec(dllexport) int overlap256_time_pipeline(
    const float* input,const float* center,std::size_t n,int mode,const float* signs,int strategy,
    std::uintptr_t index_pointer,int64_t k,int batch,float* distances,int64_t* labels,
    int repeats,double* samples) {
    if(repeats<1||!samples)return -1;
    for(int i=0;i<repeats;i++) {
        const auto start=std::chrono::steady_clock::now();
        int code=overlap256_pipeline(input,center,n,mode,signs,strategy,index_pointer,k,batch,distances,labels);
        if(code)return code;
        samples[i]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    return 0;
}
