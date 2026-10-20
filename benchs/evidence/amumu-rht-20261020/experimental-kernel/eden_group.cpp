// Experimental EDEN seven-group codec. Packed byte layout: NO C++ structs.
// Scales use ORIGINAL EDEN norm2/<code,residual>, then BF16 round-to-nearest-even.
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <immintrin.h>

static uint16_t bf16(float value) {
    uint32_t u;std::memcpy(&u,&value,4);
    if((u&0x7f800000u)==0x7f800000u&&(u&0x007fffffu))return uint16_t((u>>16)|0x40);
    return uint16_t((u+0x7fffu+((u>>16)&1))>>16);
}
static float unbf16(const uint8_t* source) {
    uint16_t b;std::memcpy(&b,source,2);uint32_t u=uint32_t(b)<<16;
    float f;std::memcpy(&f,&u,4);return f;
}
static float read_float(const uint8_t* p) {float f;std::memcpy(&f,p,4);return f;}
static int code_index(const uint8_t* code,int coordinate,int bits) {
    const int offset=coordinate*bits,shift=offset&7,byte=offset>>3;
    uint32_t v=code[byte];if(shift+bits>8)v|=uint32_t(code[byte+1])<<8;
    return (v>>shift)&((1<<bits)-1);
}
extern "C" __declspec(dllexport) void eden_bf16_convert(const float* values,uint16_t* output,std::size_t n) {
    for(std::size_t i=0;i<n;i++)output[i]=bf16(values[i]);
}
extern "C" __declspec(dllexport) int eden_group_encode(
    const float* input,const float* center,std::size_t n,int bits,int stride,int scale_format,
    const float* trained,uint8_t* output) {
    if(bits<2||bits>4||scale_format<0||scale_format>1)return -1;
    const int payload=448*bits/8,scale_bytes=scale_format?28:14;
    if(stride<payload+scale_bytes+4)return -2;
    const float* boundaries=trained+(1<<bits);
    for(std::size_t i=0;i<n;i++) {
        uint8_t* code=output+i*stride;std::memset(code,0,stride);
        float residual[448],whole_norm=0;
        for(int j=0;j<448;j++) {
            residual[j]=input[i*448+j]-(center?center[j]:0.f);
            if(!std::isfinite(residual[j]))return -3;
            whole_norm+=residual[j]*residual[j];
        }
        if(!std::isfinite(whole_norm))return -3;
        if(whole_norm<=std::numeric_limits<float>::epsilon())continue;
        std::memcpy(code+payload+scale_bytes,&whole_norm,4);
        for(int group=0;group<7;group++) {
            float norm=0;for(int j=0;j<64;j++)norm+=residual[group*64+j]*residual[group*64+j];
            if(norm<=std::numeric_limits<float>::epsilon())continue;
            const float inv=1.f/std::sqrt(norm);double dot=0;
            for(int j=0;j<64;j++) {
                float value=residual[group*64+j]*8.f*inv;
                int index=int(std::upper_bound(boundaries,boundaries+(1<<bits)-1,value)-boundaries);
                int offset=(group*64+j)*bits,byte=offset>>3,shift=offset&7;
                uint16_t packed=uint16_t(index)<<shift;
                code[byte]|=packed&255;if(shift+bits>8)code[byte+1]|=packed>>8;
                dot+=double(trained[index])*residual[group*64+j];
            }
            float scale=float(double(norm)/dot);
            // Same nonfinite-scale zero policy as original native EDEN64.
            if(!std::isfinite(scale))scale=0;
            if(scale_format)std::memcpy(code+payload+4*group,&scale,4);
            else {uint16_t rounded=bf16(scale);std::memcpy(code+payload+2*group,&rounded,2);}
        }
    }
    return 0;
}
extern "C" __declspec(dllexport) int eden_group_decode(
    const uint8_t* codes,std::size_t n,int bits,int stride,int scale_format,const float* table,
    const float* center,float* output) {
    if(bits<2||bits>4||scale_format<0||scale_format>1)return -1;
    const int payload=448*bits/8;
    if(stride<payload+(scale_format?32:18))return -2;
    for(std::size_t i=0;i<n;i++)for(int g=0;g<7;g++) {
        const uint8_t* code=codes+i*stride;
        const float scale=scale_format?read_float(code+payload+4*g):unbf16(code+payload+2*g);
        if(!std::isfinite(scale))return -3;
        for(int j=0;j<64;j++) {
            int coordinate=g*64+j;
            output[i*448+coordinate]=(center?center[coordinate]:0.f)+scale*table[code_index(code,coordinate,bits)];
        }
    }
    return 0;
}
extern "C" __declspec(dllexport) int eden_group_scores_reference(
    const float* queries,std::size_t nq,const uint8_t* codes,std::size_t nb,int bits,int stride,int scale_format,
    const float* table,const float* center,float* output) {
    const int payload=448*bits/8,scale_bytes=scale_format?28:14;
    if(bits<2||bits>4||stride<payload+scale_bytes+4)return -1;
    for(std::size_t q=0;q<nq;q++) {
        float residual[448],qnorm=0;
        for(int j=0;j<448;j++) {
            residual[j]=queries[q*448+j]-(center?center[j]:0.f);
            if(!std::isfinite(residual[j]))return -2;
            qnorm+=residual[j]*residual[j];
        }
        if(!std::isfinite(qnorm))return -2;
        for(std::size_t i=0;i<nb;i++) {
            const uint8_t* code=codes+i*stride;float cross=0;
            for(int g=0;g<7;g++) {
                const float scale=scale_format?read_float(code+payload+4*g):unbf16(code+payload+2*g);
                if(!std::isfinite(scale))return -3;
                float dot=0;for(int j=0;j<64;j++)dot+=residual[g*64+j]*table[code_index(code,g*64+j,bits)];
                cross+=scale*dot;
            }
            const float original_norm=read_float(code+payload+scale_bytes);
            if(!std::isfinite(original_norm)||original_norm<0)return -4;
            const float score=qnorm+original_norm-2.f*cross;
            output[q*nb+i]=score<0?0:score;
        }
    }
    return 0;
}
