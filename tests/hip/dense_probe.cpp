// Focused prompt-projection benchmark; no model files or engine changes.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "strata/prefill/gemm.hpp"
#include "../../src/prefill/wmma_gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=hipSuccess) { std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); std::exit(2); } } while(0)
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t bytes) { CK(hipMalloc(&p,bytes)); }
    ~Buffer() { if(p) hipFree(p); }
};
struct Shape { const char* name; bool bf16; int n,k; };

int main() {
    setvbuf(stdout,nullptr,_IONBF,0);
    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    int failures=0;
    {
        strata::prefill::Gemm gemm;
        std::string err;
        if(!gemm.init(stream,1<<20,err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 2; }
        const Shape shapes[]={
            {"hc_down",true,320,10240}, {"indexer",true,512,2560},
            {"gdn_in",false,10240,2560}, {"q_proj",false,12288,2560},
            {"kv_proj",false,512,2560}, {"out_proj",false,2560,6144},
            {"shared_gu",false,1280,2560}, {"shared_down",false,2560,640}
        };
        for(int t:{2048,8192}) for(const auto& sh:shapes) {
            const int n=sh.n,k=sh.k;
            std::mt19937 rng(20261019+t+n+k);
            std::uniform_real_distribution<float> dist(-0.25f,0.25f);
            std::vector<uint16_t> x((size_t)t*k),w((size_t)n*k);
            auto encode=[&](float v) { if(sh.bf16) return hip_bfloat16(v).data; __half h=__float2half_rn(v); uint16_t b; std::memcpy(&b,&h,2); return b; };
            for(auto& a:x) a=encode(dist(rng));
            for(auto& a:w) a=encode(dist(rng));
            Buffer dx(x.size()*2),dw(w.size()*2),dy((size_t)t*n*4);
            CK(hipMemcpyAsync(dx.p,x.data(),x.size()*2,hipMemcpyHostToDevice,stream));
            CK(hipMemcpyAsync(dw.p,w.data(),w.size()*2,hipMemcpyHostToDevice,stream));
            CK(hipStreamSynchronize(stream));
            std::vector<float> reference;
            std::vector<double> times[3];
            bool supported[3]={true,true,!sh.bf16};
            auto launch=[&](int method) {
                auto X=(const uint16_t*)dx.p,W=(const uint16_t*)dw.p; auto Y=(float*)dy.p;
                if(method==0) { if(sh.bf16) gemm.bf16(X,W,Y,t,n,k,n,0); else gemm.f16(X,W,Y,t,n,k,n,0); return true; }
                if(method==1) return sh.bf16 ? strata_wmma_gemm_bf16(X,W,Y,t,n,k,n,0,stream) : strata_wmma_gemm_f16(X,W,Y,t,n,k,n,0,stream);
                return strata_pf_gemm_f16(X,W,Y,t,n,k,n,0,stream);
            };
            // Establish numerical differences separately from timing. Only finite bounded inputs.
            for(int method=0;method<3;++method) {
                if(!supported[method]) continue;
                if(!launch(method)) { supported[method]=false; continue; }
                CK(hipStreamSynchronize(stream));
                std::vector<float> y((size_t)t*n);
                CK(hipMemcpy(y.data(),dy.p,y.size()*4,hipMemcpyDeviceToHost));
                if(method==0) reference=y;
                double d2=0,r2=0,max_abs=0;
                size_t different=0,nonfinite=0;
                for(size_t i=0;i<y.size();++i) {
                    nonfinite+=!std::isfinite(y[i]);
                    different+=std::memcmp(&y[i],&reference[i],4)!=0;
                    double d=(double)y[i]-reference[i]; d2+=d*d; r2+=(double)reference[i]*reference[i]; max_abs=std::max(max_abs,std::abs(d));
                }
                double rel=std::sqrt(d2/std::max(r2,1e-300));
                failures+=(nonfinite!=0 || rel>1e-4);
                std::printf("{\"kind\":\"numerics\",\"shape\":\"%s\",\"T\":%d,\"N\":%d,\"K\":%d,\"method\":%d,\"different\":%zu,\"nonfinite\":%zu,\"relative_l2\":%.9g,\"max_abs\":%.9g}\n",sh.name,t,n,k,method,different,nonfinite,rel,max_abs);
            }
            for(int round=0;round<3;++round) for(int step=0;step<3;++step) {
                int method=(round%2) ? 2-step : step;
                if(!supported[method]) continue;
                for(int r=0;r<3;++r) if(!launch(method)) return 3;
                hipEvent_t a,b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
                CK(hipEventRecord(a,stream));
                for(int r=0;r<20;++r) if(!launch(method)) return 3;
                CK(hipEventRecord(b,stream)); CK(hipEventSynchronize(b));
                float ms=0; CK(hipEventElapsedTime(&ms,a,b));
                times[method].push_back(ms/20.0);
                CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
            }
            for(int method=0;method<3;++method) if(supported[method]) {
                auto v=times[method]; std::sort(v.begin(),v.end());
                std::printf("{\"kind\":\"timing\",\"shape\":\"%s\",\"dtype\":\"%s\",\"T\":%d,\"N\":%d,\"K\":%d,\"method\":%d,\"median_ms\":%.6f,\"round_ms\":[%.6f,%.6f,%.6f]}\n",sh.name,sh.bf16?"bf16":"f16",t,n,k,method,v[1],times[method][0],times[method][1],times[method][2]);
            }
        }
    }
    CK(hipStreamDestroy(stream));
    std::printf("failures=%d\n",failures);
    return failures?1:0;
}
