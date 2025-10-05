#include <iostream>
#include "cnpy.h"
#include <cuda.h>
#include <vector>
#include <cmath>
#include <string>


void toga(float* dev_ptr, const float &v);

int main()
{
    cnpy::NpyArray array_a = cnpy::npy_load("rates_od1.npy"); // will need to pass the precise path from python
    float* a = array_a.data<float>(); // cast into float32 and create a pointer
    cnpy::NpyArray array_b = cnpy::npy_load("rates_od2.npy");
    float* b = array_b.data<float>();

    size_t num_neurons = sizeof(*a); // assuming both stimulus were presented to the same ntwrk configuration
    size_t num_runs = sizeof(*a[0]);

    size_t numT = 256; //hardcoded for now, should make it set by simulator later to benchmark
    dim3 numBlocks(num_runs);
    dim3 threadsPerBlock(numT);

    float* res;
    res = new float[num_runs]{0.0};

    float* g_a = nullptr; // pointers to allocate arrays to (in global mem)
    float* g_b = nullptr;
    float* g_res = nullptr;
    
    toga(g_a, a), toga(g_b, b), toga(g_res, res);

    PDV_main<numT><<numBlocks, threadsPerBlock>>();

    
    delete[] res

    return 0;
}

void toga(float* dev_ptr, const float &v)
{ // to add error logging here

    size_t v_size = sizeof(v) * sizeof(v[0]);
    
    cudaMalloc(dev_ptr, v_size);

    cudaMemcpy(dev_ptr, v, v_size, cudaMemcpyHostToDevice);

};

template <size_t numT>
__global__ void PDV_main( // ! not safe for any block dim other than 256 ! when this will work well, will add dynamic branching with templates
    const float* __restrict__ rates_a,
    const float* __restrict__ rates_b,
    const float* __restrict__ results,
    size_t num_neurons,
    size_t num_runs
)
{
    extern __shared__ float b_sum[];
    const size_t tid = threadIdx.x;
    constexpr size_t N_WARPS = numT/size_t 32;

    size_t i = threadIdx.x;
    float p_sum = 0.0f;
    // unrolled loop for summing distances, controlling which threads get which neurons
    p_sum = (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    i += numT;
    p_sum += (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    i += numT;
    p_sum += (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    i += numT;
    if(i < num_neurons)
    {
        p_sum += (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    }

    // now sum the partial sums from all threads at warp level (not necessary here, but
    // generally good for performance to operate reduction within warps using registry before going to shared memory)
    constexpr unsigned int MASK = 0xffffffff;
    #pragma unroll
    for(size_t offset = 16; offset > 0; offset /= 2)
    {
        p_sum += __shfl_down_sync(MASK, p_sum, offset);
    }

    if(tid % 32 == 0)
    {
        b_sum[tid / 32] = p_sum;
    }
    __syncthreads();

    float final_sum = 0.0f;
    #pragma unroll
    for(size_t i{0}; i < N_WARPS; i++)
    {
        final_sum += b_sum[i];
    }

    if(tid == 0)
    {
        results[blockIdx.x] = final_sum 
    }

    // now to copy results to host
};
