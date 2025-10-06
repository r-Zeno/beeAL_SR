#include <iostream>
#include "cnpy.h"
#include <cuda.h>
#include <vector>
#include <cmath>
#include <string>

template <size_t numT>
__global__ void PDV_main(
    const float* __restrict__ rates_a,
    const float* __restrict__ rates_b,
    float* __restrict__ results,
    size_t num_neurons,
    size_t N_WARPS
);
void toga(float* dev_ptr, float* host_ptr, size_t num_elements);

int main()
{
    cnpy::NpyArray array_a = cnpy::npy_load("rates_od1.npy"); // will need to pass the precise path from python
    float* a = array_a.data<float>(); // cast into float32 and create a pointer
    size_t a_num = array_a.num_vals;

    cnpy::NpyArray array_b = cnpy::npy_load("rates_od2.npy");
    float* b = array_b.data<float>();
    size_t b_num = array_b.num_vals;

    size_t num_neurons = array_b.shape[1]; // assuming both stimulus were presented to the same ntwrk configuration
    size_t num_runs = array_a.shape[0];

    size_t numT = 256; //hardcoded for now, should make it set by simulator later to benchmark
    dim3 numBlocks(num_runs);
    dim3 threadsPerBlock(numT);

    float* res;
    res = new float[num_runs]{0.0};
    size_t res_num = num_runs;

    float* g_a = nullptr; // pointers to allocate arrays to (in global mem)
    float* g_b = nullptr;
    float* g_res = nullptr;
    
    toga(g_a, a, a_num), toga(g_b, b, b_num), toga(g_res, res, res_num);

    constexpr size_t N_WARPS = numT/32;
    size_t sharedMem_size = N_WARPS * sizeof(float);
    PDV_main<numT><<numBlocks, threadsPerBlock, sharedMem_size>>(g_a, g_b, g_res, num_neurons, N_WARPS);

    cudaMemcpy(res, g_res, num_runs*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(g_a), cudaFree(g_b), cudaFree(g_res);

    // now write .npy

    delete[] res;

    return 0;
}

// toga just doesnt work, passing and modifying the pointer's pointer is a mess, remove it :(
void toga(float** dev_ptr, float* host_ptr, size_t num_elements)
{ // to add error logging here
    size_t size = num_elements*sizeof(float);

    cudaMalloc(dev_ptr, size);
    cudaMemcpy(*dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);

};

template <size_t numT>
__global__ void PDV_main( // ! not safe for any block dim other than 256 ! when this will work well, will add dynamic branching with templates
    const float* __restrict__ rates_a,
    const float* __restrict__ rates_b,
    float* __restrict__ results,
    size_t num_neurons,
    size_t N_WARPS
)
{
    extern __shared__ float b_sum[];
    const size_t tid = threadIdx.x;

    size_t i = tid + blockIdx.x*num_neurons;
    float p_sum = 0.0f;
    // unrolled loop for summing distances, controlling which threads get which neurons
    p_sum = (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    i += numT;
    p_sum += (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    i += numT;
    p_sum += (rates_a[i] - rates_b[i]) * (rates_a[i] - rates_b[i]);
    i += numT;
    size_t next_neuron_idx = tid + 3*numT;
    if(next_neuron_idx < num_neurons)
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
    
    if(tid == 0)
    {
        #pragma unroll
        for(size_t i{0}; i < N_WARPS; i++)
        {
            final_sum += b_sum[i];
        }

        results[blockIdx.x] = sqrtf(final_sum);
    }
};
