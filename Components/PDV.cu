#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

__global__ void PDV_main(
    const float* __restrict__ rates_a,
    const float* __restrict__ rates_b,
    float* __restrict__ results,
    size_t numT,
    size_t num_neurons,
    size_t N_WARPS
);

__global__ void cosD_main(
    float* __restrict__ rates_a,
    float* __restrict__ rates_b,
    float* __restrict__ results,
    size_t numT,
    size_t num_neurons,
    size_t N_WARPS
);

py::array_t<float> compute_D_gpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> rates_a,
    py::array_t<float, py::array::c_style | py::array::forcecast> rates_b,
    int selection
)
{
    size_t num_neurons = rates_a.shape()[1]; // assuming both stimulus were presented to the same ntwrk configuration
    size_t num_runs = rates_a.shape()[0];

    size_t numT = 256;
    const size_t N_WARPS = numT/32;
    dim3 numBlocks(num_runs);
    dim3 threadsPerBlock(numT);

    const float* h_a = rates_a.data();
    const float* h_b = rates_b.data();
    size_t input_size = num_runs * num_neurons * sizeof(float);

    size_t output_size = num_runs * sizeof(float);

    float* g_a = nullptr;
    float* g_b = nullptr;
    float* g_res = nullptr;

    cudaMalloc(&g_a, input_size), cudaMalloc(&g_b, input_size), cudaMalloc(&g_res, output_size);
    cudaMemcpy(g_a, h_a, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, h_b, input_size, cudaMemcpyHostToDevice);

    switch(selection)
    {
        case 0: {
            size_t sharedMem_size = N_WARPS * sizeof(float);
            PDV_main<<<numBlocks, threadsPerBlock, sharedMem_size>>>(g_a, g_b, g_res, numT, num_neurons, N_WARPS);
            break;
        }
        case 1: {
            size_t sharedMem_size = 3 * N_WARPS * sizeof(float);
            cosD_main<<<numBlocks, threadsPerBlock, sharedMem_size>>>(g_a, g_b, g_res, numT, num_neurons, N_WARPS);
            break;
        }
        default: throw std::invalid_argument("Invalid selection of distance measure for the population-wise rate distance computation");
    }

    auto h_res = py::array_t<float>(num_runs);
    float* h_res_ptr = h_res.mutable_data();

    cudaMemcpy(h_res_ptr, g_res, output_size, cudaMemcpyDeviceToHost);

    cudaFree(g_a), cudaFree(g_b), cudaFree(g_res);

    return h_res;
};

__global__ void PDV_main( // ! not safe for any block dim other than 256 ! when this will work well, will add dynamic branching with templates
    const float* __restrict__ rates_a,
    const float* __restrict__ rates_b,
    float* __restrict__ results,
    size_t numT,
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

__global__ void cosD_main(
    float* __restrict__ rates_a,
    float* __restrict__ rates_b,
    float* __restrict__ results,
    size_t numT,
    size_t num_neurons,
    size_t N_WARPS
)
{
    extern __shared__ float b_mem[];
    float* b_dotprod = b_mem;
    float* b_mA = b_dotprod + N_WARPS;
    float* b_mB = b_mA + N_WARPS;

    const size_t tid = threadIdx.x;

    float p_dotprod = 0.0f;
    float p_mA = 0.0f;
    float p_mB = 0.0f;
    // unrolling isnt really useful, as the computing time is almost null compared to the cudamemcpy time anyway
    // and keeping the loop allows for doing the comp on various pops
    for (size_t i = tid; i < num_neurons; i += numT) {
        size_t g_idx = blockIdx.x * num_neurons + i;
        
        p_dotprod += rates_a[g_idx] * rates_b[g_idx];
        p_mA += rates_a[g_idx] * rates_a[g_idx];
        p_mB += rates_b[g_idx] * rates_b[g_idx];
    }

    constexpr unsigned int MASK = 0xffffffff;
    #pragma unroll
    for(size_t offset = 16; offset > 0; offset /= 2)
    {
        p_dotprod += __shfl_down_sync(MASK, p_dotprod, offset);
        p_mA += __shfl_down_sync(MASK, p_mA, offset);
        p_mB += __shfl_down_sync(MASK, p_mB, offset);
    }
    
    if(tid % 32 == 0)
    {
        b_dotprod[tid / 32] = p_dotprod;
        b_mA[tid / 32] = p_mA;
        b_mB[tid / 32] = p_mB;
    }
    __syncthreads();

    if(tid == 0)
    {
        float final_dotprod = 0.0f;
        float final_mA = 0.0f;
        float final_mB = 0.0f;

        #pragma unroll
        for(size_t i{0}; i < N_WARPS; i++)
        {
            final_dotprod += b_dotprod[i];
            final_mA += b_mA[i];
            final_mB += b_mB[i];
        }

        final_mA = sqrtf(final_mA);
        final_mB = sqrtf(final_mB);
        final_dotprod /= (final_mA * final_mB);

        final_dotprod = fminf(1.0f, fmaxf(-1.0f, final_dotprod));

        results[blockIdx.x] = (2.0f * acosf(final_dotprod)) / M_PIf;
    }
};

PYBIND11_MODULE(pdv_cuda, m)
{
    m.doc() = "PDV computation between 2 arrays of rates (runs x neurons) on gpu";
    m.def("compute_D_gpu", &compute_D_gpu, "computes PDV on gpu for 2 rates arrays");
}
