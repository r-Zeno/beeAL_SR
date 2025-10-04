#include <iostream>
#include "cnpy.h"
#include <cuda.h>
#include <vector>
#include <cmath>
#include <string>

template <typename T>
size_t getVectorbSize(const std::vector<T> &v);
__global__ void PDV_main(float* rates_a, float* rates_b, float* results_a, float* results_b, int num_neurons, int num_runs);
__device__ float warp_reduction();
__device__ float block_reduction();

int main()
{
    cnpy::NpyArray array_a = cnpy::npy_load("rates_od1.npy"); // will need to pass the precise path from python
    float* a = array_a.data<float>(); // cast into float32 and gives a pointer
    cnpy::NpyArray array_b = cnpy::npy_load("rates_od2.npy");
    float* b = array_b.data<float>();

    unsigned int num_neurons = a.size(); // assuming both stimulus were presented to the same ntwrk configuration
    unsigned int num_runs = a[0].size();

    // just create an array instead
    std::vector<float> res_0(num_runs);
    std::vector<float> res_1(num_runs);

    dim3 numBlocks(num_runs);
    dim3 threadsPerBlock(256);

    float* g_a = nullptr; // pointers to allocate arrays to (in global mem)
    float* g_b = nullptr;
    float* g_res0 = nullptr;
    float* g_res1 = nullptr;
    size_t a_size = getVectorbSize(a); // npy data is already in a row dominant, C++/CUDA friendly form, so no need for vetor logic
    size_t b_size = getVectorbSize(b);
    size_t res0_size = getVectorbSize(res_0);
    size_t res1_size = getVectorbSize(res_1);
    cudaMalloc(&g_a, a_size);
    cudaMalloc(&g_b, b_size);
    cudaMalloc(&g_res0, res0_size);
    cudaMalloc(&g_res1, res1_size);
    cudaMemcpy(g_a, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_res0, res_0, res0_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_res1, res_1, res1_size, cudaMemcpyHostToDevice);


    

    return 0;
}

// all of these are nice, but not useful here, since npy data i discovered is already a flat array!
template <typename T>
size_t getVectorbSize(const std::vector<T> &v){
    return v.size() * sizeof(T);
};
template <typename T>
size_t getVectorbSize(const std::vector<std::vector<T>> &v){
    size_t size = 0;
    for(const auto &vv : v){
        size += getVectorbSize(vv);
    }
    return size;
};

void toga(const std::vector<float> &v, float* dest_ptr){ // performs memcopy flattening the vector if necessary
    size_t v_size = getVectorbSize(v);
    cudaMemcpy(dest_ptr, v.data(), v_size, cudaMemcpyHostToDevice);
};
void toga(const std::vector<std::vector<float>> &v, float* dest_ptr){
    float* curr_ptr = dest_ptr;

    for(const auto &vv : v){

        size_t vv_size = getVectorbSize(vv);
        cudaMemcpy(curr_ptr, vv.data(), vv_size, cudaMemcpyHostToDevice);

        curr_ptr += vv.size();
    }
};

__global__ void PDV_main(float* rates_a, float* rates_b, float* results_a, float* results_b, int num_neurons, int num_runs){
    
};
