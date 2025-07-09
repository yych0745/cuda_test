#include <iostream>
#include <cuda_runtime.h>

const int PER_THREAD_BLOCK = 256;

// shared memory
template<const int NUM_PER_BLOCK>
__global__ void reduce(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lanid = tid % 32;
    int warpid = tid / 32;
    float sum = 0;
    input = input + bid * NUM_PER_BLOCK;
    __shared__ float s_sum[32];
    for (int i = 0; i < NUM_PER_BLOCK / PER_THREAD_BLOCK; i++) {
        sum += input[tid + i * PER_THREAD_BLOCK];
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (lanid == 0) {
        s_sum[warpid] = sum;
    }
    __syncthreads();
    if (warpid == 0) {
        sum = tid * 32 < PER_THREAD_BLOCK ? s_sum[tid] : 0;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    if (tid == 0) {
        output[bid] = sum;
    }
}


bool check_equal(float a, float b) {
    return abs(a - b) < 0.005;
}


int main() {
    const int N=32*1024*1024;
    float *input = (float *)malloc(N * sizeof(float));

    // int block_num = N / PER_THREAD_BLOCK / 2;
    const int block_num = 1024 * 32;
    float *cpu_output = (float *)malloc(block_num * sizeof(float));
    float *output = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = drand48() * 2 + 0.1;
    }

    const int NUM_PER_BLOCK = N / block_num;
    for (int i = 0; i < block_num; i++) {
            cpu_output[i] = 0;
            for (int j = 0; j < NUM_PER_BLOCK; j++) {
                cpu_output[i] += input[i * NUM_PER_BLOCK + j];
            }
        }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, block_num * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(block_num);
    dim3 block(PER_THREAD_BLOCK);
    std::cout << "NUM_PER_BLOCK " << NUM_PER_BLOCK << std::endl;
    reduce<NUM_PER_BLOCK><<<grid, block>>>(d_input, d_output, N);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize(); // 确保printf输出可见
    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    bool is_equal = true;
    for (int i = 0; i < block_num; i++) {
        if (!check_equal(cpu_output[i], output[i])) {
            std::cout << "Error at block " << i << " " << cpu_output[i] << " " << output[i] << std::endl;
            is_equal = false;
        }
    }
    if (is_equal) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(cpu_output);
    return 0;
}
