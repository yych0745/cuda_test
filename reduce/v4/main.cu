#include <iostream>
#include <cuda_runtime.h>

const int PER_THREAD_BLOCK = 256;

// shared memory
__global__ void reduce(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ float s_input[PER_THREAD_BLOCK];
    input = input + bid * blockDim.x * 2;
    s_input[tid] = input[tid] + input[tid + PER_THREAD_BLOCK];
    __syncthreads();
    for (int i = 1; i < PER_THREAD_BLOCK; i *= 2) {
        int factor = PER_THREAD_BLOCK / 2 / i;
        if (tid < factor) {
            s_input[tid] += s_input[tid + factor];
        } 
        __syncthreads();
    }
    if (tid == 0) {
        output[bid] = s_input[0];
    }
}


bool check_equal(float a, float b) {
    return abs(a - b) < 0.005;
}


int main() {
    const int N=32*1024*1024;
    float *input = (float *)malloc(N * sizeof(float));

    int block_num = N / PER_THREAD_BLOCK / 2;
    float *cpu_output = (float *)malloc(block_num * sizeof(float));
    float *output = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = drand48() * 2 + 0.1;
    }
    for (int i = 0; i < block_num; i++) {
        cpu_output[i] = 0;
        for (int j = 0; j < 2 * PER_THREAD_BLOCK; j++) {
            cpu_output[i] += input[i * 2 * PER_THREAD_BLOCK + j];
        }
    }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, block_num * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid(block_num);
    dim3 block(PER_THREAD_BLOCK);

    reduce<<<grid, block>>>(d_input, d_output, N);
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
