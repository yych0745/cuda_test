#include <iostream>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// shared memory
template<const uint M_NUM_PER_BLOCK, const uint N_NUM_PER_BLOCK, const uint K_NUM_PER_BLOCK, const uint M_NUM_PER_THREAD, const uint N_NUM_PER_THREAD, const uint K_NUM_PER_THREAD>
__global__ void sgeem(float *A, float *B, float *C, int M, int N, int K) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    A = A + bidy * M_NUM_PER_BLOCK * K;
    B = B + bidx * N_NUM_PER_BLOCK;
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float a_reg[K_NUM_PER_THREAD];
    float b_reg[N_NUM_PER_THREAD];
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {{0.f}};
        // global -> shared
    for (int i = 0; i < K; i += K_NUM_PER_BLOCK) {
        for (int j = 0; j < M_NUM_PER_THREAD; ++j) {
            FETCH_FLOAT4(a_shared[(tidy) * M_NUM_PER_THREAD + j][tidx * K_NUM_PER_THREAD]) = FETCH_FLOAT4(A[K * (tidy * M_NUM_PER_THREAD + j) + i + tidx * K_NUM_PER_THREAD]);
        }
        for (int j = 0; j < K_NUM_PER_THREAD; ++j) {
            FETCH_FLOAT4(b_shared[(tidy) * K_NUM_PER_THREAD + j][tidx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(B[N * (tidy * N_NUM_PER_THREAD + j + i) + tidx * N_NUM_PER_THREAD]);
        }
        __syncthreads();
        for (int ii = 0; ii < K_NUM_PER_BLOCK; ++ii) { 
        // shared -> register
            a_reg[0] = a_shared[tidy * M_NUM_PER_THREAD][ii];
            a_reg[1] = a_shared[tidy * M_NUM_PER_THREAD + 1][ii];
            a_reg[2] = a_shared[tidy * M_NUM_PER_THREAD + 2][ii];
            a_reg[3] = a_shared[tidy * M_NUM_PER_THREAD + 3][ii];
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[ii][tidx * N_NUM_PER_THREAD]);
            if (tidx == 0 && tidy == 0) 
                int t = 0;
            // fmma
            for (int j = 0; j < M_NUM_PER_THREAD; j++) {
                for (int kk = 0; kk < N_NUM_PER_THREAD; kk++) {
                    temp[j][kk] += a_reg[j] * b_reg[kk];
                    // temp[j] += a_shared[tidy][kk] * b_shared[kk][tidx * NUM_PER_THREAD + j];
                }
            }
        }
        __syncthreads();
    }
    C = C + bidy * M_NUM_PER_BLOCK * N + bidx * N_NUM_PER_BLOCK;
    for (int i = 0; i < M_NUM_PER_THREAD; i++) {
        FETCH_FLOAT4(C[N * (tidy * M_NUM_PER_THREAD + i) + tidx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
    }
}


bool check_equal(float* res, float* c, const int n, const int m) {
    bool flag = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(res[i * n + j] - c[i * n + j]) > 0.005) {
                printf("res[%d][%d] = %f, c[%d][%d] = %f\n", i, j, res[i * n + j], i, j, c[i * n + j]);
                flag = false;
                return false;
            }
        }
    }
    return flag;
}

void sgeem_cpu(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                // printf("C[%d][%d] += A[%d][%d](%f) * B[%d][%d](%f) = %f\n", i, j, i, l, A[i * k + l], l, j, B[l * n + j], C[i * n + j]);
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matrix_print(float *A, int n, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int n = 1024;
    int m = 1024;
    int k = 1024;
    float *A = new float[m * k];
    float *B = new float[n * k];
    float *C = new float[n * m];
    float *C_cpu = new float[n * m];
    
    // 初始化输入矩阵
    for (int i = 0; i < m * k; i++) {
        A[i] = drand48() * 2 + 0.1;
        // A[i] = 0.1 * (i / k + 1);
    }
    for (int i = 0; i < n * k; i++) {
        B[i] = drand48() * 2 + 0.1;
        // B[i] = 0.1 * (i / n + 1);
    }
    // 初始化输出矩阵为0
    for (int i = 0; i < n * m; i++) {
        C[i] = 0.0f;
        C_cpu[i] = 0.0f;
    }
    sgeem_cpu(A, B, C_cpu, m, n, k);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, n * k * sizeof(float));
    cudaMalloc(&d_C, n * m * sizeof(float));
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, n * m * sizeof(float), cudaMemcpyHostToDevice);
    
    const int M_PER_THREAD_BLOCK = 16;
    const int N_PER_THREAD_BLOCK = 16;
    const int M_NUM_PER_BLOCK = 64;
    const int N_NUM_PER_BLOCK = 64;
    const int K_NUM_PER_BLOCK = 64;
    const int N_NUM_PER_THREAD = 4;
    const int M_NUM_PER_THREAD = 4;
    const int K_NUM_PER_THREAD = 4;
    dim3 block(N_PER_THREAD_BLOCK, M_PER_THREAD_BLOCK);
    dim3 grid(m / M_NUM_PER_BLOCK, n / N_NUM_PER_BLOCK);
    // dim3 grid(1, 1);
    // matrix_print(A, m, k);
    // printf("\n---------\n");
    // matrix_print(B, k, n);
    // printf("\n---------\n");
    sgeem<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, M_NUM_PER_THREAD, N_NUM_PER_THREAD, K_NUM_PER_THREAD><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    // printf("\n---------\n");
    cudaDeviceSynchronize();
    // printf("\n---------\n");
    cudaMemcpy(C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("\n---------\n");
    // matrix_print(C, m, k);
    // printf("\n---------\n");
    // matrix_print(C_cpu, m, k);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (check_equal(C_cpu, C, n, m)) {
        printf("C is equal to C\n");
    } else {
        printf("C is not equal to C\n");
    }
}
