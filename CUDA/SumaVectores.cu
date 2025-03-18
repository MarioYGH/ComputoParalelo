#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

// Kernel CUDA para sumar vectores
__global__ void vectorAdd(const float *A, const float *B, float *C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
        C[i] = A[i] + B[i];
}

// Función para verificar errores CUDA
void cudaCheck(cudaError_t err, const char* message){
    if(err != cudaSuccess){
        std::cerr << "Error (" << message << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    const int N = 1 << 24; // 16,777,216 elementos (~67 MB por vector)
    size_t size = N * sizeof(float);

    // Asignación y llenado de memoria en host
    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size);
    float *h_C = (float*) malloc(size);

    for(int i = 0; i < N; i++){
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Asignación memoria en GPU
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, size), "cudaMalloc d_A");
    cudaCheck(cudaMalloc(&d_B, size), "cudaMalloc d_B");
    cudaCheck(cudaMalloc(&d_C, size), "cudaMalloc d_C");

    // Copia de datos Host -> Device
    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy H->D d_A");
    cudaCheck(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy H->D d_B");

    // Configuración bloques y threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Tiempo GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheck(cudaDeviceSynchronize(), "Kernel launch");
    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_duration.count() << " ms" << std::endl;

    // Copia resultado Device -> Host
    cudaCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Memcpy D->H d_C");

    // Tiempo CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // Cálculo del Speedup y Eficiencia
    double speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "Speedup (CPU/GPU): " << speedup << "x" << std::endl;

    int total_cuda_cores = 2048; // Cambiar según GPU real
    double efficiency = speedup / total_cuda_cores;
    std::cout << "Eficiencia: " << efficiency * 100 << "%" << std::endl;

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
