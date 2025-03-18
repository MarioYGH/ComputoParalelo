#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

#define WIDTH 1024
#define HEIGHT 1024

// Kernel CUDA para sumar matrices
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(row < height && col < width){
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

// Función para verificar errores CUDA
void cudaCheck(cudaError_t err, const char* message){
    if(err != cudaSuccess){
        std::cerr << "Error (" << message << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(){
    const int size = WIDTH * HEIGHT * sizeof(float);

    // Asignación y llenado de memoria en host
    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size);
    float *h_C = (float*) malloc(size);

    for(int i = 0; i < WIDTH * HEIGHT; i++){
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

    // Diferentes configuraciones de hilos por bloque
    int configs[][2] = {{8, 8}, {16, 16}, {32, 64}, {64, 64}}; // Tamaño de bloques (X, Y)
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    double min_gpu_time = std::numeric_limits<double>::max();

    for (int i = 0; i < num_configs; i++) {
        int tx = configs[i][0];
        int ty = configs[i][1];

        dim3 threadsPerBlock(tx, ty);
        dim3 blocksPerGrid((WIDTH + tx - 1) / tx, (HEIGHT + ty - 1) / ty);

        auto start_gpu = std::chrono::high_resolution_clock::now();
        matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH, HEIGHT);
        cudaCheck(cudaDeviceSynchronize(), "Kernel launch");
        auto end_gpu = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
        min_gpu_time = std::min(min_gpu_time, gpu_duration.count());

        std::cout << "Config (" << tx << "x" << ty << "): GPU Time = " << gpu_duration.count() << " ms" << std::endl;
    }

    // Copia resultado Device -> Host
    cudaCheck(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Memcpy D->H d_C");

    // Tiempo CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < WIDTH * HEIGHT; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // Cálculo del Speedup y Eficiencia
    double speedup = cpu_duration.count() / min_gpu_time;
    std::cout << "Speedup (CPU/GPU): " << speedup << "x" << std::endl;

    int total_cuda_cores = 4096; 
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
