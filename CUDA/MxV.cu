#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

#define WIDTH 1024
#define HEIGHT 1024

// Memoria constante para el vector x
__constant__ float d_x[WIDTH];

// Kernel CUDA para multiplicación matriz-vector usando memoria compartida
__global__ void matrixVectorMul(const float *A, float *y, int width, int height){
    __shared__ float shared_A[32][32];  
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < height) {
        float sum = 0.0f;
        
        for (int i = 0; i < width; i += 32) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * width + (i + threadIdx.x)];
            __syncthreads(); 

            for (int j = 0; j < 32 && (i + j) < width; j++) {
                sum += shared_A[threadIdx.y][j] * d_x[i + j];
            }
            __syncthreads();
        }
        y[row] = sum;
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
    const int matrixSize = WIDTH * HEIGHT * sizeof(float);
    const int vectorSize = WIDTH * sizeof(float);
    
    // Asignación y llenado de memoria en host
    float *h_A = (float*) malloc(matrixSize);
    float *h_x = (float*) malloc(vectorSize);
    float *h_y = (float*) malloc(vectorSize);

    for(int i = 0; i < WIDTH; i++){
        h_x[i] = static_cast<float>(rand()) / RAND_MAX;
        for(int j = 0; j < HEIGHT; j++){
            h_A[i * WIDTH + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Asignación de memoria en GPU
    float *d_A, *d_y;
    cudaCheck(cudaMalloc(&d_A, matrixSize), "cudaMalloc d_A");
    cudaCheck(cudaMalloc(&d_y, vectorSize), "cudaMalloc d_y");

    // Copia de datos Host -> Device
    cudaCheck(cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice), "Memcpy H->D d_A");
    cudaCheck(cudaMemcpyToSymbol(d_x, h_x, vectorSize), "Memcpy H->D d_x (memoria constante)");

    // Configuración del kernel
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Medición de tiempo en GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrixVectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_y, WIDTH, HEIGHT);
    cudaCheck(cudaDeviceSynchronize(), "Kernel launch");
    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_duration.count() << " ms" << std::endl;

    // Copia resultado Device -> Host
    cudaCheck(cudaMemcpy(h_y, d_y, vectorSize, cudaMemcpyDeviceToHost), "Memcpy D->H d_y");

    // Tiempo CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < HEIGHT; i++){
        float sum = 0.0f;
        for(int j = 0; j < WIDTH; j++){
            sum += h_A[i * WIDTH + j] * h_x[j];
        }
        h_y[i] = sum;
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // Cálculo del Speedup
    double speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "Speedup (CPU/GPU): " << speedup << "x" << std::endl;

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);

    return 0;
}
