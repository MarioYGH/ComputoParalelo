#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>

#define BLUR_SIZE 1
#define WIDTH 9096
#define HEIGHT 9096
#define TILE_SIZE 16

// GPU: CON verificación
__global__
void blur_con_check(unsigned char* in, unsigned char* out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int pixVal = 0, pixels = 0;

    for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; ++dy) {
        for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                pixVal += in[ny * w + nx];
                pixels++;
            }
        }
    }
    out[y * w + x] = (unsigned char)(pixVal / pixels);
}

// GPU: Memoria compartida
__global__
void blur_shared_memory(unsigned char* in, unsigned char* out, int w, int h)
{
    __shared__ unsigned char sharedMem[TILE_SIZE + 2 * BLUR_SIZE][TILE_SIZE + 2 * BLUR_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x + BLUR_SIZE;
    int local_y = threadIdx.y + BLUR_SIZE;

    if (x < w && y < h) {
        sharedMem[local_y][local_x] = in[y * w + x];
        if (threadIdx.x < BLUR_SIZE) {
            sharedMem[local_y][local_x - BLUR_SIZE] = (x >= BLUR_SIZE) ? in[y * w + (x - BLUR_SIZE)] : 0;
            sharedMem[local_y][local_x + TILE_SIZE] = (x + TILE_SIZE < w) ? in[y * w + (x + TILE_SIZE)] : 0;
        }
        if (threadIdx.y < BLUR_SIZE) {
            sharedMem[local_y - BLUR_SIZE][local_x] = (y >= BLUR_SIZE) ? in[(y - BLUR_SIZE) * w + x] : 0;
            sharedMem[local_y + TILE_SIZE][local_x] = (y + TILE_SIZE < h) ? in[(y + TILE_SIZE) * w + x] : 0;
        }
    }
    __syncthreads();

    if (x < w && y < h) {
        int pixVal = 0, pixels = 0;
        for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; ++dy) {
            for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; ++dx) {
                pixVal += sharedMem[local_y + dy][local_x + dx];
                pixels++;
            }
        }
        out[y * w + x] = (unsigned char)(pixVal / pixels);
    }
}

void blur_cpu(const unsigned char* in, unsigned char* out, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int pixVal = 0, pixels = 0;
            for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; ++dy) {
                for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        pixVal += in[ny * w + nx];
                        pixels++;
                    }
                }
            }
            out[y * w + x] = (unsigned char)(pixVal / pixels);
        }
    }
}

void run_experiment() {
    unsigned char* h_image = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_out_cpu = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_out_gpu1 = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_out_gpu_shared = new unsigned char[WIDTH * HEIGHT];

    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        h_image[i] = rand() % 256;

    auto t0 = std::chrono::high_resolution_clock::now();
    blur_cpu(h_image, h_out_cpu, WIDTH, HEIGHT);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_cpu = t1 - t0;

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, WIDTH * HEIGHT);
    cudaMalloc(&d_out, WIDTH * HEIGHT);
    cudaMemcpy(d_in, h_image, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    auto g0 = std::chrono::high_resolution_clock::now();
    blur_con_check<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_gpu_check = g1 - g0;
    cudaMemcpy(h_out_gpu1, d_out, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    auto g2 = std::chrono::high_resolution_clock::now();
    blur_shared_memory<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    auto g3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_gpu_shared = g3 - g2;
    cudaMemcpy(h_out_gpu_shared, d_out, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Tiempo CPU:                " << time_cpu.count() << " s\n";
    std::cout << "GPU con verificación:      " << time_gpu_check.count() << " s\n";
    std::cout << "GPU con memoria compartida:" << time_gpu_shared.count() << " s\n\n";
    std::cout << "Speed-up GPU con check:    " << time_cpu.count() / time_gpu_check.count() << "x\n";
    std::cout << "Speed-up GPU compartida:   " << time_cpu.count() / time_gpu_shared.count() << "x\n";

    delete[] h_image;
    delete[] h_out_cpu;
    delete[] h_out_gpu1;
    delete[] h_out_gpu_shared;
}

int main() {
    run_experiment();
    return 0;
}
