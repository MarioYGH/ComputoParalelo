#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "fire_kernel.h"

void save_as_image(const std::vector<int>& state, int rows, int cols, const std::string& filename, bool is_png = true) {
    std::vector<unsigned char> image(rows * cols * 3);

    for (int i = 0; i < rows * cols; ++i) {
        int val = state[i];
        unsigned char r = 200, g = 200, b = 200;

        if (val == 0)      { r = 34;  g = 139; b = 34; }
        else if (val == 1) { r = 255; g = 0;   b = 0;  }
        else if (val == 2) { r = 50;  g = 50;  b = 50; }

        image[i * 3 + 0] = r;
        image[i * 3 + 1] = g;
        image[i * 3 + 2] = b;
    }

    if (is_png) {
        stbi_write_png(filename.c_str(), cols, rows, 3, image.data(), cols * 3);
    } else {
        stbi_write_jpg(filename.c_str(), cols, rows, 3, image.data(), 90);
    }
}

void step_simulation_cpu(std::vector<int>& state, int rows, int cols) {
    std::vector<int> new_state = state;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int idx = i * cols + j;
            if (state[idx] == 0) {
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;
                        int nidx = ni * cols + nj;
                        if (state[nidx] == 1) {
                            new_state[idx] = 1;
                            goto done;
                        }
                    }
                }
            }
            if (state[idx] == 1) {
                new_state[idx] = 2;
            }
            done:;
        }
    }

    state.swap(new_state);
}

int main() {
    int width, height, channels;
    unsigned char* image = stbi_load("../vegetation_map_4k.png", &width, &height, &channels, 1);
    if (!image) {
        std::cerr << "Error cargando la imagen." << std::endl;
        return 1;
    }

    std::vector<int> state(width * height);
    for (int i = 0; i < width * height; ++i) {
        state[i] = (image[i] > 127) ? 0 : -1;
    }

    int cx = height / 2, cy = width / 2;
    for (int dx = -10; dx <= 10; ++dx)
        for (int dy = -10; dy <= 10; ++dy)
            state[(cx + dx) * width + (cy + dy)] = 1;

    stbi_image_free(image);

    const int steps = 200;

    // SIMULACIÓN EN CPU
    std::vector<int> state_cpu = state;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < steps; ++t) {
        step_simulation_cpu(state_cpu, height, width);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    double time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "[CPU] Tiempo total: " << time_cpu << " ms" << std::endl;

    // SIMULACIÓN EN GPU
    std::vector<int> state_gpu = state;
    int* d_state;
    size_t state_size = width * height * sizeof(int);

    cudaMalloc(&d_state, state_size);
    cudaMemcpy(d_state, state_gpu.data(), state_size, cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < steps; ++t) {
        step_simulation_gpu(d_state, height, width);
    }
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    double time_gpu = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    std::cout << "[GPU] Tiempo total: " << time_gpu << " ms" << std::endl;

    cudaMemcpy(state_gpu.data(), d_state, state_size, cudaMemcpyDeviceToHost);
    cudaFree(d_state);

    // SPEED-UP
    double speedup = time_cpu / time_gpu;
    std::cout << ">> Speed-up (CPU / GPU): " << speedup << "x" << std::endl;

    //GUARDAR RESULTADO
    save_as_image(state_gpu, height, width, "result_gpu.png", true);
    std::cout << "Resultado guardado como result_gpu.png" << std::endl;

    return 0;
}
