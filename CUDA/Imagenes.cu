// -----------------------------------
// Filtros CUDA sin memoria compartida (ejercicio para clase)
// -----------------------------------
// Cargar imagen, aplicar filtros, guardar salida

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// -----------------------------------
// Filtro Gaussiano sin memoria compartida
// -----------------------------------
__global__ void filtroGaussianoGlobal(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        float kernel[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};
        for (int c = 0; c < channels; c++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int px = x + dx;
                    int py = y + dy;
                    int i = (py * width + px) * channels + c;
                    sum += input[i] * kernel[dy+1][dx+1];
                }
            }
            output[(idx * channels) + c] = sum / 16;
        }
    }
}

int main() {
    int width, height, channels;
    unsigned char* h_input = stbi_load("guepard.jpeg", &width, &height, &channels, 0);
    if (!h_input) {
        std::cerr << "No se pudo cargar la imagen.\n";
        return -1;
    }

    size_t img_size = width * height * channels;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    filtroGaussianoGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    unsigned char* h_output = new unsigned char[img_size];
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", width, height, channels, h_output, width * channels);

    std::cout << "Imagen procesada guardada como output.png\n";

    stbi_image_free(h_input);
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

// EJERCICIO PARA CLASE:
// - Reescribir el filtro anterior usando memoria compartida (__shared__)
// - Comparar tiempos de ejecución entre ambas versiones
// - Comentar dónde se reduce el número de accesos a memoria global
