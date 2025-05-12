#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <sstream> 

#define IDX(x, y, width) ((y)*(width) + (x))

// Leer archivo PGM (formato P2)
bool loadPGM(const char* filename, float** outData, int* width, int* height) {
    std::ifstream file(filename);
    if (!file) return false;

    std::string line;
    std::getline(file, line);
    if (line != "P2") return false;

    // Leer encabezado ignorando comentarios
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream dims(line);
        if (!(dims >> *width >> *height)) continue;
        break;
    }

    int maxVal = 255;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream mv(line);
        if (!(mv >> maxVal)) continue;
        break;
    }

    if (*width <= 0 || *height <= 0) return false;

    *outData = new float[(*width) * (*height)];
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        int val;
        while (ss >> val) {
            if (count < (*width) * (*height))
                (*outData)[count++] = (float)val;
        }
    }

    return count == (*width) * (*height);
}


// Guardar archivo PGM
bool savePGM(const char* filename, float* data, int width, int height) {
    std::ofstream file(filename);
    if (!file) return false;

    file << "P2\n";
    file << width << " " << height << "\n";
    file << "255\n";

    for (int i = 0; i < width * height; ++i) {
        int val = std::min(255, std::max(0, (int)data[i]));
        file << val << ((i + 1) % width == 0 ? "\n" : " ");
    }

    return true;
}

// Kernel base: copia datos (puedes reemplazar por FFT real luego)
__global__ void simpleFFTKernel(float* d_in, float* d_out, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= width || y >= height) return;

    int idx = IDX(x, y, width);
    d_out[idx] = d_in[idx]; // solo copia, puedes poner fft real aquí
}

int main() {
    const char* input = "barbara.ascii.pgm";
    const char* output = "resultado.pgm";

    float* h_input;
    int width, height;

    if (!loadPGM(input, &h_input, &width, &height) || width <= 0 || height <= 0) {
        std::cerr << "Error al cargar imagen o dimensiones inválidas.\n";
        return 1;
    }

    std::cout << "Dimensiones cargadas: " << width << " x " << height << std::endl;


    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, sizeof(float) * width * height);
    cudaMalloc(&d_output, sizeof(float) * width * height);

    cudaMemcpy(d_input, h_input, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    simpleFFTKernel<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    float* h_output = new float[width * height];
    cudaMemcpy(h_output, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    std::cout << "Primer valor del resultado: " << h_output[0] << std::endl;

    savePGM(output, h_output, width, height);

    std::cout << "Proceso completado. Imagen guardada como " << output << "\n";

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
