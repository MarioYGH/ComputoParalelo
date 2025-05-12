#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <sstream> 

#define IDX(x, y, width) ((y)*(width) + (x))

// ------------------------
// Leer archivo PGM (P2)
bool loadPGM(const char* filename, float** outData, int* width, int* height) {
    std::ifstream file(filename);
    if (!file) return false;

    std::string line;
    std::getline(file, line);
    if (line != "P2") return false;

    // Leer dimensiones
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

// ------------------------
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

// ------------------------
// Utilidades de padding
int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) power *= 2;
    return power;
}

void padImage(float*& data, int& width, int& height) {
    int newWidth = nextPowerOfTwo(width);
    int newHeight = nextPowerOfTwo(height);

    if (newWidth == width && newHeight == height) return;

    float* padded = new float[newWidth * newHeight];
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            if (x < width && y < height)
                padded[IDX(x, y, newWidth)] = data[IDX(x, y, width)];
            else
                padded[IDX(x, y, newWidth)] = 0.0f;
        }
    }

    delete[] data;
    data = padded;
    width = newWidth;
    height = newHeight;

    std::cout << "Dimensiones ajustadas a: " << width << " x " << height << " (potencias de 2)\n";
}

// Cálculo de magnitudes y normalización
void calculateMagnitudeAndNormalize(float* realData, float* imagData, float* outputData, int width, int height) {
    float maxVal = 0.0f;

    // Calculamos la magnitud de la FFT y encontramos el valor máximo
    for (int i = 0; i < width * height; ++i) {
        outputData[i] = sqrt(realData[i] * realData[i] + imagData[i] * imagData[i]);
        maxVal = std::max(maxVal, outputData[i]);
    }

    // Normalizamos la imagen para que los valores estén entre 0 y 255
    for (int i = 0; i < width * height; ++i) {
        outputData[i] = (outputData[i] / maxVal) * 255.0f;
    }
}

// ------------------------
// Kernel para la FFT 1D real en una fila
__global__ void fft1DRowKernel(float* data, float* realOut, float* imagOut, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height) return;

    // FFT 1D real en la fila y (cada hilo procesa una fila)
    extern __shared__ float sharedData[];

    int idx = IDX(x, y, width);
    sharedData[threadIdx.x] = data[idx];
    __syncthreads();

    float real = 0.0f;
    float imag = 0.0f;
    int N = width; // número de puntos en la fila

    for (int k = 0; k < N; ++k) {
        float angle = -2.0f * M_PI * k / N;
        real += sharedData[threadIdx.x] * cos(angle);
        imag += sharedData[threadIdx.x] * sin(angle);
    }

    realOut[idx] = real;
    imagOut[idx] = imag;
}

// ------------------------
// Kernel para la FFT 1D real en una columna
__global__ void fft1DColKernel(float* data, float* realOut, float* imagOut, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height) return;

    // FFT 1D real en la columna x (cada hilo procesa una columna)
    extern __shared__ float sharedData[];

    int idx = IDX(x, y, width);
    sharedData[threadIdx.y] = data[idx];
    __syncthreads();

    float real = 0.0f;
    float imag = 0.0f;
    int N = height; // número de puntos en la columna

    for (int k = 0; k < N; ++k) {
        float angle = -2.0f * M_PI * k / N;
        real += sharedData[threadIdx.y] * cos(angle);
        imag += sharedData[threadIdx.y] * sin(angle);
    }

    realOut[idx] = real;
    imagOut[idx] = imag;
}

// Función para calcular la magnitud de los números complejos
void calculateMagnitude(float* realData, float* imagData, float* outputData, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        outputData[i] = sqrt(realData[i] * realData[i] + imagData[i] * imagData[i]);
    }
}

// Función para imprimir la magnitud de la FFT en consola
void printMagnitude(float* magnitude, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = IDX(x, y, width);
            std::cout << magnitude[idx] << " ";
        }
        std::cout << std::endl;
    }
}

// ------------------------
// Función principal
int main() {
    const char* input = "barbara.ascii.pgm";
    const char* output = "resultado.pgm";

    float* h_input;
    int width, height;

    if (!loadPGM(input, &h_input, &width, &height) || width <= 0 || height <= 0) {
        std::cerr << "Error al cargar imagen o dimensiones inválidas.\n";
        return 1;
    }

    std::cout << "Dimensiones originales: " << width << " x " << height << std::endl;

    // Rellenar a potencias de 2
    padImage(h_input, width, height);

    // Reservar memoria en GPU
    float* d_input;
    float* d_realOut;
    float* d_imagOut;
    cudaMalloc(&d_input, sizeof(float) * width * height);
    cudaMalloc(&d_realOut, sizeof(float) * width * height);
    cudaMalloc(&d_imagOut, sizeof(float) * width * height);

    cudaMemcpy(d_input, h_input, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    // Aplicar FFT 1D en las filas
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    fft1DRowKernel<<<grid, block, sizeof(float) * block.x>>>(d_input, d_realOut, d_imagOut, width, height);

    // Aplicar FFT 1D en las columnas
    fft1DColKernel<<<grid, block, sizeof(float) * block.y>>>(d_input, d_realOut, d_imagOut, width, height);

    // Sincronización y copia de resultados a CPU
    cudaDeviceSynchronize();
    float* h_outputReal = new float[width * height];
    float* h_outputImag = new float[width * height];
    cudaMemcpy(h_outputReal, d_realOut, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputImag, d_imagOut, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // Calcular la magnitud de la FFT
    // Calcular magnitud y normalizar para guardarla como imagen
    float* h_magnitude = new float[width * height];
    calculateMagnitudeAndNormalize(h_outputReal, h_outputImag, h_magnitude, width, height);


    // Imprimir la magnitud en consola
    std::cout << "Magnitud de la FFT:" << std::endl;
    printMagnitude(h_magnitude, width, height);

    std::cout << "Proceso completado. Imagen guardada como " << output << "\n";

    // Guardar imagen resultante
    savePGM(output, h_magnitude, width, height);

    delete[] h_input;
    delete[] h_outputReal;
    delete[] h_outputImag;
    delete[] h_magnitude;
    cudaFree(d_input);
    cudaFree(d_realOut);
    cudaFree(d_imagOut);

    return 0;
}
