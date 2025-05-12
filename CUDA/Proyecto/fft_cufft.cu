#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <sstream>
#include <algorithm>

#define IDX(x, y, width) ((y)*(width) + (x))

// ------------------------
// Leer archivo PGM (P2)
bool loadPGM(const char* filename, float** outData, int* width, int* height) {
    std::ifstream file(filename);
    if (!file) return false;

    std::string line;
    std::getline(file, line);
    if (line != "P2") return false;

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
int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) power *= 2;
    return power;
}

// ------------------------

void printElapsedTime(const char* label, float milliseconds) {
    std::cout << label << ": " << milliseconds << " ms\n";
}

void padImage(float*& data, int& width, int& height) {
    int newWidth = nextPowerOfTwo(width);
    int newHeight = nextPowerOfTwo(height);

    if (newWidth == width && newHeight == height) return;

    float* padded = new float[newWidth * newHeight]();

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            padded[IDX(x, y, newWidth)] = data[IDX(x, y, width)];

    delete[] data;
    data = padded;
    width = newWidth;
    height = newHeight;

    std::cout << "Dimensiones ajustadas a: " << width << " x " << height << " (potencias de 2)\n";
}

void applyFFTShift(float* data, int width, int height) {
    int halfW = width / 2;
    int halfH = height / 2;

    // Swap top-left ↔ bottom-right
    for (int y = 0; y < halfH; ++y) {
        for (int x = 0; x < halfW; ++x) {
            int idx1 = y * width + x;
            int idx2 = (y + halfH) * width + (x + halfW);
            std::swap(data[idx1], data[idx2]);
        }
    }

    // Swap top-right ↔ bottom-left
    for (int y = 0; y < halfH; ++y) {
        for (int x = halfW; x < width; ++x) {
            int idx1 = y * width + x;
            int idx2 = (y + halfH) * width + (x - halfW);
            std::swap(data[idx1], data[idx2]);
        }
    }
}

int main() {
    const char* input = "barbara.ascii.pgm";
    const char* outputFFT = "resultadoFinalCuda.pgm";
    const char* outputReconstructed = "reconstruidaFinalCuda.pgm";

    float* h_input;
    int width, height;

    cudaEvent_t startFFT, endFFT;
    cudaEventCreate(&startFFT);
    cudaEventCreate(&endFFT);

    if (!loadPGM(input, &h_input, &width, &height)) {
        std::cerr << "Error al cargar imagen.\n";
        return 1;
    }

    std::cout << "Dimensiones originales: " << width << " x " << height << std::endl;
    padImage(h_input, width, height);

    int size = width * height;

    cufftReal* d_dataReal;
    cufftComplex* d_dataComplex;

    cudaMalloc(&d_dataReal, sizeof(cufftReal) * size);
    cudaMalloc(&d_dataComplex, sizeof(cufftComplex) * (width * (height / 2 + 1)));

    cudaMemcpy(d_dataReal, h_input, sizeof(float) * size, cudaMemcpyHostToDevice);

    cufftHandle planR2C;
    cufftPlan2d(&planR2C, height, width, CUFFT_R2C);

    cudaEventRecord(startFFT);
    cufftExecR2C(planR2C, d_dataReal, d_dataComplex);
    cudaEventRecord(endFFT);
    cudaEventSynchronize(endFFT);
    float fftTime = 0;
    cudaEventElapsedTime(&fftTime, startFFT, endFFT);
    std::cout << "Tiempo de ejecución FFT: " << fftTime << " ms\n";

    cufftComplex* h_complex = new cufftComplex[width * (height / 2 + 1)];
    cudaMemcpy(h_complex, d_dataComplex, sizeof(cufftComplex) * width * (height / 2 + 1), cudaMemcpyDeviceToHost);

    // ------------------------------------------
    // FIXED: Magnitude computation with symmetry
    float* h_magnitude = new float[width * height]();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int fftX = x < (width / 2 + 1) ? x : width - x;

            if (fftX >= 0 && fftX < (width / 2 + 1)) {
                int idx = y * (width / 2 + 1) + fftX;
                float real = h_complex[idx].x;
                float imag = h_complex[idx].y;
                float mag = sqrtf(real * real + imag * imag);

                h_magnitude[IDX(x, y, width)] = logf(1.0f + mag);
            }
        }
    }

    // Normalize for PGM output
    float maxVal = *std::max_element(h_magnitude, h_magnitude + size);
    for (int i = 0; i < size; ++i)
        h_magnitude[i] = 255.0f * h_magnitude[i] / maxVal;

    applyFFTShift(h_magnitude, width, height);
    savePGM(outputFFT, h_magnitude, width, height);
    std::cout << "Imagen de la magnitud FFT guardada como " << outputFFT << "\n";

    // ------------------------------------------

    cudaEvent_t startIFFT, endIFFT;
    cudaEventCreate(&startIFFT);
    cudaEventCreate(&endIFFT);
    // Inverse FFT (Reconstruction)
    cufftHandle planC2R;
    cufftPlan2d(&planC2R, height, width, CUFFT_C2R);

    cudaEventRecord(startIFFT);
    cufftExecC2R(planC2R, d_dataComplex, d_dataReal);
    cudaEventRecord(endIFFT);
    cudaEventSynchronize(endIFFT);

    float ifftTime = 0;
    cudaEventElapsedTime(&ifftTime, startIFFT, endIFFT);
    printElapsedTime("Tiempo de ejecución IFFT (reconstrucción)", ifftTime);

    float* h_reconstructed = new float[size];
    cudaMemcpy(h_reconstructed, d_dataReal, sizeof(float) * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i)
        h_reconstructed[i] /= (width * height);

    savePGM(outputReconstructed, h_reconstructed, width, height);
    std::cout << "Imagen reconstruida guardada como " << outputReconstructed << "\n";

    // Cleanup
    delete[] h_input;
    delete[] h_complex;
    delete[] h_magnitude;
    delete[] h_reconstructed;
    cudaFree(d_dataReal);
    cudaFree(d_dataComplex);
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);

    return 0;
}
