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

void calculateMagnitudeAndNormalize(float* real, float* imag, float* output, int width, int height) {
    float maxVal = 0.0f;

    for (int i = 0; i < width * height; ++i) {
        float mag = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
        output[i] = logf(1.0f + mag);
        maxVal = fmaxf(maxVal, output[i]);
    }

    for (int i = 0; i < width * height; ++i)
        output[i] = 255.0f * output[i] / maxVal;
}

void applyFFTShift(float* data, int width, int height) {
    int halfW = width / 2;
    int halfH = height / 2;

    for (int y = 0; y < halfH; ++y) {
        for (int x = 0; x < halfW; ++x) {
            // Índices de los 4 cuadrantes a intercambiar
            int idx1 = IDX(x, y, width);                       // Q1: arriba-izquierda
            int idx2 = IDX(x + halfW, y + halfH, width);       // Q4: abajo-derecha
            int idx3 = IDX(x + halfW, y, width);               // Q2: arriba-derecha
            int idx4 = IDX(x, y + halfH, width);               // Q3: abajo-izquierda

            // Intercambio de Q1 con Q4
            std::swap(data[idx1], data[idx2]);

            // Intercambio de Q2 con Q3
            std::swap(data[idx3], data[idx4]);
        }
    }
}

// ------------------------
// Kernel para la FFT 1D real en una fila
__global__ void fft1DRowKernel(float* data, float* realOut, float* imagOut, int width, int height) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (k >= width || y >= height) return;

    float real = 0.0f;
    float imag = 0.0f;

    for (int n = 0; n < width; ++n) {
        float value = data[IDX(n, y, width)];
        float angle = -2.0f * M_PI * k * n / width;
        real += value * cosf(angle);
        imag += value * sinf(angle);
    }

    int idx = IDX(k, y, width);
    realOut[idx] = real;
    imagOut[idx] = imag;
}

// ------------------------
// Kernel para la FFT 1D real en una columna
__global__ void fft1DColKernel(float* data, float* realOut, float* imagOut, int width, int height) {
    int k = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (k >= height || x >= width) return;

    float real = 0.0f;
    float imag = 0.0f;

    for (int n = 0; n < height; ++n) {
        float value = data[IDX(x, n, width)];
        float angle = -2.0f * M_PI * k * n / height;
        real += value * cosf(angle);
        imag += value * sinf(angle);
    }

    int idx = IDX(x, k, width);
    realOut[idx] = real;
    imagOut[idx] = imag;
}

// ------------------------
__global__ void ifft1DRowKernel(float* realIn, float* imagIn, float* realOut, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int n = 0; n < width; ++n) {
        float real = realIn[IDX(n, y, width)];
        float imag = imagIn[IDX(n, y, width)];
        float angle = 2.0f * M_PI * x * n / width;

        sum += real * cosf(angle) - imag * sinf(angle);  // Parte real de la IFFT
    }

    realOut[IDX(x, y, width)] = sum;  // Sin dividir aún
}

__global__ void ifft1DColKernel(float* realIn, float* imagIn, float* realOut, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int n = 0; n < height; ++n) {
        float real = realIn[IDX(x, n, width)];
        float imag = imagIn[IDX(x, n, width)];
        float angle = 2.0f * M_PI * y * n / height;

        sum += real * cosf(angle) - imag * sinf(angle);  // Parte real de la IFFT
    }

    realOut[IDX(x, y, width)] = sum;  // Sin dividir aún
}


void cropImage(float* src, float* dst, int origWidth, int origHeight, int paddedWidth) {
    for (int y = 0; y < origHeight; ++y) {
        for (int x = 0; x < origWidth; ++x) {
            dst[IDX(x, y, origWidth)] = src[IDX(x, y, paddedWidth)];
        }
    }
}
// ------------------------
// Función principal
int main() {
    const char* input = "lena.pgm";
    const char* outputFFT = "resultado.pgm";
    const char* outputReconstructed = "reconstruida.pgm";

    float* h_input;
    int width, height;

    if (!loadPGM(input, &h_input, &width, &height)) {
        std::cerr << "Error al cargar imagen.\n";
        return 1;
    }

    int originalWidth = width, originalHeight = height;
    std::cout << "Dimensiones originales: " << width << " x " << height << std::endl;

    padImage(h_input, width, height);

    float *d_input, *d_realOut, *d_imagOut;
    cudaMalloc(&d_input, sizeof(float) * width * height);
    cudaMalloc(&d_realOut, sizeof(float) * width * height);
    cudaMalloc(&d_imagOut, sizeof(float) * width * height);

    cudaMemcpy(d_input, h_input, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // Eventos para medir tiempo
    cudaEvent_t startFFT, endFFT, startIFFT, endIFFT;
    cudaEventCreate(&startFFT);
    cudaEventCreate(&endFFT);
    cudaEventCreate(&startIFFT);
    cudaEventCreate(&endIFFT);

    // ----------- FFT directa (filas y columnas) -----------
    cudaEventRecord(startFFT);

    fft1DRowKernel<<<grid, block>>>(d_input, d_realOut, d_imagOut, width, height);
    cudaDeviceSynchronize();

    float* d_tempReal;
    cudaMalloc(&d_tempReal, sizeof(float) * width * height);

    fft1DColKernel<<<grid, block>>>(d_realOut, d_tempReal, d_imagOut, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(d_realOut, d_tempReal, sizeof(float) * width * height, cudaMemcpyDeviceToDevice);
    cudaFree(d_tempReal);

    cudaEventRecord(endFFT);
    cudaEventSynchronize(endFFT);

    float elapsedFFT = 0.0f;
    cudaEventElapsedTime(&elapsedFFT, startFFT, endFFT);
    std::cout << "Tiempo FFT directa: " << elapsedFFT << " ms\n";

    // ----------- Magnitud y guardado -----------
    float* h_real = new float[width * height];
    float* h_imag = new float[width * height];
    float* h_magnitude = new float[width * height];

    cudaMemcpy(h_real, d_realOut, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_imag, d_imagOut, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    calculateMagnitudeAndNormalize(h_real, h_imag, h_magnitude, width, height);
    applyFFTShift(h_magnitude, width, height);
    savePGM(outputFFT, h_magnitude, width, height);
    std::cout << "Imagen de la magnitud FFT guardada como " << outputFFT << "\n";

    // ----------- IFFT (col + filas) -----------
    float* d_ifftTemp;
    cudaMalloc(&d_ifftTemp, sizeof(float) * width * height);

    cudaEventRecord(startIFFT);

    // Primero IFFT en columnas
    ifft1DColKernel<<<grid, block>>>(d_realOut, d_imagOut, d_ifftTemp, width, height);
    cudaDeviceSynchronize();

    // Luego IFFT en filas
    cudaMemset(d_imagOut, 0, sizeof(float) * width * height); // Imag no se usa aquí, puede ser 0
    ifft1DRowKernel<<<grid, block>>>(d_ifftTemp, d_imagOut, d_realOut, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(endIFFT);
    cudaEventSynchronize(endIFFT);
    cudaFree(d_ifftTemp);

    float elapsedIFFT = 0.0f;
    cudaEventElapsedTime(&elapsedIFFT, startIFFT, endIFFT);
    std::cout << "Tiempo IFFT (reconstrucción): " << elapsedIFFT << " ms\n";

    // ----------- Imagen reconstruida -----------
    float* h_reconstructed = new float[width * height];
    cudaMemcpy(h_reconstructed, d_realOut, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; ++i)
        h_reconstructed[i] /= (width * height);

    float* h_final = new float[originalWidth * originalHeight];
    cropImage(h_reconstructed, h_final, originalWidth, originalHeight, width);

    savePGM(outputReconstructed, h_final, originalWidth, originalHeight);
    std::cout << "Imagen reconstruida guardada como " << outputReconstructed << "\n";

    // ----------- Liberación de recursos -----------
    delete[] h_input;
    delete[] h_real;
    delete[] h_imag;
    delete[] h_magnitude;
    delete[] h_reconstructed;
    delete[] h_final;

    cudaFree(d_input);
    cudaFree(d_realOut);
    cudaFree(d_imagOut);

    cudaEventDestroy(startFFT);
    cudaEventDestroy(endFFT);
    cudaEventDestroy(startIFFT);
    cudaEventDestroy(endIFFT);

    return 0;
}

