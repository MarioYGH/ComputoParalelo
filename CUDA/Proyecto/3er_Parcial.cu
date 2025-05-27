#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <sstream>
#include <complex> // For std::complex, although not directly used with floats here
#include <algorithm> // For std::min/max, for calculateMagnitude
#include <vector> // For std::vector in spectral analysis functions
#include <cufft.h> // For temporary cuFFT usage for spectral analysis

#define IDX(x, y, width) ((y) * (width) + (x))

// The FFT_LENGTH should be the actual dimension of the row/column to transform.
// This means your padded image dimensions should be powers of 2.
// For example, if you want to transform a 256x256 image, FFT_LENGTH will be 256.
// Shared memory limit is typically 48KB or 96KB per block, so FFT_LENGTH * 2 * sizeof(float)
// must be within limits. For 256 floats (real+imag), it's 256*8 = 2KB, which is fine.
// The blockDim.x should also be FFT_LENGTH for this kernel. Max blockDim.x is 1024.
#define MAX_FFT_LENGTH 512 // Maximum FFT size a single block can handle with shared memory

// Define LOG2_MAX_FFT_LENGTH as a macro for device code
#define LOG2_MAX_FFT_LENGTH static_cast<int>(log2f(MAX_FFT_LENGTH))

// Constantes para factores de twiddle (cos y sin)
// Size needs to be MAX_FFT_LENGTH / 2
__constant__ float c_cos[MAX_FFT_LENGTH / 2];
__constant__ float c_sin[MAX_FFT_LENGTH / 2];

// Function to initialize twiddle factors for forward FFT
void initTwiddleFactors(int n) {
    if (n > MAX_FFT_LENGTH) {
        std::cerr << "Error: FFT size " << n << " exceeds MAX_FFT_LENGTH " << MAX_FFT_LENGTH << std::endl;
        exit(1);
    }
    float h_cos[MAX_FFT_LENGTH / 2];
    float h_sin[MAX_FFT_LENGTH / 2];
    for (int k = 0; k < n / 2; ++k) {
        float angle = -2.0f * M_PI * k / n; // For forward FFT
        h_cos[k] = cosf(angle);
        h_sin[k] = sinf(angle);
    }
    cudaMemcpyToSymbol(c_cos, h_cos, sizeof(float) * (n / 2));
    cudaMemcpyToSymbol(c_sin, h_sin, sizeof(float) * (n / 2));
}

// Function to initialize twiddle factors for inverse FFT
void initInverseTwiddleFactors(int n) {
    if (n > MAX_FFT_LENGTH) {
        std::cerr << "Error: IFFT size " << n << " exceeds MAX_FFT_LENGTH " << MAX_FFT_LENGTH << std::endl;
        exit(1);
    }
    float h_cos[MAX_FFT_LENGTH / 2];
    float h_sin[MAX_FFT_LENGTH / 2];
    for (int k = 0; k < n / 2; ++k) {
        float angle = 2.0f * M_PI * k / n; // For inverse FFT
        h_cos[k] = cosf(angle);
        h_sin[k] = sinf(angle);
    }
    cudaMemcpyToSymbol(c_cos, h_cos, sizeof(float) * (n / 2));
    cudaMemcpyToSymbol(c_sin, h_sin, sizeof(float) * (n / 2));
}

// Kernel for 1D FFT on rows (processes one full row per block)
__global__ void fft1DRowKernel(float* d_input_real, float* d_input_imag, float* d_output_real, float* d_output_imag, int width, int height) {
    // Each block processes one row
    int row = blockIdx.x; // Block index corresponds to row index
    if (row >= height) return;

    // Shared memory for the entire row being processed by this block
    // blockDim.x is the length of the FFT (e.g., width)
    __shared__ float s_real[MAX_FFT_LENGTH];
    __shared__ float s_imag[MAX_FFT_LENGTH];

    int tid_in_block = threadIdx.x; // Thread ID within the block

    // Load data from global memory to shared memory
    // Each thread loads one element
    if (tid_in_block < width) { // Ensure thread ID is within row bounds
        s_real[tid_in_block] = d_input_real[IDX(tid_in_block, row, width)];
        s_imag[tid_in_block] = d_input_imag[IDX(tid_in_block, row, width)];
    }
    __syncthreads(); // Wait for all threads to load data

    // Bit-reversal permutation
    int reverse_tid = 0;
    for (int i = 0; i < LOG2_MAX_FFT_LENGTH; ++i) { // Use LOG2_MAX_FFT_LENGTH
        if ((tid_in_block >> i) & 1) {
            reverse_tid |= (1 << (LOG2_MAX_FFT_LENGTH - 1 - i));
        }
    }

    if (tid_in_block < reverse_tid) { // Swap only if it's the smaller index to avoid double-swaps
        float temp_real = s_real[tid_in_block];
        float temp_imag = s_imag[tid_in_block];
        s_real[tid_in_block] = s_real[reverse_tid];
        s_imag[tid_in_block] = s_imag[reverse_tid];
        s_real[reverse_tid] = temp_real;
        s_imag[reverse_tid] = temp_imag;
    }
    __syncthreads(); // Wait for bit-reversal to complete

    // Butterfly stages (Cooley-Tukey algorithm)
    for (int s = 1; s <= LOG2_MAX_FFT_LENGTH; ++s) { // Use LOG2_MAX_FFT_LENGTH
        int m = 1 << s;       // Current FFT group size (2, 4, 8, ...)
        int m_half = m / 2;   // Half of current group size

        if (tid_in_block < width) { // Ensure thread is within actual data range
            int k = tid_in_block % m_half; // Index within a half-group
            int j = tid_in_block / m_half; // Group index

            int even_idx = j * m + k;
            int odd_idx = even_idx + m_half;

            // Only threads involved in a butterfly operation proceed
            if (odd_idx < width) { // Ensure indices are within bounds
                // Twiddle factor index: k * (N / M_current_group_size)
                // N is width, M_current_group_size is m
                float twiddle_real = c_cos[k * (width / m)];
                float twiddle_imag = c_sin[k * (width / m)];

                float t_real = s_real[odd_idx] * twiddle_real - s_imag[odd_idx] * twiddle_imag;
                float t_imag = s_real[odd_idx] * twiddle_imag + s_imag[odd_idx] * twiddle_real;

                float u_real = s_real[even_idx];
                float u_imag = s_imag[even_idx];

                s_real[even_idx] = u_real + t_real;
                s_imag[even_idx] = u_imag + t_imag;

                s_real[odd_idx] = u_real - t_real;
                s_imag[odd_idx] = u_imag - t_imag;
            }
        }
        __syncthreads(); // Synchronize after each stage
    }

    // Write results back to global memory
    if (tid_in_block < width) {
        d_output_real[IDX(tid_in_block, row, width)] = s_real[tid_in_block];
        d_output_imag[IDX(tid_in_block, row, width)] = s_imag[tid_in_block];
    }
}

// Kernel for 1D FFT on columns (processes one full column per block)
// Requires a transposed matrix for efficient access
__global__ void fft1DColKernel(float* d_input_real, float* d_input_imag, float* d_output_real, float* d_output_imag, int width, int height) {
    // Each block processes one column (from the transposed matrix, so it's a row)
    int col = blockIdx.x; // Block index corresponds to column index in the original matrix
    if (col >= width) return;

    // Shared memory for the entire column being processed by this block
    // blockDim.x is the length of the FFT (e.g., height)
    __shared__ float s_real[MAX_FFT_LENGTH];
    __shared__ float s_imag[MAX_FFT_LENGTH];

    int tid_in_block = threadIdx.x; // Thread ID within the block

    // Load data from global memory to shared memory
    // Here, width refers to the 'width' of the TRANSPOSED matrix, which is the original height
    if (tid_in_block < height) { // Ensure thread ID is within column bounds (original height)
        s_real[tid_in_block] = d_input_real[IDX(tid_in_block, col, height)]; // height is the 'width' of transposed
        s_imag[tid_in_block] = d_input_imag[IDX(tid_in_block, col, height)];
    }
    __syncthreads(); // Wait for all threads to load data

    // Bit-reversal permutation
    int reverse_tid = 0;
    for (int i = 0; i < LOG2_MAX_FFT_LENGTH; ++i) {
        if ((tid_in_block >> i) & 1) {
            reverse_tid |= (1 << (LOG2_MAX_FFT_LENGTH - 1 - i));
        }
    }
    if (tid_in_block < reverse_tid) {
        float temp_real = s_real[tid_in_block];
        float temp_imag = s_imag[tid_in_block];
        s_real[tid_in_block] = s_real[reverse_tid];
        s_imag[tid_in_block] = s_imag[reverse_tid];
        s_real[reverse_tid] = temp_real;
        s_imag[reverse_tid] = temp_imag;
    }
    __syncthreads();

    // Butterfly stages
    for (int s = 1; s <= LOG2_MAX_FFT_LENGTH; ++s) {
        int m = 1 << s;
        int m_half = m / 2;

        if (tid_in_block < height) { // Ensure thread is within actual data range (original height)
            int k = tid_in_block % m_half;
            int j = tid_in_block / m_half;

            int even_idx = j * m + k;
            int odd_idx = even_idx + m_half;

            if (odd_idx < height) {
                // Twiddle factor index: k * (N / M_current_group_size)
                // N is height, M_current_group_size is m
                float twiddle_real = c_cos[k * (height / m)];
                float twiddle_imag = c_sin[k * (height / m)];

                float t_real = s_real[odd_idx] * twiddle_real - s_imag[odd_idx] * twiddle_imag;
                float t_imag = s_real[odd_idx] * twiddle_imag + s_imag[odd_idx] * twiddle_real;

                float u_real = s_real[even_idx];
                float u_imag = s_imag[even_idx];

                s_real[even_idx] = u_real + t_real;
                s_imag[even_idx] = u_imag + t_imag;

                s_real[odd_idx] = u_real - t_real;
                s_imag[odd_idx] = u_imag - t_imag;
            }
        }
        __syncthreads();
    }

    // Write results back to global memory (to the transposed matrix)
    // height is the 'width' of the transposed matrix, width is the 'height' of transposed
    if (tid_in_block < height) {
        d_output_real[IDX(tid_in_block, col, height)] = s_real[tid_in_block];
        d_output_imag[IDX(tid_in_block, col, height)] = s_imag[tid_in_block];
    }
}

// Transposición de una matriz
__global__ void transposeKernel(float* in_real, float* in_imag, float* out_real, float* out_imag, int in_width, int in_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in_width && y < in_height) {
        out_real[IDX(y, x, in_height)] = in_real[IDX(x, y, in_width)];
        out_imag[IDX(y, x, in_height)] = in_imag[IDX(x, y, in_width)];
    }
}

// Utility: next power of 2
int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) power *= 2;
    return power;
}

// Padding image to power of 2 (and ensuring it's at least MAX_FFT_LENGTH)
void padImage(float*& data, int& width, int& height) {
    int newWidth = nextPowerOfTwo(width);
    int newHeight = nextPowerOfTwo(height);

    if (newWidth == width && newHeight == height) return;

    float* padded = new float[newWidth * newHeight](); // Initialize with zeros

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            padded[IDX(x, y, newWidth)] = data[IDX(x, y, width)];

    delete[] data;
    data = padded;
    width = newWidth;
    height = newHeight;

    std::cout << "Padded to: " << width << " x " << height << std::endl;
}

// Load PGM (P2)
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

    int maxVal;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream mv(line);
        if (!(mv >> maxVal)) continue;
        break;
    }

    *outData = new float[(*width) * (*height)];
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int val;
        while (ss >> val) {
            (*outData)[count++] = (float)val;
        }
    }

    return count == (*width) * (*height);
}

// Save PGM
bool savePGM(const char* filename, float* data, int width, int height) {
    std::ofstream file(filename);
    if (!file) return false;

    file << "P2\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        int val = static_cast<int>(std::min(255.0f, std::max(0.0f, data[i])));
        file << val << ((i + 1) % width == 0 ? "\n" : " ");
    }
    return true;
}

// Calculate magnitude spectrum
// Added FFT shift for better visualization
void calculateMagnitude(float* real, float* imag, float* output, int width, int height) {
    float maxVal = 0;
    // Apply FFT shift for visualization (DC component to center)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int original_idx = IDX(x, y, width);
            // Calculate shifted coordinates
            int shifted_x = (x + width / 2) % width;
            int shifted_y = (y + height / 2) % height;
            int shifted_idx = IDX(shifted_x, shifted_y, width);

            float magnitude = logf(1.0f + sqrtf(real[original_idx] * real[original_idx] + imag[original_idx] * imag[original_idx]));
            output[shifted_idx] = magnitude; // Store at shifted position
            maxVal = fmaxf(maxVal, magnitude);
        }
    }

    // Normalize to 0-255
    for (int i = 0; i < width * height; ++i) {
        output[i] = 255.0f * output[i] / maxVal;
    }
}

// Utility function to print elapsed time
void printElapsedTime(const char* label, float milliseconds) {
    std::cout << label << ": " << milliseconds << " ms\n";
}

// Function to calculate mean and standard deviation of an image
void calculateImageStats(float* data, int width, int height, float* outMean, float* outStdDev) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int size = width * height;

    for (int i = 0; i < size; ++i) {
        sum += data[i];
        sum_sq += (double)data[i] * data[i];
    }

    *outMean = (float)(sum / size);
    *outStdDev = sqrtf((float)(sum_sq / size - (*outMean) * (*outMean)));
}

// --- New Spectral Analysis Functions (from your cuFFT example) ---

// Helper to compute magnitude from cuFFT complex output
std::vector<float> computeMagnitudeFromCufft(cufftComplex* d_freq, int width, int height) {
    int size = width * height;
    std::vector<cufftComplex> h_freq(size); // Use std::vector for host data
    std::vector<float> magnitudes(size);

    cudaMemcpy(h_freq.data(), d_freq, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        float real = h_freq[i].x;
        float imag = h_freq[i].y;
        magnitudes[i] = sqrtf(real * real + imag * imag);
    }
    return magnitudes;
}

// This function performs the FFT shift on a 1D vector representing a 2D image
// so that the DC component is moved to the center.
std::vector<float> performFFTShift(const std::vector<float>& input_magnitude, int width, int height) {
    std::vector<float> shifted_magnitude(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int original_idx = IDX(x, y, width);
            int shifted_x = (x + width / 2) % width;
            int shifted_y = (y + height / 2) % height;
            int shifted_idx = IDX(shifted_x, shifted_y, width);
            shifted_magnitude[shifted_idx] = input_magnitude[original_idx];
        }
    }
    return shifted_magnitude;
}

float spectralCentroid(const std::vector<float>& mag, int width, int height) {
    float sum_mag = 0;
    double weighted_sum = 0; // Use double for weighted_sum to prevent overflow for large images
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = IDX(x, y, width);
            // Ensure x_coord and y_coord are relative to the center for correct frequency interpretation
            // assuming the magnitude array is already shifted (DC at center)
            float freq_x = (float)x - width / 2;
            float freq_y = (float)y - height / 2;
            float freq = sqrtf(freq_x * freq_x + freq_y * freq_y);
            weighted_sum += freq * mag[idx];
            sum_mag += mag[idx];
        }
    }
    return sum_mag > 0 ? (float)(weighted_sum / sum_mag) : 0;
}

float spectralEntropy(const std::vector<float>& mag) {
    double sum = 0; // Use double for sum to prevent overflow
    for (float val : mag) sum += val;

    double entropy = 0; // Use double for entropy calculation
    for (float val : mag) {
        float p = val / sum;
        if (p > 0)
            entropy -= p * log2f(p);
    }
    return (float)entropy;
}

float spectralEnergy(const std::vector<float>& mag) {
    double energy = 0; // Use double for energy calculation
    for (float val : mag)
        energy += (double)val * val;
    return (float)energy;
}

// Function to perform spectral analysis on an image (using cuFFT internally for magnitude)
void performSpectralAnalysis(const char* label, float* h_image, int width, int height) {
    // Allocate device memory for cuFFT processing
    float* d_input_cufft;
    cufftComplex* d_freq_cufft;
    cudaMalloc(&d_input_cufft, sizeof(float) * width * height);
    cudaMalloc(&d_freq_cufft, sizeof(cufftComplex) * width * height);

    // Copy host image data to device
    cudaMemcpy(d_input_cufft, h_image, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    // Create cuFFT plan and execute R2C FFT
    cufftHandle plan_cufft;
    cufftPlan2d(&plan_cufft, height, width, CUFFT_R2C);
    cufftExecR2C(plan_cufft, d_input_cufft, d_freq_cufft);
    cudaDeviceSynchronize(); // Ensure FFT completes

    // Compute magnitude from cuFFT results
    std::vector<float> mag = computeMagnitudeFromCufft(d_freq_cufft, width, height);

    // Perform FFT shift on the magnitude spectrum for spectral centroid calculation
    // The spectral centroid formula expects the DC component to be at the center.
    std::vector<float> shifted_mag = performFFTShift(mag, width, height);


    float centroid = spectralCentroid(shifted_mag, width, height); // Pass shifted magnitude
    float entropy = spectralEntropy(mag); // Entropy is generally independent of shift
    float energy = spectralEnergy(mag); // Energy is independent of shift

    // Output in the requested format
    std::cout << "\n Análisis espectral de la imagen (" << label << ")\n";
    std::cout << " Centroide espectral: " << centroid << "\n";
    std::cout << " Entropía espectral: " << entropy << "\n";
    std::cout << " Energía total: " << energy << "\n";

    // Destroy cuFFT plan and free device memory
    cufftDestroy(plan_cufft);
    cudaFree(d_input_cufft);
    cudaFree(d_freq_cufft);
}


int main() {
    const char* input_filename = "barbara.ascii.pgm";
    const char* output_fft_magnitude_filename = "fft_magnitude_shared_output.pgm";
    const char* output_ifft_filename = "ifft_shared_output.pgm";

    float* h_input_image_original; // Keep original image data before padding
    int original_width, original_height;
    if (!loadPGM(input_filename, &h_input_image_original, &original_width, &original_height)) {
        std::cerr << "Failed to load PGM: " << input_filename << std::endl;
        return 1;
    }
    
    // Create a copy of the original image data for analysis before padding
    float* h_input_image_padded = new float[original_width * original_height];
    std::copy(h_input_image_original, h_input_image_original + (original_width * original_height), h_input_image_padded);

    int width = original_width; // Will be modified by padImage
    int height = original_height; // Will be modified by padImage

    std::cout << "Original dimensions: " << width << " x " << height << std::endl;
    
    // --- Basic Image Stats BEFORE FFT (Mean and Standard Deviation) ---
    float original_mean, original_std_dev;
    calculateImageStats(h_input_image_padded, width, height, &original_mean, &original_std_dev);
    std::cout << "Original Image Spatial Stats (Before Padding): Mean = " << original_mean << ", Std Dev = " << original_std_dev << std::endl;

    // --- Spectral Analysis BEFORE FFT (using temporary cuFFT on padded image) ---
    // Note: padImage modifies h_input_image_padded, so call analysis after padding.
    // It's crucial to analyze the padded image for consistent FFT size.
    padImage(h_input_image_padded, width, height); // Pad the image for your custom FFT

    // Now perform spectral analysis on the padded image
    performSpectralAnalysis("Padded Original Image", h_input_image_padded, width, height);


    // Validate that padded dimensions are within MAX_FFT_LENGTH for our kernels
    if (width > MAX_FFT_LENGTH || height > MAX_FFT_LENGTH) {
        std::cerr << "Error: Padded image dimensions (" << width << "x" << height
                  << ") exceed MAX_FFT_LENGTH (" << MAX_FFT_LENGTH << "). "
                  << "Shared memory kernels cannot handle this size directly. "
                  << "Adjust MAX_FFT_LENGTH or use a different FFT strategy (e.g., cuFFT)." << std::endl;
        delete[] h_input_image_original;
        delete[] h_input_image_padded;
        return 1;
    }

    // Validate that padded dimensions are powers of 2 for our FFT algorithm
    if ((width & (width - 1)) != 0 || (height & (height - 1)) != 0) {
        std::cerr << "Error: Padded image dimensions (" << width << "x" << height
                  << ") are not powers of 2. Custom FFT kernels require power-of-2 dimensions." << std::endl;
        delete[] h_input_image_original;
        delete[] h_input_image_padded;
        return 1;
    }


    int size = width * height;

    float *d_real, *d_imag;
    float *d_transposed_real, *d_transposed_imag;

    cudaMalloc(&d_real, sizeof(float) * size);
    cudaMalloc(&d_imag, sizeof(float) * size);
    cudaMalloc(&d_transposed_real, sizeof(float) * size);
    cudaMalloc(&d_transposed_imag, sizeof(float) * size);

    cudaMemcpy(d_real, h_input_image_padded, sizeof(float) * size, cudaMemcpyHostToDevice); // Use padded image
    cudaMemset(d_imag, 0, sizeof(float) * size);

    // --- CUDA Events for Timing ---
    cudaEvent_t startFFT, endFFT;
    cudaEventCreate(&startFFT);
    cudaEventCreate(&endFFT);

    // --- 2D FFT Forward (Custom Kernels) ---
    cudaEventRecord(startFFT); // Start FFT timer

    // 1. FFT 1D en filas
    initTwiddleFactors(width);
    dim3 block_row(width); // A block processes the entire row
    dim3 grid_row(height); // One block per row

    fft1DRowKernel<<<grid_row, block_row>>>(d_real, d_imag, d_real, d_imag, width, height);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after fft1DRowKernel: " << cudaGetErrorString(err) << std::endl;
    }

    // 2. Transposición (de (W,H) a (H,W))
    dim3 block_transpose(16, 16);
    dim3 grid_transpose((width + block_transpose.x - 1) / block_transpose.x,
                        (height + block_transpose.y - 1) / block_transpose.y);

    transposeKernel<<<grid_transpose, block_transpose>>>(d_real, d_imag, d_transposed_real, d_transposed_imag, width, height);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after transposeKernel (1): " << cudaGetErrorString(err) << std::endl;
    }

    // 3. FFT 1D en columnas (que ahora son filas en la matriz transpuesta)
    initTwiddleFactors(height); // Initialize twiddle factors for the new FFT length (height)
    dim3 block_col(height);
    dim3 grid_col(width);

    fft1DColKernel<<<grid_col, block_col>>>(d_transposed_real, d_transposed_imag, d_real, d_imag, height, width); // Note: height, width here
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after fft1DColKernel: " << cudaGetErrorString(err) << std::endl;
    }

    // 4. Transposición de vuelta (de (H,W) a (W,H))
    transposeKernel<<<grid_transpose, block_transpose>>>(d_real, d_imag, d_transposed_real, d_transposed_imag, height, width); // Note: height, width here
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after transposeKernel (2): " << cudaGetErrorString(err) << std::endl;
    }
    // Now d_transposed_real and d_transposed_imag contain the 2D FFT result

    cudaEventRecord(endFFT); // End FFT timer
    cudaEventSynchronize(endFFT);
    float fftTime = 0;
    cudaEventElapsedTime(&fftTime, startFFT, endFFT);
    printElapsedTime("FFT execution time", fftTime);


    // Copy FFT results to host
    float* h_real_fft_output = new float[size];
    float* h_imag_fft_output = new float[size];
    float* h_fft_magnitude_output = new float[size];

    cudaMemcpy(h_real_fft_output, d_transposed_real, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_imag_fft_output, d_transposed_imag, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Calculate and save FFT magnitude (with shift) for visualization
    calculateMagnitude(h_real_fft_output, h_imag_fft_output, h_fft_magnitude_output, width, height);
    savePGM(output_fft_magnitude_filename, h_fft_magnitude_output, width, height);
    std::cout << "FFT magnitude output saved as " << output_fft_magnitude_filename << std::endl;

    // --- CUDA Events for Timing ---
    cudaEvent_t startIFFT, endIFFT;
    cudaEventCreate(&startIFFT);
    cudaEventCreate(&endIFFT);

    // --- 2D IFFT Inverse (Custom Kernels) ---
    cudaEventRecord(startIFFT); // Start IFFT timer

    // Input for IFFT are the FFT results (d_transposed_real, d_transposed_imag)
    // 1. IFFT 1D on rows
    initInverseTwiddleFactors(width);
    fft1DRowKernel<<<grid_row, block_row>>>(d_transposed_real, d_transposed_imag, d_real, d_imag, width, height);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after fft1DRowKernel (IFFT): " << cudaGetErrorString(err) << std::endl;
    }

    // 2. Transposition
    transposeKernel<<<grid_transpose, block_transpose>>>(d_real, d_imag, d_transposed_real, d_transposed_imag, width, height);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after transposeKernel (IFFT 1): " << cudaGetErrorString(err) << std::endl;
    }

    // 3. IFFT 1D on columns
    initInverseTwiddleFactors(height); // Initialize twiddle factors for the new IFFT length (height)
    fft1DColKernel<<<grid_col, block_col>>>(d_transposed_real, d_transposed_imag, d_real, d_imag, height, width);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after fft1DColKernel (IFFT): " << cudaGetErrorString(err) << std::endl;
    }

    // 4. Transposition back
    transposeKernel<<<grid_transpose, block_transpose>>>(d_real, d_imag, d_transposed_real, d_transposed_imag, height, width);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after transposeKernel (IFFT 2): " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(endIFFT); // End IFFT timer
    cudaEventSynchronize(endIFFT);
    float ifftTime = 0;
    cudaEventElapsedTime(&ifftTime, startIFFT, endIFFT);
    printElapsedTime("IFFT execution time", ifftTime);


    // Normalize IFFT output and copy to host
    float* h_ifft_output_image = new float[size];
    float* h_ifft_real_temp = new float[size];

    cudaMemcpy(h_ifft_real_temp, d_transposed_real, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // Normalization for 2D IFFT (1/(width * height))
    float normalization_factor = 1.0f / (width * height);
    for (int i = 0; i < size; ++i) {
        h_ifft_output_image[i] = h_ifft_real_temp[i] * normalization_factor;
    }

    savePGM(output_ifft_filename, h_ifft_output_image, width, height);
    std::cout << "IFFT output saved as " << output_ifft_filename << std::endl;

    // --- Basic Image Stats AFTER IFFT (Mean and Standard Deviation) ---
    float reconstructed_mean, reconstructed_std_dev;
    calculateImageStats(h_ifft_output_image, width, height, &reconstructed_mean, &reconstructed_std_dev);
    std::cout << "Reconstructed Image Spatial Stats: Mean = " << reconstructed_mean << ", Std Dev = " << reconstructed_std_dev << std::endl;

    // Calculate MSE and PSNR for reconstruction quality
    float mse = 0.0f;
    for (int i = 0; i < size; ++i) {
        // Ensure comparison is with the padded original image for correct MSE/PSNR calculation
        float diff = h_input_image_padded[i] - h_ifft_output_image[i];
        mse += diff * diff;
    }
    mse /= size;

    // Max pixel value is 255 for PGM (assuming 8-bit image for PSNR calculation)
    float max_pixel_value = 255.0f;
    float psnr = (mse == 0.0f) ? 100.0f : 10.0f * log10f((max_pixel_value * max_pixel_value) / mse);

    std::cout << "Reconstruction Quality: MSE = " << mse << ", PSNR = " << psnr << " dB" << std::endl;

    // --- Spectral Analysis AFTER IFFT (using temporary cuFFT on reconstructed image) ---
    performSpectralAnalysis("Reconstructed Image", h_ifft_output_image, width, height);

    performSpectralAnalysis("FFT Image", h_fft_magnitude_output, width, height);

    // Clean up
    delete[] h_input_image_original; // Original image (before padding)
    delete[] h_input_image_padded;   // Padded image (used for custom FFT input)
    delete[] h_real_fft_output;
    delete[] h_imag_fft_output;
    delete[] h_fft_magnitude_output;
    delete[] h_ifft_output_image;
    delete[] h_ifft_real_temp;

    cudaFree(d_real);
    cudaFree(d_imag);
    cudaFree(d_transposed_real);
    cudaFree(d_transposed_imag);

    // Destroy CUDA events
    cudaEventDestroy(startFFT);
    cudaEventDestroy(endFFT);
    cudaEventDestroy(startIFFT);
    cudaEventDestroy(endIFFT);

    return 0;
}
