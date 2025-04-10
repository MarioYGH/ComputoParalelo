#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fire_kernel(int* state, int rows, int cols) {
    // Posición global
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Tamaño del bloque + borde
    const int TILE_SIZE = 16;
    __shared__ int tile[TILE_SIZE + 2][TILE_SIZE + 2];

    // Índices locales
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Cargar datos al tile compartido
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        tile[ly][lx] = state[idx];

        // Borde (esquinas y lados)
        if (threadIdx.x == 0 && x > 0)
            tile[ly][lx - 1] = state[y * cols + (x - 1)];
        if (threadIdx.x == blockDim.x - 1 && x < cols - 1)
            tile[ly][lx + 1] = state[y * cols + (x + 1)];
        if (threadIdx.y == 0 && y > 0)
            tile[ly - 1][lx] = state[(y - 1) * cols + x];
        if (threadIdx.y == blockDim.y - 1 && y < rows - 1)
            tile[ly + 1][lx] = state[(y + 1) * cols + x];

        // Esquinas
        if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
            tile[ly - 1][lx - 1] = state[(y - 1) * cols + (x - 1)];
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < cols - 1 && y > 0)
            tile[ly - 1][lx + 1] = state[(y - 1) * cols + (x + 1)];
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < rows - 1)
            tile[ly + 1][lx - 1] = state[(y + 1) * cols + (x - 1)];
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < cols - 1 && y < rows - 1)
            tile[ly + 1][lx + 1] = state[(y + 1) * cols + (x + 1)];
    }

    __syncthreads();

    if (x >= cols || y >= rows) return;

    int center = tile[ly][lx];
    int new_val = center;

    if (center == 0) {
        // Buscar vecinos en fuego
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (tile[ly + dy][lx + dx] == 1) {
                    new_val = 1;
                    goto write_back;
                }
            }
        }
    } else if (center == 1) {
        new_val = 2;
    }

write_back:
    state[y * cols + x] = new_val;
}

void step_simulation_gpu(int* d_state, int rows, int cols) {
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);

    fire_kernel<<<blocks, threads>>>(d_state, rows, cols);
    cudaDeviceSynchronize();
}
