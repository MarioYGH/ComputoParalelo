// Marcar m ́ultiplos en un arreglo binario
// Objetivo: Llenar un arreglo de salida donde cada posici ́on tenga un 1 si su  ́ındice es m ́ultiplo de 5, y 0 en caso contrario.
// Instrucciones:
//  • Escribe un kernel CUDA que genere un arreglo M de tama ̃no n = 100.
//  • Cada posici ́on i de M debe contener 1 si i % 5 == 0, y 0 en otro caso.
//  • Usa blockDim.x = 25 y el m ́ınimo n ́umero de bloques necesario.
// Requisitos:
//  • El alumno debe lanzar el n ́umero correcto de hilos.
//  • El kernel debe escribir directamente en el arreglo de salida M.

#include <iostream>
#include <cuda_runtime.h>

#define N 100
#define BLOCK_SIZE 25  

__global__ void marcarMultiplosDe5(int *M, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        M[i] = (i % 5 == 0) ? 1 : 0;
    }
}

int main() {
    int h_M[N];

    int *d_M;
    cudaMalloc((void**)&d_M, N * sizeof(int));

    // Calcular número de bloques
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Lanzar kernel
    marcarMultiplosDe5<<<numBlocks, BLOCK_SIZE>>>(d_M, N);

    // Copiar resultados al host
    cudaMemcpy(h_M, d_M, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar resultado
    std::cout << "Arreglo M (1 si índice % 5 == 0, 0 en otro caso):\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_M[i] << " ";
    }
    std::cout << "\n";

    // Liberar memoria
    cudaFree(d_M);
    return 0;
}
