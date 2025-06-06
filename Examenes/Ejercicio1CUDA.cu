// Alternar signos en un arreglo
// Objetivo: Modificar un arreglo de enteros de tal forma que los elementos en posiciones pares sean positivos y los impares negativos.
// Instrucciones:
//  • Escribe un kernel CUDA que tome un arreglo A de enteros y lo modifique as ́ı:
//   – Si el  ́ındice global i es par, entonces A[i] = abs(A[i]).
//   – Si es impar, entonces A[i] = -abs(A[i]).
//  • Usa un arreglo de tama ̃no n = 128 y blockDim.x = 32.
// Requisitos:
//  • El kernel debe usar indexaci ́on con threadIdx.x, blockIdx.x y blockDim.x.
//  • Verificar que i < n.

#include <iostream>
#include <cstdlib>   // Para rand() y abs()
#include <cuda_runtime.h>

#define N 128
#define BLOCK_SIZE 32

__global__ void alternarSignos(int *A, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        if (i % 2 == 0) {
            A[i] = abs(A[i]);
        } else {
            A[i] = -abs(A[i]);
        }
    }
}

int main() {
    int h_A[N];
    
    // Inicializar arreglo con valores aleatorios entre -50 y 50
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 101 - 50;
    }

    // Mostrar arreglo original
    std::cout << "Arreglo original:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\n\n";

    // Reservar memoria en el dispositivo
    int *d_A;
    cudaMalloc((void**)&d_A, N * sizeof(int));

    // Copiar datos al dispositivo
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);

    // Lanzar kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    alternarSignos<<<numBlocks, BLOCK_SIZE>>>(d_A, N);

    // Copiar datos modificados al host
    cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar resultado
    std::cout << "Arreglo modificado:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\n";

    // Liberar memoria
    cudaFree(d_A);

    return 0;
}
