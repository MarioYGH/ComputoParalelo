#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define N 1000000  // Tamaño del arreglo
#define UMBRAL 500 // Umbral para contar valores mayores

void contar_atomic(const std::vector<int> &arr) {
    int contador = 0;

    #pragma omp parallel for shared(arr) 
    for (int i = 0; i < N; i++) {
        if (arr[i] > UMBRAL) {
            #pragma omp atomic
            contador++;
        }
    }

    std::cout << "Resultado con atomic: " << contador << std::endl;
}

void contar_critical(const std::vector<int> &arr) {
    int contador = 0;

    #pragma omp parallel for shared(arr)
    for (int i = 0; i < N; i++) {
        if (arr[i] > UMBRAL) {
            #pragma omp critical
            {
                contador++;
            }
        }
    }

    std::cout << "Resultado con critical: " << contador << std::endl;
}

int main() {
    std::vector<int> arr(N);
    std::srand(std::time(0));

    // Llenar el arreglo con valores aleatorios entre 0 y 1000
    for (int i = 0; i < N; i++) {
        arr[i] = std::rand() % 1000;
    }

    std::cout << "Iniciando conteo...\n";

    double start = omp_get_wtime();
    contar_atomic(arr);
    double end = omp_get_wtime();
    std::cout << "Tiempo con atomic: " << (end - start) << " segundos\n\n";

    start = omp_get_wtime();
    contar_critical(arr);
    end = omp_get_wtime();
    std::cout << "Tiempo con critical: " << (end - start) << " segundos\n";

    return 0;
}
