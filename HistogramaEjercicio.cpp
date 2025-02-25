#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define N 1000000  // Tamaño del arreglo
#define MAX_VAL 1000 // Valor máximo en el arreglo
#define NUM_UMBRALES 10 // Número de umbrales (rangos)

void imprimir_histograma(const std::vector<int> &conteos, const std::vector<std::pair<int, int>> &umbrales, std::string metodo) {
    std::cout << "\nHistograma para método: " << metodo << "\n";
    for (size_t i = 0; i < conteos.size(); i++) {
        std::cout << "Rango " << umbrales[i].first << "-" << umbrales[i].second << " | ";
        int barras = conteos[i] / (N / 100); // Escala para 100 caracteres
        for (int j = 0; j < barras; j++) std::cout << "#";
        std::cout << " (" << conteos[i] << " valores)\n";
    }
    std::cout << "\n";
}

std::vector<int> contar_atomic(const std::vector<int> &arr, const std::vector<std::pair<int, int>> &umbrales) {
    std::vector<int> conteos(umbrales.size(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < umbrales.size(); i++) {
        int min_umbral = umbrales[i].first;
        int max_umbral = umbrales[i].second;

        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            if (arr[j] >= min_umbral && arr[j] <= max_umbral) {
                #pragma omp atomic
                conteos[i]++;
            }
        }
    }
    return conteos;
}

std::vector<int> contar_critical(const std::vector<int> &arr, const std::vector<std::pair<int, int>> &umbrales) {
    std::vector<int> conteos(umbrales.size(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < umbrales.size(); i++) {
        int min_umbral = umbrales[i].first;
        int max_umbral = umbrales[i].second;
        int contador_local = 0;

        #pragma omp parallel for reduction(+:contador_local)
        for (int j = 0; j < N; j++) {
            if (arr[j] >= min_umbral && arr[j] <= max_umbral) {
                contador_local++;
            }
        }

        #pragma omp critical
        conteos[i] += contador_local;
    }
    return conteos;
}

int main() {
    std::vector<int> arr(N);
    std::vector<std::pair<int, int>> umbrales;

    std::srand(std::time(0));

    // Crear rangos de umbrales (0-100, 101-200, ..., 901-1000)
    for (int i = 0; i < NUM_UMBRALES; i++) {
        umbrales.push_back({i * 100, (i + 1) * 100 - 1});
    }

    // Llenar el arreglo con valores aleatorios entre 0 y MAX_VAL
    for (int i = 0; i < N; i++) {
        arr[i] = std::rand() % MAX_VAL;
    }

    std::cout << "Iniciando conteo...\n";

    double start = omp_get_wtime();
    std::vector<int> conteos_atomic = contar_atomic(arr, umbrales);
    double end = omp_get_wtime();
    std::cout << "Tiempo con atomic: " << (end - start) << " segundos\n";
    imprimir_histograma(conteos_atomic, umbrales, "Atomic");

    start = omp_get_wtime();
    std::vector<int> conteos_critical = contar_critical(arr, umbrales);
    end = omp_get_wtime();
    std::cout << "Tiempo con critical: " << (end - start) << " segundos\n";
    imprimir_histograma(conteos_critical, umbrales, "Critical");

    return 0;
}
