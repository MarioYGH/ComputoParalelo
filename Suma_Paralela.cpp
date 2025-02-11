// Autor: Ulises Olivares
// 28 de enero, 2025
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

// Definir el tamaño del arreglo
const int N = 100000000; // 100 millones de elementos

// Función para la suma en modo serial
pair<long long, double> suma_serial(const vector<int>& array) {
    long long suma = 0;
    clock_t start = clock(); // Tiempo de inicio

    for (int i = 0; i < N; i++) {
        suma += array[i];
    }

    clock_t end = clock(); // Tiempo de finalización
    double tiempo = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "Suma (Serial): " << suma << endl;
    cout << "Tiempo serial: " << tiempo << " segundos" << endl;
    return {suma, tiempo}; // Retorna la suma y el tiempo
}

// Función para la suma en modo paralelo con OpenMP
pair<long long, double> suma_paralelo(const vector<int>& array) {
    long long suma = 0;
    double start = omp_get_wtime(); // Tiempo de inicio

#pragma omp parallel for reduction(+:suma)
    for (int i = 0; i < N; i++) {
        suma += array[i];
    }

    double end = omp_get_wtime(); // Tiempo de finalización
    double tiempo = end - start;

    cout << "Suma (Paralelo): " << suma << endl;
    cout << "Tiempo paralelo: " << tiempo << " segundos" << endl;
    return {suma, tiempo}; // Retorna la suma y el tiempo
}

int main() {
    vector<int> array(N, 1); // Llenamos el array con 1s

    int opcion;
    cout << "Selecciona el modo de ejecución:\n";
    cout << "1 - Suma Serial\n";
    cout << "2 - Suma Paralela con OpenMP\n";
    cout << "3 - Ejecutar ambas, validar y calcular Speed-Up y Eficiencia\n";
    cout << "Opción: ";
    cin >> opcion;

    if (opcion == 1) {
        auto [suma, tiempo] = suma_serial(array);
    } else if (opcion == 2) {
        auto [suma, tiempo] = suma_paralelo(array);
    } else if (opcion == 3) {
        // Ejecutar ambas versiones
        auto [suma_serial_result, tiempo_serial] = suma_serial(array);
        auto [suma_paralelo_result, tiempo_paralelo] = suma_paralelo(array);

        // Validar resultados
        if (suma_serial_result == suma_paralelo_result) {
            cout << "✅ Ambas sumas son correctas y coinciden.\n";
        } else {
            cout << "❌ Error: Los resultados de las sumas no coinciden.\n";
            return 1; // Terminar si hay error
        }

        // Calcular Speed-Up y Eficiencia
        int num_hilos = omp_get_max_threads();
        double speed_up = tiempo_serial / tiempo_paralelo;
        double eficiencia = speed_up / num_hilos;

        // Mostrar resultados
        cout << "\n=== Resultados de Desempeño ===\n";
        cout << "Tiempo serial: " << tiempo_serial << " segundos\n";
        cout << "Tiempo paralelo: " << tiempo_paralelo << " segundos\n";
        cout << "Speed-Up: " << speed_up << endl;
        cout << "Eficiencia: " << eficiencia * 100 << "%\n";
        cout << "Número de hilos utilizados: " << num_hilos << endl;
    } else {
        cout << "Opción no válida.\n";
    }

    return 0;
}

