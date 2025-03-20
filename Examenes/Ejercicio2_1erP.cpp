// Ejercicio 2: Análisis paralelo de temperatura ambiental (60%)
// Crea un programa en C con OpenMP que analice datos de temperatura recogidos cada minuto durante un año completo (525,600 datos). 

// El programa debe encontrar:
// - La temperatura media anual.
// - El número de días con temperaturas superiores a 30°C.
// - El número de días con temperaturas inferiores a 0°C.
// Requisitos:
// - Genera temperaturas aleatorias en el rango de -10°C a 40°C.
// - Usa paralelismo para calcular la temperatura media y conteos eficientemente.
// - Implementa las directivas necesarias para asegurar la correcta sincronización y exclusión mutua cuando sea necesario.
// - Muestra  los resultados obtenidos al finalizar la ejecución del programa.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NUM_DATOS 525600
#define TEMP_MIN -10
#define TEMP_MAX 40

int main() {
    double *temperaturas = (double *)malloc(NUM_DATOS * sizeof(double));
    if (temperaturas == NULL) {
        printf("Error al asignar memoria.\n");
        return 1;
    }

    srand(time(NULL));

    // Generación de datos aleatorios
    for (int i = 0; i < NUM_DATOS; i++) {
        temperaturas[i] = TEMP_MIN + (rand() / (double)RAND_MAX) * (TEMP_MAX - TEMP_MIN);
    }

    double suma_temperaturas = 0.0;
    int dias_calientes = 0, dias_frios = 0;

    double start_time = omp_get_wtime();

    // Cálculo de estadísticas en paralelo
    #pragma omp parallel for reduction(+:suma_temperaturas, dias_calientes, dias_frios)
    for (int i = 0; i < NUM_DATOS; i++) {
        suma_temperaturas += temperaturas[i];
        if (temperaturas[i] > 30) dias_calientes++;
        if (temperaturas[i] < 0) dias_frios++;
    }

    double temperatura_media = suma_temperaturas / NUM_DATOS;
    double end_time = omp_get_wtime();

    printf("Temperatura media anual: %.2f°C\n", temperatura_media);
    printf("Días con temperaturas superiores a 30°C: %d\n", dias_calientes);
    printf("Días con temperaturas inferiores a 0°C: %d\n", dias_frios);
    printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);

    free(temperaturas);
    return 0;
}
