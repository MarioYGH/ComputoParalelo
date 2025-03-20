// Ejercicio 1: Simulación de votos en elecciones (40%)
// Escribe un programa en C utilizando OpenMP que simule una elección en una ciudad con 1,000,000 de votantes distribuidos en 5 candidatos. 
// Cada votante debe elegir aleatoriamente a un candidato. Utiliza paralelismo para contar los votos de forma eficiente.

// Requisitos:
// - Implementa la generación paralela de votos.
// - Cuenta los votos de cada candidato utilizando la directiva reduction.
// - Muestra el número total de votos por candidato y determina el ganador.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NUM_VOTANTES 1000000
#define NUM_CANDIDATOS 5

int main() {
    int votos[NUM_CANDIDATOS] = {0};
    int ganador = 0, max_votos = 0;
    
    // Semilla para generación de números aleatorios
    srand(time(NULL));
    
    double start_time = omp_get_wtime();
    
    // Generación de votos en paralelo
    #pragma omp parallel
    {
        int votos_local[NUM_CANDIDATOS] = {0};
        
        #pragma omp for
        for (int i = 0; i < NUM_VOTANTES; i++) {
            int voto = rand() % NUM_CANDIDATOS;
            votos_local[voto]++;
        }
        
        // Sumar votos locales a la cuenta global con reduction
        #pragma omp critical
        for (int j = 0; j < NUM_CANDIDATOS; j++) {
            votos[j] += votos_local[j];
        }
    }
    
    double end_time = omp_get_wtime();
    
    // Determinar el candidato ganador
    for (int i = 0; i < NUM_CANDIDATOS; i++) {
        if (votos[i] > max_votos) {
            max_votos = votos[i];
            ganador = i;
        }
        printf("Candidato %d: %d votos\n", i, votos[i]);
    }
    
    printf("\nGanador: Candidato %d con %d votos\n", ganador, max_votos);
    printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
    
    return 0;
}
