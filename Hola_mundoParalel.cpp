#include <iostream>
#include <omp.h>  // Necesario para OpenMP

int main() {
    // Configura el número de hilos si deseas un número específico
    //omp_set_num_threads(4);  

    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        // Sección crítica para evitar que los mensajes se mezclen en la salida
        #pragma omp critical
        std::cout << "Hola desde el hilo " << id << std::endl;
    }

    return 0;
}
