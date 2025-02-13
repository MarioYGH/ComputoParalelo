#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>

int N = 500; 

using namespace std;

// Definir el tamaño del arreglo


int main() {
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N));
    vector<vector<int>> D(N, vector<int>(N));
    vector<vector<int>> E(N, vector<int>(N));


    int numero_hilos = omp_get_max_threads();

    srand(time(NULL));

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[i][j] = rand() % 100 + 1;
            B[i][j] = rand() % 100 + 1;
        }
        
    }

    clock_t start = clock();
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            C[i][j] = 0;
            for(int k=0; k < N; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    clock_t end = clock();

    double Tiemposerie = double(end-start);
    printf("Tiempo con serie: %lf secs\n", double(end-start)/ double(CLOCKS_PER_SEC));
    

    clock_t start2 = clock();
    #pragma omp parallel for collapse(2)
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            D[i][j] = 0;
            for(int k=0; k < N; k++){
                D[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    clock_t end2 = clock();

    double Tiempoparalelocolapsado = double(end2-start2);

    printf("Tiempo con colapsado: %lf secs\n", double(end2-start2)/ double(CLOCKS_PER_SEC));
    printf("La eficiencia es de:  %lf \n", double((Tiemposerie/Tiempoparalelocolapsado))/(numero_hilos));

    clock_t start3 = clock();
    #pragma omp parallel for 
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            E[i][j] = 0;
            for(int k=0; k < N; k++){
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    clock_t end3 = clock();

    double Tiempoparalelo = double(end3-start3);

    printf("Tiempo sin colapsar: %lf secs\n", double(end3-start3)/ double(CLOCKS_PER_SEC));
    printf("La eficiencia es de:  %lf ", double((Tiemposerie/Tiempoparalelo))/(numero_hilos));

    return 0;

}
