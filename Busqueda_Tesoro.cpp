#include <iostream> 
#include <vector>
#include <ctime>
#include <omp.h>
using namespace std;

void buscarTesorosSerial(const vector<vector<int>>& matriz, int N, int M) {
    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (matriz[i][j] == 1) {
                cout << "Tesoro encontrado en: (" << i << ", " << j << ")" << endl;
            }
        }
    }
    clock_t end = clock();
    cout << "Tiempo serial: " << (double)(end - start) / CLOCKS_PER_SEC << " segundos" << endl;
}

void buscarTesorosParalelo(const vector<vector<int>>& matriz, int N, int M) {
    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (matriz[i][j] == 1) {
                #pragma omp critical
                cout << "Tesoro encontrado en: (" << i << ", " << j << ")" << endl;
            }
        }
    }
    double end = omp_get_wtime();
    cout << "Tiempo paralelo: " << (end - start) << " segundos" << endl;
}

int main() {
    srand(time(NULL));
    int N=10000, M=10000, a,b;
    int Numerotesoros = 5;
    
    vector<vector<int>> matriz(N, vector<int>(M, 0)); // Matriz NxM con ceros

    for (int i=0; i < Numerotesoros; i++) {
        a = rand() % N;
        b = rand() % M;
        matriz[a][b] = 1;
    }

    cout << "Busqueda Serial:" << endl;
    buscarTesorosSerial(matriz, N, M);

    cout << "Busqueda Paralela:" << endl;
    buscarTesorosParalelo(matriz, N, M);

    return 0;
}
