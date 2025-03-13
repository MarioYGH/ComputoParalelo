// Para compilar desde terminal, nvcc test.cu -o test 
// Para correr, ./test

#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello world from GPU!\n");
}

int main(){
    cuda_hello<<<1,15>>>();
    cudaDeviceSynchronize();
    return 0;
}
