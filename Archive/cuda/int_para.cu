/*-------------int_para.cu----------------------------------------------------//
*
*              int_para -- CUDA in parallel
*
* Purpose: Parallelize int_gpu.cu
*
*   Notes: block: parallel invocation of kernel
*           grid: set of blocks
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include<cstdlib>

using namespace std;

void random_ints(int* a, int N);

// Define global kernel, adds blocks
__global__ void add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// define the number of blocks
#define N 512

int main(void){

    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // allocates space on device for a, b, and c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // copy to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // performs calculation
    add<<<N,1>>>(d_a, d_b, d_c);

    // return to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cout << *c << endl;

    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

void random_ints(int* a, int h){

    for (int i = 0; i < N; i++){
        a[i] = rand();
    }
}
