/*------------int_add---------------------------------------------------------//
*
* Purpose: adding integers with the gpu! I am excited! Woo!
*
*-----------------------------------------------------------------------------*/

#include<iostream> 

__global__ void add(int *a, int *b, int *c){
    *c = *b + *a;
}

using namespace std;

int main(void){
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    // Allocate space on the gpu
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // input values 
    a = 2;
    b = 7;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    add<<<1,1>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 

    cout << "integer from GPU is: " <<  c << endl;

    return 0;
}
